#!/usr/bin/env python3
"""
Data Context Organ — extracted decision logic from discovery-engine.

A pure decider for the connected-data interview-context lifecycle:
  - assemble a human-readable data-context block from pre-fetched connector
    summaries (so it can be injected into a Claude interview prompt);
  - gate whether an answer is worth validating against the data;
  - normalise the two AI-assisted outputs (data-informed questions, and
    answer-vs-data validations) into their capped, well-typed shapes.

All the I/O in the original `app/services/data_context.py` (DB lookups,
connector instantiation, OAuth credential building, OpenRouter calls) is the
CALLER'S job. By the time work reaches this organ, the caller has already
fetched each connection's `summarise()` dict and/or received the raw LLM
output. The organ only performs the pure transforms and gating decisions.

Contract (per orchestrator CONTRACT.md):
  INPUT state: {
    "op": "summary" | "validate_gate"
        | "normalize_validations" | "normalize_questions",
    ...op-specific fields (see below)...
  }

  op="summary" — assemble the interview data-context string.
    "connections": [
      {"name": str, "summary": {            # summary == null -> connector
         "source": str,                     #   unbuildable, skip the entry
         "rows": int | str,
         "columns": [{"name": str, "type": str, "samples": [..]}]
      } | null},
      ...
    ]
    OUTPUT.output: {"context": str, "source_count": int}
      context == "" when no buildable connection produced a block.

  op="validate_gate" — should we spend an LLM call validating this answer?
    "has_data": bool,        # is there a non-empty data summary to compare to?
    "answer_text": str       # the interviewee's answer
    OUTPUT.output: {"should_validate": bool,
                    "reason": "no_data" | "answer_too_short" | "ok"}

  op="normalize_validations" — clamp/clean the raw LLM validation array.
    "validations": <any>     # ideally a list of dicts
    OUTPUT.output: {"validations": [{claim, data_shows, source, severity}], "dropped": int}

  op="normalize_questions" — clamp/stringify the raw LLM question array.
    "questions": <any>       # ideally a list
    OUTPUT.output: {"questions": [str, ...], "count": int}

  OUTPUT (every op): {
    "output": {...} | null,
    "rationale": str,
    "self_metric": {"confidence": float, "decision_path": str}
  }

The organ is pure:
  - Takes all inputs via JSON
  - Makes no DB/network/connector calls
  - Returns only computed advice
  - Never raises on bad input (fail-safe to the conservative empty result)
"""

from __future__ import annotations

import json
import os
import sys

# Mirror the source constants so behaviour stays bit-for-bit identical.
_MIN_ANSWER_LEN = 20          # data_context.validate_answer_against_data
_MAX_QUESTIONS = 3            # generate_data_informed_questions
_MAX_VALIDATIONS = 5          # validate_answer_against_data
_SEVERITIES = ("contradiction", "discrepancy", "note")
_DEFAULT_SEVERITY = "note"


def _op_summary(state: dict) -> dict:
    """Assemble the CONNECTED DATA SOURCES context block.

    Mirrors data_context.get_data_summary_for_interview: skip connections whose
    summary is null (unbuildable connector), format each surviving summary, and
    join with the '---' separator. Empty string when nothing survives.
    """
    connections = state.get("connections")
    coerced = False
    if not isinstance(connections, list):
        connections = []
        coerced = connections is not state.get("connections")

    parts: list[str] = []
    for conn in connections:
        if not isinstance(conn, dict):
            continue
        summary = conn.get("summary")
        if not isinstance(summary, dict):
            # null / missing summary == unbuildable connector -> skip
            continue
        name = conn.get("name", "")
        col_lines = []
        for col in summary.get("columns", []) or []:
            if not isinstance(col, dict):
                continue
            samples = ", ".join(str(s) for s in (col.get("samples") or []))
            col_lines.append(
                f"  - {col.get('name')} ({col.get('type')}): e.g. {samples}"
            )
        part = (
            f"Data source: {summary.get('source', name)}\n"
            f"Rows: {summary.get('rows', '?')}\n"
            f"Columns:\n" + "\n".join(col_lines)
        )
        parts.append(part)

    if not parts:
        return {
            "output": {"context": "", "source_count": 0},
            "rationale": "No buildable connections produced a data block.",
            "self_metric": {
                "confidence": 0.5 if coerced else 1.0,
                "decision_path": "summary_empty",
            },
        }

    context = "CONNECTED DATA SOURCES:\n\n" + "\n\n---\n\n".join(parts)
    return {
        "output": {"context": context, "source_count": len(parts)},
        "rationale": f"Assembled context from {len(parts)} data source(s).",
        "self_metric": {
            "confidence": 0.5 if coerced else 1.0,
            "decision_path": "summary_assembled",
        },
    }


def _op_validate_gate(state: dict) -> dict:
    """Decide whether an answer is worth validating against the data.

    Mirrors the two early-return guards in validate_answer_against_data:
      1. no data summary -> skip
      2. answer shorter than 20 chars (stripped) -> skip
    """
    has_data = bool(state.get("has_data"))
    answer_text = state.get("answer_text", "")
    if not isinstance(answer_text, str):
        answer_text = str(answer_text)

    if not has_data:
        return {
            "output": {"should_validate": False, "reason": "no_data"},
            "rationale": "No connected-data summary to validate the answer against.",
            "self_metric": {"confidence": 1.0, "decision_path": "gate_no_data"},
        }

    if len(answer_text.strip()) < _MIN_ANSWER_LEN:
        return {
            "output": {"should_validate": False, "reason": "answer_too_short"},
            "rationale": (
                f"Answer is shorter than {_MIN_ANSWER_LEN} chars; unlikely to carry "
                "a quantitative claim worth validating."
            ),
            "self_metric": {"confidence": 1.0, "decision_path": "gate_too_short"},
        }

    return {
        "output": {"should_validate": True, "reason": "ok"},
        "rationale": "Data present and answer long enough — validation worthwhile.",
        "self_metric": {"confidence": 1.0, "decision_path": "gate_ok"},
    }


def _op_normalize_validations(state: dict) -> dict:
    """Clamp and clean the raw LLM validation array.

    Mirrors the post-processing in validate_answer_against_data: keep only the
    first 5, drop entries that are not dicts or lack a 'claim', coerce severity
    to the allowed enum (default 'note'), and stringify the text fields.
    """
    raw = state.get("validations")
    if not isinstance(raw, list):
        return {
            "output": {"validations": [], "dropped": 0},
            "rationale": "Validation payload was not a list; returning empty.",
            "self_metric": {"confidence": 0.0, "decision_path": "validations_not_list"},
        }

    considered = raw[:_MAX_VALIDATIONS]
    result = []
    for v in considered:
        if isinstance(v, dict) and v.get("claim"):
            severity = v.get("severity")
            result.append({
                "claim": str(v.get("claim", "")),
                "data_shows": str(v.get("data_shows", "")),
                "source": str(v.get("source", "")),
                "severity": severity if severity in _SEVERITIES else _DEFAULT_SEVERITY,
            })

    dropped = len(considered) - len(result)
    return {
        "output": {"validations": result, "dropped": dropped},
        "rationale": (
            f"Normalised {len(result)} validation(s); dropped {dropped} "
            f"of the first {len(considered)} considered (cap {_MAX_VALIDATIONS})."
        ),
        "self_metric": {"confidence": 1.0, "decision_path": "validations_normalized"},
    }


def _op_normalize_questions(state: dict) -> dict:
    """Clamp and stringify the raw LLM question array.

    Mirrors generate_data_informed_questions: [str(q) for q in questions[:3]]
    when a list, else [].
    """
    raw = state.get("questions")
    if not isinstance(raw, list):
        return {
            "output": {"questions": [], "count": 0},
            "rationale": "Questions payload was not a list; returning empty.",
            "self_metric": {"confidence": 0.0, "decision_path": "questions_not_list"},
        }

    questions = [str(q) for q in raw[:_MAX_QUESTIONS]]
    return {
        "output": {"questions": questions, "count": len(questions)},
        "rationale": f"Capped to {len(questions)} question(s) (max {_MAX_QUESTIONS}).",
        "self_metric": {"confidence": 1.0, "decision_path": "questions_normalized"},
    }


_OPS = {
    "summary": _op_summary,
    "validate_gate": _op_validate_gate,
    "normalize_validations": _op_normalize_validations,
    "normalize_questions": _op_normalize_questions,
}


def decide(state: dict, context: dict | None = None) -> dict:
    """Pure data-context decider.

    Args:
        state: {"op": ..., ...op-specific fields...}
        context: unused, present for orchestrator compatibility.

    Returns:
        {"output": {...} | null, "rationale": str, "self_metric": {...}}
    """
    context = context or {}
    try:
        if not isinstance(state, dict):
            raise TypeError("state must be a dict")
        op = state.get("op")
        handler = _OPS.get(op)
        if handler is None:
            return {
                "output": None,
                "rationale": (
                    f"Unknown op {op!r}; expected one of {sorted(_OPS)}."
                ),
                "self_metric": {"confidence": 0.0, "decision_path": "unknown_op"},
            }
        return handler(state)
    except Exception as e:  # fail-safe — never raise into the orchestrator
        return {
            "output": None,
            "rationale": f"Decision logic error (fail-safe): {e}",
            "self_metric": {"confidence": 0.0, "decision_path": "error_fallback"},
        }


def main() -> int:
    path = os.environ.get("ORGAN_INPUT")
    raw = open(path).read() if path else sys.stdin.read()
    try:
        payload = json.loads(raw)
        state = payload["state"]
    except Exception as e:
        print(json.dumps({"error": f"invalid input: {e}"}), file=sys.stderr)
        return 1
    print(json.dumps(decide(state, payload.get("context")), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
