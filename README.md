# Data Context Organ

A pure decider for the connected-data interview-context lifecycle, extracted
from discovery-engine `app/services/data_context.py`.

## What it does

DataFlow Advisory lets a tenant attach data sources (CSV, Google Sheets,
VisualSoft, …) and injects a summary of that data into Claude discovery
interviews — so the AI can ask data-informed questions and flag answers that
contradict the numbers. The original service interleaves a lot of I/O (DB
lookups, connector instantiation, OAuth, OpenRouter calls) with a handful of
pure transforms and gating decisions. **This organ is only those pure parts.**

By the time work reaches the organ, the caller has already fetched each
connection's `summarise()` dict and/or received the raw LLM output. The organ
performs four pure operations, dispatched on `state.op`:

| `op`                     | Mirrors (source fn)                       | Decision |
|--------------------------|-------------------------------------------|----------|
| `summary`                | `get_data_summary_for_interview`          | Assemble the `CONNECTED DATA SOURCES:` context block from pre-fetched connector summaries; skip unbuildable (null) connectors; empty string when nothing survives. |
| `validate_gate`          | `validate_answer_against_data` (guards)   | Should we spend an LLM call validating this answer? Needs a non-empty data summary **and** an answer ≥ 20 chars. |
| `normalize_validations`  | `validate_answer_against_data` (post)     | Clamp the raw LLM validation array to ≤ 5, drop entries lacking a `claim`, coerce `severity` to the allowed enum. |
| `normalize_questions`    | `generate_data_informed_questions` (post) | Clamp the raw LLM question array to ≤ 3 and stringify. |

Everything else in the source — building connectors, Google credentials,
schema caching, the OpenRouter prompts and calls, DB writes — stays the
caller's responsibility. The organ makes **no** DB/network/connector calls.

## Input / output contract

The organ reads a JSON object with a `state` field (and an optional, ignored
`context`). It always returns:

```json
{
  "output": { },
  "rationale": "...",
  "self_metric": { "confidence": 1.0, "decision_path": "..." }
}
```

`output` is `null` for an unknown `op` or a fail-safe error. The organ never
raises on bad input — it returns the conservative empty result instead.

### `op = "summary"`

```json
{
  "state": {
    "op": "summary",
    "connections": [
      {
        "name": "Staff roster",
        "summary": {
          "source": "staff.csv",
          "rows": 200,
          "columns": [
            {"name": "department", "type": "string", "samples": ["Ops", "Sales"]}
          ]
        }
      },
      {"name": "broken connector", "summary": null}
    ]
  }
}
```

→ `output: {"context": "CONNECTED DATA SOURCES:\n\n...", "source_count": 1}`
(the `null`-summary connection is skipped, exactly as the source skips
connectors it cannot build).

### `op = "validate_gate"`

```json
{"state": {"op": "validate_gate", "has_data": true, "answer_text": "We process about fifty orders a day."}}
```

→ `output: {"should_validate": true, "reason": "ok"}`. `reason` is
`"no_data"` when `has_data` is false (checked first) or `"answer_too_short"`
when the stripped answer is under 20 chars.

### `op = "normalize_validations"`

```json
{"state": {"op": "normalize_validations", "validations": [{"claim": "...", "data_shows": "...", "source": "...", "severity": "contradiction"}]}}
```

→ `output: {"validations": [ ...≤5 cleaned dicts... ], "dropped": 0}`. Entries
without a truthy `claim` are dropped; `severity` not in
`{contradiction, discrepancy, note}` becomes `note`.

### `op = "normalize_questions"`

```json
{"state": {"op": "normalize_questions", "questions": ["q1", "q2", "q3", "q4"]}}
```

→ `output: {"questions": ["q1", "q2", "q3"], "count": 3}`.

## Running

```bash
# stdin
echo '{"state": {"op": "normalize_questions", "questions": ["a", "b"]}}' | python3 organ.py

# or a file
ORGAN_INPUT=samples/summary_two_sources.json python3 organ.py
```

## Tests

```bash
python -m pip install pytest
python -m pytest -v
```

The `conformance` GitHub Action shadow-runs the organ on every file in
`samples/` and prints each verdict + `self_metric` to the job summary, then
runs the full pytest suite.

## Provenance

Extracted from `discovery-engine/app/services/data_context.py`. The constants
(`MIN_ANSWER_LEN=20`, `MAX_QUESTIONS=3`, `MAX_VALIDATIONS=5`, the severity
enum) mirror the source so behaviour is bit-for-bit identical to the inlined
logic. Pure decision organ per the orchestrator `CONTRACT.md`.
