#!/usr/bin/env python3
"""
Connection-standard conformance check for the data-context organ.

Asserts the three guarantees the orchestrator's CONNECTORS.md asks of a
ports.json declaration:

  1. ports.json parses and is well-formed
       — top-level {"inputs": [...], "outputs": [...]};
         every input has {name, type, required}; every output has {name, type}.
  2. Every declared type exists in the (vendored) types.json vocabulary.
  3. decide() actually reads every declared input name from `state`, and
     writes exactly the declared set of output names under its `output` dict.
       — reads are verified statically (AST scan for `state.get("...")`),
         so the check doesn't depend on which op a sample happens to exercise;
       — writes are verified dynamically by running decide() across a
         representative state per op and taking the UNION of the keys each
         produces under `output` (this is a multi-op organ — no single call
         emits all eight outputs).

The check is self-testing: at the bottom it re-runs its own assertions against
deliberately-broken in-memory copies of the manifests and confirms each one is
rejected, so a future edit that weakens a guarantee turns CI red here too.

Exit 0 = conform; non-zero = a guarantee is violated.
"""
from __future__ import annotations

import ast
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------- #
# loaders
# --------------------------------------------------------------------------- #
def _load_json(name: str) -> dict:
    with open(os.path.join(HERE, name)) as f:
        return json.load(f)


def _vocabulary(types_doc: dict) -> set[str]:
    types = types_doc.get("types", types_doc)
    if not isinstance(types, dict):
        raise ValueError("types.json must carry a 'types' object (or be one)")
    return {k for k in types if not k.startswith("_")}


# --------------------------------------------------------------------------- #
# static read scan
# --------------------------------------------------------------------------- #
def _state_get_literals(organ_src: str) -> set[str]:
    """Every literal X in a top-level `state.get("X")` call within organ.py."""
    tree = ast.parse(organ_src)
    found: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if not (isinstance(fn, ast.Attribute) and fn.attr == "get"):
            continue
        # only `state.get(...)` — not conn.get / summary.get / col.get / v.get
        if not (isinstance(fn.value, ast.Name) and fn.value.id == "state"):
            continue
        if node.args and isinstance(node.args[0], ast.Constant) \
                and isinstance(node.args[0].value, str):
            found.add(node.args[0].value)
    return found


# --------------------------------------------------------------------------- #
# dynamic write scan — representative state per op, union of output keys
# --------------------------------------------------------------------------- #
_REPRESENTATIVE_STATES = [
    {
        "op": "summary",
        "connections": [
            {"name": "Staff CSV", "summary": {
                "source": "staff.csv", "rows": 200,
                "columns": [{"name": "dept", "type": "string", "samples": ["Ops"]}],
            }},
        ],
    },
    {"op": "summary", "connections": []},               # summary_empty branch
    {"op": "validate_gate", "has_data": True, "answer_text": "a" * 40},
    {"op": "validate_gate", "has_data": False, "answer_text": ""},
    {"op": "normalize_validations", "validations": [{"claim": "x"}]},
    {"op": "normalize_validations", "validations": "not-a-list"},
    {"op": "normalize_questions", "questions": ["q1", "q2", "q3", "q4"]},
    {"op": "normalize_questions", "questions": "not-a-list"},
]


def _produced_output_keys(decide) -> set[str]:
    keys: set[str] = set()
    for state in _REPRESENTATIVE_STATES:
        result = decide(state)
        out = result.get("output")
        if isinstance(out, dict):
            keys.update(out)
    return keys


# --------------------------------------------------------------------------- #
# the three guarantees
# --------------------------------------------------------------------------- #
def check_ports(ports: dict, vocab: set[str], read_literals: set[str],
                produced_outputs: set[str]) -> None:
    # (1) shape
    if not isinstance(ports, dict):
        raise AssertionError("ports.json must be a JSON object")
    inputs = ports.get("inputs")
    outputs = ports.get("outputs")
    if not isinstance(inputs, list) or not isinstance(outputs, list):
        raise AssertionError("ports.json must have list 'inputs' and 'outputs'")
    for i in inputs:
        if not isinstance(i, dict) or not {"name", "type", "required"} <= set(i):
            raise AssertionError(f"input missing name/type/required: {i!r}")
        if not isinstance(i.get("name"), str) or not isinstance(i.get("type"), str):
            raise AssertionError(f"input name/type must be strings: {i!r}")
        if not isinstance(i.get("required"), bool):
            raise AssertionError(f"input 'required' must be bool: {i!r}")
    for o in outputs:
        if not isinstance(o, dict) or not {"name", "type"} <= set(o):
            raise AssertionError(f"output missing name/type: {o!r}")
        if not isinstance(o.get("name"), str) or not isinstance(o.get("type"), str):
            raise AssertionError(f"output name/type must be strings: {o!r}")

    declared_inputs = {i["name"] for i in inputs}
    declared_outputs = {o["name"] for o in outputs}

    # (2) every declared type is in the vocabulary
    for p in inputs + outputs:
        if p["type"] not in vocab:
            raise AssertionError(
                f"type {p['type']!r} for port {p['name']!r} not in types.json "
                f"vocabulary {sorted(vocab)}"
            )

    # (3a) decide() reads exactly the declared inputs (no undeclared read,
    #      no declared-but-never-read input)
    undeclared_reads = read_literals - declared_inputs
    if undeclared_reads:
        raise AssertionError(
            f"decide() reads state keys not declared as inputs: "
            f"{sorted(undeclared_reads)}"
        )
    unread_inputs = declared_inputs - read_literals
    if unread_inputs:
        raise AssertionError(
            f"inputs declared but never read by decide(): {sorted(unread_inputs)}"
        )

    # (3b) decide() writes exactly the declared outputs (union across ops)
    undeclared_writes = produced_outputs - declared_outputs
    if undeclared_writes:
        raise AssertionError(
            f"decide() produces output keys not declared: "
            f"{sorted(undeclared_writes)}"
        )
    unproduced_outputs = declared_outputs - produced_outputs
    if unproduced_outputs:
        raise AssertionError(
            f"outputs declared but never produced by decide(): "
            f"{sorted(unproduced_outputs)}"
        )


# --------------------------------------------------------------------------- #
# self-test: each broken manifest MUST be rejected
# --------------------------------------------------------------------------- #
def _self_test(ports: dict, vocab: set[str], reads: set[str], outs: set[str]) -> None:
    import copy

    def expect_reject(label, p=None, v=None, r=None, o=None):
        try:
            check_ports(p if p is not None else ports,
                        v if v is not None else vocab,
                        r if r is not None else reads,
                        o if o is not None else outs)
        except AssertionError:
            return
        raise SystemExit(f"SELF-TEST FAILED: {label} should have been rejected")

    # a) unknown type
    bad = copy.deepcopy(ports)
    bad["inputs"][0]["type"] = "definitely_not_a_type"
    expect_reject("unknown type", p=bad)

    # b) undeclared read (organ reads something not in ports)
    expect_reject("undeclared read", r=reads | {"sneaky_unlisted_key"})

    # c) declared-but-unproduced output
    bad = copy.deepcopy(ports)
    bad["outputs"].append({"name": "phantom_output", "type": "string"})
    expect_reject("phantom output", p=bad)

    # d) malformed shape
    expect_reject("missing inputs list", p={"outputs": ports["outputs"]})


# --------------------------------------------------------------------------- #
def main() -> int:
    ports = _load_json("ports.json")
    vocab = _vocabulary(_load_json("types.json"))
    with open(os.path.join(HERE, "organ.py")) as f:
        read_literals = _state_get_literals(f.read())

    # import decide() from the organ in this directory
    sys.path.insert(0, HERE)
    from organ import decide  # noqa: E402

    produced = _produced_output_keys(decide)

    check_ports(ports, vocab, read_literals, produced)
    _self_test(ports, vocab, read_literals, produced)

    print("ports.json conformance: OK")
    print(f"  inputs  declared+read   = {sorted(read_literals)}")
    print(f"  outputs declared+produced = {sorted(produced)}")
    print(f"  vocabulary              = {sorted(vocab)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
