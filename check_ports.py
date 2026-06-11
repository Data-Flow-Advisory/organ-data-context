#!/usr/bin/env python3
"""Port-manifest conformance check (the connection-standard "stud" check).

Asserts, per CONNECTORS.md ("Conformance gains a port check"):
  1. ports.json parses and has the {inputs:[{name,type,required}], outputs:[{name,type}]} shape.
  2. Every port `type` exists in the shared vocabulary (types.json).
  3. decide() actually READS each declared input name and WRITES each declared
     output name — sampled against the organ's own committed samples/*.json.

Exit code 0 = green, 1 = the conformance Action should go red.

This module exposes the individual checks as functions so test_ports.py can
assert them under pytest too (the conformance workflow runs both this script
and the pytest suite).
"""

from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PORTS_PATH = os.path.join(HERE, "ports.json")
TYPES_PATH = os.path.join(HERE, "types.json")
ORGAN_PATH = os.path.join(HERE, "organ.py")
SAMPLES_DIR = os.path.join(HERE, "samples")


def load_ports() -> dict:
    """Check 1: ports.json parses and has the required shape."""
    with open(PORTS_PATH) as f:
        ports = json.load(f)
    if not isinstance(ports, dict):
        raise ValueError("ports.json must be a JSON object")
    for side in ("inputs", "outputs"):
        decl = ports.get(side)
        if not isinstance(decl, list):
            raise ValueError(f"ports.json '{side}' must be a list")
        for p in decl:
            if not isinstance(p, dict) or "name" not in p or "type" not in p:
                raise ValueError(f"each {side} port needs 'name' and 'type': {p!r}")
            if not isinstance(p["name"], str) or not isinstance(p["type"], str):
                raise ValueError(f"port 'name'/'type' must be strings: {p!r}")
        if side == "inputs":
            for p in decl:
                if "required" in p and not isinstance(p["required"], bool):
                    raise ValueError(f"input 'required' must be bool: {p!r}")
    return ports


def load_vocabulary() -> dict:
    with open(TYPES_PATH) as f:
        types = json.load(f)
    vocab = types.get("types")
    if not isinstance(vocab, dict) or not vocab:
        raise ValueError("types.json must carry a non-empty 'types' object")
    return vocab


def check_types_in_vocabulary(ports: dict, vocab: dict) -> None:
    """Check 2: every declared port type is a name in the vocabulary."""
    missing = []
    for side in ("inputs", "outputs"):
        for p in ports[side]:
            if p["type"] not in vocab:
                missing.append((side, p["name"], p["type"]))
    if missing:
        lines = "\n".join(f"  - {s} port '{n}' -> unknown type '{t}'" for s, n, t in missing)
        raise ValueError(
            "ports reference types absent from the vocabulary (types.json):\n" + lines
        )


def _load_samples() -> list[dict]:
    samples = []
    if not os.path.isdir(SAMPLES_DIR):
        raise ValueError("no samples/ directory to ground the port check against")
    for fn in sorted(os.listdir(SAMPLES_DIR)):
        if fn.endswith(".json"):
            with open(os.path.join(SAMPLES_DIR, fn)) as f:
                samples.append({"file": fn, "payload": json.load(f)})
    if not samples:
        raise ValueError("samples/ is empty — cannot ground the port check")
    return samples


def check_reads_and_writes(ports: dict) -> None:
    """Check 3: decide reads each declared input and writes each declared output.

    Grounded against the committed samples:
      - the union of every sample's `state` keys must equal the declared input
        names (no undeclared input is fed; no declared input is never exercised);
      - the union of every output dict's keys produced by running the samples
        must equal the declared output names;
      - additionally, each declared input name must be referenced in organ.py
        (so a key a sample happens to carry but the organ ignores can't pass).
    """
    # Import the organ lazily so a syntax error there surfaces here too.
    sys.path.insert(0, HERE)
    from organ import decide  # noqa: E402

    samples = _load_samples()
    declared_inputs = {p["name"] for p in ports["inputs"]}
    declared_outputs = {p["name"] for p in ports["outputs"]}

    seen_input_keys: set[str] = set()
    seen_output_keys: set[str] = set()
    for s in samples:
        state = s["payload"].get("state", {})
        if isinstance(state, dict):
            seen_input_keys |= set(state.keys())
        result = decide(state, s["payload"].get("context"))
        out = result.get("output")
        if isinstance(out, dict):
            seen_output_keys |= set(out.keys())

    problems = []
    undeclared_in = seen_input_keys - declared_inputs
    if undeclared_in:
        problems.append(f"sample state keys not declared as inputs: {sorted(undeclared_in)}")
    unexercised_in = declared_inputs - seen_input_keys
    if unexercised_in:
        problems.append(f"declared inputs never present in any sample: {sorted(unexercised_in)}")

    undeclared_out = seen_output_keys - declared_outputs
    if undeclared_out:
        problems.append(f"output keys produced but not declared: {sorted(undeclared_out)}")
    unwritten_out = declared_outputs - seen_output_keys
    if unwritten_out:
        problems.append(f"declared outputs never written by any sample: {sorted(unwritten_out)}")

    src = open(ORGAN_PATH).read()
    not_referenced = [n for n in declared_inputs if n not in src]
    if not_referenced:
        problems.append(f"declared inputs not referenced in organ.py: {sorted(not_referenced)}")

    if problems:
        raise ValueError("read/write port grounding failed:\n  - " + "\n  - ".join(problems))


def run_all() -> None:
    ports = load_ports()
    vocab = load_vocabulary()
    check_types_in_vocabulary(ports, vocab)
    check_reads_and_writes(ports)


def main() -> int:
    try:
        run_all()
    except Exception as e:  # noqa: BLE001 — surface any failure as a red check
        print(f"PORT CHECK FAILED: {e}", file=sys.stderr)
        return 1
    print("Port manifest OK: ports.json parses, every type is in the vocabulary, "
          "and decide reads/writes every declared name.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
