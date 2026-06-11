"""
Pytest suite for the data-context organ.

Covers the four pure operations extracted from discovery-engine
app/services/data_context.py:
  - summary               (get_data_summary_for_interview formatting)
  - validate_gate         (validate_answer_against_data early guards)
  - normalize_validations (validate_answer_against_data post-processing)
  - normalize_questions   (generate_data_informed_questions capping)
plus dispatch + fail-safe behaviour.
"""

import json
import pytest
from organ import decide


def _ok(result):
    """Common shape assertions for every organ return."""
    assert set(result) == {"output", "rationale", "self_metric"}
    assert isinstance(result["rationale"], str)
    assert "confidence" in result["self_metric"]
    assert "decision_path" in result["self_metric"]
    assert 0.0 <= result["self_metric"]["confidence"] <= 1.0


# --------------------------------------------------------------------------- #
# op = summary
# --------------------------------------------------------------------------- #
class TestSummary:
    def test_single_source(self):
        state = {
            "op": "summary",
            "connections": [{
                "name": "Staff CSV",
                "summary": {
                    "source": "staff.csv",
                    "rows": 200,
                    "columns": [
                        {"name": "dept", "type": "string", "samples": ["Ops", "Sales"]},
                        {"name": "absence", "type": "float", "samples": [0.23, 0.05]},
                    ],
                },
            }],
        }
        result = decide(state)
        _ok(result)
        ctx = result["output"]["context"]
        assert ctx.startswith("CONNECTED DATA SOURCES:")
        assert "Data source: staff.csv" in ctx
        assert "Rows: 200" in ctx
        assert "- dept (string): e.g. Ops, Sales" in ctx
        assert "- absence (float): e.g. 0.23, 0.05" in ctx
        assert result["output"]["source_count"] == 1
        assert result["self_metric"]["decision_path"] == "summary_assembled"
        assert result["self_metric"]["confidence"] == 1.0

    def test_multiple_sources_joined(self):
        state = {
            "op": "summary",
            "connections": [
                {"name": "A", "summary": {"source": "a", "rows": 1, "columns": []}},
                {"name": "B", "summary": {"source": "b", "rows": 2, "columns": []}},
            ],
        }
        result = decide(state)
        _ok(result)
        assert result["output"]["source_count"] == 2
        assert "\n\n---\n\n" in result["output"]["context"]

    def test_null_summary_skipped(self):
        """A connection with null summary (unbuildable connector) is skipped."""
        state = {
            "op": "summary",
            "connections": [
                {"name": "broken", "summary": None},
                {"name": "good", "summary": {"source": "g", "rows": 3, "columns": []}},
            ],
        }
        result = decide(state)
        assert result["output"]["source_count"] == 1
        assert "Data source: g" in result["output"]["context"]

    def test_no_connections_empty_context(self):
        result = decide({"op": "summary", "connections": []})
        _ok(result)
        assert result["output"]["context"] == ""
        assert result["output"]["source_count"] == 0
        assert result["self_metric"]["decision_path"] == "summary_empty"

    def test_all_unbuildable_empty_context(self):
        state = {
            "op": "summary",
            "connections": [{"name": "x", "summary": None}, {"name": "y", "summary": None}],
        }
        result = decide(state)
        assert result["output"]["context"] == ""
        assert result["output"]["source_count"] == 0

    def test_fallback_to_name_when_source_missing(self):
        state = {
            "op": "summary",
            "connections": [{"name": "Fallback Name", "summary": {"rows": 5, "columns": []}}],
        }
        result = decide(state)
        assert "Data source: Fallback Name" in result["output"]["context"]

    def test_missing_rows_renders_question_mark(self):
        state = {
            "op": "summary",
            "connections": [{"name": "n", "summary": {"source": "s", "columns": []}}],
        }
        result = decide(state)
        assert "Rows: ?" in result["output"]["context"]

    def test_connections_not_list_is_empty(self):
        result = decide({"op": "summary", "connections": "nope"})
        assert result["output"]["context"] == ""
        assert result["self_metric"]["confidence"] == 0.5

    def test_missing_connections_key(self):
        result = decide({"op": "summary"})
        assert result["output"]["context"] == ""

    def test_non_dict_connection_skipped(self):
        state = {
            "op": "summary",
            "connections": ["junk", {"name": "ok", "summary": {"source": "o", "rows": 1, "columns": []}}],
        }
        result = decide(state)
        assert result["output"]["source_count"] == 1

    def test_non_dict_column_skipped(self):
        state = {
            "op": "summary",
            "connections": [{"name": "n", "summary": {
                "source": "s", "rows": 1, "columns": ["bad", {"name": "c", "type": "int", "samples": [1]}],
            }}],
        }
        result = decide(state)
        assert "- c (int): e.g. 1" in result["output"]["context"]


# --------------------------------------------------------------------------- #
# op = validate_gate
# --------------------------------------------------------------------------- #
class TestValidateGate:
    def test_no_data_skips(self):
        result = decide({"op": "validate_gate", "has_data": False, "answer_text": "x" * 50})
        _ok(result)
        assert result["output"]["should_validate"] is False
        assert result["output"]["reason"] == "no_data"
        assert result["self_metric"]["decision_path"] == "gate_no_data"

    def test_short_answer_skips(self):
        result = decide({"op": "validate_gate", "has_data": True, "answer_text": "too short"})
        assert result["output"]["should_validate"] is False
        assert result["output"]["reason"] == "answer_too_short"

    def test_boundary_19_chars_skips(self):
        result = decide({"op": "validate_gate", "has_data": True, "answer_text": "a" * 19})
        assert result["output"]["should_validate"] is False

    def test_boundary_20_chars_validates(self):
        result = decide({"op": "validate_gate", "has_data": True, "answer_text": "a" * 20})
        assert result["output"]["should_validate"] is True
        assert result["output"]["reason"] == "ok"

    def test_whitespace_stripped_before_length(self):
        result = decide({"op": "validate_gate", "has_data": True, "answer_text": "   short   "})
        assert result["output"]["should_validate"] is False

    def test_long_answer_with_data_validates(self):
        result = decide({
            "op": "validate_gate",
            "has_data": True,
            "answer_text": "We process about fifty orders a day on a good week.",
        })
        assert result["output"]["should_validate"] is True
        assert result["self_metric"]["decision_path"] == "gate_ok"

    def test_no_data_takes_priority_over_length(self):
        result = decide({"op": "validate_gate", "has_data": False, "answer_text": "tiny"})
        assert result["output"]["reason"] == "no_data"

    def test_non_string_answer_coerced(self):
        result = decide({"op": "validate_gate", "has_data": True, "answer_text": 12345})
        # str(12345) == "12345" -> 5 chars -> too short
        assert result["output"]["should_validate"] is False
        assert result["output"]["reason"] == "answer_too_short"


# --------------------------------------------------------------------------- #
# op = normalize_validations
# --------------------------------------------------------------------------- #
class TestNormalizeValidations:
    def test_basic_normalization(self):
        state = {"op": "normalize_validations", "validations": [
            {"claim": "50 orders", "data_shows": "200 orders", "source": "csv", "severity": "contradiction"},
        ]}
        result = decide(state)
        _ok(result)
        v = result["output"]["validations"][0]
        assert v == {"claim": "50 orders", "data_shows": "200 orders", "source": "csv", "severity": "contradiction"}
        assert result["output"]["dropped"] == 0

    def test_caps_at_five(self):
        state = {"op": "normalize_validations", "validations": [
            {"claim": f"c{i}"} for i in range(8)
        ]}
        result = decide(state)
        assert len(result["output"]["validations"]) == 5

    def test_drops_items_without_claim(self):
        state = {"op": "normalize_validations", "validations": [
            {"claim": "kept"},
            {"data_shows": "no claim here"},
            {"claim": ""},  # empty claim is falsy -> dropped
        ]}
        result = decide(state)
        assert len(result["output"]["validations"]) == 1
        assert result["output"]["dropped"] == 2

    def test_drops_non_dict(self):
        state = {"op": "normalize_validations", "validations": ["str", 7, {"claim": "ok"}]}
        result = decide(state)
        assert len(result["output"]["validations"]) == 1
        assert result["output"]["dropped"] == 2

    def test_invalid_severity_defaults_to_note(self):
        state = {"op": "normalize_validations", "validations": [{"claim": "c", "severity": "bogus"}]}
        result = decide(state)
        assert result["output"]["validations"][0]["severity"] == "note"

    def test_missing_severity_defaults_to_note(self):
        state = {"op": "normalize_validations", "validations": [{"claim": "c"}]}
        result = decide(state)
        assert result["output"]["validations"][0]["severity"] == "note"

    def test_all_valid_severities_preserved(self):
        for sev in ("contradiction", "discrepancy", "note"):
            state = {"op": "normalize_validations", "validations": [{"claim": "c", "severity": sev}]}
            result = decide(state)
            assert result["output"]["validations"][0]["severity"] == sev

    def test_text_fields_stringified(self):
        state = {"op": "normalize_validations", "validations": [
            {"claim": 99, "data_shows": 100, "source": 1},
        ]}
        result = decide(state)
        v = result["output"]["validations"][0]
        assert v["claim"] == "99" and v["data_shows"] == "100" and v["source"] == "1"

    def test_not_a_list_returns_empty(self):
        result = decide({"op": "normalize_validations", "validations": {"claim": "x"}})
        assert result["output"]["validations"] == []
        assert result["self_metric"]["confidence"] == 0.0
        assert result["self_metric"]["decision_path"] == "validations_not_list"

    def test_missing_fields_default_empty_strings(self):
        state = {"op": "normalize_validations", "validations": [{"claim": "only"}]}
        result = decide(state)
        v = result["output"]["validations"][0]
        assert v["data_shows"] == "" and v["source"] == ""

    def test_dropped_counts_only_within_cap(self):
        # 7 items, first 5 considered, 2 of those dropped -> dropped == 2
        vals = [{"claim": "a"}, {"x": 1}, {"claim": "b"}, {"y": 2}, {"claim": "c"}, {"claim": "d"}, {"claim": "e"}]
        result = decide({"op": "normalize_validations", "validations": vals})
        assert result["output"]["dropped"] == 2
        assert len(result["output"]["validations"]) == 3


# --------------------------------------------------------------------------- #
# op = normalize_questions
# --------------------------------------------------------------------------- #
class TestNormalizeQuestions:
    def test_caps_at_three(self):
        state = {"op": "normalize_questions", "questions": ["q1", "q2", "q3", "q4", "q5"]}
        result = decide(state)
        _ok(result)
        assert result["output"]["questions"] == ["q1", "q2", "q3"]
        assert result["output"]["count"] == 3

    def test_stringifies(self):
        result = decide({"op": "normalize_questions", "questions": [1, 2.5, None]})
        assert result["output"]["questions"] == ["1", "2.5", "None"]

    def test_fewer_than_three(self):
        result = decide({"op": "normalize_questions", "questions": ["only one"]})
        assert result["output"]["count"] == 1

    def test_empty_list(self):
        result = decide({"op": "normalize_questions", "questions": []})
        assert result["output"]["questions"] == []
        assert result["output"]["count"] == 0

    def test_not_a_list_returns_empty(self):
        result = decide({"op": "normalize_questions", "questions": "not a list"})
        assert result["output"]["questions"] == []
        assert result["self_metric"]["decision_path"] == "questions_not_list"


# --------------------------------------------------------------------------- #
# dispatch + fail-safe
# --------------------------------------------------------------------------- #
class TestDispatch:
    def test_unknown_op(self):
        result = decide({"op": "frobnicate"})
        _ok(result)
        assert result["output"] is None
        assert result["self_metric"]["decision_path"] == "unknown_op"
        assert result["self_metric"]["confidence"] == 0.0

    def test_missing_op(self):
        result = decide({})
        assert result["output"] is None
        assert result["self_metric"]["decision_path"] == "unknown_op"

    def test_state_not_dict_fail_safe(self):
        result = decide("not a dict")
        assert result["output"] is None
        assert result["self_metric"]["decision_path"] == "error_fallback"

    def test_context_arg_accepted_and_ignored(self):
        result = decide({"op": "normalize_questions", "questions": ["a"]}, {"anything": True})
        assert result["output"]["questions"] == ["a"]

    def test_never_raises_on_garbage(self):
        for garbage in [None, 42, [], "x", {"op": 123}, {"op": "summary", "connections": 7}]:
            result = decide(garbage)
            assert set(result) == {"output", "rationale", "self_metric"}


# --------------------------------------------------------------------------- #
# samples are self-consistent
# --------------------------------------------------------------------------- #
class TestSamples:
    def test_committed_samples_run(self):
        import glob
        import os
        here = os.path.dirname(__file__)
        sample_paths = sorted(glob.glob(os.path.join(here, "samples", "*.json")))
        assert sample_paths, "expected committed sample files"
        for p in sample_paths:
            with open(p) as f:
                payload = json.load(f)
            result = decide(payload["state"], payload.get("context"))
            assert set(result) == {"output", "rationale", "self_metric"}


# --------------------------------------------------------------------------- #
# samples are PINNED to their expected verdicts
#
# The conformance.yml "Shadow-run" step only PRINTS each sample's verdict to
# the job summary — it never asserts it, so a regression that flipped a
# verdict (or changed an output field) would slip a green CI. These assertions
# pin the decision_path AND the load-bearing output values per sample, so any
# drift in the organ's behaviour against its committed fixtures turns CI red.
# --------------------------------------------------------------------------- #
class TestSamplesConform:
    def _run(self, name):
        import os
        here = os.path.dirname(__file__)
        with open(os.path.join(here, "samples", name)) as f:
            payload = json.load(f)
        return decide(payload["state"], payload.get("context"))

    def test_normalize_questions_capped(self):
        r = self._run("normalize_questions_capped.json")
        assert r["self_metric"]["decision_path"] == "questions_normalized"
        assert r["output"]["count"] == 3
        # 5 input questions -> capped to first 3, all stringified
        assert r["output"]["questions"] == [
            "Your data shows a 23% absence rate on Fridays — what's happening on Fridays?",
            "The orders sheet records ~200 orders a day but you mentioned 50 — where's the gap?",
            "Finance has no entries after March — did that team move systems?",
        ]

    def test_normalize_validations_contradiction(self):
        r = self._run("normalize_validations_contradiction.json")
        assert r["self_metric"]["decision_path"] == "validations_normalized"
        vals = r["output"]["validations"]
        # middle entry has no 'claim' -> dropped; 2 survive
        assert len(vals) == 2
        assert r["output"]["dropped"] == 1
        assert vals[0]["severity"] == "contradiction"
        # "made-up-severity" is not in the enum -> coerced to "note"
        assert vals[1]["severity"] == "note"
        assert vals[1]["claim"] == "Fridays are quiet"

    def test_summary_two_sources(self):
        r = self._run("summary_two_sources.json")
        assert r["self_metric"]["decision_path"] == "summary_assembled"
        # 3 connections, one null summary -> 2 buildable blocks
        assert r["output"]["source_count"] == 2
        ctx = r["output"]["context"]
        assert ctx.startswith("CONNECTED DATA SOURCES:")
        assert "Data source: staff.csv" in ctx
        assert "Data source: orders.gsheet" in ctx
        assert "broken connector" not in ctx
        assert "\n\n---\n\n" in ctx

    def test_validate_gate_ok(self):
        r = self._run("validate_gate_ok.json")
        assert r["self_metric"]["decision_path"] == "gate_ok"
        assert r["output"]["should_validate"] is True
        assert r["output"]["reason"] == "ok"


# --------------------------------------------------------------------------- #
# ports.json — connection-standard port declaration
#
# These mirror check_ports.py so the contract is enforced by the pytest suite
# as well as the dedicated conformance step. ports.json declares the keys
# decide() reads from `state` (inputs) and writes under `output` (outputs);
# every type must exist in the vendored types.json vocabulary.
# --------------------------------------------------------------------------- #
class TestPorts:
    def _load(self, name):
        import os
        here = os.path.dirname(__file__)
        with open(os.path.join(here, name)) as f:
            return json.load(f)

    def test_check_ports_passes(self):
        """The standalone conformance check exits 0 against the committed manifests."""
        import check_ports
        assert check_ports.main() == 0

    def test_ports_shape(self):
        ports = self._load("ports.json")
        assert isinstance(ports["inputs"], list) and ports["inputs"]
        assert isinstance(ports["outputs"], list) and ports["outputs"]
        for i in ports["inputs"]:
            assert {"name", "type", "required"} <= set(i)
            assert isinstance(i["required"], bool)
        for o in ports["outputs"]:
            assert {"name", "type"} <= set(o)

    def test_every_type_in_vocabulary(self):
        ports = self._load("ports.json")
        vocab = set(self._load("types.json")["types"])
        for p in ports["inputs"] + ports["outputs"]:
            assert p["type"] in vocab, f"{p['name']}: {p['type']} not in {sorted(vocab)}"

    def test_declared_inputs_match_state_reads(self):
        import check_ports
        import os
        here = os.path.dirname(__file__)
        with open(os.path.join(here, "organ.py")) as f:
            reads = check_ports._state_get_literals(f.read())
        declared = {i["name"] for i in self._load("ports.json")["inputs"]}
        assert reads == declared

    def test_declared_outputs_match_produced(self):
        import check_ports
        produced = check_ports._produced_output_keys(decide)
        declared = {o["name"] for o in self._load("ports.json")["outputs"]}
        assert produced == declared
