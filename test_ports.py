"""Pytest wrapper for the port-manifest conformance check.

Mirrors check_ports.py so the existing `python -m pytest` step covers ports
too. The conformance workflow also runs `python3 check_ports.py` directly.
"""

import check_ports


def test_ports_json_parses_and_shape():
    ports = check_ports.load_ports()
    assert ports["inputs"] and ports["outputs"]


def test_every_port_type_is_in_the_vocabulary():
    ports = check_ports.load_ports()
    vocab = check_ports.load_vocabulary()
    check_ports.check_types_in_vocabulary(ports, vocab)


def test_decide_reads_and_writes_declared_names():
    ports = check_ports.load_ports()
    check_ports.check_reads_and_writes(ports)


def test_run_all_green():
    # The exact aggregate the conformance Action runs.
    check_ports.run_all()
