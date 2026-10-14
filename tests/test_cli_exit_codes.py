import runpy
import sys

import pytest


def test_module_cli_propagates_delegated_failure(monkeypatch):
    from stacr import cli

    monkeypatch.setattr(cli, "main", lambda: 7)
    with pytest.raises(SystemExit) as result:
        runpy.run_module("stacr", run_name="__main__")
    assert result.value.code == 7


def test_protocol_cli_propagates_transfer_failure(monkeypatch):
    from experiments.protocols import openwebtext_transfer

    monkeypatch.setattr(openwebtext_transfer, "main", lambda args: 3)
    monkeypatch.setattr(sys, "argv", ["reproduce", "openwebtext"])
    with pytest.raises(SystemExit) as result:
        runpy.run_module("experiments.protocols.__main__", run_name="__main__")
    assert result.value.code == 3
