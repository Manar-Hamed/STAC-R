import json

import pytest
import yaml

from stacr.utils import parse_override


@pytest.mark.parametrize("value", ["on", "off", "ON", "OFF", "yes", "no", "Yes", "No"])
def test_override_enum_values_remain_strings(value):
    assert parse_override(value) == value


@pytest.mark.parametrize(
    "text,expected",
    [
        ("true", True),
        ("FALSE", False),
        ("True", True),
        ("42", 42),
        ("-3", -3),
        ("0.125", 0.125),
        ("1.0e-4", 1e-4),
        ("null", None),
        ('"false"', "false"),
    ],
)
def test_override_keeps_boolean_numeric_null_and_quoted_types(text, expected):
    actual = parse_override(text)
    assert actual == expected
    assert type(actual) is type(expected)


def test_override_rules_apply_inside_mappings_and_lists_without_mutating_yaml():
    result = parse_override(
        "{on: off, flags: [on, off, true, false], nested: {enabled: true, count: 4, rate: 0.25}}"
    )
    assert result == {
        "on": "off",
        "flags": ["on", "off", True, False],
        "nested": {"enabled": True, "count": 4, "rate": 0.25},
    }
    # File loading and unrelated PyYAML users retain their existing resolver.
    assert yaml.safe_load("off") is False
    assert yaml.safe_load("{on: off}") == {True: False}


def test_generic_cli_uses_override_rules_in_nested_compression(tmp_path, capsys):
    from stacr.cli import main

    config = tmp_path / "model.yaml"
    config.write_text("task: synthetic\nengine: adaptive\n", encoding="utf-8")
    main(
        [
            "train",
            "--config",
            str(config),
            "--dry-run",
            "--set",
            "compression={refresh_trigger: off, quantize_coefficients: false, rank: 8}",
            "--set",
            "target_names=[on, off]",
            "--set",
            "local_files_only=true",
            "--set",
            "steps=7",
        ]
    )
    resolved = json.loads(capsys.readouterr().out)
    assert resolved["compression"] == {
        "refresh_trigger": "off",
        "quantize_coefficients": False,
        "rank": 8,
    }
    assert resolved["target_names"] == ["on", "off"]
    assert resolved["local_files_only"] is True
    assert resolved["steps"] == 7


def test_recorded_cli_keeps_c1_off_enum_and_nested_baseline_types(capsys):
    from stacr.cli import main

    main(
        [
            "reproduce",
            "run",
            "checkpoint-boundary-wide-compressed",
            "--seeds",
            "42",
            "--dry-run",
            "--set",
            "stacr_ckpt_planner=off",
            "--set",
            "stacr_ckpt_refresh_trigger=off",
            "--set",
            "stacr_ckpt_quantize_coefficients=true",
            "--set",
            "baseline_cfg={checkpointing: transformer_hf, amp: bf16, grad_accum_steps: 2}",
        ]
    )
    resolved = json.loads(capsys.readouterr().out)["runs"][0]
    assert resolved["stacr_ckpt_planner"] == "off"
    assert resolved["stacr_ckpt_refresh_trigger"] == "off"
    assert resolved["stacr_ckpt_quantize_coefficients"] is True
    assert resolved["baseline_cfg"]["grad_accum_steps"] == 2
    assert resolved["baseline_cfg"]["amp"] == "bf16"


def test_override_parser_rejects_unsafe_yaml_tags():
    with pytest.raises(yaml.constructor.ConstructorError):
        parse_override("!!python/tuple [on, off]")
