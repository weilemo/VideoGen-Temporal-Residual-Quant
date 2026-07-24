import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("hy_worldplay_runner", ROOT / "runner.py")
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
assert SPEC.loader is not None
SPEC.loader.exec_module(runner)


@pytest.fixture
def config():
    return json.loads((ROOT / "config.json").read_text(encoding="utf-8"))


@pytest.fixture
def actions():
    return json.loads((ROOT / "actions.json").read_text(encoding="utf-8"))


def test_all_variants_keep_controlled_variables_fixed(config, actions):
    manifests = [
        runner.build_command(config, actions, variant, "turn_left")[1]
        for variant in runner.VARIANTS
    ]
    reference = manifests[0]["controlled_variables"]
    assert all(item["controlled_variables"] == reference for item in manifests)
    assert {item["quant_type"] for item in manifests} == {
        "none",
        "trq-int4",
        "trq-int2",
        "packed-naive-int4",
        "packed-naive-int2",
    }


def test_counterfactual_actions_change_pose_only(config, actions):
    left_command, left = runner.build_command(config, actions, "bf16", "turn_left")
    right_command, right = runner.build_command(config, actions, "bf16", "turn_right")
    assert left["controlled_variables"] == right["controlled_variables"]
    assert left["pose"] != right["pose"]
    left_pose_index = left_command.index("--pose") + 1
    right_pose_index = right_command.index("--pose") + 1
    assert left_command[left_pose_index] == "w-8,a-8,w-8"
    assert right_command[right_pose_index] == "w-8,d-8,w-8"


def test_temporal_partition_must_match_prediction(config, actions):
    config["temporal_context_size"] = 43
    with pytest.raises(ValueError, match="must equal pred_latent_size"):
        runner.build_command(config, actions, "bf16", "canonical")


def test_nonzero_seed_is_rejected_until_upstream_exposes_it(config, actions):
    config["seed"] = 1
    with pytest.raises(ValueError, match="hard-codes seed 0"):
        runner.build_command(config, actions, "bf16", "canonical")


def test_unknown_action_is_rejected(config, actions):
    with pytest.raises(ValueError, match="unknown action"):
        runner.build_command(config, actions, "bf16", "teleport")


def test_source_override(monkeypatch, config):
    monkeypatch.setenv("HY_WORLDPLAY_SOURCE", "/tmp/hy-worldplay")
    assert runner.source_root(config) == Path("/tmp/hy-worldplay").resolve()
