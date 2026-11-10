from pathlib import Path

import numpy as np
import pytest
import torch

from unisupermimic.deploy.export import export_onnx, load_policy
from unisupermimic.rl.algos.supermimic.networks import Policy


def test_checkpoint_loader_rejects_changed_contract(tmp_path: Path):
    policy = Policy(99, 730, hidden=(16, 16))
    good = tmp_path / "good.pt"
    torch.save(
        {
            "pi": policy.state_dict(),
            "args": {"goal_frames": 10, "hist": 10, "z_dim": 32, "policy_hidden": [16, 16]},
        },
        good,
    )
    loaded, args = load_policy(good)
    assert args["goal_frames"] == 10
    assert loaded.hist == 10

    bad = tmp_path / "bad.pt"
    torch.save(
        {
            "pi": policy.state_dict(),
            "args": {"goal_frames": 5, "hist": 10, "z_dim": 32, "policy_hidden": [16, 16]},
        },
        bad,
    )
    with pytest.raises(ValueError, match="goal_frames"):
        load_policy(bad)


def test_onnx_export_contract(tmp_path: Path):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    policy = Policy(99, 730, hidden=(16, 16))
    ckpt = tmp_path / "ckpt_000001.pt"
    torch.save(
        {
            "pi": policy.state_dict(),
            "args": {"goal_frames": 10, "hist": 10, "z_dim": 32, "policy_hidden": [16, 16]},
        },
        ckpt,
    )
    result = export_onnx(ckpt, tmp_path)
    assert result["io"] == [
        ("obs", [1, 990]),
        ("goal", [1, 730]),
        ("action", [1, 29]),
    ]
    assert result["max_abs_action_diff"] < 2e-4
    assert np.isfinite(result["max_abs_action_diff"])
