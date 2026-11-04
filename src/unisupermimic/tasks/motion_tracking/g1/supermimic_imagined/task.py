"""Asset-owned task inputs for pure frozen-world-model SuperMimic training.

This module deliberately has no UniLab environment dependency. It constructs only
the differentiable analytic MDP needed to roll the frozen world model and score
imagined states.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from unisupermimic.rl.algos.supermimic.networks import WorldModel

from ..supermimic.kinematics import G1Kinematics
from ..supermimic.mdp import MDP, RewardCfg
from ..supermimic.motion import MotionLib
from ..supermimic.robot import default_joint_pos_list


@dataclass(frozen=True)
class G1ImaginedConfig:
    """Configuration for the offline G1 imagination task."""

    motion_files: tuple[str, ...]
    mjcf_path: str
    src_fps: float = 30.0
    gamma: float = 0.99
    reset_noise: bool = True
    w_amp: float = 0.0
    amp_tol: float = 0.02
    amp_scale: float = 0.5
    amp_min_height: float = 0.8
    amp_ankles: tuple[int, int] = (3, 6)
    w_speed: float = 0.0
    speed_min: float = 2.0
    speed_tol: float = 0.3
    speed_scale: float = 1.5
    speed_bodies: tuple[int, int, int, int] = (3, 6, 9, 11)


@dataclass
class G1ImaginedTask:
    """Device-resident analytic MDP plus frozen world-model metadata."""

    mdp: MDP
    motion: MotionLib
    world_model: WorldModel
    wm_hidden: tuple[int, ...]
    wm_structured: bool


def build_g1_imagined_task(
    cfg: G1ImaginedConfig,
    checkpoint: dict[str, object],
    device: str | torch.device,
) -> G1ImaginedTask:
    if not cfg.motion_files:
        raise ValueError("G1SuperMimicImagined requires at least one motion file")

    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("wm"), dict):
        raise TypeError("the frozen checkpoint must be a mapping containing a 'wm' state")

    state = checkpoint["wm"]
    if not isinstance(state, dict):
        raise TypeError("checkpoint['wm'] must be a state-dict mapping")

    expected_input = 13 + 3 * 29 + 33
    expected_output = 12 + 2 * 29
    layers: dict[int, tuple[int, int]] = {}
    valid = True
    for name, value in state.items():
        parts = name.split(".")
        if not name.startswith("net.") or not name.endswith(".weight"):
            continue
        # LayerNorm weights are 1-D and are not hidden-layer Linear matrices.
        if value.ndim != 2:
            continue
        if len(parts) != 3 or not parts[1].isdigit() or value.ndim != 2:
            valid = False
            break
        layers[int(parts[1])] = tuple(value.shape)
    ordered = sorted(layers.items())
    hidden = tuple(shape[0] for _, shape in ordered[:-1])
    if (
        not valid
        or not hidden
        or ordered[0][1][1] != expected_input
        or ordered[-1][1] != (expected_output, hidden[-1])
    ):
        raise ValueError("cannot infer a G1-compatible world model from the checkpoint")

    # Every current WorldModel has an inc_norm buffer; only structured checkpoints
    # carry non-default target statistics for it. Presence plus a non-identity
    # variance is a stable discriminator without checkpoint schema changes.
    structured = any(
        key.startswith("inc_norm.") and key.endswith(".var") and not torch.all(state[key] == 1)
        for key in state
    )

    device = torch.device(device)
    kin = G1Kinematics(cfg.mjcf_path, device)
    motion = MotionLib(cfg.motion_files, kin, device, src_fps=cfg.src_fps)
    default_q = torch.tensor(default_joint_pos_list(), dtype=torch.float32, device=device)
    mdp = MDP(
        kin,
        motion,
        default_q,
        RewardCfg(
            w_amp=cfg.w_amp,
            amp_tol=cfg.amp_tol,
            amp_scale=cfg.amp_scale,
            amp_min_height=cfg.amp_min_height,
            amp_ankles=cfg.amp_ankles,
            w_speed=cfg.w_speed,
            speed_min=cfg.speed_min,
            speed_tol=cfg.speed_tol,
            speed_scale=cfg.speed_scale,
            speed_bodies=cfg.speed_bodies,
        ),
    )
    wm = WorldModel(
        hidden=hidden,
        num_joints=mdp.num_joints,
        context_dim=mdp.context_dim,
        structured=structured,
    ).to(device)
    wm.load_state_dict(state, strict=True)
    wm.requires_grad_(False)
    wm.eval()

    return G1ImaginedTask(mdp, motion, wm, hidden, structured)


__all__ = ["G1ImaginedConfig", "G1ImaginedTask", "build_g1_imagined_task"]
