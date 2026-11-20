"""Measure the manager-vs-direct collect-step cost on the real boneseed workload.

Builds two envs from the same training config (boneseed library, 4096 envs,
mjwarp, masked autoreset + reset graph + step-path compilation):
  1. SuperMimicManagerBasedRlEnv — the shipped manager-based path
  2. DirectCollectEnv            — same env, but the step compute bypasses the
     manager orchestration and calls the (compiled) term functions directly
Everything else (TorchEnv.step, autoreset graph, action manager, reset
lifecycle) is inherited unchanged, so the delta isolates the manager layer.
"""

from __future__ import annotations

import sys
import time
from dataclasses import replace
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from unilab.base.backend_factory import create_backend, env_backend_kwargs
from unilab.base.entity import EntityCfg

from unisupermimic.rl.contracts.supermimic import SUPERMIMIC_TRANSITION_KEY
from unisupermimic.tasks.motion_tracking.g1.supermimic.env import (
    SuperMimicManagerBasedRlEnv,
    make_supermimic_env,
)
from unisupermimic.tasks.motion_tracking.g1.supermimic.manager_terms import (
    SuperMimicCommand,
    make_current_info,
    update_start_bin_returns,
)
from unisupermimic.tasks.motion_tracking.g1.supermimic.robot import COLLISION_GEOM_NAMES

NUM_ENVS = 4096
WARMUP = 30
STEPS = 200


class DirectCollectEnv(SuperMimicManagerBasedRlEnv):
    """Manager-free step compute: termination/reward/command/obs in one pass.

    ``fn`` selects the callable set: the instance's (default-)compiled bound
    methods, or reduce-overhead (CUDA-graph) rewraps of the class methods.
    """

    fn: dict[str, object] = {}

    def _update_state_in_read_phase(self, state):
        # One cudagraph-trees generation per step: outputs of graph-compiled
        # callables below must stay live until the last consumer in this step.
        torch.compiler.cudagraph_mark_step_begin()
        command = self._supermimic_command
        read_plan = self.scene.read_plan
        if read_plan is not None and not read_plan.ready:
            read_plan.refresh()
        self._advance_episode_steps(state.info["steps"])
        # --- termination + reward (compiled term functions, pre-advance f) ---
        st = self.fn["state"]()
        self._last_direct_state = st
        action = self.action_manager.action
        reward = self.fn["reward"](st, command.f + 1, action, command.a_prev)
        hard = self.fn["termination"](st, command.f + 1)
        timeout = command.ep_step + 1 >= command.max_episode_steps
        clip_end = command.f + 1 >= command.motion.clip_end[command.motion.clip_id[command.f]]
        time_outs = timeout | clip_end
        self.reset_terminated.copy_(hard)
        self.reset_time_outs.copy_(time_outs)
        torch.logical_or(self.reset_terminated, self.reset_time_outs, out=self.reset_buf)
        self.reward_buf = reward
        # --- command update (inline of SuperMimicCommand._update_command) ---
        command.ep_disc_ret.add_(command.gamma ** command.ep_step.float() * reward)
        command.ep_ret.add_(reward)
        command.f.add_(1)
        command.ep_step.add_(1)
        command.a_prev.copy_(action)
        if command.eval_mode:
            command.last_episode_stats = {}
        else:
            command.last_episode_stats = update_start_bin_returns(
                command.bin_G,
                command.start_bin,
                command.ep_step,
                command.ep_ret,
                command.ep_disc_ret,
                hard,
                self.reset_buf,
                command.gamma,
            )
        # --- observations (post-advance f, same order as the manager path) ---
        mapped = {
            "obs": self.fn["obs"](st, command.a_prev, command.xi, command.f),
            "critic": self.fn["goal"](st, command.f, command.cfg.goal_frames),
        }
        self._copy_step_outputs(state, mapped)
        self.obs_buf = state.obs
        self._has_transition = True
        # --- transition publication (same as the manager-path override) ---
        if self._pre_action is None:
            raise RuntimeError("direct update_state requires a preceding apply_action snapshot")
        transition = self._make_transition_info(
            self._pre_action["state"],
            st,
            self._pre_action["action"],
            self._pre_action["previous_action"],
            self._pre_action["context"],
            state.reward,
            self._pre_action["frames"],
            self._pre_action["ep_step"],
            state.terminated,
            state.truncated,
        )
        transition.update(command.last_episode_stats)
        state.info[SUPERMIMIC_TRANSITION_KEY] = transition
        self._pre_action = None
        return state

    def _current_info(self):
        # Snapshot from the same step's state read (graph outputs stay live
        # within the generation); avoids a second state() read. The inherited
        # reset path calls this before any direct step has run.
        st = getattr(self, "_last_direct_state", None)
        if st is None:
            return super()._current_info()
        return make_current_info(
            st,
            self._supermimic_command.f,
            self._supermimic_command.xi,
            self._supermimic_command.a_prev,
            self._supermimic_command.ep_step,
        )


def compose_override():
    GlobalHydra.instance().clear()
    with initialize_config_dir(
        config_dir=str(ROOT / "src" / "unisupermimic" / "conf" / "supermimic"),
        version_base="1.3",
    ):
        owner = compose(
            "config",
            overrides=[
                "task=g1_super_mimic/mjwarp",
                "env.commands.motion.motion_files=[data/library/bones_seed_compact.pt]",
            ],
        )
    from unilab.base.config_adapter import EnvConfigAdapter

    return EnvConfigAdapter(
        owner, root_dir=ROOT
    ).build_task_env_cfg_override()


def build_env_cls(env_cls):
    """Mirror make_supermimic_env for the prototype subclass."""
    from unilab.base import registry
    from unilab.base.config_materialization import apply_cfg_overrides
    from unilab.envs.manager_based_rl_env import ManagerBasedRlEnvCfg

    cfg = registry.materialize_env_config("G1SuperMimic")
    assert isinstance(cfg, ManagerBasedRlEnvCfg)
    apply_cfg_overrides(cfg, compose_override())
    assert cfg.scene is not None
    robot_cfg = cfg.scene.entities["robot"]
    assert isinstance(robot_cfg, EntityCfg)
    if robot_cfg.geom_names is None:
        cfg.scene.entities["robot"] = replace(robot_cfg, geom_names=COLLISION_GEOM_NAMES)
    cfg.validate()
    backend = create_backend(
        "mjwarp",
        cfg.scene,
        NUM_ENVS,
        cfg.sim_dt,
        body_state_required=True,
        base_name=robot_cfg.root_body_name,
        **env_backend_kwargs(cfg),
    )
    return env_cls(cfg, backend, NUM_ENVS)


def bench(env, action, label):
    env.seed(0)
    env.reset()
    generator = torch.Generator(device=env.device).manual_seed(123)
    for _ in range(WARMUP):
        env.step(torch.randn((NUM_ENVS, 29), generator=generator, device=env.device) * 0.3)
    done = 0
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(STEPS):
        state = env.step(action)
        done += int((state.terminated | state.truncated).sum())
    torch.cuda.synchronize()
    ms = (time.perf_counter() - started) / STEPS * 1000
    command = env.command_manager.get_term("motion")
    assert isinstance(command, SuperMimicCommand)
    print(
        f"{label}: {ms:.2f} ms/step | dones={done} | f_sum={int(command.f.sum())} "
        f"| ep_step_mean={float(command.ep_step.float().mean()):.3f}"
    )
    return ms, state


class DirectCaptureEnv(DirectCollectEnv):
    """Local compute-segment CUDA graph: capture the whole step compute.

    The only per-step-address-unstable input is the pre-action snapshot;
    persistent buffers replace its per-step clones. Everything else the
    segment touches already lives in fixed-address carriers.
    """

    _graph = None
    _warm = 0
    _pa = None

    def apply_action(self, actions, state):
        command = self._supermimic_command
        if self._pa is None:
            self._pa = {
                "state": tuple(value.detach().clone() for value in command.state()),
                "frames": command.f.detach().clone(),
                "context": command.xi.detach().clone(),
                "previous_action": command.a_prev.detach().clone(),
                "ep_step": command.ep_step.detach().clone(),
                "action": actions.detach().clone(),
            }
        else:
            for buffer, value in zip(self._pa["state"], command.state()):
                buffer.copy_(value)
            self._pa["frames"].copy_(command.f)
            self._pa["context"].copy_(command.xi)
            self._pa["previous_action"].copy_(command.a_prev)
            self._pa["ep_step"].copy_(command.ep_step)
            self._pa["action"].copy_(actions)
        self._pre_action = self._pa
        return SuperMimicManagerBasedRlEnv.__mro__[1].apply_action(self, actions, state)

    def _update_state_in_read_phase(self, state):
        if self._graph is None:
            if self._warm < 12:
                self._warm += 1
                return super()._update_state_in_read_phase(state)
            self._graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self._graph):
                super()._update_state_in_read_phase(state)
            return state
        self._graph.replay()
        return state


def _default_fn(env):
    command = env._supermimic_command
    return {
        "state": command.state,
        "obs": command.mdp.obs,
        "goal": command.mdp.goal,
        "reward": command.mdp.reward,
        "termination": command.mdp.termination,
    }


def _graph_fn(env, mode="reduce-overhead"):
    """Rewrap the class-level MDP methods with CUDA-graph compilation."""
    command = env._supermimic_command
    mdp_cls = type(command.mdp)
    cmd_cls = type(command)
    wrap = lambda fn: torch.compile(fn, mode=mode)  # noqa: E731
    return {
        "state": wrap(cmd_cls.state.__get__(command)),
        "obs": wrap(mdp_cls.obs.__get__(command.mdp)),
        "goal": wrap(mdp_cls.goal.__get__(command.mdp)),
        "reward": wrap(mdp_cls.reward.__get__(command.mdp)),
        "termination": wrap(mdp_cls.termination.__get__(command.mdp)),
    }


def main():
    from unilab.base import registry

    registry.ensure_registries()
    torch.manual_seed(0)
    action = (torch.randn((NUM_ENVS, 29), device="cuda") * 0.3).contiguous()

    manager_env = make_supermimic_env(_manager_cfg(), num_envs=NUM_ENVS, backend_type="mjwarp")
    ms_manager, state_m = bench(manager_env, action, "manager path")
    f_m = manager_env.command_manager.get_term("motion").f.clone()
    ms_manager2, _ = bench(manager_env, action, "manager path (repeat)")
    f_m2 = manager_env.command_manager.get_term("motion").f.clone()
    print(f"manager-vs-manager f max|diff| (chaos baseline) = {int((f_m - f_m2).abs().max())}")

    direct_env = build_env_cls(DirectCollectEnv)
    direct_env.fn = _default_fn(direct_env)
    ms_direct, state_d = bench(direct_env, action, "direct  path")

    capture_env = build_env_cls(DirectCaptureEnv)
    capture_env.fn = _default_fn(capture_env)
    ms_capture, state_c = bench(capture_env, action, "direct+segment capture   ")

    print(
        f"direct speedup: {ms_manager / ms_direct:.2f}x | "
        f"segment-capture speedup: {ms_manager / ms_capture:.2f}x "
        f"({ms_manager:.2f} -> {ms_direct:.2f} -> {ms_capture:.2f} ms)"
    )
    f_d = direct_env.command_manager.get_term("motion").f
    print(
        f"manager-vs-direct f max|diff| = {int((f_m - f_d).abs().max())} (chaos-level if ~baseline)"
    )
    for env in (manager_env, direct_env, capture_env):
        env.close()
        env.backend.close()


def _manager_cfg():
    from unilab.base import registry
    from unilab.base.config_materialization import apply_cfg_overrides
    from unilab.envs.manager_based_rl_env import ManagerBasedRlEnvCfg

    cfg = registry.materialize_env_config("G1SuperMimic")
    assert isinstance(cfg, ManagerBasedRlEnvCfg)
    apply_cfg_overrides(cfg, compose_override())
    return cfg


main()
