"""Process-isolated collector for async SuperMimic training.

Transport follows UniLab's verified IPC mechanisms instead of ad-hoc
thread/stream sharing:

- ``TransitionRing`` mirrors ``unilab.rl.runtime.ipc.TensorRolloutRingBuffer``:
  per-field shared-memory torch tensors with the write/read pointer protocol
  (``signal_write_done`` / ``available`` / ``wait_for_data`` / ``advance_read``).
  Field dtypes are preserved (``f`` stays int64: BONES-SEED has 47.3M frames,
  beyond the exact integer range of fp32).
- ``SharedWeightSync`` (unilab.rl.runtime.ipc.weight_sync) broadcasts the
  policy encoder/decoder weights learner -> collector once per iteration.
- The collector runs in a spawn process with its own CUDA context, so
  DEVICE_RESIDENT backends (mjwarp) cannot race with the learner's streams;
  every cross-device transfer is an explicit copy at this boundary.
"""

from __future__ import annotations

import multiprocessing as mp
import time
from multiprocessing import shared_memory
from typing import Any

import torch
from unilab.rl.runtime.ipc.weight_sync import SharedWeightSync

_SPAWN_CTX = mp.get_context("spawn")

# SuperMimic transition fields: (dim, dtype). Mirrors replay.FIELDS plus the
# episode statistics carried per step (full per-env rows, zero where not done).
TRANSITION_FIELDS: dict[str, tuple[int | tuple[()], torch.dtype]] = {
    "s": (71, torch.float32),
    "s1": (71, torch.float32),
    "a": (29, torch.float32),
    "a_prev": (29, torch.float32),
    "xi": (33, torch.float32),
    "r": ((), torch.float32),
    "f": ((), torch.int64),
    "ep_step": ((), torch.int32),
    "done": ((), torch.bool),
    "term": ((), torch.bool),
    "ep_len": ((), torch.float32),
    "ep_ret_norm": ((), torch.float32),
}
# Norm snapshot fields (per packet): (dim source, dtype). Counts stay fp32
# exactly as RunningNorm stores them.
NORM_FIELDS: dict[str, tuple[str | tuple[()], torch.dtype]] = {
    "obs_mean": ("obs", torch.float32),
    "obs_var": ("obs", torch.float32),
    "goal_mean": ("goal", torch.float32),
    "goal_var": ("goal", torch.float32),
    "obs_count": ((), torch.float32),
    "goal_count": ((), torch.float32),
}


class TransitionRing:
    """Bounded single-writer/single-reader packet ring in shared memory.

    One slot carries one collection packet: up to ``max_steps`` env steps of
    transitions plus a policy-normalizer snapshot and scalar metadata.
    """

    def __init__(
        self,
        num_envs: int,
        max_steps: int,
        obs_dim: int,
        goal_dim: int,
        *,
        num_slots: int = 4,
        create: bool = True,
        shm_names: dict[str, str] | None = None,
        write_ptr: Any = None,
        read_ptr: Any = None,
        field_dims: dict[str, int] | None = None,
    ) -> None:
        self.num_envs = num_envs
        self.max_steps = max_steps
        self.num_slots = num_slots
        self._create = create
        self._shm_blocks: dict[str, shared_memory.SharedMemory] = {}
        self._tensors: dict[str, torch.Tensor] = {}
        dims = field_dims or {}
        fields: dict[str, tuple[int | tuple[()], torch.dtype]] = dict(TRANSITION_FIELDS)
        for name, (dim, dtype) in fields.items():
            if name in dims:
                dim = dims[name]
            dim_tuple = (dim,) if isinstance(dim, int) else dim
            self._alloc(
                f"tr:{name}",
                (num_slots, max_steps, num_envs, *dim_tuple),
                dtype,
                create,
                shm_names,
            )
        for name, (kind, dtype) in NORM_FIELDS.items():
            dim = obs_dim if kind == "obs" else goal_dim if kind == "goal" else 1
            self._alloc(f"norm:{name}", (num_slots, dim), dtype, create, shm_names)
        self._alloc("meta:n_steps", (num_slots,), torch.int64, create, shm_names)
        self._alloc("meta:has_ep", (num_slots, max_steps), torch.int64, create, shm_names)

        if create:
            self._write_ptr = _SPAWN_CTX.Value("l", 0)
            self._read_ptr = _SPAWN_CTX.Value("l", 0)
        else:
            self._write_ptr = write_ptr
            self._read_ptr = read_ptr

    def _alloc(self, key, shape, dtype, create, shm_names) -> None:
        count = 1
        for extent in shape:
            count *= int(extent)
        size = max(count * torch.empty((), dtype=dtype).element_size(), 1)
        if create:
            shm = shared_memory.SharedMemory(create=True, size=size)
        else:
            shm = shared_memory.SharedMemory(name=shm_names[key], create=False)
        self._shm_blocks[key] = shm
        self._tensors[key] = torch.frombuffer(shm.buf, dtype=dtype, count=count).reshape(shape)

    @property
    def shm_names(self) -> dict[str, str]:
        return {key: shm.name for key, shm in self._shm_blocks.items()}

    @property
    def write_ptr(self) -> Any:
        return self._write_ptr

    @property
    def read_ptr(self) -> Any:
        return self._read_ptr

    # ---- writer side (collector process) ----

    @property
    def write_slot(self) -> int:
        return int(self._write_ptr.value) % self.num_slots

    def write_buffer_views(self) -> dict[str, torch.Tensor]:
        slot = self.write_slot
        return {key: tensor[slot] for key, tensor in self._tensors.items()}

    def signal_write_done(self) -> None:
        with self._write_ptr.get_lock():
            self._write_ptr.value += 1

    # ---- reader side (learner process) ----

    def _clamp_read_ptr(self) -> None:
        write = int(self._write_ptr.value)
        oldest = max(0, write - self.num_slots)
        if int(self._read_ptr.value) >= oldest:
            return
        with self._read_ptr.get_lock():
            if int(self._read_ptr.value) < oldest:
                self._read_ptr.value = oldest

    def available(self) -> int:
        self._clamp_read_ptr()
        return min(max(0, int(self._write_ptr.value) - int(self._read_ptr.value)), self.num_slots)

    def wait_for_data(self, timeout: float = 600.0) -> bool:
        deadline = time.monotonic() + timeout
        while self.available() == 0:
            if time.monotonic() > deadline:
                return False
            time.sleep(0.001)
        return True

    def read_tensor_views(self) -> dict[str, torch.Tensor]:
        self._clamp_read_ptr()
        slot = int(self._read_ptr.value) % self.num_slots
        return {key: tensor[slot] for key, tensor in self._tensors.items()}

    def advance_read(self) -> None:
        with self._read_ptr.get_lock():
            write = int(self._write_ptr.value)
            next_read = min(int(self._read_ptr.value) + 1, write)
            self._read_ptr.value = max(next_read, max(0, write - self.num_slots))

    def cleanup(self) -> None:
        for shm in self._shm_blocks.values():
            try:
                shm.close()
                if self._create:
                    shm.unlink()
            except Exception:
                pass

    def close(self) -> None:
        for shm in self._shm_blocks.values():
            try:
                shm.close()
            except Exception:
                pass


def weight_state_subset(pi: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Encoder/decoder parameters (norm buffers flow collector -> learner)."""
    return {
        name: param.detach().cpu().clone()
        for name, param in pi.state_dict().items()
        if name.startswith(("enc.", "dec."))
    }


class CollectActor:
    """Env-side policy replica: mirrors ``SuperMimicLearner._collect_step``."""

    def __init__(self, env: Any, pi: torch.nn.Module, cfg: Any):
        self.env = env
        self.pi = pi
        self.cfg = cfg
        self.mdp = env.supermimic_mdp
        self.device = torch.device(env.device)
        self.obs_hist = torch.zeros(env.num_envs, cfg.hist, self.mdp.obs_dim, device=self.device)

    def load_weights(self, ws: SharedWeightSync) -> None:
        subset = weight_state_subset(self.pi)
        ws.read_weights_into(subset)
        self.pi.load_state_dict(subset, strict=False)

    def norm_snapshot(self) -> dict[str, torch.Tensor]:
        obs_norm, goal_norm = self.pi.obs_norm, self.pi.goal_norm
        return {
            "obs_mean": obs_norm.mean.detach().cpu().clone(),
            "obs_var": obs_norm.var.detach().cpu().clone(),
            "goal_mean": goal_norm.mean.detach().cpu().clone(),
            "goal_var": goal_norm.var.detach().cpu().clone(),
            "obs_count": obs_norm.count.detach().cpu().clone().reshape(1),
            "goal_count": goal_norm.count.detach().cpu().clone().reshape(1),
        }

    @torch.no_grad()
    def collect_step(self) -> dict[str, torch.Tensor]:
        from unisupermimic.rl.algos.supermimic.geometry import unflatten_state
        from unisupermimic.rl.contracts.supermimic import (
            SUPERMIMIC_STATE_KEY,
            SUPERMIMIC_TRANSITION_KEY,
        )

        env, mdp, cfg = self.env, self.mdp, self.cfg
        state = env.state
        if state is None:
            state = env.init_state()
        info: dict[str, torch.Tensor] = state.info[SUPERMIMIC_STATE_KEY]
        flat = unflatten_state(info["s"])
        obs = mdp.add_noise(mdp.obs(flat, info["a_prev"], info["xi"], info["f"]))
        fresh = info["ep_step"] == 0
        shifted = torch.cat((self.obs_hist[:, 1:], obs.unsqueeze(1)), 1)
        self.obs_hist = torch.where(
            fresh[:, None, None], obs.unsqueeze(1).expand_as(self.obs_hist), shifted
        )
        goal = mdp.goal(flat, info["f"], cfg.goal_frames)
        self.pi.obs_norm.update(obs)
        self.pi.goal_norm.update(goal)
        action = self.pi(self.obs_hist, goal)
        action = torch.clamp(action, -20, 20)
        step_state = env.step(action.to(self.device))
        transition = dict(step_state.info[SUPERMIMIC_TRANSITION_KEY])
        done = transition["done"]
        # Normalize episode statistics to full per-env rows (envs may emit
        # variable-length rows already masked to finished episodes).
        for key in ("ep_len", "ep_ret_norm"):
            value = transition.get(key)
            if value is None:
                value = torch.zeros(env.num_envs, device=done.device)
            elif value.shape[0] != env.num_envs:
                full = torch.zeros(env.num_envs, dtype=value.dtype, device=value.device)
                full[done] = value
                value = full
            transition[key] = value
        if "term" not in transition:
            transition["term"] = torch.zeros_like(done)
        return {key: value.detach().cpu() for key, value in transition.items()}


def collector_entry(
    *,
    env_spec: dict[str, Any],
    learner_cfg: dict[str, Any],
    weight_shapes: dict[str, tuple[int, ...]],
    weight_shm_name: str,
    weight_lock: Any,
    ring_names: dict[str, str],
    ring_dims: dict[str, Any],
    write_ptr: Any,
    read_ptr: Any,
    target_seq: Any,
    target_steps: Any,
    stop_flag: Any,
    error_conn: Any,
) -> None:
    """Spawn-process collector entry (top-level for pickling)."""
    from omegaconf import OmegaConf
    from unilab.base.config_adapter import create_env
    from unilab.base.process_device import configure_backend_process_device
    from unilab.rl.runtime.seed import apply_worker_seed
    from unilab.training import ensure_registries

    from unisupermimic.rl.algos.supermimic.learner import SuperMimicConfig
    from unisupermimic.rl.algos.supermimic.networks import Policy

    profiler = None
    try:
        if "env_factory" in env_spec:
            import importlib

            module_name, func_name = env_spec["env_factory"].rsplit(".", 1)
            env = getattr(importlib.import_module(module_name), func_name)()
        else:
            ensure_registries()
            apply_worker_seed(env_spec.get("seed"), torch_runtime=True, cuda=True)
            configure_backend_process_device(str(env_spec["sim_backend"]), str(env_spec["device"]))
            cfg = OmegaConf.create(env_spec["cfg_dict"])
            env = create_env(
                cfg,
                num_envs=int(env_spec["num_envs"]),
                env_cfg_override=env_spec["env_override"],
            )
        from unisupermimic.rl.algos.supermimic.learner import _maybe_start_profiler

        profiler = _maybe_start_profiler("collector")
        lcfg = SuperMimicConfig(**learner_cfg)
        mdp = env.supermimic_mdp
        goal_dim = mdp.goal_frame_dim * lcfg.goal_frames
        pi = Policy(
            mdp.obs_dim,
            goal_dim,
            hist=lcfg.hist,
            z_dim=lcfg.z_dim,
            num_joints=mdp.num_joints,
            hidden=lcfg.policy_hidden,
        ).to(env.device)
        if lcfg.test_hidden_dims is not None:
            from unisupermimic.rl.algos.supermimic.networks import mlp

            z_dim = pi.enc[-1].out_features
            pi.enc = mlp(goal_dim, lcfg.test_hidden_dims, z_dim).to(env.device)
            pi.dec = mlp(z_dim + mdp.obs_dim * lcfg.hist, lcfg.test_hidden_dims, mdp.num_joints).to(
                env.device
            )
        ws = SharedWeightSync(
            weight_shapes, create=False, shm_name=weight_shm_name, lock=weight_lock
        )
        actor = CollectActor(env, pi, lcfg)
        ring = TransitionRing(
            int(env_spec["num_envs"]),
            ring_dims["max_steps"],
            ring_dims["obs_dim"],
            ring_dims["goal_dim"],
            num_slots=ring_dims["num_slots"],
            create=False,
            shm_names=ring_names,
            write_ptr=write_ptr,
            read_ptr=read_ptr,
            field_dims=ring_dims.get("field_dims"),
        )
        my_seq = 0
        while int(stop_flag.value) == 0:
            target = int(target_seq.value)
            if target <= my_seq:
                time.sleep(0.0005)
                continue
            my_seq = target
            n_steps = int(target_steps.value)
            actor.load_weights(ws)
            slot = ring.write_buffer_views()
            for step in range(n_steps):
                transition = actor.collect_step()
                for name in TRANSITION_FIELDS:
                    slot[f"tr:{name}"][step].copy_(transition[name])
                slot["meta:has_ep"][step] = 1 if bool(transition["done"].any()) else 0
            for name in NORM_FIELDS:
                slot[f"norm:{name}"].copy_(actor.norm_snapshot()[name])
            slot["meta:n_steps"].fill_(n_steps)
            ring.signal_write_done()
        close = getattr(env, "close", None)
        if close is not None:
            close()
        backend = getattr(env, "backend", None)
        if backend is not None:
            backend.close()
        if profiler is not None:
            profiler.__exit__(None, None, None)
    except BaseException as exc:  # noqa: BLE001 - forwarded to the learner
        try:
            error_conn.send(repr(exc))
        except Exception:
            pass
        raise


class ProcCollectDriver:
    """Learner-side driver mirroring the threaded worker's dispatch/harvest."""

    def __init__(self, learner: Any, env_spec: dict[str, Any], learner_cfg: dict[str, Any]):
        self._learner = learner
        self._env_spec = env_spec
        self._learner_cfg = learner_cfg
        env = learner.env
        mdp = learner.mdp
        self.pending_steps = 0
        self._field_dims = {
            "s": learner.mdp.num_joints * 2 + 13,
            "s1": learner.mdp.num_joints * 2 + 13,
            "a": learner.mdp.num_joints,
            "a_prev": learner.mdp.num_joints,
            "xi": learner.mdp.context_dim,
        }
        max_steps = max(4, int(2.0 / learner.cfg.utd) + 1)
        self._ring = TransitionRing(
            env.num_envs,
            max_steps,
            mdp.obs_dim,
            mdp.goal_frame_dim * learner.cfg.goal_frames,
            create=True,
            field_dims=self._field_dims,
        )
        self._weight_shapes = {k: v.shape for k, v in weight_state_subset(learner.pi).items()}
        self._ws = SharedWeightSync.from_state_dict(weight_state_subset(learner.pi))
        self._target_seq = _SPAWN_CTX.Value("l", 0)
        self._target_steps = _SPAWN_CTX.Value("l", 0)
        self._stop_flag = _SPAWN_CTX.Value("l", 0)
        self._error_parent, error_child = _SPAWN_CTX.Pipe(duplex=False)
        ring_dims = {
            "max_steps": max_steps,
            "obs_dim": mdp.obs_dim,
            "goal_dim": mdp.goal_frame_dim * learner.cfg.goal_frames,
            "num_slots": self._ring.num_slots,
            "field_dims": self._field_dims,
        }
        self._proc = _SPAWN_CTX.Process(
            target=collector_entry,
            kwargs=dict(
                env_spec=env_spec,
                learner_cfg=learner_cfg,
                weight_shapes=self._weight_shapes,
                weight_shm_name=self._ws.name,
                weight_lock=self._ws._lock,
                ring_names=self._ring.shm_names,
                ring_dims=ring_dims,
                write_ptr=self._ring.write_ptr,
                read_ptr=self._ring.read_ptr,
                target_seq=self._target_seq,
                target_steps=self._target_steps,
                stop_flag=self._stop_flag,
                error_conn=error_child,
            ),
            daemon=True,
            name="supermimic-proc-collect",
        )

    def start(self) -> None:
        self._proc.start()

    def dispatch(self, n_steps: int) -> None:
        self.pending_steps = n_steps
        self._ws.write_weights(weight_state_subset(self._learner.pi))
        self._target_steps.value = n_steps
        with self._target_seq.get_lock():
            self._target_seq.value += 1

    def harvest(self) -> None:
        learner = self._learner
        while not self._ring.wait_for_data(timeout=2.0):
            self._check_child_error()
            if not self._proc.is_alive():
                raise RuntimeError("proc collector died without producing a packet")
        views = self._ring.read_tensor_views()
        n_steps = int(views["meta:n_steps"])
        for step in range(n_steps):
            transition = {
                name: views[f"tr:{name}"][step].to(learner.device) for name in TRANSITION_FIELDS
            }
            learner._apply_wm_norms(transition)
            learner.buffer.add_staged(transition)
            learner.stats["r"].append(transition["r"].mean() / learner.mdp.max_step_reward)
            if int(views["meta:has_ep"][step]):
                done = transition["done"]
                learner.stats["ep_len"].append(transition["ep_len"][done].mean())
                learner.stats["ep_ret"].append(transition["ep_ret_norm"][done].mean())
                learner.stats["term"].append(transition["term"].float().sum())
                learner.stats["done"].append(done.float().sum())
        learner.buffer.commit(n_steps)
        learner.sim_steps += n_steps * learner.env.num_envs
        pi = learner.pi
        pi.obs_norm.mean.copy_(views["norm:obs_mean"].to(learner.device))
        pi.obs_norm.var.copy_(views["norm:obs_var"].to(learner.device))
        pi.obs_norm.count.copy_(views["norm:obs_count"].reshape(()).to(learner.device))
        pi.goal_norm.mean.copy_(views["norm:goal_mean"].to(learner.device))
        pi.goal_norm.var.copy_(views["norm:goal_var"].to(learner.device))
        pi.goal_norm.count.copy_(views["norm:goal_count"].reshape(()).to(learner.device))
        self._ring.advance_read()
        self.pending_steps = 0
        self._check_child_error()

    def _check_child_error(self) -> None:
        if self._error_parent.poll():
            raise RuntimeError("proc collector failed: " + str(self._error_parent.recv()))

    def shutdown(self) -> None:
        self._stop_flag.value = 1
        self._proc.join(timeout=10.0)
        if self._proc.is_alive():
            self._proc.terminate()
            self._proc.join(timeout=5.0)
        self._ring.cleanup()
        self._ws.cleanup()
