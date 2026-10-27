"""Prepare BONES-SEED G1 motions for SuperMimic training."""

from __future__ import annotations

import argparse
import csv
import math
import multiprocessing as mp
import os
import tarfile
import tempfile
import time
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import numpy as np
import torch

from unisupermimic.tasks.motion_tracking.g1.supermimic.robot import G1_JOINT_NAMES

REPO_ID = "bones-studio/seed"
ARCHIVE_NAME = "g1.tar.gz"
BONES_HEADER_PREFIX = "Frame,root_translateX"
DEFAULT_MODEL = Path(__file__).resolve().parents[1] / "assets/robots/g1/supermimic.xml"
REF_DIM = 71

SONIC_KEYWORDS = [
    "bed", "bike", "chair", "climb", "com_up_50cm", "sitting", "step_on", "seat",
    "table", "_sit_", "sit_", "ladder", "crutch", "_bed_", "_ride_", "scooter",
    "stepdown", "box_hspu", "50cm_box_", "on_box", "fall_from", "on_1m", "form_box",
    "off_1m", "230m", "jump_over_obstacle_", "lift_crate_come_up_", "stair", "box_jump",
    "box_dips", "walking_on_edge", "push_obstacle",
]  # fmt: skip
SONIC_ACROBATICS = [
    "acrobatics_", "cartwheel", "handstand", "handstand_ff_", "safety_roll",
    "kozak_dance", "jump_to_shoulder_roll", "monkey_jump",
]  # fmt: skip
PAPER_EXTRA = ["_box_", "box_", "_stair", "jump_off_50cm", "off_50cm", "fence", "look_over"]
KEYWORDS = sorted(set(SONIC_KEYWORDS + SONIC_ACROBATICS + PAPER_EXTRA))

ELEVATED_Z = 0.20
ELEVATED_S = 0.6
PENETRATE_Z_HARD = -0.12
PENETRATE_Z_SOFT = -0.08
PENETRATE_FRAC = 0.80
FLOAT_Z = 0.02
FLOAT_S = 0.5
UPRIGHT_GZ = 0.9
STAND_Z = (0.60, 0.85)
STILL_V = 0.1
STILL_QD = 0.5


@dataclass
class ResampledMotion:
    fps: float
    root_pos: np.ndarray
    root_quat: np.ndarray
    joint_pos: np.ndarray
    root_lin_vel_w: np.ndarray
    root_ang_vel_w: np.ndarray
    joint_vel: np.ndarray

    @property
    def num_frames(self) -> int:
        return int(self.root_pos.shape[0])


@dataclass
class BodyStates:
    body_names: list[str]
    pos_w: np.ndarray
    quat_w: np.ndarray
    lin_vel_w: np.ndarray
    ang_vel_w: np.ndarray


def load_g1_model(xml_path: str | Path = DEFAULT_MODEL):
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(xml_path))  # pyright: ignore[reportAttributeAccessIssue]
    names = [model.joint(index).name for index in range(model.njnt)]
    if (
        model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE  # pyright: ignore[reportAttributeAccessIssue]
        or names[1:] != G1_JOINT_NAMES
    ):
        raise ValueError("BONES-SEED conversion requires the G1 free joint and natural joint order")
    if model.nq != 36 or model.nv != 35:
        raise ValueError(f"Unexpected G1 model dimensions: nq={model.nq}, nv={model.nv}")
    return model


def quat_xyzw_to_wxyz(quaternion: np.ndarray) -> np.ndarray:
    return quaternion[..., [3, 0, 1, 2]]


def quat_wxyz_to_xyzw(quaternion: np.ndarray) -> np.ndarray:
    return quaternion[..., [1, 2, 3, 0]]


def quat_mul(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = first.T
    w2, x2, y2, z2 = second.T
    return np.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        axis=-1,
    )


def quat_conjugate(quaternion: np.ndarray) -> np.ndarray:
    return quaternion * np.array([1.0, -1.0, -1.0, -1.0])


def rotvec_from_quat(quaternion: np.ndarray) -> np.ndarray:
    quaternion = quaternion / np.linalg.norm(quaternion, axis=-1, keepdims=True)
    quaternion = np.where(quaternion[..., :1] < 0, -quaternion, quaternion)
    vector = quaternion[..., 1:]
    magnitude = np.linalg.norm(vector, axis=-1)
    angle = 2.0 * np.arctan2(magnitude, quaternion[..., 0])
    scale = np.where(magnitude > 1e-9, angle / np.maximum(magnitude, 1e-9), 2.0)
    return vector * scale[..., None]


def quat_rotate_inverse(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    w, x, y, z = quaternion.T
    rotation = np.empty(quaternion.shape[:-1] + (3, 3), dtype=np.float64)
    rotation[..., 0, 0] = 1 - 2 * (y * y + z * z)
    rotation[..., 0, 1] = 2 * (x * y - w * z)
    rotation[..., 0, 2] = 2 * (x * z + w * y)
    rotation[..., 1, 0] = 2 * (x * y + w * z)
    rotation[..., 1, 1] = 1 - 2 * (x * x + z * z)
    rotation[..., 1, 2] = 2 * (y * z - w * x)
    rotation[..., 2, 0] = 2 * (x * z - w * y)
    rotation[..., 2, 1] = 2 * (y * z + w * x)
    rotation[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return np.einsum("...ji,...j->...i", rotation, vector)


def _euler_xyz_degrees_to_quat(euler: np.ndarray) -> np.ndarray:
    angles = np.deg2rad(euler) / 2.0
    cx, cy, cz = np.cos(angles).T
    sx, sy, sz = np.sin(angles).T
    w = cz * cy * cx + sz * sy * sx
    x = cz * cy * sx - sz * sy * cx
    y = cz * sy * cx + sz * cy * sx
    z = sz * cy * cx - cz * sy * sx
    return np.stack((x, y, z, w), axis=-1)


def load_csv(path: str | Path) -> np.ndarray:
    path = Path(path)
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader, [])
    if header and header[0] == "Frame":
        columns = header
        joint_columns = [column for column in columns if column.endswith("_joint_dof")]
        expected = [f"{name}_dof" for name in G1_JOINT_NAMES]
        if joint_columns != expected:
            raise ValueError(f"{path}: joint columns differ from the G1 natural order")
        indices = {name: columns.index(name) for name in columns}
        values = np.atleast_2d(np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float64))
        position = values[:, [indices[f"root_translate{axis}"] for axis in "XYZ"]] / 100.0
        euler = values[:, [indices[f"root_rotate{axis}"] for axis in "XYZ"]]
        quaternion = _euler_xyz_degrees_to_quat(euler)
        joints = np.deg2rad(values[:, [columns.index(name) for name in joint_columns]])
        raw = np.concatenate((position, quaternion, joints), axis=1)
    else:
        raw = np.atleast_2d(np.loadtxt(path, delimiter=",", dtype=np.float64))
    if raw.ndim != 2 or raw.shape[1] != 36 or raw.shape[0] < 2:
        raise ValueError(
            f"{path}: expected at least two rows and 36 motion columns, got {raw.shape}"
        )
    if not np.isfinite(raw).all():
        raise ValueError(f"{path}: motion contains non-finite values")
    return raw


def resample_motion(raw: np.ndarray, input_fps: float, output_fps: float) -> ResampledMotion:
    if (
        not math.isfinite(input_fps)
        or not math.isfinite(output_fps)
        or min(input_fps, output_fps) <= 0
    ):
        raise ValueError("Input and output fps must be finite and positive")
    n_in = raw.shape[0]
    duration = (n_in - 1) / input_fps
    n_out = math.floor(duration * output_fps + 1e-6) + 1
    times = np.arange(n_out, dtype=np.float64) / output_fps
    source = np.minimum(times * input_fps, n_in - 1)
    idx0 = np.floor(source).astype(np.int64)
    idx1 = np.minimum(idx0 + 1, n_in - 1)
    blend = (source - idx0)[:, None]

    quaternion = quat_xyzw_to_wxyz(raw[:, 3:7])
    quaternion /= np.linalg.norm(quaternion, axis=-1, keepdims=True)
    root_pos = raw[idx0, :3] * (1 - blend) + raw[idx1, :3] * blend
    joint_pos = raw[idx0, 7:] * (1 - blend) + raw[idx1, 7:] * blend
    q0, q1 = quaternion[idx0], quaternion[idx1]
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0, -q1, q1)
    dot = np.abs(dot)
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta = np.sin(theta)
    linear = dot > 0.9995
    safe_sin = np.where(linear, 1.0, sin_theta)
    w0 = np.where(linear, 1 - blend, np.sin((1 - blend) * theta) / safe_sin)
    w1 = np.where(linear, blend, np.sin(blend * theta) / safe_sin)
    root_quat = w0 * q0 + w1 * q1
    root_quat /= np.linalg.norm(root_quat, axis=-1, keepdims=True)

    dt = 1.0 / output_fps
    root_lin_vel = np.gradient(root_pos, dt, axis=0)
    joint_vel = np.gradient(joint_pos, dt, axis=0)
    root_ang_vel = _so3_derivative(root_quat, dt)
    return ResampledMotion(
        output_fps, root_pos, root_quat, joint_pos, root_lin_vel, root_ang_vel, joint_vel
    )


def _so3_derivative(quaternion: np.ndarray, dt: float) -> np.ndarray:
    if quaternion.shape[0] < 3:
        return np.zeros((quaternion.shape[0], 3), dtype=quaternion.dtype)
    relative = quat_mul(quaternion[2:], quat_conjugate(quaternion[:-2]))
    omega = rotvec_from_quat(relative) / (2.0 * dt)
    return np.concatenate((omega[:1], omega, omega[-1:]), axis=0)


def forward_kinematics(model, motion: ResampledMotion) -> BodyStates:
    import mujoco

    data = mujoco.MjData(model)  # pyright: ignore[reportAttributeAccessIssue]
    frames = motion.num_frames
    bodies = model.nbody - 1
    pos_w = np.empty((frames, bodies, 3))
    quat_w = np.empty((frames, bodies, 4))
    lin_w = np.empty((frames, bodies, 3))
    ang_w = np.empty((frames, bodies, 3))
    root_id = model.jnt_bodyid[0]
    ang_vel_b = quat_rotate_inverse(motion.root_quat, motion.root_ang_vel_w)
    for frame in range(frames):
        data.qpos[:3] = motion.root_pos[frame]
        data.qpos[3:7] = motion.root_quat[frame]
        data.qpos[7:] = motion.joint_pos[frame]
        data.qvel[:3] = motion.root_lin_vel_w[frame]
        data.qvel[3:6] = ang_vel_b[frame]
        data.qvel[6:] = motion.joint_vel[frame]
        mujoco.mj_kinematics(model, data)  # pyright: ignore[reportAttributeAccessIssue]
        mujoco.mj_comPos(model, data)  # pyright: ignore[reportAttributeAccessIssue]
        mujoco.mj_comVel(model, data)  # pyright: ignore[reportAttributeAccessIssue]
        pos_w[frame] = data.xpos[1:]
        quat_w[frame] = data.xquat[1:]
        cvel = data.cvel[1:]
        offset = data.subtree_com[root_id][None, :] - data.xpos[1:]
        lin_w[frame] = cvel[:, 3:6] - np.cross(cvel[:, :3], offset)
        ang_w[frame] = cvel[:, :3]
    names = [model.body(index).name for index in range(1, model.nbody)]
    return BodyStates(names, pos_w, quat_w, lin_w, ang_w)


def convert_csv_to_npz(
    csv_path: str | Path,
    out_path: str | Path,
    input_fps: float = 120.0,
    output_fps: float = 50.0,
    *,
    model=None,
) -> dict[str, object]:
    model = load_g1_model() if model is None else model
    raw = load_csv(csv_path)
    motion = resample_motion(raw, input_fps, output_fps)
    bodies = forward_kinematics(model, motion)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = out_path.with_name(out_path.name + ".tmp")
    try:
        with temporary.open("wb") as stream:
            np.savez(
                stream,
                fps=np.array([output_fps], dtype=np.float32),
                joint_pos=motion.joint_pos.astype(np.float32),
                joint_vel=motion.joint_vel.astype(np.float32),
                body_pos_w=bodies.pos_w.astype(np.float32),
                body_quat_w=bodies.quat_w.astype(np.float32),
                body_lin_vel_w=bodies.lin_vel_w.astype(np.float32),
                body_ang_vel_w=bodies.ang_vel_w.astype(np.float32),
                joint_names=np.array(G1_JOINT_NAMES),
                body_names=np.array(bodies.body_names),
                source=np.array(str(csv_path)),
                input_fps=np.array([input_fps], dtype=np.float32),
            )
        os.replace(temporary, out_path)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "source": str(csv_path),
        "out": str(out_path),
        "input_frames": int(raw.shape[0]),
        "output_frames": motion.num_frames,
        "duration_s": (motion.num_frames - 1) / output_fps,
    }


_MODEL = None


def _init_worker(model_path: str) -> None:
    global _MODEL
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    _MODEL = load_g1_model(model_path)


def _convert_one(job: tuple[str, str, float, float]) -> dict[str, object]:
    source, output, input_fps, output_fps = job
    try:
        result = convert_csv_to_npz(source, output, input_fps, output_fps, model=_MODEL)
        return {**result, "ok": True, "error": ""}
    except Exception as exc:
        return {
            "source": source,
            "out": output,
            "input_frames": 0,
            "output_frames": 0,
            "duration_s": 0.0,
            "ok": False,
            "error": repr(exc),
        }


def convert_directory(
    input_dir: Path,
    output_dir: Path,
    *,
    input_fps: float = 120.0,
    output_fps: float = 50.0,
    workers: int = 16,
    pattern: str = "**/*.csv",
    overwrite: bool = False,
    limit: int | None = None,
    model_path: Path = DEFAULT_MODEL,
) -> list[dict[str, object]]:
    files = sorted(input_dir.glob(pattern))
    if limit is not None:
        files = files[:limit]
    if not files:
        raise FileNotFoundError(f"No files matching {pattern} under {input_dir}")
    names = [file.stem for file in files]
    if len(names) != len(set(names)):
        raise ValueError("BONES-SEED clip names must be unique across input subdirectories")
    output_dir.mkdir(parents=True, exist_ok=True)
    jobs = [
        (str(file), str(output_dir / f"{file.stem}.npz"), input_fps, output_fps)
        for file in files
        if overwrite or not (output_dir / f"{file.stem}.npz").is_file()
    ]
    print(f"{len(files)} CSV clips, {len(jobs)} to convert, {workers} workers")
    results: list[dict[str, object]] = []
    started = time.monotonic()
    if workers == 1:
        _init_worker(str(model_path))
        results = [_convert_one(job) for job in jobs]
    elif jobs:
        with mp.Pool(workers, initializer=_init_worker, initargs=(str(model_path),)) as pool:
            for index, result in enumerate(pool.imap_unordered(_convert_one, jobs, chunksize=4), 1):
                results.append(result)
                if index % 200 == 0 or index == len(jobs):
                    elapsed = time.monotonic() - started
                    print(f"  {index}/{len(jobs)} converted in {elapsed:.0f}s")
    failures = [result for result in results if not result["ok"]]
    for failure in failures[:20]:
        print(f"FAILED {failure['source']}: {failure['error']}")
    write_conversion_manifest(output_dir)
    if failures:
        raise RuntimeError(f"Failed to convert {len(failures)} of {len(jobs)} clips")
    return results


def write_conversion_manifest(output_dir: Path) -> None:
    rows = []
    for path in sorted(output_dir.glob("*.npz")):
        with np.load(path) as data:
            frames = int(data["joint_pos"].shape[0])
            fps = float(data["fps"][0])
            input_fps = float(data["input_fps"][0])
        rows.append((path.stem, frames, fps, (frames - 1) / fps, input_fps))
    with (output_dir / "manifest.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("name", "frames", "fps", "duration_s", "input_fps"))
        writer.writerows(rows)
    hours = sum(row[3] for row in rows) / 3600.0
    print(f"Manifest: {len(rows)} clips, {hours:.3f} h")


def name_filter(name: str) -> str | None:
    lowered = name.lower()
    for keyword in KEYWORDS:
        if keyword in lowered:
            return f"name:{keyword}"
    return None


def longest_run(mask: np.ndarray) -> int:
    padded = np.concatenate(
        (np.zeros(1, dtype=np.int8), mask.astype(np.int8), np.zeros(1, dtype=np.int8))
    )
    changes = np.flatnonzero(np.diff(padded))
    return int((changes[1::2] - changes[::2]).max(initial=0))


def _foot_geoms(model) -> list[int]:
    import mujoco

    body_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)  # pyright: ignore[reportAttributeAccessIssue]
        for name in ("left_ankle_roll_link", "right_ankle_roll_link")
    ]
    geoms = [
        geom
        for geom in range(model.ngeom)
        if model.geom_bodyid[geom] in body_ids
        and (model.geom_contype[geom] != 0 or model.geom_conaffinity[geom] != 0)
    ]
    supported = {
        mujoco.mjtGeom.mjGEOM_SPHERE,  # pyright: ignore[reportAttributeAccessIssue]
        mujoco.mjtGeom.mjGEOM_CAPSULE,  # pyright: ignore[reportAttributeAccessIssue]
    }
    if not geoms or any(model.geom_type[geom] not in supported for geom in geoms):
        raise ValueError("Foot collision geometry must contain spheres or capsules")
    return geoms


def _foot_sole_height(npz_path: str | Path, gate: np.ndarray, model) -> np.ndarray:
    import mujoco

    data = mujoco.MjData(model)  # pyright: ignore[reportAttributeAccessIssue]
    geoms = _foot_geoms(model)
    with np.load(npz_path) as motion:
        names = [str(name) for name in motion["body_names"]]
        pelvis = names.index("pelvis")
        root_pos = motion["body_pos_w"][:, pelvis]
        root_quat = motion["body_quat_w"][:, pelvis]
        joint_pos = motion["joint_pos"]
    indices = np.nonzero(gate)[0]
    heights = np.empty(len(indices))
    for output, frame in enumerate(indices):
        data.qpos[:3] = root_pos[frame]
        data.qpos[3:7] = root_quat[frame]
        data.qpos[7:] = joint_pos[frame]
        mujoco.mj_kinematics(model, data)  # pyright: ignore[reportAttributeAccessIssue]
        center_z = data.geom_xpos[geoms, 2]
        radius = model.geom_size[geoms, 0]
        half_length = model.geom_size[geoms, 1]
        axis_z = data.geom_xmat[geoms].reshape(-1, 3, 3)[:, 2, 2]
        is_capsule = (
            model.geom_type[geoms] == mujoco.mjtGeom.mjGEOM_CAPSULE  # pyright: ignore[reportAttributeAccessIssue]
        )
        heights[output] = (center_z - half_length * np.abs(axis_z) * is_capsule - radius).min()
    return heights


def feasibility_filter(npz_path: str | Path, model) -> str | None:
    with np.load(npz_path) as motion:
        names = [str(name) for name in motion["body_names"]]
        pos = motion["body_pos_w"]
        quat = motion["body_quat_w"]
        joint_vel = motion["joint_vel"]
        lin_vel = motion["body_lin_vel_w"]
        fps = float(motion["fps"][0])
    zmin = pos[:, :, 2].min(axis=1)
    left = names.index("left_ankle_roll_link")
    right = names.index("right_ankle_roll_link")
    pelvis = names.index("pelvis")
    if left < 0 or right < 0:
        raise ValueError("Missing foot bodies")
    if longest_run(zmin > ELEVATED_Z) >= ELEVATED_S * fps:
        return "elevated_support"
    if (zmin < PENETRATE_Z_HARD).any() or (zmin < PENETRATE_Z_SOFT).mean() > PENETRATE_FRAC:
        return "ground_penetration"
    quaternion = quat[:, pelvis]
    gravity_z = -(1.0 - 2.0 * (quaternion[:, 1] ** 2 + quaternion[:, 2] ** 2))
    upright = gravity_z < -UPRIGHT_GZ
    standing = (pos[:, pelvis, 2] > STAND_Z[0]) & (pos[:, pelvis, 2] < STAND_Z[1])
    still = (np.linalg.norm(lin_vel[:, pelvis], axis=1) < STILL_V) & (
        np.sqrt((joint_vel**2).mean(axis=1)) < STILL_QD
    )
    gate = upright & standing & still
    if longest_run(gate) >= FLOAT_S * fps:
        floating = gate.copy()
        floating[gate] = _foot_sole_height(npz_path, gate, model) > FLOAT_Z
        if longest_run(floating) >= FLOAT_S * fps:
            return "floating_feet"
    return None


def _filter_one(job: tuple[str, str]) -> tuple[str, int, str | None]:
    path, model_path = job
    global _MODEL
    if _MODEL is None:
        _MODEL = load_g1_model(model_path)
    reason = name_filter(Path(path).stem)
    if reason is None:
        reason = feasibility_filter(path, _MODEL)
    with np.load(path) as motion:
        frames = int(motion["joint_pos"].shape[0])
    return Path(path).stem, frames, reason


def filter_directory(
    npz_dir: Path, *, workers: int = 16, out: Path | None = None, model_path: Path = DEFAULT_MODEL
) -> Path:
    files = sorted(str(path) for path in npz_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No NPZ clips under {npz_dir}")
    jobs = [(path, str(model_path)) for path in files]
    if workers == 1:
        rows = [_filter_one(job) for job in jobs]
    else:
        with mp.Pool(workers, initializer=_init_worker, initargs=(str(model_path),)) as pool:
            rows = pool.map(_filter_one, jobs, chunksize=64)
    out = out or npz_dir / "manifest_filtered.csv"
    with out.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("name", "frames", "keep", "reason"))
        for name, frames, reason in rows:
            writer.writerow((name, frames, int(reason is None), reason or ""))
    counts = Counter(reason for _, _, reason in rows)
    kept = sum(reason is None for _, _, reason in rows)
    name_removed = sum(bool(reason and reason.startswith("name:")) for _, _, reason in rows)
    print(f"Filtered {len(rows)} -> {kept} clips ({name_removed} rejected by name)")
    print(f"Reasons: {dict(counts.most_common(40))}")
    print(f"Saved {out}")
    return out


def discover_npz(
    directory: Path, manifest: Path | None = None, names: Sequence[str] | None = None
) -> list[Path]:
    files = sorted(directory.glob("*.npz"))
    if manifest is not None:
        with manifest.open(encoding="utf-8", newline="") as stream:
            keep = {row["name"] for row in csv.DictReader(stream) if row.get("keep") == "1"}
        files = [path for path in files if path.stem in keep]
        missing = keep - {path.stem for path in files}
        if missing:
            raise FileNotFoundError(
                f"Manifest clips missing from {directory}: {sorted(missing)[:5]}"
            )
    if names is not None:
        wanted = set(names)
        files = [path for path in files if path.stem in wanted]
    return files


def _read_npz_reference(path: Path) -> np.ndarray:
    with np.load(path) as motion:
        names = [str(name) for name in motion["body_names"]]
        pelvis = names.index("pelvis")
        return np.concatenate(
            (
                motion["body_pos_w"][:, pelvis],
                motion["body_quat_w"][:, pelvis],
                motion["joint_pos"],
                motion["body_lin_vel_w"][:, pelvis],
                motion["body_ang_vel_w"][:, pelvis],
                motion["joint_vel"],
            ),
            axis=1,
        )


def pack_library(
    npz_dir: Path, out: Path, *, manifest: Path | None = None, workers: int = 1
) -> Path:
    if workers < 1:
        raise ValueError("workers must be positive")
    manifest = manifest or npz_dir / "manifest_filtered.csv"
    files = discover_npz(npz_dir, manifest)
    if not files:
        raise ValueError("No clips selected for the compact library")
    infos = []
    total_frames = 0
    for path in files:
        with np.load(path) as motion:
            frames = int(motion["joint_pos"].shape[0])
            fps = float(motion["fps"][0])
        if not math.isclose(fps, 50.0):
            raise ValueError(f"{path}: compact SuperMimic libraries require 50 fps, got {fps}")
        infos.append((path.stem, total_frames, frames))
        total_frames += frames
    reference = torch.empty((total_frames, REF_DIM), dtype=torch.float16)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        references = pool.map(_read_npz_reference, files)
        for index, (path, ref) in enumerate(zip(files, references), 1):
            if ref.shape != (infos[index - 1][2], REF_DIM) or not np.isfinite(ref).all():
                raise ValueError(f"{path}: invalid or non-finite reference state {ref.shape}")
            compact = ref.astype(np.float16)
            if not np.isfinite(compact).all():
                raise ValueError(f"{path}: values exceed the fp16 range")
            start = infos[index - 1][1]
            reference[start : start + len(ref)] = torch.from_numpy(compact)
            if index % 500 == 0 or index == len(files):
                print(f"  {index}/{len(files)} packed")
    payload = {
        "ref": reference,
        "names": [name for name, _, _ in infos],
        "starts": [start for _, start, _ in infos],
        "lengths": [length for _, _, length in infos],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=out.parent, prefix=out.name + ".", suffix=".tmp", delete=False
    ) as stream:
        temporary = Path(stream.name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, out)
    finally:
        temporary.unlink(missing_ok=True)
    size_gb = reference.numel() * reference.element_size() / 1e9
    print(f"Saved {out}: {len(files)} clips, {total_frames} frames, {size_gb:.2f} GB fp16")
    return out


def download_dataset(output_dir: Path, cache_dir: Path) -> None:
    from huggingface_hub import hf_hub_download

    archive = Path(
        hf_hub_download(
            repo_id=REPO_ID,
            filename=ARCHIVE_NAME,
            repo_type="dataset",
            local_dir=cache_dir,
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as bundle:
        members = bundle.getmembers()
        for member in members:
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts or not (member.isfile() or member.isdir()):
                raise ValueError(f"Unsafe archive member: {member.name}")
        bundle.extractall(output_dir)
    print(f"Extracted {ARCHIVE_NAME} to {output_dir}")


def _path(value: str) -> Path:
    return Path(value).expanduser()


def _add_convert_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input-dir", type=_path, default=Path("data/raw/bones_seed"))
    parser.add_argument("--output-dir", type=_path, default=Path("data/npz/bones_seed"))
    parser.add_argument("--input-fps", type=float, default=120.0)
    parser.add_argument("--output-fps", type=float, default=50.0)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--pattern", default="**/*.csv")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--model", type=_path, default=DEFAULT_MODEL)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    download = subparsers.add_parser("download", help="Download and extract g1.tar.gz")
    download.add_argument("--output-dir", type=_path, default=Path("data/raw/bones_seed"))
    download.add_argument("--cache-dir", type=_path, default=Path("data/.cache/bones_seed"))
    convert = subparsers.add_parser("convert", help="Convert raw CSV clips to 50 Hz NPZ")
    _add_convert_arguments(convert)
    filtering = subparsers.add_parser("filter", help="Apply name and feasibility filters")
    filtering.add_argument("--npz-dir", type=_path, default=Path("data/npz/bones_seed"))
    filtering.add_argument("--workers", type=int, default=16)
    filtering.add_argument("--out", type=_path)
    filtering.add_argument("--model", type=_path, default=DEFAULT_MODEL)
    pack = subparsers.add_parser("pack", help="Pack kept clips into an fp16 library")
    pack.add_argument("--npz-dir", type=_path, default=Path("data/npz/bones_seed"))
    pack.add_argument("--manifest", type=_path)
    pack.add_argument("--out", type=_path, default=Path("data/library/bones_seed_compact.pt"))
    pack.add_argument("--workers", type=int, default=1)
    all_steps = subparsers.add_parser("all", help="Run download, convert, filter, and pack")
    _add_convert_arguments(all_steps)
    all_steps.add_argument("--cache-dir", type=_path, default=Path("data/.cache/bones_seed"))
    all_steps.add_argument("--skip-download", action="store_true")
    all_steps.add_argument(
        "--library", type=_path, default=Path("data/library/bones_seed_compact.pt")
    )
    args = parser.parse_args(argv)
    if args.command == "download":
        download_dataset(args.output_dir, args.cache_dir)
    elif args.command == "convert":
        convert_directory(
            args.input_dir,
            args.output_dir,
            input_fps=args.input_fps,
            output_fps=args.output_fps,
            workers=args.workers,
            pattern=args.pattern,
            overwrite=args.overwrite,
            limit=args.limit,
            model_path=args.model,
        )
    elif args.command == "filter":
        filter_directory(args.npz_dir, workers=args.workers, out=args.out, model_path=args.model)
    elif args.command == "pack":
        pack_library(args.npz_dir, args.out, manifest=args.manifest, workers=args.workers)
    else:
        if not args.skip_download:
            download_dataset(args.input_dir, args.cache_dir)
        convert_directory(
            args.input_dir,
            args.output_dir,
            input_fps=args.input_fps,
            output_fps=args.output_fps,
            workers=args.workers,
            pattern=args.pattern,
            overwrite=args.overwrite,
            limit=args.limit,
            model_path=args.model,
        )
        manifest = filter_directory(args.output_dir, workers=args.workers, model_path=args.model)
        pack_library(args.output_dir, args.library, manifest=manifest, workers=args.workers)


if __name__ == "__main__":
    main()
