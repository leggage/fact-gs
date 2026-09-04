"""Export FaCT-GS density Gaussians in the Graphdeco SIBR PLY layout.

SIBR expects opacity and spherical-harmonic colour, while CT Gaussians carry a
positive attenuation density. This module keeps geometry exact, uses neutral
white degree-0 SH colour, and maps density only to opacity.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Iterable

import numpy as np


SH_C0 = 0.28209479177387814
PLY_PROPERTIES = (
    "x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2",
    "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
)


def _density_rgb(value: np.ndarray, low: float, high: float) -> np.ndarray:
    """Small perceptually ordered blue-cyan-yellow-red map (no matplotlib)."""
    t = np.clip((value - low) / max(high - low, np.finfo(np.float32).eps), 0.0, 1.0)
    stops = np.asarray(
        [[0.05, 0.08, 0.45], [0.00, 0.70, 0.90], [0.95, 0.90, 0.10], [0.75, 0.02, 0.02]],
        dtype=np.float32,
    )
    u = t * (len(stops) - 1)
    index = np.minimum(u.astype(np.int32), len(stops) - 2)
    frac = (u - index)[:, None]
    return stops[index] * (1.0 - frac) + stops[index + 1] * frac


def export_sibr_ply(
    path: os.PathLike | str,
    xyz: np.ndarray,
    density: np.ndarray,
    scale: np.ndarray,
    rotation: np.ndarray,
    *,
    density_percentiles: tuple[float, float] = (1.0, 99.0),
    opacity_floor: float = 0.05,
) -> dict:
    """Write a degree-0 binary PLY accepted by ``SIBR_gaussianViewer_app``.

    ``scale`` and ``density`` must already be activated (positive physical
    values); ``rotation`` must be a normalised wxyz quaternion.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    xyz = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
    density = np.asarray(density, dtype=np.float32).reshape(-1)
    scale = np.asarray(scale, dtype=np.float32).reshape(-1, 3)
    rotation = np.asarray(rotation, dtype=np.float32).reshape(-1, 4)
    n = xyz.shape[0]
    if not (density.size == scale.shape[0] == rotation.shape[0] == n):
        raise ValueError("xyz, density, scale and rotation lengths differ")
    if n == 0:
        raise ValueError("cannot export an empty Gaussian model")

    finite = np.isfinite(density)
    if not finite.any():
        raise ValueError("density contains no finite values")
    low, high = np.percentile(density[finite], density_percentiles).astype(float)
    if high <= low:
        high = low + max(abs(low), 1.0) * 1e-6
    norm = np.clip((density - low) / (high - low), 0.0, 1.0)
    # CT has no intrinsic RGB colour; carry density exclusively by opacity.
    rgb = np.ones((n, 3), dtype=np.float32)
    sh_dc = (rgb - 0.5) / SH_C0
    opacity = np.clip(opacity_floor + (1.0 - opacity_floor) * norm, 1e-6, 1.0 - 1e-6)
    opacity_logit = np.log(opacity / (1.0 - opacity))
    log_scale = np.log(np.maximum(scale, np.finfo(np.float32).tiny))
    quat_norm = np.linalg.norm(rotation, axis=1, keepdims=True)
    rotation = rotation / np.maximum(quat_norm, np.finfo(np.float32).eps)

    rows = np.concatenate(
        [xyz, np.zeros_like(xyz), sh_dc, opacity_logit[:, None], log_scale, rotation], axis=1
    ).astype("<f4", copy=False)
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {n}"]
    header.extend(f"property float {name}" for name in PLY_PROPERTIES)
    header.extend(["comment fact_gs_color density", "end_header"])
    with path.open("wb") as handle:
        handle.write(("\n".join(header) + "\n").encode("ascii"))
        handle.write(rows.tobytes(order="C"))

    stats = {
        "count": n,
        "density_min": float(np.nanmin(density)),
        "density_max": float(np.nanmax(density)),
        "density_color_low": low,
        "density_color_high": high,
        "scale_min": np.nanmin(scale, axis=0).astype(float).tolist(),
        "scale_max": np.nanmax(scale, axis=0).astype(float).tolist(),
    }
    with path.with_suffix(".json").open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)
    return stats


def _camera_value(camera, *names):
    for name in names:
        if hasattr(camera, name):
            return getattr(camera, name)
    raise AttributeError(f"camera has none of the required attributes: {names}")


def _camera_to_json(camera, camera_id: int) -> dict:
    """Convert an R2-Gaussian Camera/CameraInfo to Graphdeco camera JSON."""
    rotation = np.asarray(camera.R, dtype=np.float64).reshape(3, 3)
    translation = np.asarray(camera.T, dtype=np.float64).reshape(3)
    world_to_camera = np.eye(4, dtype=np.float64)
    # R2-Gaussian stores R transposed for GLM/CUDA, like Graphdeco 3DGS.
    world_to_camera[:3, :3] = rotation.T
    world_to_camera[:3, 3] = translation
    camera_to_world = np.linalg.inv(world_to_camera)

    width = int(_camera_value(camera, "image_width", "width"))
    height = int(_camera_value(camera, "image_height", "height"))
    fov_x = float(_camera_value(camera, "FoVx", "FovX"))
    fov_y = float(_camera_value(camera, "FoVy", "FovY"))
    return {
        "id": int(camera_id),
        # Images are disabled in the CT viewer. Reuse the one valid placeholder
        # name so SIBR's dataset parser still sees a complete scene.
        "img_name": "reference.ppm",
        "width": width,
        "height": height,
        "fx": width / (2.0 * math.tan(fov_x / 2.0)),
        "fy": height / (2.0 * math.tan(fov_y / 2.0)),
        "position": camera_to_world[:3, 3].tolist(),
        "rotation": camera_to_world[:3, :3].tolist(),
    }


def export_sibr_cameras(model_dir: os.PathLike | str, cameras: Iterable) -> list[dict]:
    """Write the real CT cameras and put a mid-spiral side view first."""
    root = Path(model_dir).resolve()
    entries = [_camera_to_json(camera, i) for i, camera in enumerate(cameras)]
    if not entries:
        raise ValueError("cannot export an empty CT camera list")

    # InteractiveCameraHandler starts from the first input camera. Put the
    # camera nearest the median source z first, while retaining every camera
    # for SIBR's Top view trajectory display.
    z_values = np.asarray([entry["position"][2] for entry in entries])
    middle = int(np.argmin(np.abs(z_values - np.median(z_values))))
    entries = entries[middle:] + entries[:middle]
    for camera_id, entry in enumerate(entries):
        entry["id"] = camera_id
    (root / "viewer_scene").mkdir(parents=True, exist_ok=True)
    with (root / "viewer_scene" / "cameras.json").open("w", encoding="utf-8") as handle:
        json.dump(entries, handle, indent=2)
        handle.write("\n")
    return entries


def export_sibr_orbit_cameras(
    model_dir: os.PathLike | str,
    bbox,
    *,
    orbit_radius: float,
    count: int = 64,
    resolution: int = 1024,
    fov_margin: float = 1.1,
) -> list[dict]:
    """Write square, same-z inspection cameras whose FOV contains the volume."""
    root = Path(model_dir).resolve()
    bounds = np.asarray(bbox, dtype=np.float64).reshape(2, 3)
    center = bounds.mean(axis=0)
    half_diagonal = float(np.linalg.norm((bounds[1] - bounds[0]) / 2.0))
    if count < 1 or resolution < 1 or fov_margin <= 1.0:
        raise ValueError("orbit count/resolution must be positive and fov_margin must exceed 1")
    # Keep the requested scanner-like radius where possible, but guarantee that
    # the camera stays outside the margin-expanded bounding sphere.
    radius = max(float(orbit_radius), half_diagonal * fov_margin * 1.05)
    ratio = min(half_diagonal * fov_margin / radius, 0.999)
    fov = 2.0 * math.asin(ratio)
    focal = resolution / (2.0 * math.tan(fov / 2.0))
    entries = []
    up_hint = np.asarray([0.0, 0.0, -1.0])
    for camera_id, angle in enumerate(np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)):
        position = center + radius * np.asarray([math.cos(angle), math.sin(angle), 0.0])
        forward = center - position
        forward /= np.linalg.norm(forward)
        right = np.cross(up_hint, forward)
        right /= np.linalg.norm(right)
        up = np.cross(forward, right)
        camera_to_world = np.column_stack((right, up, forward))
        entries.append(
            {
                "id": camera_id,
                "img_name": "reference.ppm",
                "width": int(resolution),
                "height": int(resolution),
                "fx": focal,
                "fy": focal,
                "position": position.tolist(),
                "rotation": camera_to_world.tolist(),
            }
        )
    (root / "viewer_scene").mkdir(parents=True, exist_ok=True)
    with (root / "viewer_scene" / "cameras.json").open("w", encoding="utf-8") as handle:
        json.dump(entries, handle, indent=2)
        handle.write("\n")
    return entries


def export_gaussian_model(
    model,
    model_path: os.PathLike | str,
    step: int,
    *,
    cameras: Iterable | None = None,
    viewer_bbox=None,
    viewer_orbit_radius: float | None = None,
    viewer_camera_mode: str = "orbit",
) -> dict:
    """Export an in-memory :class:`GaussianModel` as one SIBR iteration."""
    target = Path(model_path) / "sibr" / "point_cloud" / f"iteration_{int(step)}" / "point_cloud.ply"
    to_numpy = lambda value: value.detach().float().cpu().numpy()
    stats = export_sibr_ply(
        target,
        to_numpy(model.get_xyz),
        to_numpy(model.get_density),
        to_numpy(model.get_scaling),
        to_numpy(model.get_rotation),
    )
    ensure_minimal_sibr_scene(
        Path(model_path) / "sibr",
        cameras=cameras,
        viewer_bbox=viewer_bbox,
        viewer_orbit_radius=viewer_orbit_radius,
        viewer_camera_mode=viewer_camera_mode,
    )
    return stats


def ensure_minimal_sibr_scene(
    model_dir: os.PathLike | str,
    *,
    cameras: Iterable | None = None,
    viewer_bbox=None,
    viewer_orbit_radius: float | None = None,
    viewer_camera_mode: str = "spiral",
) -> None:
    """Create the tiny COLMAP scaffold required for free-camera SIBR viewing."""
    root = Path(model_dir).resolve()
    sparse = root / "viewer_scene" / "sparse" / "0"
    images = root / "viewer_scene" / "images"
    sparse.mkdir(parents=True, exist_ok=True)
    images.mkdir(parents=True, exist_ok=True)
    (sparse / "cameras.txt").write_text("1 PINHOLE 1024 1024 800 800 512 512\n", encoding="ascii")
    (sparse / "images.txt").write_text("1 1 0 0 0 0 0 4 1 reference.ppm\n\n", encoding="ascii")
    (sparse / "points3D.txt").write_text("", encoding="ascii")
    # SIBR's Gaussian scene parser identifies a dataset by cameras.json.
    # Keep one neutral camera so the free-camera viewer can load CT Gaussians
    # even though this project does not use photographic input images.
    if viewer_camera_mode == "orbit":
        if viewer_bbox is None or viewer_orbit_radius is None:
            raise ValueError("orbit viewer cameras require bbox and orbit radius")
        export_sibr_orbit_cameras(
            root, viewer_bbox, orbit_radius=viewer_orbit_radius
        )
    elif viewer_camera_mode == "spiral" and cameras is not None:
        export_sibr_cameras(root, cameras)
    elif viewer_camera_mode not in {"spiral", "fallback"}:
        raise ValueError(f"unknown SIBR viewer camera mode: {viewer_camera_mode}")
    else:
        camera_file = root / "viewer_scene" / "cameras.json"
        # Never replace previously exported real cameras with the fallback.
        if not camera_file.exists():
            camera_file.write_text(
                '[{"id":0,"img_name":"reference.ppm","width":1024,"height":1024,'
                '"fx":800.0,"fy":800.0,"position":[0.0,0.0,-5.0],'
                '"rotation":[[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]]}]\n',
                encoding="ascii",
            )
    ppm = images / "reference.ppm"
    if not ppm.exists():
        with ppm.open("wb") as handle:
            handle.write(b"P6\n1 1\n255\n\x00\x00\x00")
    cfg = f"Namespace(source_path='{root / 'viewer_scene'}', sh_degree=0, white_background=False)\n"
    (root / "cfg_args").write_text(cfg, encoding="utf-8")


def load_fact_gs_pickle(path: os.PathLike | str) -> dict:
    """Load a trusted FaCT-GS checkpoint and activate its raw parameters."""
    import pickle

    with Path(path).open("rb") as handle:
        data = pickle.load(handle)
    raw_density = np.asarray(data["density"], dtype=np.float32)
    density = np.logaddexp(0.0, raw_density)
    raw_scale = np.asarray(data["scale"], dtype=np.float32)
    bound = data.get("scale_bound")
    if bound is None:
        scale = np.exp(raw_scale)
    else:
        lo, hi = np.asarray(bound, dtype=np.float32)
        scale = (1.0 / (1.0 + np.exp(-raw_scale))) * (hi - lo) + lo
    rotation = np.asarray(data["rotation"], dtype=np.float32)
    rotation /= np.maximum(np.linalg.norm(rotation, axis=1, keepdims=True), 1e-12)
    return {"xyz": data["xyz"], "density": density, "scale": scale, "rotation": rotation}
