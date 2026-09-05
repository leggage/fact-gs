#!/usr/bin/env python3
"""Convert one or all FaCT-GS point_cloud.pickle snapshots for SIBR."""

from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import yaml

# NumPy 2 checkpoints name this private module ``numpy._core.numeric`` while
# the CUDA training environment currently ships NumPy 1.x.  The array pickle
# representation itself is compatible.
if not hasattr(np, "_core"):
    import numpy.core as _numpy_core
    import numpy.core.numeric as _numpy_core_numeric

    sys.modules.setdefault("numpy._core", _numpy_core)
    sys.modules.setdefault("numpy._core.numeric", _numpy_core_numeric)

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fact_gs.utils.sibr_export import ensure_minimal_sibr_scene, export_sibr_ply, load_fact_gs_pickle


def load_or_compute_densify_gradient(source: Path, training_config: Path | None):
    """Return a per-Gaussian view-average gradient, caching recomputation."""
    with source.open("rb") as handle:
        snapshot = pickle.load(handle)
    saved = snapshot.get("densify_grad")
    if saved is not None:
        saved = np.asarray(saved, dtype=np.float32).reshape(-1)
        if saved.size == np.asarray(snapshot["xyz"]).shape[0]:
            return saved, "saved_training_window"

    cache = source.parent / "densify_grads_view_average.npz"
    if cache.is_file():
        with np.load(cache) as values:
            gradient = np.asarray(values["grads"], dtype=np.float32).reshape(-1)
        if gradient.size == np.asarray(snapshot["xyz"]).shape[0]:
            return gradient, "all_train_views_cache"

    if training_config is None:
        raise SystemExit(
            "This older snapshot has no saved densification gradient. Pass "
            "--training-config PATH_TO/.hydra/config.yaml to recompute it."
        )

    # TIGRE must be initialized before torch-dependent scene imports on this
    # project; keep these heavyweight imports out of density-only exports.
    import tigre  # noqa: F401
    from omegaconf import OmegaConf
    from tqdm import tqdm

    from fact_gs.r2_gaussian.dataset import SceneRecon
    from fact_gs.r2_gaussian.gaussian import GaussianModel
    from fact_gs.utils.densify_gradient import view_average_densify_gradient

    config = OmegaConf.load(training_config)
    if bool(getattr(config.optim, "use_fused_ssim", False)):
        raise SystemExit("Post-training gradient recomputation currently requires use_fused_ssim=false")
    scene = SceneRecon(config.model, shuffle=False)
    model = GaussianModel()
    model.load_ply(str(source))
    scene.gaussians = model
    cameras = scene.getTrainCameras()
    bar = tqdm(total=len(cameras), desc="View-average densify gradient")

    def update(done, total):
        bar.n = done
        bar.refresh()

    try:
        gradient, denominator = view_average_densify_gradient(
            model,
            cameras,
            lambda_dssim=float(config.optim.lambda_dssim),
            lambda_frequency=float(getattr(config.optim, "lambda_frequency", 0.0)),
            frequency_highpass_cutoff=float(
                getattr(config.optim, "frequency_highpass_cutoff", 0.1)
            ),
            progress=update,
        )
    finally:
        bar.close()
    np.savez(
        cache,
        grads=gradient,
        denom=denominator,
        kind=np.asarray("all_train_views"),
        config=np.asarray(str(training_config)),
    )
    return gradient, "all_train_views"


def load_ct_scene(data_dir: Path, geometry_path: Path | None, *, load_cameras: bool):
    """Use FaCT-GS's authoritative pose loader without constructing CUDA Cameras."""
    with (data_dir / "meta_data.json").open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    scanner = metadata["scanner"]
    if "dVoxel" not in scanner:
        scanner["dVoxel"] = (np.asarray(scanner["sVoxel"]) / scanner["nVoxel"]).tolist()
    if "dDetector" not in scanner:
        scanner["dDetector"] = (
            np.asarray(scanner["sDetector"]) / scanner["nDetector"]
        ).tolist()
    scene_scale = 2.0 / max(scanner["sVoxel"])
    for key in (
        "dVoxel", "sVoxel", "sDetector", "dDetector", "offOrigin",
        "offDetector", "DSD", "DSO",
    ):
        scanner[key] = (np.asarray(scanner[key]) * scene_scale).tolist()
    geometry = {}
    if geometry_path is not None and geometry_path.is_file():
        with geometry_path.open("r", encoding="utf-8") as handle:
            geometry = yaml.safe_load(handle) or {}
    cameras = []
    if load_cameras:
        from fact_gs.r2_gaussian.dataset.dataset_readers import readCTameras

        cameras = readCTameras(metadata, str(data_dir), False, scene_scale, geometry)["train"]
    center = np.asarray(scanner["offOrigin"], dtype=np.float64)
    size = np.asarray(scanner["sVoxel"], dtype=np.float64)
    bbox = np.stack((center - size / 2.0, center + size / 2.0))
    return cameras, bbox, float(scanner["DSO"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path, help="FaCT-GS model directory")
    parser.add_argument("--step", type=int, help="only convert this step")
    parser.add_argument(
        "--data", type=Path,
        help="dataset containing meta_data.json; exports the real CT spiral cameras",
    )
    parser.add_argument(
        "--geometry", type=Path,
        help="geometry override YAML (defaults to MODEL/geometry_used.yml)",
    )
    parser.add_argument(
        "--cameras-only", action="store_true",
        help="only regenerate scene/cameras.json (requires --data)",
    )
    parser.add_argument(
        "--camera-mode", choices=("orbit", "spiral"), default="orbit",
        help="orbit covers the full volume; spiral reproduces acquisition cameras",
    )
    parser.add_argument(
        "--color-by",
        choices=("density", "densify-gradient"),
        default="density",
        help="diagnostic scalar encoded as ellipsoid pseudo-colour",
    )
    parser.add_argument(
        "--training-config",
        type=Path,
        help="resolved Hydra config used to recompute missing densification gradients",
    )
    args = parser.parse_args()
    if args.cameras_only and args.data is None:
        parser.error("--cameras-only requires --data")
    output = args.model / "sibr"
    cameras = None
    bbox = None
    orbit_radius = None
    if args.data is not None:
        geometry_path = args.geometry or (args.model / "geometry_used.yml")
        cameras, bbox, orbit_radius = load_ct_scene(
            args.data, geometry_path, load_cameras=args.camera_mode == "spiral"
        )
    ensure_minimal_sibr_scene(
        output,
        cameras=cameras,
        viewer_bbox=bbox,
        viewer_orbit_radius=orbit_radius,
        viewer_camera_mode=args.camera_mode if args.data is not None else "fallback",
    )
    if args.cameras_only:
        camera_file = output / "viewer_scene" / "cameras.json"
        with camera_file.open("r", encoding="utf-8") as handle:
            output_count = len(json.load(handle))
        print(f"{output_count} {args.camera_mode} viewer cameras -> {camera_file}")
        return
    if args.step is None:
        inputs = sorted((args.model / "point_cloud").glob("step_*/point_cloud.pickle"))
    else:
        inputs = [args.model / "point_cloud" / f"step_{args.step}" / "point_cloud.pickle"]
    if not inputs:
        raise SystemExit(f"No snapshots found below {args.model / 'point_cloud'}")
    for source in inputs:
        if not source.is_file():
            raise SystemExit(f"Missing snapshot: {source}")
        match = re.fullmatch(r"step_(\d+)", source.parent.name)
        step = int(match.group(1))
        values = load_fact_gs_pickle(source)
        if args.color_by == "densify-gradient":
            gradient, gradient_kind = load_or_compute_densify_gradient(
                source, args.training_config
            )
            values["color_value"] = gradient
            values["color_label"] = "densify_gradient"
        target = output / "point_cloud" / f"iteration_{step}" / "point_cloud.ply"
        stats = export_sibr_ply(target, **values)
        suffix = (
            f" ({gradient_kind})" if args.color_by == "densify-gradient" else ""
        )
        print(
            f"step {step}: {stats['count']} Gaussians, "
            f"colour={stats['color_label']}{suffix} -> {target}"
        )


if __name__ == "__main__":
    main()
