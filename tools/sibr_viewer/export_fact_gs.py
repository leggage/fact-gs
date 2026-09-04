#!/usr/bin/env python3
"""Convert one or all FaCT-GS point_cloud.pickle snapshots for SIBR."""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fact_gs.utils.sibr_export import ensure_minimal_sibr_scene, export_sibr_ply, load_fact_gs_pickle


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
        target = output / "point_cloud" / f"iteration_{step}" / "point_cloud.ply"
        stats = export_sibr_ply(target, **values)
        print(f"step {step}: {stats['count']} Gaussians -> {target}")


if __name__ == "__main__":
    main()
