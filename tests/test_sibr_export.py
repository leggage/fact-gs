import json
import numpy as np
from types import SimpleNamespace

from fact_gs.r2_gaussian.dataset.dataset_readers import angle2pose
from fact_gs.utils.sibr_export import (
    PLY_PROPERTIES,
    export_sibr_cameras,
    export_sibr_orbit_cameras,
    export_sibr_ply,
)


def test_export_sibr_ply_geometry_and_layout(tmp_path):
    xyz = np.asarray([[1, 2, 3], [-1, 0, 2]], dtype=np.float32)
    density = np.asarray([[0.1], [2.0]], dtype=np.float32)
    scale = np.asarray([[0.2, 0.3, 0.4], [1.0, 2.0, 3.0]], dtype=np.float32)
    rotation = np.asarray([[2, 0, 0, 0], [1, 0, 0, 0]], dtype=np.float32)
    output = tmp_path / "point_cloud.ply"

    stats = export_sibr_ply(output, xyz, density, scale, rotation, density_percentiles=(0, 100))
    raw = output.read_bytes()
    header, body = raw.split(b"end_header\n", 1)
    assert b"element vertex 2" in header
    assert all(f"property float {name}".encode() in header for name in PLY_PROPERTIES)
    rows = np.frombuffer(body, dtype="<f4").reshape(2, len(PLY_PROPERTIES))
    np.testing.assert_allclose(rows[:, :3], xyz)
    np.testing.assert_allclose(np.exp(rows[:, 10:13]), scale)
    np.testing.assert_allclose(rows[:, 13:17], [[1, 0, 0, 0], [1, 0, 0, 0]])
    assert stats["count"] == 2
    assert json.loads(output.with_suffix(".json").read_text())["density_max"] == 2.0


def test_export_sibr_cameras_preserves_spiral_and_starts_from_side(tmp_path):
    cameras = []
    dso = 3.6
    for angle, z in [(0.0, -1.0), (np.pi / 2, 0.0), (np.pi, 1.0)]:
        c2w = angle2pose(dso, angle, z)
        w2c = np.linalg.inv(c2w)
        cameras.append(
            SimpleNamespace(
                R=w2c[:3, :3].T,
                T=w2c[:3, 3],
                image_width=888,
                image_height=64,
                FoVx=1.0,
                FoVy=0.2,
            )
        )

    entries = export_sibr_cameras(tmp_path, cameras)
    positions = np.asarray([entry["position"] for entry in entries])
    np.testing.assert_allclose(np.linalg.norm(positions[:, :2], axis=1), dso)
    np.testing.assert_allclose(sorted(positions[:, 2]), [-1.0, 0.0, 1.0])
    assert abs(entries[0]["position"][2]) < 1e-8
    assert np.linalg.norm(entries[0]["position"][:2]) > 3.5

    written = json.loads((tmp_path / "viewer_scene" / "cameras.json").read_text())
    assert len(written) == 3
    assert [entry["id"] for entry in written] == [0, 1, 2]


def test_orbit_cameras_share_z_look_at_center_and_cover_bbox(tmp_path):
    bbox = np.asarray([[-1.0, -1.0, -2.0], [1.0, 1.0, 0.0]])
    center = bbox.mean(axis=0)
    entries = export_sibr_orbit_cameras(
        tmp_path, bbox, orbit_radius=3.6, count=8, resolution=1024
    )
    positions = np.asarray([entry["position"] for entry in entries])
    rotations = np.asarray([entry["rotation"] for entry in entries])
    np.testing.assert_allclose(positions[:, 2], center[2])
    np.testing.assert_allclose(np.linalg.norm(positions[:, :2], axis=1), 3.6)
    forward = rotations[:, :, 2]
    expected = center - positions
    expected /= np.linalg.norm(expected, axis=1, keepdims=True)
    np.testing.assert_allclose(forward, expected, atol=1e-7)

    fov = 2.0 * np.arctan(1024 / (2.0 * entries[0]["fx"]))
    sphere_angle = np.arcsin(np.linalg.norm((bbox[1] - bbox[0]) / 2.0) / 3.6)
    assert fov / 2.0 > sphere_angle
    assert entries[0]["width"] == entries[0]["height"] == 1024
