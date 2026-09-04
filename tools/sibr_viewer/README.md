# FaCT-GS SIBR viewer bridge

This bridge lets the Graphdeco SIBR real-time viewer load CT Gaussians. Geometry
(position, activated anisotropic scale, and quaternion rotation) is preserved.
Positive CT density is currently mapped to display opacity (low density is more
transparent); the exported degree-0 SH colour is white. The adjacent
`point_cloud.json` records the exact density/scale range. A blue-cyan-yellow-red
helper exists in the exporter but is not currently wired into the PLY output.

For a reproducible installation on a machine without SIBR, the exact local
patches, dependencies, validation steps, known issues, and planned live viewer
protocol are documented in [MIGRATION.md](MIGRATION.md).

## Existing model

```bash
conda activate fact-gs
python tools/sibr_viewer/export_fact_gs.py /path/to/model
tools/sibr_viewer/run_viewer.sh /path/to/model 5000
```

For an existing CT model, generate square circular-cone inspection cameras that
cover the full volume (without touching PLY snapshots):

```bash
python tools/sibr_viewer/export_fact_gs.py /path/to/model \
  --data /path/to/r2gs/dataset --cameras-only
```

Omit the step to open the largest exported iteration. In the floating SIBR
panel choose **Ellipsoids** to inspect anisotropic sizes and use **Scaling
Modifier** to separate overlapping Gaussians. Opacity represents density.
Change iterations by restarting with another step.

Training-time exports use this full-volume orbit by default. Pass
`eval.sibr_camera_mode=spiral` during training, or `--camera-mode spiral` to the
conversion command, only when the narrow acquisition trajectory itself is what
you want to inspect. With neither training geometry nor `--data`, a single
`[0, 0, -5]` fallback keeps legacy exports loadable.

## Build the official viewer on Ubuntu

The machine already has the upstream source at
`/home/xielei/gaussian-splatting/SIBR_viewers`. The wrapper builds and installs
it inside this repository (`third_party/SIBR_viewers/{build,install}`):

```bash
sudo apt install -y libglew-dev libassimp-dev libboost-all-dev libgtk-3-dev \
  libopencv-dev libglfw3-dev libavdevice-dev libavcodec-dev libeigen3-dev \
  libxxf86vm-dev libembree-dev
tools/sibr_viewer/build_viewer.sh
```

For another checkout set `SIBR_SOURCE_DIR`; override the final executable with
`SIBR_VIEWER_BIN=/other/path/SIBR_gaussianViewer_app`.

## G4 + outside-budget training snapshots

Add these overrides to the normal `ldctc002/real/ntrain=1000` command:

```text
optim.densification_method=improved optim.use_las=true
optim.outside_gaussian_budget=0.1
eval.sibr_export=true eval.sibr_export_interval=1000
```

Each PLY is roughly 68 bytes per Gaussian. An interval of 1000 is a practical
default; exporting every densification event can consume tens of gigabytes.
