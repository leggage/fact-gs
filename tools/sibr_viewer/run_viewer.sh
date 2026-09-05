#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 MODEL_DIR [STEP] [extra SIBR arguments...]" >&2
  exit 2
fi

model_dir=$1
shift
step_args=()
if [[ $# -gt 0 && $1 =~ ^[0-9]+$ ]]; then
  step_args=(--iteration "$1")
  shift
fi

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
# The bundled viewer is built against CUDA 12.8.  Keep Conda's CUDA runtime
# (for example 11.8 used by training) from being selected at launch time.
if [[ -d /usr/local/cuda-12.8/lib64 ]]; then
  export LD_LIBRARY_PATH="/usr/local/cuda-12.8/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
local_viewer="$repo_root/third_party/SIBR_viewers/install/bin/SIBR_gaussianViewer_app"
upstream_viewer=/home/xielei/gaussian-splatting/SIBR_viewers/install/bin/SIBR_gaussianViewer_app
if [[ -x "$local_viewer" ]]; then
  default_viewer=$local_viewer
else
  default_viewer=$upstream_viewer
fi
viewer=${SIBR_VIEWER_BIN:-$default_viewer}
if [[ ! -x "$viewer" ]]; then
  echo "SIBR viewer not found at $viewer; run tools/sibr_viewer/build_viewer.sh first." >&2
  exit 1
fi

# Non-photographic CT exports may only contain cameras.json.  GaussianView
# still asks the SIBR scene parser for the tiny COLMAP scaffold, so create it
# lazily here without overwriting real camera metadata.
scene_dir="$model_dir/sibr/viewer_scene"
sparse_dir="$scene_dir/sparse/0"
if [[ ! -f "$sparse_dir/cameras.txt" || ! -f "$sparse_dir/images.txt" || ! -f "$sparse_dir/points3D.txt" ]]; then
  mkdir -p "$sparse_dir"
  [[ -f "$sparse_dir/cameras.txt" ]] || printf '%s\n' '1 PINHOLE 1024 1024 800 800 512 512' > "$sparse_dir/cameras.txt"
  [[ -f "$sparse_dir/images.txt" ]] || printf '%s\n\n' '1 1 0 0 0 0 0 4 1 reference.ppm' > "$sparse_dir/images.txt"
  [[ -f "$sparse_dir/points3D.txt" ]] || : > "$sparse_dir/points3D.txt"
fi

# GaussianView also requires a valid proxy mesh.  CT exports have no surface
# mesh, so use a conservative unit box; it is only a navigation/depth proxy.
proxy_mesh="$scene_dir/input.ply"
if [[ ! -s "$proxy_mesh" ]]; then
  {
    printf '%s\n' 'ply' 'format ascii 1.0' 'element vertex 8' \
      'property float x' 'property float y' 'property float z' \
      'element face 12' 'property list uchar int vertex_indices' 'end_header'
    printf '%s\n' '-2.0 -2.0 -3.0' '2.0 -2.0 -3.0' '2.0 2.0 -3.0' '-2.0 2.0 -3.0' \
      '-2.0 -2.0 2.0' '2.0 -2.0 2.0' '2.0 2.0 2.0' '-2.0 2.0 2.0'
    printf '%s\n' '3 0 1 2' '3 0 2 3' '3 4 6 5' '3 4 7 6' \
      '3 0 4 5' '3 0 5 1' '3 1 5 6' '3 1 6 2' \
      '3 2 6 7' '3 2 7 3' '3 3 7 4' '3 3 4 0'
  } > "$proxy_mesh"
fi

exec "$viewer" -m "$model_dir/sibr" "${step_args[@]}" "$@"
