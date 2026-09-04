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

exec "$viewer" -m "$model_dir/sibr" "${step_args[@]}" "$@"
