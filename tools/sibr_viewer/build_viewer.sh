#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
source_dir=${SIBR_SOURCE_DIR:-$repo_root/third_party/SIBR_viewers/source}
build_dir=${SIBR_BUILD_DIR:-$repo_root/third_party/SIBR_viewers/build}
install_dir=${SIBR_INSTALL_DIR:-$repo_root/third_party/SIBR_viewers/install}
cuda_compat=$repo_root/tools/sibr_viewer/cuda11_glibc_compat.h

if [[ ! -f "$source_dir/CMakeLists.txt" ]]; then
  echo "SIBR source not found at $source_dir; set SIBR_SOURCE_DIR." >&2
  exit 1
fi

CC=${CC:-/usr/bin/gcc-15} CXX=${CXX:-/usr/bin/g++-15} cmake -S "$source_dir" -B "$build_dir" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_ROOT="$install_dir" \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-11 \
  -DCMAKE_CUDA_FLAGS="-include $cuda_compat" \
	-DBUILD_IBR_REMOTE=OFF \
	-DSIBR_USE_EGL=OFF \
  -DBoost_NO_BOOST_CMAKE=ON
cmake --build "$build_dir" -j"$(nproc)" --target install
echo "Installed SIBR viewer below $install_dir"
