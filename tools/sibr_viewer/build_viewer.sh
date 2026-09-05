#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
source_dir=${SIBR_SOURCE_DIR:-$repo_root/third_party/SIBR_viewers/source}
build_dir=${SIBR_BUILD_DIR:-$repo_root/third_party/SIBR_viewers/build}
install_dir=${SIBR_INSTALL_DIR:-$repo_root/third_party/SIBR_viewers/install}
cuda_compat=$repo_root/tools/sibr_viewer/cuda11_glibc_compat.h
cuda_home=${CUDA_HOME:-/usr/local/cuda-12.8}
cuda_architectures=${SIBR_CUDA_ARCHITECTURES:-120}
cc=${CC:-/usr/bin/gcc-11}
cxx=${CXX:-/usr/bin/g++-11}
cuda_host_cxx=${CUDAHOSTCXX:-$cxx}
build_jobs=${SIBR_BUILD_JOBS:-4}

if [[ ! -f "$source_dir/CMakeLists.txt" ]]; then
  echo "SIBR source not found at $source_dir; set SIBR_SOURCE_DIR." >&2
  exit 1
fi
if [[ ! -f "$source_dir/src/projects/gaussianviewer/CMakeLists.txt" ]]; then
  echo "SIBR gaussianviewer project is missing at $source_dir/src/projects/gaussianviewer." >&2
  exit 1
fi
if [[ ! -x "$cuda_home/bin/nvcc" ]]; then
  echo "CUDA compiler not found at $cuda_home/bin/nvcc; set CUDA_HOME." >&2
  exit 1
fi

cuda_flags=()
if [[ "$cuda_home" == *"11.8" ]]; then
  cuda_flags=(-DCMAKE_CUDA_FLAGS=-include\ $cuda_compat)
fi

CC="$cc" CXX="$cxx" cmake -S "$source_dir" -B "$build_dir" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_ROOT="$install_dir" \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DCMAKE_CUDA_COMPILER="$cuda_home/bin/nvcc" \
  -DCMAKE_CUDA_HOST_COMPILER="$cuda_host_cxx" \
  -DCMAKE_CUDA_ARCHITECTURES="$cuda_architectures" \
  "${cuda_flags[@]}" \
  -DBUILD_IBR_REMOTE=OFF \
  -DSIBR_USE_EGL=OFF \
  -DBoost_NO_BOOST_CMAKE=ON
cmake --build "$build_dir" -j"$build_jobs" --target install
echo "Installed SIBR viewer below $install_dir"
