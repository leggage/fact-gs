/* CUDA 11.8 declares names that glibc 2.41+ added with different noexcept
 * specifications. Pre-include CUDA's declarations under private names. */
#pragma once
#define cospi __cuda11_compat_cospi
#define sinpi __cuda11_compat_sinpi
#define rsqrt __cuda11_compat_rsqrt
#define cospif __cuda11_compat_cospif
#define sinpif __cuda11_compat_sinpif
#define rsqrtf __cuda11_compat_rsqrtf
#include <cuda_runtime.h>
#undef cospi
#undef sinpi
#undef rsqrt
#undef cospif
#undef sinpif
#undef rsqrtf
