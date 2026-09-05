"""FaCT-GS public rendering entry points.

Keep the CUDA-backed modules lazy so CPU-only utilities such as the SIBR
checkpoint exporter and its tests do not require the training extensions just
to import ``fact_gs``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = ("rasterize_proj", "voxelize_vol")

if TYPE_CHECKING:
    from .rasterize import rasterize_proj
    from .voxelize import voxelize_vol


def __getattr__(name: str):
    if name == "rasterize_proj":
        from .rasterize import rasterize_proj

        return rasterize_proj
    if name == "voxelize_vol":
        from .voxelize import voxelize_vol

        return voxelize_vol
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
