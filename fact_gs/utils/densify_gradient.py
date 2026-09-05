"""Utilities for reproducing the view-space densification statistic."""

from __future__ import annotations

import numpy as np
import torch

from fact_gs import rasterize_proj
from fact_gs.r2_gaussian.utils.loss_utils import (
    frequency_magnitude_loss,
    l1_loss,
    ssim,
)


def view_average_densify_gradient(
    model,
    cameras,
    *,
    lambda_dssim: float = 0.0,
    lambda_frequency: float = 0.0,
    frequency_highpass_cutoff: float = 0.1,
    progress=None,
):
    """Average the training densification gradient over camera visibility.

    For every visible Gaussian and camera this accumulates
    ``mean(abs(dL/du), abs(dL/dv))`` and divides by the number of cameras in
    which that Gaussian was visible. This is the same statistic consumed by
    ``GaussianModel.densify_and_prune*``; only the averaging window differs
    when all cameras are requested after training.
    """
    count = model.get_xyz.shape[0]
    accum = torch.zeros((count, 1), dtype=torch.float32, device="cuda")
    denom = torch.zeros_like(accum)

    for index, camera in enumerate(cameras):
        for parameter in (model._xyz, model._density, model._scaling, model._rotation):
            parameter.grad = None

        package = rasterize_proj(camera, model)
        prediction = package["render"]
        target = camera.original_image.cuda()
        loss = l1_loss(prediction, target)
        if lambda_dssim > 0:
            loss = loss + float(lambda_dssim) * (1.0 - ssim(prediction, target))
        if lambda_frequency > 0:
            loss = loss + float(lambda_frequency) * frequency_magnitude_loss(
                prediction,
                target,
                highpass_cutoff=float(frequency_highpass_cutoff),
            )
        loss.backward()

        gradient = package["viewspace_points"].grad
        visible = package["visibility_filter"] & torch.isfinite(gradient[:, :2]).all(dim=1)
        with torch.no_grad():
            accum[visible] += torch.abs(gradient[visible, :2]).mean(dim=-1, keepdim=True)
            denom[visible] += 1
        if progress is not None:
            progress(index + 1, len(cameras))

    average = torch.where(denom > 0, accum / torch.clamp_min(denom, 1.0), 0.0)
    return (
        average.squeeze(-1).detach().cpu().numpy().astype(np.float32),
        denom.squeeze(-1).detach().cpu().numpy().astype(np.int32),
    )
