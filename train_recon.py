import hydra
import os
import os.path as osp
import time
from random import randint
from tqdm import tqdm
import numpy as np
from fused_ssim import fused_ssim
import tigre ## For some reason, tigre needs to be imported before torch to avoid GPU errors
import torch
import torch.nn.functional as F
import sys
import yaml
from torch.utils.tensorboard import SummaryWriter

sys.path.append("./")
from fact_gs.r2_gaussian.gaussian import GaussianModel, initialize_gaussian_from_proj, initialize_gaussian, initialize_gaussian_from_prior
from fact_gs.r2_gaussian.utils.general_utils import safe_state
from fact_gs.r2_gaussian.dataset import SceneRecon
from fact_gs.r2_gaussian.utils.loss_utils import (
    frequency_magnitude_loss,
    l1_loss,
    ssim,
    tv_3d_loss,
)
from fact_gs.r2_gaussian.utils.image_utils import metric_vol, metric_proj

from fact_gs import rasterize_proj, voxelize_vol
from fact_gs.utils.profile import setup_profiler
from fact_gs.utils.vol_utils import save_volume, visualize_gaussian_footprint, visualize_gaussian_position, save_error_maps
from fact_gs.utils.sibr_export import export_gaussian_model



@hydra.main(config_path="config", config_name="default_recon.yaml", version_base=None)
def train_recon(config):
    """Hydra entry point for reconstruction training."""
    os.makedirs(config.model.model_path, exist_ok=True)

    safe_state(False)
    torch.autograd.set_detect_anomaly(False)  

    tb_writer = None
    if config.tensorboard.enabled:
        tensorboard_path = osp.join(config.model.model_path, "tensorboard")
        tb_writer = SummaryWriter(tensorboard_path)
        print(f"TensorBoard logs: {tensorboard_path}")

    profiler = setup_profiler(config.profile, config.model.model_path)

    try:
        if profiler is not None:
            with profiler:
                optimize(config, profiler, tb_writer)
        else:
            optimize(config, tb_writer=tb_writer)
    finally:
        if tb_writer is not None:
            tb_writer.flush()
            tb_writer.close()


def record_densify_event(step, xyz, grads, model_path, tb_writer, grad_threshold):
    """每个 densification 事件记录决策梯度（xyz_gradient_accum/denom，densify 重置前）。

    grads 行数对应 densify 前的旧高斯数（densification_postfix 会重置 accum 并扩容）。
    保存 compact 记录：三轴梯度加权位置剖面、梯度直方图、摘要标量；
    同时写 tensorboard 直方图（scene/densify_grad_norm）与 above-threshold 计数。
    """
    g = grads.squeeze(-1).detach().float()
    x = xyz.detach().float()
    n = int(g.numel())
    above = int((g >= grad_threshold).sum())
    mean_v = float(g.mean())
    p99 = float(torch.quantile(g, 0.99))
    if tb_writer is not None:
        tb_writer.add_histogram("scene/densify_grad_norm", g.cpu(), step)
        tb_writer.add_scalar("scene/densify_grad_above_thresh", above, step)
        tb_writer.add_scalar("scene/densify_grad_mean", mean_v, step)
        tb_writer.add_scalar("scene/densify_grad_p99", p99, step)
        tb_writer.add_scalar("scene/densify_grad_max", float(g.max()), step)
    g_np = g.cpu().numpy()
    x_np = x.cpu().numpy()
    prof = np.stack([np.histogram(x_np[:, a], bins=256, weights=g_np)[0] for a in range(3)])
    hist_counts, hist_edges = np.histogram(g_np, bins=64)
    out_dir = osp.join(model_path, "grad_maps")
    os.makedirs(out_dir, exist_ok=True)
    np.savez(
        osp.join(out_dir, f"step_{step:06d}.npz"),
        profiles=prof.astype(np.float32),
        hist_counts=hist_counts.astype(np.float32),
        hist_edges=hist_edges.astype(np.float32),
        summary=np.asarray([step, n, above, mean_v, p99]),
    )


def save_densify_grads(scene, eval_save_path, last_densify, step):
    """eval 时保存当前可用于 densification 决策的完整 (xyz, grads) 梯度。

    优先用当前 accum 窗口（denom>0 的高斯）；若恰逢 densification 事件
    （accum 已被 postfix 重置为 0），则保存最近一次事件的决策梯度。
    """
    g = scene.gaussians
    denom = g.denom
    valid = None
    if denom.numel() > 0:
        valid = denom.squeeze(-1) > 0
    if valid is not None and bool(valid.any()):
        xyz = g.get_xyz[valid].detach().cpu().numpy()
        grads = (g.xyz_gradient_accum[valid] / denom[valid]).squeeze(-1).detach().cpu().numpy()
        kind = "accum_window"
    elif last_densify is not None and last_densify[0] == step:
        _, xyz_b, grads_b = last_densify
        xyz = xyz_b.detach().cpu().numpy()
        grads = grads_b.squeeze(-1).detach().cpu().numpy()
        kind = "densify_decision"
    else:
        return
    np.savez(
        osp.join(eval_save_path, "densify_grads.npz"),
        xyz=xyz.astype(np.float32),
        grads=grads.astype(np.float32),
        kind=np.asarray(kind),
        step=np.asarray(step),
    )


# --- ImprovedGS 移植（improved-densification 分支；官方实现 XiaoBin2001/Improved-GS）---
EDGE_KERNEL = torch.tensor(
    [[[-1.0, -1.0, -1.0], [-1.0, 8.0, -1.0], [-1.0, -1.0, -1.0]]],
    dtype=torch.float32,
).unsqueeze(0)


def normalize_to_unit_range(t):
    """nan/inf 清理后 min-max 归一化到 [0,1]；全等输入返回 0。"""
    sanitized = torch.nan_to_num(t.detach().float(), nan=0.0, posinf=0.0, neginf=0.0)
    if sanitized.numel() == 0:
        return sanitized
    mn, mx = sanitized.amin(), sanitized.amax()
    if mx - mn <= 0.0:
        return torch.zeros_like(sanitized)
    return (sanitized - mn) / (mx - mn)


def compute_proj_edge_map(gt_image):
    """GT 投影的拉普拉斯边缘图（ImprovedGS EAS 移植；对 CT 浮点投影：
    先 min-max 到 [0,1]，拉普拉斯滤波，clamp [0,1]，再 min-max 归一化）。
    gt_image: [C,H,W]；返回 [H,W] 归一化边缘权重。"""
    img = gt_image[:1].detach().float()
    img = normalize_to_unit_range(img)
    edge = F.conv2d(img.unsqueeze(0), EDGE_KERNEL.to(img.device), padding=1)
    edge = torch.clamp(edge, min=0.0, max=1.0)
    return normalize_to_unit_range(edge).squeeze(0).squeeze(0)


def compute_eas_scores(gaussians, scene, edge_maps, sample_cams):
    """EAS 边缘感知得分（论文式 6/7 忠实版，无需 CUDA 内核改动）。

    官方 S_ij = Σ_p ω_p·α_render（α_render = 渲染权重 T·α）。由链式法则：
    ∂ℓ/∂d = Σ_p ω·T·c·σ'(d) = α(1−α)·Σ_p ω·T（灰度 c=1），故
        S_i = α_i·Σ_p ω·T = ∂ℓ/∂d_i / (1−α_i)
    对每个采样视角：ℓ = Σ ω·I；∂ℓ/∂density 用 autograd（保留 (1−α) 除法的
    数值稳定 clamp）；跨视角直接平均（式 7，无 min-max 归一化，保留绝对
    贡献度）。返回 [N] 得分（不可见高斯为 0）；Σ≤0 回退 None。
    每次 densify 事件额外成本 ≈ sample_cams 次 渲染+反向（≈+5% 总计算）。
    """
    cams = scene.getTrainCameras()
    n_cams = len(cams)
    k = min(int(sample_cams), n_cams)
    if k <= 0 or not edge_maps or len(edge_maps) != n_cams:
        return None
    idx = np.random.choice(n_cams, k, replace=False)
    alpha = gaussians.get_density.detach().squeeze(-1)  # 当前 opacity（乘子）
    scores = torch.zeros(
        gaussians.get_xyz.shape[0], device="cuda", dtype=torch.float32
    )
    for i in idx:
        cam = cams[i]
        with torch.enable_grad():
            render_pkg = rasterize_proj(cam, gaussians)
            img = render_pkg["render"]  # [C,H,W]
            w = edge_maps[i].to(img.device)
            if w.ndim == 2:
                w = w.unsqueeze(0)  # 匹配单通道投影
            l = (img * w).sum()
        g = torch.autograd.grad(l, gaussians._density, retain_graph=False)[0]
        s = g.squeeze(-1) / (1.0 - alpha).clamp(min=1e-3)  # = α·ΣωT
        vis = render_pkg["visibility_filter"]
        scores[vis] += s[vis] / k
    if scores.sum() <= 0:
        return None  # 采样视角均无边缘信号（如全均匀投影）→ 调用方回退梯度得分
    return scores


def compute_improved_budget(step, budget_absolute, densify_from_step,
                            densify_until_step, warmup_offset):
    """Growth Control 预算（ImprovedGS 移植）：√progress 爬坡到 budget_absolute。

    progress = (step − from) / (until − offset − from)，区间外饱和；
    budget_absolute<=0 时返回极大值（不设预算上限，退化为全候选分裂）。
    """
    if budget_absolute <= 0:
        return int(1e18)
    end = densify_until_step - warmup_offset
    if end <= densify_from_step:
        return int(budget_absolute)
    progress = (step - densify_from_step) / float(end - densify_from_step)
    progress = min(max(progress, 0.0), 1.0)
    if progress >= 1.0:
        return int(budget_absolute)
    return max(int(progress ** 0.5 * budget_absolute), 1)


def run_improved_densification(gaussians, scene, edge_maps, optim_args, step,
                               densify_until_step, max_num_gaussians_limit,
                               densify_scale_threshold, max_scale,
                               outside_box, outside_budget):
    """ImprovedGS 移植的 densify 事件（Growth Control + LAS + EAS 调度入口）。

    计算 EAS 得分（eas_use!=off 时）、√爬坡预算、资格（梯度 OR 边缘 top
    分位，qualify 模式），调用 gaussians.densify_and_prune_improved。
    返回决策梯度（与 legacy densify_and_prune 一致，供事件记录复用）。
    """
    eas_use = str(getattr(optim_args, "eas_use", "off"))
    budget_absolute = int(getattr(optim_args, "improved_budget_absolute", 0) or 0)
    if budget_absolute <= 0 and max_num_gaussians_limit:
        budget_absolute = int(max_num_gaussians_limit)
    warmup_offset = int(getattr(optim_args, "improved_budget_warmup_until_offset", 500))
    budget = compute_improved_budget(
        step,
        budget_absolute,
        int(optim_args.densify_from_step),
        densify_until_step,
        warmup_offset,
    )

    scores = None
    if eas_use != "off" and edge_maps:
        scores = compute_eas_scores(
            gaussians, scene, edge_maps,
            int(getattr(optim_args, "eas_sample_cams", 10)),
        )
        # EAS 得分即分裂概率权重（论文式 7）：候选仍由梯度阈值（绝对值
        # 梯度）决定，long_axis_split 按 scores 做 multinomial 抽样。
        # 旧的 qualify-mask 语义（EAS 得分 OR 进候选池）已废弃。

    return gaussians.densify_and_prune_improved(
        scores,
        optim_args.densify_grad_threshold,
        optim_args.density_min_threshold,
        budget,
        step,
        densify_until_step,
        optim_args.max_screen_size,
        max_scale,
        densify_scale_threshold,
        None,  # bbox — disabled（与 legacy 一致：init 坐标可能不匹配归一化 bbox）
        outside_box=outside_box,
        outside_budget=outside_budget,
        use_las=bool(getattr(optim_args, "use_las", True)),
        split_distance=float(getattr(optim_args, "split_distance", 0.45)),
        density_reduction=float(getattr(optim_args, "split_density_reduction", 0.6)),
    )


# 出生/剪枝编码（与 gaussian_model.py 事件记录一致）
BIRTH_TYPE_CODE = {"clone": 1, "split": 2}
PRUNE_TAG_CODE = {"prune": 1, "split_parent": 2}
PRUNE_REASON_NAME = {1: "density", 2: "bbox", 3: "screen", 4: "scale"}


def save_population_event(scene, step, xyz_pre, xyz_grad_pre, gaussians,
                          scale_pre=None, density_pre=None):
    """densification 事件的人口记账存档（population_events/step_XXXXXX.npz）。

    xyz_pre / xyz_post：事件前后的完整 xyz 快照。两次相邻事件之间无结构变化
    （densify/prune 只发生在事件内），因此 xyz_pre(事件 e+1) 与 xyz_post(事件 e)
    逐行对齐 → 幸存高斯的真实位移（plan §17：actual displacement 优先）。
    xyz_grad_pre：事件前一步反向传播的世界坐标梯度 ∂L/∂xyz（形状与 xyz_pre 对齐；
    None 表示该步 xyz 无梯度）。
    scale_pre/density_pre/post：事件前后的尺度/密度快照（研究边界大尺度动力学）。
    birth_*：本次事件新出生高斯（clone=1/split=2）及其 parent 位置。
    prune_*：本次事件被删高斯的位置与原因（tag 1=常规剪枝[reason 码见
    PRUNE_REASON_NAME]，tag 2=split 后被移除的 parent）。
    """
    events = list(gaussians.event_log)
    gaussians.event_log = []
    payload = {
        "step": np.asarray(step),
        "xyz_pre": xyz_pre.cpu().numpy(),
        "xyz_post": gaussians.get_xyz.detach().cpu().numpy(),
    }
    if xyz_grad_pre is not None:
        payload["xyz_grad_pre"] = xyz_grad_pre.cpu().numpy()
    if scale_pre is not None:
        payload["scale_pre"] = scale_pre.cpu().numpy()
        payload["scale_post"] = gaussians.get_scaling.detach().cpu().numpy()
        payload["density_pre"] = density_pre.cpu().numpy()
        payload["density_post"] = gaussians.get_density.detach().cpu().numpy()
    births = [e for e in events if e["kind"] == "birth"]
    prunes = [e for e in events if e["kind"] == "prune"]
    if births:
        payload["birth_xyz"] = np.concatenate([e["xyz"] for e in births], 0)
        payload["birth_type"] = np.concatenate(
            [np.full(len(e["xyz"]), BIRTH_TYPE_CODE[e["type"]], dtype=np.int8)
             for e in births]
        )
        payload["birth_parent_xyz"] = np.concatenate(
            [e["parent_xyz"] for e in births], 0
        )
    if prunes:
        payload["prune_xyz"] = np.concatenate([e["xyz"] for e in prunes], 0)
        payload["prune_tag"] = np.concatenate(
            [np.full(len(e["xyz"]), PRUNE_TAG_CODE[e["tag"]], dtype=np.int8)
             for e in prunes]
        )
        reasons = [e["reason"] for e in prunes if e["reason"] is not None]
        if reasons:
            payload["prune_reason"] = np.concatenate(reasons, 0)
    out_dir = osp.join(scene.model_path, "population_events")
    os.makedirs(out_dir, exist_ok=True)
    np.savez(osp.join(out_dir, f"step_{step:06d}.npz"), **payload)


def optimize(config, profiler=None, tb_writer=None):
    """Run the full reconstruction optimization loop.

    Args:
        config: Hydra config with ``model``, ``optim`` and ``eval`` sections.
        profiler: Optional torch profiler context returned by ``setup_profiler``.
    """
    model_args, optim_args, eval_args = config.model, config.optim, config.eval
    scene = SceneRecon(model_args, shuffle=False)
    # Record the camera geometry corrections used for this run so evaluation
    # can reproduce them later.
    with open(osp.join(model_args.model_path, "geometry_used.yml"), "w") as handle:
        yaml.safe_dump(dict(model_args.geometry), handle)
    training_time_seconds = 0.0
    eval_and_save_time_seconds = 0.0

    print(f"Data source path: {model_args.data_source_path}")
    print(f"Model save path: {scene.model_path}")

    scanner_cfg = scene.scanner_cfg
    bbox = scene.bbox
    volume_to_world = max(scanner_cfg["sVoxel"])

    # 干预实验（研究用）：重建体积框（缩放坐标），供 clamp / 框外预算 / 出生 clamp 使用
    sv_box = np.asarray(scanner_cfg["sVoxel"], dtype=float)
    off_box = np.asarray(scanner_cfg["offOrigin"], dtype=float)
    box_lo_np = off_box - sv_box / 2.0
    box_hi_np = off_box + sv_box / 2.0

    # Experiment F（plan §12，研究用）：位置 clamp 到重建体积框
    clamp_box = bool(getattr(model_args, "position_clamp_box", False))
    clamp_lo = clamp_hi = None
    if clamp_box:
        clamp_lo = torch.tensor(box_lo_np, dtype=torch.float32, device="cuda")
        clamp_hi = torch.tensor(box_hi_np, dtype=torch.float32, device="cuda")
        print(f"Position clamp enabled: box lo={clamp_lo.tolist()} hi={clamp_hi.tolist()}")

    # 2a 干预（研究用）：框外高斯预算上限（densify 选择排除）
    outside_budget = float(getattr(optim_args, "outside_gaussian_budget", 0.0) or 0.0)
    outside_box = None
    if outside_budget > 0:
        outside_box = torch.stack([
            torch.tensor(box_lo_np, dtype=torch.float32, device="cuda"),
            torch.tensor(box_hi_np, dtype=torch.float32, device="cuda"),
        ])
        print(f"Outside gaussian budget enabled (2a): {outside_budget:.2f}")
    scale_bound = None
    if model_args.scale_min > 0 and model_args.scale_max > 0:
        scale_bound = np.array([model_args.scale_min, model_args.scale_max]) * volume_to_world
    voxelizefunc = lambda x: voxelize_vol(
        x,
        scanner_cfg["offOrigin"],
        scanner_cfg["nVoxel"],
        scanner_cfg["sVoxel"],
    )

    # Gaussian densification params
    max_scale = optim_args.max_scale * volume_to_world if optim_args.max_scale else None
    densify_scale_threshold = (
        optim_args.densify_scale_threshold * volume_to_world
        if optim_args.densify_scale_threshold
        else None
    )

    # Set up Gaussians
    gaussians = GaussianModel(scale_bound)
    init_mode = model_args.init_mode
    if init_mode == "auto":
        dataset_name = osp.basename(osp.normpath(model_args.data_source_path))
        init_path = osp.join(
            model_args.data_source_path, f"init_{dataset_name}.npy"
        )
        # Match the r2_gaussian/r2_gaussian_spiral cold-start baseline.  Their
        # initialize_pcd.py uniformly samples foreground voxels from an FDK
        # volume, saves the result, and then trains from that file.  "intensity"
        # is the historical FaCT-GS name for this uniform sampling strategy.
        init_mode = "precomputed" if osp.exists(init_path) else "intensity"
        print(f"Auto initialization selected '{init_mode}'.")

    if init_mode == "precomputed":
        initialize_gaussian(gaussians, model_args, None)
    elif init_mode in ["gradient", "intensity"]:
        # Pass the resolved mode explicitly.  Previously the auto fallback
        # still passed model_args.init_mode == "auto" into sample_vol(), where
        # it matched no sampling branch and left sampled_indices undefined.
        initialize_gaussian_from_proj(
            gaussians, model_args, optim_args, scene, init_mode=init_mode
        )
    elif init_mode == "prior":
        initialize_gaussian_from_prior(gaussians, model_args)
    else:
        raise ValueError(f"Unknown initialization mode {model_args.init_mode}")
    scene.gaussians = gaussians
    gaussians.training_setup(optim_args)

    # ImprovedGS 移植（improved-densification 分支）：densify 调度 + EAS 边缘图
    densification_method = str(getattr(optim_args, "densification_method", "legacy"))
    eas_use = str(getattr(optim_args, "eas_use", "off"))
    edge_maps = []
    if densification_method == "improved" and eas_use != "off":
        t0 = time.perf_counter()
        edge_maps = [
            compute_proj_edge_map(cam.original_image.cpu())
            for cam in scene.getTrainCameras()
        ]
        print(
            f"[ImprovedGS] Precomputed {len(edge_maps)} projection edge maps "
            f"for EAS ({time.perf_counter() - t0:.1f}s)"
        )

    # 2b 干预（研究用）：densify 出生的子代 clamp 回框内（box 见上方 208 行）
    if bool(getattr(model_args, "clamp_child_positions", False)):
        gaussians.clamp_birth_box = (
            torch.tensor(box_lo_np, dtype=torch.float32, device="cuda"),
            torch.tensor(box_hi_np, dtype=torch.float32, device="cuda"),
        )
        print("Child-position clamp enabled (2b): births clamped to box, "
              "optimizer moves remain free")

    # Translate the configured percentage to an absolute cap for densification
    max_num_gaussians_limit = None
    max_num_gaussians_absolute = getattr(
        optim_args, "max_num_gaussians_absolute", None
    )
    if max_num_gaussians_absolute is not None:
        max_num_gaussians_limit = int(max_num_gaussians_absolute)
    elif optim_args.max_num_gaussians is not None:
        initial_gaussians = getattr(model_args, "num_gaussians", None)
        if initial_gaussians is None or initial_gaussians <= 0:
            initial_gaussians = gaussians.get_xyz.shape[0]
        max_num_gaussians_limit = int(initial_gaussians * optim_args.max_num_gaussians)
    densify_until_step = int(optim_args.densify_until_step_percent * optim_args.steps)

    # Set up loss
    use_tv = optim_args.lambda_tv > 0
    if use_tv:
        print("Use total variation loss")
        tv_vol_size = optim_args.tv_vol_size
        tv_vol_nVoxel = torch.tensor([tv_vol_size, tv_vol_size, tv_vol_size])
        tv_vol_sVoxel = torch.tensor(scanner_cfg["dVoxel"]) * tv_vol_nVoxel
        reduction_value = 3*(tv_vol_size-1)*tv_vol_size**2


    ckpt_save_path = osp.join(scene.model_path, "ckpt")
    os.makedirs(ckpt_save_path, exist_ok=True)
    viewpoint_stack = None
    progress_bar = tqdm(range(0, optim_args.steps), desc="Train", leave=False)
    latest_eval_metrics = None
    densify_record_enabled = bool(getattr(eval_args, "record_densify_grads", False))
    population_record_enabled = bool(getattr(eval_args, "record_population", False))
    sibr_export_enabled = bool(getattr(eval_args, "sibr_export", False))
    sibr_export_interval = int(getattr(eval_args, "sibr_export_interval", 1000))
    sibr_camera_mode = str(getattr(eval_args, "sibr_camera_mode", "orbit"))
    if sibr_export_enabled and sibr_export_interval <= 0:
        raise ValueError("eval.sibr_export_interval must be positive")
    if population_record_enabled:
        gaussians.event_log_enabled = True
        gaussians.event_log = []
    last_densify = None  # (step, xyz_before, decision_grads) 最近一次 densification 事件
    for step in range(optim_args.steps + 1):
        step_time_start = time.perf_counter()
        # Update learning rate
        gaussians.update_learning_rate(step)

        # Get one camera for training
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        # Render X-ray projection
        render_pkg = rasterize_proj(viewpoint_cam, gaussians)
        image, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg["render"],
            render_pkg["viewspace_points"],
            render_pkg["visibility_filter"],
            render_pkg["radii"],
        )

        # Compute loss
        gt_image = viewpoint_cam.original_image.cuda()
        loss = {"total": 0.0}
        render_loss = l1_loss(image, gt_image)
        loss["render"] = render_loss
        loss["total"] += loss["render"]
        if optim_args.lambda_dssim > 0:
            if getattr(optim_args, "use_fused_ssim", False):
                ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
            else:
                # Match the zero-padded 11x11 SSIM used by the reported
                # r2_gaussian_spiral baseline.  Rasterization remains fused.
                ssim_value = ssim(image, gt_image)
            loss_dssim = 1.0 - ssim_value
            loss["dssim"] = loss_dssim
            loss["total"] = loss["total"] + optim_args.lambda_dssim * loss_dssim
        lambda_frequency = float(getattr(optim_args, "lambda_frequency", 0.0))
        if lambda_frequency > 0:
            loss_frequency = frequency_magnitude_loss(
                image,
                gt_image,
                highpass_cutoff=float(
                    getattr(optim_args, "frequency_highpass_cutoff", 0.1)
                ),
            )
            loss["frequency"] = loss_frequency
            loss["total"] = loss["total"] + lambda_frequency * loss_frequency

        # 3D TV loss
        if use_tv:
            # Randomly get the tiny volume center
            tv_vol_center = (bbox[0] + tv_vol_sVoxel / 2) + (
                bbox[1] - tv_vol_sVoxel - bbox[0]
            ) * torch.rand(3)
            
            vol_pred = voxelize_vol(
                gaussians,
                tv_vol_center,
                tv_vol_nVoxel,
                tv_vol_sVoxel,
            )["vol"]
            loss_tv = tv_3d_loss(vol_pred, reduction_value=reduction_value)
            loss["tv"] = loss_tv
            loss["total"] = loss["total"] + optim_args.lambda_tv * loss_tv
        loss["total"].backward()

        with torch.no_grad():
            # Adaptive control
            gaussians.max_radii2D[visibility_filter] = torch.max(
                gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
            )

            # Optional densification
            if optim_args.densify_gaussians:
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                if step < densify_until_step:
                    if (
                        step > optim_args.densify_from_step
                        and step % optim_args.densification_interval == 0
                    ):
                        # 记录 densification 决策梯度：densify_and_prune 返回
                        # xyz_gradient_accum/denom（postfix 重置前的值）
                        xyz_before = gaussians.get_xyz.detach()
                        xyz_grad_before = gaussians._xyz.grad
                        if xyz_grad_before is not None:
                            # densify 会替换参数张量；先克隆本步的 world 梯度
                            xyz_grad_before = xyz_grad_before.detach().clone()
                        scale_before = gaussians.get_scaling.detach()
                        density_before = gaussians.get_density.detach()
                        if densification_method == "improved":
                            densify_grads = run_improved_densification(
                                gaussians,
                                scene,
                                edge_maps,
                                optim_args,
                                step,
                                densify_until_step,
                                max_num_gaussians_limit,
                                densify_scale_threshold,
                                max_scale,
                                outside_box,
                                outside_budget,
                            )
                        else:
                            densify_grads = gaussians.densify_and_prune(
                                optim_args.densify_grad_threshold,
                                optim_args.density_min_threshold,
                                optim_args.max_screen_size,
                                max_scale,
                                max_num_gaussians_limit,
                                densify_scale_threshold,
                                None,  # bbox — disabled: init coords may not match scene_scale-normalized bbox
                                outside_box=outside_box,
                                outside_budget=outside_budget,
                            )
                        last_densify = (step, xyz_before, densify_grads)
                        if densify_record_enabled:
                            record_densify_event(
                                step,
                                xyz_before,
                                densify_grads,
                                scene.model_path,
                                tb_writer,
                                optim_args.densify_grad_threshold,
                            )
                        if population_record_enabled:
                            save_population_event(
                                scene, step, xyz_before, xyz_grad_before, gaussians,
                                scale_before, density_before,
                            )
                        # print(f"Number of Gaussians after densification: {gaussians.get_density.shape[0]}")
                        # print(f"Scale after densification, max: {gaussians.get_scaling.max().item():.4f}, mean: {gaussians.get_scaling.mean().item():.4f}, min: {gaussians.get_scaling.min().item():.4f}")
                        # print(f"Density after densification, max: {gaussians.get_density.max().item():.4e}, mean: {gaussians.get_density.mean().item():.4e}, min: {gaussians.get_density.min().item():.4e}")   
                

            # Optimization
            if step < optim_args.steps:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
                # Experiment F（plan §12）：把高斯中心 clamp 到重建体积框内，
                # 诊断 volume-outside 高斯是否是被优化/出生"挤出"的（研究用）。
                if clamp_box:
                    with torch.no_grad():
                        gaussians._xyz.clamp_(min=clamp_lo, max=clamp_hi)

            training_time_seconds += time.perf_counter() - step_time_start
            step_non_training_time_start = time.perf_counter()

            # Save gaussians
            if step == optim_args.steps:
                tqdm.write(f"[STEP {step}] Saving Gaussians")
                scene.save(step, voxelizefunc, vol_format="tiff")

            if sibr_export_enabled and (
                step % sibr_export_interval == 0 or step == optim_args.steps
            ):
                stats = export_gaussian_model(
                    gaussians,
                    scene.model_path,
                    step,
                    cameras=scene.getTrainCameras(),
                    viewer_bbox=scene.bbox.detach().cpu().numpy(),
                    viewer_orbit_radius=float(scanner_cfg["DSO"]),
                    viewer_camera_mode=sibr_camera_mode,
                )
                tqdm.write(
                    f"[STEP {step}] SIBR snapshot: {stats['count']} Gaussians, "
                    f"density [{stats['density_min']:.3g}, {stats['density_max']:.3g}]"
                )

            # Progress bar
            if step % 10 == 0:
                progress_bar.set_postfix(
                    {
                        "loss": f"{loss['total'].item():.1e}",
                        "pts": f"{gaussians.get_density.shape[0]:2.1e}",
                    }
                )
                progress_bar.update(10)
            if step == optim_args.steps:
                progress_bar.close()

            # Logging
            metrics = {}
            for l in loss:
                metrics["loss_" + l] = loss[l].item()
            for param_group in gaussians.optimizer.param_groups:
                metrics[f"lr_{param_group['name']}"] = param_group["lr"]
            if tb_writer is not None and step % config.tensorboard.log_interval == 0:
                for name, value in metrics.items():
                    group = "loss" if name.startswith("loss_") else "learning_rate"
                    tb_writer.add_scalar(f"{group}/{name}", value, step)
                tb_writer.add_scalar(
                    "scene/num_gaussians", gaussians.get_xyz.shape[0], step
                )
            time_limit_seconds = getattr(optim_args, "training_time_limit_seconds", 0.0)
            time_limit_hit = time_limit_seconds > 0 and training_time_seconds >= time_limit_seconds

            eval_metrics = log_training_status(
                step,
                metrics,
                training_time_seconds,
                eval_and_save_time_seconds,
                optim_args.steps,
                eval_args,
                scene,
                lambda x, y: rasterize_proj(x, y),
                voxelizefunc,
                init_mode,
                force_eval=time_limit_hit,
                tb_writer=tb_writer,
                last_densify=last_densify,
            )

            if eval_metrics is not None:
                latest_eval_metrics = eval_metrics

            eval_and_save_time_seconds += time.perf_counter() - step_non_training_time_start

            # Early stopping based on 3D SSIM
            if optim_args.ssim3d_early_stop and eval_metrics is not None:
                ssim_value = eval_metrics.get("ssim_3d")
                threshold = getattr(optim_args, "ssim3d_early_stop_threshold", 0.0)
                if ssim_value is not None and threshold > 0 and ssim_value >= threshold:
                    iteration_str = f"{eval_metrics['iteration']:.2f}"
                    tqdm.write(
                        f"[STEP {step}] Early stopping triggered at iteration {iteration_str} "
                        f"(SSIM={eval_metrics['ssim_3d']:.4f} >= {threshold:.3f})"
                    )
                    if step != optim_args.steps:
                        scene.save(step, voxelizefunc, vol_format="tiff")
                    save_final_metrics(scene.model_path, eval_metrics, training_time_seconds)
                    break

            # Early stopping based on accumulated training time
            if time_limit_hit:
                tqdm.write(
                    f"[STEP {step}] Stopping because training time "
                    f"{training_time_seconds:.2f}s exceeded limit {time_limit_seconds:.2f}s"
                )
                if step != optim_args.steps:
                    scene.save(step, voxelizefunc, vol_format="tiff")
                metrics_to_save = eval_metrics or latest_eval_metrics
                if metrics_to_save is not None:
                    save_final_metrics(scene.model_path, metrics_to_save, training_time_seconds)
                break

            # Research-only early stop: 停在第 N 步（该步 eval 已在循环内完成）。
            # 注意 optim.steps 保持不变（LR 衰减等按完整步数走），保证 0→N 轨迹
            # 与完整训练的前 N 步一致，可用于机制拆分实验（plan §7/§11/§12）。
            early_stop_after = int(getattr(optim_args, "early_stop_after_steps", 0) or 0)
            if early_stop_after > 0 and step >= early_stop_after:
                tqdm.write(f"[STEP {step}] Stopping (early_stop_after_steps={early_stop_after})")
                if step != optim_args.steps:
                    scene.save(step, voxelizefunc, vol_format="tiff")
                metrics_to_save = eval_metrics or latest_eval_metrics
                if metrics_to_save is not None:
                    save_final_metrics(scene.model_path, metrics_to_save, training_time_seconds)
                break

            if profiler is not None:
                profiler.step()
    progress_bar.close()
    torch.cuda.empty_cache()

def save_final_metrics(model_path, metrics, training_time_seconds):
    """Save final evaluation metrics to a YAML file for easy retrieval."""
    if not metrics:
        return
    eval_save_path = osp.dirname(model_path)
    yaml_name = f"{osp.basename(model_path)}_metrics_final.yml"
    # Prefer explicit keys; fall back to the test-split projection metrics.
    psnr_2d = metrics.get("psnr_2d", metrics.get("render_test_psnr_2d"))
    ssim_2d = metrics.get("ssim_2d", metrics.get("render_test_ssim_2d"))
    payload = {
        "psnr_3d": metrics.get("psnr_3d"),
        "ssim_3d": metrics.get("ssim_3d"),
        "psnr_2d": psnr_2d,
        "ssim_2d": ssim_2d,
        "time_training_seconds": training_time_seconds,
    }
    with open(osp.join(eval_save_path, yaml_name), "w") as f:
        yaml.dump(payload, f, default_flow_style=False, sort_keys=False)

def log_training_status(
    step,
    metrics_train,
    training_time_seconds,
    eval_and_save_time_seconds,
    max_steps,
    eval_args,
    scene: SceneRecon,
    renderFunc,
    voxelizeFunc,
    init_mode,
    force_eval=False,
    tb_writer=None,
    last_densify=None,
):
    """Evaluate/visualize model checkpoints and persist metrics.

    Args:
        step: Current global training step.
        metrics_train: Dictionary with scalar loss/learning-rate values.
        training_time_seconds: Total time spent on training.
        eval_and_save_time_seconds: Total time spent on evaluation and saving.
        max_steps: Final step count that signals completion.
        eval_args: Evaluation sub-config that controls cadence and visualization.
        scene: Scene wrapper that stores Gaussians, volumes and metadata.
        renderFunc: Callable that renders projections given a camera and model.
        voxelizeFunc: Callable that voxelizes the global Gaussian model.
        init_mode: String describing how Gaussians were initialized (for logging).
        last_densify: Optional (step, xyz, grads) of the latest densification event.
    """

    iter_num = step / len(scene.getTrainCameras())

    eval_metrics = None

    should_eval = force_eval or (
        (eval_args.eval_in_training and step % eval_args.every_n_steps == 0 and step != 0)
        or (eval_args.eval_end and step == max_steps)
        or (eval_args.eval_start and step == 0)
        or (eval_args.extra_eval_iter_num is not None and iter_num == eval_args.extra_eval_iter_num)
    )

    if should_eval:
        # Evaluate 2D rendering performance
        if step == 0:
            eval_save_path = osp.join(scene.model_path, "eval", f"init_{init_mode}")
        else:
            eval_save_path = osp.join(scene.model_path, "eval", f"step_{step:06d}")
        os.makedirs(eval_save_path, exist_ok=True)
        torch.cuda.empty_cache()

        #Dump time to yaml
        time_dict = {
            "training_time_seconds": float(training_time_seconds),
            "eval_and_save_time_seconds": float(eval_and_save_time_seconds),
        }
        with open(osp.join(eval_save_path, f"time.yml"), "w") as f:
            yaml.dump(time_dict, f, default_flow_style=False, sort_keys=False)

        # 记录当前 densification 决策梯度（完整 per-Gaussian xyz+grad）
        if getattr(eval_args, "record_densify_grads", False) and step > 0:
            save_densify_grads(scene, eval_save_path, last_densify, step)

        validation_configs = [
            {"name": "render_train", "cameras": scene.getTrainCameras()},
            {"name": "render_test", "cameras": scene.getTestCameras()},
        ]
        metrics_2d = {}
        psnr_2d, ssim_2d = None, None
        for config in validation_configs:
            if config["cameras"] and len(config["cameras"]) > 0:
                images = []
                gt_images = []
                image_show_2d = []
                # Render projections
                show_idx = np.linspace(0, len(config["cameras"]), 7).astype(int)[1:-1]
                for idx, viewpoint in enumerate(config["cameras"]):
                    image = renderFunc(
                        viewpoint,
                        scene.gaussians,
                    )["render"]
                    gt_image = viewpoint.original_image.to("cuda")
                    images.append(image)
                    gt_images.append(gt_image)
                images = torch.concat(images, 0).permute(1, 2, 0)
                gt_images = torch.concat(gt_images, 0).permute(1, 2, 0)
                psnr_2d, psnr_2d_projs = metric_proj(gt_images, images, "psnr")
                ssim_2d, ssim_2d_projs = metric_proj(gt_images, images, "ssim")
                eval_dict_2d = {
                    "psnr_2d": psnr_2d,
                    "ssim_2d": ssim_2d,
                    "psnr_2d_projs": psnr_2d_projs,
                    "ssim_2d_projs": ssim_2d_projs,
                }
                metrics_2d[config["name"]] = eval_dict_2d
                with open(
                    osp.join(eval_save_path, f"eval2d_{config['name']}.yml"),
                    "w",
                ) as f:
                    yaml.dump(
                        eval_dict_2d, f, default_flow_style=False, sort_keys=False
                    )
                tqdm.write(
                    f"[STEP {step}] {config['name']}: "
                    f"PSNR2D {psnr_2d:.3f}, SSIM2D {ssim_2d:.4f}"
                )

        # Evaluate 3D reconstruction performance
        voxelize_pkg = voxelizeFunc(scene.gaussians)
        vol_pred = voxelize_pkg["vol"]
        if scene.scanner_cfg.get("coord_left", False):
            vol_pred = torch.flip(vol_pred, dims=[0])
        vol_gt = scene.vol_gt
        psnr_3d, _ = metric_vol(vol_gt, vol_pred, "psnr")
        ssim_3d, ssim_3d_axis = metric_vol(vol_gt, vol_pred, "ssim")
        eval_metrics = {
            "psnr_3d": float(psnr_3d),
            "ssim_3d": float(ssim_3d),
            "ssim_3d_x": float(ssim_3d_axis[0]),
            "ssim_3d_y": float(ssim_3d_axis[1]),
            "ssim_3d_z": float(ssim_3d_axis[2]),
            "step": float(step),
            "iteration": float(iter_num),
        }
        for split_name, split_metrics in metrics_2d.items():
            eval_metrics[f"{split_name}_psnr_2d"] = float(split_metrics["psnr_2d"])
            eval_metrics[f"{split_name}_ssim_2d"] = float(split_metrics["ssim_2d"])
        eval_dict = {
            "psnr_3d": eval_metrics["psnr_3d"],
            "ssim_3d": eval_metrics["ssim_3d"],
            "ssim_3d_x": eval_metrics["ssim_3d_x"],
            "ssim_3d_y": eval_metrics["ssim_3d_y"],
            "ssim_3d_z": eval_metrics["ssim_3d_z"],
        }
        with open(osp.join(eval_save_path, "eval3d.yml"), "w") as f:
            yaml.dump(eval_dict, f, default_flow_style=False, sort_keys=False)

        if tb_writer is not None:
            for split_name, split_metrics in metrics_2d.items():
                tb_writer.add_scalar(
                    f"projection/{split_name}_psnr_2d",
                    split_metrics["psnr_2d"],
                    step,
                )
                tb_writer.add_scalar(
                    f"projection/{split_name}_ssim_2d",
                    split_metrics["ssim_2d"],
                    step,
                )
            tb_writer.add_scalar("reconstruction/psnr_3d", psnr_3d, step)
            tb_writer.add_scalar("reconstruction/ssim_3d", ssim_3d, step)
            for axis_name, axis_value in zip("xyz", ssim_3d_axis):
                tb_writer.add_scalar(
                    f"reconstruction/ssim_3d_{axis_name}", axis_value, step
                )
            center = vol_gt.shape[0] // 2
            gt_slice = vol_gt[center].detach().float().cpu()
            pred_slice = vol_pred[center].detach().float().cpu()
            comparison = torch.stack(
                [gt_slice, pred_slice, torch.abs(gt_slice - pred_slice)], dim=0
            )
            tb_writer.add_image(
                "reconstruction/center_slice_gt_pred_error", comparison, step
            )
            tb_writer.add_histogram(
                "scene/density", scene.gaussians.get_density.detach().cpu(), step
            )
            tb_writer.flush()
        
        tqdm.write(
            f"[STEP {step}] Iter: {int(np.floor(iter_num))}, Training Time: {training_time_seconds:.2f}s. Evaluating: psnr3d {psnr_3d:.3f}, ssim3d {ssim_3d:.3f}, psnr2d {psnr_2d:.3f}, ssim2d {ssim_2d:.3f}"
        )

        if eval_args.visualize_at_eval:
            vol_save_path = osp.join(eval_save_path, "vol_pred.tiff")
            save_volume(vol_pred, vol_save_path, save_preview=True, save_volume=True)
            if step == 0:
                vol_gt_save_path = osp.join(scene.model_path, "vol_gt.tiff")
                save_volume(vol_gt, vol_gt_save_path, save_preview=True, save_volume=True)
        
        if eval_args.visualize_gaussians:
            scene.gaussians.save_ply(osp.join(eval_save_path, "point_cloud.pickle"))
            gauss_position_path = osp.join(eval_save_path, "gaussian_positions/")
            gauss_footprint_path = osp.join(eval_save_path, "gaussian_footprints/")
            error_maps_path = osp.join(eval_save_path, "error_maps/")
            os.makedirs(gauss_position_path, exist_ok=True)
            os.makedirs(gauss_footprint_path, exist_ok=True)
            os.makedirs(error_maps_path, exist_ok=True)

            visualize_gaussian_position(gauss_position_path, 
                                        scene.gaussians.get_xyz,
                                        voxelize_pkg["radii"], 
                                        scene.scanner_cfg["sVoxel"],
                                        scene.scanner_cfg["dVoxel"],
                                        scene.scanner_cfg["offOrigin"],)
            visualize_gaussian_footprint(gauss_footprint_path, 
                                        scene.gaussians.get_xyz, 
                                        voxelize_pkg["radii"], 
                                        voxelize_pkg["conics"],
                                        scene.gaussians.get_density, 
                                        scene.scanner_cfg["sVoxel"],
                                        scene.scanner_cfg["dVoxel"],    
                                        scene.scanner_cfg["offOrigin"],)
            save_error_maps(error_maps_path, vol_pred, vol_gt)

        if eval_args.extra_eval_iter_num is not None and iter_num == eval_args.extra_eval_iter_num:
            eval_dict = {
                "psnr_3d": eval_metrics["psnr_3d"],
                "ssim_3d": eval_metrics["ssim_3d"],
                "training_time_seconds": training_time_seconds,
                "eval_and_save_time_seconds": eval_and_save_time_seconds,
            }
            eval_save_path = osp.dirname(scene.model_path)
            yaml_name = f"{osp.basename(scene.model_path)}_metrics_{int(iter_num)}.yml"
            with open(osp.join(eval_save_path, yaml_name), "w") as f:
                yaml.dump(eval_dict, f, default_flow_style=False, sort_keys=False)

        if step == max_steps:
            save_final_metrics(scene.model_path, eval_metrics, training_time_seconds)

    return eval_metrics

if __name__ == "__main__":
    train_recon()
    
