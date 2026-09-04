"""绘制 densification 决策梯度（xyz_gradient_accum/denom）的时空演化图。

输入：模型目录（train_recon.py 开启 eval.record_densify_grads 后生成）：
- grad_maps/step_XXXXXX.npz   —— 每个 densification 事件（step 500..15000 每 100 步）的
  compact 记录：profiles(3,256) 三轴梯度加权位置剖面（xyz 列序: 0=x, 1=y, 2=z）、
  hist_counts/edges 梯度直方图、summary [step, n, above_thresh, mean, p99]
- eval/step_XXXXXX/densify_grads.npz —— eval 步的完整 (xyz, grads)（accum_window 或
  densify_decision，kind 字段区分）。xyz 为缩放世界坐标
  （= mm × scene_scale，scene_scale=2/max(sVoxel)，1 缩放单位 = max(sVoxel)/2 mm），
  列序 (x, y, z)，轴向 z 是第 3 列。

输出（--output 目录）：
- grad_summary_vs_step.png    —— above-threshold 计数 / mean / p99 / max 随事件步演化
- grad_zprofile_evolution.png —— z 轴梯度加权剖面的时空图（事件步 × z，mm 标注）
- grad_hist_evolution.png     —— 梯度直方图叠加演化（颜色越深越晚）
- eval_grad_planes_stepXXXXXX_{xy,xz,yz}.png —— 每个 eval 步，梯度加权位置密度在
  xy/xz/yz 三个平面上的中间切片图（来自完整 densify_grads.npz，空间网格 256³）。
  坐标轴标注物理方向（x→ / y↑ / z↑），刻度为 mm（场景单位 = mm × object_scale/1000）。

说明：图里的"网格"是把高斯位置按坐标分到 256×256×256 的直方图网格（histogram bin），
每格(bin)累计落入其中的高斯的梯度权重和；切片图就是某个平面上这个 256×256 网格的
热力图，横纵轴不再叫"bin"，而是标注了真实物理方向与 mm 刻度。

用法:
    MPLCONFIGDIR=/tmp/spiral-gs-matplotlib python experiments/helpers/plot_grad_maps.py \
        --model models/real/ldctl004/spiral/ntrain1000/factgs_det4x_cold \
        --output output/real/ldctl004/spiral/ntrain1000/grad_maps
"""

import argparse
import glob
import json
import os

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, PowerNorm
except ImportError:  # pragma: no cover
    plt = None

BINS = 256
SCENE_TO_MM = 20.0  # 旧上游约定（object_scale=50）；本管线 mm/缩放单位 = max(sVoxel)/2，volume_grid 模式由 scanner 自动给出（可被 --scale-mm 覆盖）

# 体积体素索引（plot_roi 的 ROI 图默认切片位置）
SLICE_VOXEL = 128


def scanner_grid(scanner):
    """体积网格（训练缩放空间，scene_scale=2/max(sVoxel)）+ mm 换算系数。

    与 plot_c002_grad_compare_residuals.py 的 scanner_grid 相同：voxelize kernel
    把 offOrigin 当体积中心，体积世界范围 = [offOrigin−sVoxel/2, offOrigin+sVoxel/2]。
    返回 lo/span（缩放世界坐标，列序 x,y,z）与 mm 换算
    （scene_scale = 2/max(sVoxel) 直接作用于 mm 值 → 1 缩放单位 = max(sVoxel)/2 mm）。
    """
    sVoxel = np.asarray(scanner["sVoxel"], dtype=float)
    offOrigin = np.asarray(scanner["offOrigin"], dtype=float)
    ss = 2.0 / sVoxel.max()
    sv = sVoxel * ss
    lo = offOrigin * ss - sv / 2.0
    span = sv
    mm_per_unit = sVoxel.max() / 2.0
    return lo, span, mm_per_unit


def load_events(model_dir):
    """读取 grad_maps/ 下的全部事件记录，按 step 排序。"""
    files = sorted(glob.glob(os.path.join(model_dir, "grad_maps", "step_*.npz")))
    events = []
    for f in files:
        z = np.load(f)
        step, n, above, mean_v, p99 = z["summary"]
        events.append({
            "step": int(step), "n": int(n), "above": int(above),
            "mean": float(mean_v), "p99": float(p99),
            "profiles": z["profiles"],  # (3, 256) 轴序: 0=x,1=y,2=z（xyz 列序）
            "hist_counts": z["hist_counts"], "hist_edges": z["hist_edges"],
        })
    return events


def load_eval_grads(model_dir):
    """读取 eval 目录下的完整梯度记录：[(step, kind, xyz(N,3), grads(N,))]。"""
    out = []
    for d in sorted(glob.glob(os.path.join(model_dir, "eval", "step_*"))):
        f = os.path.join(d, "densify_grads.npz")
        if not os.path.isfile(f):
            continue
        z = np.load(f)
        step = int(z["step"].item())
        kind = str(z["kind"].item())
        out.append((step, kind, z["xyz"].astype(np.float64), z["grads"].astype(np.float64)))
    out.sort(key=lambda t: t[0])
    return out


def plot_summary(events, out_path):
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))
    steps = [e["step"] for e in events]
    for ax, key, title in (
        (axes[0], "above", "above densify_grad_threshold"),
        (axes[1], "mean", "mean grad"),
        (axes[2], "p99", "p99 grad"),
        (axes[3], "n", "num gaussians (at event)"),
    ):
        ax.plot(steps, [e[key] for e in events], "o-", ms=3, lw=1.2)
        ax.set_xlabel("densification event step")
        ax.set_title(title)
        ax.grid(alpha=0.3)
    axes[0].axhline(0, color="k", lw=0.8, ls="--")
    fig.suptitle("Densification decision gradients over time", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


def plot_zprofile(events, out_path, scale_mm=SCENE_TO_MM, axis=2):
    """事件步 × z 的梯度加权剖面时空图（profiles 列序: 0=x, 1=y, 2=z）。

    剖面权重同样跨数量级 → LogNorm(vmin=最小正值) 提亮低值。
    """
    steps = [e["step"] for e in events]
    prof = np.stack([e["profiles"][axis] for e in events])  # (events, 256)
    pos = prof[prof > 0]
    vmin = float(pos.min()) if pos.size else 1e-12
    vmax = float(pos.max()) if pos.size else 1.0
    norm = LogNorm(vmin=vmin, vmax=vmax)
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(prof.T, aspect="auto", origin="lower", cmap="magma",
                   norm=norm, extent=[steps[0], steps[-1], 0, prof.shape[1]])
    ax.set_xlabel("densification event step")
    ax.set_ylabel("z (uniform 256 bins over position range)")
    ax.set_title("grad-weighted z-profile over densification events (z = axial, log scale)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"saved {out_path}")


def plot_hist_evolution(events, out_path):
    fig, ax = plt.subplots(figsize=(9, 5))
    steps = [e["step"] for e in events]
    for e in events:
        edges = e["hist_edges"]
        counts = e["hist_counts"][: len(edges) - 1]
        centers = (edges[:-1] + edges[1:]) / 2
        alpha = 0.4 + 0.6 * (e["step"] - min(steps)) / max(1, max(steps) - min(steps))
        ax.semilogy(centers, counts, lw=1.1, color="tab:blue", alpha=alpha)
    ax.set_xlabel("gradient norm (xyz_gradient_accum/denom)")
    ax.set_ylabel("count (log)")
    ax.set_title("densify gradient distribution over events (darker = later)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"saved {out_path}")


def _annotate_plane(ax, plane, scale_mm):
    """按平面标注物理轴方向与 mm 刻度（世界坐标列序 xyz=(x,y,z)）。

    plane: ('xy'|'xz'|'yz')，切片图的行列 = 切片后剩余体积轴的顺序（行=第一剩余轴，
    列=第二剩余轴，与 plot_roi get_slice / grad_compare 残差图同序，不转置）：
    xy 行=x 列=y；xz 行=x 列=z；yz 行=y 列=z。
    场景单位 → mm: × scale_mm。刻度用占位 extent（调用方用 imshow extent 覆盖）。
    """
    axes_mm = {0: ("x (mm) →", "x"), 1: ("y (mm) →", "y"), 2: ("z (mm) →", "z")}
    if plane == "xy":   # 行=x, 列=y
        row, col = 0, 1
    elif plane == "xz":  # 行=x, 列=z
        row, col = 0, 2
    else:               # yz: 行=y, 列=z
        row, col = 1, 2
    ax.set_xlabel(axes_mm[col][0])
    ax.set_ylabel({0: "x (mm) ↑", 1: "y (mm) ↑", 2: "z (mm) ↑"}[row])
    ax.set_xticks([])
    ax.set_yticks([])


def eval_spatial_maps(eval_recs, out_prefix, voxel_bins=64, gamma=0.3, pct=99.9,
                      lo=None, span=None, scale_mm=SCENE_TO_MM, volume_grid=False):
    """每个 eval 步：梯度加权位置在 xy/xz/yz 三个平面的中间切片图（物理轴标注+mm 刻度）。

    两点保证可读性：
    1. 网格默认 64³（而非 256³）：500k 高斯在 256³ 网格里中间切片只有 ~3% 占用，
       画出来是稀疏散点；64³ 下 ~34% 占用，是连续的密度场（--bins 可调）。
    2. 梯度权重跨多个数量级，线性色标会全黑：用 PowerNorm(gamma<1) 提亮低值，
       色标上限取"当前切片自身"非零值的 pct 分位数（不能取全体积的非零分位——
       其他切片的高密度会把本切片压黑）。

    volume_grid=True（--dataset 指定）时用体积网格（lo/span 由 scanner_grid 给出）：
    切片取 (voxel_bins−1)//2 号 bin，其中心 = 体积中心平面，与 plot_roi 的 ROI 图
    （体积体素 128）切片位置对应；mm 刻度 = 缩放单位 × max(sVoxel)×10。
    体积外的高斯被 histogramdd 丢弃。默认（无 --dataset）用数据范围分箱。
    """
    for step, kind, xyz, grads in eval_recs:
        if volume_grid:
            slice_bin = (voxel_bins - 1) // 2
        else:
            # 世界坐标 → 直方图网格（bin），范围=该步数据范围
            lo = xyz.min(axis=0)
            span = np.ptp(xyz, axis=0)
            slice_bin = None
        idx = np.floor((xyz - lo) / (span + 1e-12) * (voxel_bins - 1)).astype(int)
        if volume_grid:
            # 体积网格模式：裁剪体积外高斯 + 显式 range，保证切片 = 体积中心平面
            inside = ((idx >= 0) & (idx < voxel_bins)).all(axis=1)
            idx, grads = idx[inside], grads[inside]
            H, _ = np.histogramdd(idx, bins=[voxel_bins] * 3,
                                  range=[(0, voxel_bins - 1)] * 3,
                                  weights=grads)  # 轴序 (x, y, z)
        else:
            H, _ = np.histogramdd(idx, bins=[voxel_bins] * 3, weights=grads)  # 轴序 (x, y, z)
        planes = {"xy": (2, (0, 1)), "xz": (1, (0, 2)), "yz": (0, (1, 2))}
        for plane, (keep_axis, (row_w, col_w)) in planes.items():
            b = slice_bin if slice_bin is not None else H.shape[keep_axis] // 2
            s = np.take(H, b, axis=keep_axis)  # (row_w, col_w)，行=第一剩余轴
            nonzero = s[s > 0]
            vmax = float(np.quantile(nonzero, pct / 100.0)) if nonzero.size else 1.0
            norm = PowerNorm(gamma=gamma, vmin=0, vmax=vmax)
            # mm 范围：切片平面保留的两个世界轴
            extent = [lo[col_w] * scale_mm, (lo[col_w] + span[col_w]) * scale_mm,
                      lo[row_w] * scale_mm, (lo[row_w] + span[row_w]) * scale_mm]
            fig, ax = plt.subplots(figsize=(7, 6))
            im = ax.imshow(s, origin="lower", cmap="inferno", norm=norm, extent=extent,
                           aspect=(extent[1] - extent[0]) / (extent[3] - extent[2]))
            _annotate_plane(ax, plane, scale_mm)
            plane_desc = f"volume voxel {SLICE_VOXEL}" if volume_grid else "mid-plane"
            ax.set_title(f"grad-weighted position density, {plane} {plane_desc}\n"
                         f"(step {step}, {kind}, PowerNorm γ={gamma}, vmax=p{pct})")
            fig.colorbar(im, ax=ax, fraction=0.046, label="cumulative grad weight")
            fig.tight_layout()
            out = f"{out_prefix}_step{step:06d}_{kind}_{plane}.png"
            fig.savefig(out, dpi=150)
            plt.close(fig)
            print(f"saved {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="模型目录")
    ap.add_argument("--output", required=True, help="输出目录")
    ap.add_argument("--scale-mm", type=float, default=SCENE_TO_MM,
                    help="缩放世界单位→mm 换算系数（volume_grid 模式由 scanner 给出 = max(sVoxel)/2，忽略此值）")
    ap.add_argument("--gamma", type=float, default=0.3,
                    help="PowerNorm 的 gamma（<1 提亮低值，默认 0.3）")
    ap.add_argument("--pct", type=float, default=99.9,
                    help="色标上限取非零值分位数（默认 p99.9，避免少数大值压黑全图）")
    ap.add_argument("--bins", type=int, default=64,
                    help="空间直方图网格边长（默认 64，切片占用 ~34%%；256 只有 ~3%% 稀疏散点）")
    ap.add_argument("--dataset", default=None,
                    help="数据集目录(含 meta_data.json)。指定后按体积网格(缩放空间)分箱，"
                         "切片位置与 plot_roi 的 ROI 图(体素 128)一致；mm 刻度取 scanner 换算")
    args = ap.parse_args()

    if plt is None:
        raise SystemExit("matplotlib 不可用")

    os.makedirs(args.output, exist_ok=True)
    events = load_events(args.model)
    if events:
        plot_summary(events, os.path.join(args.output, "grad_summary_vs_step.png"))
        plot_zprofile(events, os.path.join(args.output, "grad_zprofile_evolution.png"),
                      scale_mm=args.scale_mm)
        plot_hist_evolution(events, os.path.join(args.output, "grad_hist_evolution.png"))
        print(f"events: {len(events)} (step {events[0]['step']}..{events[-1]['step']})")
    else:
        print("warning: grad_maps/ 下没有事件记录")

    eval_recs = load_eval_grads(args.model)
    if eval_recs:
        lo = span = None
        scale_mm = args.scale_mm
        volume_grid = args.dataset is not None
        if volume_grid:
            with open(os.path.join(args.dataset, "meta_data.json")) as fh:
                lo, span, scale_mm = scanner_grid(json.load(fh)["scanner"])
            print(f"volume grid: lo={lo} span={span} mm_per_unit={scale_mm:.1f}")
        eval_spatial_maps(eval_recs, os.path.join(args.output, "eval_grad_planes"),
                          voxel_bins=args.bins, gamma=args.gamma, pct=args.pct,
                          lo=lo, span=span, scale_mm=scale_mm, volume_grid=volume_grid)
        for step, kind, _, grads in eval_recs:
            print(f"  eval step {step} ({kind}): N={len(grads)} "
                  f"mean={grads.mean():.3e} p99={np.quantile(grads, 0.99):.3e}")
    else:
        print("warning: eval 目录下没有 densify_grads.npz")


if __name__ == "__main__":
    main()
