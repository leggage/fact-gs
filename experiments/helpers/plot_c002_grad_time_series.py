"""ldctc002 冷/暖启动 NG(x) 与每高斯梯度的时间序列（只画 yz 平面）。

行 = cold/warm；列 = 时间：init(step 0) → eval 5000 → ... → 30000。
eval 每 5000 步一次（没有 eval3000），densification 事件到 step 14900 为止，
所以各 eval 步 densify_grads.npz 的 accum_window 约从上次 densify 起算，
step 30000 的窗口 ≈ step 15000→30000。

数据来源：
- init 位置（step 0）：
  - cold = 数据集 init_r2gs.npy（R2 等价的 FDK foreground 均匀采样，50000 点）
  - warm = vol_fit_fdk1000/point_cloud/step_500/point_cloud.pickle（FDK prior 的
    volume-fit 高斯，50000 点）
- eval 步位置/梯度 = 模型 eval/step_XXXXXX/densify_grads.npz（xyz, grads=accum/denom）
- init 步没有 densify_grads（不渲染无梯度），所以每高斯梯度图从 step 5000 起

输出（--output-dir，默认 grad_compare 目录）：
- ng_time_series_yz.png          —— NG(x) yz 中间切片，7 列（含 init）
- grad_per_gaussian_time_series_yz.png —— G/NG yz 中间切片，6 列（5k..30k）
- stdout —— 每时间点的 z 分片高斯数表（直接检验"一开始边缘高斯是否就少"）

分箱与其余脚本一致：64³ 体积网格，裁剪体积外高斯 + 显式 range（bin 31 =
体积中心平面 = 体素 128），yz 切片行=y 列=z（同 plot_roi / grad_compare）。

用法:
    MPLCONFIGDIR=/tmp/spiral-gs-matplotlib python experiments/helpers/plot_c002_grad_time_series.py
"""

import argparse
import json
import os
import pickle
import sys

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import PowerNorm
except ImportError:  # pragma: no cover
    plt = None

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from plot_c002_grad_factors import (  # noqa: E402
    scanner_grid, grad_hist, annotate_axes, AXIS_LABEL, PLANES,
)

BASE = os.path.join(ROOT, "models/real/ldctc002/spiral/ntrain1000")
DEFAULT_COLD = os.path.join(BASE, "factgs_det_native_cold")
DEFAULT_WARM = os.path.join(BASE, "factgs_det_native_warm_fdk1000")
DEFAULT_DATASET = os.path.join(ROOT, "data/real/ldctc002/spiral/ntrain1000/r2gs")
DEFAULT_INIT_COLD = os.path.join(DEFAULT_DATASET, "init_r2gs.npy")
DEFAULT_INIT_WARM = os.path.join(BASE, "vol_fit_fdk1000", "point_cloud",
                                 "step_500", "point_cloud.pickle")
DEFAULT_OUTDIR = os.path.join(ROOT, "output/real/ldctc002/spiral/ntrain1000/grad_compare")

BINS = 64
COUNT_BINS = 256
EVAL_STEPS = [5000, 10000, 15000, 20000, 25000, 30000]


def load_init_positions(path):
    """读取 init 高斯位置 (N,3)：支持 init_*.npy（前 3 列）与 point_cloud.pickle。"""
    if path.endswith(".npy"):
        return np.load(path)[:, :3].astype(np.float64)
    with open(path, "rb") as fh:
        d = pickle.load(fh)
    xyz = d["xyz"]
    if hasattr(xyz, "detach"):
        xyz = xyz.detach().cpu().numpy()
    return np.asarray(xyz, dtype=np.float64)


def plot_series(panels, times, name, suptitle, lo, span, mm, out_dir,
                gamma=0.3, pct=99.9, ncols=None):
    """行 = cold/warm，列 = 时间，每面板 = yz 中间切片（行=y, 列=z）。"""
    keep, row_a, col_a = 0, 1, 2  # yz 平面：保留轴 0(x)，行=y，列=z
    ncols = ncols or len(times)
    slice_bin = (BINS - 1) // 2

    nz = np.concatenate([q[q > 0] for _, row in panels.items() for q in row.values()])
    vmax = float(np.quantile(nz, pct / 100.0)) if nz.size else 1.0
    norm = PowerNorm(gamma=gamma, vmin=0, vmax=vmax)

    fig, axes = plt.subplots(2, ncols, figsize=(2.8 * ncols, 8.2), squeeze=False)
    for ri, (label, row) in enumerate(panels.items()):
        for ci in range(ncols):
            ax = axes[ri][ci]
            t = times[ci] if ci < len(times) else None
            q = row.get(t)
            if q is None:
                ax.axis("off")
                continue
            s = np.take(q, slice_bin, axis=keep)
            ext = annotate_axes(ax, row_a, col_a, lo, span, mm)
            im = ax.imshow(s, origin="lower", cmap="inferno", norm=norm, extent=ext,
                           aspect=(ext[1] - ext[0]) / (ext[3] - ext[2]))
            if ci == 0:
                ax.set_ylabel(f"{label}\n" + AXIS_LABEL[row_a][0].replace("→", "↑"))
            if ri == 0:
                ax.set_title(t)
    for ci in range(ncols):
        fig.colorbar(im, ax=axes[:, ci], fraction=0.046)
    fig.suptitle(suptitle, y=0.995)
    out = os.path.join(out_dir, name)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")



def plot_all_projection_series(xyz_by_time, times, lo, span, mm, out_dir,
                               bins=COUNT_BINS, pct=99.9, weights_by_time=None,
                               name="gaussian_count", label="gaussian count",
                               cmap="viridis", gamma=None):
    """按平面输出任意模型/节点的全量（含框外）计数或加权投影。"""
    models = tuple(xyz_by_time)
    xyz_all = np.concatenate([xyz for row in xyz_by_time.values()
                              for xyz in row.values()])
    bounds_lo = np.minimum(lo, xyz_all.min(axis=0))
    bounds_hi = np.maximum(lo + span, xyz_all.max(axis=0))
    pad = np.maximum((bounds_hi - bounds_lo) * 0.01, 1e-6)
    bounds_lo -= pad
    bounds_hi += pad

    for plane, (_, row_a, col_a) in PLANES.items():
        hist = {model: {} for model in models}
        for model in models:
            for step in times:
                xyz = xyz_by_time[model][step]
                if weights_by_time is not None and step not in weights_by_time[model]:
                    hist[model][step] = None
                    continue
                weights = None if weights_by_time is None else weights_by_time[model][step]
                h, _, _ = np.histogram2d(
                    xyz[:, row_a], xyz[:, col_a], bins=bins, weights=weights,
                    range=((bounds_lo[row_a], bounds_hi[row_a]),
                           (bounds_lo[col_a], bounds_hi[col_a])))
                expected = len(xyz) if weights is None else weights.sum()
                assert np.isclose(h.sum(), expected, rtol=1e-5)
                hist[model][step] = h
        nz = np.concatenate([h[h > 0] for row in hist.values()
                             for h in row.values() if h is not None])
        vmax = float(np.quantile(nz, pct / 100.0)) if nz.size else 1.0
        norm = PowerNorm(gamma=gamma, vmin=0, vmax=vmax) if gamma else None
        fig, axes = plt.subplots(len(models), len(times),
                                 figsize=(4 * len(times), 4 * len(models)), squeeze=False)
        ext = [bounds_lo[col_a] * mm, bounds_hi[col_a] * mm,
               bounds_lo[row_a] * mm, bounds_hi[row_a] * mm]
        im = None
        for ri, model in enumerate(models):
            for ci, step in enumerate(times):
                ax = axes[ri][ci]
                h = hist[model][step]
                if h is None:
                    ax.set_facecolor("black")
                    ax.text(0.5, 0.5, "N/A\n(no gradients at init)", color="white",
                            ha="center", va="center", transform=ax.transAxes)
                    ax.set_xticks([]); ax.set_yticks([])
                else:
                    im = ax.imshow(h, origin="lower", cmap=cmap, norm=norm,
                                   vmin=None if norm else 0, vmax=None if norm else vmax,
                                   extent=ext, aspect="equal")
                    ax.add_patch(plt.Rectangle(
                        (lo[col_a] * mm, lo[row_a] * mm), span[col_a] * mm,
                        span[row_a] * mm, fill=False, color="white", ls="--", lw=0.8))
                if ri == 0:
                    ax.set_title("init" if step == 0 else f"{step // 1000}k")
                if ci == 0:
                    ax.set_ylabel(f"{model}\n" + AXIS_LABEL[row_a][0].replace("→", "↑"))
                if ri == len(models) - 1:
                    ax.set_xlabel(AXIS_LABEL[col_a][0])
        fig.colorbar(im, ax=axes, fraction=0.012, pad=0.01, label=label)
        scale = f"PowerNorm gamma={gamma}" if gamma else "linear"
        fig.suptitle(f"{label}: all-gaussian {plane} projection "
                     f"({bins}x{bins}; {scale}; shared p{pct}; dashed = reconstruction box)")
        out = os.path.join(out_dir, f"{name}_time_series_{plane}.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"saved {out}")


def plot_selected_gradient_panels(xyz_by_time, grads_by_time, times, lo, span,
                                  mm, out_dir, bins=COUNT_BINS, pct=99.9,
                                  threshold=None, gamma=0.15):
    """四行（cold/warm × 时间）三列（xy/xz/yz）的梯度密度图。"""
    xyz_all = np.concatenate([xyz_by_time[m][t] for m in ("cold", "warm")
                              for t in times])
    bounds_lo = np.minimum(lo, xyz_all.min(axis=0))
    bounds_hi = np.maximum(lo + span, xyz_all.max(axis=0))
    pad = np.maximum((bounds_hi - bounds_lo) * 0.01, 1e-6)
    bounds_lo -= pad
    bounds_hi += pad

    panels = {}
    for model in ("cold", "warm"):
        for step in times:
            xyz, grads = xyz_by_time[model][step], grads_by_time[model][step]
            keep = np.ones(len(grads), dtype=bool) if threshold is None else grads >= threshold
            panels[model, step] = {}
            for plane, (_, row_a, col_a) in PLANES.items():
                h, _, _ = np.histogram2d(
                    xyz[keep, row_a], xyz[keep, col_a], bins=bins,
                    weights=grads[keep],
                    range=((bounds_lo[row_a], bounds_hi[row_a]),
                           (bounds_lo[col_a], bounds_hi[col_a])))
                assert np.isclose(h.sum(), grads[keep].sum(), rtol=1e-5)
                panels[model, step][plane] = h
            print(f"{model} {step}: kept {keep.sum()}/{len(keep)}")

    nz = np.concatenate([h[h > 0] for row in panels.values() for h in row.values()])
    vmax = float(np.quantile(nz, pct / 100.0)) if nz.size else 1.0
    norm = PowerNorm(gamma=gamma, vmin=0, vmax=vmax)
    rows = [(model, step) for step in times for model in ("cold", "warm")]
    fig, axes = plt.subplots(len(rows), 3, figsize=(16, 4.2 * len(rows)), squeeze=False)
    for ri, (model, step) in enumerate(rows):
        for ci, (plane, (_, row_a, col_a)) in enumerate(PLANES.items()):
            ax = axes[ri][ci]
            ext = [bounds_lo[col_a] * mm, bounds_hi[col_a] * mm,
                   bounds_lo[row_a] * mm, bounds_hi[row_a] * mm]
            im = ax.imshow(panels[model, step][plane], origin="lower", cmap="inferno",
                           norm=norm, extent=ext, aspect="equal")
            ax.add_patch(plt.Rectangle(
                (lo[col_a] * mm, lo[row_a] * mm), span[col_a] * mm,
                span[row_a] * mm, fill=False, color="white", ls="--", lw=0.8))
            ax.set_title(f"{plane} — {model} {step // 1000}k")
            ax.set_xlabel(AXIS_LABEL[col_a][0])
            ax.set_ylabel(AXIS_LABEL[row_a][0].replace("→", "↑"))
            fig.colorbar(im, ax=ax, fraction=0.046)
    suffix = "all" if threshold is None else f"above_{threshold:g}"
    selection = "all gaussians" if threshold is None else f"grads >= {threshold:g} only"
    fig.suptitle(f"Gradient density at 5k and 30k ({selection}; {bins}x{bins}; "
                 f"shared p{pct}; PowerNorm gamma={gamma})", y=0.998)
    out = os.path.join(out_dir, f"gradient_density_5k_30k_{suffix}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def z_table(ng_by_time, lo, span, bins, zs=8):
    """每时间点 z 分片高斯数（冷/暖并排），自下而上。"""
    zsp = bins // zs
    print("\nz 分片高斯数随时间（cold / warm）：")
    header = f"{'z (mm)':>14}"
    for t in ng_by_time["cold"]:
        header += f" | {str(t):>13}"
    print(header)
    for j in range(zs):
        sl = slice(j * zsp, (j + 1) * zsp)
        z0 = (lo[2] + j * zsp / (bins - 1) * span[2]) * 9.5
        z1 = (lo[2] + ((j + 1) * zsp - 1) / (bins - 1) * span[2]) * 9.5
        line = f"{z0:7.1f}..{z1:6.1f}"
        for t in ng_by_time["cold"]:
            c = int(ng_by_time["cold"][t][:, :, sl].sum())
            w = int(ng_by_time["warm"][t][:, :, sl].sum())
            line += f" | {c:>6}/{w:<6}"
        print(line)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cold-dir", default=DEFAULT_COLD)
    ap.add_argument("--warm-dir", default=DEFAULT_WARM)
    ap.add_argument("--init-cold", default=DEFAULT_INIT_COLD)
    ap.add_argument("--init-warm", default=DEFAULT_INIT_WARM)
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--output-dir", default=DEFAULT_OUTDIR)
    ap.add_argument("--bins", type=int, default=BINS)
    ap.add_argument("--gamma", type=float, default=0.3)
    ap.add_argument("--pct", type=float, default=99.9)
    ap.add_argument("--densify-threshold", type=float, default=5e-5)
    args = ap.parse_args()

    if plt is None:
        raise SystemExit("matplotlib 不可用")
    bins = args.bins
    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.dataset, "meta_data.json")) as fh:
        scanner = json.load(fh)["scanner"]
    lo, span, mm = scanner_grid(scanner)

    ng_by_time = {"cold": {}, "warm": {}}
    g_by_time = {"cold": {}, "warm": {}}
    xyz_by_time = {"cold": {}, "warm": {}}
    grads_by_time = {"cold": {}, "warm": {}}

    # ---- init（step 0）----
    xyz_c0 = load_init_positions(args.init_cold)
    xyz_w0 = load_init_positions(args.init_warm)
    ng_by_time["cold"]["init"] = grad_hist(xyz_c0, None, lo, span, bins, False)
    ng_by_time["warm"]["init"] = grad_hist(xyz_w0, None, lo, span, bins, False)
    print(f"init cold: N={len(xyz_c0)} (init_r2gs.npy)")
    print(f"init warm: N={len(xyz_w0)} (vol_fit_fdk1000 step_500)")

    # ---- eval 步 ----
    for mdir, label in ((args.cold_dir, "cold"), (args.warm_dir, "warm")):
        for step in EVAL_STEPS:
            z = np.load(os.path.join(mdir, "eval", f"step_{step:06d}",
                                     "densify_grads.npz"))
            xyz_by_time[label][step] = z["xyz"]
            grads_by_time[label][step] = z["grads"]
            ng_by_time[label][step] = grad_hist(z["xyz"], None, lo, span, bins, False)
            g_by_time[label][step] = grad_hist(z["xyz"], z["grads"], lo, span, bins, True)
            print(f"{label} step {step}: N={len(z['xyz'])} kind={z['kind'].item()}")

    plot_all_projection_series(xyz_by_time, EVAL_STEPS, lo, span, mm,
                               args.output_dir)
    plot_all_projection_series(
        xyz_by_time, EVAL_STEPS, lo, span, mm, args.output_dir,
        weights_by_time=grads_by_time, name="gradient_density",
        label="sum(accum/denom)", cmap="inferno", gamma=args.gamma)
    selected_steps = [5000, 30000]
    plot_selected_gradient_panels(
        xyz_by_time, grads_by_time, selected_steps, lo, span, mm, args.output_dir,
        pct=args.pct)
    plot_selected_gradient_panels(
        xyz_by_time, grads_by_time, selected_steps, lo, span, mm, args.output_dir,
        pct=args.pct, threshold=args.densify_threshold)

    # ---- 图 1: NG 时间序列（含 init，7 列）----
    times_ng = ["init"] + EVAL_STEPS
    plot_series(ng_by_time, times_ng, "ng_time_series_yz.png",
                "NG(x) yz mid-slice over time (rows: cold/warm, cols: init -> 30k)\n"
                f"PowerNorm gamma={args.gamma}, vmax=p{args.pct} (shared)",
                lo, span, mm, args.output_dir, gamma=args.gamma, pct=args.pct)

    # ---- 图 2: 每高斯梯度 G/NG 时间序列（init 无梯度记录，6 列）----
    eps = 1e-12
    grad_ratio = {
        label: {t: g_by_time[label][t] / (ng_by_time[label][t] + eps)
                for t in EVAL_STEPS}
        for label in ("cold", "warm")
    }
    plot_series(grad_ratio, EVAL_STEPS, "grad_per_gaussian_time_series_yz.png",
                "G/NG yz mid-slice over time (rows: cold/warm, cols: 5k -> 30k)\n"
                "mean position-gradient norm per gaussian; "
                f"PowerNorm gamma={args.gamma}, vmax=p{args.pct} (shared)",
                lo, span, mm, args.output_dir, gamma=args.gamma, pct=args.pct)

    z_table(ng_by_time, lo, span, bins)


if __name__ == "__main__":
    main()
