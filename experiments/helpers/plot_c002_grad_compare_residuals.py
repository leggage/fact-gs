"""ldctc002 冷/暖启动 —— 梯度图对比 + 残差图（物理轴标注）。

输入：两个模型目录（train_recon.py 开启 eval.record_densify_grads 后生成）
- eval/step_XXXXXX/densify_grads.npz —— 每 eval 步的完整 (xyz, grads)，
  xyz 为缩放世界坐标（= mm × scene_scale，scene_scale=2/max(sVoxel)，
  1 缩放单位 = max(sVoxel)/2 mm），列序 (x, y, z)，轴向 z 为第 3 列
- eval/step_XXXXXX/vol_pred.tiff + 模型目录 vol_gt.tiff —— 重建体积与 GT

输出（--output-dir）：
- grad_cold_vs_warm_{xy,xz,yz}_step*.png  —— 冷 | 暖 | 暖-冷 梯度密度切片对比（三平面各一张）
- residual_cold_warm_{xy,xz,yz}_step*.png —— GT | |冷−GT| | |暖−GT| 残差图（三平面各一张）

坐标标注：每个平面图都标出横轴/纵轴的真实物理方向（x→ / y→ / z→）与 mm 刻度；
梯度图在"体积网格"（scanner_grid：缩放空间 offOrigin±sVoxel/2）上分 bin，
切片取体积中心 bin = plot_roi 的 ROI 图 / 残差图的体积体素 128 位置（同一物理平面）。
体积轴约定（单高斯标定验证）: 0=x, 1=y, 2=z —— grad yz ↔ ROI ax0, grad xz ↔ ROI ax1,
grad xy ↔ ROI ax2。面板行列与 plot_roi 的 get_slice 一致：行=切片后剩余的第一个
体积轴、列=第二个（如 yz 平面：行=y, 列=z），避免与 ROI 图/残差图出现 90° 转置。

用法:
    MPLCONFIGDIR=/tmp/spiral-gs-matplotlib python experiments/helpers/plot_c002_grad_compare_residuals.py
"""

import argparse
import json
import os
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
BASE = os.path.join(ROOT, "models/real/ldctc002/spiral/ntrain1000")
DEFAULT_COLD = os.path.join(BASE, "factgs_det_native_cold")
DEFAULT_WARM = os.path.join(BASE, "factgs_det_native_warm_fdk1000")
DEFAULT_DATASET = os.path.join(ROOT, "data/real/ldctc002/spiral/ntrain1000/r2gs")

# 体积体素索引（ROI 图 plot_roi 的默认切片）
SLICE_VOXEL = 128

BINS = 64  # 空间直方图网格边长（默认 64，切片占用 ~34%；256 只有 ~3% 稀疏散点）

# 平面 → (切片保留的世界轴, 行对应世界轴, 列对应世界轴)  世界轴: 0=x, 1=y, 2=z
# 行=切片后剩余的第一个体积轴、列=第二个（与 plot_roi get_slice / 残差图同序，
# 不转置：yz 行=y 列=z，xz 行=x 列=z，xy 行=x 列=y）
PLANES = {"xy": (2, 0, 1), "xz": (1, 0, 2), "yz": (0, 1, 2)}
AXIS_LABEL = {0: ("x (mm) →", "x"), 1: ("y (mm) →", "y"), 2: ("z (mm) →", "z")}


def load_scanner(dataset):
    with open(os.path.join(dataset, "meta_data.json")) as fh:
        return json.load(fh)["scanner"]


def scanner_grid(scanner):
    """体积网格（训练用缩放空间，scene_scale=2/max(sVoxel)）+ mm 换算系数。

    SceneRecon（dataset_readers.py）加载时把 scanner 几何整体乘 scene_scale；
    voxelize kernel 把世界坐标映射到体素：voxel = (world - off + sVoxel/2)/dVoxel，
    即体积世界范围 = [off - sVoxel/2, off + sVoxel/2]（off 是体积中心，不是角点）。

    返回 lo/span（缩放世界坐标，列序 x,y,z）与 mm 换算系数
    （scene_scale = 2/max(sVoxel) 直接作用于 mm 值 → 1 缩放单位 = max(sVoxel)/2 mm）。
    """
    sVoxel = np.asarray(scanner["sVoxel"], dtype=float)
    offOrigin = np.asarray(scanner["offOrigin"], dtype=float)
    ss = 2.0 / sVoxel.max()
    sv = sVoxel * ss
    off = offOrigin * ss
    lo = off - sv / 2.0
    span = sv
    mm_per_unit = sVoxel.max() / 2.0
    return lo, span, mm_per_unit


def grad_map(xyz, grads, bins, lo, span):
    """高斯位置(x,y,z)按梯度加权投到 bins³ 直方图网格 → 密度图。

    必须用体积网格（scanner_grid 返回的 lo/span）分箱：所有 run 共用同一网格，
    且 bins//2 切片 = 体积体素 SLICE_VOXEL（bins 整除 256 时恒成立），
    与 ROI 图 / 残差图的切片位置完全一致。
    体积外的高斯（eval 步实测冷 ~15% / 暖 ~9% 漂移出体积）先按 idx 裁剪丢弃，
    再用显式 range=[0,bins-1] 分箱——否则 histogramdd 会按数据范围扩大网格，
    bin 中心偏移、且体积外点被挤进图内。
    """
    idx = np.floor((xyz - lo) / (span + 1e-12) * (bins - 1)).astype(int)
    inside = ((idx >= 0) & (idx < bins)).all(axis=1)
    idx, grads = idx[inside], grads[inside]
    H, _ = np.histogramdd(idx, bins=[bins] * 3, range=[(0, bins - 1)] * 3,
                          weights=grads)
    return H


def annotate_axes(ax, row_axis, col_axis, lo, span, scale_mm):
    """标注横轴(col)与纵轴(row)的物理方向与 mm 刻度。"""
    extent = [lo[col_axis] * scale_mm, (lo[col_axis] + span[col_axis]) * scale_mm,
              lo[row_axis] * scale_mm, (lo[row_axis] + span[row_axis]) * scale_mm]
    ax.set_xlabel(AXIS_LABEL[col_axis][0])
    ax.set_ylabel(AXIS_LABEL[row_axis][0].replace("→", "↑"))
    return extent


def load_volume(path):
    from plot_roi import load_volume as _lv
    return _lv(path)


def plot_grad_compare(cold_rec, warm_rec, scanner, out_dir, step, gamma=0.3, pct=99.9, bins=BINS):
    """冷 vs 暖：xy/xz/yz 三平面梯度密度切片（冷 | 暖 | 暖−冷）。

    共享"体积网格"（scanner_grid：lo=offOrigin−sVoxel/2, span=sVoxel，训练缩放空间）：
    - 冷/暖 bin 到同一网格 → 每张图三个面板是同一物理位置；
    - 切片取 (bins−1)//2 号 bin，其中心 = 体积中心平面，与 plot_roi 的 ROI 图 /
      残差图的体积体素 128 切片对应（bin 中心与体素中心差 <1 bin，~1mm）；
    - 体积外的高斯（实测 ~16% 漂移出体积）被 histogramdd 丢弃，只画体积内梯度；
    - 面板行列 = 切片后剩余体积轴的顺序（PLANES 第二/三列），不转置，
      与 plot_roi 的 get_slice 及本文件的残差图完全同序。

    梯度权重跨多个数量级，线性色标会全黑：密度图用 PowerNorm(gamma<1)，
    色标上限取两图非零值的 pct 分位数（共享，保证冷/暖可比）；
    差图用对称 PowerNorm（0 保持白色），上限取 |diff| 的 pct 分位数。
    """
    xyz_c, g_c = cold_rec
    xyz_w, g_w = warm_rec
    lo, span, mm_per_unit = scanner_grid(scanner)
    H_c = grad_map(xyz_c, g_c, bins=bins, lo=lo, span=span)
    H_w = grad_map(xyz_w, g_w, bins=bins, lo=lo, span=span)
    slice_bin = (bins - 1) // 2  # 中心 = 体积中心平面 = ROI 体素 128 所在平面
    for plane, (keep, row_a, col_a) in PLANES.items():
        s_c = np.take(H_c, slice_bin, axis=keep)
        s_w = np.take(H_w, slice_bin, axis=keep)
        diff = s_w - s_c
        nz = np.concatenate([s_c[s_c > 0], s_w[s_w > 0]])
        dmax = float(np.quantile(nz, pct / 100.0)) if nz.size else 1.0
        lim = float(np.quantile(np.abs(diff[diff != 0]), pct / 100.0)) if np.any(diff != 0) else 1.0
        norm_pos = PowerNorm(gamma=gamma, vmin=0, vmax=dmax)
        norm_diff = PowerNorm(gamma=gamma, vmin=-lim, vmax=lim)
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.6), gridspec_kw={"wspace": 0.32})
        for ax, data, cmap, norm, title in (
            (axes[0], s_c, "inferno", norm_pos, "cold"),
            (axes[1], s_w, "inferno", norm_pos, "warm"),
            (axes[2], diff, "RdBu_r", norm_diff, "warm − cold"),
        ):
            ext = annotate_axes(ax, row_a, col_a, lo, span, mm_per_unit)
            im = ax.imshow(data, origin="lower", cmap=cmap, norm=norm, extent=ext,
                           aspect=(ext[1] - ext[0]) / (ext[3] - ext[2]))
            ax.set_title(title)
            fig.colorbar(im, ax=ax, fraction=0.046, label="grad weight")
        fig.suptitle(f"densify decision gradient density, {plane} plane @ volume voxel {SLICE_VOXEL} (step {step})\n"
                     f"PowerNorm γ={gamma}, vmax=p{pct} (shared)", y=1.02)
        out = os.path.join(out_dir, f"grad_cold_vs_warm_{plane}_step{step:06d}.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"saved {out}")


def plot_residuals(cold_pred, warm_pred, gt, scanner, out_dir, step, gamma=0.3, pct=99.9):
    """GT | |冷−GT| | |暖−GT| 残差图（三平面）。

    体积轴约定（单高斯标定验证）: 0=x, 1=y, 2=z。对体积轴 a 的中间切片(体素128)：
    a=0 → yz 平面(行=y, 列=z)；a=1 → xz 平面(行=x, 列=z)；a=2 → xy 平面(行=x, 列=y)。
    mm 范围 = 体积世界范围 [offOrigin−sVoxel/2, offOrigin+sVoxel/2]
    （scanner 的 sVoxel/offOrigin 本身就是 mm，不需要再换算；
    offOrigin 是体积中心不是角点；旧版把 off 当角点且乘了多余的 20，mm 标注全错）。
    """
    sv = np.asarray(scanner["sVoxel"], dtype=float)
    off = np.asarray(scanner["offOrigin"], dtype=float)
    lo_w = off - sv / 2.0
    hi_w = off + sv / 2.0
    # (切片体积轴, 行世界轴, 列世界轴, 平面名) —— 体积轴=世界轴（标定验证）
    views = [(0, 1, 2, "yz"), (1, 0, 2, "xz"), (2, 0, 1, "xy")]
    for vol_axis, row_w, col_w, plane in views:
        s_gt = np.take(gt, gt.shape[vol_axis] // 2, axis=vol_axis)
        s_c = np.take(cold_pred, cold_pred.shape[vol_axis] // 2, axis=vol_axis)
        s_w = np.take(warm_pred, warm_pred.shape[vol_axis] // 2, axis=vol_axis)
        r_c = np.abs(s_c - s_gt)
        r_w = np.abs(s_w - s_gt)
        nz = np.concatenate([r_c[r_c > 0], r_w[r_w > 0]])
        rmax = float(np.quantile(nz, pct / 100.0)) if nz.size else 1.0
        norm_res = PowerNorm(gamma=gamma, vmin=0, vmax=rmax)
        ext = [lo_w[col_w], hi_w[col_w], lo_w[row_w], hi_w[row_w]]
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.6), gridspec_kw={"wspace": 0.32})
        for ax, data, cmap, norm, title in (
            (axes[0], s_gt, "gray", PowerNorm(gamma=1.0, vmin=0, vmax=1), "GT"),
            (axes[1], r_c, "hot", norm_res, "|cold − GT|"),
            (axes[2], r_w, "hot", norm_res, "|warm − GT|"),
        ):
            im = ax.imshow(data, origin="lower", cmap=cmap, norm=norm, extent=ext,
                           aspect=(ext[1] - ext[0]) / (ext[3] - ext[2]))
            ax.set_xlabel(AXIS_LABEL[col_w][0])
            ax.set_ylabel(AXIS_LABEL[row_w][0].replace("→", "↑"))
            ax.set_title(title)
            fig.colorbar(im, ax=ax, fraction=0.046)
        fig.suptitle(f"residual maps, {plane} mid-plane @ step {step}\n"
                     f"PowerNorm γ={gamma}, vmax=p{pct} (shared)", y=1.02)
        out = os.path.join(out_dir, f"residual_cold_warm_{plane}_step{step:06d}.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"saved {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cold-dir", default=DEFAULT_COLD)
    ap.add_argument("--warm-dir", default=DEFAULT_WARM)
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--step", default="step_030000")
    ap.add_argument("--output-dir",
                    default=os.path.join(ROOT, "output/real/ldctc002/spiral/ntrain1000/grad_compare"))
    ap.add_argument("--gamma", type=float, default=0.3,
                    help="PowerNorm 的 gamma（<1 提亮低值，默认 0.3）")
    ap.add_argument("--pct", type=float, default=99.9,
                    help="色标上限取非零值分位数（默认 p99.9，避免少数大值压黑全图）")
    ap.add_argument("--bins", type=int, default=64,
                    help="空间直方图网格边长（默认 64，切片占用 ~34%%；256 只有 ~3%% 稀疏散点）")
    args = ap.parse_args()

    if plt is None:
        raise SystemExit("matplotlib 不可用")
    os.makedirs(args.output_dir, exist_ok=True)
    scanner = load_scanner(args.dataset)

    evals = {}
    for name, mdir in (("cold", args.cold_dir), ("warm", args.warm_dir)):
        rec = np.load(os.path.join(mdir, "eval", args.step, "densify_grads.npz"))
        evals[name] = (rec["xyz"], rec["grads"])
        print(f"{name}: {rec['kind'].item()} N={len(rec['grads'])}")

    plot_grad_compare(evals["cold"], evals["warm"], scanner, args.output_dir,
                      int(args.step.split("_")[1]), gamma=args.gamma, pct=args.pct,
                      bins=args.bins)

    gt = load_volume(os.path.join(args.cold_dir, "vol_gt.tiff"))
    cold_pred = load_volume(os.path.join(args.cold_dir, "eval", args.step, "vol_pred.tiff"))
    warm_pred = load_volume(os.path.join(args.warm_dir, "eval", args.step, "vol_pred.tiff"))
    plot_residuals(cold_pred, warm_pred, gt, scanner, args.output_dir,
                   int(args.step.split("_")[1]), gamma=args.gamma, pct=args.pct)


if __name__ == "__main__":
    main()
