"""ldctc002 冷/暖启动梯度差异的因子分解：Nv(x) / NG(x) / 三种归一化梯度图。

背景：梯度图每个 bin 的值 = Σ_{高斯∈bin} (xyz_gradient_accum/denom)，
同时受"该位置的有效投影数 Nv(x)"和"该位置的高斯数 NG(x)"双重影响
（没有高斯载体，损失梯度无法传导，图上就是空白）。本脚本把两个因子拆开：

  1. Nv(x)           每体素的几何有效投影数（圆锥束穿过体素中心的训练视角数，冷暖共享）
  2. NG(x)           每 bin 的高斯数量（occupancy）
  3. Σ accum/denom   视角平均位置梯度范数的空间密度（= 现有梯度图 grad_cold_vs_warm_*）
  4. (3)/NG(x)       每个高斯的平均位置梯度范数（消除高斯数量差异）
  5. (3)/(Nv(x)·NG(x))  再除以有效投影数（近似同时消除两个因子；denom 是逐高斯的
                     可见视角数，这里用逐体素的几何覆盖 Nv 近似，见文末讨论）

输出（--output-dir，默认 grad_compare 目录）：
- factors_nviews.png / factors_gaussian_count.png / factors_grad_density.png /
  factors_grad_per_gaussian.png / factors_grad_normalized.png
  —— NG 图为包含框外高斯的 xy/xz/yz 全量投影，其余图为三平面中间切片
  （bin 31 = 体积体素 128，与 grad_compare 的梯度图/残差图同一平面位置、
  同一行列轴序：行=第一剩余轴、列=第二剩余轴）
- grad_factors.npz —— Nv / NG / G 与比值数组、lo/span
- stdout —— 按 z 分 8 片的数值剖面（Nv、NG、每高斯平均梯度），检验
  "冷启动 z 上下边缘梯度缺失"假说

实现注记：
- 几何与训练完全一致：meta_data scanner 值 × scene_scale=2/max(sVoxel)，
  每视角 c2w = angle2pose(dso, angle, z_shift·scene_scale)（dataset_readers.py 同款），
  可见性判据 = 体素中心投影落在探测器矩形内（|x|/z ≤ tan(FovX/2), |y|/z ≤ tan(FovY/2)），
  与 rasterizer 的 visibility_filter(radii>0) 对点中心的判据一致；
- coord_left 的镜像不影响覆盖计数（探测器矩形对称）；
- 比值图对 NG=0 / Nv=0 的 bin 置 0。

用法:
    MPLCONFIGDIR=/tmp/spiral-gs-matplotlib python experiments/helpers/plot_c002_grad_factors.py
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
DEFAULT_OUTDIR = os.path.join(ROOT, "output/real/ldctc002/spiral/ntrain1000/grad_compare")

BINS = 64  # 与 grad_compare 的空间直方图网格一致
COUNT_BINS = 256
SLICE_VOXEL = 128  # 中间切片对应的体积体素（bins=64 整除 256 时 bin 31 中心 = 体素 128 平面）

# 平面 → (切片保留的世界轴, 行对应世界轴, 列对应世界轴)  世界轴: 0=x, 1=y, 2=z
# 行=第一剩余轴、列=第二剩余轴（与 plot_roi get_slice / grad_compare 同序，不转置）
PLANES = {"xy": (2, 0, 1), "xz": (1, 0, 2), "yz": (0, 1, 2)}
AXIS_LABEL = {0: ("x (mm) →", "x"), 1: ("y (mm) →", "y"), 2: ("z (mm) →", "z")}


def angle2pose(DSO, angle, z_shift=0.0):
    """与 fact_gs/r2_gaussian/dataset/dataset_readers.py 完全相同的 c2w 构造。"""
    phi1 = -np.pi / 2
    R1 = np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(phi1), -np.sin(phi1)],
        [0.0, np.sin(phi1), np.cos(phi1)],
    ])
    phi2 = np.pi / 2
    R2 = np.array([
        [np.cos(phi2), -np.sin(phi2), 0.0],
        [np.sin(phi2), np.cos(phi2), 0.0],
        [0.0, 0.0, 1.0],
    ])
    R3 = np.array([
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle), np.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ])
    rot = np.dot(np.dot(R3, R2), R1)
    trans = np.array([DSO * np.cos(angle), DSO * np.sin(angle), z_shift])
    transform = np.eye(4)
    transform[:3, :3] = rot
    transform[:3, 3] = trans
    return transform


def scanner_grid(scanner):
    """体积网格（缩放空间）lo/span 与 mm 换算（1 缩放单位 = max(sVoxel)/2 mm）。"""
    sVoxel = np.asarray(scanner["sVoxel"], dtype=float)
    offOrigin = np.asarray(scanner["offOrigin"], dtype=float)
    ss = 2.0 / sVoxel.max()
    sv = sVoxel * ss
    lo = offOrigin * ss - sv / 2.0
    return lo, sv, sVoxel.max() / 2.0


def bin_centers(lo, span, bins):
    """histogramdd 风格网格的 bin 中心：idx = floor((x-lo)/span*(bins-1))。"""
    ax = [lo[a] + (np.arange(bins, dtype=np.float64) + 0.5) * span[a] / (bins - 1)
          for a in range(3)]
    gx, gy, gz = np.meshgrid(ax[0], ax[1], ax[2], indexing="ij")
    return np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)  # (bins³, 3)


def compute_nviews(centers, scanner, meta_data, bins, progress=True):
    """Nv(x)：圆锥束穿过体素中心的训练视角数（冷暖共享）。

    与 readCTameras 一致：scanner 几何 × scene_scale，逐视角
    c2w = angle2pose(dso, angle, z_shift·scene_scale)；
    可见性 = 体素中心投影落在探测器矩形内（z_cam>0 且 |x|/z≤tan(FovX/2)、
    |y|/z≤tan(FovY/2)）。探测器矩形对 coord_left 镜像对称，无需处理。
    """
    ss = 2.0 / max(scanner["sVoxel"])
    dso = float(scanner["DSO"]) * ss
    dsd = float(scanner["DSD"]) * ss
    sdet = np.asarray(scanner["sDetector"], dtype=float) * ss
    tan_x = (sdet[1] / 2.0) / dsd
    tan_y = (sdet[0] / 2.0) / dsd

    nv = np.zeros(len(centers), dtype=np.int32)
    n = len(meta_data["proj_train"])
    for i, frame in enumerate(meta_data["proj_train"]):
        if progress and (i % 100 == 0):
            print(f"  Nv: view {i}/{n}", flush=True)
        c2w = angle2pose(dso, frame["angle"], float(frame.get("z_shift", 0.0)) * ss)
        R, t = c2w[:3, :3], c2w[:3, 3]
        rel = centers - t
        x = rel @ R[:, 0]
        y = rel @ R[:, 1]
        z = rel @ R[:, 2]
        vis = (z > 0) & (np.abs(x) <= tan_x * z) & (np.abs(y) <= tan_y * z)
        nv += vis.astype(np.int32)
    if progress:
        print(f"  Nv: view {n}/{n}", flush=True)
    return nv.reshape(bins, bins, bins)


def grad_hist(xyz, grads, lo, span, bins, weighted):
    """体积网格分箱：裁剪体积外高斯 + 显式 range=[0,bins-1]（与 grad_map 同）。

    否则 histogramdd 按数据范围扩网格（eval 步 ~9-15% 高斯在体积外），
    bin 31 不再是体积中心平面。
    """
    idx = np.floor((xyz - lo) / (span + 1e-12) * (bins - 1)).astype(int)
    inside = ((idx >= 0) & (idx < bins)).all(axis=1)
    idx = idx[inside]
    w = grads[inside] if weighted else None
    H, _ = np.histogramdd(idx, bins=[bins] * 3, range=[(0, bins - 1)] * 3,
                          weights=w)
    return H


def annotate_axes(ax, row_axis, col_axis, lo, span, scale_mm):
    extent = [lo[col_axis] * scale_mm, (lo[col_axis] + span[col_axis]) * scale_mm,
              lo[row_axis] * scale_mm, (lo[row_axis] + span[row_axis]) * scale_mm]
    ax.set_xlabel(AXIS_LABEL[col_axis][0])
    ax.set_ylabel(AXIS_LABEL[row_axis][0].replace("→", "↑"))
    return extent


def plot_quantity(q_cold, q_warm, name, suptitle, lo, span, mm, out_dir,
                  cmap="inferno", gamma=0.3, pct=99.9, shared_only=False):
    """每个量一张图：列 = xy/xz/yz 中间切片，行 = cold/warm（shared_only 时单行）。"""
    slice_bin = (q_cold.shape[0] - 1) // 2
    rows = [("shared", q_cold)] if shared_only else [("cold", q_cold), ("warm", q_warm)]
    # 共享色标上限（跨行跨平面），保证 cold/warm 可比
    nz = np.concatenate([q[q > 0] for _, q in rows]) if rows[0][0] == "shared" \
        else np.concatenate([q[q > 0] for _, q in rows] + [q_warm[q_warm > 0]])
    vmax = float(np.quantile(nz, pct / 100.0)) if nz.size else 1.0
    norm = PowerNorm(gamma=gamma, vmin=0, vmax=vmax) if gamma is not None \
        else plt.Normalize(vmin=0, vmax=vmax)

    fig, axes = plt.subplots(len(rows), 3, figsize=(18, 6.0 * len(rows)),
                             squeeze=False)
    for ri, (label, q) in enumerate(rows):
        for ci, (plane, (keep, row_a, col_a)) in enumerate(PLANES.items()):
            ax = axes[ri][ci]
            s = np.take(q, slice_bin, axis=keep)
            ext = annotate_axes(ax, row_a, col_a, lo, span, mm)
            im = ax.imshow(s, origin="lower", cmap=cmap, norm=norm, extent=ext,
                           aspect=(ext[1] - ext[0]) / (ext[3] - ext[2]))
            ax.set_title(f"{plane}" + (f" ({label})" if not shared_only else ""))
            fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(suptitle, y=0.995)
    out = os.path.join(out_dir, f"{name}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")



def plot_gaussian_count(evals, lo, span, mm, out_dir, bins, pct):
    """全量高斯的三向投影直方图；共享范围保证 cold/warm 可直接比较。"""
    xyz_all = np.concatenate([xyz for xyz, _ in evals.values()])
    bounds_lo = np.minimum(lo, xyz_all.min(axis=0))
    bounds_hi = np.maximum(lo + span, xyz_all.max(axis=0))
    pad = np.maximum((bounds_hi - bounds_lo) * 0.01, 1e-6)
    bounds_lo -= pad
    bounds_hi += pad

    hist = {}
    for label, (xyz, _) in evals.items():
        hist[label] = {}
        for plane, (_, row_a, col_a) in PLANES.items():
            h, _, _ = np.histogram2d(
                xyz[:, row_a], xyz[:, col_a], bins=bins,
                range=((bounds_lo[row_a], bounds_hi[row_a]),
                       (bounds_lo[col_a], bounds_hi[col_a])))
            assert int(h.sum()) == len(xyz)
            hist[label][plane] = h

    nz = np.concatenate([h[h > 0] for by_plane in hist.values()
                         for h in by_plane.values()])
    vmax = float(np.quantile(nz, pct / 100.0)) if nz.size else 1.0
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), squeeze=False)
    for ri, label in enumerate(("cold", "warm")):
        for ci, (plane, (_, row_a, col_a)) in enumerate(PLANES.items()):
            ax = axes[ri][ci]
            ext = [bounds_lo[col_a] * mm, bounds_hi[col_a] * mm,
                   bounds_lo[row_a] * mm, bounds_hi[row_a] * mm]
            im = ax.imshow(hist[label][plane], origin="lower", cmap="viridis",
                           vmin=0, vmax=vmax, extent=ext, aspect="equal")
            ax.add_patch(plt.Rectangle(
                (lo[col_a] * mm, lo[row_a] * mm), span[col_a] * mm,
                span[row_a] * mm, fill=False, color="white", ls="--", lw=1))
            ax.set_xlabel(AXIS_LABEL[col_a][0])
            ax.set_ylabel(AXIS_LABEL[row_a][0].replace("→", "↑"))
            ax.set_title(f"{plane} ({label})")
            fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"NG: all-gaussian projected count ({bins}x{bins} histogram)\n"
                 f"linear vmax=p{pct} (shared); dashed = reconstruction box", y=0.995)
    out = os.path.join(out_dir, "factors_gaussian_count.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def z_profiles(nv, ng_c, ng_w, g_c, g_w, bins, lo, span):
    """按 z 分 8 片输出数值剖面：Nv / NG / 每高斯平均梯度（检验 z 边缘假说）。"""
    zs = bins // 8
    print("\nz-slab 剖面（8 片，自下而上）: "
          "Nv=每体素平均有效视角数, NG=高斯总数, <grad>=Σgrad/NG")
    print(f"{'z range (mm)':>16} | {'Nv':>7} | {'NG cold':>8} {'NG warm':>8} | "
          f"{'<grad> cold':>11} {'<grad> warm':>11}")
    for j in range(8):
        sl = slice(j * zs, (j + 1) * zs)
        z0 = (lo[2] + (j * zs) / (bins - 1) * span[2]) * 9.5
        z1 = (lo[2] + ((j + 1) * zs - 1) / (bins - 1) * span[2]) * 9.5
        nv_j = nv[:, :, sl].mean()
        ngc = int(ng_c[:, :, sl].sum())
        ngw = int(ng_w[:, :, sl].sum())
        gc = g_c[:, :, sl].sum()
        gw = g_w[:, :, sl].sum()
        print(f"{z0:7.1f}..{z1:7.1f} | {nv_j:7.1f} | {ngc:8d} {ngw:8d} | "
              f"{gc / ngc:11.3e} {gw / ngw:11.3e}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cold-dir", default=DEFAULT_COLD)
    ap.add_argument("--warm-dir", default=DEFAULT_WARM)
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--step", default="step_005000")
    ap.add_argument("--output-dir", default=DEFAULT_OUTDIR)
    ap.add_argument("--bins", type=int, default=BINS)
    ap.add_argument("--count-bins", type=int, default=COUNT_BINS,
                    help="NG 全量投影图的每轴分辨率")
    ap.add_argument("--gamma", type=float, default=0.3,
                    help="梯度类图的 PowerNorm gamma（<1 提亮低值）")
    ap.add_argument("--pct", type=float, default=99.9,
                    help="共享色标上限取非零值分位数")
    args = ap.parse_args()

    if plt is None:
        raise SystemExit("matplotlib 不可用")
    if args.count_bins < 256:
        raise SystemExit("--count-bins 必须 >= 256")
    bins = args.bins
    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.dataset, "meta_data.json")) as fh:
        meta = json.load(fh)
    scanner = meta["scanner"]
    lo, span, mm = scanner_grid(scanner)

    evals = {}
    for name, mdir in (("cold", args.cold_dir), ("warm", args.warm_dir)):
        rec = np.load(os.path.join(mdir, "eval", args.step, "densify_grads.npz"))
        evals[name] = (rec["xyz"], rec["grads"])
        print(f"{name}: N={len(rec['grads'])} kind={rec['kind'].item()}")

    # ---- Nv(x)：几何有效投影数（冷暖共享） ----
    centers = bin_centers(lo, span, bins)
    nv = compute_nviews(centers, scanner, meta, bins)
    print(f"Nv: min={nv.min()} max={nv.max()} mean={nv.mean():.1f}")

    # ---- NG(x) 与 G(x)=Σ accum/denom ----
    ng, gg = {}, {}
    for name, (xyz, grads) in evals.items():
        ng[name] = grad_hist(xyz, grads, lo, span, bins, weighted=False)
        gg[name] = grad_hist(xyz, grads, lo, span, bins, weighted=True)
        print(f"NG {name}: total={ng[name].sum():.0f}  G {name}: total={gg[name].sum():.3e}")

    g_c, g_w = gg["cold"], gg["warm"]
    ng_c, ng_w = ng["cold"], ng["warm"]
    eps = 1e-12

    # ---- 图 1-2: Nv / NG（线性色标） ----
    plot_quantity(nv, None, "factors_nviews",
                  "Nv(x): effective train views per voxel (geometric coverage, "
                  "shared by cold/warm)\n"
                  f"linear vmax=p{args.pct} (mid-slices, volume voxel {SLICE_VOXEL})",
                  lo, span, mm, args.output_dir, cmap="viridis", gamma=None,
                  pct=args.pct, shared_only=True)
    plot_gaussian_count(evals, lo, span, mm, args.output_dir,
                        args.count_bins, args.pct)

    # ---- 图 3-5: 三种梯度归一化 ----
    plot_quantity(g_c, g_w, "factors_grad_density",
                  "G(x) = sum(accum/denom): spatial density of view-averaged "
                  "position gradient norm\n"
                  f"PowerNorm gamma={args.gamma}, vmax=p{args.pct} (shared)",
                  lo, span, mm, args.output_dir, gamma=args.gamma, pct=args.pct)
    plot_quantity(g_c / (ng_c + eps), g_w / (ng_w + eps),
                  "factors_grad_per_gaussian",
                  "G(x)/NG(x): mean position gradient norm per gaussian\n"
                  f"PowerNorm gamma={args.gamma}, vmax=p{args.pct} (shared)",
                  lo, span, mm, args.output_dir, gamma=args.gamma, pct=args.pct)
    plot_quantity(g_c / (nv * ng_c + eps), g_w / (nv * ng_w + eps),
                  "factors_grad_normalized",
                  "G(x)/(Nv(x)*NG(x)): normalized by effective views and "
                  "gaussian count\n"
                  f"PowerNorm gamma={args.gamma}, vmax=p{args.pct} (shared)",
                  lo, span, mm, args.output_dir, gamma=args.gamma, pct=args.pct)

    # ---- 数值剖面 + 存盘 ----
    z_profiles(nv, ng_c, ng_w, g_c, g_w, bins, lo, span)
    np.savez(os.path.join(args.output_dir, "grad_factors.npz"),
             nv=nv, ng_cold=ng_c, ng_warm=ng_w,
             g_cold=g_c, g_warm=g_w, lo=lo, span=span)


if __name__ == "__main__":
    main()
