"""Experiment A（plan §7）：volume-outside Gaussian 时间序列 + 六方向拆解。

输入：
- init 位置：cold = 数据集 init_r2gs.npy；warm = vol_fit_fdk1000 step_500 pickle
- eval 步完整位置：模型 eval/step_XXXXXX/densify_grads.npz 的 xyz（全部高斯，含体积外）
时间点：0(init), 5000, 10000, 15000, 20000, 25000, 30000
（densification 事件每 100 步一次且不保存完整 xyz，只有 eval 步有全量快照）

统计：N_out(t)、R_out(t)=N_out/N_total，并拆六个方向
x_low/x_high/y_low/y_high/z_low/z_high（相对重建体积 lo..lo+span）。

输出（--output-dir）：
- outside_growth.png —— 每模型一行：6 方向 stacked 面积图（对数 y）+ N_out 总量线
- outside_direction_scatter_yz.png —— step 5000/10000/30000 的体积外高斯 yz 散点
  （按方向着色，cold/warm 两行；体积框用灰线标出）
- outside_growth_table.csv —— 全部数值
- stdout —— 表格摘要

用法:
    MPLCONFIGDIR=/tmp/spiral-gs-matplotlib python experiments/helpers/analyze_outside_growth.py
"""

import argparse
import csv
import json
import os
import pickle
import sys

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

BASE = os.path.join(ROOT, "models/real/ldctc002/spiral/ntrain1000")
DEFAULT_COLD = os.path.join(BASE, "factgs_det_native_cold")
DEFAULT_WARM = os.path.join(BASE, "factgs_det_native_warm_fdk1000")
DEFAULT_DATASET = os.path.join(ROOT, "data/real/ldctc002/spiral/ntrain1000/r2gs")
DEFAULT_INIT_COLD = os.path.join(DEFAULT_DATASET, "init_r2gs.npy")
DEFAULT_INIT_WARM = os.path.join(BASE, "vol_fit_fdk1000", "point_cloud",
                                 "step_500", "point_cloud.pickle")
DEFAULT_OUTDIR = os.path.join(ROOT, "output/real/ldctc002/spiral/ntrain1000/grad_compare")

EVAL_STEPS = [5000, 10000, 15000, 20000, 25000, 30000]
DIRS = ["x_low", "x_high", "y_low", "y_high", "z_low", "z_high"]
DIR_COLORS = {
    "x_low": "#4C72B0", "x_high": "#DD8452",
    "y_low": "#55A868", "y_high": "#C44E52",
    "z_low": "#8172B3", "z_high": "#CCB974",
}


def load_xyz(path):
    if path.endswith(".npy"):
        return np.load(path)[:, :3].astype(np.float64)
    with open(path, "rb") as fh:
        d = pickle.load(fh)
    xyz = d["xyz"]
    if hasattr(xyz, "detach"):
        xyz = xyz.detach().cpu().numpy()
    return np.asarray(xyz, dtype=np.float64)


def direction_counts(xyz, lo, hi):
    """六方向计数（角点高斯会同时计入多个方向）；n_out 取并集单独算。"""
    out = {}
    out["x_low"] = int((xyz[:, 0] < lo[0]).sum())
    out["x_high"] = int((xyz[:, 0] > hi[0]).sum())
    out["y_low"] = int((xyz[:, 1] < lo[1]).sum())
    out["y_high"] = int((xyz[:, 1] > hi[1]).sum())
    out["z_low"] = int((xyz[:, 2] < lo[2]).sum())
    out["z_high"] = int((xyz[:, 2] > hi[2]).sum())
    outside = np.zeros(len(xyz), dtype=bool)
    for a in range(3):
        outside |= (xyz[:, a] < lo[a]) | (xyz[:, a] > hi[a])
    out["_n_out"] = int(outside.sum())
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cold-dir", default=DEFAULT_COLD)
    ap.add_argument("--warm-dir", default=DEFAULT_WARM)
    ap.add_argument("--init-cold", default=DEFAULT_INIT_COLD)
    ap.add_argument("--init-warm", default=DEFAULT_INIT_WARM)
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--output-dir", default=DEFAULT_OUTDIR)
    args = ap.parse_args()

    if plt is None:
        raise SystemExit("matplotlib 不可用")
    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.dataset, "meta_data.json")) as fh:
        scanner = json.load(fh)["scanner"]
    sv = np.asarray(scanner["sVoxel"], float)
    off = np.asarray(scanner["offOrigin"], float)
    ss = 2.0 / sv.max()
    lo = off * ss - sv * ss / 2.0
    hi = lo + sv * ss

    rows = []  # (model, step, total, n_out, ratio, dirs...)
    data = {"cold": {}, "warm": {}}
    for label, path in (("cold", args.init_cold), ("warm", args.init_warm)):
        xyz = load_xyz(path)
        data[label][0] = xyz
    for mdir, label in ((args.cold_dir, "cold"), (args.warm_dir, "warm")):
        for step in EVAL_STEPS:
            z = np.load(os.path.join(mdir, "eval", f"step_{step:06d}",
                                     "densify_grads.npz"))
            data[label][step] = z["xyz"]

    for label in ("cold", "warm"):
        for t, xyz in data[label].items():
            dc = direction_counts(xyz, lo, hi)
            n_out = dc.pop("_n_out")  # 并集，非各方向求和（角点会多方向计数）
            rows.append([label, t, len(xyz), n_out, n_out / len(xyz)] +
                        [dc[d] for d in DIRS])

    # ---- stdout 表格 ----
    print(f"volume lo={lo.round(3)} hi={hi.round(3)} (scaled units)")
    print("model step    total   N_out  R_out    " +
          " ".join(f"{d:>8}" for d in DIRS))
    for r in rows:
        label, t, tot, nout, ratio = r[:5]
        print(f"{label:4} {t:>6} {tot:>7} {nout:>6} {ratio:7.2%}   " +
              " ".join(f"{v:>8}" for v in r[5:]))

    # ---- csv ----
    csv_path = os.path.join(args.output_dir, "outside_growth_table.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "step", "total", "n_out", "r_out"] + DIRS)
        w.writerows(rows)
    print(f"saved {csv_path}")

    # ---- 图 1: stacked area ----
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for ri, label in enumerate(("cold", "warm")):
        ax = axes[ri]
        sub = [r for r in rows if r[0] == label]
        ts = [r[1] for r in sub]
        stacks = {d: [r[5 + i] for r in sub] for i, d in enumerate(DIRS)}
        bottom = np.zeros(len(ts))
        for d in DIRS:
            vals = np.asarray(stacks[d], float)
            ax.fill_between(ts, bottom, bottom + vals, label=d,
                            color=DIR_COLORS[d], alpha=0.85, lw=0)
            bottom = bottom + vals
        ax.set_yscale("log")
        ax.set_title(f"{label} volume-outside gaussians by direction")
        ax.set_ylabel("count (log)")
        ax.legend(ncol=6, fontsize=8, loc="upper left")
        ax.grid(alpha=0.3, which="both")
    axes[1].set_xlabel("step")
    fig.suptitle("Experiment A: N_out(t) growth and 6-direction decomposition "
                 "(0=init; volume = reconstruction box)", y=0.995)
    out = os.path.join(args.output_dir, "outside_growth.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")

    # ---- 图 2: yz 散点（方向着色）----
    steps_show = [5000, 10000, 30000]
    fig, axes = plt.subplots(2, 3, figsize=(14, 9), squeeze=False)
    for ri, label in enumerate(("cold", "warm")):
        for ci, t in enumerate(steps_show):
            ax = axes[ri][ci]
            xyz = data[label][t]
            outside = np.zeros(len(xyz), dtype=bool)
            for a in range(3):
                outside |= (xyz[:, a] < lo[a]) | (xyz[:, a] > hi[a])
            xyz_o = xyz[outside]
            # 按方向分块画（一个点可能同时在两个方向外，归属第一个命中）
            tag = np.full(len(xyz_o), "", dtype=object)
            for d, cond in (
                ("x_low", xyz_o[:, 0] < lo[0]),
                ("x_high", xyz_o[:, 0] > hi[0]),
                ("y_low", xyz_o[:, 1] < lo[1]),
                ("y_high", xyz_o[:, 1] > hi[1]),
                ("z_low", xyz_o[:, 2] < lo[2]),
                ("z_high", xyz_o[:, 2] > hi[2]),
            ):
                tag[cond & (tag == "")] = d
            for d in DIRS:
                m = tag == d
                ax.scatter(xyz_o[m, 2], xyz_o[m, 1], s=0.4, alpha=0.5,
                           color=DIR_COLORS[d], label=d if ri == 0 and ci == 0 else None,
                           rasterized=True)
            ax.set_xlim(lo[2] - 0.3, hi[2] + 0.3)
            ax.set_ylim(lo[1] - 0.3, hi[1] + 0.3)
            ax.axvline(lo[2], color="gray", lw=0.8, ls="--")
            ax.axvline(hi[2], color="gray", lw=0.8, ls="--")
            ax.axhline(lo[1], color="gray", lw=0.8, ls="--")
            ax.axhline(hi[1], color="gray", lw=0.8, ls="--")
            if ri == 0:
                ax.set_title(f"step {t}")
            if ci == 0:
                ax.set_ylabel(f"{label}\nz →")
            if ri == 1:
                ax.set_xlabel("y →")
    axes[0][0].legend(fontsize=7, markerscale=8, loc="upper right")
    fig.suptitle("yz scatter of volume-outside gaussians (colored by first-hit "
                 "direction; dashed = volume box)", y=0.995)
    out = os.path.join(args.output_dir, "outside_direction_scatter_yz.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
