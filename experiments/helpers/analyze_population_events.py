"""用 population_events/step_*.npz 做 Experiment B/C/D 分析（plan §8-§10）。

对齐关系（关键不变量）：densify/prune 只发生在事件内部；相邻两事件之间只有
optimizer.step。因此：
  xyz_post(事件 e) 与 xyz_pre(事件 e+1) 逐行同 id → 幸存高斯的真实位移
  Δp = xyz_pre(e+1) - xyz_post(e)（plan §17：actual displacement 优先于梯度）。
事件内的出生（birth_*，clone=1/split=2，含 parent）与剪枝（prune_*，
tag 1=prune[reason 1..4]，tag 2=split_parent）由 gaussian_model 事件钩子记录。

输出（--output-dir，默认 grad_compare 目录）：
1. nout_fine_curve.png —— 事件分辨率 N_out/R_out 曲线 + 六方向（填补 Exp A 的
   0→5k 空隙；0 点取 init 文件）
2. displacement_hists.png —— 幸存高斯逐窗口位移：|Δp| 与 Δz 直方图（cold vs warm）
3. boundary_gradient_direction.png —— 边界带高斯的世界梯度 outward 分量统计
   （d = -g·n；z_low/z_high/x/y 边界，cold/warm，事件序列）
4. z_accounting.png —— 8 个 z bin 的 birth / prune / net-migration / net-change
   分解（事件序列，cold/warm 对照）
5. births_outside.png —— 出生与剪枝中位于体积外的数量随时间（出生在体积外
   直接回答"densify 直接在体积外创建"假说）

全部数值同时写入 stdout 摘要与 --output-dir/population_accounting_summary.csv。

用法:
    MPLCONFIGDIR=/tmp/spiral-gs-matplotlib python experiments/helpers/analyze_population_events.py \
        --model models/real/ldctc002/spiral/ntrain1000/factgs_det_native_cold_rec \
        --label cold [--init-xyz .../init_r2gs.npy]
"""

import argparse
import glob
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
from plot_c002_grad_factors import scanner_grid  # noqa: E402

DEFAULT_DATASET = os.path.join(ROOT, "data/real/ldctc002/spiral/ntrain1000/r2gs")
DEFAULT_OUTDIR = os.path.join(ROOT, "output/real/ldctc002/spiral/ntrain1000/grad_compare")

BIRTH_NAME = {1: "clone", 2: "split"}
PRUNE_NAME = {1: "prune", 2: "split_parent"}
REASON_NAME = {1: "density", 2: "bbox", 3: "screen", 4: "scale"}


def load_events(model_dir):
    files = sorted(glob.glob(os.path.join(model_dir, "population_events", "step_*.npz")))
    events = []
    for f in files:
        z = np.load(f)
        events.append({
            "step": int(z["step"].item()),
            "xyz_pre": z["xyz_pre"],
            "xyz_post": z["xyz_post"],
            "grad": z["xyz_grad_pre"] if "xyz_grad_pre" in z else None,
            "birth_xyz": z["birth_xyz"] if "birth_xyz" in z else None,
            "birth_type": z["birth_type"] if "birth_type" in z else None,
            "birth_parent_xyz": z["birth_parent_xyz"] if "birth_parent_xyz" in z else None,
            "prune_xyz": z["prune_xyz"] if "prune_xyz" in z else None,
            "prune_tag": z["prune_tag"] if "prune_tag" in z else None,
            "prune_reason": z["prune_reason"] if "prune_reason" in z else None,
        })
    return events


def outside_mask(xyz, lo, hi):
    m = np.zeros(len(xyz), dtype=bool)
    for a in range(3):
        m |= (xyz[:, a] < lo[a]) | (xyz[:, a] > hi[a])
    return m


def zbin_of(xyz, lo, span, bins):
    """z 分箱编码：0 = 体积下方外，1..bins = 体积内 bin，bins+1 = 体积上方外。"""
    idx = np.floor((xyz[:, 2] - lo[2]) / span[2] * (bins - 1)).astype(int)
    idx = np.clip(idx, 0, bins - 1) + 1
    idx[xyz[:, 2] < lo[2]] = 0
    idx[xyz[:, 2] > lo[2] + span[2]] = bins + 1
    return idx


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="population_events 所在模型目录")
    ap.add_argument("--label", default="model", help="图例标签（cold/warm）")
    ap.add_argument("--init-xyz", default=None, help="init 位置（npy 前3列 或 pickle）")
    ap.add_argument("--max-step", type=int, default=None,
                    help="只分析 step <= 该值的事件（用于训练进行中截取 0→N 窗口）")
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--output-dir", default=DEFAULT_OUTDIR)
    args = ap.parse_args()

    if plt is None:
        raise SystemExit("matplotlib 不可用")

    with open(os.path.join(args.dataset, "meta_data.json")) as fh:
        scanner = json.load(fh)["scanner"]
    lo, span, _ = scanner_grid(scanner)
    hi = lo + span
    os.makedirs(args.output_dir, exist_ok=True)

    events = load_events(args.model)
    if args.max_step is not None:
        events = [e for e in events if e["step"] <= args.max_step]
    print(f"{args.label}: {len(events)} events, steps "
          f"{events[0]['step']}..{events[-1]['step']}")

    # ---- 事件分辨率 N_out 曲线 ----
    curve = []
    for e in events:
        n = len(e["xyz_post"])
        m = outside_mask(e["xyz_post"], lo, hi)
        curve.append((e["step"], n, int(m.sum()), float(m.mean())))
    print("\nstep  N_total  N_out  R_out")
    for s, n, no, r in curve[::10]:
        print(f"{s:>6} {n:>7} {no:>6} {r:7.2%}")
    if curve:
        s, n, no, r = curve[-1]
        print(f"{s:>6} {n:>7} {no:>6} {r:7.2%}")

    # ---- 位移（幸存者，事件间窗口，id 对齐）----
    disp_mag, disp_z = [], []
    for e1, e2 in zip(events[:-1], events[1:]):
        n = min(len(e1["xyz_post"]), len(e2["xyz_pre"]))
        d = e2["xyz_pre"][:n] - e1["xyz_post"][:n]
        disp_mag.append(np.linalg.norm(d, axis=1))
        disp_z.append(d[:, 2])
    disp_mag = np.concatenate(disp_mag)
    disp_z = np.concatenate(disp_z)
    print(f"\ndisplacement survivors: |Δp| mean={disp_mag.mean():.4g} "
          f"p50={np.median(disp_mag):.4g} p99={np.quantile(disp_mag, 0.99):.4g}")
    print(f"Δz: mean={disp_z.mean():.4g} p50={np.median(disp_z):.4g} "
          f"P(Δz>0)={float((disp_z > 0).mean()):.3f}")

    # ---- 梯度方向（边界带）----
    margin = 0.05  # 边界带宽度（缩放单位）
    gstats = []
    for e in events:
        if e["grad"] is None or len(e["grad"]) != len(e["xyz_pre"]):
            continue
        xyz, g = e["xyz_pre"], e["grad"]
        d = -g  # 更新方向 = -∇L
        for name, nvec, cond in (
            ("z_low", np.array([0, 0, -1.0]), xyz[:, 2] < lo[2] + margin),
            ("z_high", np.array([0, 0, 1.0]), xyz[:, 2] > hi[2] - margin),
            ("x_low", np.array([-1.0, 0, 0]), xyz[:, 0] < lo[0] + margin),
            ("x_high", np.array([1.0, 0, 0]), xyz[:, 0] > hi[0] - margin),
            ("y_low", np.array([0, -1.0, 0]), xyz[:, 1] < lo[1] + margin),
            ("y_high", np.array([0, 1.0, 0]), xyz[:, 1] > hi[1] - margin),
        ):
            d_out = d[cond] @ nvec
            if len(d_out) == 0:
                continue
            gstats.append((e["step"], name, float(d_out.mean()),
                           float((d_out > 0).mean()), len(d_out)))
    if gstats:
        print("\nboundary outward-update stats (d=-g·n), first/last event per side:")
        for name in ("z_low", "z_high", "x_low", "y_low"):
            sub = [r for r in gstats if r[1] == name]
            if not sub:
                continue
            first, last = sub[0], sub[-1]
            print(f"  {name}: step {first[0]}: E[d_out]={first[2]:+.3g} "
                  f"P(d_out>0)={first[3]:.2f} (n={first[4]}) | "
                  f"step {last[0]}: E[d_out]={last[2]:+.3g} P={last[3]:.2f}")

    # ---- z-bin 人口记账 ----
    zb = 8
    for e in events:
        e["pre_bin"] = zbin_of(e["xyz_pre"], lo, span, zb)
        e["post_bin"] = zbin_of(e["xyz_post"], lo, span, zb)
    acct = []
    for ei in range(len(events) - 1):
        e1, e2 = events[ei], events[ei + 1]
        n = min(len(e1["xyz_post"]), len(e2["xyz_pre"]))
        b0, b1 = e1["post_bin"][:n], e2["pre_bin"][:n]
        moved = b1 != b0
        mig = np.bincount(b1[moved], minlength=zb + 2) - np.bincount(b0[moved], minlength=zb + 2)
        births = np.bincount(
            zbin_of(e2["birth_xyz"], lo, span, zb), minlength=zb + 2
        ) if e2["birth_xyz"] is not None else np.zeros(zb + 2)
        prunes = np.bincount(
            zbin_of(e2["prune_xyz"], lo, span, zb), minlength=zb + 2
        ) if e2["prune_xyz"] is not None else np.zeros(zb + 2)
        dn = np.bincount(e2["post_bin"], minlength=zb + 2) - np.bincount(
            e1["post_bin"], minlength=zb + 2)
        acct.append((e2["step"], dn, births, prunes, mig))
    # 守恒检验：dn 应 = births - prunes + mig（幸存者迁移）
    errs = [np.abs(a[1] - (a[2] - a[3] + a[4])).sum() for a in acct]
    print(f"\nz-accounting conservation error (sum |ΔN - (B-P+M)| per event): "
          f"max={max(errs)}, mean={np.mean(errs):.2f}")

    # ---- 出生/剪枝在体积外的数量 ----
    print("\nstep births_out births_total prunes_out prunes_total")
    for e in events:
        bo = int(outside_mask(e["birth_xyz"], lo, hi).sum()) if e["birth_xyz"] is not None else 0
        bt = 0 if e["birth_xyz"] is None else len(e["birth_xyz"])
        po = int(outside_mask(e["prune_xyz"], lo, hi).sum()) if e["prune_xyz"] is not None else 0
        pt = 0 if e["prune_xyz"] is None else len(e["prune_xyz"])
        if bt or pt:
            print(f"{e['step']:>6} {bo:>10} {bt:>12} {po:>10} {pt:>12}")

    # ================= 图 =================
    steps = [e["step"] for e in events]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(steps, [c[3] * 100 for c in curve], "o-", ms=3)
    ax.set_xlabel("step"); ax.set_ylabel("R_out (%)")
    ax.set_title(f"{args.label}: volume-outside ratio at event resolution")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(args.output_dir, f"nout_fine_curve_{args.label}.png")
    fig.savefig(out, dpi=150); plt.close(fig); print(f"saved {out}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(np.log10(disp_mag[disp_mag > 0]), bins=80, color="tab:blue", alpha=0.8)
    axes[0].set_xlabel("log10 |Δp| (per 100-step window)"); axes[0].set_title(args.label)
    axes[1].hist(disp_z, bins=80, color="tab:orange", alpha=0.8)
    axes[1].set_xlabel("Δz (per 100-step window)"); axes[1].set_title(args.label)
    fig.suptitle("survivor displacement between densification events")
    fig.tight_layout()
    out = os.path.join(args.output_dir, f"displacement_hists_{args.label}.png")
    fig.savefig(out, dpi=150); plt.close(fig); print(f"saved {out}")

    if gstats:
        fig, axes = plt.subplots(2, 2, figsize=(13, 8))
        for ax, name in zip(axes.ravel(), ("z_low", "z_high", "x_low", "y_low")):
            sub = [(r[0], r[2], r[3]) for r in gstats if r[1] == name]
            if not sub:
                continue
            ss = [r[0] for r in sub]
            ax.plot(ss, [r[1] for r in sub], "o-", ms=3, label="E[d_out]")
            ax.plot(ss, [r[2] for r in sub], "s-", ms=3, label="P(d_out>0)")
            ax.axhline(0, color="k", lw=0.8, ls="--")
            ax.set_title(f"{name} boundary, {args.label}"); ax.legend()
            ax.grid(alpha=0.3)
        fig.suptitle("outward update component d=-g·n on boundary bands")
        fig.tight_layout()
        out = os.path.join(args.output_dir, f"boundary_gradient_direction_{args.label}.png")
        fig.savefig(out, dpi=150); plt.close(fig); print(f"saved {out}")

    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    for ax, key, title in (
        (axes[0], 2, "births per z-bin"),
        (axes[1], 3, "prunes per z-bin"),
        (axes[2], 4, "survivor net migration per z-bin (in - out)"),
        (axes[3], 1, "net change ΔN per z-bin"),
    ):
        M = np.stack([a[key] for a in acct]).T
        im = ax.imshow(M, aspect="auto", origin="lower", cmap="RdBu_r",
                       extent=[steps[1], steps[-1], -0.5, zb + 0.5])
        ax.set_ylabel("z-bin (0=bottom)"); ax.set_title(f"{title} — {args.label}")
        fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    out = os.path.join(args.output_dir, f"z_accounting_{args.label}.png")
    fig.savefig(out, dpi=150); plt.close(fig); print(f"saved {out}")

    np.savez(os.path.join(args.output_dir, f"population_stats_{args.label}.npz"),
             steps=np.asarray(steps), nout=np.asarray([c[2] for c in curve]),
             disp_mag=disp_mag, disp_z=disp_z,
             gstats=np.asarray(gstats, dtype=object))


if __name__ == "__main__":
    main()
