"""对比冷/暖启动训练动态：densification 程度与强度空间时间分布。

输入：两个训练好的模型目录（各自含 tensorboard/ 事件文件与 eval/step_XXXXX/ 序列）。
- densification 曲线:  tensorboard `scene/num_gaussians`（每 10 步一次）
- 强度分布时间演化:  tensorboard `scene/density` 直方图（每个 eval 步一次）
- 空间时间分布:      eval/step_XXXXX/vol_pred.tiff 的轴向(z, axis0)平均强度剖面
                      随 eval 步的演化（含 warm - cold 差分图）
- 指标曲线:          eval yml 的 psnr3d/ssim3d/psnr2d/ssim2d

用法:
    MPLCONFIGDIR=/tmp/spiral-gs-matplotlib python experiments/helpers/analyze_warm_cold_dynamics.py \
        --cold models/real/ldctc001/spiral/ntrain1000/factgs_det_native_cold \
        --warm models/real/ldctc001/spiral/ntrain1000/factgs_det_native_warm_fdk1000
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from plot_roi import load_volume  # noqa: E402

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None


def read_scalar(events_dir, tag):
    """从 TensorBoard 事件文件读取 (step, value) 序列。"""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    acc = EventAccumulator(events_dir, size_guidance={"scalars": 0, "histograms": 0})
    acc.Reload()
    if tag not in acc.Tags().get("scalars", []):
        return None
    evs = acc.Scalars(tag)
    return np.asarray([(e.step, e.value) for e in evs], dtype=float)


def read_histogram(events_dir, tag):
    """读取直方图事件：返回 [(step, bin_edges, counts)]，按 step 排序。"""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    acc = EventAccumulator(events_dir, size_guidance={"scalars": 0, "histograms": 0})
    acc.Reload()
    if tag not in acc.Tags().get("histograms", []):
        return []
    out = []
    for e in acc.Histograms(tag):
        h = e.histogram_value
        edges = np.asarray(h.bucket_limit, dtype=float)
        counts = np.asarray(h.bucket, dtype=float)
        out.append((e.step, edges, counts))
    out.sort(key=lambda x: x[0])
    return out


def eval_steps(run_dir):
    steps = sorted(
        int(d.split("_")[1])
        for d in os.listdir(os.path.join(run_dir, "eval"))
        if d.startswith("step_") and os.path.isdir(os.path.join(run_dir, "eval", d))
    )
    return steps


def volume_z_profile(vol):
    """axis0（z）方向的平均强度剖面。"""
    return vol.mean(axis=(1, 2))


def read_yml_metrics(eval_dir):
    out = {}
    for fname, keys in (("eval3d.yml", ("psnr_3d", "ssim_3d")),
                        ("eval2d_render_test.yml", ("psnr_2d", "ssim_2d"))):
        path = os.path.join(eval_dir, fname)
        if not os.path.isfile(path):
            continue
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                if ":" not in line:
                    continue
                k, _, v = line.partition(":")
                if k.strip() in keys:
                    try:
                        out[k.strip()] = float(v.strip())
                    except ValueError:
                        pass
    return out


def densification_stats(steps_counts):
    steps, counts = steps_counts[:, 0], steps_counts[:, 1]
    return {
        "init": int(counts[0]),
        "max": int(counts.max()),
        "max_step": int(steps[counts.argmax()]),
        "final": int(counts[-1]),
        "growth_ratio": float(counts[-1] / counts[0]) if counts[0] else 0.0,
        "max_growth_ratio": float(counts.max() / counts[0]) if counts[0] else 0.0,
        "pruned_below_init": bool((counts < counts[0]).any()),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cold", required=True)
    ap.add_argument("--warm", required=True)
    ap.add_argument("--output", default=None,
                    help="输出目录（默认 output/real/ldctc001/spiral/ntrain1000/dynamics）")
    ap.add_argument("--z-axis", type=int, default=0, help="轴向剖面使用的轴（默认 0=z）")
    args = ap.parse_args()

    if plt is None:
        raise SystemExit("matplotlib 不可用")

    cold_dir, warm_dir = os.path.abspath(args.cold), os.path.abspath(args.warm)
    out_dir = args.output or os.path.join(
        ROOT := os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "output", "real", "ldctc001", "spiral", "ntrain1000", "dynamics")
    os.makedirs(out_dir, exist_ok=True)

    # ---------------- 1) densification 曲线 ----------------
    series = {}
    for name, run in (("cold", cold_dir), ("warm", warm_dir)):
        tb = glob.glob(os.path.join(run, "tensorboard", "events.out.tfevents.*"))
        if not tb:
            raise FileNotFoundError(f"{name} 缺少 tensorboard 事件文件: {run}/tensorboard")
        series[name] = read_scalar(os.path.dirname(tb[0]), "scene/num_gaussians")
        if series[name] is None:
            raise RuntimeError(f"{name} 的 tensorboard 中没有 scene/num_gaussians")

    stats = {n: densification_stats(s) for n, s in series.items()}
    print("=== densification 程度（num_gaussians）===")
    for n in ("cold", "warm"):
        s = stats[n]
        print(f"  {n:5s}: init={s['init']} max={s['max']}@step{s['max_step']} "
              f"final={s['final']} growth={s['growth_ratio']:.2f}x "
              f"max_growth={s['max_growth_ratio']:.2f}x pruned_below_init={s['pruned_below_init']}")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, color, ls in (("cold", "tab:blue", "-"), ("warm", "tab:red", "-")):
        st, cnt = series[name].T
        ax.plot(st, cnt, color=color, ls=ls, lw=1.6, label=f"{name} (final {stats[name]['final']:,})")
    ax.set_xlabel("training step")
    ax.set_ylabel("num_gaussians")
    ax.set_title("Densification: cold vs warm start (ldctc001 det4 n1000)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "densification_curve.png"), dpi=200)
    plt.close(fig)

    # ---------------- 2) 指标曲线 ----------------
    ev_steps = {n: eval_steps(r) for n, r in (("cold", cold_dir), ("warm", warm_dir))}
    metrics = {n: {s: read_yml_metrics(os.path.join(r, "eval", f"step_{s:06d}"))
                   for s in ev_steps[n]} for n, r in (("cold", cold_dir), ("warm", warm_dir))}
    print("=== eval 指标（每 5000 步）===")
    for s in sorted(set(ev_steps["cold"]) & set(ev_steps["warm"])):
        row = f"  step {s:6d}: "
        for m in ("psnr_3d", "ssim_3d", "psnr_2d", "ssim_2d"):
            row += f"{m} " + "/".join(f"{metrics[n][s].get(m, float('nan')):.3f}" for n in ("cold", "warm")) + "  "
        print(row)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for m, ax in zip(("psnr_3d", "ssim_3d"), axes):
        for name, color in (("cold", "tab:blue"), ("warm", "tab:red")):
            ss = sorted(metrics[name])
            ax.plot(ss, [metrics[name][s].get(m, np.nan) for s in ss],
                    marker="o", color=color, lw=1.5, label=name)
        ax.set_title(m)
        ax.set_xlabel("step")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "metrics_vs_step.png"), dpi=200)
    plt.close(fig)

    # ---------------- 3) 强度空间时间分布（vol_pred 轴向剖面） ----------------
    profiles = {}
    for name, run in (("cold", cold_dir), ("warm", warm_dir)):
        steps = sorted(set(ev_steps[name]))
        prof = np.stack([volume_z_profile(load_volume(os.path.join(
            run, "eval", f"step_{s:06d}", "vol_pred.tiff"))) for s in steps], axis=0)
        profiles[name] = (np.asarray(steps), prof)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    vmax = max(p.max() for _, p in profiles.values())
    for ax, (name, color) in zip(axes[:2], (("cold", "tab:blue"), ("warm", "tab:red"))):
        steps, prof = profiles[name]
        im = ax.imshow(prof.T, aspect="auto", origin="lower", vmin=0, vmax=vmax,
                       extent=[steps[0], steps[-1], 0, prof.shape[1]], cmap="magma")
        ax.set_title(f"{name} start\nmean intensity per z-slice vs step")
        ax.set_xlabel("eval step")
        ax.set_ylabel(f"z voxel (axis {args.z_axis})")
        fig.colorbar(im, ax=ax, fraction=0.046)
    steps_c, pc = profiles["cold"]
    steps_w, pw = profiles["warm"]
    if len(steps_c) == len(steps_w) and np.allclose(steps_c, steps_w):
        diff = pw - pc
        lim = np.abs(diff).max() or 1.0
        im = axes[2].imshow(diff.T, aspect="auto", origin="lower", cmap="RdBu_r",
                            vmin=-lim, vmax=lim,
                            extent=[steps_w[0], steps_w[-1], 0, diff.shape[1]])
        axes[2].set_title("warm - cold\nintensity spatiotemporal diff")
        axes[2].set_xlabel("eval step")
        fig.colorbar(im, ax=axes[2], fraction=0.046)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "intensity_spatiotemporal.png"), dpi=200)
    plt.close(fig)

    # ---------------- 4) density 直方图时间演化 ----------------
    hists = {}
    for name, run in (("cold", cold_dir), ("warm", warm_dir)):
        tb = glob.glob(os.path.join(run, "tensorboard", "events.out.tfevents.*"))[0]
        hists[name] = read_histogram(os.path.dirname(tb), "scene/density")
    print("=== scene/density 直方图均值（每个 eval 步）===")
    for name in ("cold", "warm"):
        means = []
        for step, edges, counts in hists[name]:
            centers = (edges[:-1] + edges[1:]) / 2 if len(edges) > 1 else edges
            counts = np.asarray(counts)[: len(centers)]  # bucket_limit 比 bucket 多一个边界
            total = counts.sum()
            means.append((step, float((centers * counts).sum() / total) if total else 0.0))
        print(f"  {name:5s}: " + "  ".join(f"{s}:{m:.3f}" for s, m in means))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, (name, color) in zip(axes, (("cold", "tab:blue"), ("warm", "tab:red"))):
        for step, edges, counts in hists[name]:
            centers = (edges[:-1] + edges[1:]) / 2
            counts = np.asarray(counts)[: len(centers)]
            ax.semilogy(centers, counts, lw=1.2, color=color,
                        alpha=0.55 + 0.45 * (step / max(1, max((s for s, _, _ in hists[name])))))
        ax.set_title(f"{name}: scene/density hist over steps (darker=later)")
        ax.set_xlabel("gaussian density")
        ax.set_ylim(1, None)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("count")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "density_hist_evolution.png"), dpi=200)
    plt.close(fig)

    with open(os.path.join(out_dir, "dynamics_summary.json"), "w") as fh:
        json.dump({"densification": {n: stats[n] for n in stats},
                   "eval_steps": ev_steps,
                   "metrics": {n: {str(s): m for s, m in metrics[n].items()} for n in metrics}},
                  fh, indent=2, default=str)
    print(f"\noutputs -> {out_dir}")


if __name__ == "__main__":
    main()
