
# Visualize results produced by ppo_carl_mario training.
# Reads:
#   - progress.csv (SB3)
#   - train_episodes.csv (per-episode)
#   - eval_results.jsonl (per-eval per-context)
# Saves figures into: <run_dir>/figs
#
# Usage:
#   python visualize_mario_results.py --run_dir logs/mario_benchmark/inertia_MMDD_HHMM
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt


def read_csv_dicts(path: Path) -> List[Dict[str, str]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except FileNotFoundError:
        print(f"[WARN] CSV file not found: {path}")
        return []
    except Exception as e:
        print(f"[WARN] Failed reading CSV {path}: {e}")
        return []


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                out.append(json.loads(line))
    except FileNotFoundError:
        print(f"[WARN] JSONL file not found: {path}")
    except Exception as e:
        print(f"[WARN] Failed reading JSONL {path}: {e}")
    return out


def to_float_list(rows: List[Dict[str, str]], key: str) -> List[float]:
    out: List[float] = []
    for r in rows:
        v = r.get(key, None)
        if v is None or v == "":
            out.append(float("nan"))
        else:
            try:
                out.append(float(v))
            except Exception:
                out.append(float("nan"))
    return out


def rolling_mean(values: List[float], window: int) -> np.ndarray:
    a = np.asarray(values, dtype=float)
    n = a.size
    if n == 0 or window <= 1:
        return a
    # min_periods=1 style cumulative
    cs = np.cumsum(a)
    out = np.empty(n, dtype=float)
    for i in range(n):
        start = max(0, i - window + 1)
        s = cs[i] - (cs[start - 1] if start > 0 else 0.0)
        out[i] = s / (i - start + 1)
    return out


def plot_progress(run_dir: Path, figs_dir: Path) -> List[Path]:
    out_paths: List[Path] = []
    prog_path = run_dir / "progress.csv"
    rows = read_csv_dicts(prog_path)
    if not rows:
        return out_paths

    # X-axis: total timesteps
    t = []
    for i, r in enumerate(rows):
        v = r.get("time/total_timesteps", None)
        if v is not None and v != "":
            t.append(float(v))
        else:
            v2 = r.get("total_timesteps", None)
            if v2 is not None and v2 != "":
                t.append(float(v2))
            else:
                t.append(float(i))

    # Rollout means
    rew_mean = to_float_list(rows, "rollout/ep_rew_mean")
    len_mean = to_float_list(rows, "rollout/ep_len_mean")
    if any(not math.isnan(x) for x in rew_mean) or any(not math.isnan(x) for x in len_mean):
        plt.figure(figsize=(7, 4))
        if any(not math.isnan(x) for x in rew_mean):
            plt.plot(t, rew_mean, label="ep_rew_mean")
        if any(not math.isnan(x) for x in len_mean):
            plt.plot(t, len_mean, label="ep_len_mean")
        plt.xlabel("Total timesteps")
        plt.ylabel("Value")
        plt.title("SB3 rollout means")
        plt.legend()
        plt.tight_layout()
        p = figs_dir / "progress_rollout_means.png"
        plt.savefig(p, dpi=150)
        plt.close()
        out_paths.append(p)

    # Other training metrics
    metric_keys = [
        "train/approx_kl",
        "train/clip_fraction",
        "train/entropy_loss",
        "train/value_loss",
        "time/fps",
        "train/learning_rate",
    ]
    for k in metric_keys:
        vals = to_float_list(rows, k)
        if any(not math.isnan(x) for x in vals):
            plt.figure(figsize=(7, 3))
            plt.plot(t, vals)
            plt.xlabel("Total timesteps")
            plt.ylabel(k)
            plt.tight_layout()
            p = figs_dir / f"{k.replace('/','_')}.png"
            plt.savefig(p, dpi=150)
            plt.close()
            out_paths.append(p)

    return out_paths


def plot_train_episodes(run_dir: Path, figs_dir: Path, roll: int = 100) -> List[Path]:
    out_paths: List[Path] = []
    ep_csv = run_dir / "train_episodes.csv"
    rows = read_csv_dicts(ep_csv)
    if not rows:
        return out_paths

    try:
        ep_idx = [int(r.get("episode_idx", i)) for i, r in enumerate(rows)]
    except Exception:
        ep_idx = list(range(len(rows)))

    returns = to_float_list(rows, "return")
    comps = to_float_list(rows, "completion")

    if returns:
        plt.figure(figsize=(8, 4))
        plt.plot(ep_idx, returns, alpha=0.25, label="return")
        rm = rolling_mean(returns, roll)
        plt.plot(ep_idx, rm, label=f"return (roll={roll})")
        plt.xlabel("Episode")
        plt.ylabel("Return")
        plt.title("Training Return per Episode")
        plt.legend()
        plt.tight_layout()
        p = figs_dir / "train_return_rolling.png"
        plt.savefig(p, dpi=150)
        plt.close()
        out_paths.append(p)

    if comps:
        plt.figure(figsize=(8, 4))
        plt.plot(ep_idx, comps, alpha=0.25, label="completion")
        rm = rolling_mean(comps, roll)
        plt.plot(ep_idx, rm, label=f"completion (roll={roll})")
        plt.xlabel("Episode")
        plt.ylabel("Completion")
        plt.title("Training Completion per Episode")
        plt.ylim(0, 1.0)
        plt.legend()
        plt.tight_layout()
        p = figs_dir / "train_completion_rolling.png"
        plt.savefig(p, dpi=150)
        plt.close()
        out_paths.append(p)

    # Context distribution
    try:
        ctx_ids = [int(r.get("context_id", -1)) for r in rows]
        counts = Counter(ctx_ids)
        if counts:
            xs = sorted(counts.keys())
            ys = [counts[k] for k in xs]
            plt.figure(figsize=(6, 4))
            plt.bar([str(x) for x in xs], ys)
            plt.title("Context frequency (train episodes)")
            plt.xlabel("context_id")
            plt.ylabel("count")
            plt.tight_layout()
            p = figs_dir / "train_context_distribution.png"
            plt.savefig(p, dpi=150)
            plt.close()
            out_paths.append(p)
    except Exception:
        pass

    return out_paths


def plot_eval(run_dir: Path, figs_dir: Path) -> List[Path]:
    out_paths: List[Path] = []
    eval_path = run_dir / "eval_results.jsonl"
    erec = read_jsonl(eval_path)
    if not erec:
        return out_paths

    tags = sorted(set(r.get("tag", "eval") for r in erec))
    for metric in ["return_mean", "completion_mean"]:
        for tag in tags:
            # Collect per inertia time series
            by_inertia: Dict[float, List[Tuple[float, float]]] = defaultdict(list)
            for r in erec:
                if r.get("tag", "eval") != tag:
                    continue
                ts = r.get("timestep", None)
                val = r.get(metric, None)
                inertia = r.get("mario_inertia", None)
                if ts is None or val is None or inertia is None:
                    continue
                try:
                    ts = float(ts)
                    inertia = float(inertia)
                    val = float(val)
                except Exception:
                    continue
                by_inertia[inertia].append((ts, val))

            if not by_inertia:
                continue

            plt.figure(figsize=(8, 4))
            for inertia, arr in sorted(by_inertia.items(), key=lambda kv: kv[0]):
                arr = sorted(arr, key=lambda x: x[0])
                xs = [a[0] for a in arr]
                ys = [a[1] for a in arr]
                plt.plot(xs, ys, label=f"inertia={inertia:.2f}")
            plt.xlabel("Timestep")
            plt.ylabel(metric)
            plt.title(f"Eval {metric} over time [{tag}]")
            plt.legend(ncol=2, fontsize=8)
            plt.tight_layout()
            p = figs_dir / f"eval_{metric}_{tag}.png"
            plt.savefig(p, dpi=150)
            plt.close()
            out_paths.append(p)

    # Final per-context bars (last eval per tag)
    for tag in tags:
        last_ts = None
        for r in erec:
            if r.get("tag", None) == tag:
                try:
                    ts = float(r.get("timestep", float("nan")))
                    if not math.isnan(ts):
                        last_ts = ts if last_ts is None else max(last_ts, ts)
                except Exception:
                    continue

        if last_ts is None:
            continue

        sub = []
        for r in erec:
            if r.get("tag", None) == tag:
                try:
                    ts = float(r.get("timestep", float("nan")))
                except Exception:
                    continue
                if ts == last_ts:
                    sub.append(r)

        if not sub:
            continue

        # Sort by inertia for consistent order
        try:
            sub.sort(key=lambda r: float(r.get("mario_inertia", float("inf"))))
        except Exception:
            pass

        inertias = []
        for r in sub:
            try:
                inertias.append(float(r.get("mario_inertia", float("nan"))))
            except Exception:
                inertias.append(float("nan"))

        for metric in ["return_mean", "completion_mean"]:
            vals = []
            for r in sub:
                try:
                    vals.append(float(r.get(metric, float("nan"))))
                except Exception:
                    vals.append(float("nan"))
            plt.figure(figsize=(7, 4))
            labels = [f"{i:.2f}" if not math.isnan(i) else "NA" for i in inertias]
            plt.bar(labels, vals)
            plt.xlabel("mario_inertia")
            plt.ylabel(metric)
            plt.title(f"Final eval {metric} [{tag}] at ts={int(last_ts)}")
            if metric == "completion_mean":
                plt.ylim(0, 1.0)
            plt.tight_layout()
            p = figs_dir / f"eval_final_{metric}_{tag}.png"
            plt.savefig(p, dpi=150)
            plt.close()
            out_paths.append(p)

    return out_paths


def main():
    parser = argparse.ArgumentParser(description="Visualize Mario PPO results.")
    parser.add_argument("--run_dir", type=str, required=True, help="Path to a single run dir (e.g., logs/mario_benchmark/inertia_MMDD_HHMM)")
    parser.add_argument("--roll", type=int, default=100, help="Rolling window for episode plots")
    parser.add_argument("--show", action="store_true", help="Show figures interactively after saving")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.exists():
        print(f"[ERROR] run_dir does not exist: {run_dir}")
        return

    figs_dir = run_dir / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Using run_dir={run_dir}")
    print(f"[INFO] Saving figures to {figs_dir}")

    created: List[Path] = []
    created += plot_progress(run_dir, figs_dir)
    created += plot_train_episodes(run_dir, figs_dir, roll=args.roll)
    created += plot_eval(run_dir, figs_dir)

    if created:
        print("[INFO] Saved figures:")
        for p in created:
            print(f"  - {p}")
    else:
        print("[WARN] No figures were generated. Check that progress.csv, train_episodes.csv, and/or eval_results.jsonl exist in the run_dir.")

    if args.show:
        # Optional interactive display
        print("[INFO] Displaying figures (close windows to exit)...")
        plt.show()


if __name__ == "__main__":
    main()