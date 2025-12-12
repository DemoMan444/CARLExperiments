# run_multi_seeds.py
from __future__ import annotations
import json
import time
from pathlib import Path
from copy import deepcopy
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt

import numpy as np

from Inertiaevaluation import TrainingConfig, train


# ============================================================================
# CONFIGURATION TOGGLE
# ============================================================================
# Set to True for quick testing (verify code works)
# Set to False for full experiment (proper evaluation)
QUICK_TEST = False

# Smoothing for episode-level reward curve (higher = smoother)
EPISODE_RETURN_SMOOTH_WINDOW = 50
# ============================================================================

def t_critical_975(df: int) -> float:
    """
    Two-sided 95% CI critical value: t_{0.975, df}.

    With small numbers of seeds (e.g., 5–7), this is more appropriate than the
    normal approximation (1.96).
    """
    if df <= 0:
        return 0.0

    table = {
        1: 12.706,
        2: 4.303,
        3: 3.182,
        4: 2.776,
        5: 2.571,
        6: 2.447,
        7: 2.365,
        8: 2.306,
        9: 2.262,
        10: 2.228,
        11: 2.201,
        12: 2.179,
        13: 2.160,
        14: 2.145,
        15: 2.131,
        16: 2.120,
        17: 2.110,
        18: 2.101,
        19: 2.093,
        20: 2.086,
        21: 2.080,
        22: 2.074,
        23: 2.069,
        24: 2.064,
        25: 2.060,
        26: 2.056,
        27: 2.052,
        28: 2.048,
        29: 2.045,
        30: 2.042,
        40: 2.021,
        60: 2.000,
        120: 1.980,
    }
    if df in table:
        return table[df]
    if df > 120:
        return 1.960

    # Choose nearest available df (keeps this dependency-free and stable).
    nearest = min(table.keys(), key=lambda k: abs(k - df))
    return table[nearest]


def get_config() -> TrainingConfig:
    """Get configuration based on QUICK_TEST toggle."""
    cfg = TrainingConfig()
    
    if QUICK_TEST:
        # Quick test settings - minimal resources for code verification
        cfg.total_timesteps = 5_000
        cfg.n_eval_episodes = 1
        cfg.n_train_maps = 2
        cfg.n_test_maps = 2
        cfg.eval_freq = 7_500
        cfg.print_freq = 1_000
        cfg.checkpoint_freq = 5_000
        cfg.n_envs = 4
    else:
        # Full experiment settings - proper evaluation
        cfg.total_timesteps = 40_000
        cfg.n_eval_episodes = 5
        cfg.n_train_maps = 5
        cfg.n_test_maps = 5
        cfg.eval_freq = 50_000
        cfg.print_freq = 5_000
        cfg.checkpoint_freq = 30_000
        cfg.n_envs = 8
    
    return cfg


def get_seeds() -> list[int]:
    """Get seeds based on QUICK_TEST toggle."""
    if QUICK_TEST:
        return [0, 1]  # fewer seeds for quick test
    else:
        return [0, 1, 2, 3, 4]  # more seeds for full experiment


def get_log_dir() -> str:
    """Get log directory based on QUICK_TEST toggle."""
    if QUICK_TEST:
        return "./logs/mario_inertia_quick"
    else:
        return "./logs/mario_inertia_full"


# ============================================================================
# Core Functions (unchanged)
# ============================================================================

def run_single_seed(
    seed: int,
    base_log_dir: str,
    base_config: TrainingConfig | None = None,
):
    """
    Run one full training with a given seed and return (seed, results, run_dir).

    base_log_dir is the *experiment-level* directory under which each seed
    gets its own subfolder, so runs don't overwrite each other.
    """
    if base_config is None:
        base_config = TrainingConfig()

    cfg = deepcopy(base_config)

    # Important: make log_dir unique per seed so your timestamp-based run_id
    # doesn't collide when you run multiple trainings.
    cfg.seed = seed
    cfg.log_dir = str(Path(base_log_dir) / f"seed_{seed}")

    model, final_results, run_dir = train(cfg)

    # Ensure per-episode rewards are available as a granular training curve
    episode_history = final_results.get("episode_history", [])
    try:
        ep_path = Path(run_dir) / "episode_history.json"
        if episode_history and not ep_path.exists():
            with ep_path.open("w") as f:
                json.dump(episode_history, f, indent=2)
    except Exception:
        pass

    # Save final_results as JSON to the run directory
    out_path = Path(run_dir) / "final_results.json"
    with out_path.open("w") as f:
        json.dump(final_results, f, indent=2)

    return {
        "seed": seed,
        "results": final_results,
        "run_dir": str(run_dir),
    }


def aggregate_runs(runs: list[dict]) -> dict:
    """
    Aggregates across multiple seeds:
      - per-inertia: mean_return, mean_completion, success_rate (mean & std over seeds)
      - per-group: same, by generalization type (id/interp/extra/...)
      - training_curve: reward & completion vs steps with across-seed mean/std/95% CI
    """
    # --- final evaluation aggregates (stochastic) ---
    per_inertia_vals = defaultdict(lambda: {"returns": [], "completions": [], "success_rates": []})
    per_group_vals = defaultdict(lambda: {"returns": [], "completions": [], "success_rates": []})

    # --- training curve aggregates ---
    per_timestep_vals = defaultdict(lambda: {"returns": [], "completions": []})

    for run in runs:
        res = run["results"]["stochastic"]

        # Per-inertia
        for inertia, stats in res["per_inertia"].items():
            inertia_f = float(inertia)
            per_inertia_vals[inertia_f]["returns"].append(float(stats.get("mean_return", float("nan"))))
            per_inertia_vals[inertia_f]["completions"].append(float(stats.get("mean_completion", float("nan"))))
            per_inertia_vals[inertia_f]["success_rates"].append(float(stats.get("success_rate", float("nan"))))

        # Per generalization group
        for gtype, stats in res["per_group"].items():
            per_group_vals[gtype]["returns"].append(float(stats.get("mean_return", float("nan"))))
            per_group_vals[gtype]["completions"].append(float(stats.get("mean_completion", float("nan"))))
            per_group_vals[gtype]["success_rates"].append(float(stats.get("success_rate", float("nan"))))

        # Training curve
        curve = run["results"].get("training_curve", [])
        for pt in curve:
            t = int(pt["timesteps"])
            mr = pt.get("mean_return", None)
            mc = pt.get("mean_completion", None)

            if mr is not None:
                per_timestep_vals[t]["returns"].append(float(mr))
            if mc is not None:
                per_timestep_vals[t]["completions"].append(float(mc))

    def summarize(values: list[float]) -> tuple[float, float]:
        arr = np.array(values, dtype=float)
        arr = arr[~np.isnan(arr)]
        if arr.size == 0:
            return float("nan"), float("nan")
        mean = float(arr.mean())
        std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
        return mean, std

    agg: dict = {
        "per_inertia": {},
        "per_group": {},
        "training_curve": {},
        "n_runs": len(runs),
    }

    # --- per inertia ---
    for inertia, vals in per_inertia_vals.items():
        r_mean, r_std = summarize(vals["returns"])
        c_mean, c_std = summarize(vals["completions"])
        s_mean, s_std = summarize(vals["success_rates"])
        agg["per_inertia"][inertia] = {
            "mean_return_mean": r_mean,
            "mean_return_std": r_std,
            "mean_completion_mean": c_mean,
            "mean_completion_std": c_std,
            "success_rate_mean": s_mean,
            "success_rate_std": s_std,
            "n": len(vals["returns"]),
        }

    # --- per group ---
    for gtype, vals in per_group_vals.items():
        r_mean, r_std = summarize(vals["returns"])
        c_mean, c_std = summarize(vals["completions"])
        s_mean, s_std = summarize(vals["success_rates"])
        agg["per_group"][gtype] = {
            "mean_return_mean": r_mean,
            "mean_return_std": r_std,
            "mean_completion_mean": c_mean,
            "mean_completion_std": c_std,
            "success_rate_mean": s_mean,
            "success_rate_std": s_std,
            "n": len(vals["returns"]),
        }

    # --- training curve over seeds ---
    timesteps = sorted(per_timestep_vals.keys())
    curve_out = {
        "timesteps": [],
        "mean_return": [],
        "mean_return_std": [],
        "mean_return_ci": [],
        "mean_completion": [],
        "mean_completion_std": [],
        "mean_completion_ci": [],
        "n_seeds": len(runs),
    }

    for t in timesteps:
        r_vals = np.array(per_timestep_vals[t]["returns"], dtype=float)
        c_vals = np.array(per_timestep_vals[t]["completions"], dtype=float)

        r_vals = r_vals[~np.isnan(r_vals)]
        c_vals = c_vals[~np.isnan(c_vals)]

        if r_vals.size > 0:
            r_mean = float(r_vals.mean())
            r_std = float(r_vals.std(ddof=1)) if r_vals.size > 1 else 0.0
            r_se = r_std / np.sqrt(r_vals.size) if r_vals.size > 1 else 0.0
            r_ci = t_critical_975(int(r_vals.size) - 1) * r_se if r_vals.size > 1 else 0.0
        else:
            r_mean = r_std = r_ci = float("nan")

        if c_vals.size > 0:
            c_mean = float(c_vals.mean())
            c_std = float(c_vals.std(ddof=1)) if c_vals.size > 1 else 0.0
            c_se = c_std / np.sqrt(c_vals.size) if c_vals.size > 1 else 0.0
            c_ci = t_critical_975(int(c_vals.size) - 1) * c_se if c_vals.size > 1 else 0.0
        else:
            c_mean = c_std = c_ci = float("nan")

        curve_out["timesteps"].append(int(t))
        curve_out["mean_return"].append(r_mean)
        curve_out["mean_return_std"].append(r_std)
        curve_out["mean_return_ci"].append(r_ci)
        curve_out["mean_completion"].append(c_mean)
        curve_out["mean_completion_std"].append(c_std)
        curve_out["mean_completion_ci"].append(c_ci)

    agg["training_curve"] = curve_out

    return agg


def run_multi_seed_experiment(
    seeds: list[int],
    base_log_dir: str,
    base_config: TrainingConfig,
):
    """
    High-level wrapper:
      - runs one training per seed
      - aggregates the final evals
      - saves aggregate results as JSON
    """
    base = Path(base_log_dir)
    base.mkdir(parents=True, exist_ok=True)

    # Create a run-specific directory so multiple runs don't overwrite each other.
    run_id = time.strftime("multi_%m%d_%H%M%S")
    run_dir = base / run_id
    # Avoid collisions if you launch multiple runs within the same second.
    if run_dir.exists():
        run_dir = base / f"{run_id}_{int(time.time() * 1000) % 1_000_000:06d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nRun directory: {run_dir}")

    all_runs = []
    for seed in seeds:
        print(f"=== Starting run for seed={seed} ===")
        run_out = run_single_seed(seed, base_log_dir=str(run_dir), base_config=base_config)
        all_runs.append(run_out)

    agg = aggregate_runs(all_runs)

    # Save aggregated results
    agg_path = run_dir / "aggregate_results.json"
    with agg_path.open("w") as f:
        json.dump(agg, f, indent=2)

    print(f"\nAggregated results saved to: {agg_path}")

    plot_multi_seed_training_and_eval(all_runs, agg, str(run_dir))
    print(f"Plots saved into: {run_dir}")

    return all_runs, agg


def plot_multi_seed_training_and_eval(
    all_runs: list[dict],
    agg: dict,
    base_log_dir: str,
):
    base = Path(base_log_dir)
    base.mkdir(parents=True, exist_ok=True)

    def running_mean(y: list[float], window: int) -> np.ndarray:
        """
        Causal running mean with the same length as input.

        Unlike np.convolve(..., mode="valid"), this does not drop the first
        (window - 1) points, so the curve doesn't appear to "start late" when
        the smoothing window is large.
        """
        arr = np.asarray(y, dtype=float)
        if window <= 1 or arr.size == 0:
            return arr

        w = int(window)
        # cumsum with a leading 0 so we can slice sums cleanly
        csum = np.cumsum(np.insert(arr, 0, 0.0))
        idx = np.arange(1, arr.size + 1)
        start = np.maximum(0, idx - w)
        sums = csum[idx] - csum[start]
        counts = idx - start
        return sums / counts

    # ---- Episode-level return curve (smoothed running average) ----
    has_episode_history = any(run["results"].get("episode_history") for run in all_runs)
    if has_episode_history:
        # Collect per-seed smoothed curves (episode return vs episode-end timesteps)
        seed_curves: list[tuple[np.ndarray, np.ndarray]] = []

        plt.figure(figsize=(7, 5))
        for run in all_runs:
            eps = run["results"].get("episode_history", [])
            if not eps:
                continue

            pts = [
                (int(pt["timesteps"]), float(pt["return"]))
                for pt in eps
                if pt.get("timesteps") is not None and pt.get("return") is not None
            ]
            if not pts:
                continue
            pts.sort(key=lambda x: x[0])
            ts = [t for t, _ in pts]
            rs = [r for _, r in pts]

            smooth_r = running_mean(rs, EPISODE_RETURN_SMOOTH_WINDOW)
            smooth_t = np.asarray(ts, dtype=int)
            seed_label = run.get("seed", None)
            label = f"seed {seed_label}" if seed_label is not None else None
            plt.plot(smooth_t, smooth_r, alpha=0.8, linewidth=1.2, label=label)
            seed_curves.append((smooth_t, np.asarray(smooth_r, dtype=float)))

        plt.xlabel("Environment steps")
        plt.ylabel(f"Episode return (running mean, window={EPISODE_RETURN_SMOOTH_WINDOW})")
        plt.title("Training episode return vs steps (smoothed, per-seed)")
        plt.grid(True, alpha=0.3)
        plt.legend(loc="best", fontsize=8, frameon=False)
        plt.tight_layout()
        plt.savefig(base / "training_episode_return_running_mean_multi_seed.png", dpi=150)
        plt.close()

        # ---- Aggregate mean + 95% CI over seeds (aligned on a shared timestep grid) ----
        if len(seed_curves) >= 2:
            # Shared grid = all episode-end timesteps observed across seeds
            grid = np.unique(np.concatenate([t for t, _ in seed_curves])).astype(int)

            # Forward-fill each seed's curve onto the grid (value at or before timestep)
            per_seed_vals = []
            for t_seed, r_seed in seed_curves:
                idx = np.searchsorted(t_seed, grid, side="right") - 1
                vals = np.where(idx >= 0, r_seed[idx], np.nan)
                per_seed_vals.append(vals)

            mat = np.vstack(per_seed_vals)  # shape: (n_seeds, n_grid)
            n = np.sum(~np.isnan(mat), axis=0)
            mean = np.nanmean(mat, axis=0)

            # sample std with ddof=1 where n>=2
            std = np.nanstd(mat, axis=0, ddof=1)
            se = np.where(n > 1, std / np.sqrt(n), 0.0)
            tcrit = np.array([t_critical_975(int(k) - 1) if k > 1 else 0.0 for k in n], dtype=float)
            ci = tcrit * se

            valid = n > 1
            if np.any(valid):
                plt.figure(figsize=(7, 5))
                plt.plot(grid[valid], mean[valid], color="C0", linewidth=1.8, label="mean episode return")
                plt.fill_between(
                    grid[valid],
                    mean[valid] - ci[valid],
                    mean[valid] + ci[valid],
                    color="C0",
                    alpha=0.2,
                    label="95% CI (t)",
                )
                plt.xlabel("Environment steps")
                plt.ylabel(f"Episode return (running mean, window={EPISODE_RETURN_SMOOTH_WINDOW})")
                plt.title("Training episode return vs steps (smoothed, mean ± 95% CI)")
                plt.grid(True, alpha=0.3)
                plt.legend(loc="best", fontsize=8, frameon=False)
                plt.tight_layout()
                plt.savefig(base / "training_episode_return_running_mean_ci_multi_seed.png", dpi=150)
                plt.close()

    curve = agg.get("training_curve", {})
    if curve and curve.get("timesteps"):
        ts = np.array(curve["timesteps"], dtype=int)
        mean_r = np.array(curve["mean_return"], dtype=float)
        ci_r = np.array(curve["mean_return_ci"], dtype=float)
        mean_c = np.array(curve["mean_completion"], dtype=float)
        ci_c = np.array(curve["mean_completion_ci"], dtype=float)

        # ---- Plot training return vs steps ----
        plt.figure(figsize=(7, 5))
        for run in all_runs:
            c = run["results"].get("training_curve", [])
            if not c:
                continue
            t_run = [pt["timesteps"] for pt in c]
            r_run = [pt["mean_return"] for pt in c]
            plt.plot(t_run, r_run, color="gray", alpha=0.3, linewidth=1)

        plt.plot(ts, mean_r, color="C0", label="mean return")
        plt.fill_between(ts, mean_r - ci_r, mean_r + ci_r, color="C0", alpha=0.2, label="95% CI")
        plt.xlabel("Environment steps")
        plt.ylabel("Window mean return")
        plt.title("Training return vs steps (multi-seed)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(base / "training_return_multi_seed.png", dpi=150)
        plt.close()

        # ---- Plot training completion vs steps ----
        if not np.all(np.isnan(mean_c)):
            plt.figure(figsize=(7, 5))
            for run in all_runs:
                c = run["results"].get("training_curve", [])
                if not c:
                    continue
                t_run = [pt["timesteps"] for pt in c]
                c_run = [
                    (pt["mean_completion"] if pt.get("mean_completion") is not None else np.nan)
                    for pt in c
                ]
                if np.all(np.isnan(c_run)):
                    continue
                plt.plot(t_run, c_run, color="gray", alpha=0.3, linewidth=1)

            plt.plot(ts, mean_c, color="C1", label="mean completion")
            plt.fill_between(ts, mean_c - ci_c, mean_c + ci_c, color="C1", alpha=0.2, label="95% CI")
            plt.xlabel("Environment steps")
            plt.ylabel("Window mean completion")
            plt.title("Training completion vs steps (multi-seed)")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(base / "training_completion_multi_seed.png", dpi=150)
            plt.close()

    # ---- Final evaluation: completion vs inertia ----
    per_inertia = agg.get("per_inertia", {})
    if per_inertia:
        inertias = sorted(per_inertia.keys())
        comp_mean = np.array([per_inertia[i]["mean_completion_mean"] for i in inertias], dtype=float)
        comp_std = np.array([per_inertia[i]["mean_completion_std"] for i in inertias], dtype=float)
        ns = np.array([per_inertia[i]["n"] for i in inertias], dtype=float)
        se = np.where(ns > 1, comp_std / np.sqrt(ns), 0.0)
        ci = np.array(
            [t_critical_975(int(n) - 1) * s if n > 1 else 0.0 for n, s in zip(ns, se)],
            dtype=float,
        )

        plt.figure(figsize=(7, 5))
        plt.errorbar(
            inertias,
            comp_mean,
            yerr=ci,
            fmt="o-",
            capsize=4,
            color="C2",
            ecolor="C2",
            label="completion (95% CI)",
        )
        plt.xlabel("Inertia")
        plt.ylabel("Completion")
        plt.title("Final evaluation: completion vs inertia (multi-seed, stochastic)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(base / "eval_completion_per_inertia_multi_seed.png", dpi=150)
        plt.close()


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    # Get configuration based on QUICK_TEST toggle
    cfg = get_config()
    seeds = get_seeds()
    log_dir = get_log_dir()
    
    # Print mode info
    mode = "QUICK TEST" if QUICK_TEST else "FULL EXPERIMENT"
    print("=" * 60)
    print(f"MODE: {mode}")
    print("=" * 60)
    print(f"  Seeds:              {seeds}")
    print(f"  Total timesteps:    {cfg.total_timesteps:,}")
    print(f"  Eval episodes:      {cfg.n_eval_episodes}")
    print(f"  Train maps:         {cfg.n_train_maps}")
    print(f"  Test maps:          {cfg.n_test_maps}")
    print(f"  Eval frequency:     {cfg.eval_freq:,}")
    print(f"  Print frequency:    {cfg.print_freq:,}")
    print(f"  Checkpoint freq:    {cfg.checkpoint_freq:,}")
    print(f"  Parallel envs:      {cfg.n_envs}")
    print(f"  Log directory:      {log_dir}")
    print("=" * 60)
    
    # Run experiment
    all_runs, agg = run_multi_seed_experiment(
        seeds=seeds,
        base_log_dir=log_dir,
        base_config=cfg,
    )
    
    # Print results
    print("\nAggregate per-inertia results (stochastic):")
    for inertia, stats in sorted(agg["per_inertia"].items()):
        print(
            f"  inertia={inertia:.3f} -> "
            f"return={stats['mean_return_mean']:.2f} ± {stats['mean_return_std']:.2f}, "
            f"completion={stats['mean_completion_mean']:.2%} ± {stats['mean_completion_std']:.2%}, "
            f"success={stats['success_rate_mean']:.2%} ± {stats['success_rate_std']:.2%}, "
            f"n={stats['n']}"
        )

    print("\nAggregate per-group results (stochastic):")
    for g, stats in agg["per_group"].items():
        print(
            f"  {g:8s} -> "
            f"return={stats['mean_return_mean']:.2f} ± {stats['mean_return_std']:.2f}, "
            f"completion={stats['mean_completion_mean']:.2%} ± {stats['mean_completion_std']:.2%}, "
            f"success={stats['success_rate_mean']:.2%} ± {stats['success_rate_std']:.2%}, "
            f"n={stats['n']}"
        )
