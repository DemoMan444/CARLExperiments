# run_multi_seeds.py
from __future__ import annotations
import json
from pathlib import Path
from copy import deepcopy
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt

import numpy as np

from Inertiaevaluation import TrainingConfig, train


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

    # Optional: adjust any other variations of "initialization model" here,
    # e.g. starting from a pre-trained model, different hyperparams, etc.

    model, final_results, run_dir = train(cfg)

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
    # timestep -> {"returns": [...], "completions": [...]}
    per_timestep_vals = defaultdict(lambda: {"returns": [], "completions": []})

    for run in runs:
        # Use stochastic evaluation
        res = run["results"]["stochastic"]

        # Per-inertia
        for inertia, stats in res["per_inertia"].items():
            inertia_f = float(inertia)
            per_inertia_vals[inertia_f]["returns"].append(float(stats.get("mean_return", float("nan"))))
            per_inertia_vals[inertia_f]["completions"].append(float(stats.get("mean_completion", float("nan"))))
            per_inertia_vals[inertia_f]["success_rates"].append(float(stats.get("success_rate", float("nan"))))

        # Per generalization group (id, interp, extra, ...)
        for gtype, stats in res["per_group"].items():
            per_group_vals[gtype]["returns"].append(float(stats.get("mean_return", float("nan"))))
            per_group_vals[gtype]["completions"].append(float(stats.get("mean_completion", float("nan"))))
            per_group_vals[gtype]["success_rates"].append(float(stats.get("success_rate", float("nan"))))

        # Training curve (if present)
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

        # returns
        if r_vals.size > 0:
            r_mean = float(r_vals.mean())
            r_std = float(r_vals.std(ddof=1)) if r_vals.size > 1 else 0.0
            r_se = r_std / np.sqrt(r_vals.size) if r_vals.size > 1 else 0.0
            r_ci = 1.96 * r_se if r_vals.size > 1 else 0.0
        else:
            r_mean = r_std = r_ci = float("nan")

        # completions
        if c_vals.size > 0:
            c_mean = float(c_vals.mean())
            c_std = float(c_vals.std(ddof=1)) if c_vals.size > 1 else 0.0
            c_se = c_std / np.sqrt(c_vals.size) if c_vals.size > 1 else 0.0
            c_ci = 1.96 * c_se if c_vals.size > 1 else 0.0
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
    base_log_dir: str = "./logs/mario_inertia_multi",
):
    """
    High-level wrapper:
      - runs one training per seed
      - aggregates the final evals
      - saves aggregate results as JSON
    """
    Path(base_log_dir).mkdir(parents=True, exist_ok=True)

    # your base config for all runs
    base_cfg = TrainingConfig()
    # you can tweak these per experiment:
    base_cfg.total_timesteps = 40_000
    # base_cfg.use_context = True
    # base_cfg.test_on_train_maps = False

    all_runs = []
    for seed in seeds:
        print(f"=== Starting run for seed={seed} ===")
        run_out = run_single_seed(seed, base_log_dir=base_log_dir, base_config=base_cfg)
        all_runs.append(run_out)

    agg = aggregate_runs(all_runs)

    # Save aggregated results
    agg_path = Path(base_log_dir) / "aggregate_results.json"
    with agg_path.open("w") as f:
        json.dump(agg, f, indent=2)

    print(f"\nAggregated results saved to: {agg_path}")

    plot_multi_seed_training_and_eval(all_runs, agg, base_log_dir)
    print(f"Plots saved into: {base_log_dir}")

    return all_runs, agg

def plot_multi_seed_training_and_eval(
    all_runs: list[dict],
    agg: dict,
    base_log_dir: str,
):
    base = Path(base_log_dir)
    base.mkdir(parents=True, exist_ok=True)

    curve = agg.get("training_curve", {})
    if curve and curve.get("timesteps"):
        ts = np.array(curve["timesteps"], dtype=int)
        mean_r = np.array(curve["mean_return"], dtype=float)
        ci_r = np.array(curve["mean_return_ci"], dtype=float)
        mean_c = np.array(curve["mean_completion"], dtype=float)
        ci_c = np.array(curve["mean_completion_ci"], dtype=float)

        # ---- Plot training return vs steps ----
        plt.figure(figsize=(7, 5))
        # individual seeds (faint)
        for run in all_runs:
            c = run["results"].get("training_curve", [])
            if not c:
                continue
            t_run = [pt["timesteps"] for pt in c]
            r_run = [pt["mean_return"] for pt in c]
            plt.plot(t_run, r_run, color="gray", alpha=0.3, linewidth=1)

        # aggregate
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
        # Skip if we don't have completion data
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
        ci = 1.96 * se

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

if __name__ == "__main__":
    # Example: 5 independent runs with different initializations
    seeds = [0, 1, 2, 3, 4]
    all_runs, agg = run_multi_seed_experiment(seeds)
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
    