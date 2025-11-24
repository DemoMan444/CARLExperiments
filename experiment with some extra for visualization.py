# - Config/contexts saving
# - SB3 CSV logger (progress.csv)
# - Per-episode CSV logger (train_episodes.csv)
# - Per-eval JSONL logger (eval_results.jsonl)
# - Basic matplotlib plots saved under run_dir/figs

from __future__ import annotations
import numpy as np
import json
import time
from pathlib import Path
from typing import List, Dict, Sequence, Optional, Tuple, Any
import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass, asdict
from collections import deque, defaultdict, Counter
import math
import csv
import os

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure as sb3_configure

import torch

# Matplotlib (headless)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Optional image libs
try:
    import cv2
    _HAS_CV2 = True
except Exception:
    cv2 = None
    _HAS_CV2 = False

try:
    from PIL import Image
    _HAS_PIL = True
except Exception:
    Image = None
    _HAS_PIL = False

# CARL imports
from carl.envs.mario.pcg_smb_env import MarioEnv
from carl.envs.carl_env import CARLEnv
from carl.utils.types import Contexts
from carl.context.context_space import (
    CategoricalContextFeature,
    ContextFeature,
    UniformFloatContextFeature,
    UniformIntegerContextFeature,
)
from carl.context.selection import AbstractSelector
from carl.envs.mario.pcg_smb_env.toadgan.toad_gan import generate_level


# ---------------------------
# Simple dual logger (console + file)
# ---------------------------
class SimpleLogger:
    def __init__(self, file_path: Optional[Path] = None):
        self.file_path = Path(file_path) if file_path else None
        self.fp = None
        if self.file_path:
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            # line-buffered
            self.fp = open(self.file_path, "a", buffering=1, encoding="utf-8")

    def log(self, msg: str):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line)
        if self.fp:
            try:
                self.fp.write(line + "\n")
            except Exception:
                pass

    def close(self):
        if self.fp:
            self.fp.close()
            self.fp = None


LOGGER: Optional[SimpleLogger] = None


def set_logger(file_path: Optional[Path]):
    global LOGGER
    LOGGER = SimpleLogger(file_path)


def log(msg: str):
    if LOGGER is not None:
        LOGGER.log(msg)
    else:
        print(msg)


# ---------------------------
# Configuration
# ---------------------------
@dataclass
class TrainingConfig:
    # Context counts (will be overwritten by actual dict lengths in setup_training)
    n_train_maps: int = 5
    n_test_maps: int = 5
    level_width: int = 100

    # Training
    total_timesteps: int = 40_000
    n_envs: int = 8

    # PPO
    learning_rate: float = 2.5e-4
    n_steps: int = 512
    batch_size: int = 128
    n_epochs: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 1

    # Evaluation
    eval_freq: int = 10_000
    n_eval_episodes: int = 3

    # Logging
    log_dir: str = "./logs/mario_benchmark"

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42

    # Inertia-specific settings
    # Train on three ranges: 0.6–0.7, 0.9–1.0, 1.2–1.3
    train_inertia_ranges: Tuple[Tuple[float, float], ...] = (
        (0.6, 0.7),
        (0.9, 1.0),
        (1.2, 1.3),
    )
    # Evaluate on these exact inertia values
    eval_inertias: Tuple[float, ...] = (0.5, 1.4, 1.1)

    # Simple logging toggles
    print_freq: int = 10_000         # summary every N timesteps
    progress_window: int = 50        # moving window for mean stats
    episode_summaries: bool = True   # print line when episode finishes
    sanity_check: bool = True        # run a quick single-env sanity check
    save_final_model: bool = True   # set True if you want to save the final model


LEVEL_HEIGHT = 16


# ---------------------------
# CARLMarioEnv (no noisy prints; keeps episode_stats in info)
# ---------------------------
class CARLMarioEnv(CARLEnv):
    metadata = {
        "render_modes": ["rgb_array", "tiny_rgb_array"],
        "render_fps": 24,
    }

    def __init__(
        self,
        env: MarioEnv = None,
        contexts: Contexts | None = None,
        obs_context_features: list[str] | None = None,
        obs_context_as_dict: bool = True,
        context_selector: AbstractSelector | type[AbstractSelector] | None = None,
        context_selector_kwargs: dict | None = None,
        **kwargs,
    ):
        if env is None:
            env = MarioEnv(levels=[])
        super().__init__(
            env=env,
            contexts=contexts,
            obs_context_features=obs_context_features,
            obs_context_as_dict=obs_context_as_dict,
            context_selector=context_selector,
            context_selector_kwargs=context_selector_kwargs,
            **kwargs,
        )
        self.levels: List[str] = []
        self.episode_stats = {"steps": 0, "reward": 0.0, "completion": 0.0}

    def _update_context(self) -> None:
        self.env: MarioEnv
        self.context = CARLMarioEnv.get_context_space().insert_defaults(self.context)

        if not self.levels:
            for context in self.contexts.values():
                level_width = int(context["level_width"])
                level_index = int(context["level_index"])
                noise_seed = int(context["noise_seed"])

                level, _ = generate_level(
                    width=level_width,
                    height=LEVEL_HEIGHT,
                    level_index=level_index,
                    seed=noise_seed,
                    filter_unplayable=True,
                )
                self.levels.append(level)

        self.env.mario_state = int(self.context["mario_state"])
        self.env.mario_inertia = float(self.context["mario_inertia"])
        self.env.levels = [self.levels[self.context_id]]

    def reset(self, **kwargs):
        self.episode_stats = {"steps": 0, "reward": 0.0, "completion": 0.0}
        obs, info = super().reset(**kwargs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        # Update episode stats
        completion_from_info = float(info.get("completed", info.get("completion", 0.0)))
        self.episode_stats["steps"] += 1
        self.episode_stats["reward"] += float(reward)
        self.episode_stats["completion"] = max(self.episode_stats["completion"], completion_from_info)

        if terminated or truncated:
            info["episode_stats"] = {
                "steps": int(self.episode_stats["steps"]),
                "reward": float(self.episode_stats["reward"]),
                "completion": float(self.episode_stats["completion"]),
            }
            info["context_id"] = int(self.context_id)
            info["level_width"] = int(self.context["level_width"])

        return obs, reward, terminated, truncated, info

    @staticmethod
    def get_context_features() -> dict[str, ContextFeature]:
        return {
            "level_width": UniformIntegerContextFeature(
                "level_width", 50, 500, default_value=200
            ),
            "level_index": CategoricalContextFeature(
                "level_index", choices=np.arange(0, 14), default_value=0
            ),
            "noise_seed": UniformIntegerContextFeature(
                "noise_seed", 0, 2**31 - 1, default_value=0
            ),
            "mario_state": CategoricalContextFeature(
                "mario_state", choices=[0, 1, 2], default_value=0
            ),
            "mario_inertia": UniformFloatContextFeature(
                "mario_inertia", lower=0.5, upper=1.5, default_value=0.89
            ),
        }


# ---------------------------
# Enhanced Observation Wrapper: grayscale + resize + CHW + frame stack
# ---------------------------
class MarioObsAdapter(gym.ObservationWrapper):
    def __init__(
        self,
        env: gym.Env,
        ctx_keys: Sequence[str] = ("level_width", "level_index", "mario_state", "mario_inertia"),
        normalize_context: bool = True,
        frame_stack: int = 4,
        to_gray: bool = True,
        resize_to: Tuple[int, int] = (84, 84),  # (W, H)
    ):
        super().__init__(env)
        self.ctx_keys = list(ctx_keys)
        self.normalize_context = normalize_context
        self.frame_stack = frame_stack
        self.to_gray = to_gray
        self.resize_to = resize_to

        # context normalization ranges
        self._ctx_norm: Dict[str, Tuple[float, float]] = {}
        features = CARLMarioEnv.get_context_features()
        for k in self.ctx_keys:
            feat = features[k]
            if hasattr(feat, "lower") and hasattr(feat, "upper"):
                lo, hi = float(feat.lower), float(feat.upper)
            elif hasattr(feat, "choices"):
                choices = list(feat.choices)
                lo, hi = float(min(choices)), float(max(choices))
            else:
                lo, hi = 0.0, 1.0
            self._ctx_norm[k] = (lo, hi)

        # Find the image key (best-effort)
        self._img_key_candidates = ["obs", "image", "img", "rgb", "frame"]
        self._img_key = None
        if isinstance(self.env.observation_space, spaces.Dict):
            for k in self._img_key_candidates:
                sp = self.env.observation_space.spaces.get(k, None)
                if isinstance(sp, spaces.Box) and sp.shape and len(sp.shape) == 3:
                    self._img_key = k
                    break

        # Build output observation space: CHW uint8 + optional ctx vector
        if self.resize_to is None:
            raise ValueError("Please set resize_to=(W,H) to define fixed image size.")
        out_c = 1 if self.to_gray else 3
        W, H = self.resize_to

        img_space = spaces.Box(
            low=0,
            high=255,
            shape=(out_c * self.frame_stack, H, W),
            dtype=np.uint8,
        )

        if len(self.ctx_keys) == 0:
            # No context branch at all
            self.observation_space = spaces.Dict({"img": img_space})
        else:
            ctx_space = spaces.Box(
                low=0.0,
                high=1.0,
                shape=(len(self.ctx_keys),),
                dtype=np.float32,
            )
            self.observation_space = spaces.Dict({"img": img_space, "ctx": ctx_space})

        self.frames: deque[np.ndarray] = deque(maxlen=self.frame_stack)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.frames.clear()
        return self.observation(obs), info

    def _extract_img(self, obs):
        if isinstance(obs, dict):
            if self._img_key is not None and self._img_key in obs:
                return obs[self._img_key]
            for k in self._img_key_candidates:
                if k in obs and isinstance(obs[k], np.ndarray) and obs[k].ndim >= 2:
                    self._img_key = k
                    return obs[k]
            for v in obs.values():
                if isinstance(v, np.ndarray) and v.ndim >= 2:
                    return v
        elif isinstance(obs, np.ndarray) and obs.ndim >= 2:
            return obs
        raise RuntimeError("Could not find image-like entry in observation.")

    def _to_hwc(self, img: np.ndarray) -> np.ndarray:
        # If CHW -> HWC, else assume HWC already
        if img.ndim == 3 and img.shape[0] in (1, 3) and img.shape[2] not in (1, 3):
            return np.transpose(img, (1, 2, 0))
        return img

    def _resize_hwc(self, x: np.ndarray) -> np.ndarray:
        W, H = self.resize_to
        if _HAS_CV2:
            return cv2.resize(x, (W, H), interpolation=cv2.INTER_AREA)
        elif _HAS_PIL:
            pil = Image.fromarray(x)
            return np.array(pil.resize((W, H), resample=Image.BILINEAR))
        else:
            # Fallback: naive nearest-neighbor resizing
            h0, w0 = x.shape[:2]
            ys = (np.linspace(0, h0 - 1, H)).astype(np.int32)
            xs = (np.linspace(0, w0 - 1, W)).astype(np.int32)
            return x[ys][:, xs]

    def _to_chw_uint8(self, x: np.ndarray) -> np.ndarray:
        # x: either HxW (gray) or HxWxC
        if x.ndim == 2:
            y = x[np.newaxis, :, :]
        else:
            y = np.transpose(x, (2, 0, 1))
        return y.astype(np.uint8)

    def _process_img(self, img: np.ndarray) -> np.ndarray:
        x = self._to_hwc(img)  # HWC
        # Ensure uint8 range for CV/PIL ops
        if x.dtype != np.uint8:
            x = np.clip(x, 0, 255).astype(np.uint8)

        # Grayscale
        if self.to_gray:
            if x.ndim == 3 and x.shape[2] == 3:
                if _HAS_CV2:
                    x = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
                else:
                    # Luma transform
                    x = np.dot(x[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
            elif x.ndim == 3 and x.shape[2] == 1:
                x = x[..., 0]
        # Resize
        if self.resize_to is not None:
            x = self._resize_hwc(x)
        # CHW uint8
        y = self._to_chw_uint8(x)
        return y

    def _extract_context(self) -> Dict[str, float]:
        if hasattr(self.env, "context") and isinstance(self.env.context, dict):
            return self.env.context
        return {}

    def _ctx_to_vec(self, ctx_dict: Dict[str, float]) -> np.ndarray:
        vec = []
        for k in self.ctx_keys:
            x = float(ctx_dict.get(k, 0.0))
            if self.normalize_context:
                lo, hi = self._ctx_norm[k]
                x = (x - lo) / (hi - lo) if hi > lo else 0.0
            vec.append(x)
        return np.array(vec, dtype=np.float32)

    def observation(self, obs):
        img = self._extract_img(obs)
        img = self._process_img(img)  # CHW uint8 (C,H,W)
        # frame stack along channel axis
        self.frames.append(img)
        while len(self.frames) < self.frame_stack:
            self.frames.append(img)
        img_stack = np.concatenate(list(self.frames), axis=0)  # (C*k, H, W)

        if len(self.ctx_keys) == 0:
            # No context branch
            return {"img": img_stack}
        else:
            ctx_vec = self._ctx_to_vec(self._extract_context())
            return {"img": img_stack, "ctx": ctx_vec}


# ---------------------------
# Action repeat (frame skip)
# ---------------------------
class ActionRepeat(gym.Wrapper):
    def __init__(self, env, repeat: int = 4, max_pool: bool = True):
        super().__init__(env)
        self.repeat = repeat
        self.max_pool = max_pool

    def step(self, action):
        total_r = 0.0
        terminated = truncated = False
        last_obs = None
        pooled_img = None
        info = {}
        for _ in range(self.repeat):
            obs, r, term, trunc, info = self.env.step(action)
            total_r += float(r)
            if self.max_pool and isinstance(obs, dict) and "img" in obs:
                pooled_img = obs["img"] if pooled_img is None else np.maximum(pooled_img, obs["img"])
            last_obs = obs
            terminated = term or terminated
            truncated = trunc or truncated
            if term or trunc:
                break
        if self.max_pool and isinstance(last_obs, dict) and "img" in last_obs and pooled_img is not None:
            obs = dict(last_obs)
            obs["img"] = pooled_img
        else:
            obs = last_obs
        return obs, total_r, terminated, truncated, info


# ---------------------------
# Context generation (only varying mario_inertia)
# ---------------------------
def build_inertia_train_contexts(
    n_maps: int,
    level_width: int,
    inertia_ranges: Sequence[Tuple[float, float]],
    seed: int = 42,
) -> Dict[int, Dict[str, Any]]:
    """
    Build training contexts as a *cross product* of:
      - n_maps different maps (different noise_seed each)
      - the given inertia ranges

    Total contexts = n_maps * len(inertia_ranges).
    """
    np.random.seed(seed)
    ctxs: Dict[int, Dict[str, Any]] = {}
    ctx_id = 0

    for map_idx in range(n_maps):
        # one distinct map per map_idx
        noise_seed = seed + map_idx

        for (low, high) in inertia_ranges:
            inertia = float(np.random.uniform(low, high))

            ctxs[ctx_id] = {
                "level_width": int(level_width),
                "level_index": 0,          # keep the same level_index
                "noise_seed": noise_seed,  # this fixes the map identity
                "mario_state": 0,
                "mario_inertia": inertia,
            }
            ctx_id += 1

    return ctxs

def build_inertia_test_contexts(
    n_maps: int,
    inertia_values: Sequence[float],
    level_width: int,
    seed: int = 43,
) -> Dict[int, Dict[str, Any]]:
    """
    Build test contexts as a *cross product* of:
      - n_maps different test maps (different noise_seed each, offset from train)
      - the given inertia values

    Total contexts = n_maps * len(inertia_values).
    """
    ctxs: Dict[int, Dict[str, Any]] = {}
    ctx_id = 0

    for map_idx in range(n_maps):
        # ensure test maps are different from train maps
        noise_seed = seed + 1000 + map_idx

        for inertia in inertia_values:
            ctxs[ctx_id] = {
                "level_width": int(level_width),
                "level_index": 0,
                "noise_seed": noise_seed,
                "mario_state": 0,
                "mario_inertia": float(inertia),
            }
            ctx_id += 1

    return ctxs

# ---------------------------
# Simple callbacks (console + txt via log())
# ---------------------------
class TrainingPrinterCallback(BaseCallback):
    """
    Enhanced training printer:
    - Keeps the original episode-weighted sliding-window metrics
    - Adds uniform-per-context rolling metrics (each context weighted equally)
    - Prints the context distribution over the last window to reveal bias
    """
    def __init__(self, print_freq=10_000, window=50, episode_summaries=True):
        super().__init__()
        self.print_freq = print_freq
        self.window = window
        self.episode_summaries = episode_summaries
        self._last_print_ts = 0
        self._last_mean_return = None

        self.ep_returns = deque(maxlen=self.window)
        self.ep_lengths = deque(maxlen=self.window)
        self.ep_completions = deque(maxlen=self.window)
        self.total_episodes = 0

        # For uniform-per-context metrics
        self.ctx_returns: Dict[int, deque] = defaultdict(lambda: deque(maxlen=self.window))
        self.ctx_completions: Dict[int, deque] = defaultdict(lambda: deque(maxlen=self.window))
        # For distribution of contexts in the window
        self.last_ctx_ids: deque = deque(maxlen=self.window)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if info is None:
                continue
            if "episode" in info:
                ep_r = float(info["episode"].get("r", 0.0))
                ep_l = int(info["episode"].get("l", 0))
                completion = None
                if "episode_stats" in info:
                    completion = float(info["episode_stats"].get("completion", 0.0))

                self.ep_returns.append(ep_r)
                self.ep_lengths.append(ep_l)
                if completion is not None:
                    self.ep_completions.append(completion)
                self.total_episodes += 1

                ctx_id = info.get("context_id", None)
                if ctx_id is not None:
                    self.ctx_returns[ctx_id].append(ep_r)
                    if completion is not None:
                        self.ctx_completions[ctx_id].append(completion)
                    self.last_ctx_ids.append(ctx_id)

                if self.episode_summaries:
                    ctx_str = f" ctx={ctx_id if ctx_id is not None else 'NA'}"
                    comp_str = f" comp={completion:.2%}" if completion is not None else ""
                    log(f"Episode done: r={ep_r:.2f} len={ep_l}{comp_str}{ctx_str}")

        if (self.num_timesteps - self._last_print_ts) >= self.print_freq:
            mean_r = float(np.mean(self.ep_returns)) if len(self.ep_returns) > 0 else float("nan")
            mean_l = float(np.mean(self.ep_lengths)) if len(self.ep_lengths) > 0 else float("nan")
            mean_c = float(np.mean(self.ep_completions)) if len(self.ep_completions) > 0 else float("nan")

            # Uniform-per-context means (equal weight to each context)
            per_ctx_r = [np.mean(v) for v in self.ctx_returns.values() if len(v) > 0]
            uniform_ctx_mean_r = float(np.mean(per_ctx_r)) if len(per_ctx_r) > 0 else float("nan")
            per_ctx_c = [np.mean(v) for v in self.ctx_completions.values() if len(v) > 0]
            uniform_ctx_mean_c = float(np.mean(per_ctx_c)) if len(per_ctx_c) > 0 else float("nan")

            # Context distribution in the last window episodes
            counts = Counter(self.last_ctx_ids)
            total = sum(counts.values())
            dist_str = "NA"
            if total > 0:
                # Sort by context id for consistency
                parts = [f"{k}:{(v/total):.0%}" for k, v in sorted(counts.items())]
                dist_str = " ".join(parts)

            arrow = ""
            if self._last_mean_return is not None and not np.isnan(mean_r):
                if mean_r > self._last_mean_return + 1e-6:
                    arrow = "↑"
                elif mean_r < self._last_mean_return - 1e-6:
                    arrow = "↓"
                else:
                    arrow = "→"

            log("-" * 64)
            log(f"Train @ {self.num_timesteps:,} steps")
            log(f"  Episodes (total): {self.total_episodes:,}")
            log(f"  Window({self.window}) mean return (episode-weighted): {mean_r:.2f} {arrow}")
            log(f"  Window({self.window}) mean length: {mean_l:.1f}")
            if not np.isnan(mean_c):
                log(f"  Window({self.window}) mean completion: {mean_c:.2%}")
            log(f"  Uniform-per-context mean return: {uniform_ctx_mean_r:.2f}")
            if not np.isnan(uniform_ctx_mean_c):
                log(f"  Uniform-per-context mean completion: {uniform_ctx_mean_c:.2%}")
            log(f"  Context distribution (last {self.window} eps): {dist_str}")
            log("-" * 64)

            self._last_mean_return = mean_r
            self._last_print_ts = self.num_timesteps

        return True


class ContextualEvalCallback(BaseCallback):
    """
    Evaluate per-context with configurable deterministic vs stochastic policy.
    Also write results to JSONL (one record per context at each eval).
    """
    def __init__(
        self,
        eval_env: gym.Env,
        eval_contexts: List[int],
        eval_freq: int = 10_000,
        n_eval_episodes: int = 2,
        deterministic: bool = True,
        tag: str = "",
        results_path: Optional[Path] = None,
        ctx_lookup: Optional[Dict[int, Dict[str, Any]]] = None,
    ):
        super().__init__()
        self.eval_env = eval_env
        self.eval_contexts = eval_contexts
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.deterministic = deterministic
        self.tag = tag
        self._last_eval_ts = 0
        self.results_path = Path(results_path) if results_path else None
        self.ctx_lookup = ctx_lookup or {}

        self._fp = None

    def _on_training_start(self) -> None:
        if self.results_path:
            self.results_path.parent.mkdir(parents=True, exist_ok=True)
            # Append JSONL
            self._fp = open(self.results_path, "a", encoding="utf-8")

    def _on_training_end(self) -> None:
        if self._fp:
            self._fp.close()
            self._fp = None

    def _on_step(self) -> bool:
        if (self.num_timesteps - self._last_eval_ts) >= self.eval_freq:
            results = self._evaluate()
            self._last_eval_ts = self.num_timesteps

            all_returns = []
            all_completions = []
            for ctx_results in results.values():
                all_returns.extend(ctx_results["returns"])
                all_completions.extend(ctx_results["completions"])

            mean_return = float(np.mean(all_returns)) if len(all_returns) > 0 else float("nan")
            mean_completion = float(np.mean(all_completions)) if len(all_completions) > 0 else float("nan")

            log("=" * 50)
            log(f"Eval{(' ['+self.tag+']') if self.tag else ''} @ Timestep {self.num_timesteps:,}  (deterministic={self.deterministic})")
            log(f"  Mean return: {mean_return:.2f}")
            log(f"  Mean completion: {mean_completion:.2%}")
            log("Per-context eval (mean over episodes):")
            for ctx_id, ctx_results in results.items():
                rctx = float(np.mean(ctx_results["returns"])) if len(ctx_results["returns"]) > 0 else float("nan")
                cctx = float(np.mean(ctx_results["completions"])) if len(ctx_results["completions"]) > 0 else float("nan")
                log(f"  ctx={ctx_id}: return={rctx:.2f}  completion={cctx:.2%}")
            log("=" * 50)
            with torch.no_grad():
                total_abs = 0.0
                for p in self.model.policy.parameters():
                    total_abs += p.abs().sum().item()
            log(f"Policy param |abs| sum: {total_abs:.3e}")
            if hasattr(self.model.policy, 'optimizer'):
                for i, param_group in enumerate(self.model.policy.optimizer.param_groups):
                    log(f"Optimizer lr group {i}: {param_group['lr']:.6f}")

            # Log gradient norms during training
            total_grad_norm = 0.0
            for p in self.model.policy.parameters():
                if p.grad is not None:
                    total_grad_norm += p.grad.norm().item() ** 2
            total_grad_norm = total_grad_norm ** 0.5
            log(f"Total gradient norm: {total_grad_norm:.3e}")

            # NEW: write JSONL
            if self._fp:
                for ctx_id, ctx_results in results.items():
                    inertia = float(self.ctx_lookup.get(ctx_id, {}).get("mario_inertia", float("nan")))
                    arr_r = np.array(ctx_results["returns"], dtype=float)
                    arr_c = np.array(ctx_results["completions"], dtype=float)
                    record = {
                        "timestep": int(self.num_timesteps),
                        "tag": self.tag,
                        "deterministic": bool(self.deterministic),
                        "context_id": int(ctx_id),
                        "mario_inertia": inertia,
                        "return_mean": float(np.nanmean(arr_r)) if arr_r.size else float("nan"),
                        "return_std": float(np.nanstd(arr_r)) if arr_r.size else float("nan"),
                        "completion_mean": float(np.nanmean(arr_c)) if arr_c.size else float("nan"),
                        "completion_std": float(np.nanstd(arr_c)) if arr_c.size else float("nan"),
                        "length_mean": float(np.mean(ctx_results["lengths"])) if len(ctx_results["lengths"]) else float("nan"),
                        "n_episodes": int(len(ctx_results["returns"])),
                    }
                    self._fp.write(json.dumps(record) + "\n")
                self._fp.flush()

        return True

    def _evaluate(self) -> Dict[int, Dict[str, List[float]]]:
        results = {}

        for ctx_id in self.eval_contexts:
            ctx_returns = []
            ctx_completions = []
            ctx_lengths = []

            for _ in range(self.n_eval_episodes):
                # Reset with context
                try:
                    obs, info = self.eval_env.reset(options={"context_id": ctx_id})
                except Exception:
                    self.eval_env.unwrapped.context_id = ctx_id
                    if hasattr(self.eval_env.unwrapped, "_update_context"):
                        self.eval_env.unwrapped._update_context()
                    obs, info = self.eval_env.reset()

                done = False
                total_reward = 0.0
                steps = 0
                max_steps = 5000

                while not done and steps < max_steps:
                    action, _ = self.model.predict(obs, deterministic=self.deterministic)
                    obs, reward, terminated, truncated, info = self.eval_env.step(action)
                    total_reward += float(reward)
                    steps += 1
                    done = terminated or truncated

                ctx_returns.append(total_reward)
                ctx_lengths.append(steps)
                completion = 0.0
                if "episode_stats" in info:
                    completion = info["episode_stats"].get("completion", 0.0)
                ctx_completions.append(completion)

            results[ctx_id] = {
                "returns": ctx_returns,
                "completions": ctx_completions,
                "lengths": ctx_lengths,
            }

        return results


# ---------------------------
# Per-episode CSV Logger Callback
# ---------------------------
class CSVEpisodeLoggerCallback(BaseCallback):
    def __init__(self, out_path: Path, ctx_lookup: Dict[int, Dict[str, Any]] | None = None):
        super().__init__()
        self.out_path = Path(out_path)
        self.ctx_lookup = ctx_lookup or {}
        self.fp = None
        self.writer = None
        self.ep_idx = 0

    def _on_training_start(self) -> None:
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        exists = self.out_path.exists()
        self.fp = open(self.out_path, "a", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(
            self.fp,
            fieldnames=[
                "timestep",
                "wall_time",
                "episode_idx",
                "context_id",
                "mario_inertia",
                "return",
                "length",
                "completion",
            ],
        )
        if not exists:
            self.writer.writeheader()

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        now = time.time()
        for info in infos:
            if info is None:
                continue
            if "episode" in info:
                ep_r = float(info["episode"].get("r", 0.0))
                ep_l = int(info["episode"].get("l", 0))
                completion = float(info.get("episode_stats", {}).get("completion", 0.0))
                ctx_id = info.get("context_id", None)
                inertia = float("nan")
                if ctx_id is not None:
                    inertia = float(self.ctx_lookup.get(int(ctx_id), {}).get("mario_inertia", float("nan")))
                row = {
                    "timestep": int(self.num_timesteps),
                    "wall_time": now,
                    "episode_idx": self.ep_idx,
                    "context_id": int(ctx_id) if ctx_id is not None else -1,
                    "mario_inertia": inertia,
                    "return": ep_r,
                    "length": ep_l,
                    "completion": completion,
                }
                self.writer.writerow(row)
                self.fp.flush()
                self.ep_idx += 1
        return True

    def _on_training_end(self) -> None:
        if self.fp:
            self.fp.close()
            self.fp = None


# ---------------------------
# Training Utilities
# ---------------------------
def make_env_fn(contexts: Dict[int, Dict], ctx_keys: List[str], monitor: bool = True):
    def _init():
        env = CARLMarioEnv(
            contexts=contexts,
            obs_context_features=ctx_keys,
            obs_context_as_dict=True,
        )
        # Vision preprocessing (grayscale + resize + CHW + frame stack)
        env = MarioObsAdapter(
            env,
            ctx_keys=ctx_keys,
            normalize_context=True,
            frame_stack=4,
            to_gray=True,
            resize_to=(84, 84),
        )
        # Action repeat (frame skip)
        env = ActionRepeat(env, repeat=4, max_pool=True)
        if monitor:
            env = Monitor(env)
        return env
    return _init


def sanity_check_env(contexts: Dict[int, Dict], ctx_keys: List[str]) -> None:
    log("Running sanity check on a single env...")
    env = make_env_fn(contexts, ctx_keys, monitor=False)()
    try:
        log(f"Action space: {env.action_space}")
        log(f"Observation space: {env.observation_space}")

        obs, info = env.reset()
        if isinstance(obs, dict):
            keys = list(obs.keys())
            log(f"Obs keys: {keys}")
            if "img" in obs:
                log(f"  img shape: {getattr(obs['img'], 'shape', None)} dtype: {obs['img'].dtype}")
            if "ctx" in obs:
                log(f"  ctx shape: {getattr(obs['ctx'], 'shape', None)} dtype: {obs['ctx'].dtype}")
        else:
            log(f"Obs type: {type(obs)}")

        a = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(a)
        log(f"One random step -> reward={float(reward):.3f}, done={terminated or truncated}, info keys={list(info.keys())}")
    finally:
        try:
            env.close()
        except Exception as e:
            log(f"Warning: env.close() raised: {e}. Ignoring.")
    log("Sanity check OK.")


def linear_schedule(initial_value: float):
    # SB3 schedule: progress goes from 1 (start) to 0 (end)
    def f(progress: float):
        return initial_value * progress
    return f


def setup_training(config: TrainingConfig):
    # ---------------------------------------------------------
    # 1. Create a Unique Run Directory based on time
    # ---------------------------------------------------------
    # Format: inertia_MMDD_HHMM (e.g., inertia_1025_1430)
    run_id = "inertia_" + time.strftime("%m%d_%H%M")
    
    # The base directory
    base_path = Path(config.log_dir)
    
    # The specific folder for THIS run
    run_dir = base_path / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # 2. Set up the Text Logger in the unique folder
    # ---------------------------------------------------------
    set_logger(run_dir / "training_output.txt")
    log(f"Initialized run directory: {run_dir}")

    # ---------------------------------------------------------
    # 3. Standard Setup (Seeds, Contexts, Envs)
    # ---------------------------------------------------------
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    # Create contexts
    log("Generating training contexts (varying only mario_inertia)...")
    train_contexts = build_inertia_train_contexts(
        n_maps=config.n_train_maps,
        level_width=config.level_width,
        inertia_ranges=config.train_inertia_ranges,
        seed=config.seed + 1,
    )

    log("Generating test contexts (OOD mario_inertia values)...")
    test_contexts = build_inertia_test_contexts(
        n_maps=config.n_test_maps,
        inertia_values=config.eval_inertias,
        level_width=config.level_width,
        seed=config.seed + 1,
    )

    config.n_train_maps = len(train_contexts)
    config.n_test_maps = len(test_contexts)
    ctx_keys = ["mario_inertia"]

    # Save config and contexts to the run directory
    (run_dir / "config.json").write_text(json.dumps(asdict(config), indent=2))
    (run_dir / "train_contexts.json").write_text(json.dumps(train_contexts, indent=2))
    (run_dir / "test_contexts.json").write_text(json.dumps(test_contexts, indent=2))

    log("Training contexts (all):")
    for k in sorted(train_contexts.keys()):
        log(f"  id={k}: {train_contexts[k]}")
    log("Test contexts (all):")
    for k in sorted(test_contexts.keys()):
        log(f"  id={k}: {test_contexts[k]}")

    if config.sanity_check:
        sanity_check_env(train_contexts, ctx_keys)

    log(f"Creating {config.n_envs} parallel environments...")
    env_fns = [make_env_fn(train_contexts, ctx_keys) for _ in range(config.n_envs)]

    if config.n_envs == 1:
        train_env = DummyVecEnv(env_fns)
    else:
        train_env = SubprocVecEnv(env_fns)

    train_env = VecNormalize(train_env, norm_obs=False, norm_reward=True, clip_reward=10.0)

    log("Creating evaluation environment...")
    eval_env = make_env_fn(test_contexts, ctx_keys)()

    # ---------------------------------------------------------
    # 4. Model Setup
    # ---------------------------------------------------------
    try:
        from stable_baselines3.common.torch_layers import CombinedExtractor
    except ImportError:
        from stable_baselines3.common.torch_layers import MultiInputExtractor as CombinedExtractor

    policy_kwargs = dict(
        features_extractor_class=CombinedExtractor,
        features_extractor_kwargs=dict(cnn_output_dim=256),
        net_arch=dict(pi=[256], vf=[256]),
        activation_fn=torch.nn.ReLU,
        ortho_init=False,
        normalize_images=True,
    )

    log("Creating PPO model...")
    
    # We point TensorBoard to the unique run_dir. 
    # SB3 will create a subfolder inside it (e.g., run_dir/PPO_1)
    model = PPO(
        policy="MultiInputPolicy",
        env=train_env,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        learning_rate=linear_schedule(config.learning_rate),
        ent_coef=config.ent_coef,
        gamma=config.gamma,
        n_epochs=config.n_epochs,
        clip_range=config.clip_range,
        gae_lambda=config.gae_lambda,
        max_grad_norm=config.max_grad_norm,
        vf_coef=config.vf_coef,
        target_kl=0.01,
        seed=config.seed,
        device=config.device,
        policy_kwargs=policy_kwargs,
        tensorboard_log=str(run_dir),  # <--- Logs go into the unique folder
        verbose=0,
    )

    # Configure SB3 logger to also output CSV (progress.csv) and stdout
    new_logger = sb3_configure(str(run_dir), ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)

    eval_callback_greedy = ContextualEvalCallback(
        eval_env=eval_env,
        eval_contexts=list(test_contexts.keys()),
        eval_freq=100_000,
        n_eval_episodes=1,
        deterministic=True,
        tag="greedy",
        results_path=run_dir / "eval_results.jsonl",
        ctx_lookup=test_contexts,
    )
    eval_callback_stoch = ContextualEvalCallback(
        eval_env=eval_env,
        eval_contexts=list(test_contexts.keys()),
        eval_freq=config.eval_freq,
        n_eval_episodes=config.n_eval_episodes,
        deterministic=False,
        tag="stochastic",
        results_path=run_dir / "eval_results.jsonl",
        ctx_lookup=test_contexts,
    )

    train_printer = TrainingPrinterCallback(
        print_freq=config.print_freq,
        window=config.progress_window,
        episode_summaries=config.episode_summaries,
    )

    csv_ep_logger = CSVEpisodeLoggerCallback(out_path=run_dir / "train_episodes.csv",
                                             ctx_lookup=train_contexts)

    callbacks = [train_printer, eval_callback_greedy, eval_callback_stoch, csv_ep_logger]

    # Return run_dir so train() knows where to save the final model
    return model, train_env, eval_env, callbacks, train_contexts, test_contexts, run_dir


def train(config: TrainingConfig):
    log("=" * 50)
    log("SETTING UP TRAINING")
    log("=" * 50)
    
    # Unpack the new return value (run_dir)
    model, train_env, eval_env, callbacks, train_contexts, test_contexts, run_dir = setup_training(config)

    log("Training Configuration:")
    log(f"  Train contexts: {config.n_train_maps}")
    log(f"  Test contexts: {config.n_test_maps}")
    log(f"  Parallel envs: {config.n_envs}")
    log(f"  Level width: {config.level_width}")
    log(f"  Total timesteps: {config.total_timesteps:,}")
    log(f"  Device: {config.device}")
    log(f"  Run Directory: {run_dir}")

    log("=" * 50)
    log("STARTING TRAINING")
    log("=" * 50)

    model.learn(
        total_timesteps=config.total_timesteps,
        callback=callbacks,
        progress_bar=False,
        reset_num_timesteps=False,
        tb_log_name="tb"
    )

    log("=" * 50)
    log("FINAL EVALUATION")
    log("=" * 50)
    final_greedy = evaluate_final(model, eval_env, test_contexts, n_episodes=config.n_eval_episodes, deterministic=True, tag="greedy")
    final_stoch = evaluate_final(model, eval_env, test_contexts, n_episodes=config.n_eval_episodes, deterministic=False, tag="stochastic")
    final_results = {"greedy": final_greedy, "stochastic": final_stoch}

    if config.save_final_model:
        # Save the model inside the unique run directory
        model_path = run_dir / "final_model.zip"
        model.save(model_path)
        
        # Optionally save VecNormalize stats
        try:
            train_env.save(run_dir / "vec_normalize.pkl")
        except Exception:
            pass
        log(f"Model saved to {model_path}")

    train_env.close()
    eval_env.close()
    
    # Generate basic plots
    try:
        generate_basic_plots(run_dir)
        log(f"Saved figures under: {run_dir / 'figs'}")
    except Exception as e:
        log(f"Plot generation failed: {e}")

    return model, final_results, run_dir

def evaluate_final(
    model: PPO,
    eval_env: gym.Env,
    contexts: Dict[int, Dict],
    n_episodes: int = 2,
    deterministic: bool = True,
    tag: str = "",
) -> Dict[str, Any]:
    all_results = []

    for ctx_id, ctx_params in contexts.items():
        for ep in range(n_episodes):
            try:
                obs, info = eval_env.reset(options={"context_id": ctx_id})
            except Exception:
                eval_env.unwrapped.context_id = ctx_id
                if hasattr(eval_env.unwrapped, "_update_context"):
                    eval_env.unwrapped._update_context()
                obs, info = eval_env.reset()

            done = False
            total_reward = 0.0
            steps = 0
            max_steps = 5000

            while not done and steps < max_steps:
                action, _ = model.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                total_reward += float(reward)
                steps += 1
                done = terminated or truncated

            completion = 0.0
            if "episode_stats" in info:
                completion = info["episode_stats"].get("completion", 0.0)

            all_results.append({
                "context_id": ctx_id,
                "context_params": ctx_params,
                "episode": ep,
                "return": total_reward,
                "length": steps,
                "completion": completion,
            })

    returns = [r["return"] for r in all_results]
    completions = [r["completion"] for r in all_results]
    lengths = [r["length"] for r in all_results]

    summary = {
        "mean_return": float(np.mean(returns)) if len(returns) > 0 else float("nan"),
        "std_return": float(np.std(returns)) if len(returns) > 0 else float("nan"),
        "mean_completion": float(np.mean(completions)) if len(completions) > 0 else float("nan"),
        "std_completion": float(np.std(completions)) if len(completions) > 0 else float("nan"),
        "mean_length": float(np.mean(lengths)) if len(lengths) > 0 else float("nan"),
        "success_rate": float(np.mean([c > 0.9 for c in completions])) if len(completions) > 0 else float("nan"),
        "episodes": all_results,
        "deterministic": deterministic,
        "tag": tag,
    }

    log("=" * 50)
    log(f"FINAL EVALUATION RESULTS{(' ['+tag+']') if tag else ''} (deterministic={deterministic})")
    log("=" * 50)
    log(f"Mean Return: {summary['mean_return']:.2f} ± {summary['std_return']:.2f}")
    log(f"Mean Completion: {summary['mean_completion']:.2%} ± {summary['std_completion']:.2%}")
    log(f"Success Rate (>90%): {summary['success_rate']:.2%}")
    log(f"Mean Episode Length: {summary['mean_length']:.1f}")
    log("=" * 50)

    return summary


# ---------------------------
# Basic plotting utilities (no pandas)
# ---------------------------
def _read_csv_dicts(path: Path) -> List[Dict[str, str]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except Exception:
        return []

def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                out.append(json.loads(line))
    except Exception:
        pass
    return out

def _to_float_list(rows: List[Dict[str, str]], key: str) -> List[float]:
    out = []
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

def _rolling_mean(values: List[float], window: int) -> np.ndarray:
    a = np.asarray(values, dtype=float)
    n = a.size
    if n == 0:
        return a
    # min_periods=1
    cs = np.cumsum(a)
    out = np.empty(n, dtype=float)
    for i in range(n):
        start = max(0, i - window + 1)
        s = cs[i] - (cs[start - 1] if start > 0 else 0.0)
        out[i] = s / (i - start + 1)
    return out

def generate_basic_plots(run_dir: Path):
    run_dir = Path(run_dir)
    figs = run_dir / "figs"
    figs.mkdir(exist_ok=True)

    # 1) SB3 progress.csv
    prog_path = run_dir / "progress.csv"
    if prog_path.exists():
        rows = _read_csv_dicts(prog_path)
        if rows:
            # X-axis: time/total_timesteps or total_timesteps or index
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
            rew_mean = _to_float_list(rows, "rollout/ep_rew_mean")
            len_mean = _to_float_list(rows, "rollout/ep_len_mean")
            if any(not math.isnan(x) for x in rew_mean) or any(not math.isnan(x) for x in len_mean):
                plt.figure(figsize=(7,4))
                if any(not math.isnan(x) for x in rew_mean):
                    plt.plot(t, rew_mean, label="ep_rew_mean")
                if any(not math.isnan(x) for x in len_mean):
                    plt.plot(t, len_mean, label="ep_len_mean")
                plt.xlabel("Total timesteps")
                plt.ylabel("Value")
                plt.title("SB3 rollout means")
                plt.legend()
                plt.tight_layout()
                plt.savefig(figs / "progress_rollout_means.png", dpi=150)
                plt.close()

            # Other training metrics if present
            for k in ["train/approx_kl","train/clip_fraction","train/entropy_loss","train/value_loss","time/fps","train/learning_rate"]:
                vals = _to_float_list(rows, k)
                if any(not math.isnan(x) for x in vals):
                    plt.figure(figsize=(7,3))
                    plt.plot(t, vals)
                    plt.xlabel("Total timesteps")
                    plt.ylabel(k)
                    plt.tight_layout()
                    plt.savefig(figs / f"{k.replace('/','_')}.png", dpi=150)
                    plt.close()

    # 2) Per-episode training (train_episodes.csv)
    ep_csv = run_dir / "train_episodes.csv"
    if ep_csv.exists():
        rows = _read_csv_dicts(ep_csv)
        if rows:
            ep_idx = [int(r.get("episode_idx", i)) for i, r in enumerate(rows)]
            returns = _to_float_list(rows, "return")
            comps = _to_float_list(rows, "completion")

            # Plot return with rolling mean
            if returns:
                plt.figure(figsize=(8,4))
                plt.plot(ep_idx, returns, alpha=0.25, label="return")
                rm = _rolling_mean(returns, 100)
                plt.plot(ep_idx, rm, label="return (roll=100)")
                plt.xlabel("Episode")
                plt.ylabel("Return")
                plt.title("Training Return per Episode")
                plt.legend()
                plt.tight_layout()
                plt.savefig(figs / "train_return_rolling.png", dpi=150)
                plt.close()

            # Plot completion with rolling mean
            if comps:
                plt.figure(figsize=(8,4))
                plt.plot(ep_idx, comps, alpha=0.25, label="completion")
                rm = _rolling_mean(comps, 100)
                plt.plot(ep_idx, rm, label="completion (roll=100)")
                plt.xlabel("Episode")
                plt.ylabel("Completion")
                plt.title("Training Completion per Episode")
                plt.ylim(0, 1.0)
                plt.legend()
                plt.tight_layout()
                plt.savefig(figs / "train_completion_rolling.png", dpi=150)
                plt.close()

            # Context distribution
            ctx_ids = [int(r.get("context_id", -1)) for r in rows]
            counts = Counter(ctx_ids)
            if counts:
                xs = sorted(counts.keys())
                ys = [counts[k] for k in xs]
                plt.figure(figsize=(6,4))
                plt.bar([str(x) for x in xs], ys)
                plt.title("Context frequency (train episodes)")
                plt.xlabel("context_id")
                plt.ylabel("count")
                plt.tight_layout()
                plt.savefig(figs / "train_context_distribution.png", dpi=150)
                plt.close()

    # 3) Evaluation over time, per inertia (eval_results.jsonl)
    eval_path = run_dir / "eval_results.jsonl"
    erec = _read_jsonl(eval_path)
    if erec:
        # Group by tag and inertia
        tags = sorted(set(r.get("tag", "eval") for r in erec))
        for metric in ["return_mean", "completion_mean"]:
            for tag in tags:
                # collect per inertia
                by_inertia: Dict[float, List[Tuple[float, float]]] = defaultdict(list)
                for r in erec:
                    if r.get("tag", "eval") != tag:
                        continue
                    ts = float(r.get("timestep", float("nan")))
                    val = r.get(metric, None)
                    inertia = r.get("mario_inertia", None)
                    if inertia is None or val is None or math.isnan(ts):
                        continue
                    by_inertia[float(inertia)].append((ts, float(val)))
                if not by_inertia:
                    continue
                plt.figure(figsize=(8,4))
                for inertia, arr in sorted(by_inertia.items(), key=lambda kv: kv[0]):
                    arr = sorted(arr, key=lambda x: x[0])
                    if not arr:
                        continue
                    xs = [a[0] for a in arr]
                    ys = [a[1] for a in arr]
                    plt.plot(xs, ys, label=f"inertia={inertia:.2f}")
                plt.xlabel("Timestep")
                plt.ylabel(metric)
                plt.title(f"Eval {metric} over time [{tag}]")
                plt.legend(ncol=2, fontsize=8)
                plt.tight_layout()
                plt.savefig(figs / f"eval_{metric}_{tag}.png", dpi=150)
                plt.close()

        # Final per-context bar (last eval only, per tag)
        for tag in tags:
            last_ts = max((r.get("timestep", 0) for r in erec if r.get("tag") == tag), default=None)
            if last_ts is None:
                continue
            sub = [r for r in erec if r.get("tag") == tag and r.get("timestep") == last_ts]
            if not sub:
                continue
            # Sort by inertia
            sub = sorted(sub, key=lambda r: r.get("mario_inertia", 0))
            inertias = [float(r.get("mario_inertia", float("nan"))) for r in sub]
            for metric in ["return_mean", "completion_mean"]:
                vals = [float(r.get(metric, float("nan"))) for r in sub]
                plt.figure(figsize=(7,4))
                labels = [f"{i:.2f}" if not math.isnan(i) else "NA" for i in inertias]
                plt.bar(labels, vals)
                plt.xlabel("mario_inertia")
                plt.ylabel(metric)
                plt.title(f"Final eval {metric} [{tag}] at ts={int(last_ts)}")
                if metric == "completion_mean":
                    plt.ylim(0, 1.0)
                plt.tight_layout()
                plt.savefig(figs / f"eval_final_{metric}_{tag}.png", dpi=150)
                plt.close()


# ---------------------------
# Main
# ---------------------------
def main():
    # Reduce thread oversubscription when using SubprocVecEnv
    try:
        torch.set_num_threads(1)
    except Exception:
        pass

    config = TrainingConfig()

    print("FAST ITERATION MODE") # using print here because log isn't setup yet
    
    try:
        model, results, run_dir = train(config)
        log("=" * 50)
        log("TRAINING COMPLETED!")
        log("=" * 50)
        log(f"All outputs saved to: {run_dir}")
        log(f"  - Text Log: {run_dir}/training_output.txt")
        log(f"  - Config/Contexts: {run_dir}/config.json, train_contexts.json, test_contexts.json")
        log(f"  - SB3 CSV: {run_dir}/progress.csv")
        log(f"  - Episodes CSV: {run_dir}/train_episodes.csv")
        log(f"  - Eval JSONL: {run_dir}/eval_results.jsonl")
        log(f"  - Plots: {run_dir}/figs")
        log(f"  - Model: {run_dir}/final_model.zip")
        log(f"  - TensorBoard: {run_dir}/tb_1")
        log(f"Run: tensorboard --logdir {Path(config.log_dir)}") 
    finally:
        if LOGGER:
            LOGGER.close()

if __name__ == "__main__":
    main()
