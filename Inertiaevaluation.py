# ppo_carl_mario_inertia_generalization.py
from __future__ import annotations
import contextlib
import os
import warnings
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
from pyvirtualdisplay.abstractdisplay import XStartError  # add at top of file

import torch
import torch.nn as nn

# Reduce noisy import-time output in subprocess workers (SubprocVecEnv).
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("GYM_DISABLE_WARNINGS", "1")

warnings.filterwarnings(
    "ignore",
    message=r"pkg_resources is deprecated as an API\..*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"Module distance not found\..*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"You are using `torch\.load` with `weights_only=False`.*",
    category=FutureWarning,
)


@contextlib.contextmanager
def _suppress_output():
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield


with _suppress_output():
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
    from stable_baselines3.common.monitor import Monitor

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

# CARL imports (quiet to avoid optional-dependency spam in workers)
with _suppress_output():
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
    n_train_maps: int = 5         # number of different maps used to BUILD train contexts
    n_test_maps: int = 5          # only used if testing on NEW maps
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
    eval_freq: int = 50_000
    n_eval_episodes: int = 3

    # Logging
    log_dir: str = "./logs/mario_benchmark"
    checkpoint_freq: int = 20_000   # <-- how often (in env steps) to save checkpoints


    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42

    # Inertia-specific settings
    # Train on two ranges: 0.8–0.9 and 1.1–1.2
    train_inertia_ranges: Tuple[Tuple[float, float], ...] = (
        (0.8, 0.9),
        (1.1, 1.2),
    )
    # If True, sample a new mario_inertia every episode (from the ranges above).
    # For performance/reproducibility we rotate through a fixed pre-sampled pool.
    resample_train_inertia_each_episode: bool = True
    train_inertia_pool_size: int = 100

    # Final eval on training maps: budget controls (keeps runtime similar to test eval)
    train_final_eval_n_inertias: int = 100
    train_final_eval_n_episodes: int = 1
    # Evaluate on this grid of inertias
    eval_inertias: Tuple[float, ...] = (0.6, 0.85, 1, 1.15, 1.4)

    # Simple logging toggles
    print_freq: int = 10_000         # summary every N timesteps
    progress_window: int = 50        # moving window for mean stats
    episode_summaries: bool = False   # print line when episode finishes
    save_final_model: bool = True    # save the final model

    # Experiment switches
    # If True, the observation will include a "ctx" vector with (normalized) mario_inertia
    use_context: bool = True
    # Which CARL context keys to include in the "ctx" vector (order matters).
    # Common choices: ("mario_inertia",) or ("level_index", "mario_state", "mario_inertia")
    context_keys: Tuple[str, ...] = ("mario_inertia",)

    # How to fuse vision + context in the policy:
    # - "concat": SB3 CombinedExtractor (default behavior)
    # - "hadamard": cGate/Hadamard (element-wise) fusion in a custom extractor
    feature_fusion: str = "hadamard"  # "concat" or "hadamard"
    extractor_features_dim: int = 256
    context_hidden_dim: int = 256
    hadamard_gate_activation: str = "relu"  # "relu" or "sigmoid"

    # If True: test on the SAME maps as training (same noise_seed etc.), only inertia changes.
    # If False: test on NEW maps (different noise_seed) as in the original version.
    test_on_train_maps: bool = False

    # If True, do an additional final evaluation on the exact training contexts
    # (same maps AND the same inertia values sampled for training).
    final_eval_on_train_contexts: bool = True


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
        # success flag added
        self.episode_stats = {"steps": 0, "reward": 0.0, "completion": 0.0, "success": False}

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
        # reset success flag as well
        self.episode_stats = {"steps": 0, "reward": 0.0, "completion": 0.0, "success": False}
        obs, info = super().reset(**kwargs)
        return obs, info

    def close(self):
        try:
            return super().close()
        except XStartError:
            # pyvirtualdisplay wasn't started (common in headless SubprocVecEnv workers)
            return None

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        # Update episode stats
        completion_from_info = float(info.get("completed", info.get("completion", 0.0)))
        self.episode_stats["steps"] += 1
        self.episode_stats["reward"] += float(reward)
        self.episode_stats["completion"] = max(self.episode_stats["completion"], completion_from_info)
        # full level success if completion is essentially 1.0
        self.episode_stats["success"] = self.episode_stats["success"] or (completion_from_info >= 0.999)

        if terminated or truncated:
            info["episode_stats"] = {
                "steps": int(self.episode_stats["steps"]),
                "reward": float(self.episode_stats["reward"]),
                "completion": float(self.episode_stats["completion"]),
                "success": bool(self.episode_stats["success"]),
            }
            info["context_id"] = int(self.context_id)
            info["level_width"] = int(self.context["level_width"])
            info["mario_inertia"] = float(self.context["mario_inertia"])

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
# cGate / Hadamard Fusion Extractor
# ---------------------------
class HadamardGateExtractor(BaseFeaturesExtractor):
    """
    Multi-input extractor that fuses vision ("img") and context ("ctx") using a
    Hadamard (element-wise) product.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        features_dim: int = 256,
        context_hidden_dim: int = 256,
        gate_activation: str = "relu",
    ):
        super().__init__(observation_space, features_dim)

        if not isinstance(observation_space, spaces.Dict):
            raise TypeError("HadamardGateExtractor expects a Dict observation space.")
        if "img" not in observation_space.spaces or "ctx" not in observation_space.spaces:
            raise KeyError('HadamardGateExtractor requires observation keys "img" and "ctx".')

        img_space = observation_space.spaces["img"]
        ctx_space = observation_space.spaces["ctx"]

        if not isinstance(img_space, spaces.Box) or img_space.shape is None or len(img_space.shape) != 3:
            raise TypeError('Observation space "img" must be a Box(C,H,W).')
        if not isinstance(ctx_space, spaces.Box) or ctx_space.shape is None or len(ctx_space.shape) != 1:
            raise TypeError('Observation space "ctx" must be a 1D Box.')

        in_channels, height, width = img_space.shape
        ctx_dim = int(ctx_space.shape[0])

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, height, width)
            n_flat = int(self.cnn(dummy).shape[1])

        self.context_proj = nn.Sequential(
            nn.LayerNorm(ctx_dim),
            nn.Linear(ctx_dim, int(context_hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(context_hidden_dim), n_flat),
        )

        gate_activation = str(gate_activation).lower().strip()
        if gate_activation == "relu":
            self.gate_activation: nn.Module = nn.ReLU()
        elif gate_activation == "sigmoid":
            self.gate_activation = nn.Sigmoid()
        else:
            raise ValueError('gate_activation must be "relu" or "sigmoid".')

        self.final_layer = nn.Sequential(
            nn.Linear(n_flat, int(features_dim)),
            nn.ReLU(),
        )

        self._features_dim = int(features_dim)

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        img = observations["img"]
        ctx = observations["ctx"]

        if img.dtype == torch.uint8:
            img = img.float() / 255.0
        else:
            img = img.float()
        ctx = ctx.float()

        frame_feat = self.cnn(img)
        ctx_feat = self.gate_activation(self.context_proj(ctx))
        fused = frame_feat * ctx_feat
        return self.final_layer(fused)


# ---------------------------
# Context generation (only varying mario_inertia)
# ---------------------------
def build_inertia_train_contexts(
    n_maps: int,
    level_width: int,
    inertia_ranges: Sequence[Tuple[float, float]],
    seed: int = 42,
    sample_inertia: bool = True,
) -> Dict[int, Dict[str, Any]]:
    """
    Build training contexts as a *cross product* of:
      - n_maps different maps (different noise_seed each)
      - the given inertia ranges

    Total contexts = n_maps * len(inertia_ranges).
    """
    rng = np.random.RandomState(int(seed))
    ctxs: Dict[int, Dict[str, Any]] = {}
    ctx_id = 0

    for map_idx in range(n_maps):
        # one distinct map per map_idx
        noise_seed = seed + map_idx

        for range_idx, (low, high) in enumerate(inertia_ranges):
            low_f = float(low)
            high_f = float(high)
            inertia = (
                float(rng.uniform(low_f, high_f))
                if sample_inertia
                else float((low_f + high_f) / 2.0)
            )

            ctxs[ctx_id] = {
                "level_width": int(level_width),
                "level_index": 0,          # keep the same level_index
                "noise_seed": noise_seed,  # this fixes the map identity
                "mario_state": 0,
                "mario_inertia": inertia,
                "train_range_index": int(range_idx),
            }
            ctx_id += 1

    return ctxs


def build_inertia_pools(
    inertia_ranges: Sequence[Tuple[float, float]],
    pool_size: int,
    seed: int,
) -> list[list[float]]:
    rng = np.random.RandomState(int(seed))
    pools: list[list[float]] = []
    for (low, high) in inertia_ranges:
        low_f = float(low)
        high_f = float(high)
        vals = rng.uniform(low_f, high_f, size=int(pool_size)).astype(np.float32)
        rng.shuffle(vals)
        pools.append([float(x) for x in vals])
    return pools


def _unique_train_maps(train_contexts: Dict[int, Dict[str, Any]]) -> list[Dict[str, Any]]:
    maps: list[Dict[str, Any]] = []
    seen = set()
    for ctx in train_contexts.values():
        key = (
            int(ctx["level_width"]),
            int(ctx["level_index"]),
            int(ctx["noise_seed"]),
            int(ctx["mario_state"]),
        )
        if key in seen:
            continue
        seen.add(key)
        maps.append(
            {
                "level_width": int(ctx["level_width"]),
                "level_index": int(ctx["level_index"]),
                "noise_seed": int(ctx["noise_seed"]),
                "mario_state": int(ctx["mario_state"]),
            }
        )
    return maps


def build_train_eval_contexts_sampled(
    train_contexts: Dict[int, Dict[str, Any]],
    inertia_values: Sequence[float],
    seed: int,
) -> Dict[int, Dict[str, Any]]:
    """
    Build ~len(inertia_values) contexts on the training maps, pairing each inertia
    with a map (rotating through unique training maps). This avoids the full
    cross-product (maps x inertias) for runtime.
    """
    maps = _unique_train_maps(train_contexts)
    if not maps:
        return {}
    inertias = [float(x) for x in inertia_values]
    rng = np.random.RandomState(int(seed))
    rng.shuffle(inertias)

    ctxs: Dict[int, Dict[str, Any]] = {}
    for i, inertia in enumerate(inertias):
        map_params = maps[i % len(maps)]
        ctxs[int(i)] = {
            **map_params,
            "mario_inertia": float(inertia),
            "gen_type": "train",
        }
    return ctxs


class TrainInertiaResampleWrapper(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        contexts: Dict[int, Dict[str, Any]],
        inertia_pools: list[list[float]],
        seed: int,
    ):
        super().__init__(env)
        self.contexts = contexts
        self.context_ids = [int(k) for k in contexts.keys()]
        self.inertia_pools = inertia_pools
        self.pool_pos = [0 for _ in inertia_pools]
        self.rng = np.random.RandomState(int(seed))
        self.ctx_to_range_idx: Dict[int, int] = {}
        for ctx_id, ctx in contexts.items():
            ridx = ctx.get("train_range_index", 0)
            self.ctx_to_range_idx[int(ctx_id)] = int(ridx)

    def reset(self, **kwargs):
        options = kwargs.get("options") or {}
        if "context_id" in options:
            ctx_id = int(options["context_id"])
        else:
            ctx_id = int(self.rng.choice(self.context_ids))
            kwargs["options"] = dict(options)
            kwargs["options"]["context_id"] = ctx_id

        range_idx = int(self.ctx_to_range_idx.get(ctx_id, 0))
        pool = self.inertia_pools[range_idx]
        pos = int(self.pool_pos[range_idx])
        inertia = float(pool[pos])
        self.pool_pos[range_idx] = (pos + 1) % len(pool)
        self.contexts[ctx_id]["mario_inertia"] = inertia

        return self.env.reset(**kwargs)


def build_inertia_test_contexts(
    inertia_values: Sequence[float],
    n_maps: int,  # number of different maps per inertia
    level_width: int,
    seed: int = 43,
) -> Dict[int, Dict[str, Any]]:
    """
    Test contexts on NEW maps: for each inertia, create n_maps contexts with different noise_seed.
    """
    ctxs: Dict[int, Dict[str, Any]] = {}
    ctx_id = 0

    for map_idx in range(n_maps):
        noise_seed = seed + 10_000 + map_idx  # Different map per map_idx

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


def build_test_on_train_maps(
    train_contexts: Dict[int, Dict[str, Any]],
    inertia_values: Sequence[float],
) -> Dict[int, Dict[str, Any]]:
    """
    Build test contexts that reuse the SAME maps as in train_contexts
    (same level_width, level_index, noise_seed, mario_state)
    but sweep over the given inertia_values.

    This isolates the effect of changing mario_inertia from any map change.
    """
    ctxs: Dict[int, Dict[str, Any]] = {}
    ctx_id = 0

    # Identify unique maps by (level_width, level_index, noise_seed, mario_state)
    seen_maps = set()
    for ctx in train_contexts.values():
        key = (
            int(ctx["level_width"]),
            int(ctx["level_index"]),
            int(ctx["noise_seed"]),
            int(ctx["mario_state"]),
        )
        if key in seen_maps:
            continue
        seen_maps.add(key)
        level_width, level_index, noise_seed, mario_state = key

        for inertia in inertia_values:
            ctxs[ctx_id] = {
                "level_width": level_width,
                "level_index": level_index,
                "noise_seed": noise_seed,
                "mario_state": mario_state,
                "mario_inertia": float(inertia),
            }
            ctx_id += 1

    return ctxs


def classify_inertia(
    inertia: float,
    train_ranges: Sequence[Tuple[float, float]],
) -> str:
    """
    Classify inertia as:
      - 'id'      : inside any training interval
      - 'interp'  : between min and max train inertia but in no interval
      - 'extra'   : outside [min_train, max_train]
    """
    min_lo = min(lo for lo, _ in train_ranges)
    max_hi = max(hi for _, hi in train_ranges)

    for lo, hi in train_ranges:
        if lo <= inertia <= hi:
            return "id"

    if min_lo < inertia < max_hi:
        return "interp"

    return "extra"


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
        self.episode_history: list[dict] = []

        # For uniform-per-context metrics
        self.ctx_returns: Dict[int, deque] = defaultdict(lambda: deque(maxlen=self.window))
        self.ctx_completions: Dict[int, deque] = defaultdict(lambda: deque(maxlen=self.window))
        # For distribution of contexts in the window
        self.last_ctx_ids: deque = deque(maxlen=self.window)
        self.history: list[dict] = []   # each entry: {"timesteps": int, "mean_return": ..., "mean_completion": ...}


    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if info is None:
                continue
            if "episode" in info:
                ep_r = float(info["episode"].get("r", 0.0))
                ep_l = int(info["episode"].get("l", 0))
                completion = None
                success = None
                inertia = None
                if "episode_stats" in info:
                    stats = info["episode_stats"]
                    completion = float(stats.get("completion", 0.0))
                    success = bool(stats.get("success", completion >= 0.999))
                if "mario_inertia" in info:
                    inertia = float(info["mario_inertia"])

                self.ep_returns.append(ep_r)
                self.ep_lengths.append(ep_l)
                if completion is not None:
                    self.ep_completions.append(completion)
                self.total_episodes += 1

                ctx_id = info.get("context_id", None)
                self.episode_history.append(
                    {
                        "timesteps": int(self.num_timesteps),
                        "episode": int(self.total_episodes),
                        "return": float(ep_r),
                        "length": int(ep_l),
                        "completion": (None if completion is None else float(completion)),
                        "success": (None if success is None else bool(success)),
                        "mario_inertia": (None if inertia is None else float(inertia)),
                        "context_id": (None if ctx_id is None else int(ctx_id)),
                    }
                )
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

            mean_c_val = None if np.isnan(mean_c) else float(mean_c)
            uniform_ctx_mean_r_val = None if np.isnan(uniform_ctx_mean_r) else float(uniform_ctx_mean_r)
            uniform_ctx_mean_c_val = None if np.isnan(uniform_ctx_mean_c) else float(uniform_ctx_mean_c)

            self.history.append(
                {
                    "timesteps": int(self.num_timesteps),
                    "mean_return": float(mean_r),
                    "mean_completion": mean_c_val,
                    "uniform_ctx_mean_return": uniform_ctx_mean_r_val,
                    "uniform_ctx_mean_completion": uniform_ctx_mean_c_val,
                }
            )

        return True


class ContextualEvalCallback(BaseCallback):
    """
    Evaluate per-context with configurable deterministic vs stochastic policy.
    Now aggregates results:
      1. Per individual context (as before)
      2. Per inertia value (averaged across maps)
      3. Per generalization type (id/interp/extra, averaged across maps and inertias)
      4. Tracks full-level success rate as well.
    """
    def __init__(
        self,
        eval_env: gym.Env,
        eval_contexts: List[int],
        eval_freq: int = 10_000,
        n_eval_episodes: int = 2,
        deterministic: bool = True,
        tag: str = "",
        ctx_dict: Optional[Dict[int, Dict[str, Any]]] = None,
    ):
        super().__init__()
        self.eval_env = eval_env
        self.eval_contexts = eval_contexts
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.deterministic = deterministic
        self.tag = tag
        self._last_eval_ts = 0

        # Dict of contexts so we can read inertia and gen_type
        self.ctx_dict = ctx_dict or {}

    def _on_step(self) -> bool:
        if (self.num_timesteps - self._last_eval_ts) >= self.eval_freq:
            results = self._evaluate()
            self._last_eval_ts = self.num_timesteps

            # Collect all returns, completions, successes
            all_returns = []
            all_completions = []
            all_successes = []
            for ctx_results in results.values():
                all_returns.extend(ctx_results["returns"])
                all_completions.extend(ctx_results["completions"])
                all_successes.extend(ctx_results["successes"])

            mean_return = float(np.mean(all_returns)) if len(all_returns) > 0 else float("nan")
            mean_completion = float(np.mean(all_completions)) if len(all_completions) > 0 else float("nan")
            mean_success = float(np.mean(all_successes)) if len(all_successes) > 0 else float("nan")

            # ----------------------------------------------------------
            # Aggregate by inertia value (average across maps)
            # ----------------------------------------------------------
            inertia_agg: Dict[float, Dict[str, List[float]]] = defaultdict(
                lambda: {"returns": [], "completions": [], "successes": [], "gen_type": None}
            )
            for ctx_id, ctx_results in results.items():
                inertia = ctx_results.get("inertia", float("nan"))
                gen_type = ctx_results.get("gen_type", "unknown")
                inertia_agg[inertia]["returns"].extend(ctx_results["returns"])
                inertia_agg[inertia]["completions"].extend(ctx_results["completions"])
                inertia_agg[inertia]["successes"].extend(ctx_results["successes"])
                inertia_agg[inertia]["gen_type"] = gen_type

            # ----------------------------------------------------------
            # Aggregate by generalization type (average across maps AND inertias)
            # ----------------------------------------------------------
            type_agg: Dict[str, Dict[str, List[float]]] = defaultdict(
                lambda: {"returns": [], "completions": [], "successes": []}
            )
            for ctx_id, ctx_results in results.items():
                gen_type = ctx_results.get("gen_type", "unknown")
                type_agg[gen_type]["returns"].extend(ctx_results["returns"])
                type_agg[gen_type]["completions"].extend(ctx_results["completions"])
                type_agg[gen_type]["successes"].extend(ctx_results["successes"])

            # ----------------------------------------------------------
            # Logging
            # ----------------------------------------------------------
            log("=" * 60)
            log(f"Eval{(' ['+self.tag+']') if self.tag else ''} @ Timestep {self.num_timesteps:,}  (deterministic={self.deterministic})")
            log(f"  Overall mean return: {mean_return:.2f}")
            log(f"  Overall mean completion: {mean_completion:.2%}")
            log(f"  Overall success rate (full level): {mean_success:.2%}")

            # Per inertia (averaged across maps)
            log("-" * 60)
            log("Per-inertia eval (averaged over maps):")
            for inertia in sorted(inertia_agg.keys()):
                agg = inertia_agg[inertia]
                r_mean = float(np.mean(agg["returns"])) if agg["returns"] else float("nan")
                c_mean = float(np.mean(agg["completions"])) if agg["completions"] else float("nan")
                s_mean = float(np.mean(agg["successes"])) if agg["successes"] else float("nan")
                n_eps = len(agg["returns"])
                gen_type = agg["gen_type"] or "unknown"
                log(
                    f"  inertia={inertia:.3f} ({gen_type}): "
                    f"return={r_mean:.2f}  completion={c_mean:.2%}  "
                    f"success={s_mean:.2%}  (n={n_eps} episodes)"
                )

            # Per generalization type (averaged across maps AND inertias)
            log("-" * 60)
            log("Per-type eval (averaged over maps and inertias):")
            for gen_type in ["id", "interp", "extra"]:
                if gen_type not in type_agg:
                    continue
                agg = type_agg[gen_type]
                r_mean = float(np.mean(agg["returns"])) if agg["returns"] else float("nan")
                c_mean = float(np.mean(agg["completions"])) if agg["completions"] else float("nan")
                s_mean = float(np.mean(agg["successes"])) if agg["successes"] else float("nan")
                n_eps = len(agg["returns"])
                log(
                    f"  {gen_type:8s}: return={r_mean:.2f}  "
                    f"completion={c_mean:.2%}  success={s_mean:.2%}  (n={n_eps} episodes)"
                )

            log("=" * 60)

            # Debug info (optional)
            with torch.no_grad():
                total_abs = 0.0
                for p in self.model.policy.parameters():
                    total_abs += p.abs().sum().item()
            log(f"Policy param |abs| sum: {total_abs:.3e}")

            total_grad_norm = 0.0
            for p in self.model.policy.parameters():
                if p.grad is not None:
                    total_grad_norm += p.grad.norm().item() ** 2
            total_grad_norm = total_grad_norm ** 0.5
            log(f"Total gradient norm: {total_grad_norm:.3e}")

        return True

    def _evaluate(self) -> Dict[int, Dict[str, Any]]:
        results = {}

        for ctx_id in self.eval_contexts:
            ctx_returns = []
            ctx_completions = []
            ctx_lengths = []
            ctx_successes = []

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
                success = False
                if "episode_stats" in info:
                    stats = info["episode_stats"]
                    completion = stats.get("completion", 0.0)
                    # Fallback: treat completion ~1.0 as success if not explicitly provided
                    success = bool(stats.get("success", completion >= 0.999))
                ctx_completions.append(completion)
                ctx_successes.append(1.0 if success else 0.0)

            ctx_meta = self.ctx_dict.get(ctx_id, {})
            inertia = float(ctx_meta.get("mario_inertia", float("nan")))
            gen_type = ctx_meta.get("gen_type", "unknown")

            results[ctx_id] = {
                "returns": ctx_returns,
                "completions": ctx_completions,
                "successes": ctx_successes,
                "lengths": ctx_lengths,
                "inertia": inertia,
                "gen_type": gen_type,
            }

        return results

class PeriodicCheckpointCallback(BaseCallback):
    """
    Save model (and VecNormalize stats, if used) every `save_freq` environment timesteps.
    """
    def __init__(self, save_freq: int, save_path: str | Path, name_prefix: str = "checkpoint", verbose: int = 0):
        super().__init__(verbose)
        self.save_freq = int(save_freq)
        self.save_path = Path(save_path)
        self.name_prefix = name_prefix
        self._last_save = 0

    def _on_training_start(self) -> None:
        # Ensure directory exists
        self.save_path.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        # self.num_timesteps is the total number of environment steps seen so far
        if (self.num_timesteps - self._last_save) >= self.save_freq:
            model_path = self.save_path / f"{self.name_prefix}_{self.num_timesteps}.zip"
            self.model.save(model_path)

            # If using VecNormalize, save its statistics too
            try:
                from stable_baselines3.common.vec_env import VecNormalize
                if isinstance(self.training_env, VecNormalize):
                    vecnorm_path = self.save_path / f"vecnormalize_{self.num_timesteps}.pkl"
                    self.training_env.save(vecnorm_path)
            except Exception:
                pass

            log(f"Checkpoint saved at {self.num_timesteps:,} steps to {model_path}")
            self._last_save = self.num_timesteps

        return True
# ---------------------------
# Training Utilities
# ---------------------------
def make_env_fn(
    contexts: Dict[int, Dict],
    ctx_keys: List[str],
    monitor: bool = True,
    *,
    resample_train_inertia: bool = False,
    inertia_pools: Optional[list[list[float]]] = None,
    base_seed: int = 0,
    env_rank: int = 0,
):
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
            frame_stack=2,
            to_gray=True,
            resize_to=(84, 84),
        )
        # Action repeat (frame skip)
        env = ActionRepeat(env, repeat=4, max_pool=True)
        if resample_train_inertia:
            if inertia_pools is None:
                raise ValueError("resample_train_inertia requested but inertia_pools is None")
            env = TrainInertiaResampleWrapper(
                env,
                contexts=contexts,
                inertia_pools=inertia_pools,
                seed=int(base_seed) + 10_000 * int(env_rank) + 123,
            )
        if monitor:
            env = Monitor(env)
        return env
    return _init




def linear_schedule(initial_value: float):
    # SB3 schedule: progress goes from 1 (start) to 0 (end)
    def f(progress: float):
        return initial_value * progress
    return f


def setup_training(config: TrainingConfig):
    # ---------------------------------------------------------
    # 1. Create a Unique Run Directory based on time
    # ---------------------------------------------------------
    run_id = "inertia_" + time.strftime("%m%d_%H%M%S")
    base_path = Path(config.log_dir)
    run_dir = base_path / run_id
    # Avoid collisions if multiple runs start within the same second.
    if run_dir.exists():
        run_dir = base_path / f"{run_id}_{int(time.time() * 1000) % 1_000_000:06d}"
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

    # Create training contexts
    log("Generating training contexts (varying only mario_inertia)...")
    train_contexts = build_inertia_train_contexts(
        n_maps=config.n_train_maps,
        level_width=config.level_width,
        inertia_ranges=config.train_inertia_ranges,
        seed=config.seed + 1,
        sample_inertia=not bool(config.resample_train_inertia_each_episode),
    )

    inertia_pools: Optional[list[list[float]]] = None
    train_inertia_values: Optional[list[float]] = None
    if config.resample_train_inertia_each_episode:
        inertia_pools = build_inertia_pools(
            inertia_ranges=config.train_inertia_ranges,
            pool_size=config.train_inertia_pool_size,
            seed=config.seed + 777,
        )
        train_inertia_values = sorted(float(v) for pool in inertia_pools for v in pool)
        log(
            "Training inertia resampling enabled: "
            f"{len(config.train_inertia_ranges)} ranges x {config.train_inertia_pool_size} values (rotating pool)"
        )

    # Create test contexts: either on TRAIN MAPS or NEW MAPS
    if config.test_on_train_maps:
        log("Generating TEST contexts on TRAIN MAPS (same maps, varying mario_inertia)...")
        test_contexts = build_test_on_train_maps(
            train_contexts=train_contexts,
            inertia_values=config.eval_inertias,
        )
    else:
        log("Generating TEST contexts on NEW MAPS (different maps, varying mario_inertia)...")
        test_contexts = build_inertia_test_contexts(
            n_maps=config.n_test_maps,
            inertia_values=config.eval_inertias,
            level_width=config.level_width,
            seed=config.seed + 1,
        )

    # Tag each test context as in-distribution / interpolation / extrapolation
    type_counts = {"id": 0, "interp": 0, "extra": 0}
    for ctx_id, ctx in test_contexts.items():
        inertia = float(ctx["mario_inertia"])
        gtype = classify_inertia(inertia, config.train_inertia_ranges)
        ctx["gen_type"] = gtype
        type_counts[gtype] += 1

    # Overwrite with real counts (#contexts, not literally #maps)
    config.n_train_maps = len(train_contexts)
    config.n_test_maps = len(test_contexts)

    # Decide which context keys to include in the observation
    ctx_keys: List[str] = list(config.context_keys) if config.use_context else []
    if ctx_keys:
        available = set(CARLMarioEnv.get_context_features().keys())
        unknown = [k for k in ctx_keys if k not in available]
        if unknown:
            raise ValueError(f"Unknown context_keys in config: {unknown}. Available: {sorted(available)}")
    log(f"Using context keys in observations: {ctx_keys if ctx_keys else 'NONE (vision only)'}")

    log("Training contexts (all):")
    for k in sorted(train_contexts.keys()):
        log(f"  id={k}: {train_contexts[k]}")

    log("Test contexts (all):")
    for k in sorted(test_contexts.keys()):
        ctx = test_contexts[k]
        log(
            f"  id={k}: {ctx} "
            f"--> gen_type={ctx['gen_type']}, inertia={ctx['mario_inertia']:.3f}"
        )
    log(f"Test contexts by type: {type_counts}")

    log(f"Creating {config.n_envs} parallel environments...")
    env_fns = [
        make_env_fn(
            train_contexts,
            ctx_keys,
            resample_train_inertia=bool(config.resample_train_inertia_each_episode),
            inertia_pools=inertia_pools,
            base_seed=int(config.seed),
            env_rank=int(i),
        )
        for i in range(config.n_envs)
    ]

    if config.n_envs == 1:
        train_env = DummyVecEnv(env_fns)
    else:
        train_env = SubprocVecEnv(env_fns)

    train_env = VecNormalize(train_env, norm_obs=False, norm_reward=True, clip_reward=10.0)

    log("Creating evaluation environment...")
    eval_env = make_env_fn(test_contexts, ctx_keys)()

    train_eval_contexts = train_contexts
    train_eval_inertias: Optional[list[float]] = None
    if config.resample_train_inertia_each_episode and train_inertia_values is not None:
        n = int(config.train_final_eval_n_inertias)
        if n <= 0:
            train_eval_contexts = {}
        else:
            if len(train_inertia_values) <= n:
                train_eval_inertias = list(train_inertia_values)
            else:
                rng = np.random.RandomState(int(config.seed) + 999)
                idx = rng.choice(len(train_inertia_values), size=n, replace=False)
                train_eval_inertias = [float(train_inertia_values[i]) for i in idx]
            train_eval_contexts = build_train_eval_contexts_sampled(
                train_contexts=train_contexts,
                inertia_values=train_eval_inertias,
                seed=int(config.seed) + 1001,
            )
        log(
            f"Train final eval contexts: {len(train_eval_contexts)} "
            f"(n_inertias={len(train_eval_inertias) if train_eval_inertias is not None else 0})"
        )

    train_eval_env = make_env_fn(train_eval_contexts, ctx_keys)()

    # ---------------------------------------------------------
    # 4. Model Setup
    # ---------------------------------------------------------
    fusion = str(config.feature_fusion).lower().strip()
    if fusion not in ("concat", "hadamard"):
        raise ValueError('config.feature_fusion must be "concat" or "hadamard".')

    if fusion == "hadamard":
        if not ctx_keys:
            raise ValueError('Hadamard fusion requires config.use_context=True and non-empty config.context_keys.')
        policy_kwargs = dict(
            features_extractor_class=HadamardGateExtractor,
            features_extractor_kwargs=dict(
                features_dim=int(config.extractor_features_dim),
                context_hidden_dim=int(config.context_hidden_dim),
                gate_activation=str(config.hadamard_gate_activation),
            ),
            net_arch=dict(pi=[256], vf=[256]),
            activation_fn=nn.ReLU,
            ortho_init=False,
            normalize_images=True,
        )
    else:
        try:
            from stable_baselines3.common.torch_layers import CombinedExtractor
        except ImportError:
            from stable_baselines3.common.torch_layers import MultiInputExtractor as CombinedExtractor

        policy_kwargs = dict(
            features_extractor_class=CombinedExtractor,
            features_extractor_kwargs=dict(cnn_output_dim=int(config.extractor_features_dim)),
            net_arch=dict(pi=[256], vf=[256]),
            activation_fn=nn.ReLU,
            ortho_init=False,
            normalize_images=True,
        )

    log("Creating PPO model...")
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
        tensorboard_log=str(run_dir),
        verbose=0,
    )

    eval_callback_stoch = ContextualEvalCallback(
        eval_env=eval_env,
        eval_contexts=list(test_contexts.keys()),
        eval_freq=config.eval_freq,
        n_eval_episodes=config.n_eval_episodes,
        deterministic=False,
        tag="stochastic",
        ctx_dict=test_contexts,
    )

    train_printer = TrainingPrinterCallback(
        print_freq=config.print_freq,
        window=config.progress_window,
        episode_summaries=config.episode_summaries,
    )

    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_callback = PeriodicCheckpointCallback(
        save_freq=config.checkpoint_freq,
        save_path=checkpoint_dir,
        name_prefix="ppo_inertia",
    )

    callbacks = [train_printer, eval_callback_stoch, checkpoint_callback]

    return (
        model,
        train_env,
        eval_env,
        train_eval_env,
        callbacks,
        train_contexts,
        train_eval_contexts,
        test_contexts,
        run_dir,
        train_printer,
        train_inertia_values,
    )



def evaluate_final(
    model: PPO,
    eval_env: gym.Env,
    contexts: Dict[int, Dict],
    n_episodes: int = 2,
    deterministic: bool = False,
    tag: str = "",
    inertia_ranges: Optional[Sequence[Tuple[float, float]]] = None,
    summarize_inertia_by_ranges: bool = False,
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
            success = False
            if "episode_stats" in info:
                stats = info["episode_stats"]
                completion = stats.get("completion", 0.0)
                # fallback: treat completion close to 1 as success
                success = bool(stats.get("success", completion >= 0.999))

            all_results.append({
                "context_id": ctx_id,
                "context_params": ctx_params,
                "episode": ep,
                "return": total_reward,
                "length": steps,
                "completion": completion,
                "success": success,
                "gen_type": ctx_params.get("gen_type", "all"),
            })

    returns = [r["return"] for r in all_results]
    completions = [r["completion"] for r in all_results]
    lengths = [r["length"] for r in all_results]
    successes = [1.0 if r["success"] else 0.0 for r in all_results]

    summary = {
        "mean_return": float(np.mean(returns)) if len(returns) > 0 else float("nan"),
        "std_return": float(np.std(returns)) if len(returns) > 0 else float("nan"),
        "mean_completion": float(np.mean(completions)) if len(completions) > 0 else float("nan"),
        "std_completion": float(np.std(completions)) if len(completions) > 0 else float("nan"),
        "mean_length": float(np.mean(lengths)) if len(lengths) > 0 else float("nan"),
        # success here is explicitly "full level completion"
        "success_rate": float(np.mean(successes)) if len(successes) > 0 else float("nan"),
        "episodes": all_results,
        "deterministic": deterministic,
        "tag": tag,
    }

    from collections import defaultdict

    # Per generalization-type summary (id / interp / extra)
    group_stats = defaultdict(lambda: {"returns": [], "completions": [], "lengths": [], "successes": []})
    for r in all_results:
        g = r["gen_type"]
        group_stats[g]["returns"].append(r["return"])
        group_stats[g]["completions"].append(r["completion"])
        group_stats[g]["lengths"].append(r["length"])
        group_stats[g]["successes"].append(1.0 if r["success"] else 0.0)

    per_group = {}
    for g, vals in group_stats.items():
        if len(vals["returns"]) == 0:
            continue
        per_group[g] = {
            "mean_return": float(np.mean(vals["returns"])),
            "mean_completion": float(np.mean(vals["completions"])),
            "mean_length": float(np.mean(vals["lengths"])),
            "success_rate": float(np.mean(vals["successes"])),
            "n_episodes": len(vals["returns"]),
        }
    summary["per_group"] = per_group

    # Per-inertia summary
    inertia_stats = defaultdict(lambda: {"returns": [], "completions": [], "lengths": [], "successes": []})
    for r in all_results:
        inertia = float(r["context_params"]["mario_inertia"])
        inertia_stats[inertia]["returns"].append(r["return"])
        inertia_stats[inertia]["completions"].append(r["completion"])
        inertia_stats[inertia]["lengths"].append(r["length"])
        inertia_stats[inertia]["successes"].append(1.0 if r["success"] else 0.0)

    per_inertia = {}
    for inertia, vals in inertia_stats.items():
        if len(vals["returns"]) == 0:
            continue
        per_inertia[float(inertia)] = {
            "mean_return": float(np.mean(vals["returns"])),
            "mean_completion": float(np.mean(vals["completions"])),
            "mean_length": float(np.mean(vals["lengths"])),
            "success_rate": float(np.mean(vals["successes"])),
            "n_episodes": len(vals["returns"]),
        }
    summary["per_inertia"] = per_inertia

    # Per training-range summary (optional)
    if inertia_ranges is not None and len(inertia_ranges) > 0:
        range_stats = defaultdict(lambda: {"returns": [], "completions": [], "lengths": [], "successes": []})
        for r in all_results:
            inertia = float(r["context_params"]["mario_inertia"])
            range_idx: Optional[int] = None
            for i, (lo, hi) in enumerate(inertia_ranges):
                if float(lo) <= inertia <= float(hi):
                    range_idx = int(i)
                    break
            key = (
                f"range_{range_idx}:{float(inertia_ranges[range_idx][0]):.3f}-{float(inertia_ranges[range_idx][1]):.3f}"
                if range_idx is not None
                else "other"
            )
            range_stats[key]["returns"].append(r["return"])
            range_stats[key]["completions"].append(r["completion"])
            range_stats[key]["lengths"].append(r["length"])
            range_stats[key]["successes"].append(1.0 if r["success"] else 0.0)

        per_range: Dict[str, Any] = {}
        for key, vals in range_stats.items():
            if len(vals["returns"]) == 0:
                continue
            per_range[key] = {
                "mean_return": float(np.mean(vals["returns"])),
                "mean_completion": float(np.mean(vals["completions"])),
                "mean_length": float(np.mean(vals["lengths"])),
                "success_rate": float(np.mean(vals["successes"])),
                "n_episodes": len(vals["returns"]),
            }
        summary["per_range"] = per_range

    # Logging
    log("=" * 50)
    log(f"FINAL EVALUATION RESULTS{(' ['+tag+']') if tag else ''} (deterministic={deterministic})")
    log("=" * 50)
    log(f"Mean Return: {summary['mean_return']:.2f} ± {summary['std_return']:.2f}")
    log(f"Mean Completion: {summary['mean_completion']:.2%} ± {summary['std_completion']:.2%}")
    log(f"Success Rate (full level): {summary['success_rate']:.2%}")
    log(f"Mean Episode Length: {summary['mean_length']:.1f}")

    if len(per_group) > 1:
        log("Per generalization-type summary:")
        for g, s in per_group.items():
            log(
                f"  {g}: "
                f"mean_return={s['mean_return']:.2f}, "
                f"mean_completion={s['mean_completion']:.2%}, "
                f"success_rate={s['success_rate']:.2%}, "
                f"n_episodes={s['n_episodes']}"
            )

    if len(per_inertia) > 0:
        if summarize_inertia_by_ranges and "per_range" in summary:
            log("Per-range summary (collapsed from per-inertia):")
            for key in sorted(summary["per_range"].keys()):
                s = summary["per_range"][key]
                log(
                    f"  {key}: "
                    f"mean_return={s['mean_return']:.2f}, "
                    f"mean_completion={s['mean_completion']:.2%}, "
                    f"success_rate={s['success_rate']:.2%}, "
                    f"n_episodes={s['n_episodes']}"
                )
            log(f"(Per-inertia details saved in JSON: {len(per_inertia)} values)")
        else:
            log("Per-inertia summary:")
            for inertia in sorted(per_inertia.keys()):
                s = per_inertia[inertia]
                log(
                    f"  inertia={inertia:.3f}: "
                    f"mean_return={s['mean_return']:.2f}, "
                    f"mean_completion={s['mean_completion']:.2%}, "
                    f"success_rate={s['success_rate']:.2%}, "
                    f"n_episodes={s['n_episodes']}"
                )

    log("=" * 50)

    return summary


def summarize_training_episodes(
    episode_history: list[dict],
    contexts: Dict[int, Dict[str, Any]],
    tag: str = "training",
) -> Dict[str, Any]:
    episodes = []
    for ep in episode_history:
        ctx_id = ep.get("context_id", None)
        if ctx_id is None or ctx_id not in contexts:
            continue
        ctx_params = dict(contexts[int(ctx_id)])
        if ep.get("mario_inertia", None) is not None:
            ctx_params["mario_inertia"] = float(ep["mario_inertia"])
        episodes.append(
            {
                "timesteps": int(ep.get("timesteps", 0)),
                "episode": int(ep.get("episode", 0)),
                "context_id": int(ctx_id),
                "context_params": ctx_params,
                "return": float(ep.get("return", 0.0)),
                "length": int(ep.get("length", 0)),
                "completion": ep.get("completion", None),
                "success": ep.get("success", None),
            }
        )

    returns = [float(r["return"]) for r in episodes]
    lengths = [int(r["length"]) for r in episodes]
    completions = [
        float(r["completion"]) for r in episodes if r.get("completion", None) is not None
    ]
    successes = [
        1.0 for r in episodes if r.get("success", None) is True
    ] + [
        0.0 for r in episodes if r.get("success", None) is False
    ]

    summary: Dict[str, Any] = {
        "tag": tag,
        "n_episodes": int(len(episodes)),
        "mean_return": float(np.mean(returns)) if returns else float("nan"),
        "std_return": float(np.std(returns)) if returns else float("nan"),
        "mean_completion": float(np.mean(completions)) if completions else float("nan"),
        "std_completion": float(np.std(completions)) if completions else float("nan"),
        "mean_length": float(np.mean(lengths)) if lengths else float("nan"),
        "success_rate": float(np.mean(successes)) if successes else float("nan"),
    }

    # Per context summary (episode-weighted within context)
    per_context: Dict[int, Dict[str, Any]] = defaultdict(lambda: {"returns": [], "completions": [], "successes": [], "lengths": [], "inertias": []})
    for r in episodes:
        ctx_id = int(r["context_id"])
        per_context[ctx_id]["returns"].append(float(r["return"]))
        per_context[ctx_id]["lengths"].append(int(r["length"]))
        if r.get("completion", None) is not None:
            per_context[ctx_id]["completions"].append(float(r["completion"]))
        if r.get("success", None) is True:
            per_context[ctx_id]["successes"].append(1.0)
        elif r.get("success", None) is False:
            per_context[ctx_id]["successes"].append(0.0)
        inertia = r.get("context_params", {}).get("mario_inertia", None)
        if inertia is not None:
            per_context[ctx_id]["inertias"].append(float(inertia))

    per_context_out: Dict[int, Dict[str, Any]] = {}
    for ctx_id, vals in per_context.items():
        per_context_out[int(ctx_id)] = {
            "mean_return": float(np.mean(vals["returns"])) if vals["returns"] else float("nan"),
            "mean_completion": float(np.mean(vals["completions"])) if vals["completions"] else float("nan"),
            "success_rate": float(np.mean(vals["successes"])) if vals["successes"] else float("nan"),
            "mean_length": float(np.mean(vals["lengths"])) if vals["lengths"] else float("nan"),
            "mean_inertia": float(np.mean(vals["inertias"])) if vals["inertias"] else float("nan"),
            "n_episodes": int(len(vals["returns"])),
        }
    summary["per_context"] = per_context_out

    # Uniform-per-context aggregates (each context weighted equally)
    ctx_mean_returns = [v["mean_return"] for v in per_context_out.values() if not np.isnan(v["mean_return"])]
    ctx_mean_completions = [v["mean_completion"] for v in per_context_out.values() if not np.isnan(v["mean_completion"])]
    ctx_success_rates = [v["success_rate"] for v in per_context_out.values() if not np.isnan(v["success_rate"])]
    summary["uniform_ctx_mean_return"] = float(np.mean(ctx_mean_returns)) if ctx_mean_returns else float("nan")
    summary["uniform_ctx_mean_completion"] = float(np.mean(ctx_mean_completions)) if ctx_mean_completions else float("nan")
    summary["uniform_ctx_success_rate"] = float(np.mean(ctx_success_rates)) if ctx_success_rates else float("nan")

    # Per inertia summary (episode-weighted across all maps with that inertia)
    inertia_stats: Dict[float, Dict[str, list[float]]] = defaultdict(lambda: {"returns": [], "completions": [], "successes": [], "lengths": []})
    for r in episodes:
        inertia = float(r["context_params"].get("mario_inertia", float("nan")))
        inertia_stats[inertia]["returns"].append(float(r["return"]))
        inertia_stats[inertia]["lengths"].append(int(r["length"]))
        if r.get("completion", None) is not None:
            inertia_stats[inertia]["completions"].append(float(r["completion"]))
        if r.get("success", None) is True:
            inertia_stats[inertia]["successes"].append(1.0)
        elif r.get("success", None) is False:
            inertia_stats[inertia]["successes"].append(0.0)

    per_inertia: Dict[float, Dict[str, Any]] = {}
    for inertia, vals in inertia_stats.items():
        per_inertia[float(inertia)] = {
            "mean_return": float(np.mean(vals["returns"])) if vals["returns"] else float("nan"),
            "mean_completion": float(np.mean(vals["completions"])) if vals["completions"] else float("nan"),
            "success_rate": float(np.mean(vals["successes"])) if vals["successes"] else float("nan"),
            "mean_length": float(np.mean(vals["lengths"])) if vals["lengths"] else float("nan"),
            "n_episodes": int(len(vals["returns"])),
        }
    summary["per_inertia"] = per_inertia

    # Per training range summary (if train_range_index is available)
    per_range_stats: Dict[int, Dict[str, list[float]]] = defaultdict(lambda: {"returns": [], "completions": [], "successes": [], "lengths": [], "inertias": []})
    for r in episodes:
        ctx_id = int(r["context_id"])
        ridx = contexts.get(ctx_id, {}).get("train_range_index", None)
        if ridx is None:
            continue
        ridx_i = int(ridx)
        per_range_stats[ridx_i]["returns"].append(float(r["return"]))
        per_range_stats[ridx_i]["lengths"].append(int(r["length"]))
        if r.get("completion", None) is not None:
            per_range_stats[ridx_i]["completions"].append(float(r["completion"]))
        if r.get("success", None) is True:
            per_range_stats[ridx_i]["successes"].append(1.0)
        elif r.get("success", None) is False:
            per_range_stats[ridx_i]["successes"].append(0.0)
        inertia = r.get("context_params", {}).get("mario_inertia", None)
        if inertia is not None:
            per_range_stats[ridx_i]["inertias"].append(float(inertia))

    per_range_out: Dict[int, Dict[str, Any]] = {}
    for ridx, vals in per_range_stats.items():
        per_range_out[int(ridx)] = {
            "mean_return": float(np.mean(vals["returns"])) if vals["returns"] else float("nan"),
            "mean_completion": float(np.mean(vals["completions"])) if vals["completions"] else float("nan"),
            "success_rate": float(np.mean(vals["successes"])) if vals["successes"] else float("nan"),
            "mean_length": float(np.mean(vals["lengths"])) if vals["lengths"] else float("nan"),
            "mean_inertia": float(np.mean(vals["inertias"])) if vals["inertias"] else float("nan"),
            "n_episodes": int(len(vals["returns"])),
        }
    summary["per_range"] = per_range_out

    return summary


def train(config: TrainingConfig):
    log("=" * 50)
    log("SETTING UP TRAINING")
    log("=" * 50)

    (
        model,
        train_env,
        eval_env,
        train_eval_env,
        callbacks,
        train_contexts,
        train_eval_contexts,
        test_contexts,
        run_dir,
        train_printer,
        train_inertia_values,
    ) = setup_training(config)

    log("Training Configuration:")
    log(f"  Train contexts: {config.n_train_maps}")
    log(f"  Test contexts: {config.n_test_maps}")
    log(f"  Parallel envs: {config.n_envs}")
    log(f"  Seed: {config.seed}")
    log(f"  Level width: {config.level_width}")
    log(f"  Total timesteps: {config.total_timesteps:,}")
    log(f"  Device: {config.device}")
    log(f"  Use context in obs: {config.use_context}")
    if config.use_context:
        log(f"  Context keys: {list(config.context_keys)}")
        log(f"  Feature fusion: {config.feature_fusion}")
    log(f"  Test on train maps: {config.test_on_train_maps}")
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

    training_curve = getattr(train_printer, "history", [])
    episode_history = getattr(train_printer, "episode_history", [])
    training_episode_summary = summarize_training_episodes(
        episode_history=episode_history,
        contexts=train_contexts,
        tag="training_rollout",
    )
    try:
        curve_path = run_dir / "training_curve.json"
        with curve_path.open("w") as f:
            json.dump(training_curve, f, indent=2)
        log(f"Saved training curve to {curve_path}")
    except Exception as e:
        log(f"Could not save training curve: {e}")

    try:
        ep_path = run_dir / "episode_history.json"
        with ep_path.open("w") as f:
            json.dump(episode_history, f, indent=2)
        log(f"Saved episode history to {ep_path}")
    except Exception as e:
        log(f"Could not save episode history: {e}")

    log("=" * 50)
    log("FINAL EVALUATION")
    log("=" * 50)
    final_stoch = evaluate_final(
        model,
        eval_env,
        test_contexts,
        n_episodes=config.n_eval_episodes,
        deterministic=False,
        tag="stochastic",
    )

    final_train_stoch = None
    if config.final_eval_on_train_contexts:
        log("=" * 50)
        log("FINAL EVALUATION (TRAIN CONTEXTS)")
        log("=" * 50)
        final_train_stoch = evaluate_final(
            model,
            train_eval_env,
            train_eval_contexts,
            n_episodes=int(config.train_final_eval_n_episodes),
            deterministic=False,
            tag="train_stochastic",
            inertia_ranges=config.train_inertia_ranges,
            summarize_inertia_by_ranges=True,
        )

    final_results = {
        "stochastic": final_stoch,
        "train_stochastic": final_train_stoch,
        "training_rollout": training_episode_summary,
        "training_curve": training_curve,
        "episode_history": episode_history,
    }
    if train_inertia_values is not None:
        final_results["train_inertia_values"] = train_inertia_values
    if config.resample_train_inertia_each_episode and config.final_eval_on_train_contexts:
        final_results["train_final_eval_n_inertias"] = int(config.train_final_eval_n_inertias)
        final_results["train_final_eval_n_episodes"] = int(config.train_final_eval_n_episodes)

    try:
        out_path = run_dir / "final_results.json"
        with out_path.open("w") as f:
            json.dump(final_results, f, indent=2)
        log(f"Saved final results to {out_path}")
    except Exception as e:
        log(f"Could not save final results JSON: {e}")

    if config.save_final_model:
        model_path = run_dir / "final_model.zip"
        model.save(model_path)

        try:
            train_env.save(run_dir / "vec_normalize.pkl")
        except Exception:
            pass
        log(f"Model saved to {model_path}")

    # --- make closing non-fatal ---
    for env, name in (
        (train_env, "train_env"),
        (eval_env, "eval_env"),
        (train_eval_env, "train_eval_env"),
    ):
        try:
            env.close()
        except XStartError as e:
            log(f"Ignoring XStartError when closing {name}: {e}")
        except Exception as e:
            log(f"Non-fatal error when closing {name}: {e}")

    return model, final_results, run_dir


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
    # You can toggle these manually if you want:
    # config.use_context = True          # to feed mario_inertia as input
    # config.feature_fusion = "hadamard"  # cGate/Hadamard fusion (requires use_context=True)
    # config.test_on_train_maps = True   # to test only physics shift on same maps

    print("FAST ITERATION MODE")  # using print here because log isn't setup yet

    try:
        model, results, run_dir = train(config)
        log("=" * 50)
        log("TRAINING COMPLETED!")
        log("=" * 50)
        log(f"All outputs saved to: {run_dir}")
        log(f"  - Text Log: {run_dir}/training_output.txt")
        log(f"  - Model: {run_dir}/final_model.zip")
        log(f"  - TensorBoard: {run_dir}/tb_1")
        log(f"Run: tensorboard --logdir {config.log_dir}")
    finally:
        if LOGGER:
            LOGGER.close()


if __name__ == "__main__":
    main()
