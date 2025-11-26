from __future__ import annotations
from typing import List
import numpy as np
import pygame
import time

from carl.context.context_space import (
    CategoricalContextFeature,
    ContextFeature,
    UniformFloatContextFeature,
    UniformIntegerContextFeature,
)
from carl.context.selection import AbstractSelector
from carl.envs.carl_env import CARLEnv
from carl.envs.mario.pcg_smb_env import MarioEnv
from carl.envs.mario.pcg_smb_env.toadgan.toad_gan import generate_level
from carl.utils.types import Contexts

LEVEL_HEIGHT = 16


class CARLMarioEnv(CARLEnv):
    metadata = {
        "render_modes": ["rgb_array", "tiny_rgb_array"],
        "render_fps": 24,
    }

    def __init__(
        self,
        env: MarioEnv = None,
        contexts: Contexts | None = None,
        obs_context_features: (
            list[str] | None
        ) = None,
        obs_context_as_dict: bool = True,
        context_selector: AbstractSelector | type[AbstractSelector] | None = None,
        context_selector_kwargs: dict = None,
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

    def _update_context(self) -> None:
        self.env: MarioEnv
        self.context = CARLMarioEnv.get_context_space().insert_defaults(self.context)
        if not self.levels:
            for context in self.contexts.values():
                level, _ = generate_level(
                    width=context["level_width"],
                    height=LEVEL_HEIGHT,
                    level_index=context["level_index"],
                    seed=context["noise_seed"],
                    filter_unplayable=True,
                )
                self.levels.append(level)
        self.env.mario_state = self.context["mario_state"]
        
        # <-- Manually override inertia here -->
        #self.env.mario_inertia = 0.95 
        print("self.env.mario_inertia ", self.env.mario_inertia)
        self.env.levels = [self.levels[self.context_id]]


    @staticmethod
    def get_context_features() -> dict[str, ContextFeature]:
        return {
            "level_width": UniformIntegerContextFeature(
                "level_width", 16, 1000, default_value=100
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
                "mario_inertia", lower=0.5, upper=15, default_value=0.89
            ),
        }


# --- Build a few example contexts ---
def build_mario_contexts(n=3, width=100):
    ctxs = {}
    for i in range(n):
        ctxs[i] = {
            "level_width": width,
            "level_index": i % 14,
            "noise_seed": np.random.randint(0, 2**31 - 1),
            "mario_state": 0,
            "mario_inertia": 0.89,
        }
    return ctxs


# NES-like button order
BUTTONS = ["B", "A", "SELECT", "START", "UP", "DOWN", "LEFT", "RIGHT"]
IDX = {b: i for i, b in enumerate(BUTTONS)}


def make_action(env, keys):
    """
    Map keyboard input to discrete actions (0-9)
    """
    space = env.action_space
    
    # This environment uses Discrete(10), not MultiBinary
    if hasattr(space, "n") and type(space).__name__ == "Discrete":
        right = keys[pygame.K_RIGHT]
        left = keys[pygame.K_LEFT]
        down = keys[pygame.K_DOWN]
        jump = keys[pygame.K_SPACE]
        speed = keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] or keys[pygame.K_z]
        
        # Priority order (most complex action first)
        if right and speed and jump:
            return 5  # RIGHTSPEEDJUMP
        if right and jump:
            return 4  # RIGHTJUMP
        if right and speed:
            return 3  # RIGHTSPEED
        if right:
            return 2  # RIGHT
        
        if left and speed and jump:
            return 8  # LEFTSPEEDJUMP
        if left and jump:
            return 7  # LEFTJUMP
        if left:
            return 6  # LEFT
        
        if jump:
            return 9  # JUMP
        
        if down:
            return 1  # DOWN
        
        return 0  # NOOP
    
    # Fallback
    return space.sample()

def open_pygame_window(frame_shape, scale=3, caption="CARL Mario"):
    (h, w) = frame_shape[:2]
    screen = pygame.display.set_mode((w * scale, h * scale))
    pygame.display.set_caption(caption)
    return screen, (w, h, scale)


def blit_frame(screen, frame, dims):
    w, h, scale = dims
    surf = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
    surf = pygame.transform.scale(surf, (w * scale, h * scale))
    screen.blit(surf, (0, 0))
    pygame.display.flip()


if __name__ == "__main__":
    contexts = build_mario_contexts(n=3, width=100)
    env = CARLMarioEnv(contexts=contexts)

    pygame.init()
    clock = pygame.time.Clock()
    fps = env.metadata.get("render_fps", 24)

    obs, info = env.reset()
    frame = env.render()
    if frame is None:
        frame = env.render()

    screen, dims = open_pygame_window(
        frame.shape,
        scale=3,
        caption=f"CARL Mario | inertia={env.env.mario_inertia:.2f}"
    )

    running = True
    total_reward = 0.0
    steps = 0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_ESCAPE]:
            running = False
        INERTIA_MIN = 0.5
        INERTIA_MAX = 2
        INERTIA_STEP = 0.5

        
        if keys[pygame.K_a]:  # decrease inertia
            env.env.mario_inertia = max(INERTIA_MIN, env.env.mario_inertia - INERTIA_STEP)
            print(env.env.mario_inertia)
            env.reset()
        if keys[pygame.K_b]:  # increase inertia
            env.env.mario_inertia = min(INERTIA_MAX, env.env.mario_inertia + INERTIA_STEP)
            print(env.env.mario_inertia)

        action = make_action(env, keys)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

        frame = env.render()
        if frame is not None:
            blit_frame(screen, frame, dims)

        if steps % 120 == 0 or terminated or truncated:
            print(f"steps={steps} total_reward={total_reward:+.2f}")

        if terminated or truncated:
            obs, info = env.reset()
            total_reward = 0.0
            steps = 0

        clock.tick(fps)

    env.close()
    pygame.quit()