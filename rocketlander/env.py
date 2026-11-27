"""Gymnasium wrapper around the legacy gym RocketLander environment.

The physics, the difficulty levels and the default reward live in
gym/envs/box2d/rocket_lander.py. This module only adapts the API
(reset -> (obs, info), step -> 5-tuple) and lets you plug a custom reward:

    def my_reward(s, base_reward):
        # s: dict of named quantities (see RocketLander.step "info"), e.g.
        # s["x"], s["velocity_y"], s["angle"], s["landed"], s["game_over"], s["fuel"]...
        return base_reward - 0.1 * abs(s["x"])

    env = make_env(level=3, reward_fn=my_reward)
"""

from __future__ import annotations

from typing import Callable, Optional

import gymnasium
import numpy as np
from gymnasium import spaces
from gymnasium.wrappers import TimeLimit

from gym.envs.box2d.rocket_lander import FPS, MAX_STEPS, RocketLander
from rocketlander.episode import EpisodeTracker

RewardFn = Callable[[dict, float], float]

STATE_NAMES = [
    "x", "y", "angle", "left_leg_contact", "right_leg_contact",
    "throttle", "gimbal", "velocity_x", "velocity_y", "angular_velocity",
]


class RocketLanderEnv(gymnasium.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": FPS}

    def __init__(
        self,
        level: int = 0,
        continuous: bool = True,
        reward_fn: Optional[RewardFn] = None,
        render_mode: Optional[str] = None,
        reward_params: Optional[dict] = None,
        frame_skip: int = 4,
        config: Optional[dict] = None,
        challenge: Optional[str] = None,
        keep_trajectory: bool = False,
    ):
        """
        reward_params: overrides of the default reward weights (REWARD in rocket_lander.py).
        config: overrides of the scenario (DEFAULT_CONFIG in rocket_lander.py): zone width,
            start position / speed, touchdown speed limit, end_on_landing...
        frame_skip: the agent decides every `frame_skip` physics frames (60 frames = 1 s)
            and its action is repeated in between. Rewards of those frames are summed.
            Fewer decisions per episode = shorter credit-assignment horizon = much easier
            to learn.
        challenge: id of a challenge (rocketlander.challenges) used to judge each episode:
            the final info holds "summary" and "challenge" = {"ok", "text"}.
        keep_trajectory: also keep a light trajectory in the summary (for visualisation).
        """
        assert render_mode in (None, *self.metadata["render_modes"])
        self.level = level
        self.continuous = continuous
        self.reward_fn = reward_fn
        self.render_mode = render_mode
        self.frame_skip = frame_skip
        self.memory: dict = {}
        self.challenge = challenge
        self.tracker = EpisodeTracker(keep_trajectory=keep_trajectory)
        self.lander = RocketLander(level_number=level, continuous=continuous,
                                   reward_params=reward_params, config=config)
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=self.lander.observation_space.shape, dtype=np.float32
        )
        if continuous:
            # [gimbal, throttle, side thruster], each in [-1, 1]
            self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        else:
            # 0/1 gimbal +/-, 2/3 throttle +/-, 4/5 side thruster left/right, 6 nothing
            self.action_space = spaces.Discrete(7)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self.lander.seed(seed)
        obs = self.lander.reset()
        self.memory = {}  # free scratch space for custom reward functions, reset every episode
        self.tracker.reset()
        if self.render_mode == "human":
            self.render()
        return obs, dict(self.lander.info)

    def step(self, action):
        total, base_total = 0.0, 0.0
        for _ in range(self.frame_skip):
            obs, base, terminated, info = self.lander.step(action)
            base_total += base
            self.tracker.update(info)
            if self.reward_fn is not None:
                info = dict(info, memory=self.memory)
                total += float(self.reward_fn(info, base))
            else:
                total += base
            if self.render_mode == "human":
                self.render()
            if terminated:
                break
        info = dict(info, base_reward=base_total)
        info.pop("memory", None)
        if terminated:
            from rocketlander.challenges import judge

            info["summary"] = summary = self.tracker.summary()
            info["challenge"] = judge(self.challenge, summary)
        return obs, total, terminated, False, info

    def render(self):
        if self.render_mode is None:
            return None
        return self.lander.render(mode=self.render_mode)

    def close(self):
        self.lander.close()


def make_env(
    level: int = 0,
    continuous: bool = True,
    reward_fn: Optional[RewardFn] = None,
    render_mode: Optional[str] = None,
    max_episode_steps: Optional[int] = None,
    reward_params: Optional[dict] = None,
    frame_skip: int = 4,
    config: Optional[dict] = None,
    challenge: Optional[str] = None,
    keep_trajectory: bool = False,
) -> gymnasium.Env:
    """Create a RocketLander env. Episodes last at most MAX_STEPS physics frames (the
    env ends them itself, as a failure); `max_episode_steps` (in agent decisions) is an
    extra safety limit."""
    env = RocketLanderEnv(level, continuous, reward_fn, render_mode, reward_params, frame_skip, config,
                          challenge, keep_trajectory)
    if max_episode_steps is None:
        max_episode_steps = env.lander.config["max_steps"] // frame_skip + 10
    return TimeLimit(env, max_episode_steps=max_episode_steps)


gymnasium.register(
    id="RocketLander-v0",
    entry_point="rocketlander.env:RocketLanderEnv",
    max_episode_steps=MAX_STEPS // 4 + 10,
)
