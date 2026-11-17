"""Training / evaluation helpers shared by the notebook and the scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from stable_baselines3 import A2C, PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from rocketlander.env import RewardFn, make_env

ALGOS = {"PPO": PPO, "A2C": A2C, "SAC": SAC}

# Small MLP policies on CPU: extra torch threads only add contention (measured ~1.6x
# slower with the default thread count). Envs are stepped in-process for the same reason.
torch.set_num_threads(1)


class LandingStatsCallback(BaseCallback):
    """Logs landing statistics to TensorBoard at the end of every episode and, if
    `progress_path` is given, keeps a small JSON file with the learning curve (read by
    the web app)."""

    def __init__(self, window: int = 100, progress_path: Optional[str] = None,
                 total_timesteps: Optional[int] = None, every: int = 4096):
        super().__init__()
        self.window = window
        self.progress_path = Path(progress_path) if progress_path else None
        self.total_timesteps = total_timesteps
        self.every = every
        self.outcomes: list[dict] = []
        self.history: list[dict] = []
        self._last_write = 0

    def _rates(self) -> dict:
        recent = self.outcomes[-self.window :]
        if not recent:
            return {}
        out = {f"{k}_rate": float(np.mean([o[k] for o in recent]))
               for k in ("success", "crash", "timeout", "touch_down")}
        out["episode_return"] = float(np.mean([o["return"] for o in recent]))
        ok = [o for o in recent if o["success"]]
        if ok:
            out["abs_x_when_landed"] = float(np.mean([o["abs_x"] for o in ok]))
            out["fuel_left_when_landed"] = float(np.mean([o["fuel"] for o in ok]))
        return out

    def _on_step(self) -> bool:
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if not done:
                continue
            self.outcomes.append(
                {
                    "success": float(info.get("landed_full", False)),
                    "crash": float(info.get("game_over", False)),
                    "timeout": float(info.get("timeout", False) or info.get("TimeLimit.truncated", False)),
                    "touch_down": float(info.get("touch_down", False)),
                    "abs_x": abs(info.get("x", np.nan)),
                    "fuel": info.get("fuel", np.nan),
                    "return": info.get("episode", {}).get("r", np.nan),
                }
            )
            for key, value in self._rates().items():
                self.logger.record(f"landing/{key}", value)
        if self.progress_path and self.num_timesteps - self._last_write >= self.every:
            self._write()
        return True

    def _on_training_end(self) -> None:
        if self.progress_path:
            self._write(status="done")

    def _write(self, status: str = "running") -> None:
        self._last_write = self.num_timesteps
        rates = self._rates()
        if rates:
            self.history.append({"t": self.num_timesteps, **rates})
        data = {"status": status, "timesteps": self.num_timesteps,
                "total_timesteps": self.total_timesteps, "episodes": len(self.outcomes),
                "latest": rates, "history": self.history}
        tmp = self.progress_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data), encoding="utf8")
        tmp.replace(self.progress_path)


def make_vec(
    level: int = 0,
    continuous: bool = True,
    reward_fn: Optional[RewardFn] = None,
    n_envs: int = 4,
    seed: int = 0,
    subprocess: bool = False,
    reward_params: Optional[dict] = None,
    frame_skip: int = 4,
    config: Optional[dict] = None,
):
    return make_vec_env(
        make_env,
        n_envs=n_envs,
        seed=seed,
        env_kwargs=dict(level=level, continuous=continuous, reward_fn=reward_fn,
                        reward_params=reward_params, frame_skip=frame_skip, config=config),
        vec_env_cls=SubprocVecEnv if subprocess else None,
    )


def evaluate(model, level=0, continuous=True, n_episodes=100, seed=10_000, reward_fn=None,
             reward_params=None, frame_skip=4, config=None):
    """Runs the deterministic policy and returns a dict of landing statistics."""
    env = make_env(level=level, continuous=continuous, reward_fn=reward_fn, reward_params=reward_params,
                   frame_skip=frame_skip, config=config)
    stats = {"success": 0, "crash": 0, "timeout": 0, "returns": [], "abs_x_landed": []}
    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        done, ret = False, 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(action)
            ret += r
            done = term or trunc
        stats["returns"].append(ret)
        if info["landed_full"]:
            stats["success"] += 1
            stats["abs_x_landed"].append(abs(info["x"]))
        elif info["game_over"]:
            stats["crash"] += 1
        else:
            stats["timeout"] += 1
    env.close()
    n = n_episodes
    return {
        "success_rate": stats["success"] / n,
        "crash_rate": stats["crash"] / n,
        "timeout_rate": stats["timeout"] / n,
        "mean_return": float(np.mean(stats["returns"])),
        "mean_abs_x_when_landed": float(np.mean(stats["abs_x_landed"])) if stats["abs_x_landed"] else None,
    }


def record_episode(model, path, level=0, continuous=True, seed=0, every=1, frame_skip=4,
                   config=None, reward_fn=None):
    """Plays one episode, saves it as .mp4 or .gif, returns (path, trajectory)."""
    import imageio.v2 as imageio

    env = make_env(level=level, continuous=continuous, render_mode="rgb_array", frame_skip=frame_skip,
                   config=config, reward_fn=reward_fn)
    obs, info = env.reset(seed=seed)
    frames, traj, done, t = [], [], False, 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(action)
        traj.append(dict(info, reward=r, action=np.asarray(action).tolist()))
        if t % every == 0 or term or trunc:
            frames.append(env.render())
        done = term or trunc
        t += 1
    env.close()
    frames += [frames[-1]] * 15  # hold the last frame
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fps = 60 // (every * frame_skip)
    if path.suffix == ".gif":
        imageio.mimsave(path, frames, duration=1 / fps, loop=0)
    else:
        imageio.mimsave(path, frames, fps=fps, macro_block_size=1)
    return path, traj
