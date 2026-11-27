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
                 total_timesteps: Optional[int] = None, every: int = 4096, keep_episodes: int = 40):
        super().__init__()
        self.keep_episodes = keep_episodes
        self.episodes: list[dict] = []  # last episodes with their trajectory (episodes.json)
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
        judged = [o["challenge"] for o in recent if o["challenge"] is not None]
        if judged:
            out["challenge_rate"] = float(np.mean(judged))
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
            summary = info.get("summary") or {}
            verdict = (info.get("challenge") or {}).get("ok")
            self.outcomes.append(
                {
                    "challenge": None if verdict is None else float(verdict),
                    "success": float(summary.get("landed", info.get("landed_full", False))),
                    "crash": float(info.get("game_over", False)),
                    "timeout": float(info.get("timeout", False) or info.get("TimeLimit.truncated", False)),
                    "touch_down": float(info.get("touch_down", False)),
                    "abs_x": abs(info.get("x", np.nan)),
                    "fuel": info.get("fuel", np.nan),
                    "return": info.get("episode", {}).get("r", np.nan),
                }
            )
            if summary.get("trajectory"):
                self.episodes.append({
                    "t": self.num_timesteps, "outcome": summary.get("outcome"),
                    "challenge": info.get("challenge"), "return": info.get("episode", {}).get("r"),
                    "trajectory": summary["trajectory"],
                })
                self.episodes = self.episodes[-self.keep_episodes :]
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
        _write_json(self.progress_path, data)
        if self.episodes:
            _write_json(self.progress_path.with_name("episodes.json"), self.episodes)


class LiveSnapshotCallback(BaseCallback):
    """Saves the current policy every `every` steps to live_model.zip, so the web app can
    show the agent playing while it learns."""

    def __init__(self, out_dir, every: int = 16_384):
        super().__init__()
        self.out_dir = Path(out_dir)
        self.every = every
        self._last = -10**9
        self.count = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last >= self.every:
            self._save()
        return True

    def _on_training_end(self) -> None:
        self._save()

    def _save(self) -> None:
        self._last = self.num_timesteps
        self.count += 1
        tmp = self.out_dir / "live_model.tmp.zip"
        self.model.save(tmp)
        _replace(tmp, self.out_dir / "live_model.zip")
        _write_json(self.out_dir / "live.json", {"timesteps": self.num_timesteps, "snapshot": self.count})


def _write_json(path: Path, data) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data), encoding="utf8")
    _replace(tmp, path)


def _replace(src: Path, dst: Path, attempts: int = 20) -> None:
    """Atomic replace that survives Windows briefly locking the target (antivirus scan,
    the web app reading it): retry, and give up silently on this snapshot if it stays locked."""
    import time

    for i in range(attempts):
        try:
            src.replace(dst)
            return
        except PermissionError:
            time.sleep(0.05 * (i + 1))
    src.unlink(missing_ok=True)


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
    challenge: Optional[str] = None,
    keep_trajectory: bool = False,
):
    return make_vec_env(
        make_env,
        n_envs=n_envs,
        seed=seed,
        env_kwargs=dict(level=level, continuous=continuous, reward_fn=reward_fn,
                        reward_params=reward_params, frame_skip=frame_skip, config=config,
                        challenge=challenge, keep_trajectory=keep_trajectory),
        vec_env_cls=SubprocVecEnv if subprocess else None,
    )


def evaluate(model, level=0, continuous=True, n_episodes=100, seed=10_000, reward_fn=None,
             reward_params=None, frame_skip=4, config=None, challenge=None):
    """Runs the deterministic policy and returns a dict of landing statistics (and the
    challenge success rate when a challenge is given)."""
    env = make_env(level=level, continuous=continuous, reward_fn=reward_fn, reward_params=reward_params,
                   frame_skip=frame_skip, config=config, challenge=challenge)
    stats = {"success": 0, "crash": 0, "timeout": 0, "returns": [], "abs_x_landed": [], "challenge": []}
    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        done, ret = False, 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(action)
            ret += r
            done = term or trunc
        stats["returns"].append(ret)
        verdict = (info.get("challenge") or {}).get("ok")
        if verdict is not None:
            stats["challenge"].append(float(verdict))
        if (info.get("summary") or {}).get("landed", info["landed_full"]):
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
        "challenge": challenge,
        "challenge_rate": float(np.mean(stats["challenge"])) if stats["challenge"] else None,
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
