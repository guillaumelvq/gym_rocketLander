"""Rocket RL Lab: local web app to watch, compare, play and train rocket landing agents.

    uv run python -m rocketlander.app          # http://127.0.0.1:8000

Runs only on localhost: it executes the reward code typed in the page.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import gym.envs.box2d.rocket_lander as RL
from rocketlander.autopilot import Autopilot
from rocketlander.challenges import CHALLENGES, CodeReward, compile_reward, judge
from rocketlander.controls import KeyboardPilot
from rocketlander.episode import EpisodeTracker
from rocketlander.presets import AUTOPILOT_SUCCESS, LEVEL_HELP, PRESETS, level_summary

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs"
PLAYERS = ROOT / "players"  # human attempts (attempts.jsonl)
STATIC = Path(__file__).parent / "static"
FPS = RL.FPS

app = FastAPI(title="Rocket RL Lab")
app.mount("/static", StaticFiles(directory=STATIC), name="static")

CONFIG_DOC = {
    "ship_width": ("Largeur de la zone", 0.2, 2.0, 0.05, "x la largeur normale (~100 m)"),
    "start_x": ("Position de départ", -0.4, 0.4, 0.01, "fraction de l'écran, 0 = au-dessus du bateau"),
    "start_x_spread": ("Dispersion du départ", 0.0, 0.3, 0.01, "± aléatoire autour de la position"),
    "start_height": ("Hauteur de départ", 0.15, 0.97, 0.01, "fraction de l'écran (~385 m)"),
    "start_speed": ("Vitesse de chute initiale", 0.0, 60.0, 1.0, "m/s"),
    "start_spin": ("Rotation initiale max", 0.0, 2.0, 0.05, "rad/s, aléatoire"),
    "start_drift": ("Dérive horizontale", 0.0, 0.5, 0.01, "fraction de la vitesse initiale"),
    "max_touchdown_speed": ("Vitesse max au contact", 1.0, 20.0, 0.1, "m/s, au-delà les pieds cassent (vide = jamais)"),
    "hard_landing_breaks": ("Pieds fragiles", None, None, None, "cassent si les jambes sont trop comprimées"),
    "leg_scale": ("Longueur des jambes", 0.3, 1.5, 0.05, "x la longueur normale"),
    "end_on_landing": ("Fin à l'atterrissage", None, None, None, "l'épisode s'arrête après 1 s posé"),
    "max_steps": ("Durée max", 300, 3000, 60, "frames (60 = 1 s)"),
    "fuel": ("Carburant", 100, 1500, 10, "unités"),
}
REWARD_DOC = {
    "shaping": ("Guidage (distance + vitesse + inclinaison)", 0.0, 5.0, 0.1),
    "time": ("Pénalité par instant en vol", 0.0, 0.05, 0.001),
    "descent": ("Guidage « ralentir près du sol »", 0.0, 20.0, 0.5),
    "success": ("Bonus atterrissage", 0.0, 50.0, 1.0),
    "center": ("Bonus centre", 0.0, 20.0, 0.5),
    "crash": ("Pénalité crash", 0.0, 50.0, 1.0),
    "crash_speed": ("Pénalité crash × vitesse d'impact", 0.0, 30.0, 0.5),
    "timeout": ("Pénalité temps écoulé", 0.0, 50.0, 1.0),
}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


# --------------------------------------------------------------------------- metadata


@app.get("/api/meta")
def meta() -> dict:
    return {
        "default_config": RL.DEFAULT_CONFIG,
        "levels": {i: RL.level_config(i) for i in range(21)},
        "level_summaries": {i: level_summary(i) for i in range(21)},
        "level_help": LEVEL_HELP,
        "presets": {name: {**p, "autopilot_success": AUTOPILOT_SUCCESS.get(name)} for name, p in PRESETS.items()},
        "config_doc": CONFIG_DOC,
        "reward_defaults": RL.REWARD,
        "reward_doc": REWARD_DOC,
        "challenges": CHALLENGES,
        "cpu_count": os.cpu_count(),
        "max_parallel_trainings": MAX_PARALLEL,
    }


# --------------------------------------------------------------------------- runs / models


def _read_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf8"))
    except (OSError, ValueError):
        return None


def _run_info(d: Path) -> dict:
    meta_ = _read_json(d / "meta.json") or {"name": d.name}
    progress = _read_json(d / "progress.json") or {}
    proc = TRAININGS.get(d.name)
    if proc is not None and proc.poll() is None:
        status = "running"
    elif progress.get("status") == "done" or (d / "eval.json").exists():
        status = "done"
    elif d.name in STOPPED:
        status = "stopped"
    elif proc is not None:
        status = "failed"
    else:  # started outside the app (scripts/train.py) or app restarted
        status = "running" if progress and time.time() - (d / "progress.json").stat().st_mtime < 120 else "stopped"
    timing = _timing(d, meta_, progress, status)
    return {
        **meta_,
        "name": d.name,
        "status": status,
        "has_best": (d / "best_model.zip").exists(),
        "has_last": (d / "last_model.zip").exists(),
        "has_live": (d / "live_model.zip").exists(),
        "live": _read_json(d / "live.json"),
        **timing,
        "progress": progress,
        "eval": _read_json(d / "eval.json"),
        "reward_code": (d / "reward.py").read_text(encoding="utf8") if (d / "reward.py").exists() else None,
        "mtime": (d / "meta.json").stat().st_mtime if (d / "meta.json").exists() else d.stat().st_mtime,
    }


def _timing(d: Path, meta_: dict, progress: dict, status: str) -> dict:
    """Elapsed time and a rough ETA from the progress file."""
    started = START_TIMES.get(d.name)
    if started is None:
        try:
            started = time.mktime(time.strptime(meta_.get("created", ""), "%Y-%m-%dT%H:%M:%S"))
        except (ValueError, OverflowError):
            started = (d / "meta.json").stat().st_mtime if (d / "meta.json").exists() else None
    if started is None:
        return {"elapsed_s": None, "eta_s": None}
    end = time.time() if status == "running" else (
        (d / "progress.json").stat().st_mtime if (d / "progress.json").exists() else time.time())
    elapsed = max(0.0, end - started)
    done, total = progress.get("timesteps") or 0, progress.get("total_timesteps") or meta_.get("timesteps")
    eta = None
    if status == "running" and done > 0 and total:
        eta = max(0.0, (total - done) * elapsed / done)
    return {"elapsed_s": round(elapsed), "eta_s": None if eta is None else round(eta)}


@app.get("/api/runs")
def list_runs() -> list[dict]:
    if not RUNS.exists():
        return []
    dirs = [d for d in RUNS.iterdir() if d.is_dir() and (d / "meta.json").exists()]
    runs = [_run_info(d) for d in dirs]
    for r in runs:
        r["progress"] = {k: v for k, v in r["progress"].items() if k != "history"}
    return sorted(runs, key=lambda r: r["mtime"], reverse=True)


@app.get("/api/runs/{name}")
def get_run(name: str) -> dict:
    d = _run_dir(name)
    return _run_info(d)


@app.get("/api/runs/{name}/episodes")
def get_episodes(name: str) -> dict:
    """Last training episodes (outcome, challenge verdict, trajectory) written by train.py."""
    d = _run_dir(name)
    return {"episodes": _read_json(d / "episodes.json") or [], "live": _read_json(d / "live.json")}


@app.get("/api/runs/{name}/log")
def get_log(name: str) -> dict:
    log = _run_dir(name) / "train.log"
    text = log.read_text(encoding="utf8", errors="replace") if log.exists() else ""
    return {"log": text[-6000:]}


def _run_dir(name: str) -> Path:
    d = (RUNS / name).resolve()
    if d.parent != RUNS.resolve() or not d.is_dir():
        raise HTTPException(404, f"unknown run {name!r}")
    return d


# --------------------------------------------------------------------------- training

# Each training uses about one core (envs stepped in-process, torch limited to 1 thread).
MAX_PARALLEL = max(1, (os.cpu_count() or 2) - 2)
TRAININGS: dict[str, subprocess.Popen] = {}
STOPPED: set[str] = set()
START_TIMES: dict[str, float] = {}


class TrainRequest(BaseModel):
    name: Optional[str] = None
    level: int = 1
    continuous: bool = True
    config: dict[str, Any] = Field(default_factory=dict)
    reward: dict[str, float] = Field(default_factory=dict)
    reward_code: Optional[str] = None
    timesteps: int = 300_000
    algo: str = "PPO"
    gamma: float = 0.999
    n_envs: int = 16
    preset: Optional[str] = None  # scenario label shown in the page (metadata only)
    challenge: Optional[str] = None  # challenge used to judge the episodes (curves, verdicts)
    init_run: Optional[str] = None  # start from this run's model instead of a blank agent
    init_which: str = "best"  # which model of init_run: best | last | live


def _model_path(d: Path, which: str = "best") -> Optional[Path]:
    """Model file of a run: the requested one, else the best available."""
    for w in (which, "best", "last", "live"):
        p = d / f"{w}_model.zip"
        if p.exists():
            return p
    return None


@app.post("/api/train")
def start_training(req: TrainRequest) -> dict:
    running = [n for n, p in TRAININGS.items() if p.poll() is None]
    if len(running) >= MAX_PARALLEL:
        raise HTTPException(429, f"{len(running)} entraînements déjà en cours (max {MAX_PARALLEL})")
    unknown = set(req.config) - set(RL.DEFAULT_CONFIG)
    if unknown:
        raise HTTPException(400, f"paramètres inconnus : {sorted(unknown)}")
    if req.challenge is not None and req.challenge not in CHALLENGES:
        raise HTTPException(400, f"défi inconnu : {req.challenge!r}")
    init_path = None
    if req.init_run:
        init_dir = _run_dir(req.init_run)
        init_path = _model_path(init_dir, req.init_which)
        if init_path is None:
            raise HTTPException(400, f"le run {req.init_run!r} n'a pas de modèle sauvegardé")
        init_meta = _read_json(init_dir / "meta.json") or {}
        # same networks: the algorithm and the action space come from the starting agent
        req.algo = init_meta.get("algo", req.algo)
        req.continuous = init_meta.get("continuous", req.continuous)
    if req.reward_code:
        try:
            fn = compile_reward(req.reward_code)
            fn(_probe_info(req.level, req.config), 0.0)
        except Exception as exc:  # report any error of the user's code
            raise HTTPException(400, f"erreur dans la fonction reward : {type(exc).__name__}: {exc}")
    slug = "".join(c if c.isalnum() or c in "-_" else "_" for c in (req.name or "run")).strip("_")[:40] or "run"
    name = f"{slug}_{time.strftime('%Y%m%d-%H%M%S')}"
    out = RUNS / name
    out.mkdir(parents=True)
    cmd = [sys.executable, str(ROOT / "scripts" / "train.py"), "--name", name, "--algo", req.algo,
           "--level", str(req.level), "--timesteps", str(int(req.timesteps)), "--gamma", str(req.gamma),
           "--n-envs", str(req.n_envs), "--config", json.dumps(req.config), "--reward", json.dumps(req.reward),
           "--no-progress-bar"]
    if not req.continuous:
        cmd.append("--discrete")
    if req.preset:
        cmd += ["--preset", req.preset[:60]]
    if req.challenge:
        cmd += ["--challenge", req.challenge]
    if init_path is not None:
        cmd += ["--init-model", str(init_path)]
    # ~30 snapshots per training so the live view visibly evolves
    cmd += ["--snapshot-every", str(max(4096, int(req.timesteps) // 30))]
    if req.reward_code:
        (out / "reward.py").write_text(req.reward_code, encoding="utf8")
        cmd += ["--reward-code", str(out / "reward.py")]
    log = open(out / "train.log", "w", encoding="utf8")
    flags = subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 0
    TRAININGS[name] = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, creationflags=flags)
    START_TIMES[name] = time.time()
    return {"name": name, "init_model": None if init_path is None else init_path.name}


@app.post("/api/runs/{name}/stop")
def stop_training(name: str) -> dict:
    proc = TRAININGS.get(name)
    if proc is None or proc.poll() is not None:
        raise HTTPException(404, "pas d'entraînement en cours pour ce run")
    proc.terminate()
    STOPPED.add(name)
    return {"stopped": name}


@app.delete("/api/runs/{name}")
def delete_run(name: str) -> dict:
    d = _run_dir(name)
    proc = TRAININGS.get(name)
    if proc is not None and proc.poll() is None:
        raise HTTPException(409, "arrête l'entraînement avant de le supprimer")
    shutil.rmtree(d)
    return {"deleted": name}


def _probe_info(level: int, config: dict) -> dict:
    env = RL.RocketLander(level_number=level, config=config)
    env.seed(0)
    env.reset()
    return dict(env.info, memory={}, base_reward=0.0)


# --------------------------------------------------------------------------- human attempts

_attempts_lock = threading.Lock()


def _effective_config(level: int, config: dict) -> dict:
    """Full scenario of a human attempt (humans always use direct control)."""
    cfg = {**RL.level_config(level), **{k: v for k, v in (config or {}).items() if k in RL.DEFAULT_CONFIG}}
    cfg["control"] = "direct"
    return cfg


def scenario_signature(level: int, config: dict) -> str:
    """Identifies a scenario for the leaderboard: same signature = same rules and start."""
    cfg = _effective_config(level, config)
    # numbers normalized: the page sends 700 where Python has 700.0
    norm = {k: (round(float(v), 4) if isinstance(v, (int, float)) and not isinstance(v, bool) else v)
            for k, v in cfg.items()}
    blob = json.dumps(norm, sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def _rle_encode(actions: list[list[float]]) -> list[list]:
    out: list[list] = []
    for a in actions:
        a = [float(x) for x in a]
        if out and out[-1][1] == a:
            out[-1][0] += 1
        else:
            out.append([1, a])
    return out


def _rle_decode(rle: list[list]) -> list[list[float]]:
    return [list(a) for n, a in rle for _ in range(int(n))]


def _attempts_file() -> Path:
    return PLAYERS / "attempts.jsonl"


def load_attempts() -> list[dict]:
    path = _attempts_file()
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf8").splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def save_attempt(attempt: dict) -> None:
    with _attempts_lock:
        PLAYERS.mkdir(parents=True, exist_ok=True)
        with open(_attempts_file(), "a", encoding="utf8") as fh:
            fh.write(json.dumps(attempt) + "\n")


def _rank_key(a: dict):
    return (a["outcome"] != "success", -a["score"], a["duration"])


def _summary(a: dict) -> dict:
    return {k: v for k, v in a.items() if k not in ("actions", "trajectory")}


@app.get("/api/attempts")
def list_attempts(level: Optional[int] = None, config: Optional[str] = None, signature: Optional[str] = None,
                  limit: int = 50) -> dict:
    sig = signature
    if sig is None and level is not None:
        sig = scenario_signature(level, json.loads(config) if config else {})
    items = [a for a in load_attempts() if sig is None or a["signature"] == sig]
    items.sort(key=_rank_key)
    ranked = [dict(_summary(a), rank=i + 1) for i, a in enumerate(items[:limit])]
    return {"signature": sig, "count": len(items), "attempts": ranked}


@app.get("/api/attempts/{attempt_id}")
def get_attempt(attempt_id: str) -> dict:
    for a in load_attempts():
        if a["id"] == attempt_id:
            return a
    raise HTTPException(404, "tentative inconnue")


@app.delete("/api/attempts/{attempt_id}")
def delete_attempt(attempt_id: str) -> dict:
    with _attempts_lock:
        items = load_attempts()
        kept = [a for a in items if a["id"] != attempt_id]
        if len(kept) == len(items):
            raise HTTPException(404, "tentative inconnue")
        PLAYERS.mkdir(parents=True, exist_ok=True)
        _attempts_file().write_text("".join(json.dumps(a) + "\n" for a in kept), encoding="utf8")
    return {"deleted": attempt_id}


def _rank_of(attempt: dict) -> tuple[int, int]:
    same = [a for a in load_attempts() if a["signature"] == attempt["signature"]]
    same.sort(key=_rank_key)
    ids = [a["id"] for a in same]
    return (ids.index(attempt["id"]) + 1 if attempt["id"] in ids else len(ids)), len(ids)


def autopilot_trajectory(level: int, config: dict, seed: int, reward_params: Optional[dict] = None,
                         every: int = 3) -> dict:
    """Runs the autopilot headless on a scenario: the reference curve shown next to a human."""
    env = RL.RocketLander(level_number=level, config=_effective_config(level, config), reward_params=reward_params)
    env.seed(seed)
    obs = env.reset()
    pilot = Autopilot(env, target_touchdown_speed=1.0, margin=3.0)
    traj, ret, done = [], 0.0, False
    while not done:
        obs, r, done, info = env.step(pilot.predict(obs)[0])
        ret += r
        if env.stepnumber % every == 0 or done:
            traj.append([env.stepnumber, round(info["altitude"], 2), round(info["velocity_y"] * RL.START_SPEED, 2)])
    out = {"outcome": _outcome(env.info), "score": round(ret, 3), "trajectory": traj}
    env.close()
    return out


# --------------------------------------------------------------------------- live simulation


class Agent:
    """Something that picks an action from the current RocketLander state."""

    continuous = True
    frame_skip = 1

    def act(self, env: RL.RocketLander, obs: np.ndarray):
        raise NotImplementedError


class HumanAgent(Agent):
    """Keyboard pilot. The page sends the held keys; actions are computed here, frame by
    frame, so that the attempt can be recorded and replayed exactly."""

    def __init__(self, mode: str = "simple"):
        self.pilot = KeyboardPilot(mode)
        self.raw_action: Optional[list[float]] = None  # legacy {type: action} messages

    def act(self, env, obs):
        if self.raw_action is not None:
            return list(self.raw_action)
        return self.pilot.action(env)


class ReplayAgent(Agent):
    """Plays back the per-frame actions of a recorded human attempt."""

    def __init__(self, attempt: dict):
        self.attempt = attempt
        self.actions = _rle_decode(attempt["actions"])
        self._start: Optional[int] = None

    def act(self, env, obs):
        if self._start is None:
            self._start = env.stepnumber
        i = env.stepnumber - self._start
        return self.actions[i] if i < len(self.actions) else [0.0, -1.0, 0.0]


class RandomAgent(Agent):
    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    def act(self, env, obs):
        return self.rng.uniform(-1, 1, size=3).astype(np.float32)


class AutopilotAgent(Agent):
    def __init__(self, env: RL.RocketLander):
        self.pilot = Autopilot(env, target_touchdown_speed=1.0, margin=3.0)

    def act(self, env, obs):
        return self.pilot.predict(obs)[0]


class ModelAgent(Agent):
    _cache: dict[tuple, Any] = {}

    def __init__(self, run: str, which: str):
        from stable_baselines3 import A2C, PPO, SAC

        d = _run_dir(run)
        meta_ = _read_json(d / "meta.json") or {}
        if which == "live":
            # the latest snapshot of a training: reloaded whenever the file changes
            path = d / "live_model.zip"
            if not path.exists():
                raise HTTPException(404, "pas encore d'instantané : l'agent n'a pas fini son premier cycle")
        else:
            path = _model_path(d, which)
            if path is None:
                raise HTTPException(404, "ce run n'a pas encore de modèle sauvegardé")
        key = (str(path), path.stat().st_mtime)
        if key not in self._cache:
            cls = {"A2C": A2C, "SAC": SAC}.get(meta_.get("algo"), PPO)
            for attempt in range(5):  # the training may be rewriting the file right now
                try:
                    self._cache[key] = cls.load(path, device="cpu")
                    break
                except (OSError, ValueError, EOFError, KeyError):
                    if attempt == 4:
                        raise
                    time.sleep(0.2)
            # keep the cache small: live snapshots change every few seconds
            for k in [k for k in self._cache if k[0] == str(path) and k != key]:
                del self._cache[k]
        self.model = self._cache[key]
        self.meta = meta_
        self.continuous = meta_.get("continuous", True)
        self.frame_skip = int(meta_.get("frame_skip", 4))
        self.reward_code = (d / "reward.py").read_text(encoding="utf8") if (d / "reward.py").exists() else None
        self.snapshot = _read_json(d / "live.json") if which == "live" else None

    def act(self, env, obs):
        action, _ = self.model.predict(obs, deterministic=True)
        return action


def _poly(body) -> list[list[float]]:
    t = body.transform
    return [[round(c, 3) for c in (t * v)] for v in body.fixtures[0].shape.vertices]


def _world(env: RL.RocketLander) -> dict:
    static = lambda b: [[round(x, 3), round(y, 3)] for x, y in b.fixtures[0].shape.vertices]  # noqa: E731
    return {
        "W": RL.W, "H": RL.H,
        "water": static(env.water), "ship": static(env.ship),
        "containers": [static(c) for c in env.containers],
        "deck_y": env.helipad_y, "ship_x1": env.helipad_x1, "ship_x2": env.helipad_x2,
        "rocket_width": RL.ROCKET_WIDTH, "rocket_height": RL.ROCKET_HEIGHT,
        "engine_width": RL.ENGINE_WIDTH, "engine_height": RL.ENGINE_HEIGHT,
        "thruster_height": RL.THRUSTER_HEIGHT,
    }


def _frame(env: RL.RocketLander, ret: float, custom_ret: Optional[float]) -> dict:
    lander = env.lander
    struts = []
    for leg, side in zip(env.legs, (-1, 1)):
        a = lander.transform * (side * RL.ROCKET_WIDTH / 2, RL.ROCKET_HEIGHT / 8)
        b = leg.transform * (side * env.leg_length * 0.8, 0)
        struts.append([[round(a[0], 3), round(a[1], 3)], [round(b[0], 3), round(b[1], 3)]])
    i = env.info
    return {
        "type": "frame",
        "t": env.stepnumber,
        "lander": _poly(lander),
        "legs": [_poly(leg) for leg in env.legs],
        "struts": struts,
        "rocket": {"x": round(lander.position[0], 3), "y": round(lander.position[1], 3),
                   "angle": round(lander.angle, 4), "gimbal": round(float(env.gimbal), 4),
                   "power": round(float(env.power), 3), "side": env.force_dir},
        "hud": {
            "altitude": round(i["altitude"], 2),
            "vx": round(i["velocity_x"] * RL.START_SPEED, 2),
            "vy": round(i["velocity_y"] * RL.START_SPEED, 2),
            "angle_deg": round(math.degrees(lander.angle), 1),
            "fuel": round(i["fuel"], 3), "throttle": round(float(env.throttle), 3),
            "left": i["left_leg_contact"], "right": i["right_leg_contact"],
            "landed_ticks": i["landed_ticks"], "rotation": round(i["rotation"], 2),
            "x_on_ship": round(i["x_on_ship"], 3),
        },
        "return": round(ret, 3),
        "custom_return": None if custom_ret is None else round(custom_ret, 3),
    }


def _outcome(info: dict) -> str:
    if info.get("landed_full"):
        return "success"
    if info.get("broken_leg"):
        return "broken_leg"
    if info.get("outside"):
        return "outside"
    if info.get("game_over"):
        return "crash"
    return "timeout"


@app.websocket("/ws/sim")
async def simulate(ws: WebSocket) -> None:
    """One live episode per 'start' message. Client -> server messages:
    {type: start, agent: human|autopilot|random|model|replay, run, which, attempt, level, config,
     seed, speed, controls (simple|expert), player, countdown}
    {type: keys, keys: [up, down, left, right, side_left, side_right]}   (human agent)
    {type: go}   (starts a human episode after the countdown)
    {type: speed, value}  {type: stop}
    """
    await ws.accept()
    session: dict[str, Any] = {"task": None, "agent": None, "speed": 1.0}

    async def run_episode(msg: dict) -> None:
        try:
            agent_kind = msg.get("agent", "autopilot")
            seed = int(msg.get("seed", 0))
            level = int(msg.get("level", 1))
            config = {k: v for k, v in (msg.get("config") or {}).items() if k in RL.DEFAULT_CONFIG}
            reward_params = {k: float(v) for k, v in (msg.get("reward") or {}).items() if k in RL.REWARD}
            if agent_kind == "replay":
                attempt = get_attempt(str(msg.get("attempt")))
                agent: Agent = ReplayAgent(attempt)
                seed, level = int(attempt["seed"]), int(attempt["level"])
                config = attempt["config"]
                reward_params = attempt.get("reward_params") or {}
            elif agent_kind in ("model", "live"):
                agent: Agent = ModelAgent(msg["run"], "live" if agent_kind == "live" else msg.get("which", "best"))
                # a model is shown in the scenario it was trained on, unless the page overrides it
                if not msg.get("override_scenario"):
                    level = int(agent.meta.get("level", level))
                    config = agent.meta.get("config", {}) or {}
            elif agent_kind == "human":
                agent = HumanAgent(msg.get("controls", "simple"))
                config = _effective_config(level, config)
            elif agent_kind == "random":
                agent = RandomAgent(np.random.default_rng(seed))
            else:
                agent = None  # autopilot needs the env
            env = RL.RocketLander(level_number=level, continuous=getattr(agent, "continuous", True),
                                  reward_params=(agent.meta.get("reward") if isinstance(agent, ModelAgent)
                                                 else reward_params),
                                  config=config)
            if agent is None:
                agent = AutopilotAgent(env)
            session["agent"] = agent
            env.seed(seed)
            obs = env.reset()
            # a trained agent is judged on the challenge it was trained for
            challenge = (agent.meta.get("challenge") if isinstance(agent, ModelAgent) else None) \
                or msg.get("challenge") or None
            if challenge not in CHALLENGES:
                challenge = None
            tracker = EpisodeTracker(keep_trajectory=False)
            code = msg.get("reward_code") or getattr(agent, "reward_code", None)
            custom = CodeReward(code) if code else None
            memory: dict = {}
            ret, custom_ret = 0.0, (0.0 if custom else None)
            await ws.send_json({"type": "init", "world": _world(env), "config": env.config,
                                "level": level, "agent": agent_kind, "seed": seed,
                                "frame_skip": agent.frame_skip,
                                "challenge": challenge,
                                "snapshot": getattr(agent, "snapshot", None),
                                "attempt": _summary(agent.attempt) if isinstance(agent, ReplayAgent) else None})
            await ws.send_json(_frame(env, ret, custom_ret))
            if msg.get("countdown"):
                # the page shows 3-2-1 over the first frame, then sends {type: go}
                session["go"] = asyncio.Event()
                await session["go"].wait()
            is_human = isinstance(agent, HumanAgent)
            applied: list[list[float]] = []
            trajectory: list[list[float]] = []
            touchdown_vy: Optional[float] = None
            action = agent.act(env, obs)
            done, t0, frames_done = False, time.perf_counter(), 0
            while not done:
                speed = session["speed"]
                n = max(1, int(round(speed))) if speed >= 1 else 1
                for _ in range(n):
                    if env.stepnumber % agent.frame_skip == 0:
                        action = agent.act(env, obs)
                    if is_human:
                        applied.append([float(x) for x in action])
                    obs, r, done, info = env.step(action)
                    ret += r
                    tracker.update(info)
                    if touchdown_vy is None and (info["left_leg_contact"] or info["right_leg_contact"]):
                        touchdown_vy = info["velocity_y"] * RL.START_SPEED
                    if is_human and (env.stepnumber % 3 == 0 or done):
                        trajectory.append([env.stepnumber, round(info["altitude"], 2),
                                           round(info["velocity_y"] * RL.START_SPEED, 2)])
                    if custom:
                        try:
                            custom_ret += custom(dict(info, memory=memory), r)
                        except Exception as exc:
                            await ws.send_json({"type": "error", "message": f"reward: {exc}"})
                            custom = None
                    frames_done += 1
                    if done:
                        break
                await ws.send_json(_frame(env, ret, custom_ret))
                # pace to real time x speed
                target = t0 + frames_done / (FPS * max(speed, 0.05))
                await asyncio.sleep(max(0.0, target - time.perf_counter()))
            info = env.info
            end = {"type": "end", "outcome": _outcome(info), "return": round(ret, 3),
                   "custom_return": None if custom_ret is None else round(custom_ret, 3),
                   "steps": env.stepnumber, "x_on_ship": round(info["x_on_ship"], 3),
                   "fuel": round(info["fuel"], 3), "rotation": round(info["rotation"], 2),
                   "touchdown_vy": None if touchdown_vy is None else round(touchdown_vy, 2),
                   "touchdown_limit": env.config.get("max_touchdown_speed")}
            summary = tracker.summary()
            verdict = judge(challenge, summary)
            end["summary"] = summary
            end["challenge"] = {"id": challenge, "title": CHALLENGES[challenge]["title"] if challenge else "Atterrir",
                                **verdict}
            if is_human:
                attempt = {
                    "id": uuid.uuid4().hex[:10],
                    "name": (str(msg.get("player") or "Anonyme").strip() or "Anonyme")[:30],
                    "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "signature": scenario_signature(level, config),
                    "seed": seed, "level": level, "config": env.config, "reward_params": reward_params,
                    "controls": agent.pilot.mode,
                    "outcome": end["outcome"], "score": end["return"], "custom_score": end["custom_return"],
                    "duration": round(env.stepnumber / FPS, 2), "fuel": end["fuel"],
                    "x_on_ship": end["x_on_ship"], "touchdown_vy": end["touchdown_vy"],
                    "touchdown_limit": end["touchdown_limit"],
                    "actions": _rle_encode(applied), "trajectory": trajectory,
                }
                save_attempt(attempt)
                rank, total = _rank_of(attempt)
                reference = await asyncio.to_thread(autopilot_trajectory, level, config, seed, reward_params)
                end["attempt"] = dict(_summary(attempt), rank=rank, total=total, trajectory=trajectory)
                end["autopilot"] = reference
            await ws.send_json(end)
            env.close()
        except asyncio.CancelledError:
            raise
        except HTTPException as exc:
            await ws.send_json({"type": "error", "message": exc.detail})
        except Exception as exc:  # surface anything to the page instead of a silent socket
            await ws.send_json({"type": "error", "message": f"{type(exc).__name__}: {exc}"})

    try:
        while True:
            msg = await ws.receive_json()
            kind = msg.get("type")
            if kind == "start":
                if session["task"]:
                    session["task"].cancel()
                session["speed"] = float(msg.get("speed", session["speed"]))
                session["task"] = asyncio.create_task(run_episode(msg))
            elif kind == "keys" and isinstance(session["agent"], HumanAgent):
                session["agent"].pilot.set_keys(msg.get("keys") or [])
            elif kind == "action" and isinstance(session["agent"], HumanAgent):
                session["agent"].raw_action = [float(x) for x in np.clip(np.asarray(msg["action"], dtype=np.float64), -1, 1)]
            elif kind == "go" and session.get("go") is not None:
                session["go"].set()
            elif kind == "speed":
                session["speed"] = float(msg["value"])
            elif kind == "stop" and session["task"]:
                session["task"].cancel()
    except WebSocketDisconnect:
        pass
    finally:
        if session["task"]:
            session["task"].cancel()
