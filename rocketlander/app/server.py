"""Rocket RL Lab: local web app to watch, compare, play and train rocket landing agents.

    uv run python -m rocketlander.app          # http://127.0.0.1:8000

Runs only on localhost: it executes the reward code typed in the page.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import gym.envs.box2d.rocket_lander as RL
from rocketlander.autopilot import Autopilot
from rocketlander.challenges import CHALLENGES, CodeReward, compile_reward
from rocketlander.presets import AUTOPILOT_SUCCESS, LEVEL_HELP, PRESETS, level_summary

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs"
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
    return {
        **meta_,
        "name": d.name,
        "status": status,
        "has_best": (d / "best_model.zip").exists(),
        "has_last": (d / "last_model.zip").exists(),
        "progress": progress,
        "eval": _read_json(d / "eval.json"),
        "reward_code": (d / "reward.py").read_text(encoding="utf8") if (d / "reward.py").exists() else None,
        "mtime": (d / "meta.json").stat().st_mtime if (d / "meta.json").exists() else d.stat().st_mtime,
    }


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


@app.post("/api/train")
def start_training(req: TrainRequest) -> dict:
    running = [n for n, p in TRAININGS.items() if p.poll() is None]
    if len(running) >= MAX_PARALLEL:
        raise HTTPException(429, f"{len(running)} entraînements déjà en cours (max {MAX_PARALLEL})")
    unknown = set(req.config) - set(RL.DEFAULT_CONFIG)
    if unknown:
        raise HTTPException(400, f"paramètres inconnus : {sorted(unknown)}")
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
    if req.reward_code:
        (out / "reward.py").write_text(req.reward_code, encoding="utf8")
        cmd += ["--reward-code", str(out / "reward.py")]
    log = open(out / "train.log", "w", encoding="utf8")
    flags = subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 0
    TRAININGS[name] = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, creationflags=flags)
    return {"name": name}


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


# --------------------------------------------------------------------------- live simulation


class Agent:
    """Something that picks an action from the current RocketLander state."""

    continuous = True
    frame_skip = 1

    def act(self, env: RL.RocketLander, obs: np.ndarray):
        raise NotImplementedError


class HumanAgent(Agent):
    def __init__(self):
        self.action = np.array([0.0, -1.0, 0.0], dtype=np.float32)

    def act(self, env, obs):
        return self.action


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
        from stable_baselines3 import A2C, PPO

        d = _run_dir(run)
        meta_ = _read_json(d / "meta.json") or {}
        path = d / f"{which}_model.zip"
        if not path.exists():
            path = d / "last_model.zip" if (d / "last_model.zip").exists() else d / "best_model.zip"
        if not path.exists():
            raise HTTPException(404, "ce run n'a pas encore de modèle sauvegardé")
        key = (str(path), path.stat().st_mtime)
        if key not in self._cache:
            cls = A2C if meta_.get("algo") == "A2C" else PPO
            self._cache[key] = cls.load(path, device="cpu")
        self.model = self._cache[key]
        self.meta = meta_
        self.continuous = meta_.get("continuous", True)
        self.frame_skip = int(meta_.get("frame_skip", 4))
        self.reward_code = (d / "reward.py").read_text(encoding="utf8") if (d / "reward.py").exists() else None

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
    {type: start, agent: human|autopilot|random|model, run, which, level, config, seed, speed}
    {type: action, action: [gimbal, throttle, side]}   (human agent)
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
            if agent_kind == "model":
                agent: Agent = ModelAgent(msg["run"], msg.get("which", "best"))
                # a model is shown in the scenario it was trained on, unless the page overrides it
                if not msg.get("override_scenario"):
                    level = int(agent.meta.get("level", level))
                    config = agent.meta.get("config", {}) or {}
            elif agent_kind == "human":
                agent = HumanAgent()
            elif agent_kind == "random":
                agent = RandomAgent(np.random.default_rng(seed))
            else:
                agent = None  # autopilot needs the env
            env = RL.RocketLander(level_number=level, continuous=getattr(agent, "continuous", True),
                                  reward_params=(agent.meta.get("reward") if agent_kind == "model"
                                                 else {k: float(v) for k, v in (msg.get("reward") or {}).items()
                                                       if k in RL.REWARD}),
                                  config=config)
            if agent is None:
                agent = AutopilotAgent(env)
            session["agent"] = agent
            env.seed(seed)
            obs = env.reset()
            code = msg.get("reward_code") or getattr(agent, "reward_code", None)
            custom = CodeReward(code) if code else None
            memory: dict = {}
            ret, custom_ret = 0.0, (0.0 if custom else None)
            await ws.send_json({"type": "init", "world": _world(env), "config": env.config,
                                "level": level, "agent": agent_kind, "seed": seed,
                                "frame_skip": agent.frame_skip})
            await ws.send_json(_frame(env, ret, custom_ret))
            action = agent.act(env, obs)
            done, t0, frames_done = False, time.perf_counter(), 0
            while not done:
                speed = session["speed"]
                n = max(1, int(round(speed))) if speed >= 1 else 1
                for _ in range(n):
                    if env.stepnumber % agent.frame_skip == 0:
                        action = agent.act(env, obs)
                    obs, r, done, info = env.step(action)
                    ret += r
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
            await ws.send_json({"type": "end", "outcome": _outcome(info), "return": round(ret, 3),
                                "custom_return": None if custom_ret is None else round(custom_ret, 3),
                                "steps": env.stepnumber, "x_on_ship": round(info["x_on_ship"], 3),
                                "fuel": round(info["fuel"], 3), "rotation": round(info["rotation"], 2)})
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
            elif kind == "action" and isinstance(session["agent"], HumanAgent):
                session["agent"].action = np.clip(np.asarray(msg["action"], dtype=np.float32), -1, 1)
            elif kind == "speed":
                session["speed"] = float(msg["value"])
            elif kind == "stop" and session["task"]:
                session["task"].cancel()
    except WebSocketDisconnect:
        pass
    finally:
        if session["task"]:
            session["task"].cancel()
