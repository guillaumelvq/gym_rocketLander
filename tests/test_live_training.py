"""Laboratory: live snapshots in the arena, challenge verdicts, trainings started from an
existing agent."""

import asyncio
import json
import os
import socket
import threading
import time
import urllib.error
import urllib.request

import pytest
import uvicorn
import websockets
from stable_baselines3 import PPO

from rocketlander import make_env
from rocketlander.app import server
from rocketlander.challenges import CHALLENGES

LOW = {"start_spin": 0.0, "start_height": 0.3, "start_speed": 5.0, "start_x_spread": 0.0}


@pytest.fixture(scope="module")
def base_url():
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    srv = uvicorn.Server(uvicorn.Config(server.app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()
    for _ in range(100):
        if srv.started:
            break
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    srv.should_exit = True
    thread.join(5)


def _request(base, method, path, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(base + path, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def _episode(base, start):
    """Plays one episode over the websocket, returns (init message, end message)."""

    async def run():
        async with websockets.connect(base.replace("http", "ws") + "/ws/sim", max_size=None) as ws:
            await ws.send(json.dumps(start))
            init = None
            while True:
                m = json.loads(await ws.recv())
                if m["type"] == "init":
                    init = m
                elif m["type"] == "end":
                    return init, m
                elif m["type"] == "error":
                    raise AssertionError(m["message"])

    return asyncio.run(run())


def _save_model(path, seed):
    PPO("MlpPolicy", make_env(level=0), n_steps=64, batch_size=32, seed=seed, device="cpu").save(path)


@pytest.fixture
def runs(tmp_path, monkeypatch):
    """A runs/ folder with one run 'r1' (PPO, salto challenge) that has a live snapshot."""
    root = tmp_path / "runs"
    d = root / "r1"
    d.mkdir(parents=True)
    (d / "meta.json").write_text(json.dumps({
        "name": "r1", "algo": "PPO", "level": 0, "continuous": True, "frame_skip": 4,
        "config": LOW, "challenge": "salto", "timesteps": 100000,
    }), encoding="utf8")
    _save_model(d / "live_model.zip", seed=0)
    _save_model(d / "best_model.zip", seed=0)
    (d / "live.json").write_text(json.dumps({"timesteps": 4096, "snapshot": 1}), encoding="utf8")
    monkeypatch.setattr(server, "RUNS", root)
    return root


def test_live_agent_reloads_new_snapshots(base_url, runs):
    start = {"type": "start", "agent": "live", "run": "r1", "seed": 1, "speed": 8}
    init, end = _episode(base_url, start)
    assert init["snapshot"] == {"timesteps": 4096, "snapshot": 1}
    assert init["challenge"] == "salto"  # judged on the challenge it is trained for
    assert end["challenge"]["id"] == "salto" and end["challenge"]["ok"] is False

    # the training writes a new snapshot: the next episode must use it
    live = runs / "r1" / "live_model.zip"
    _save_model(live, seed=1)
    later = time.time() + 5
    os.utime(live, (later, later))
    (runs / "r1" / "live.json").write_text(json.dumps({"timesteps": 8192, "snapshot": 2}), encoding="utf8")
    init2, _ = _episode(base_url, start)
    assert init2["snapshot"]["snapshot"] == 2
    keys = [k for k in server.ModelAgent._cache if k[0] == str(live)]
    assert keys == [(str(live), live.stat().st_mtime)]  # old snapshot dropped from the cache


def test_live_agent_without_snapshot_reports_an_error(base_url, runs):
    (runs / "r1" / "live_model.zip").unlink()
    with pytest.raises(AssertionError, match="instantané"):
        _episode(base_url, {"type": "start", "agent": "live", "run": "r1", "seed": 1, "speed": 8})


@pytest.mark.parametrize("challenge,ok", [("landing", True), ("salto", False)])
def test_end_message_holds_the_challenge_verdict(base_url, challenge, ok):
    _, end = _episode(base_url, {"type": "start", "agent": "autopilot", "level": 0, "config": LOW,
                                 "seed": 2, "speed": 8, "challenge": challenge})
    verdict = end["challenge"]
    assert verdict["id"] == challenge and verdict["title"] == CHALLENGES[challenge]["title"]
    assert verdict["ok"] is ok and verdict["text"]
    assert end["summary"]["landed"] is True


class FakePopen:
    calls = []

    def __init__(self, cmd, **kwargs):
        FakePopen.calls.append(cmd)

    def poll(self):
        return None

    def terminate(self):
        pass


def test_training_from_an_existing_agent_builds_the_command(base_url, runs, monkeypatch):
    monkeypatch.setattr(server.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(server, "TRAININGS", {})
    FakePopen.calls.clear()
    status, data = _request(base_url, "POST", "/api/train", {
        "name": "suite", "level": 0, "config": LOW, "challenge": "salto", "init_run": "r1",
        "timesteps": 30000, "algo": "A2C", "continuous": False,
        "reward_code": CHALLENGES["salto"]["code"],
    })
    assert status == 200, data
    assert data["init_model"] == "best_model.zip"
    cmd = FakePopen.calls[-1]
    assert cmd[cmd.index("--challenge") + 1] == "salto"
    assert cmd[cmd.index("--init-model") + 1] == str((runs / "r1" / "best_model.zip").resolve())
    assert cmd[cmd.index("--algo") + 1] == "PPO"  # taken from the starting agent
    assert "--discrete" not in cmd                 # same action space as the starting agent
    assert int(cmd[cmd.index("--snapshot-every") + 1]) >= 4096
    assert (runs / data["name"] / "reward.py").exists()


def test_training_rejects_unknown_challenge_and_missing_model(base_url, runs, monkeypatch):
    monkeypatch.setattr(server.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(server, "TRAININGS", {})
    assert _request(base_url, "POST", "/api/train", {"challenge": "moonwalk"})[0] == 400
    (runs / "r2").mkdir()
    (runs / "r2" / "meta.json").write_text(json.dumps({"name": "r2"}), encoding="utf8")
    assert _request(base_url, "POST", "/api/train", {"init_run": "r2"})[0] == 400
    assert _request(base_url, "POST", "/api/train", {"init_run": "nope"})[0] == 404


def test_episodes_endpoint_and_run_info(base_url, runs):
    episodes = [{"t": 4096, "outcome": "crash", "challenge": {"ok": False, "text": "crash"},
                 "return": -3.0, "trajectory": [[0.0, 100.0, 0.0], [1.0, 0.0, 0.1]]}]
    (runs / "r1" / "episodes.json").write_text(json.dumps(episodes), encoding="utf8")
    status, data = _request(base_url, "GET", "/api/runs/r1/episodes")
    assert status == 200 and data["episodes"] == episodes and data["live"]["snapshot"] == 1
    status, info = _request(base_url, "GET", "/api/runs/r1")
    assert info["has_live"] is True and info["live"]["snapshot"] == 1 and "eta_s" in info
