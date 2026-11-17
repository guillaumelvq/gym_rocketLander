"""Human play mode: keyboard mapping, attempt recording, leaderboard and exact replays."""

import asyncio
import json
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

import pytest
import uvicorn
import websockets

import gym.envs.box2d.rocket_lander as RL
from rocketlander.app import server
from rocketlander.controls import KeyboardPilot

CALM = {"start_spin": 0.0, "start_height": 0.6, "start_speed": 10.0, "start_x_spread": 0.0}


def _hold(mode, keys, frames=120):
    env = RL.RocketLander(level_number=0, config=CALM)
    env.seed(0)
    env.reset()
    pilot = KeyboardPilot(mode)
    pilot.set_keys(keys)
    for _ in range(frames):
        env.step(pilot.action(env))
    return env.lander.angle, env.lander.linearVelocity[1]


@pytest.mark.parametrize("mode,left,right", [("simple", "left", "right"), ("expert", "side_left", "side_right")])
def test_controls_signs(mode, left, right):
    angle0, vy0 = _hold(mode, [])
    assert abs(angle0) < 0.05
    assert _hold(mode, [left])[0] > 0.3      # nose to the left = angle increases
    assert _hold(mode, [right])[0] < -0.3
    assert _hold(mode, ["up"])[1] > vy0 + 10  # the engine slows the fall


def test_simple_mode_holds_attitude():
    env = RL.RocketLander(level_number=0, config=CALM)
    env.seed(0)
    env.reset()
    pilot = KeyboardPilot("simple")
    pilot.set_keys(["left"])
    for _ in range(30):
        env.step(pilot.action(env))
    spin = abs(env.lander.angularVelocity)
    pilot.set_keys([])
    for _ in range(60):
        env.step(pilot.action(env))
    assert abs(env.lander.angularVelocity) < spin / 3


class Client:
    """Minimal HTTP + websocket client against a live uvicorn server (no extra dependency)."""

    def __init__(self, base):
        self.base = base

    def _req(self, method, path, params=None):
        url = self.base + path + ("?" + urllib.parse.urlencode(params) if params else "")
        try:
            with urllib.request.urlopen(urllib.request.Request(url, method=method)) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read() or b"{}")

    def get(self, path, params=None):
        return self._req("GET", path, params)[1]

    def delete(self, path):
        return self._req("DELETE", path)[0]

    def play(self, start, script=None):
        """One episode over the websocket; `script(frame_index)` returns the keys to hold."""

        async def run():
            async with websockets.connect(self.base.replace("http", "ws") + "/ws/sim", max_size=None) as ws:
                await ws.send(json.dumps(start))
                n = 0
                while True:
                    m = json.loads(await ws.recv())
                    if m["type"] == "frame":
                        n += 1
                        if script:
                            await ws.send(json.dumps({"type": "keys", "keys": script(n)}))
                    elif m["type"] == "end":
                        return m
                    elif m["type"] == "error":
                        raise AssertionError(m["message"])

        return asyncio.run(run())


@pytest.fixture(scope="module")
def live_server(tmp_path_factory):
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


@pytest.fixture
def client(live_server, tmp_path, monkeypatch):
    monkeypatch.setattr(server, "PLAYERS", tmp_path / "players")
    return Client(live_server)


def _human_attempt(client, name="Ada", script=lambda n: ["up"] if n % 7 < 3 else [], config=CALM, seed=3):
    return client.play({"type": "start", "agent": "human", "player": name, "level": 1, "config": config,
                        "seed": seed, "speed": 8, "controls": "simple"}, script)


def test_attempt_is_recorded_and_replayed_exactly(client):
    end = _human_attempt(client)
    attempt = end["attempt"]
    assert attempt["name"] == "Ada" and attempt["rank"] == 1 and attempt["total"] == 1
    assert end["autopilot"]["trajectory"] and attempt["trajectory"]
    stored = client.get(f"/api/attempts/{attempt['id']}")
    assert stored["actions"] and stored["seed"] == 3

    replay = client.play({"type": "start", "agent": "replay", "attempt": attempt["id"], "speed": 8})
    for key in ("outcome", "return", "steps", "x_on_ship", "fuel"):
        assert replay[key] == end[key], key


def test_leaderboard_ranks_and_filters(client):
    crash = _human_attempt(client, "Crash", script=lambda n: [])  # engine never on
    other = _human_attempt(client, "Autre", config={**CALM, "ship_width": 0.5})
    a = crash["attempt"]
    board = client.get("/api/attempts", params={"level": 1, "config": json.dumps(CALM)})
    assert [x["id"] for x in board["attempts"]] == [a["id"]]
    assert board["signature"] == a["signature"] != other["attempt"]["signature"]
    assert "actions" not in board["attempts"][0]
    assert client.delete(f"/api/attempts/{a['id']}") == 200
    assert client.get("/api/attempts", params={"signature": a["signature"]})["count"] == 0
    assert client.delete(f"/api/attempts/{a['id']}") == 404


def test_successes_rank_first():
    attempts = [
        {"outcome": "crash", "score": 50.0, "duration": 3.0},
        {"outcome": "success", "score": 1.0, "duration": 9.0},
        {"outcome": "success", "score": 5.0, "duration": 9.0},
    ]
    ranked = sorted(attempts, key=server._rank_key)
    assert [a["score"] for a in ranked] == [5.0, 1.0, 50.0]


def test_rle_roundtrip():
    actions = [[0.0, -1.0, 0.0]] * 5 + [[0.25, 1.0, -1.0]] * 2 + [[0.0, -1.0, 0.0]]
    assert server._rle_decode(server._rle_encode(actions)) == actions
