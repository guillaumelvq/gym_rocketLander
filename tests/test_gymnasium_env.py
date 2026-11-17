import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from rocketlander import RocketLanderEnv, make_env


@pytest.mark.parametrize("continuous", [True, False])
@pytest.mark.parametrize("level", [0, 5, 15])
def test_gymnasium_api(level, continuous):
    check_env(RocketLanderEnv(level=level, continuous=continuous), skip_render_check=True)


def test_custom_reward_and_info():
    seen = []

    def reward_fn(s, base_reward):
        seen.append(s)
        return 42.0

    env = make_env(level=3, reward_fn=reward_fn, frame_skip=1)
    obs, info = env.reset(seed=0)
    assert {"x", "y", "angle", "fuel", "landed_full", "game_over"} <= set(info)
    _, r, _, _, info = env.step(env.action_space.sample())
    assert r == 42.0 and "base_reward" in info and seen


def test_time_limit():
    env = make_env(max_episode_steps=10)
    env.reset(seed=0)
    for _ in range(10):
        _, _, term, trunc, _ = env.step(np.zeros(3, dtype=np.float32))
    assert trunc and not term


def test_rgb_render():
    env = make_env(render_mode="rgb_array")
    env.reset(seed=0)
    frame = env.render()
    env.close()
    assert frame.shape == (720, 500, 3)
    assert frame.std() > 10  # not a blank / mis-projected frame


@pytest.mark.parametrize("level", [1, 9])
def test_fast_touchdown_breaks_legs(level):
    env = make_env(level=level)
    env.reset(seed=0)
    term, info = False, {}
    while not term:  # free fall, engine off
        _, _, term, trunc, info = env.step(np.array([0, -1, 0], dtype=np.float32))
        if trunc:
            break
    assert info["game_over"] and not info["landed_full"]
