import numpy as np
import pytest

from gym.envs.box2d.rocket_lander import RocketLander


def rollout(env, actions):
    obs, rews = [], []
    for a in actions:
        o, r, d, _ = env.step(a)
        obs.append(o)
        rews.append(r)
        if d:
            break
    return np.array(obs), np.array(rews)


@pytest.mark.parametrize("level", [0, 1, 2, 3])
def test_spaces_and_types(level):
    env = RocketLander(level_number=level)
    o = env.reset()
    assert o.shape == env.observation_space.shape and o.dtype == np.float32
    assert env.observation_space.contains(o)
    o, r, d, info = env.step(env.action_space.sample())
    assert isinstance(r, float) and -1.0 <= r <= 1.0
    assert isinstance(d, bool) and isinstance(info, dict)
    env.close()


@pytest.mark.parametrize("level", [0, 3])
def test_seed_is_deterministic(level):
    actions = [np.array([0.3, 0.8, 0.0], dtype=np.float32)] * 200
    runs = []
    for _ in range(2):
        env = RocketLander(level_number=level)
        env.seed(123)
        env.reset()
        runs.append(rollout(env, actions))
    np.testing.assert_array_equal(runs[0][0], runs[1][0])
    np.testing.assert_array_equal(runs[0][1], runs[1][1])


def test_different_seeds_differ():
    starts = []
    for seed in (1, 2):
        env = RocketLander(level_number=1)
        env.seed(seed)
        starts.append(env.reset())
    assert not np.allclose(starts[0], starts[1])


def test_actions_are_clipped():
    a = RocketLander()
    b = RocketLander()
    a.seed(0), b.seed(0)
    a.reset(), b.reset()
    oa, _ = rollout(a, [np.array([5.0, 5.0, 5.0])] * 50)
    ob, _ = rollout(b, [np.array([1.0, 1.0, 1.0])] * 50)
    np.testing.assert_array_equal(oa, ob)


def test_discrete_mode():
    env = RocketLander(continuous=False)
    env.seed(0)
    env.reset()
    for _ in range(100):
        _, r, d, _ = env.step(env.action_space.sample())
        assert -1.0 <= r <= 1.0
        if d:
            break


def test_free_fall_ends_with_crash():
    env = RocketLander()
    env.seed(0)
    env.reset()
    _, rews = rollout(env, [np.array([0.0, -1.0, 0.0])] * 2000)
    assert env.game_over and env.good_landings == 0
    assert rews[-1] == -1.0  # crash is penalized
    assert rews.sum() < 0.0  # crashing is never a net win


def test_ground_contact_is_not_farmable():
    """Resting on the ship must not pay ~+1 per step (old +100/leg bug)."""
    env = RocketLander()
    env.seed(0)
    env.reset()
    env.legs[0].ground_contact = True
    _, r, _, _ = env.step(np.zeros(3))
    _, r2, _, _ = env.step(np.zeros(3))
    assert r2 < 0.5

