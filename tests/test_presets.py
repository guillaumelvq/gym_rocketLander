import pytest

from gym.envs.box2d.rocket_lander import DEFAULT_CONFIG
from rocketlander.presets import AUTOPILOT_SUCCESS, PRESETS, level_summary, measure

NAMES = list(PRESETS)


def test_presets_are_complete_configs():
    for p in PRESETS.values():
        assert set(p["config"]) == set(DEFAULT_CONFIG)
        assert p["title"] and p["description"]


def test_presets_get_harder():
    cfgs = [PRESETS[n]["config"] for n in NAMES]
    for easy, hard in zip(cfgs, cfgs[1:]):
        assert hard["start_height"] >= easy["start_height"]
        assert hard["start_speed"] >= easy["start_speed"]
        assert hard["ship_width"] <= easy["ship_width"]
        assert hard["leg_scale"] <= easy["leg_scale"]
        assert hard["fuel"] <= easy["fuel"]
        assert hard["start_x_spread"] >= easy["start_x_spread"]
        assert hard["hard_landing_breaks"] >= easy["hard_landing_breaks"]
        assert hard["max_touchdown_speed"] <= easy["max_touchdown_speed"]


def test_recorded_autopilot_rates_decrease():
    rates = [AUTOPILOT_SUCCESS[n] for n in NAMES]
    assert all(r > 0 for r in rates)
    assert rates == sorted(rates, reverse=True)


@pytest.mark.parametrize("name", ["decouverte", "difficile", "cauchemar"])
def test_presets_are_feasible(name):
    # the hand-written autopilot must land at least sometimes
    assert measure(n_seeds=5, names=[name])[name] > 0


def test_level_summary():
    assert "incassables" in level_summary(0)
    assert "7.6" in level_summary(1)
    assert "jambes" in level_summary(15)


def test_meta_exposes_presets():
    from rocketlander.app.server import meta as meta_endpoint

    meta = meta_endpoint()  # the /api/meta handler (no HTTP client needed)
    assert set(meta["presets"]) == set(PRESETS)
    assert meta["presets"]["facile"]["autopilot_success"] == AUTOPILOT_SUCCESS["facile"]
    assert len(meta["level_summaries"]) == 21 and meta["level_help"]
