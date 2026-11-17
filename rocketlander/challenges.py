"""Exercises for the workshop: each challenge is a scenario (env config) + a starting
reward function written in plain Python, that participants edit to obtain a behaviour.

A reward function has the signature

    def reward(s, base_reward):
        ...
        return value

called at every physics frame (60 per second). `s` holds the named quantities of the
current frame (see RocketLander.step "info"), plus `s["memory"]`, a dict you can use to
remember things during the episode (it is emptied at every new episode).
"""

from __future__ import annotations

import math
from typing import Callable, Optional

RewardFn = Callable[[dict, float], float]

# Easier start than the default (345 m at 40 m/s): makes the exercises learnable in a few
# minutes on a laptop.
EASY_START = {"start_height": 0.35, "start_speed": 10.0, "start_spin": 0.3}

CHALLENGES: dict[str, dict] = {
    "landing": {
        "title": "Atterrir",
        "goal": "Se poser sur le bateau et y rester 1 seconde.",
        "level": 1,
        "config": dict(EASY_START),
        "code": '''def reward(s, base_reward):
    # La reward de base : se rapprocher et ralentir rapporte un peu à chaque instant,
    # atterrir rapporte gros, se crasher coûte cher.
    return base_reward
''',
    },
    "center": {
        "title": "Viser le centre",
        "goal": "Atterrir le plus près possible du centre du bateau.",
        "level": 1,
        "config": {**EASY_START, "start_x": 0.25},
        "code": '''def reward(s, base_reward):
    r = base_reward
    if s["landed_full"]:
        # x_on_ship : 0 = centre, -1 / +1 = bords du bateau
        r += 10 * (1 - abs(s["x_on_ship"]))
    return r
''',
    },
    "edge": {
        "title": "S'arrêter au bord",
        "goal": "Atterrir tout au bord droit du bateau, sans tomber.",
        "level": 1,
        "config": dict(EASY_START),
        "code": '''def reward(s, base_reward):
    r = base_reward
    # x_on_ship = +1 au bord droit : on veut se poser vers +0.85
    target = 0.85
    r -= 0.02 * abs(s["x_on_ship"] - target)
    if s["landed_full"]:
        r += 20 * max(0.0, 1 - 4 * abs(s["x_on_ship"] - target))
    return r
''',
    },
    "salto": {
        "title": "Faire un salto",
        "goal": "Faire un tour complet sur soi-même, puis atterrir.",
        "level": 0,
        "config": {"start_height": 0.7, "start_speed": 5.0, "start_spin": 0.0},
        "code": '''def reward(s, base_reward):
    m = s["memory"]
    turns = abs(s["rotation"])            # tours effectués depuis le départ
    best = m.get("best_turns", 0.0)
    r = 0.0
    if turns > best:                      # récompense chaque progrès dans la rotation
        r += 10 * (turns - best)
        m["best_turns"] = turns
    if turns >= 1 and not m.get("flip_done"):
        m["flip_done"] = True
        r += 10                           # salto réussi
    if m.get("flip_done"):
        r += base_reward                  # ensuite : atterrir
    elif s["game_over"]:
        r -= 5
    return r
''',
    },
    "one_foot": {
        "title": "Un pied puis l'autre",
        "goal": "Toucher le bateau avec un seul pied, attendre une demi-seconde, puis poser le second.",
        "level": 0,
        "config": dict(EASY_START),
        "code": '''def reward(s, base_reward):
    m = s["memory"]
    left, right = s["left_leg_contact"], s["right_leg_contact"]
    r = base_reward
    if (left != right) and "first_touch" not in m:
        m["first_touch"] = s["step"]       # premier pied posé
    if left and right and "both" not in m:
        m["both"] = s["step"]
        gap = (m["both"] - m.get("first_touch", m["both"])) / 60
        r += 10 * min(gap, 0.5) / 0.5       # bonus si >= 0.5 s entre les deux pieds
    return r
''',
    },
    "hover": {
        "title": "Vol stationnaire",
        "goal": "Rester immobile à 100 m au-dessus du bateau le plus longtemps possible.",
        "level": 0,
        "config": {"start_height": 0.5, "start_speed": 5.0, "start_spin": 0.3,
                   "end_on_landing": False, "max_steps": 900},
        "code": '''def reward(s, base_reward):
    if s["game_over"]:
        return -10
    err = abs(s["altitude"] - 100) / 100 + abs(s["velocity_y"]) + abs(s["angle"])
    return 0.02 * max(0.0, 1 - err)
''',
    },
    "fuel": {
        "title": "Économiser le carburant",
        "goal": "Atterrir en gardant le plus de carburant possible.",
        "level": 1,
        "config": dict(EASY_START),
        "code": '''def reward(s, base_reward):
    r = base_reward
    if s["landed_full"]:
        r += 20 * s["fuel"]          # fuel : 1 = réservoir plein
    return r
''',
    },
    "hacking": {
        "title": "Démo : reward hacking",
        "goal": "Piège ! « Toucher le bateau rapporte des points » et l'épisode ne s'arrête "
                "pas à l'atterrissage. Que va faire l'agent ?",
        "level": 0,
        "config": {**EASY_START, "end_on_landing": False, "max_steps": 900},
        "code": '''def reward(s, base_reward):
    r = 0.0
    if s["left_leg_contact"] or s["right_leg_contact"]:
        r += 0.05              # chaque instant avec un pied sur le bateau rapporte
    if s["game_over"]:
        r -= 5
    return r
''',
    },
}


def compile_reward(code: str) -> RewardFn:
    """Turns the source code of a `reward(s, base_reward)` function into a callable."""
    namespace: dict = {"math": math, "abs": abs, "min": min, "max": max}
    exec(compile(code, "<reward>", "exec"), namespace)  # noqa: S102 - local, user-written code
    fn = namespace.get("reward")
    if not callable(fn):
        raise ValueError("the code must define a function `reward(s, base_reward)`")
    return fn


class CodeReward:
    """Picklable reward function built from source code."""

    def __init__(self, code: str):
        self.code = code
        self._fn: Optional[RewardFn] = None

    def __call__(self, s: dict, base_reward: float) -> float:
        if self._fn is None:
            self._fn = compile_reward(self.code)
        return float(self._fn(s, base_reward))

    def __getstate__(self):
        return {"code": self.code}

    def __setstate__(self, state):
        self.code = state["code"]
        self._fn = None
