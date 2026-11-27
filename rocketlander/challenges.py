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
        "goal": "Faire UN tour complet sur soi-même (pas plus), puis atterrir.",
        "level": 0,
        "config": {"start_height": 0.7, "start_speed": 5.0, "start_spin": 0.0},
        "code": '''def reward(s, base_reward):
    m = s["memory"]
    turns = abs(s["rotation"])            # tours effectués depuis le départ
    best = m.get("best_turns", 0.0)
    r = 0.0
    # 1) les progrès de rotation rapportent... mais seulement jusqu'à 1 tour.
    #    Piège : sans les min(), chaque tour en plus rapporte encore
    #    et l'agent apprend à tourner sans s'arrêter.
    if turns > best:
        r += 10 * (min(turns, 1.0) - min(best, 1.0))
        m["best_turns"] = turns
    if turns >= 1 and not m.get("flip_done"):
        m["flip_done"] = True
        r += 10                           # salto réussi
    # 2) tourner au-delà d'un tour et quart coûte, à chaque instant
    if turns > 1.25:
        r -= 0.05
    # 3) une fois le salto fait : se redresser et atterrir (reward de base)
    if m.get("flip_done"):
        r += base_reward
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


# --------------------------------------------------------------------------- judging
# Each check reads the episode summary (rocketlander.episode.EpisodeTracker.summary) and
# returns (ok, text). ok is None when there is no right answer (demo challenges).

def _landing(s):
    return s["landed"], "posé" if s["landed"] else _fail(s)


def _center(s):
    x = abs(s["x_on_ship"])
    return s["landed"] and x < 0.2, f"écart au centre {x:.2f}" if s["landed"] else _fail(s)


def _edge(s):
    x = s["x_on_ship"]
    ok = s["landed"] and 0.65 <= x <= 1.0
    return ok, f"posé à {x:+.2f} (bord droit = +1)" if s["landed"] else _fail(s)


def _salto(s):
    turns = s["max_rotation"]
    ok = 1.0 <= turns < 1.5 and s["landed"]
    end = "puis posé" if s["landed"] else _fail(s)
    return ok, f"{turns:.2f} tour, {end}"


def _one_foot(s):
    gap = s["contact_gap_s"]
    if not s["landed"]:
        return False, _fail(s)
    ok = gap is not None and gap >= 0.5
    return ok, f"{gap or 0:.2f} s entre les deux pieds"


def _hover(s):
    return s["hover_s"] >= 5.0, f"{s['hover_s']:.1f} s en vol stationnaire à 100 m"


def _fuel(s):
    ok = s["landed"] and s["fuel"] >= 0.5
    return ok, f"carburant restant {s['fuel']:.0%}" if s["landed"] else _fail(s)


def _hacking(s):
    return None, f"{s['contact_s']:.1f} s avec un pied sur le bateau, posé : {'oui' if s['landed'] else 'non'}"


def _fail(s):
    return {"crash": "crash", "broken_leg": "pied cassé", "outside": "sorti de l'écran",
            "timeout": "temps écoulé"}.get(s["outcome"], s["outcome"])


CHECKS = {
    "landing": _landing, "center": _center, "edge": _edge, "salto": _salto,
    "one_foot": _one_foot, "hover": _hover, "fuel": _fuel, "hacking": _hacking,
}
for _id, _ch in CHALLENGES.items():
    _ch.setdefault("success", {
        "landing": "posé 1 s", "center": "posé à moins de 0,2 du centre", "edge": "posé entre +0,65 et +1",
        "salto": "1 tour (entre 1 et 1,5) puis posé", "one_foot": "≥ 0,5 s entre les deux pieds",
        "hover": "≥ 5 s à 100 m ± 15 m", "fuel": "posé avec ≥ 50 % de carburant",
        "hacking": "pas de bonne réponse : observer",
    }[_id])


def judge(challenge: Optional[str], summary: dict) -> dict:
    """{'ok': bool | None, 'text': str} for a challenge id, or the plain landing check."""
    ok, text = CHECKS.get(challenge or "landing", _landing)(summary)
    return {"ok": ok, "text": text}
