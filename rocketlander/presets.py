"""Named difficulty presets for the workshop.

A preset is a complete scenario (every key of DEFAULT_CONFIG): the web app and the training
scripts pass it as `config`, on top of level 0. Presets are ordered from easiest to hardest
and each one is feasible: the hand-written autopilot lands in most episodes (see
AUTOPILOT_SUCCESS, measured with `uv run python -m rocketlander.presets`).

The numeric levels 0-20 (level_config in rocket_lander.py) remain available as "advanced
levels"; LEVEL_HELP explains what each number changes.
"""

from __future__ import annotations

from gym.envs.box2d.rocket_lander import DEFAULT_CONFIG, level_config


def _preset(**overrides) -> dict:
    cfg = dict(DEFAULT_CONFIG)
    unknown = set(overrides) - set(cfg)
    assert not unknown, unknown
    cfg.update(overrides)
    return cfg


# name -> (description, full config). Order = difficulty.
PRESETS: dict[str, dict] = {
    "decouverte": {
        "title": "Découverte",
        "description": "Départ bas et lent, grande zone, pieds solides : pour prendre en main.",
        "config": _preset(
            start_height=0.35, start_speed=10.0, start_spin=0.2, start_x_spread=0.0, start_drift=0.0,
            ship_width=1.5, max_touchdown_speed=8.0, hard_landing_breaks=False, leg_scale=1.2, fuel=1000.0,
        ),
    },
    "facile": {
        "title": "Facile",
        "description": "Un peu plus haut et plus vite, pieds cassés au-delà de 7 m/s au contact.",
        "config": _preset(
            start_height=0.5, start_speed=15.0, start_spin=0.4, start_x_spread=0.02, start_drift=0.0,
            ship_width=1.2, max_touchdown_speed=7.0, hard_landing_breaks=False, leg_scale=1.0, fuel=900.0,
        ),
    },
    "moyen": {
        "title": "Moyen",
        "description": "Départ aux deux tiers de la hauteur à 25 m/s, légère dérive, limite 6 m/s.",
        "config": _preset(
            start_height=0.7, start_speed=25.0, start_spin=0.7, start_x_spread=0.04, start_drift=0.1,
            ship_width=1.0, max_touchdown_speed=6.0, hard_landing_breaks=False, leg_scale=1.0, fuel=800.0,
        ),
    },
    "difficile": {
        "title": "Difficile",
        "description": "Départ tout en haut à 40 m/s, zone réduite, limite 5 m/s au contact.",
        "config": _preset(
            start_height=0.95, start_speed=40.0, start_spin=1.0, start_x_spread=0.05, start_drift=0.2,
            ship_width=0.8, max_touchdown_speed=5.0, hard_landing_breaks=False, leg_scale=1.0, fuel=700.0,
        ),
    },
    "expert": {
        "title": "Expert",
        "description": "Zone étroite, jambes plus courtes qui cassent aussi si trop comprimées, moins de carburant.",
        "config": _preset(
            start_height=0.95, start_speed=40.0, start_spin=1.2, start_x_spread=0.06, start_drift=0.2,
            ship_width=0.6, max_touchdown_speed=5.0, hard_landing_breaks=True, leg_scale=0.8, fuel=600.0,
        ),
    },
    "cauchemar": {
        "title": "Cauchemar",
        "description": "Zone minuscule, départ très décalé, jambes courtes, peu de carburant.",
        "config": _preset(
            start_height=0.95, start_speed=40.0, start_spin=1.2, start_x_spread=0.1, start_drift=0.3,
            ship_width=0.4, max_touchdown_speed=5.0, hard_landing_breaks=True, leg_scale=0.6, fuel=500.0,
        ),
    },
}

# Autopilot success rate on 30 seeds (Autopilot(target_touchdown_speed=1.0, margin=3.0)),
# refreshed with `uv run python -m rocketlander.presets`.
AUTOPILOT_SUCCESS: dict[str, float] = {
    "decouverte": 1.00, "facile": 1.00, "moyen": 1.00,
    "difficile": 0.97, "expert": 0.87, "cauchemar": 0.40,
}

LEVEL_HELP = (
    "Le numéro du niveau règle la difficulté d'un coup : "
    "0 = les pieds ne cassent jamais ; 1 à 9 = les pieds cassent si la vitesse verticale au contact "
    "dépasse 8 − 0,4 × niveau m/s (7,6 au niveau 1, 4,4 au niveau 9) et le départ devient plus aléatoire ; "
    "10 = en plus, les jambes cassent si elles sont trop comprimées ; "
    "11 à 20 = en plus, les jambes raccourcissent (×0,93 au niveau 11 … ×0,3 au niveau 20)."
)


def level_summary(level: int) -> str:
    """One-line French explanation of a numeric level."""
    cfg = level_config(level)
    parts = []
    lim = cfg["max_touchdown_speed"]
    parts.append("pieds incassables" if lim is None else f"pieds cassés au-delà de {lim:.1f} m/s au contact")
    if cfg["hard_landing_breaks"]:
        parts.append("et si les jambes sont trop comprimées")
    if cfg["leg_scale"] < 1:
        parts.append(f"jambes ×{cfg['leg_scale']:.2f}")
    if cfg["start_x_spread"] > 0.03:
        parts.append(f"départ ±{cfg['start_x_spread'] * 100:.0f} % de l'écran")
    return ", ".join(parts)


def measure(n_seeds: int = 30, names=None) -> dict[str, float]:
    """Autopilot success rate of each preset (feasibility check)."""
    from rocketlander import make_env
    from rocketlander.autopilot import Autopilot

    out = {}
    for name in names or PRESETS:
        cfg = PRESETS[name]["config"]
        ok = 0
        for seed in range(n_seeds):
            env = make_env(level=0, config=cfg)
            pilot = Autopilot(env, target_touchdown_speed=1.0, margin=3.0)
            obs, info = env.reset(seed=seed)
            done = False
            while not done:
                action, _ = pilot.predict(obs)
                obs, _, term, trunc, info = env.step(action)
                done = term or trunc
            ok += bool(info["landed_full"])
            env.close()
        out[name] = ok / n_seeds
    return out


if __name__ == "__main__":
    for name, rate in measure().items():
        print(f"{name:12s} {rate:.0%}")
