"""Keyboard piloting, computed server side frame by frame so that a human attempt can be
recorded and replayed exactly.

Two schemes:

- "simple" (3 keys): ↑ engine on (off when released), ← / → rotate the rocket. When no
  rotation key is held, an assist damps the rotation so the rocket keeps its attitude.
- "expert": ↑ / ↓ raise / lower the throttle (it stays where it is), ← / → orient the
  engine (gimbal), Q (or A) / D fire the side thrusters.

Signs (direct control): a positive gimbal makes the angle decrease (nose to the right),
side thruster +1 makes it increase (nose to the left).
"""

from __future__ import annotations

KEYS = {"up", "down", "left", "right", "side_left", "side_right"}
MODES = ("simple", "expert")


def _clip(v: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return lo if v < lo else hi if v > hi else v


class KeyboardPilot:
    def __init__(self, mode: str = "simple"):
        self.mode = mode if mode in MODES else "simple"
        self.keys: set[str] = set()
        self.throttle = 0.0  # expert mode: throttle level kept between key presses

    def set_keys(self, keys) -> None:
        self.keys = {k for k in keys if k in KEYS}

    def action(self, env) -> list[float]:
        """Action [gimbal, throttle, side] for the next physics frame of `env` (a RocketLander)."""
        k = self.keys
        if self.mode == "expert":
            if "up" in k:
                self.throttle = min(1.0, self.throttle + 0.02)
            if "down" in k:
                self.throttle = max(0.0, self.throttle - 0.03)
            gimbal = float(("right" in k) - ("left" in k))
            side = float(("side_left" in k) - ("side_right" in k))
            return [gimbal, self.throttle if self.throttle > 1e-3 else -1.0, side]

        throttle = 1.0 if "up" in k else -1.0
        turn = ("left" in k) - ("right" in k)  # +1 = nose to the left (angle increases)
        if turn:
            return [float(-turn), throttle, float(turn)]
        # attitude hold: damp the angular velocity
        w = float(env.lander.angularVelocity)
        gimbal = _clip(1.5 * w)
        side = -1.0 if w > 0.05 else 1.0 if w < -0.05 else 0.0
        return [gimbal, throttle, side]
