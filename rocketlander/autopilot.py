"""Hand-written controller (no learning). Useful as a baseline / sanity check that a
task is physically feasible, and for live demos ("this is what classic control does")."""

from __future__ import annotations

import numpy as np

import gym.envs.box2d.rocket_lander as RL


class Autopilot:
    """Suicide-burn style lander for the continuous action space.

    Works on the raw Box2D state, so it needs the underlying RocketLander (`env.unwrapped.lander`).
    """

    def __init__(self, env, target_touchdown_speed: float = 2.0, margin: float = 1.3,
                 target_x_on_ship: float = 0.0):
        """target_x_on_ship: where to land, 0 = ship centre, -1 / +1 = ship ends."""
        self.env = env
        self.target_x_on_ship = target_x_on_ship
        self.v_td = target_touchdown_speed  # m/s at touchdown
        self.margin = margin  # >1 brakes earlier than the theoretical limit

    @property
    def lander_env(self):
        env = getattr(self.env, "unwrapped", self.env)
        # gymnasium wrapper -> its RocketLander; a bare RocketLander is used as is
        return env.lander if isinstance(env.lander, RL.RocketLander) else env

    def predict(self, obs, deterministic=True):
        e = self.lander_env
        body = e.lander
        m = body.mass + sum(l.mass for l in e.legs)
        g = -e.world.gravity[1]
        x, y = body.position
        vx, vy = body.linearVelocity
        # height of the leg tips above the deck (legs reach ~ROCKET_WIDTH*... below the body origin)
        h = max(0.0, y - e.shipheight - 1.0)
        a_max = RL.MAIN_ENGINE_POWER / m - g  # net upward deceleration at full power
        # descent-speed profile: v(h) such that full-power braking ends at v_td
        v_target = -np.sqrt(self.v_td ** 2 + 2 * (a_max / self.margin) * h)
        # wanted vertical acceleration to track the profile (+ hold position horizontally)
        ay = 4.0 * (v_target - vy)
        thrust = m * (ay + g)
        throttle_cmd = (thrust / RL.MAIN_ENGINE_POWER - RL.MIN_THROTTLE) / (1 - RL.MIN_THROTTLE)
        throttle_cmd = float(np.clip(throttle_cmd, 0.0, 1.0))
        d_throttle = np.clip((throttle_cmd - e.throttle) * RL.FPS / 0.5, -1, 1)

        # attitude: tilt slightly against horizontal drift, gimbal + side thrusters keep it
        x_err = x - (RL.W / 2 + self.target_x_on_ship * e.ship_width / 2)
        angle_target = np.clip(0.02 * x_err + 0.08 * vx, -0.15, 0.15)
        ang_err = body.angle - angle_target
        # positive gimbal pushes the engine so that the angle decreases
        gimbal_cmd = np.clip(1.5 * ang_err + 0.8 * body.angularVelocity, -RL.GIMBAL_THRESHOLD, RL.GIMBAL_THRESHOLD)
        d_gimbal = np.clip((gimbal_cmd - e.gimbal) * RL.FPS / 0.15, -1, 1)
        side = 0.0
        spin = ang_err + 0.5 * body.angularVelocity
        # side thruster +1 pushes the nose so that the angle increases
        if spin > 0.05:
            side = -1.0
        elif spin < -0.05:
            side = 1.0
        if e.config["control"] == "direct":
            throttle_action = throttle_cmd if throttle_cmd > 0 else -1.0
            return np.array([gimbal_cmd / RL.GIMBAL_THRESHOLD, throttle_action, side], dtype=np.float32), None
        return np.array([d_gimbal, d_throttle, side], dtype=np.float32), None
