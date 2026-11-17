import math
import os

import numpy as np
import pyglet
import Box2D
from Box2D.b2 import (
    edgeShape,
    circleShape,
    fixtureDef,
    polygonShape,
    revoluteJointDef,
    distanceJointDef,
    contactListener,
)
import gym
from gym import spaces
from gym.utils import seeding


"""

The objective of this environment is to land a rocket on a ship.

STATE VARIABLES
The state consists of the following variables:
    - x position
    - y position
    - angle
    - first leg ground contact indicator
    - second leg ground contact indicator
    - throttle
    - engine gimbal
If VEL_STATE is set to true, the velocities are included:
    - x velocity
    - y velocity
    - angular velocity
all state variables are roughly in the range [-1, 1]
    
CONTROL INPUTS
Discrete control inputs are:
    - gimbal left
    - gimbal right
    - throttle up
    - throttle down
    - use first control thruster
    - use second control thruster
    - no action
    
Continuous control inputs are:
    - gimbal (left/right)
    - throttle (up/down)
    - control thruster (left/right)

"""

CONTINUOUS = True
VEL_STATE = True  # Add velocity info to state
FPS = 60
SCALE_S = 0.35  # Temporal Scaling, lower is faster - adjust forces appropriately
INITIAL_RANDOM = 0.0  # Random scaling of initial velocity, higher is more difficult

MAX_STEPS = 1000  # episode length; not landing in time counts as a failure

# Default reward. Per-step terms are small, terminal terms are large.
REWARD = {
    "shaping": 1.0,   # weight of the (distance + speed + tilt) potential
    "time": 0.01,     # per step spent in the air
    "success": 10.0,  # landed and stayed 1 s on the ship
    "center": 5.0,    # extra, full when landing on the ship's center
    "crash": 5.0,     # hull touched, broken leg, left the screen
    "timeout": 5.0,   # still flying after MAX_STEPS
}

FUEL_CAPACITY = 700.0  # consumed by `power` (0.4-1.0) each step the main engine runs
LANDED_SPEED = 0.2  # (in units of START_SPEED) max speed to count as "landed"

# Scenario parameters. A difficulty level is a preset of these values (see level_config);
# any of them can be overridden with RocketLander(config={...}).
DEFAULT_CONFIG = {
    "ship_width": 1.0,             # landing zone width, x the default ship (~102 m)
    "start_x": 0.0,                # start position, fraction of the screen width from the center
    "start_x_spread": 0.03,        # +/- random offset added to start_x
    "start_height": 0.95,          # fraction of the screen height (~385 m)
    "start_speed": 40.0,           # initial falling speed, m/s
    "start_spin": 1.0,             # max random initial angular velocity, rad/s
    "start_drift": 0.0,            # random initial horizontal speed, fraction of start_speed
    "max_touchdown_speed": None,   # m/s, legs break above this vertical speed (None: never)
    "hard_landing_breaks": False,  # legs break if their joints are over-compressed
    "leg_scale": 1.0,              # leg length multiplier
    "end_on_landing": True,        # end the episode after 1 s landed (success)
    "max_steps": 1500,             # physics frames (60 = 1 s = 25 s); then the episode fails
    "fuel": FUEL_CAPACITY,
    "control": "direct",           # continuous actions: "direct" setpoints or "rate" of change
}


def level_config(level: int) -> dict:
    """Difficulty presets, monotonic: each level is at least as hard as the previous one.

        0       legs never break (crashing at full speed on the ship still "lands")
        1-9     legs break above a touchdown speed: 7.6 m/s at level 1 ... 4.4 m/s at level 9;
                more random start position and drift
        10-20   same 4.4 m/s limit + legs break on hard landings; from 11 the legs get
                shorter (x0.93 per level, down to x0.3 at level 20)
    """
    cfg = dict(DEFAULT_CONFIG)
    level = int(level)
    if level >= 1:
        cfg["max_touchdown_speed"] = 8.0 - 0.4 * min(level, 9)
    if level >= 2:
        cfg["start_x_spread"] = 0.03 * (1 + 0.1 * min(level, 10))
        cfg["start_drift"] = 0.2
        cfg["start_spin"] = 1.2
    if level >= 10:
        cfg["hard_landing_breaks"] = True
    if level >= 11:
        cfg["leg_scale"] = max(0.3, 1.0 - 0.07 * (level - 10))
    return cfg

START_HEIGHT = 1000.0
START_SPEED = 40.0

# ROCKET
GIMBAL_SLEW = 1.0 / 60  # max gimbal change per frame in "direct" control (rad)
THROTTLE_SLEW = 3.0 / 60  # max throttle change per frame in "direct" control
MIN_THROTTLE = 0.4
GIMBAL_THRESHOLD = 0.4
MAIN_ENGINE_POWER = 1600 * SCALE_S * 1.0
SIDE_ENGINE_POWER = 100 / FPS * SCALE_S * 2.0

ROCKET_WIDTH = 3.66 * SCALE_S
ROCKET_HEIGHT = ROCKET_WIDTH / 3.7 * 47.9
ENGINE_HEIGHT = ROCKET_WIDTH * 0.5
ENGINE_WIDTH = ENGINE_HEIGHT * 0.7
THRUSTER_HEIGHT = ROCKET_HEIGHT * 0.86

# LEGS
LEG_LENGTH = ROCKET_WIDTH * 4.2
BASE_ANGLE = -0.27
SPRING_ANGLE = 0.27
LEG_AWAY = ROCKET_WIDTH / 2

# SHIP
SHIP_HEIGHT = ROCKET_WIDTH
SHIP_WIDTH = SHIP_HEIGHT * 80

# VIEWPORT
VIEWPORT_H = 720
VIEWPORT_W = 500
H = 1.1 * START_HEIGHT * SCALE_S
W = float(VIEWPORT_W) / VIEWPORT_H * H

# SMOKE FOR VISUALS
MAX_SMOKE_LIFETIME = 2 * FPS

MEAN = np.array(
    [-0.034, -0.15, -0.016, 0.0024, 0.0024, 0.137, -0.02, -0.01, -0.8, 0.002]
)
VAR = np.sqrt(
    np.array([0.08, 0.33, 0.0073, 0.0023, 0.0023, 0.8, 0.085, 0.0088, 0.063, 0.076])
)


class ContactDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env

    def BeginContact(self, contact):
        if (
            self.env.water in [contact.fixtureA.body, contact.fixtureB.body]
            or self.env.lander in [contact.fixtureA.body, contact.fixtureB.body]
            or self.env.containers[0] in [contact.fixtureA.body, contact.fixtureB.body]
            or self.env.containers[1] in [contact.fixtureA.body, contact.fixtureB.body]
        ):
            self.env.game_over = True
        else:
            for i in range(2):
                if self.env.legs[i] in [contact.fixtureA.body, contact.fixtureB.body]:
                    self.env.legs[i].ground_contact = True

    def EndContact(self, contact):
        for i in range(2):
            if self.env.legs[i] in [contact.fixtureA.body, contact.fixtureB.body]:
                self.env.legs[i].ground_contact = False


class RocketLander(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": FPS}

    """
    level_number: difficulty preset, see level_config().
    config: overrides of the scenario parameters, see DEFAULT_CONFIG.
    reward_params: overrides of the default reward weights, see REWARD.
    """

    def __init__(self, level_number=0, continuous=True, reward_params=None, config=None):
        self.level_number = level_number
        self.config = {**level_config(level_number), **(config or {})}
        unknown = set(self.config) - set(DEFAULT_CONFIG)
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        self.reward_params = {**REWARD, **(reward_params or {})}
        self.seed()
        self.viewer = None
        self.episode_number = 0

        self.world = Box2D.b2World()
        self.water = None
        self.lander = None
        self.engine = None
        self.ship = None
        self.legs = []
        self.state = []
        self.continuous = continuous
        self.landed = False
        self.landed_fraction = []
        self.good_landings = 0
        self.total_landed_ticks = 0
        self.landed_ticks = 0
        self.done = False
        # Observations are standardized with MEAN/VAR, so they are not bounded to [-1, 1].
        obs_dim = 10 if VEL_STATE else 7
        high = np.full(obs_dim, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        if self.continuous:
            self.action_space = spaces.Box(-1, +1, (3,), dtype=np.float32)
        else:
            self.action_space = spaces.Discrete(7)

        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    _seed = seed  # backward compatibility

    def _destroy(self):
        if not self.water:
            return
        self.world.contactListener = None
        self.world.DestroyBody(self.water)
        self.water = None
        self.world.DestroyBody(self.lander)
        self.lander = None
        self.world.DestroyBody(self.ship)
        self.ship = None
        self.world.DestroyBody(self.legs[0])
        self.world.DestroyBody(self.legs[1])
        self.legs = []
        self.world.DestroyBody(self.containers[0])
        self.world.DestroyBody(self.containers[1])
        self.containers = []

    def reset(self):
        self._destroy()
        self.world.contactListener_keepref = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_keepref
        self.game_over = False
        self.prev_shaping = None
        self.episode_number += 1
        self.throttle = 0
        self.gimbal = 0.0
        self.total_landed_ticks += self.landed_ticks
        self.landed_ticks = 0
        cfg = self.config
        self.total_fuel = float(cfg["fuel"])
        self.touch_down = False
        self.total_rotation = 0.0  # cumulated rotation since the start, in turns (+ = counter-clockwise)
        self.ship_width = SHIP_WIDTH * cfg["ship_width"]
        self.leg_length = LEG_LENGTH * cfg["leg_scale"]
        self.stepnumber = 0
        self.smoke = []

        # self.terrainheigth = self.np_random.uniform(H / 20, H / 10)
        self.terrainheigth = H / 20
        self.shipheight = self.terrainheigth + SHIP_HEIGHT
        ship_pos = W / 2
        self.helipad_x1 = ship_pos - self.ship_width / 2
        self.helipad_x2 = self.helipad_x1 + self.ship_width
        self.helipad_y = self.terrainheigth + SHIP_HEIGHT

        self.water = self.world.CreateStaticBody(
            fixtures=fixtureDef(
                shape=polygonShape(
                    vertices=(
                        (0, 0),
                        (W, 0),
                        (W, self.terrainheigth),
                        (0, self.terrainheigth),
                    )
                ),
                friction=0.1,
                restitution=0.0,
            )
        )
        self.water.color1 = rgb(70, 96, 176)

        self.ship = self.world.CreateStaticBody(
            fixtures=fixtureDef(
                shape=polygonShape(
                    vertices=(
                        (self.helipad_x1, self.terrainheigth),
                        (self.helipad_x2, self.terrainheigth),
                        (self.helipad_x2, self.terrainheigth + SHIP_HEIGHT),
                        (self.helipad_x1, self.terrainheigth + SHIP_HEIGHT),
                    )
                ),
                friction=0.5,
                restitution=0.0,
            )
        )

        self.containers = []
        for side in [-1, 1]:
            self.containers.append(
                self.world.CreateStaticBody(
                    fixtures=fixtureDef(
                        shape=polygonShape(
                            vertices=(
                                (
                                    ship_pos + side * 0.95 * self.ship_width / 2,
                                    self.helipad_y,
                                ),
                                (
                                    ship_pos + side * 0.95 * self.ship_width / 2,
                                    self.helipad_y + SHIP_HEIGHT,
                                ),
                                (
                                    ship_pos
                                    + side * 0.95 * self.ship_width / 2
                                    - side * SHIP_HEIGHT,
                                    self.helipad_y + SHIP_HEIGHT,
                                ),
                                (
                                    ship_pos
                                    + side * 0.95 * self.ship_width / 2
                                    - side * SHIP_HEIGHT,
                                    self.helipad_y,
                                ),
                            )
                        ),
                        friction=0.2,
                        restitution=0.0,
                    )
                )
            )
            self.containers[-1].color1 = rgb(206, 206, 2)

        self.ship.color1 = (0.2, 0.2, 0.2)

        spread = cfg["start_x_spread"]
        initial_x = W / 2 + W * (cfg["start_x"] + self.np_random.uniform(-spread, spread))
        initial_x = float(np.clip(initial_x, 0.05 * W, 0.95 * W))
        initial_y = H * float(np.clip(cfg["start_height"], 0.15, 0.97))
        self.lander = self.world.CreateDynamicBody(
            position=(initial_x, initial_y),
            angle=0.0,
            fixtures=fixtureDef(
                shape=polygonShape(
                    vertices=(
                        (-ROCKET_WIDTH / 2, 0),
                        (+ROCKET_WIDTH / 2, 0),
                        (ROCKET_WIDTH / 2, +ROCKET_HEIGHT),
                        (-ROCKET_WIDTH / 2, +ROCKET_HEIGHT),
                    )
                ),
                density=1.0,
                friction=0.5,
                categoryBits=0x0010,
                maskBits=0x001,
                restitution=0.0,
            ),
        )

        self.lander.color1 = rgb(230, 230, 230)

        leg_length_modified = self.leg_length
        for i in [-1, +1]:
            leg = self.world.CreateDynamicBody(
                position=(initial_x - i * LEG_AWAY, initial_y + ROCKET_WIDTH * 0.2),
                angle=(i * BASE_ANGLE),
                fixtures=fixtureDef(
                    shape=polygonShape(
                        vertices=(
                            (0, 0),
                            (0, leg_length_modified / 25),
                            (i * leg_length_modified, 0),
                            (i * leg_length_modified, -leg_length_modified / 20),
                            (i * leg_length_modified / 3, -leg_length_modified / 7),
                        )
                    ),
                    density=1,
                    restitution=0.0,
                    friction=0.2,
                    categoryBits=0x0020,
                    maskBits=0x001,
                ),
            )
            leg.ground_contact = False
            leg.color1 = (0.25, 0.25, 0.25)
            rjd = revoluteJointDef(
                bodyA=self.lander,
                bodyB=leg,
                localAnchorA=(i * LEG_AWAY, ROCKET_WIDTH * 0.2),
                localAnchorB=(0, 0),
                enableLimit=True,
                maxMotorTorque=2500.0,
                motorSpeed=-0.05 * i,
                enableMotor=True,
            )
            djd = distanceJointDef(
                bodyA=self.lander,
                bodyB=leg,
                anchorA=(i * LEG_AWAY, ROCKET_HEIGHT / 8),
                anchorB=leg.fixtures[0].body.transform * (i * leg_length_modified, 0),
                collideConnected=False,
                frequencyHz=0.01,
                dampingRatio=0.9,
            )
            if i == 1:
                rjd.lowerAngle = -SPRING_ANGLE
                rjd.upperAngle = 0
            else:
                rjd.lowerAngle = 0
                rjd.upperAngle = +SPRING_ANGLE
            leg.joint = self.world.CreateJoint(rjd)
            leg.joint2 = self.world.CreateJoint(djd)

            self.legs.append(leg)

        self.lander.linearVelocity = (
            self.np_random.uniform(-1, 1) * cfg["start_drift"] * cfg["start_speed"],
            -float(cfg["start_speed"]),
        )
        self.lander.angularVelocity = cfg["start_spin"] * self.np_random.uniform(-1, 1)

        self.drawlist = (
            self.legs + [self.water] + [self.ship] + self.containers + [self.lander]
        )

        if self.continuous:
            return self.step([0, 0, 0])[0]
        else:
            return self.step(6)[0]

    def step(self, action):

        self.force_dir = 0

        if self.continuous:
            # plain floats: numpy calls on 3 scalars are the main cost of a step
            action = [min(1.0, max(-1.0, float(a))) for a in action]
            if self.config["control"] == "direct":
                # action = setpoint: gimbal angle, throttle (<= 0: engine off). The
                # actuators follow it at a limited speed.
                g_target = action[0] * GIMBAL_THRESHOLD
                t_target = max(0.0, action[1])
                self.gimbal += min(GIMBAL_SLEW, max(-GIMBAL_SLEW, g_target - self.gimbal))
                self.throttle += min(THROTTLE_SLEW, max(-THROTTLE_SLEW, t_target - self.throttle))
            else:
                # action = rate of change of gimbal / throttle
                self.gimbal += action[0] * 0.15 / FPS
                self.throttle += action[1] * 0.5 / FPS
            if action[2] > 0.5:
                self.force_dir = 1
            elif action[2] < -0.5:
                self.force_dir = -1
        else:
            if action == 0:
                self.gimbal += 0.05
            elif action == 1:
                self.gimbal -= 0.05
            elif action == 2:
                self.throttle += 0.1
            elif action == 3:
                self.throttle -= 0.1
            elif action == 4:  # left
                self.force_dir = -1
            elif action == 5:  # right
                self.force_dir = 1

        self.gimbal = min(GIMBAL_THRESHOLD, max(-GIMBAL_THRESHOLD, float(self.gimbal)))
        self.throttle = min(1.0, max(0.0, float(self.throttle)))
        self.power = (
            0
            if self.throttle == 0.0 or self.total_fuel <= 0
            else MIN_THROTTLE + self.throttle * (1 - MIN_THROTTLE)
        )

        # main engine force
        force_pos = (self.lander.position[0], self.lander.position[1])
        force = (
            -math.sin(self.lander.angle + self.gimbal) * MAIN_ENGINE_POWER * self.power,
            math.cos(self.lander.angle + self.gimbal) * MAIN_ENGINE_POWER * self.power,
        )
        self.lander.ApplyForce(force=force, point=force_pos, wake=False)

        # control thruster force
        lx, ly = self.lander.position
        force_pos_c = (
            lx + THRUSTER_HEIGHT * math.sin(self.lander.angle),
            ly + THRUSTER_HEIGHT * math.cos(self.lander.angle),
        )
        force_c = (
            -self.force_dir * math.cos(self.lander.angle) * SIDE_ENGINE_POWER,
            self.force_dir * math.sin(self.lander.angle) * SIDE_ENGINE_POWER,
        )
        self.lander.ApplyLinearImpulse(impulse=force_c, point=force_pos_c, wake=False)

        self.world.Step(1.0 / FPS, 60, 60)

        pos = self.lander.position
        vx, vy = self.lander.linearVelocity
        vel_l = (vx / START_SPEED, vy / START_SPEED)
        vel_a = self.lander.angularVelocity
        x_distance = (pos.x - W / 2) / W
        y_distance = (pos.y - self.shipheight) / (H - self.shipheight)

        angle = (self.lander.angle / math.pi) % 2
        if angle > 1:
            angle -= 2

        state = [
            2 * x_distance,
            2 * (y_distance - 0.5),
            angle,
            1.0 if self.legs[0].ground_contact else 0.0,
            1.0 if self.legs[1].ground_contact else 0.0,
            2 * (self.throttle - 0.5),
            (self.gimbal / GIMBAL_THRESHOLD),
        ]
        if VEL_STATE:
            state.extend([vel_l[0], vel_l[1], vel_a])
        self.state = state
        # REWARD -------------------------------------------------------------------------------------------------------
        # state variables for reward
        distance = math.hypot(3 * x_distance, y_distance)  # weight x position more
        speed = math.hypot(*vel_l)
        groundcontact = self.legs[0].ground_contact or self.legs[1].ground_contact
        cfg = self.config
        self.total_rotation += self.lander.angularVelocity / FPS / (2 * np.pi)
        brokenleg = False
        if cfg["hard_landing_breaks"] and groundcontact:
            # hard landing: leg joints pushed past their limits
            brokenleg = self.legs[0].joint.angle < 0 or self.legs[1].joint.angle > 0
        if cfg["max_touchdown_speed"] is not None and groundcontact:
            # touchdown too fast
            brokenleg = brokenleg or abs(self.lander.linearVelocity[1]) > cfg["max_touchdown_speed"]
        if groundcontact:
            self.touch_down = True
        outside = abs(pos.x - W / 2) > W / 2 or pos.y > H
        self.total_fuel = max(0.0, self.total_fuel - self.power)
        fuelcost = 0.1 * (0 * self.power + abs(self.force_dir)) / FPS
        self.landed = (
            self.legs[0].ground_contact
            and self.legs[1].ground_contact
            and speed < LANDED_SPEED
        )
        done = False
        self.done = False
        success = False
        timeout = False

        if outside or brokenleg:
            self.game_over = True

        R = self.reward_params
        reward = -fuelcost

        if self.game_over:
            done = True
        else:
            # Potential-based shaping: each step pays the improvement of
            # (distance to the ship + speed + tilt), so getting closer and slowing
            # down are rewarded along the way without changing the best policy.
            shaping = -R["shaping"] * (distance + speed + abs(angle))
            shaping += 0.1 * (self.legs[0].ground_contact + self.legs[1].ground_contact)
            if self.prev_shaping is not None:
                reward += shaping - self.prev_shaping
            self.prev_shaping = shaping

            if self.landed:
                self.landed_ticks += 1
            else:
                self.landed_ticks = 0
            if self.landed_ticks == FPS:
                success = True
                self.good_landings += 1
                done = cfg["end_on_landing"]
            if not done and self.stepnumber + 1 >= cfg["max_steps"]:
                # running out of time is a failure, otherwise hovering is a safe bet
                timeout = True
                done = True

        if not groundcontact:
            reward -= R["time"]
        if success:
            # up to +success, and up to +center more for landing on the ship's center
            reward += R["success"] + R["center"] * (1 - min(1.0, abs(pos.x - W / 2) / (self.ship_width / 2)))
        elif self.game_over:
            reward -= R["crash"]
        elif timeout:
            reward -= R["timeout"]
        self.done = done

        # REWARD -------------------------------------------------------------------------------------------------------

        self.stepnumber += 1

        # Everything a custom reward function may need, in physical-ish units
        # (same scale as `self.state`, i.e. before observation standardization).
        info = {
            "x": state[0],  # -1 (left edge) .. 1 (right edge), 0 = ship center
            "y": state[1],  # -1 = ship deck .. ~1 = start height
            "angle": angle,  # in units of pi, 0 = upright
            "left_leg_contact": bool(self.legs[0].ground_contact),
            "right_leg_contact": bool(self.legs[1].ground_contact),
            "throttle": self.throttle,  # 0..1
            "gimbal": self.gimbal / GIMBAL_THRESHOLD,  # -1..1
            "velocity_x": float(vel_l[0]),  # in units of START_SPEED
            "velocity_y": float(vel_l[1]),  # negative = falling
            "angular_velocity": float(vel_a),
            "distance": float(distance),  # to the ship center, x weighted x3
            "speed": float(speed),
            "power": float(self.power),
            "side_thruster": self.force_dir,  # -1, 0, 1
            "fuel": self.total_fuel / max(1e-9, float(cfg["fuel"])),  # 1 = full tank
            "x_on_ship": float((pos.x - W / 2) / (self.ship_width / 2)),  # -1 / +1 = ship edges
            "x_meters": float(pos.x - W / 2),
            "altitude": float(pos.y - self.shipheight),  # m above the deck (body base)
            "rotation": float(self.total_rotation),  # turns since the start, + = counter-clockwise
            "touch_down": self.touch_down,  # a leg has touched the ship this episode
            "landed": bool(self.landed),  # both legs on the ship, slow
            "landed_ticks": self.landed_ticks,
            "landed_full": self.landed_ticks >= FPS,  # success: landed for 1 s
            "game_over": bool(self.game_over),
            "timeout": timeout,  # did not land within max_steps
            "outside": bool(outside),
            "broken_leg": bool(brokenleg),
            "step": self.stepnumber,
        }

        self.info = info

        state = (np.array(state) - MEAN[: len(state)]) / VAR[: len(state)]
        return state.astype(np.float32), float(reward), done, info

    def render(self, mode="human", close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        from gym.envs.classic_control import rendering

        if self.viewer is None:

            self.viewer = rendering.Viewer(
                VIEWPORT_W, VIEWPORT_H, visible=(mode == "human")
            )

            self.viewer.set_bounds(0, W, 0, H)

            sky = rendering.FilledPolygon(((0, 0), (0, H), (W, H), (W, 0)))
            self.sky_color = rgb(126, 150, 233)
            sky.set_color(*self.sky_color)
            self.sky_color_half_transparent = (
                np.array((np.array(self.sky_color) + rgb(255, 255, 255))) / 2
            )
            self.viewer.add_geom(sky)

            self.rockettrans = rendering.Transform()

            engine = rendering.FilledPolygon(
                (
                    (0, 0),
                    (ENGINE_WIDTH / 2, -ENGINE_HEIGHT),
                    (-ENGINE_WIDTH / 2, -ENGINE_HEIGHT),
                )
            )
            self.enginetrans = rendering.Transform()
            engine.add_attr(self.enginetrans)
            engine.add_attr(self.rockettrans)
            engine.set_color(0.4, 0.4, 0.4)
            self.viewer.add_geom(engine)

            self.fire = rendering.FilledPolygon(
                (
                    (ENGINE_WIDTH * 0.4, 0),
                    (-ENGINE_WIDTH * 0.4, 0),
                    (-ENGINE_WIDTH * 1.2, -ENGINE_HEIGHT * 5),
                    (0, -ENGINE_HEIGHT * 8),
                    (ENGINE_WIDTH * 1.2, -ENGINE_HEIGHT * 5),
                )
            )
            self.fire.set_color(*rgb(255, 230, 107))
            self.firescale = rendering.Transform(scale=(1, 1))
            self.firetrans = rendering.Transform(translation=(0, -ENGINE_HEIGHT))
            self.fire.add_attr(self.firescale)
            self.fire.add_attr(self.firetrans)
            self.fire.add_attr(self.enginetrans)
            self.fire.add_attr(self.rockettrans)

            smoke = rendering.FilledPolygon(
                (
                    (ROCKET_WIDTH / 2, THRUSTER_HEIGHT * 1),
                    (ROCKET_WIDTH * 3, THRUSTER_HEIGHT * 1.03),
                    (ROCKET_WIDTH * 4, THRUSTER_HEIGHT * 1),
                    (ROCKET_WIDTH * 3, THRUSTER_HEIGHT * 0.97),
                )
            )
            smoke.set_color(*self.sky_color_half_transparent)
            self.smokescale = rendering.Transform(scale=(1, 1))
            smoke.add_attr(self.smokescale)
            smoke.add_attr(self.rockettrans)
            self.viewer.add_geom(smoke)

            self.gridfins = []
            for i in (-1, 1):
                finpoly = (
                    (i * ROCKET_WIDTH * 1.1, THRUSTER_HEIGHT * 1.01),
                    (i * ROCKET_WIDTH * 0.4, THRUSTER_HEIGHT * 1.01),
                    (i * ROCKET_WIDTH * 0.4, THRUSTER_HEIGHT * 0.99),
                    (i * ROCKET_WIDTH * 1.1, THRUSTER_HEIGHT * 0.99),
                )
                gridfin = rendering.FilledPolygon(finpoly)
                gridfin.add_attr(self.rockettrans)
                gridfin.set_color(0.25, 0.25, 0.25)
                self.gridfins.append(gridfin)

        if self.stepnumber % round(FPS / 10) == 0 and self.power > 0:
            s = [
                MAX_SMOKE_LIFETIME * self.power,  # total lifetime
                0,  # current lifetime
                self.power * (1 + 0.2 * np.random.random()),  # size
                np.array(self.lander.position)
                + self.power
                * ROCKET_WIDTH
                * 10
                * np.array(
                    (
                        np.sin(self.lander.angle + self.gimbal),
                        -np.cos(self.lander.angle + self.gimbal),
                    )
                )
                + self.power * 5 * (np.random.random(2) - 0.5),
            ]  # position
            self.smoke.append(s)

        for s in list(self.smoke):
            s[1] += 1
            if s[1] > s[0]:
                self.smoke.remove(s)
                continue
            t = rendering.Transform(translation=(s[3][0], s[3][1] + H * s[1] / 2000))
            self.viewer.draw_circle(
                radius=0.05 * s[1] + s[2],
                color=self.sky_color
                + (1 - (2 * s[1] / s[0] - 1) ** 2)
                / 3
                * (self.sky_color_half_transparent - self.sky_color),
            ).add_attr(t)

        self.viewer.add_onetime(self.fire)
        for g in self.gridfins:
            self.viewer.add_onetime(g)

        for obj in self.drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                path = [trans * v for v in f.shape.vertices]
                self.viewer.draw_polygon(path, color=obj.color1)

        for l in zip(self.legs, [-1, 1]):
            path = [
                self.lander.fixtures[0].body.transform
                * (l[1] * ROCKET_WIDTH / 2, ROCKET_HEIGHT / 8),
                l[0].fixtures[0].body.transform
                * (l[1] * self.leg_length * 0.8, 0),
            ]
            self.viewer.draw_polyline(
                path, color=self.ship.color1, linewidth=1 if START_HEIGHT > 500 else 2
            )

        self.viewer.draw_polyline(
            (
                (self.helipad_x2, self.terrainheigth + SHIP_HEIGHT),
                (self.helipad_x1, self.terrainheigth + SHIP_HEIGHT),
            ),
            color=rgb(206, 206, 2),
            linewidth=1,
        )

        self.rockettrans.set_translation(*self.lander.position)
        self.rockettrans.set_rotation(self.lander.angle)
        self.enginetrans.set_rotation(self.gimbal)
        self.firescale.set_scale(newx=1, newy=self.power * np.random.uniform(1, 1.3))
        self.smokescale.set_scale(newx=self.force_dir, newy=1)

        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


def rgb(r, g, b):
    return float(r) / 255, float(g) / 255, float(b) / 255
