"""Per-episode bookkeeping shared by training, evaluation and the web app: what happened
during an episode (outcome, flips, touchdown speed, legs order, hover time...) and a
light trajectory for visualisation. Challenges are judged on this summary."""

from __future__ import annotations

from typing import Optional

START_SPEED = 40.0  # m/s, unit of the velocities in `info`
FPS = 60
TRAJECTORY_EVERY = 6  # keep one point every N physics frames


def outcome_of(info: dict) -> str:
    if info.get("landed_full"):
        return "success"
    if info.get("broken_leg"):
        return "broken_leg"
    if info.get("outside"):
        return "outside"
    if info.get("game_over"):
        return "crash"
    return "timeout"


class EpisodeTracker:
    """Feed it the `info` dict of every physics frame (RocketLander.step)."""

    def __init__(self, keep_trajectory: bool = True):
        self.keep_trajectory = keep_trajectory
        self.reset()

    def reset(self) -> None:
        self.frames = 0
        self.max_rotation = 0.0
        self.touchdown_vy: Optional[float] = None
        self.first_contact_frame: Optional[int] = None
        self.first_contact_leg: Optional[str] = None
        self.both_contact_frame: Optional[int] = None
        self.contact_frames = 0
        self.hover_frames = 0
        self.landed_ever = False
        self.last: dict = {}
        self.trajectory: list[list[float]] = []

    def update(self, info: dict) -> None:
        self.frames += 1
        self.last = info
        self.max_rotation = max(self.max_rotation, abs(info.get("rotation", 0.0)))
        left, right = info.get("left_leg_contact"), info.get("right_leg_contact")
        vy = info.get("velocity_y", 0.0) * START_SPEED
        if (left or right) and self.first_contact_frame is None:
            self.first_contact_frame = self.frames
            self.first_contact_leg = "both" if left and right else ("left" if left else "right")
            self.touchdown_vy = vy
        if left and right and self.both_contact_frame is None:
            self.both_contact_frame = self.frames
        if left or right:
            self.contact_frames += 1
        if abs(info.get("altitude", 0.0) - 100.0) < 15.0 and abs(vy) < 3.0 and not info.get("game_over"):
            self.hover_frames += 1
        if info.get("landed_full"):
            self.landed_ever = True
        if self.keep_trajectory and (self.frames % TRAJECTORY_EVERY == 1):
            self.trajectory.append([round(info.get("x_meters", 0.0), 1), round(info.get("altitude", 0.0), 1),
                                    round(info.get("angle", 0.0), 3)])

    def summary(self) -> dict:
        i = self.last
        gap = None
        if self.first_contact_frame is not None and self.both_contact_frame is not None:
            gap = (self.both_contact_frame - self.first_contact_frame) / FPS
        out = {
            "outcome": "success" if self.landed_ever and not i.get("game_over") else outcome_of(i),
            "landed": self.landed_ever,
            "steps": self.frames,
            "duration_s": round(self.frames / FPS, 2),
            "x_on_ship": round(i.get("x_on_ship", 0.0), 3),
            "fuel": round(i.get("fuel", 0.0), 3),
            "max_rotation": round(self.max_rotation, 3),
            "touchdown_vy": None if self.touchdown_vy is None else round(self.touchdown_vy, 2),
            "first_leg": self.first_contact_leg,
            "contact_gap_s": None if gap is None else round(gap, 2),
            "contact_s": round(self.contact_frames / FPS, 2),
            "hover_s": round(self.hover_frames / FPS, 2),
        }
        if self.keep_trajectory:
            if i:
                self.trajectory.append([round(i.get("x_meters", 0.0), 1), round(i.get("altitude", 0.0), 1),
                                        round(i.get("angle", 0.0), 3)])
            out["trajectory"] = self.trajectory
        return out
