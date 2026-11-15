#!/usr/bin/env python3
"""
CALIBRATION CHECK - how far off is the camera calibration, measured with the SAME motion as final_run.py
==========================================================================================================

The arm is moved exactly the way final_run.py moves it to a peg:
  1 lift (skipped when no lift is possible from where the arm is - e.g. pointing down close to the base)
  2 ORIENT: one joint move to a point on the line base -> target, at the safe height (target + --hover), already
    with the final gripper pitch (0..5 deg, see below)
  3 straight out over the target (base angle and pitch fixed)
  4 straight down; the last PRE_MM (20 mm) slowly
  Every step is planned BEFORE anything moves. If it cannot be solved from the current position, other start
  positions (reach-out 130-260 mm from the base) are tried, like final_run.py. A fast closed-form check rejects
  unreachable targets at once. Each step is re-planned from the real joints just before it runs, and every move must
  really arrive (calib_check's Checker, which final_run.py also uses).

joint1: switched to Operating Mode 4 (multi-turn) at start if it is single-turn, and kept on ONE side of 0 within
  --joint1-limit (210 deg) - the same rule as final_run.py. A target outside that range is refused with a suggestion.
gripper pitch: between 0 and 5 deg (0 = horizontal, + = tilted down, like the teleop's 'p'); the current pitch if
  it is inside and works, else the closest one that works for the target AND the points above it (--pitch,
  --pitch-range MIN MAX).

THE OFFSET MEASUREMENT (why it changed): the tip is clicked in the camera image, and the camera ray through that
  pixel is cut with the horizontal plane at the tip's real height (known from the joints). The difference is the
  calibration's x, y error at that height - the error final_run.py makes when it picks a peg there. The depth camera
  is NOT used at the tip: on the thin gripper it mostly measures the table behind it, which gave ~25 mm of false
  z error. z is measured by touch in the marker mode (MODE 1).
  error = where the robot really is - where camera + calibration put it.  A constant error is removed with
      python3 calib_camera_node.py --solve omx_calibration.json --offset DX DY DZ     (the summary prints it)

START (ROS 2 sourced; final_run.py, the teleop, the camera node and the peg detector must NOT run):
  T1: ros2 launch open_manipulator_x_bringup hardware.launch.py
  T2: python3 calib_check.py omx_calibration.json --goto -150 50 75
  The torque is switched ON at start (the arm first holds its pose); --return / q / the end of MODE 1-3 bring the
  arm back to that start pose.

CHECK + OFFSET UPDATE (the usual way):   python3 calib_check.py omx_calibration.json --check
  The arm visits CHECK_POINTS (4 points spread over the work area; or your own: --check --points "x,y,z x,y,z ...").
  At each point: ENTER = freeze the image and CLICK the point between the finger tips (2-3 clicks, averaged),
  n = next point, q = finish early. At the end, back to the start pose and the summary per point. If the error is a
  constant shift measured at >= 3 points, the new offset (x, y; z kept as it is) is shown and, after y, WRITTEN into
  omx_calibration.json by calib_camera_node.py --solve ... --offset ... (backup omx_calibration.json.bak-<time>; the
  file is put back if anything fails). --yes = no question, --no-update = report only.
  The marker mode (touch) also updates z the same way.

MODE 4 - ONE POINT:   --goto X Y Z [PITCH]     (--goto alone = DEFAULT_POINTS[0])
      python3 calib_check.py omx_calibration.json --goto -150 50 75
      python3 calib_check.py omx_calibration.json --goto -150 50 75 3          gripper pitch 3 deg
      python3 calib_check.py omx_calibration.json --goto -150 50 75 --return   back to the start pose at the end
      python3 calib_check.py omx_calibration.json --goto -150 50 75 --no-camera
  Camera window: GREEN = where the calibration expects the point, RED = where it puts the robot tip now. At the point
  a picture is saved (goto_view.png); ENTER = freeze and CLICK the point between the finger tips -> x, y error;
  ENTER again = another click; q / ESC = finish (summary + offset).

MODE 1 - MARKERS (full 3D by touch):   python3 calib_check.py omx_calibration.json
  The camera measures every ArUco marker centre; the arm goes 10 mm above it (stops above it first: ENTER), you
  nudge the tip onto the centre (w/s x, a/d y, r/f z, [ ] step 0.5-5 mm), ENTER records it. n = skip, c = click an
  extra point on the board, q = quit.

MODE 2 - FIXED POINTS:   --points "x,y,z x,y,z ..."   (--points alone = DEFAULT_POINTS)
  The arm visits the points one by one; at each ENTER = click the tip (as in MODE 4), n = next.

MODE 3 - TYPE POINTS IN THE TERMINAL:   --terminal   (with --points: those first)
      -150 50 75 [pitch]   go there            d 0 10 0   relative to the last point (short moves = one line)
      s   stop the move     tip   where the tip is     home   back to the start pose     q   quit
  In the camera window ENTER = click the tip and measure.

Ctrl+C in the terminal = EMERGENCY STOP: the arm holds where it is and the program exits without moving.
final_run.py imports this file (Checker, Arm, make_ros, ...): keep those names and arguments when editing.
"""
import argparse
import json
import math
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import cv2
import numpy as np

import calib_camera_node as _n1          # same folder; works with older and newer copies of it

WIN, Aruco, overlay, point_3d = _n1.WIN, _n1.Aruco, _n1.overlay, _n1.point_3d
preflight = getattr(_n1, "preflight", lambda: None)


def open_camera(fps=15):
    oc = getattr(_n1, "open_camera", None)
    if oc is None:
        return _n1.Camera()
    try:
        return oc(fps)
    except TypeError:                     # older calib_camera_node.py: no frame-rate option
        return oc()


def _marker_at(markers, p, r=60):
    """Detected marker under / nearest to a clicked point (within r px)."""
    best = None
    for mid, c in markers:
        inside = cv2.pointPolygonTest(c.astype(np.float32).reshape(-1, 1, 2), (float(p[0]), float(p[1])), False) >= 0
        d = 0.0 if inside else float(np.linalg.norm(c.mean(0) - p))
        if d <= r and (best is None or d < best[0]):
            best = (d, mid, c)
    return None if best is None else (best[1], best[2])


HOVER_MM = 50.0          # first stop above the target
SAFE_MM = 10.0           # automatic descent stops this far above the target; the last mm are yours
MAX_NUDGE_MM = 40.0      # how far you may nudge away from the target
DIRECT_MM = 30.0         # MODE 3: moves shorter than this go as one straight line (no lift / hover)
# ---- trajectories (same scheme as square_final_path.py) ----
MAX_JOINT_VEL = 0.6      # rad/s
MAX_JOINT_ACC = 1.5      # rad/s^2
CART_SPEED_MM = 40.0     # mm/s, peak tip speed of straight lines (lift, travel up)   (--speed)
SLOW_SPEED_MM = 15.0     # mm/s, last descent onto the point and nudges
ANG_SPEED = 0.5          # rad/s, pitch change along a line
MIN_SEGMENT_T = 1.0      # s, shortest move
NUDGE_T = 0.4            # s, shortest nudge
DT = 0.02                # s between trajectory points
WAYPOINT_MM = 4.0        # mm between IK solutions on a straight line
MAX_STEP_RAD = 0.3       # a joint jumping more than this between two waypoints = elbow flip -> refused
PITCH_TOL_DEG = 0.3      # accepted gripper pitch error
PITCH_MIN_DEG = 0.0      # allowed gripper pitch during the check (0 = horizontal, + = tilted down), like the
PITCH_MAX_DEG = 5.0      # teleop's "p" (--pitch-range MIN MAX to change)
SETTLE_S = 3.0           # after each move: wait up to this long for the joints to really arrive
SETTLE_TOL_RAD = 0.015   # ... within this (0.9 deg)
W_PITCH = 0.1            # IK weight: 1 rad of pitch counts like 100 mm of position
# ---- joint1 (base) ----
MAX_BASE_TURN_DEG = 200.0    # refuse any single move that turns the base more (wires)
JOINT1_DXL_ID = 11           # Dynamixel ID of joint1 (its Operating Mode is read at start)
JOINT1_NEGATIVE_ONLY = True  # multi-turn: joint1 on ONE side of 0: [JOINT1_MIN_DEG, 0] (or the mirror image) - as
JOINT1_MIN_DEG = -210.0      # final_run.py (cables); --joint1-limit. +162.8 from -162.9 is reached as -197.2 (34 deg)
JOINT1_SOFT_LIMIT_DEG = 210.0  # (used when JOINT1_NEGATIVE_ONLY is False: [-this, +this])
# ---- final_run.py's planning (CalibRun) ----
START_R = (130.0, 260.0)     # orient pose: reach-out (mm from the joint1 axis) of the current position, limited
START_R_TRY = (180.0, 220.0, 150.0, 250.0, 130.0)   # other start positions tried if IK / a move fails
PRE_MM = 20.0                # the last PRE_MM of the descent are slow (--slow)
DIP_MM = 15.0                # a joint move may not sag more than this below its lower end point
SEAM_MARGIN_DEG = 0.5        # single-turn: joint1 stays this far inside +-180 deg. AT 180 deg the servo reading
                             # flips between +180 and -180 and the controller jumps a full turn (violent motion)
# MODE 2 (--points): fixed tip coordinates in the robot base frame (mm). Change them so they lie above your
# workspace and inside the camera view; keep z well above the table (the tip goes exactly there).
DEFAULT_POINTS = [[-236.0, 116.0, 60.0]]
# --check: 4 points spread over the work area (both sides of 180 deg, near and far, ordered so the base turns one way)
CHECK_POINTS = [[-236.0, 116.0, 60.0], [-220.0, 80.0, 70.0], [-150.0, 50.0, 75.0], [-170.0, -40.0, 70.0]]
CLICKS_PER_POINT = 2          # suggested clicks at each point (they are averaged)
MIN_UPDATE_POINTS = 3         # the calibration file is only updated from at least this many points
CLICK_SCATTER_WARN_MM = 6.0   # clicks at one point further apart than this: warn
MIN_POINT_Z = 20.0
SOFT_LIMITS = {'joint1': (-np.pi, np.pi), 'joint2': (-2.05, 1.57), 'joint3': (-1.55, 1.53), 'joint4': (-1.80, 2.00)}
STOP_WORDS = ("s", "stop")

TERMINAL_HELP = """
---------------------------------------------------------------------------
 TYPE IN THIS TERMINAL (mm, robot base frame), then press Enter:
   x y z [pitch]  move the tip there        e.g.  -190 -30 75   or  -190,-30,75,10
                  (pitch in deg, 0..5: 0 = gripper horizontal, + = tilted down)
   d dx dy dz     move relative to the last point   e.g.  d 0 10 0
   s              STOP the current move (arm holds where it is)
   tip            print where the tip is now
   home           back to the start pose
   help           this text
   q              quit (arm goes up and back to the start pose)
 Ctrl+C = EMERGENCY STOP: arm holds, program exits without moving
 In the CAMERA window: ENTER = freeze & click the tip to measure,  q = quit
---------------------------------------------------------------------------"""


class PlanError(RuntimeError):
    pass


class Quit(Exception):
    """'q' pressed: stop, then go up and back to the start pose."""


class StopMotion(Exception):
    """'s' typed in the terminal: stop this move, stay in the program."""


def wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def j1_window(multiturn, q1=None):
    """Allowed joint1 range [lo, hi] in rad. Multi-turn, one-sided: joint1 stays on the side of 0 where it is now
    (negative: [JOINT1_MIN_DEG, 0]; if it starts positive: the mirror image [0, -JOINT1_MIN_DEG])."""
    if not multiturn:
        return -math.pi - 1e-9, math.pi + 1e-9
    if JOINT1_NEGATIVE_ONLY:
        if q1 is not None and q1 > math.radians(0.5):
            return -1e-9, -math.radians(JOINT1_MIN_DEG)
        return math.radians(JOINT1_MIN_DEG), 1e-9
    lim = math.radians(JOINT1_SOFT_LIMIT_DEG)
    return -lim, lim


def unwrap_j1(target, ref, window):
    """joint1 = target + k*360 deg that lies inside window and is closest to ref."""
    lo, hi = window
    cands = [target + k * 2 * math.pi for k in (-2, -1, 0, 1, 2)]
    cands = [c for c in cands if lo - 1e-9 <= c <= hi + 1e-9]
    if not cands:
        raise PlanError(f'joint1 {math.degrees(target):+.1f} deg has no equivalent inside '
                        f'[{math.degrees(lo):+.0f}, {math.degrees(hi):+.0f}] deg')
    return min(cands, key=lambda c: abs(c - ref))


# ============================== quintic trajectories (from square_final_path.py) ==============================
@dataclass
class Traj:
    t: np.ndarray
    Q: np.ndarray
    Qd: np.ndarray
    Qdd: np.ndarray

    @property
    def T(self):
        return float(self.t[-1])

    @property
    def q_end(self):
        return self.Q[-1].copy()


def quintic(t, T):
    """Rest-to-rest quintic time scaling s(t) in [0,1] with s', s'' = 0 at both ends."""
    tau = np.clip(t / T, 0.0, 1.0)
    s = 10 * tau ** 3 - 15 * tau ** 4 + 6 * tau ** 5
    sd = (30 * tau ** 2 - 60 * tau ** 3 + 30 * tau ** 4) / T
    sdd = (60 * tau - 180 * tau ** 2 + 120 * tau ** 3) / T ** 2
    return s, sd, sdd


def plan_joint(q0, q1, vmax, amax, tmin, dt):
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
    dq = q1 - q0
    m = float(np.max(np.abs(dq)))
    if m < 1e-3:
        return None
    # quintic peaks: v = 1.875*d/T,  a = 5.7735*d/T^2
    T = max(tmin, 1.875 * m / vmax, math.sqrt(5.7735 * m / amax))
    n = max(2, int(math.ceil(T / dt)))
    t = np.linspace(0.0, T, n + 1)
    s, sd, sdd = quintic(t, T)
    return Traj(t, q0 + np.outer(s, dq), np.outer(sd, dq), np.outer(sdd, dq))


def hermite_path(Q):
    """C1 joint-space path through IK waypoints Q (uniform in path parameter s)."""
    Q = np.asarray(Q, float)
    N = len(Q)
    h = 1.0 / (N - 1)
    M = np.gradient(Q, np.linspace(0.0, 1.0, N), axis=0) if N > 2 else np.vstack([Q[1] - Q[0]] * 2)

    def ev(s):
        s = np.clip(np.asarray(s, float), 0.0, 1.0)
        i = np.minimum((s / h).astype(int), N - 2)
        u = ((s - i * h) / h)[:, None]
        p0, p1, m0, m1 = Q[i], Q[i + 1], M[i] * h, M[i + 1] * h
        P = ((2 * u ** 3 - 3 * u ** 2 + 1) * p0 + (u ** 3 - 2 * u ** 2 + u) * m0
             + (-2 * u ** 3 + 3 * u ** 2) * p1 + (u ** 3 - u ** 2) * m1)
        D1 = ((6 * u ** 2 - 6 * u) * p0 + (3 * u ** 2 - 4 * u + 1) * m0
              + (-6 * u ** 2 + 6 * u) * p1 + (3 * u ** 2 - 2 * u) * m1) / h
        D2 = ((12 * u - 6) * p0 + (6 * u - 4) * m0 + (-12 * u + 6) * p1 + (6 * u - 2) * m1) / h ** 2
        return P, D1, D2
    return ev


def retime_quintic(Q, T0, vmax, amax, dt):
    """Put a quintic time law s(t) on a geometric path given as IK waypoints."""
    ev = hermite_path(Q)
    T = T0
    for _ in range(12):
        n = max(2, int(math.ceil(T / dt)))
        t = np.linspace(0.0, T, n + 1)
        s, sd, sdd = quintic(t, T)
        P, D1, D2 = ev(s)
        Qd = D1 * sd[:, None]
        Qdd = D2 * (sd ** 2)[:, None] + D1 * sdd[:, None]
        P[0], P[-1] = Q[0], Q[-1]
        Qd[0] = Qd[-1] = 0.0
        Qdd[0] = Qdd[-1] = 0.0
        r = max(float(np.max(np.abs(Qd))) / vmax, math.sqrt(float(np.max(np.abs(Qdd))) / amax))
        if r <= 1.001:
            return Traj(t, P, Qd, Qdd)
        T *= 1.05 * r
    raise PlanError('could not satisfy joint velocity/acceleration limits')


# ============================== arm model from the URDF ==============================
def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def _axis_angle(a, t):
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * K @ K


class Arm:
    """Forward / inverse kinematics of the chain base_frame -> ee_frame, built from the URDF text."""

    def __init__(self, urdf, base, ee, tip_offset_mm):
        root = ET.fromstring(urdf)
        by_child = {}
        for j in root.findall("joint"):
            by_child[j.find("child").get("link")] = j
        chain, link = [], ee
        while link != base:
            j = by_child.get(link)
            if j is None:
                raise ValueError(f"no joint chain from '{base}' to '{ee}' in the URDF")
            chain.append(j)
            link = j.find("parent").get("link")
        self.joints = []
        for j in reversed(chain):
            o = j.find("origin")
            xyz = np.array([float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()])
            rpy = [float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()]
            ax = j.find("axis")
            axis = np.array([float(v) for v in (ax.get("xyz") if ax is not None else "1 0 0").split()])
            lim = j.find("limit")
            lo = float(lim.get("lower", -np.pi)) if lim is not None else -np.pi
            hi = float(lim.get("upper", np.pi)) if lim is not None else np.pi
            name = j.get("name")
            if name in SOFT_LIMITS:                       # same soft limits as the teleop
                lo, hi = max(lo, SOFT_LIMITS[name][0]), min(hi, SOFT_LIMITS[name][1])
            self.joints.append({"name": name, "type": j.get("type"), "R0": _rpy(*rpy), "t0": xyz,
                                "axis": axis, "lo": lo, "hi": hi})
        self.names = [j["name"] for j in self.joints if j["type"] in ("revolute", "continuous")]
        self.lo = np.array([j["lo"] for j in self.joints if j["name"] in self.names])
        self.hi = np.array([j["hi"] for j in self.joints if j["name"] in self.names])
        self.j1_urdf = (float(self.lo[0]), float(self.hi[0]))   # joint1 limits from the URDF
        self.lo[0], self.hi[0] = -np.pi, np.pi           # joint1 is solved single-turn, then unwrapped
        self.offset = np.asarray(tip_offset_mm, float) / 1000.0
        # direction the tool points, in the ee frame: from the last moving joint towards the tip
        q0 = np.zeros(len(self.names))
        T, frames = self.fk(q0, frames=True)
        d = T[:3, :3].T @ (self.tip(q0) - frames[-1])
        self.tool = d / np.linalg.norm(d) if np.linalg.norm(d) > 1e-6 else np.array([1.0, 0, 0])
        self.j1_xy = frames[0][:2].copy()                # joint1 axis in the base frame (m)

    def fk(self, q, frames=False):
        T, k, joint_pos = np.eye(4), 0, []
        for j in self.joints:
            A = np.eye(4)
            A[:3, :3], A[:3, 3] = j["R0"], j["t0"]
            T = T @ A
            if j["type"] in ("revolute", "continuous"):
                joint_pos.append(T[:3, 3].copy())
                Rj = np.eye(4)
                Rj[:3, :3] = _axis_angle(j["axis"], q[k])
                T = T @ Rj
                k += 1
        return (T, joint_pos) if frames else T

    def tip(self, q):
        T = self.fk(q)
        return T[:3, 3] + T[:3, :3] @ self.offset

    def down(self, q):
        """0 when the tool points straight down, 2 when straight up."""
        T = self.fk(q)
        return float((T[:3, :3] @ self.tool)[2] + 1.0)

    def pitch(self, q):
        """Gripper pitch in the arm's vertical plane: 0 = horizontal outward, +pi/2 = straight down
        (= joint2 + joint3 + joint4 on the OpenMANIPULATOR-X)."""
        d = self.fk(q)[:3, :3] @ self.tool
        h = d[0] * math.cos(q[0]) + d[1] * math.sin(q[0])
        return math.atan2(-d[2], h)

    def _seeds(self, target_m, q_seed):
        s0 = np.array(q_seed, float)
        s0[0] = wrap(s0[0])
        seeds = [np.clip(s0, self.lo, self.hi)]
        yaw = math.atan2(target_m[1] - self.j1_xy[1], target_m[0] - self.j1_xy[0])
        for s in ([0.0, 0.0, 0.0], [-0.5, 0.5, 1.0], [0.5, -0.5, 1.0], [0.3, 0.3, 0.8], [-1.0, 0.7, 0.3],
                  [0.6, -0.6, 0.8]):
            seeds.append(np.clip(np.array([yaw] + s[:len(self.names) - 1]), self.lo, self.hi))
        return seeds

    def ik(self, target_m, q_now):
        """Joint angles that put the TIP on target_m, preferring the tool pointing down. -> (q, error_mm)"""
        best = None
        for q0 in self._seeds(target_m, q_now):
            for w_o in (0.05, 0.0):                       # with orientation preference, then position only
                q = q0.copy()
                for _ in range(150):
                    e = target_m - self.tip(q)
                    r = np.r_[e, -w_o * self.down(q)] if w_o else e
                    J = np.zeros((len(r), len(q)))
                    for i in range(len(q)):
                        dq = np.zeros(len(q))
                        dq[i] = 1e-5
                        f1 = np.r_[self.tip(q + dq), w_o * self.down(q + dq)] if w_o else self.tip(q + dq)
                        f0 = np.r_[self.tip(q), w_o * self.down(q)] if w_o else self.tip(q)
                        J[:, i] = (f1 - f0) / 1e-5
                    step = J.T @ np.linalg.solve(J @ J.T + 1e-6 * np.eye(len(r)), r)
                    q = np.clip(q + np.clip(step, -0.2, 0.2), self.lo, self.hi)
                    if np.linalg.norm(step) < 1e-7:
                        break
                err = float(np.linalg.norm(target_m - self.tip(q))) * 1000
                score = (err > 0.5, err if err > 0.5 else self.down(q))
                if best is None or score < best[0]:
                    best = (score, q, err)
                if err <= 0.5:
                    break
        return best[1], best[2]

    def _f(self, q):
        return np.r_[self.tip(q), W_PITCH * self.pitch(q)]

    def ik_pitch(self, target_m, pitch, q_seed, chained=False):
        """Joint angles that put the TIP on target_m with gripper pitch `pitch` (rad).
        chained: start only from q_seed (straight lines). -> (q, position error mm, pitch error rad)"""
        target_m = np.asarray(target_m, float)
        seeds = self._seeds(target_m, q_seed)
        s0 = seeds[0]
        if chained:
            seeds = seeds[:1]
        tol = math.radians(PITCH_TOL_DEG)
        best = None
        for q in seeds:
            q = q.copy()
            for _ in range(100):
                f0 = self._f(q)
                r = np.r_[target_m - f0[:3], W_PITCH * wrap(pitch - f0[3] / W_PITCH)]
                if np.linalg.norm(r[:3]) < 1e-5 and abs(r[3]) < W_PITCH * 1e-4:
                    break
                J = np.zeros((4, len(q)))
                for i in range(len(q)):
                    dq = np.zeros(len(q))
                    dq[i] = 1e-6
                    J[:, i] = (self._f(q + dq) - f0) / 1e-6
                step = J.T @ np.linalg.solve(J @ J.T + 1e-8 * np.eye(4), r)
                q = np.clip(q + np.clip(step, -0.2, 0.2), self.lo, self.hi)
                if np.linalg.norm(step) < 1e-9:
                    break
            perr = float(np.linalg.norm(target_m - self.tip(q))) * 1000
            aerr = abs(wrap(pitch - self.pitch(q)))
            good = perr <= 0.5 and aerr <= tol
            d = q - s0
            d[0] = wrap(d[0])
            key = (not good, float(np.linalg.norm(d)) if good else perr + math.degrees(aerr))
            if best is None or key < best[0]:
                best = (key, q, perr, aerr)
            if good and chained:
                break
        return best[1], best[2], best[3]


# ============================== ROS side ==============================
def make_ros(args):
    import rclpy
    import rclpy.time
    from action_msgs.msg import GoalStatus
    from control_msgs.action import FollowJointTrajectory
    from rclpy.action import ActionClient
    from rclpy.duration import Duration
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, QoSProfile
    from sensor_msgs.msg import JointState
    from std_msgs.msg import String
    from std_srvs.srv import SetBool
    from tf2_ros import Buffer, TransformListener
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

    class CheckNode(Node):
        SUCCEEDED = GoalStatus.STATUS_SUCCEEDED

        def __init__(self):
            super().__init__("calib_check")
            self.lock, self.joints, self.urdf = threading.Lock(), {}, None
            self.create_subscription(JointState, "/joint_states", self._js, 10)
            qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
            self.create_subscription(String, "/robot_description", lambda m: setattr(self, "urdf", m.data), qos)
            self.pub = self.create_publisher(JointTrajectory, "/arm_controller/joint_trajectory", 10)
            self.arm_ac = ActionClient(self, FollowJointTrajectory, "/arm_controller/follow_joint_trajectory")
            self.tf = Buffer()
            self.tfl = TransformListener(self.tf, self)
            self.torque_cli = self.create_client(SetBool, "/dynamixel_hardware_interface/set_dxl_torque")

        def _wait_future(self, fut, seconds):
            t0 = time.time()
            while not fut.done() and time.time() - t0 < seconds:
                time.sleep(0.01)
            return fut.done()

        def torque_on(self):
            """Same as the teleop's 't': servo service, torque ON. -> message"""
            if not self.torque_cli.wait_for_service(timeout_sec=2.0):
                return None
            fut = self.torque_cli.call_async(SetBool.Request(data=True))
            if not self._wait_future(fut, 3.0):
                return "torque service call timed out"
            r = fut.result()
            return ("torque ON" if r.success else "torque ON failed") + (f": {r.message}" if r.message else "")

        def read_dxl_item(self, dxl_id, item, tries=3):
            """One control-table item of a Dynamixel (via the driver's get_dxl_data service) or None."""
            try:
                from dynamixel_interfaces.srv import GetDataFromDxl
            except ImportError:
                return None
            if not hasattr(self, "_get_cli"):
                self._get_cli = self.create_client(GetDataFromDxl, "/dynamixel_hardware_interface/get_dxl_data")
            if not self._get_cli.wait_for_service(timeout_sec=2.0):
                return None
            for _ in range(tries):                         # the driver answers from its read cycle: retry
                req = GetDataFromDxl.Request()
                req.id = int(dxl_id)
                req.item_name = item
                req.timeout_sec = 1.0
                fut = self._get_cli.call_async(req)
                if self._wait_future(fut, 3.0):
                    res = fut.result()
                    if res is not None and res.result:
                        return int(res.item_data)
                time.sleep(0.2)
            return None

        def read_dxl_mode(self, dxl_id):
            """Operating Mode of a Dynamixel (3 single-turn, 4 multi-turn) or None."""
            return self.read_dxl_item(dxl_id, "Operating Mode")

        def set_dxl_item(self, dxl_id, item, value):
            """Queue a control-table write in the driver (set_dxl_data). NOTE: for any item except 'Torque Enable'
            the driver switches that servo's torque OFF first and leaves it off. -> True if accepted"""
            try:
                from dynamixel_interfaces.srv import SetDataToDxl
            except ImportError:
                return False
            if not hasattr(self, "_set_cli"):
                self._set_cli = self.create_client(SetDataToDxl, "/dynamixel_hardware_interface/set_dxl_data")
            if not self._set_cli.wait_for_service(timeout_sec=2.0):
                return False
            req = SetDataToDxl.Request()
            req.id = int(dxl_id)
            req.item_name = item
            req.item_data = int(value)
            fut = self._set_cli.call_async(req)
            return self._wait_future(fut, 3.0) and fut.result() is not None and bool(fut.result().result)

        def _js(self, m):
            with self.lock:
                for n, p in zip(m.name, m.position):
                    self.joints[n] = p

        def q(self, names):
            with self.lock:
                return np.array([self.joints[n] for n in names]) if all(n in self.joints for n in names) else None

        def tf_ee(self, base, ee):
            try:
                t = self.tf.lookup_transform(base, ee, rclpy.time.Time())
            except Exception:
                return None
            tr, ro = t.transform.translation, t.transform.rotation
            x, y, z, w = ro.x, ro.y, ro.z, ro.w
            T = np.eye(4)
            T[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                         [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                         [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
            T[:3, 3] = [tr.x, tr.y, tr.z]
            return T

        def send(self, names, q, seconds):
            """One-point command on the topic (used before the torque comes on)."""
            msg = JointTrajectory()
            msg.joint_names = list(names)
            pt = JointTrajectoryPoint()
            pt.positions = [float(v) for v in q]
            pt.time_from_start = Duration(seconds=float(seconds)).to_msg()
            msg.points = [pt]
            self.pub.publish(msg)

        def send_traj(self, names, tr):
            """Full trajectory (positions, velocities, accelerations). -> goal future, or None if sent on the topic"""
            jt = JointTrajectory()
            jt.joint_names = list(names)
            for k in range(1, len(tr.t)):                  # skip t=0: the controller blends from the current state
                pt = JointTrajectoryPoint()
                pt.positions = [float(v) for v in tr.Q[k]]
                pt.velocities = [float(v) for v in tr.Qd[k]]
                pt.accelerations = [float(v) for v in tr.Qdd[k]]
                pt.time_from_start = Duration(nanoseconds=int(round(float(tr.t[k]) * 1e9))).to_msg()
                jt.points.append(pt)
            if self.arm_ac.server_is_ready():
                goal = FollowJointTrajectory.Goal()
                goal.trajectory = jt
                goal.goal_time_tolerance = Duration(seconds=1.0).to_msg()
                return self.arm_ac.send_goal_async(goal)
            self.pub.publish(jt)
            return None

        def hold(self, names, q):
            """Stay exactly at q now (preempts any running trajectory)."""
            try:
                jt = JointTrajectory()
                jt.joint_names = list(names)
                pt = JointTrajectoryPoint()
                pt.positions = [float(v) for v in q]
                pt.velocities = [0.0] * len(q)
                pt.time_from_start = Duration(seconds=0.1).to_msg()
                jt.points = [pt]
                if self.arm_ac.server_is_ready():
                    goal = FollowJointTrajectory.Goal()
                    goal.trajectory = jt
                    self.arm_ac.send_goal_async(goal)
                self.pub.publish(jt)                       # backup path through the topic
            except Exception as e:
                print(f"  hold command failed: {e}")

    try:
        from rclpy.signals import SignalHandlerOptions
        rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    except (ImportError, TypeError):
        rclpy.init()
    node = CheckNode()
    exe = SingleThreadedExecutor()
    exe.add_node(node)

    def spin():
        try:
            exe.spin()
        except Exception:
            pass

    th = threading.Thread(target=spin, daemon=True)
    th.start()
    return rclpy, node, exe, th


def wait_for(cond, seconds, what):
    t0 = time.time()
    while time.time() - t0 < seconds:
        v = cond()
        if v is not None:
            return v
        time.sleep(0.05)
    raise SystemExit(f"Timed out waiting for {what}.")


# ============================== check ==============================
# =============================================================================================================
#  Checker = the motion machinery SHARED WITH final_run.py (its PegSortRun is a subclass of Checker):
#  URDF arm model, IK, straight lines, quintic joint moves, execute (every move must really arrive), joint1 mode 4.
#  final_run.py calls: current_q, tip_now_mm, hold, solve, j1_lim, plan_joint_to, plan_line, execute, _tick, show,
#  switch_joint1_mode4, _estop, close_ros  -  keep their names and arguments.
#  The calibration check itself is CalibRun (below), planned like final_run.py.
# =============================================================================================================
class Checker:
    def __init__(self, args):
        """Start-up as in final_run.py: arm model checked against TF, servo reads made reliable, joint1 mode,
        hold the current pose, torque ON, joint1 to multi-turn for the session."""
        with open(args.calib) as f:
            self.cal = json.load(f)
        self.T = np.array(self.cal["BASE_T_CAMERA_mm"], float)           # includes the saved fine offset
        self.Ti = np.linalg.inv(self.T)
        self.base = self.cal.get("base_frame", "world")
        self.args = args
        self.inbox = None                                                # terminal mode only
        self.view_target = None                                          # camera point drawn GREEN while moving
        self.headless = False                                            # camera works but no window (no display)
        self.stop_evt = threading.Event()
        self.multiturn = False
        self.spin_th = None
        self.results = []
        self.rclpy, self.node, self.exe, self.spin_th = make_ros(args)
        urdf = wait_for(lambda: self.node.urdf, 5, "/robot_description (is hardware.launch.py running?)")
        self.arm = Arm(urdf, self.base, args.ee_frame, args.tip_offset)
        self.names = self.arm.names
        q = wait_for(lambda: self.node.q(self.names), 10, "/joint_states (is hardware.launch.py running?)")
        T_tf = wait_for(lambda: self.node.tf_ee(self.base, args.ee_frame), 5, f"TF {self.base}->{args.ee_frame}")
        d = np.linalg.norm(self.arm.fk(q)[:3, 3] - T_tf[:3, 3]) * 1000
        print(f"Arm model from the URDF: joints {self.names}; model vs live TF: {d:.2f} mm")
        if d > 2.0:
            raise SystemExit("The URDF model does not match the live TF - check --ee-frame / the base frame.")
        self._consensus_reads()
        mode = self.node.read_dxl_mode(JOINT1_DXL_ID)
        if mode is None:
            self.multiturn = bool(args.joint1_multiturn)
            print(f"joint1 Operating Mode: not readable - using {'multi' if self.multiturn else 'single'}-turn")
        else:
            self.multiturn = mode == 4
            print(f"joint1 Operating Mode: {mode} -> {'multi-turn' if self.multiturn else 'single-turn (+-180 deg)'}")
        q = self.current_q()
        self.window = j1_window(self.multiturn, q[0])
        # ---- torque ON before anything moves: first hold the CURRENT pose (no jump), then enable the servos ----
        self.node.send(self.names, q, 0.5)
        time.sleep(0.6)
        msg = self.node.torque_on()
        print(msg if msg is not None else "torque service not found - make sure the torque is ON")
        time.sleep(0.5)
        wait_for(lambda: self.node.q(self.names), 5, "/joint_states")
        if not self.multiturn and not args.keep_single_turn:     # multi-turn so +-180 deg is crossed the short way
            self.switch_joint1_mode4()
        if not self.node.arm_ac.wait_for_server(timeout_sec=3.0):
            print("action /arm_controller/follow_joint_trajectory not found - trajectories go on the topic")
        q = self.current_q()
        lo, hi = self.window
        print(f"joint1 {math.degrees(q[0]):+.1f} deg, allowed [{math.degrees(lo):+.0f}, {math.degrees(hi):+.0f}] deg "
              f"({'multi-turn' if self.multiturn else 'single-turn'}); base turns over {MAX_BASE_TURN_DEG:.0f} deg "
              f"refused")
        if not lo - 0.01 <= q[0] <= hi + 0.01:
            print("WARNING: joint1 is outside its allowed range - moves will be refused. Restart hardware.launch.py "
                  "or change --joint1-limit.")
        self.park = q.copy()
        self.q_cmd = q.copy()

    def _consensus_reads(self):
        """(as in final_run.py) The ROBOTIS driver can answer a get_dxl_data request with the value of an EARLIER
        request for the same item. Read until two answers in a row agree."""
        raw = getattr(self.node, "read_dxl_item", None)
        if raw is None:
            return

        def read(dxl_id, item, tries=6):
            vals = []
            for _ in range(tries):
                v = raw(dxl_id, item, 1)
                if v is not None:
                    vals.append(v)
                    if len(vals) >= 2 and vals[-1] == vals[-2]:
                        return v
            return vals[-1] if vals else None
        self.node.read_dxl_item = read

    # ---- state ----
    def current_q(self):
        """Measured joints; joint1 wrapped to +-180 deg in single-turn mode (as in square_final_path.py)."""
        q = self.node.q(self.names)
        if q is None:
            raise PlanError("no /joint_states")
        if not self.multiturn:
            q[0] = wrap(q[0])
        return q

    def tip_now_mm(self):
        q = self.node.q(self.names)
        return None if q is None else self.arm.tip(q) * 1000

    def switch_joint1_mode4(self):
        """Switch the joint1 servo (ID 11) to Operating Mode 4 = extended position (multi-turn) for THIS session, so
        the base can cross +-180 deg the short way (e.g. -163 -> -197 deg = 34 deg instead of 326 deg the long way).
        Only joint1 loses torque, for about one second; its axis is vertical, so nothing falls.
        The servo keeps mode 4 (it is stored in the servo): the Humble open_manipulator_x_system.ros2_control.xacro
        does not set an Operating Mode, so restarts and power cycles do not change it back."""
        print("joint1: switching to Operating Mode 4 (multi-turn) for this session ...")
        q = self.current_q()
        self.node.send(self.names, q, 0.3)                       # controller holds exactly where the arm is
        time.sleep(0.4)
        if not self.node.set_dxl_item(JOINT1_DXL_ID, "Operating Mode", 4):   # driver turns joint1 torque OFF first
            print("  could not write Operating Mode (service /dynamixel_hardware_interface/set_dxl_data missing or "
                  "refused) - staying single-turn")
        time.sleep(0.6)
        mode = self.node.read_dxl_mode(JOINT1_DXL_ID)
        q_new = self.node.q(self.names)                          # raw reading, not wrapped
        if q_new is not None and abs(q_new[0] - q[0]) > math.radians(3):
            print(f"  note: joint1 reading changed {math.degrees(q[0]):+.1f} -> {math.degrees(q_new[0]):+.1f} deg "
                  f"while switching - holding at the new reading")
        # torque back ON for joint1 only, with the controller holding the reading it has NOW (no jump)
        hold_q = q_new if q_new is not None else q
        self.node.send(self.names, hold_q, 0.3)
        time.sleep(0.4)
        self.node.set_dxl_item(JOINT1_DXL_ID, "Torque Enable", 1)
        time.sleep(0.6)
        torque = self.node.read_dxl_item(JOINT1_DXL_ID, "Torque Enable")
        if torque != 1:
            self.node.set_dxl_item(JOINT1_DXL_ID, "Torque Enable", 1)             # one retry
            time.sleep(0.6)
            torque = self.node.read_dxl_item(JOINT1_DXL_ID, "Torque Enable")
        if torque != 1:
            print("  WARNING: could not confirm joint1 torque is ON - press 't' in the teleop or restart "
                  "hardware.launch.py before moving")
        if mode == 4:
            self.multiturn = True
            q1 = self.node.q(self.names)[0]
            self.window = j1_window(True, q1)
            print(f"  joint1 is now multi-turn (Operating Mode 4), torque {'ON' if torque == 1 else 'NOT confirmed'}. "
                  f"The servo keeps this setting;\n"
                  f"  other programs (teleop) may see joint1 angles beyond +-180 deg.")
        else:
            print(f"  joint1 Operating Mode reads {mode} - NOT switched, staying single-turn")

    def hold(self):
        q = self.node.q(self.names)
        if q is not None:
            if not self.multiturn:
                q[0] = wrap(q[0])
            self.node.hold(self.names, q)
            self.q_cmd = q

    # ---- planning (shared with final_run.py) ----
    def solve(self, p_mm, pitch, q_ref, chained=False):
        """IK for tip p_mm (base frame) with gripper pitch (rad); joint1 unwrapped next to q_ref."""
        p_mm = np.asarray(p_mm, float)
        tol = math.radians(PITCH_TOL_DEG)
        q, perr, aerr = self.arm.ik_pitch(p_mm / 1000.0, pitch, q_ref, chained)
        if chained and (perr > 0.5 or aerr > tol):
            q, perr, aerr = self.arm.ik_pitch(p_mm / 1000.0, pitch, q_ref, chained=False)
        if perr > 1.0:
            raise PlanError(f"x={p_mm[0]:.1f} y={p_mm[1]:.1f} z={p_mm[2]:.1f} mm with pitch "
                            f"{math.degrees(pitch):.1f} deg is out of reach (closest {perr:.1f} mm away)")
        if aerr > tol:
            raise PlanError(f"x={p_mm[0]:.1f} y={p_mm[1]:.1f} z={p_mm[2]:.1f} mm is reachable only with pitch "
                            f"{math.degrees(self.arm.pitch(q)):.1f} deg, not {math.degrees(pitch):.1f} deg "
                            f"(joint limits) - give another pitch")
        q = q.copy()
        q[0] = unwrap_j1(q[0], q_ref[0], self.window)
        lo, hi = self.j1_lim()
        if not lo <= q[0] <= hi:                                 # on / next to the +-180 deg seam (single-turn)
            q[0] = min(max(q[0], lo), hi)
            err = float(np.linalg.norm(self.arm.tip(q) * 1000 - p_mm))
            if err > 1.0:
                r = math.hypot(p_mm[0] - self.arm.j1_xy[0] * 1000, p_mm[1] - self.arm.j1_xy[1] * 1000)
                need = math.ceil(r * math.sin(math.radians(SEAM_MARGIN_DEG)) + 1.0)
                raise PlanError(
                    f"x={p_mm[0]:.1f} y={p_mm[1]:.1f} mm lies on the joint1 +-180 deg seam (straight behind the base). "
                    f"A single-turn joint1 must stay {SEAM_MARGIN_DEG} deg away from 180 deg (there its reading flips "
                    f"sign and the controller jumps a full turn). Use y >= +{need} or y <= -{need} mm, or set joint1 "
                    f"to multi-turn (Operating Mode 4)")
        return q

    def j1_lim(self):
        """Range joint1 may be commanded to: single-turn keeps SEAM_MARGIN_DEG away from +-180 deg."""
        if self.multiturn:
            return self.window
        m = math.radians(SEAM_MARGIN_DEG)
        return max(-math.pi + m, self.arm.j1_urdf[0]), min(math.pi - m, self.arm.j1_urdf[1])

    def plan_joint_to(self, q0, q_goal):
        tgt = np.array(q_goal, float)
        tgt[0] = unwrap_j1(tgt[0], q0[0], self.window)
        lo, hi = self.j1_lim()
        tgt[0] = min(max(tgt[0], lo), hi)
        turn = abs(tgt[0] - q0[0])
        if turn > math.radians(MAX_BASE_TURN_DEG):
            raise PlanError(
                f"the base would turn {math.degrees(turn):.0f} deg (joint1 {math.degrees(q0[0]):+.1f} -> "
                f"{math.degrees(tgt[0]):+.1f} deg): the point is on the other side of the +-180 deg seam. "
                f"Refused (wires). Options: (1) turn torque off, hold the arm and rotate the base by hand to that "
                f"side, then run again; (2) let joint1 be switched to multi-turn (do not use --keep-single-turn; "
                f"restart hardware.launch.py if the switch failed) so it crosses 180 deg the SHORT way; "
                f"(3) raise --joint1-limit if the short way needs more than +-{JOINT1_SOFT_LIMIT_DEG:.0f} deg; "
                f"(4) if the cables and the camera are clear, allow the long way round with "
                f"--max-base-turn {math.ceil(math.degrees(turn) / 10) * 10:.0f}")
        return plan_joint(q0, tgt, MAX_JOINT_VEL, MAX_JOINT_ACC, MIN_SEGMENT_T, DT)

    def plan_line(self, q0, p_mm, pitch, speed, tmin):
        """Straight line of the tip in [x, y, z, pitch]: IK every WAYPOINT_MM, quintic in time."""
        q0 = np.asarray(q0, float)
        x0 = np.r_[self.arm.tip(q0) * 1000, self.arm.pitch(q0)]
        d = np.r_[np.asarray(p_mm, float), pitch] - x0
        d[3] = wrap(d[3])
        L = float(np.linalg.norm(d[:3]))
        if L < 0.2 and abs(d[3]) < math.radians(0.1):
            return None
        n = max(2, int(math.ceil(max(L / WAYPOINT_MM, abs(d[3]) / math.radians(1.0)))))
        Q = [q0.copy()]
        for i in range(1, n + 1):
            x = x0 + d * (i / n)
            q = self.solve(x[:3], x[3], Q[-1], chained=True)
            step = np.abs(q - Q[-1])
            if step[0] > math.pi:
                raise PlanError("the straight line crosses the joint1 +-180 deg seam behind the robot - refused")
            if float(np.max(step)) > MAX_STEP_RAD:
                raise PlanError(f"IK jumped {math.degrees(float(np.max(step))):.0f} deg between two waypoints "
                                f"{WAYPOINT_MM:.0f} mm apart (elbow flip / joint limit) - not a straight line")
            Q.append(q)
        T0 = max(tmin, 1.875 * L / speed, 1.875 * abs(d[3]) / ANG_SPEED)
        return retime_quintic(np.array(Q), T0, MAX_JOINT_VEL, MAX_JOINT_ACC, DT)

    # ---- motion ----
    def _tick(self, cam, aruco, lines, target_cam=None):
        """Keep the window live while waiting; q -> Quit, 's' typed in the terminal -> StopMotion."""
        if cam is not None and not self.headless:
            self.show(cam, aruco, lines, target_cam if target_cam is not None else self.view_target)
            k = cv2.waitKey(1) & 0xFF
        else:
            time.sleep(0.02)
            k = 255
        if k == ord("q"):
            raise Quit
        if self.stop_evt.is_set():
            self.stop_evt.clear()
            raise StopMotion

    def execute(self, tr, cam, aruco, label):
        """Send one trajectory and wait until the controller reports it done."""
        if tr is None:
            return True
        hint = "q = stop & quit" + ("   (terminal: s + Enter = stop, Ctrl+C = emergency stop)"
                                    if self.inbox is not None else "")
        lines = [f"moving: {label}  ({tr.T:.1f} s)", hint]
        print(f"  moving: {label}  ({tr.T:.1f} s)", flush=True)
        # ---- safety: never send a trajectory the controller would have to JUMP into ----
        q_now = self.current_q()
        jump = np.abs(tr.Q[0] - q_now)
        if float(np.max(jump)) > 0.05:
            print(f"  {label}: REFUSED - the plan starts {math.degrees(float(np.max(jump))):.1f} deg away from where "
                  f"joint{int(np.argmax(jump)) + 1} is now (arm moved since planning?)")
            return False
        if not self.multiturn:
            if abs(q_now[0]) > math.pi - math.radians(SEAM_MARGIN_DEG) / 2:
                print(f"  {label}: REFUSED - joint1 is at {math.degrees(q_now[0]):+.2f} deg, right on the +-180 deg "
                      f"seam where its reading flips sign. Turn the base a few degrees away from 180 (teleop, or by "
                      f"hand with torque off) and run again.")
                return False
            lo, hi = self.j1_lim()
            if tr.Q[:, 0].min() < lo - 1e-6 or tr.Q[:, 0].max() > hi + 1e-6:
                print(f"  {label}: REFUSED - joint1 would go within {SEAM_MARGIN_DEG} deg of +-180 deg")
                return False
        fut = self.node.send_traj(self.names, tr)
        t0, gh, rfut, ok = time.time(), None, None, True
        try:
            while True:
                self._tick(cam, aruco, lines)
                el = time.time() - t0
                if fut is not None:                                 # action interface
                    if gh is None:
                        if fut.done():
                            gh = fut.result()
                            if not gh.accepted:
                                print(f"  {label}: the arm controller rejected the trajectory")
                                ok = False
                                break
                            rfut = gh.get_result_async()
                        elif el > 5.0:
                            print(f"  {label}: the arm controller did not acknowledge the trajectory")
                            ok = False
                            break
                    elif rfut.done():
                        res = rfut.result()
                        if not (res.status == self.node.SUCCEEDED and res.result.error_code == 0):
                            print(f"  {label}: controller status={res.status} code={res.result.error_code} "
                                  f"{res.result.error_string}")
                            ok = False
                        break
                else:                                               # topic interface: watch the joints
                    q = self.current_q()
                    if el > tr.T + 0.3 and float(np.max(np.abs(q - tr.q_end))) < 0.02:
                        break
                if el > tr.T + 10.0:
                    print(f"  {label}: trajectory timed out")
                    ok = False
                    break
        except BaseException:                                       # Quit / StopMotion / Ctrl+C: hold here
            if gh is not None:
                try:
                    gh.cancel_goal_async()
                except Exception:
                    pass
            self.hold()
            raise
        if not ok:
            self.hold()
            return False
        # ---- the controller says "done" when the trajectory TIME is over - wait until the joints really are there ----
        t1, err = time.time(), None
        while time.time() - t1 < SETTLE_S:
            err = np.abs(self.current_q() - tr.q_end)
            if float(np.max(err)) < SETTLE_TOL_RAD:
                break
            self._tick(cam, aruco, [f"settling: {label}"])
        else:
            q_now = self.current_q()
            tip_err = float(np.linalg.norm(self.arm.tip(q_now) - self.arm.tip(tr.q_end))) * 1000
            print(f"  {label}: not at the end of the move after {SETTLE_S:.0f} s ({tip_err:.1f} mm away) - "
                  f"stopped there")
            self.hold()
            return False
        self.q_cmd = tr.q_end
        return True

    # ---- drawing ----
    def show(self, cam, aruco, lines, target_cam=None):
        """Live frame + GREEN tilted cross (target as the calibration sees it) + RED cross (where the calibration
        puts the robot tip now, from the joint angles). -> the drawn image"""
        img, _ = cam.grab()
        h, w = img.shape[:2]
        lines = list(lines)
        uv_t = None
        if target_cam is not None and target_cam[2] > 0:
            uv_t = cam.project(target_cam)[0]
            if not (0 <= uv_t[0] < w and 0 <= uv_t[1] < h):
                lines.append("(the GREEN target is outside the camera view)")
        vis = overlay(img, lines, aruco.detect(img))
        if uv_t is not None:
            cv2.drawMarker(vis, (int(uv_t[0]), int(uv_t[1])), (0, 255, 0), cv2.MARKER_TILTED_CROSS, 22, 2)
        tip = self.tip_now_mm()
        if tip is not None:                                             # robot tip seen through the calibration
            Pc = self.Ti[:3, :3] @ tip + self.Ti[:3, 3]
            if Pc[2] > 0:
                uv = cam.project(Pc)[0]
                cv2.drawMarker(vis, (int(uv[0]), int(uv[1])), (0, 0, 255), cv2.MARKER_CROSS, 24, 2)
        if not self.headless:
            cv2.imshow(WIN, vis)
        return vis

    def _estop(self):
        self.hold()
        print("\n\n  ###########  EMERGENCY STOP (Ctrl+C) - arm holding where it is  ###########\n")
        return False                                                  # do not move back to the start pose

    def close_ros(self):
        """Stop the spin thread BEFORE destroying the node (otherwise rclpy aborts: 'terminate called ...')."""
        try:
            self.exe.shutdown(timeout_sec=2.0)
        except Exception:
            pass
        if self.spin_th is not None:
            self.spin_th.join(timeout=2.0)
        try:
            self.node.destroy_node()
        except Exception:
            pass
        try:
            if self.rclpy.ok():
                self.rclpy.shutdown()
        except Exception:
            pass


# =============================================================================================================
#  Closed-form reach check of joints 2-4 - the same as final_run.py's PlanarIK
# =============================================================================================================
class PlanarIK:
    """Closed-form IK of the OpenMANIPULATOR-X's joints 2-4 (one vertical plane, joint1 faces the target). Used to
    REJECT unreachable targets in microseconds (the numerical IK needs ~0.5 s to give up); the joints themselves
    still come from the numerical IK. Checked against the URDF model at start (random poses); if that check fails it
    is switched off and everything goes to the numerical IK."""

    def __init__(self, arm):
        self.arm, self.ok = arm, False
        try:
            if len(arm.names) != 4 or any(np.any(np.abs(j["R0"] - np.eye(3)) > 1e-9) for j in arm.joints):
                return
            q0 = np.zeros(4)
            _, fr = arm.fk(q0, frames=True)
            tip = arm.tip(q0)
            P = [fr[1], fr[2], fr[3], tip]
            if any(abs(x[1] - fr[0][1]) > 1e-4 for x in P):
                return
            pl = [np.array([x[0] - fr[0][0], x[2]]) for x in P]         # (r, z) at q = 0 (arm along +x)
            v = [pl[k + 1] - pl[k] for k in range(3)]
            self.S = pl[0]
            self.L = [float(np.linalg.norm(x)) for x in v]
            self.phi0 = [math.atan2(x[1], x[0]) for x in v]
            self.sg = [float(np.sign(j["axis"][1])) for j in arm.joints if j["name"] in arm.names][1:]
            if any(x == 0 for x in self.sg):
                return
            self.lo, self.hi = arm.lo[1:].copy(), arm.hi[1:].copy()
            rng = np.random.default_rng(1)
            for _ in range(60):                                         # self-check against the URDF model
                q = np.r_[rng.uniform(-math.pi, math.pi), rng.uniform(self.lo, self.hi)]
                Pt, a = arm.tip(q) * 1000, arm.pitch(q)
                if not any(np.linalg.norm(arm.tip(qs) * 1000 - Pt) < 0.5 for qs in self.solutions(Pt, a)):
                    return
            self.ok = True
        except Exception:
            self.ok = False

    def solutions(self, P_mm, pitch, slack_mm=0.0, slack_rad=0.0):
        """Joint vectors (joint1 = facing the target) that put the tip on P_mm with this pitch."""
        p = np.asarray(P_mm, float) / 1000.0
        d = p[:2] - self.arm.j1_xy
        yaw, r = math.atan2(d[1], d[0]), float(np.hypot(d[0], d[1]))
        out = []
        for side, rr, aa in ((0.0, r, pitch), (math.pi, -r, pitch)):   # facing it, or reaching back over
            ph3 = -aa
            W = np.array([rr, p[2]]) - self.L[2] * np.array([math.cos(ph3), math.sin(ph3)])
            dv = W - self.S
            D = float(np.linalg.norm(dv))
            l1, l2 = self.L[0], self.L[1]
            cb = (D * D - l1 * l1 - l2 * l2) / (2 * l1 * l2)
            tol = slack_mm / 1000.0 * (D + 1e-9) / (l1 * l2)
            if cb > 1 + tol or cb < -1 - tol:
                continue
            cb = min(1.0, max(-1.0, cb))
            for b in (math.acos(cb), -math.acos(cb)):
                ph1 = math.atan2(dv[1], dv[0]) - math.atan2(l2 * math.sin(b), l1 + l2 * math.cos(b))
                ph2 = ph1 + b
                c1 = self.phi0[0] - ph1
                c2 = self.phi0[1] - ph2
                c3 = self.phi0[2] - ph3
                q = np.array([c1, c2 - c1, c3 - c2]) * np.array(self.sg)
                q = (q + math.pi) % (2 * math.pi) - math.pi
                if np.all(q >= self.lo - slack_rad) and np.all(q <= self.hi + slack_rad):
                    out.append(np.r_[wrap(yaw + side), q])
        return out

    def maybe(self, P_mm, pitch):
        """False only if the target is surely unreachable (2 mm / 0.5 deg slack for the numerical IK)."""
        return not self.ok or bool(self.solutions(P_mm, pitch, slack_mm=2.0, slack_rad=math.radians(0.5)))


@dataclass
class Move:
    name: str
    kind: str                      # lift | joint | line | pause
    target: object = None          # robot point (mm)
    pitch: float = 0.0             # rad
    speed: float = None            # mm/s (lines)
    dip: str = "error"             # joint moves: 'error' | 'warn' if the tip sags more than DIP_MM
    q: object = None               # joint vector (back to the start pose)


# =============================================================================================================
#  CalibRun = the calibration check, every move planned like final_run.py's pick
# =============================================================================================================
class CalibRun(Checker):
    def __init__(self, args):
        super().__init__(args)
        self.fastik = PlanarIK(self.arm)
        if not self.fastik.ok:
            print("note: closed-form reach check not usable with this URDF - planning uses the numerical IK only")
        lo, hi = self.pitch_range()
        tip = self.arm.tip(self.park) * 1000
        print(f"start pose (returned to with --return / at the end): tip x={tip[0]:.0f} y={tip[1]:.0f} "
              f"z={tip[2]:.0f} mm, gripper pitch {math.degrees(self.arm.pitch(self.park)):+.1f} deg; "
              f"pitch used during the check {math.degrees(lo):.0f}..{math.degrees(hi):.0f} deg")

    def solve(self, p_mm, pitch, q_ref, chained=False):
        """(final_run.py) the numerical IK, but targets the closed-form check rules out fail at once, and where the
        numerical IK gives up while the closed form has a solution, that one is taken."""
        fk = getattr(self, "fastik", None)
        if fk is not None and not fk.maybe(p_mm, pitch):
            raise PlanError(f"x={p_mm[0]:.1f} y={p_mm[1]:.1f} z={p_mm[2]:.1f} mm with pitch "
                            f"{math.degrees(pitch):.1f} deg is out of reach (arm length / joint limits)")
        try:
            return super().solve(p_mm, pitch, q_ref, chained)
        except PlanError as e:
            if fk is None or not fk.ok or ("out of reach" not in str(e) and "reachable only" not in str(e)):
                raise
            best = None
            for q in fk.solutions(p_mm, pitch):
                try:
                    q[0] = unwrap_j1(q[0], q_ref[0], self.window)
                except PlanError:
                    continue
                lo, hi = self.j1_lim()
                if lo <= q[0] <= hi:
                    dq = float(np.max(np.abs(q - np.asarray(q_ref, float))))
                    if best is None or dq < best[0]:
                        best = (dq, q)
            if best is None:
                raise
            return best[1]

    # ---------------------------- gripper orientation ----------------------------
    def pitch_range(self):
        """Allowed gripper pitch (rad): PITCH_MIN_DEG .. PITCH_MAX_DEG, or --pitch-range MIN MAX."""
        r = getattr(self.args, "pitch_range", None) or [PITCH_MIN_DEG, PITCH_MAX_DEG]
        return math.radians(min(r)), math.radians(max(r))

    def gripper_pitch(self, points_mm, q_ref, want=None):
        """ONE gripper pitch inside the allowed range with which EVERY point can be reached (target and the points
        above it, so the descent keeps the pitch). want / --pitch: only that one. Otherwise the current pitch if
        inside and it works, else the closest pitch in the range that works (0.5 deg steps). Like the teleop's 'p':
        0 = horizontal, + = tilted down."""
        lo, hi = self.pitch_range()
        if want is None and self.args.pitch is not None:
            want = math.radians(self.args.pitch)
        if want is not None:
            if not lo - 1e-6 <= want <= hi + 1e-6:
                raise PlanError(f"gripper pitch {math.degrees(want):.1f} deg is outside the allowed "
                                f"{math.degrees(lo):.0f}..{math.degrees(hi):.0f} deg (--pitch-range)")
            cands = [want]
        else:
            first = min(max(self.arm.pitch(q_ref), lo), hi)
            grid = np.linspace(lo, hi, int(round(math.degrees(hi - lo) / 0.5)) + 1)
            cands = [first] + sorted((float(a) for a in grid), key=lambda a: abs(a - first))
        last = None
        for a in cands:
            try:
                for p in points_mm:
                    self.solve(p, a, q_ref)
                return a
            except PlanError as e:
                if "seam" in str(e) or "no equivalent" in str(e):
                    raise
                last = e
        p = np.asarray(points_mm[-1], float)
        raise PlanError(f"x={p[0]:.1f} y={p[1]:.1f} z={p[2]:.1f} mm ({self.radius(p):.0f} mm from the base) and the "
                        f"points above it cannot be reached with the gripper pitch {math.degrees(lo):.0f}.."
                        f"{math.degrees(hi):.0f} deg ({last})")

    # ---------------------------- geometry (final_run.py) ----------------------------
    def ray_point(self, xy, r, z):
        """Point at distance r from the joint1 axis, on the vertical plane through xy (the arm faces xy)."""
        j1 = self.arm.j1_xy * 1000
        u = np.asarray(xy[:2], float) - j1
        n = float(np.linalg.norm(u))
        u = u / n if n > 1e-6 else np.array([-1.0, 0.0])
        return np.r_[j1 + r * u, z]

    def radius(self, P):
        return float(np.linalg.norm(np.asarray(P[:2], float) - self.arm.j1_xy * 1000))

    def check_base_angle(self, P, what):
        """PlanError with a concrete suggestion if joint1 cannot face P inside its allowed range (cables)."""
        lo, hi = self.window
        if hi - lo >= 2 * math.pi - 1e-6:
            return
        j1 = self.arm.j1_xy * 1000
        d = np.asarray(P[:2], float) - j1
        th = math.atan2(d[1], d[0])
        if any(lo - 1e-9 <= th + k * 2 * math.pi <= hi + 1e-9 for k in (-2, -1, 0, 1, 2)):
            return
        r = float(np.linalg.norm(d))
        edges = []
        for e, inward in ((lo, 1.0), (hi, -1.0)):
            diff = (th - e + math.pi) % (2 * math.pi) - math.pi
            edges.append((abs(diff), e + inward * math.radians(3.0)))
        gap, e = min(edges)
        sx, sy = j1 + r * np.array([math.cos(e), math.sin(e)])
        raise PlanError(f"{what} is at a base angle of {math.degrees(th):+.0f} deg, outside the range joint1 may turn "
                        f"[{math.degrees(lo):+.0f}, {math.degrees(hi):+.0f}] deg (kept on one side of 0 for the "
                        f"cables, as in final_run.py; --joint1-limit). Use a point {math.degrees(gap) + 3:.0f} deg "
                        f"further round the base, e.g. x={sx:+.0f} y={sy:+.0f} mm (same distance from the base)")

    # ---------------------------- planning (final_run.py) ----------------------------
    def plan_move(self, s, q0):
        """One motion step from joints q0 (preview AND, re-planned from the real joints, for running)."""
        sp = s.speed or self.args.speed
        if s.kind == "lift":                                  # straight up first ...
            tip = self.arm.tip(q0) * 1000
            if s.target[2] - tip[2] < 1.0:
                return None
            p = np.r_[tip[:2], s.target[2]]
            try:
                return self.plan_line(q0, p, s.pitch, sp, MIN_SEGMENT_T)
            except PlanError:
                try:
                    return self.plan_joint_to(q0, self.solve(p, s.pitch, q0))
                except PlanError as e:
                    if "base would turn" in str(e) or "seam" in str(e):
                        raise
                    return None                               # ... no lift from this pose: the joint move goes directly
        if s.kind == "line":
            return self.plan_line(q0, s.target, s.pitch, sp, MIN_SEGMENT_T)
        if s.kind == "joint":
            tr = self.plan_joint_to(q0, s.q if s.q is not None else self.solve(s.target, s.pitch, q0))
            self._dip(tr, s)
            return tr
        return None

    def _dip(self, tr, s):
        if tr is None:
            return
        z = np.array([self.arm.tip(q)[2] for q in list(tr.Q[::4]) + [tr.Q[-1]]]) * 1000
        floor = min(z[0], z[-1]) - DIP_MM
        if z.min() < floor:
            msg = f"'{s.name}': the joint move would sag {min(z[0], z[-1]) - z.min():.0f} mm below its lower end"
            if s.dip == "error":
                raise PlanError(msg)
            print(f"  note: {msg}")

    def preview(self, moves, q0):
        """Plan every motion step from q0 -> (trajs, q_end, seconds). Raises PlanError naming the step."""
        q, total, trajs = np.asarray(q0, float).copy(), 0.0, []
        for s in moves:
            tr = None
            if s.kind in ("lift", "line", "joint"):
                try:
                    tr = self.plan_move(s, q)
                except PlanError as e:
                    raise PlanError(f"step '{s.name}': {e}")
                if tr is not None:
                    q, total = tr.q_end, total + tr.T
            trajs.append(tr)
        return trajs, q, total

    def plan_to(self, P, want=None, name="the point", pause=False):
        """lift, ORIENT (joint move, facing P, final pitch, safe height), straight over P, straight down (last
        PRE_MM slow). Tried from the current reach-out first, then START_R_TRY. -> (moves, trajs, seconds, pitch,
        notes). Raises PlanError (nothing has moved)."""
        q0 = self.current_q()
        P = np.asarray(P, float)
        z_safe = P[2] + self.args.hover
        above = np.r_[P[:2], z_safe]
        pre = np.r_[P[:2], P[2] + PRE_MM] if self.args.hover > PRE_MM + 1.0 else None
        self.check_base_angle(P, name)
        a = self.gripper_pitch([above] + ([pre] if pre is not None else []) + [P], q0, want)
        tip0 = self.arm.tip(q0) * 1000
        r_list = []
        for r in (min(max(self.radius(tip0), START_R[0]), START_R[1]),) + START_R_TRY:
            if all(abs(r - x) > 15 for x in r_list):
                r_list.append(r)
        errs = []
        for r in r_list:
            O = self.ray_point(P, r, z_safe)
            moves = [Move("lift", "lift", O, a),
                     Move(f"orient: face {name}, gripper pitch {math.degrees(a):+.1f} deg", "joint", O, a, dip="warn"),
                     Move(f"over {name} (same orientation)", "line", above, a)]
            if pause:
                moves.append(Move(f"check: above {name}", "pause"))
            if pre is not None:
                moves.append(Move(f"down to {PRE_MM:.0f} mm above {name}", "line", pre, a))
            moves.append(Move(f"down to {name} (slow)", "line", P, a, self.args.slow))
            try:
                trajs, _, total = self.preview(moves, q0)
            except PlanError as e:
                if "base would turn" in str(e) or "no equivalent" in str(e) or "seam" in str(e):
                    raise                                     # no start position changes that
                errs.append((r, e))
                continue
            notes = []
            if r != r_list[0]:
                notes.append(f"start position: the current one could not be solved ({errs[0][1]}), using a "
                             f"reach-out of {r:.0f} mm instead")
            return moves, trajs, total, a, notes
        raise PlanError(f"no start position works for {name}: from the current position ({errs[0][1]})"
                        + (f"; also tried reach-out {', '.join(f'{r:.0f}' for r, _ in errs[1:])} mm"
                           if len(errs) > 1 else ""))

    def describe(self, moves, trajs, total, a, notes):
        for n in notes:
            print(f"  note: {n}")
        print(f"  plan (final_run motion), gripper pitch {math.degrees(a):+.1f} deg, ~{total:.1f} s of motion:")
        for k, (s, tr) in enumerate(zip(moves, trajs), 1):
            extra = f"{tr.T:5.2f} s" if tr is not None else ("" if s.kind == "pause" else "(skipped / there)")
            print(f"    {k}. {s.name:<46s} {extra}")

    # ---------------------------- running ----------------------------
    def run_moves(self, moves, cam, aruco, on_pause=None):
        """Run the steps; each one re-planned from where the arm really is just before it runs. -> True if all done
        (False: the arm holds where the failing step stopped). Quit / StopMotion / Ctrl+C pass through."""
        n = len(moves)
        for i, s in enumerate(moves, 1):
            print(f"  [{i}/{n}] {s.name}", flush=True)
            if s.kind == "pause":
                if on_pause is not None and not on_pause():
                    return False
                continue
            try:
                tr = self.plan_move(s, self.current_q())
            except PlanError as e:
                self.hold()
                print(f"      cannot plan '{s.name}' from here: {e}")
                return False
            if not self.execute(tr, cam, aruco, s.name):
                print(f"      '{s.name}' did not finish - arm holding")
                return False
        return True

    def go_point(self, P, want, cam, aruco, name, on_pause=None):
        """Plan (final_run motion) and run it. -> (ok, pitch). PlanError before anything moves."""
        moves, trajs, total, a, notes = self.plan_to(P, want, name, pause=on_pause is not None)
        self.describe(moves, trajs, total, a, notes)
        return self.run_moves(moves, cam, aruco, on_pause), a

    def short_move(self, P, want, cam, aruco, name):
        """Terminal mode: a target close to the tip (<= DIRECT_MM) is one straight line; else go_point."""
        q0 = self.current_q()
        if np.linalg.norm(np.asarray(P) - self.arm.tip(q0) * 1000) <= DIRECT_MM:
            try:
                a = self.gripper_pitch([np.asarray(P, float)], q0, want)
                tr = self.plan_line(q0, P, a, self.args.slow, MIN_SEGMENT_T)
                return self.execute(tr, cam, aruco, name)
            except PlanError:
                pass
        return self.go_point(P, want, cam, aruco, name)[0]

    def lift_up(self, dz, cam, aruco):
        q = self.current_q()
        tip = self.arm.tip(q) * 1000
        return self.run_moves([Move(f"up {dz:.0f} mm", "lift", np.r_[tip[:2], tip[2] + dz], self.arm.pitch(q))],
                              cam, aruco)

    def back_to_start(self, cam, aruco):
        q = self.current_q()
        tip = self.arm.tip(q) * 1000
        return self.run_moves([Move("lift 30 mm", "lift", np.r_[tip[:2], tip[2] + 30.0], self.arm.pitch(q)),
                               Move("back to the start pose", "joint", q=self.park.copy(), dip="warn")], cam, aruco)

    # ---------------------------- camera ----------------------------
    def open_view(self):
        """Camera + window. Any problem -> (None, None): the robot part still runs without the camera."""
        try:
            preflight()
            cam = open_camera(self.args.fps)
            aruco = Aruco(self.cal.get("dictionary", "DICT_4X4_50"))
        except SystemExit as e:
            print(f"camera not available ({e}) - continuing WITHOUT the camera")
            return None, None
        except Exception as e:
            print(f"camera not available ({type(e).__name__}: {e}) - continuing WITHOUT the camera")
            return None, None
        try:
            cv2.namedWindow(WIN)
        except Exception as e:                              # camera OK, but no window possible (X display)
            msg = str(e).strip().splitlines()[-1] if str(e).strip() else type(e).__name__
            self.headless = True
            print(f"no camera WINDOW ({msg}) - the picture at the point is SAVED to {self.args.snapshot} (no "
                  f"clicking). For the window: on the HOST run   xhost +local:   then start again")
        return cam, aruco

    def c2r(self, Pc):
        return self.T[:3, :3] @ np.asarray(Pc, float) + self.T[:3, 3]

    def r2c(self, Pb):
        return self.Ti[:3, :3] @ np.asarray(Pb, float) + self.Ti[:3, 3]

    def measure_click(self, cam, click, P, name):
        """x, y error of the calibration at the tip's height. The camera ray through the clicked pixel is cut with the
        horizontal plane z = P[2] (the tip's real height, from the joints): that is where camera + calibration put
        the tip. No depth reading at the thin tip (it mostly sees the table behind it)."""
        d_c = np.array([(click[0] - cam.ppx) / cam.fx, (click[1] - cam.ppy) / cam.fy, 1.0])
        o, d = self.T[:3, 3], self.T[:3, :3] @ d_c                  # camera centre and ray direction, robot frame
        if abs(d[2]) < 1e-6 or (P[2] - o[2]) / d[2] <= 0:
            print("  the clicked ray does not reach the tip's height - click again")
            return None
        Pb = o + d * (P[2] - o[2]) / d[2]
        e = np.asarray(P, float) - Pb
        e[2] = 0.0
        uv_exp = cam.project(self.r2c(P))[0]
        px = float(np.hypot(*(np.asarray(click) - uv_exp)))
        r = {"name": name, "robot_mm": np.round(P, 1).tolist(), "camera_says_mm": np.round(Pb, 1).tolist(),
             "error_mm": np.round(e, 2).tolist(), "xy_mm": round(float(np.linalg.norm(e[:2])), 2),
             "pixel_error": round(px, 1), "click_px": [float(click[0]), float(click[1])]}
        print(f"  {name}: robot tip at x={P[0]:.1f} y={P[1]:.1f} z={P[2]:.1f}; camera + calibration put it at "
              f"x={Pb[0]:.1f} y={Pb[1]:.1f} (same height)\n"
              f"    ERROR dx={e[0]:+.1f} dy={e[1]:+.1f} mm = {r['xy_mm']:.1f} mm  ({px:.0f} px from the GREEN cross)")
        return r

    def click_and_measure(self, cam, P, name, res):
        color, _ = cam.capture(10)
        uv_exp = cam.project(self.r2c(P))[0]
        click = click_point(color, "CLICK the point between the finger tips (what the robot holds), ENTER = ok, "
                                   "c = cancel", marks=[(uv_exp, (0, 255, 0))])
        if click is not None:
            tip = self.tip_now_mm()                     # where the robot really is (joints), not the command
            r = self.measure_click(cam, click, tip if tip is not None else np.asarray(P, float), name)
            if r is not None:
                res.append(r)

    def snapshot(self, vis):
        try:
            cv2.imwrite(self.args.snapshot, vis)
            print(f"  camera view saved to {self.args.snapshot}")
        except Exception as e:
            print(f"  (could not save {self.args.snapshot}: {e})")

    # ================= MODE 4: one point (--goto) =================
    def run_goto(self, P, pitch):
        P = np.asarray(P, float)
        Pc = self.r2c(P)
        go_home = bool(self.args.return_home)
        print(f"\ngoto: robot x={P[0]:.1f} y={P[1]:.1f} z={P[2]:.1f} mm ({self.base}), {self.radius(P):.0f} mm from "
              f"the base\n  (Ctrl+C in this terminal = emergency stop,  q in the camera window = stop)")
        cam = aruco = None
        if not self.args.no_camera:
            cam, aruco = self.open_view()
        self.view_target = Pc if Pc[2] > 0 else None
        res = []
        try:
            try:
                ok, _ = self.go_point(P, pitch, cam, aruco, "the point")
            except PlanError as e:
                print(f"  not moving: {e}")
                ok = False
            if ok:
                t = self.tip_now_mm()
                print(f"  reached: x={t[0]:.1f} y={t[1]:.1f} z={t[2]:.1f} mm ({np.linalg.norm(t - P):.1f} mm from the "
                      f"command, joints), gripper pitch {math.degrees(self.arm.pitch(self.current_q())):+.1f} deg")
            if cam is not None and ok:
                time.sleep(0.3)
                lines = [f"at x={P[0]:.0f} y={P[1]:.0f} z={P[2]:.0f}.  GREEN = calibration's point,  RED = robot tip",
                         "ENTER = freeze & click the tip to measure,  q / ESC = finish"]
                self.snapshot(self.show(cam, aruco, lines, self.view_target))
                if not self.headless:
                    print("  camera window: ENTER = freeze & click the point between the finger tips,  q / ESC = finish")
                    while True:
                        self.show(cam, aruco, lines, self.view_target)
                        k = cv2.waitKey(20) & 0xFF
                        if k in (ord("q"), 27):
                            break
                        if k in (10, 13):
                            self.click_and_measure(cam, P, "the point", res)
                            print("  ENTER = measure again,  q / ESC = finish")
            if ok and go_home and self.args.wait > 0 and (cam is None or self.headless):
                print(f"  staying {self.args.wait:.0f} s, then back to the start pose ...")
                time.sleep(self.args.wait)
        except Quit:
            self.hold()
            print("\nstopped (q) - arm holding.")
        except StopMotion:
            self.hold()
        except KeyboardInterrupt:
            go_home = self._estop()
        finally:
            self._finish(cam, aruco, (lambda: self.summary_points(res)) if res else (lambda: None), go_home)

    # ================= MODE 2: fixed points =================
    def run_points(self, points):
        """MODE 2 / --check: visit the points one after the other (final_run motion); at each, click the tip a few
        times (ENTER), n = next point. At the end: summary per point and the offset update of the calibration file."""
        cam, aruco = self.open_view()
        if cam is None or self.headless:
            raise SystemExit("the point check needs the camera window to click the tip (on the HOST: xhost +local:)")
        res, go_home = [], True
        print(f"\nCHECK at {len(points)} point(s). At each point: ENTER = freeze the image and click the point between "
              f"the finger tips (do it {CLICKS_PER_POINT}-3 times), n = next point, q = finish.")
        try:
            for k, P in enumerate(points, 1):
                P = np.asarray(P, float)
                name = f"point {k}"
                self.view_target = self.r2c(P)
                print(f"\n{name}/{len(points)}: robot x={P[0]:.1f} y={P[1]:.1f} z={P[2]:.1f} mm, "
                      f"{self.radius(P):.0f} mm from the base")
                key = self._ask(cam, aruco, [f"{name}/{len(points)}: x={P[0]:.0f} y={P[1]:.0f} z={P[2]:.0f}",
                                             "ENTER = go there,  n = skip this point,  q = finish"], "\nnq")
                if key == "q":
                    raise Quit
                if key == "n":
                    continue
                try:
                    ok, _ = self.go_point(P, None, cam, aruco, name)
                except PlanError as e:
                    print(f"  skipped: {e}")
                    continue
                if not ok:
                    continue
                time.sleep(0.3)
                while True:
                    n = sum(1 for r in res if r["name"] == name)
                    key = self._ask(cam, aruco, [f"{name}: GREEN = calibration's point,  RED = robot tip,  "
                                                 f"clicks here: {n}",
                                                 "ENTER = click the tip,  n = next point,  q = finish"], "\nnq")
                    if key == "q":
                        raise Quit
                    if key == "n":
                        if n < CLICKS_PER_POINT:
                            print(f"  {name}: {n} click(s) - {CLICKS_PER_POINT} or more make the result steadier")
                        break
                    self.click_and_measure(cam, P, name, res)
        except Quit:
            print("\nstopped.")
        except StopMotion:
            self.hold()
        except KeyboardInterrupt:
            go_home = self._estop()
        finally:
            self._finish(cam, aruco, lambda: self.summary_points(res), go_home)

    def _ask(self, cam, aruco, lines, keys):
        """Live window until one of keys ('\\n' = ENTER) is pressed."""
        while True:
            self.show(cam, aruco, lines, self.view_target)
            k = cv2.waitKey(20) & 0xFF
            ch = "\n" if k in (10, 13) else (chr(k) if k != 255 else "")
            if ch and ch in keys:
                return ch

    # ================= MODE 1: markers, full 3D by touch =================
    def check_target(self, cam, aruco, name, Pc):
        target = self.c2r(Pc)                                           # camera -> robot base (mm)
        up = np.array([0, 0, 1.0])
        self.view_target = np.asarray(Pc, float)
        print(f"\n{name}: camera ({Pc[0]:.1f}, {Pc[1]:.1f}, {Pc[2]:.1f}) mm -> robot "
              f"x={target[0]:.1f} y={target[1]:.1f} z={target[2]:.1f} mm ({self.base})")
        key = self._ask(cam, aruco, [f"{name}: robot x={target[0]:.0f} y={target[1]:.0f} z={target[2]:.0f} mm",
                                     "ENTER = go above it,  n = skip,  q = quit"], "\nnq")
        if key == "q":
            raise Quit
        if key == "n":
            return

        def hover_ok():
            k = self._ask(cam, aruco, [f"above {name}: is it roughly over the marker?",
                                       f"ENTER = go down to {SAFE_MM:.0f} mm above it,  n = skip,  q = quit"], "\nnq")
            if k == "q":
                raise Quit
            return k == "\n"
        try:
            ok, a = self.go_point(target + up * SAFE_MM, None, cam, aruco, name, on_pause=hover_ok)
        except PlanError as e:
            print(f"  skipped: {e}")
            return
        if not ok:
            self.lift_up(20.0, cam, aruco)
            return
        steps, si = [0.5, 1.0, 2.0, 5.0], 2
        cmd = target + up * SAFE_MM
        print("  bring the tip onto the marker centre: w/s a/d (x, y), r/f (up/down), [ ] step; ENTER = touching")
        while True:
            self.show(cam, aruco, [f"{name}: put the tip ON the marker centre, then ENTER",
                                   f"w/s x+-  a/d y+-  r/f up/down  step {steps[si]} mm [ ]   n skip  q quit",
                                   f"commanded: {cmd[0] - target[0]:+.1f} {cmd[1] - target[1]:+.1f} "
                                   f"{cmd[2] - target[2]:+.1f} mm from the calibrated target"], Pc)
            k = cv2.waitKey(20) & 0xFF
            ch = chr(k) if k != 255 else ""
            move = {"w": (0, 1), "s": (0, -1), "a": (1, 1), "d": (1, -1), "r": (2, 1), "f": (2, -1)}.get(ch)
            if move:
                new = cmd.copy()
                new[move[0]] += move[1] * steps[si]
                if np.linalg.norm(new - target) > MAX_NUDGE_MM:
                    print(f"  more than {MAX_NUDGE_MM:.0f} mm from the target - not moving further.")
                    continue
                try:
                    tr = self.plan_line(self.current_q(), new, a, self.args.slow, NUDGE_T)
                except PlanError as e:
                    print(f"  cannot nudge there: {e}")
                    continue
                if self.execute(tr, cam, aruco, "nudge"):
                    cmd = new
            elif ch == "[":
                si = max(0, si - 1)
            elif ch == "]":
                si = min(len(steps) - 1, si + 1)
            elif ch == "n":
                break
            elif ch == "q":
                raise Quit
            elif k in (10, 13):
                time.sleep(0.5)                                         # let the arm settle
                samples = []
                for _ in range(10):
                    samples.append(self.tip_now_mm())
                    time.sleep(0.03)
                Pm = np.mean(samples, axis=0)
                e = Pm - target
                r = {"name": name, "target_mm": target.round(2).tolist(), "touch_mm": Pm.round(2).tolist(),
                     "error_mm": e.round(2).tolist(), "dist_mm": round(float(np.linalg.norm(e)), 2),
                     "xy_mm": round(float(np.linalg.norm(e[:2])), 2), "camera_mm": np.round(Pc, 2).tolist()}
                self.results.append(r)
                print(f"  ERROR {name}: {r['dist_mm']:.1f} mm  (dx={e[0]:+.1f} dy={e[1]:+.1f} dz={e[2]:+.1f}; "
                      f"in the table plane {r['xy_mm']:.1f} mm)")
                break
        self.lift_up(self.args.hover, cam, aruco)

    def run(self):
        cam, aruco = self.open_view()
        if cam is None or self.headless:
            raise SystemExit("the marker mode needs the camera window (xhost +local: on the host) - or use --goto")
        go_home = True
        try:
            if self._ask(cam, aruco, ["Arm OUT OF VIEW of the board?  ENTER = capture the board,  q = quit"],
                         "\nq") == "q":
                return
            color, depth = cam.capture(15)
            targets = []
            for mid, c in sorted(aruco.detect(color), key=lambda m: m[0]):
                P, info = point_3d(cam, depth, c.mean(0), c)
                if P is not None:
                    targets.append((f"marker {mid}", P))
            print(f"{len(targets)} markers measured: {[t[0] for t in targets]}")
            for name, Pc in targets:
                self.check_target(cam, aruco, name, Pc)
            self.view_target = None
            while True:
                k = self._ask(cam, aruco, ["c = click an extra point on the board,  ENTER / q = finish"], "\nqc")
                if k != "c":
                    break
                p = click_point(color)
                if p is not None:
                    hit = _marker_at(aruco.detect(color), p, r=0)
                    poly = hit[1] if hit else np.array([[p[0] + 12 * np.cos(t), p[1] + 12 * np.sin(t)]
                                                        for t in np.linspace(0, 2 * np.pi, 24)])
                    P, info = point_3d(cam, depth, p, poly)
                    if P is not None:
                        self.check_target(cam, aruco, f"point {len(self.results) + 1}", P)
        except Quit:
            print("\nstopped.")
        except StopMotion:
            self.hold()
        except KeyboardInterrupt:
            go_home = self._estop()
        finally:
            self._finish(cam, aruco, self.summary, go_home)

    # ================= MODE 3: points typed in the terminal =================
    def run_terminal(self, first_points=()):
        cam = aruco = None
        if not self.args.no_camera:
            cam, aruco = self.open_view()
        use_win = cam is not None and not self.headless
        self.inbox = queue.Queue()
        for p in first_points:
            self.inbox.put(" ".join(f"{v:g}" for v in p))

        def reader():                                                # terminal input, without blocking the camera
            while True:
                try:
                    line = sys.stdin.readline()
                except Exception:
                    line = ""
                if not line:                                         # EOF (Ctrl+D)
                    self.inbox.put("q")
                    return
                line = line.strip()
                if line.lower() in STOP_WORDS:
                    self.stop_evt.set()                              # handled at once, even during a move
                elif line:
                    self.inbox.put(line)

        threading.Thread(target=reader, daemon=True).start()

        def prompt():
            print("\npoint (x y z [pitch]) > ", end="", flush=True)

        print(TERMINAL_HELP)
        tip = self.tip_now_mm()
        print(f"tip now: x={tip[0]:.1f} y={tip[1]:.1f} z={tip[2]:.1f} mm, pitch "
              f"{math.degrees(self.arm.pitch(self.current_q())):+.1f} deg")
        prompt()
        res, n, go_home = [], 0, True
        P, last_cmd, name = None, tip.copy(), ""
        try:
            while True:
                if use_win:
                    lines = ["type a point in the TERMINAL:  x y z [pitch]  (mm),  q = quit"]
                    if P is not None:
                        lines += [f"{name}: x={P[0]:.0f} y={P[1]:.0f} z={P[2]:.0f}   GREEN = calibration's point",
                                  "ENTER here = freeze & click the tip to measure"]
                    self.show(cam, aruco, lines, self.view_target)
                    key = cv2.waitKey(20) & 0xFF
                else:
                    time.sleep(0.02)
                    key = 255
                if key == ord("q"):
                    raise Quit
                self.stop_evt.clear()                                # 's' while standing still: nothing to stop
                if key in (10, 13) and P is not None and use_win:
                    self.click_and_measure(cam, P, name, res)
                    prompt()
                    continue
                try:
                    line = self.inbox.get_nowait()
                except queue.Empty:
                    continue
                cmd = line.lower().strip()
                if cmd in ("q", "quit", "exit"):
                    break
                try:
                    if cmd in ("h", "help", "?"):
                        print(TERMINAL_HELP)
                    elif cmd in ("tip", "where", "p"):
                        t = self.tip_now_mm()
                        print(f"  tip now: x={t[0]:.1f} y={t[1]:.1f} z={t[2]:.1f} mm, pitch "
                              f"{math.degrees(self.arm.pitch(self.current_q())):+.1f} deg")
                    elif cmd in ("home", "park", "start"):
                        if self.back_to_start(cam, aruco):
                            print("  at the start pose.")
                        P, self.view_target = None, None
                        last_cmd = self.tip_now_mm()
                    else:
                        relative = cmd.startswith("d")
                        nums = [float(v) for v in re.findall(r"[-+]?\d+(?:\.\d*)?|[-+]?\.\d+", cmd)]
                        if (relative and len(nums) != 3) or (not relative and len(nums) not in (3, 4)) or \
                                (cmd[0].isalpha() and not relative):
                            print(f"  did not understand '{line}' - type x y z [pitch], e.g.  -150 50 75")
                            prompt()
                            continue
                        target = last_cmd + np.array(nums) if relative else np.array(nums[:3])
                        pitch = math.radians(nums[3]) if (not relative and len(nums) == 4) else None
                        if target[2] < MIN_POINT_Z and not self.args.allow_low:
                            print(f"  z={target[2]:.1f} is below {MIN_POINT_Z:.0f} mm - too close to the table "
                                  f"(start with --allow-low if you are sure)")
                            prompt()
                            continue
                        n += 1
                        name = f"point {n}"
                        self.view_target = self.r2c(target)
                        print(f"  {name}: x={target[0]:.1f} y={target[1]:.1f} z={target[2]:.1f} mm, "
                              f"{self.radius(target):.0f} mm from the base")
                        try:
                            ok = self.short_move(target, pitch, cam, aruco, name)
                        except PlanError as e:
                            print(f"  not moving: {e}")
                            ok = False
                        if ok:
                            P, last_cmd = target, target.copy()
                            t = self.tip_now_mm()
                            print(f"  reached: x={t[0]:.1f} y={t[1]:.1f} z={t[2]:.1f} mm "
                                  f"({np.linalg.norm(t - target):.1f} mm from the command)")
                except StopMotion:
                    t = self.tip_now_mm()
                    print(f"\n  STOPPED - arm holding at x={t[0]:.1f} y={t[1]:.1f} z={t[2]:.1f} mm.")
                    P, last_cmd = None, t
                prompt()
        except Quit:
            print("\nstopped.")
        except KeyboardInterrupt:
            go_home = self._estop()
        finally:
            self._finish(cam, aruco, (lambda: self.summary_points(res)) if res else (lambda: None), go_home)

    # ---------------------------- end / summaries ----------------------------
    def _finish(self, cam, aruco, summary, go_home=True):
        """Back to the start pose (unless E-stop / not wanted), summary, close everything."""
        if go_home:
            try:
                print("back to the start pose ...")
                self.back_to_start(cam, aruco)
            except BaseException as e:
                self.hold()
                print("  (not returned to the start pose:", repr(e), ")")
        else:
            print("arm left where it is (torque ON, holding).")
        summary()
        if cam is not None:
            if not self.headless:
                try:
                    cv2.destroyAllWindows()
                except Exception:
                    pass
            try:
                cam.stop()
            except Exception:
                pass
        self.close_ros()

    # ---------------------------- offset update of the calibration file ----------------------------
    def offset_cmd(self, off):
        return (f"python3 calib_camera_node.py --solve {self.args.calib} --offset {off[0]:.1f} {off[1]:.1f} "
                f"{off[2]:.1f}")

    def confirm(self, question):
        """y/N in the terminal (also in the terminal mode, where a thread reads the lines)."""
        if self.args.yes:
            print(f"{question} y  (--yes)")
            return True
        if self.inbox is not None:
            while not self.inbox.empty():
                self.inbox.get_nowait()
            print(question, end=" ", flush=True)
            try:
                ans = self.inbox.get(timeout=120)
            except queue.Empty:
                ans = ""
        else:
            try:
                ans = input(question + " ")
            except EOFError:
                ans = ""
        return ans.strip().lower() in ("y", "yes")

    def apply_offset(self, new_off, what):
        """Write the new offset into the calibration file with calib_camera_node.py --solve (the tool that made the
        file - it recomputes BASE_T_CAMERA_mm with it). Backup first; checked afterwards; restored if anything fails."""
        path = self.args.calib
        old = np.asarray(self.cal.get("offset_mm", [0.0, 0.0, 0.0]), float)
        new_off = np.round(np.asarray(new_off, float), 1)
        print(f"\n  OFFSET UPDATE ({what}):  {path}  offset_mm  {np.round(old, 1).tolist()}  ->  {new_off.tolist()}")
        if self.args.no_update:
            print(f"  --no-update: file NOT changed. To apply it:\n     {self.offset_cmd(new_off)}")
            return False
        if not self.confirm(f"  write it to {path}? [y/N]"):
            print(f"  file NOT changed. To apply it later:\n     {self.offset_cmd(new_off)}")
            return False
        tool = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calib_camera_node.py")
        if not os.path.exists(tool):
            print(f"  {tool} not found - file NOT changed. Run in the folder with calib_camera_node.py:\n"
                  f"     {self.offset_cmd(new_off)}")
            return False
        backup = f"{path}.bak-{time.strftime('%Y%m%d-%H%M%S')}"
        shutil.copy2(path, backup)
        with open(path) as f:
            before = json.load(f)
        cmd = [sys.executable, tool, "--solve", path, "--offset"] + [f"{v:.1f}" for v in new_off]
        print("  running: " + " ".join(cmd[1:]), flush=True)
        problem, out = None, ""
        try:
            p = subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=180)
            out = (p.stdout or "") + (p.stderr or "")
            if p.returncode != 0:
                problem = f"calib_camera_node.py exited with code {p.returncode}"
        except Exception as e:
            problem = f"could not run calib_camera_node.py ({type(e).__name__}: {e})"
        after = None
        if problem is None:
            try:
                with open(path) as f:
                    after = json.load(f)
                got = np.asarray(after.get("offset_mm", [np.nan] * 3), float)
                if not np.allclose(got, new_off, atol=0.06):
                    problem = f"offset_mm in the file is {got.tolist()}, not {new_off.tolist()}"
            except Exception as e:
                problem = f"the file cannot be read back ({type(e).__name__}: {e})"
        if problem is not None:
            shutil.copy2(backup, path)
            tail = "\n".join("     | " + line for line in out.strip().splitlines()[-8:])
            print(f"  FAILED: {problem}. The file was put back as it was (backup {backup}).")
            if tail:
                print(tail)
            print(f"  Apply it by hand:\n     {self.offset_cmd(new_off)}")
            return False
        t0 = np.array(before["BASE_T_CAMERA_mm"], float)[:3, 3]
        t1 = np.array(after["BASE_T_CAMERA_mm"], float)[:3, 3]
        dt, want = t1 - t0, new_off - old
        print(f"  WRITTEN. offset_mm = {new_off.tolist()}; BASE_T_CAMERA_mm moved by "
              f"({dt[0]:+.1f}, {dt[1]:+.1f}, {dt[2]:+.1f}) mm (offset change {np.round(want, 1).tolist()}).")
        if np.linalg.norm(dt - want) > 2.0:
            print("  note: the camera position moved differently from the offset change - calib_camera_node.py "
                  "re-solved the calibration; run the check again to see the result.")
        print(f"  backup of the old file: {backup}")
        print("  final_run.py uses the new calibration the next time it starts. Run this check again to confirm.")
        self.cal = after
        return True

    def summary_points(self, res):
        """Click results: averaged per point (each point counts once), bias over the points, offset update."""
        if not res:
            print("\nno measurements.")
            return
        groups = {}
        for r in res:
            groups.setdefault(r["name"], []).append(r)
        rows = []                                                  # (name, robot, mean error, clicks, scatter)
        for name, rs in groups.items():
            E = np.array([r["error_mm"] for r in rs], float)
            m = E.mean(0)
            sc = float(np.max(np.linalg.norm((E - m)[:, :2], axis=1))) if len(rs) > 1 else 0.0
            rows.append((name, np.mean([r["robot_mm"] for r in rs], axis=0), m, len(rs), sc))
        M = np.array([row[2] for row in rows])
        d = np.linalg.norm(M[:, :2], axis=1)
        b = M.mean(0)
        rest = np.linalg.norm((M - b)[:, :2], axis=1)
        rms, rms_rest = float(np.sqrt((d ** 2).mean())), float(np.sqrt((rest ** 2).mean()))
        print("\n================ CALIBRATION CHECK (x, y at the tip's height) ================")
        for i, (name, robot, m, n, sc) in enumerate(rows):
            warn = "   <- clicks disagree, click more carefully" if sc > CLICK_SCATTER_WARN_MM else ""
            print(f"  {name:9s} robot x={robot[0]:+7.1f} y={robot[1]:+7.1f} z={robot[2]:5.1f}   error dx={m[0]:+5.1f} "
                  f"dy={m[1]:+5.1f} = {d[i]:4.1f} mm   ({n} click{'s' if n != 1 else ''}, spread {sc:.1f} mm){warn}")
        print(f"  {len(rows)} point(s): mean {d.mean():.1f} mm, RMS {rms:.1f} mm, max {d.max():.1f} mm")
        print(f"  average error (bias): dx={b[0]:+.1f} dy={b[1]:+.1f} mm   "
              f"(error = where the robot really is - where camera + calibration put it)")
        if len(rows) >= 2:
            print(f"  after removing that bias: RMS {rms_rest:.1f} mm, max {rest.max():.1f} mm")
        with open(self.args.out, "w") as f:
            json.dump({"calibration": self.args.calib, "mode": "click (x, y at the tip height)", "results": res,
                       "per_point_error_mm": [np.round(row[2], 2).tolist() for row in rows],
                       "bias_mm": np.round(b, 2).tolist(), "rms_mm": rms}, f, indent=2)
        print(f"  saved to {self.args.out}")
        old = np.asarray(self.cal.get("offset_mm", [0.0, 0.0, 0.0]), float)
        new_off = old + np.r_[b[:2], 0.0]                          # z is not measured by clicking: kept as it is
        if rms <= 3.0:
            print("  -> good: within ~3 mm, no offset change needed.")
        elif len(rows) < MIN_UPDATE_POINTS:
            print(f"  -> measured at {len(rows)} point(s); an automatic offset update needs {MIN_UPDATE_POINTS} spread-"
                  f"out points (--check). The x, y shift so far would be:\n     {self.offset_cmd(new_off)}")
        elif rms_rest > 0.6 * rms or rms_rest > 8.0:
            print("  -> the error changes from point to point - not a constant shift an offset can remove. Redo the "
                  "calibration with more points spread over the workspace (board at 2-3 places, one raised).")
        else:
            print(f"  -> a constant shift: dx={b[0]:+.1f} dy={b[1]:+.1f} mm. z of the offset stays "
                  f"{old[2]:.1f} (clicking does not measure z).")
            self.apply_offset(new_off, f"x, y from {len(rows)} points")
        print("==============================================================================")

    def summary(self):
        """Touch results (marker mode): full 3D bias, offset update including z."""
        if not self.results:
            print("\nno measurements.")
            return
        E = np.array([r["error_mm"] for r in self.results])
        d = np.linalg.norm(E, axis=1)
        b = E.mean(0)
        rest = np.linalg.norm(E - b, axis=1)
        rms, rms_rest = float(np.sqrt((d ** 2).mean())), float(np.sqrt((rest ** 2).mean()))
        print("\n==================== CALIBRATION CHECK (touch, 3D) ====================")
        for r in self.results:
            e = r["error_mm"]
            print(f"  {r['name']:12s} error {r['dist_mm']:5.1f} mm   (dx={e[0]:+5.1f} dy={e[1]:+5.1f} dz={e[2]:+5.1f})")
        print(f"  {len(d)} points: mean {d.mean():.1f} mm, RMS {rms:.1f} mm, max {d.max():.1f} mm")
        print(f"  average error (bias): dx={b[0]:+.1f} dy={b[1]:+.1f} dz={b[2]:+.1f} mm")
        if len(d) >= 2:
            print(f"  after removing that bias: RMS {rms_rest:.1f} mm, max {rest.max():.1f} mm")
        with open(self.args.out, "w") as f:
            json.dump({"calibration": self.args.calib, "mode": "touch (3D)", "results": self.results,
                       "bias_mm": b.round(2).tolist(), "rms_mm": rms, "max_mm": float(d.max())}, f, indent=2)
        print(f"  saved to {self.args.out}")
        old = np.asarray(self.cal.get("offset_mm", [0.0, 0.0, 0.0]), float)
        if rms <= 3.0:
            print("  -> good: the calibration is within ~3 mm, no offset change needed.")
        elif len(d) < MIN_UPDATE_POINTS:
            print(f"  -> {len(d)} point(s); an automatic offset update needs {MIN_UPDATE_POINTS}. The shift so far "
                  f"would be:\n     {self.offset_cmd(old + b)}")
        elif rms_rest > 0.6 * rms or rms_rest > 8.0:
            print("  -> the error is not a simple shift: redo the calibration with more points spread over the "
                  "workspace (board at 2-3 places, one raised).")
        else:
            self.apply_offset(old + b, f"x, y, z by touch at {len(d)} points")
        print("=======================================================================")

def click_point(color, text="CLICK the point (the arm will go 10 mm above it), ENTER = ok, c = cancel", marks=()):
    st = {}
    cv2.setMouseCallback(WIN, lambda e, x, y, *a: st.update(p=(x, y)) if e == cv2.EVENT_LBUTTONDOWN else None)
    try:
        while True:
            vis = overlay(color, [text])
            for (u, v), colr in marks:
                cv2.drawMarker(vis, (int(u), int(v)), colr, cv2.MARKER_TILTED_CROSS, 22, 2)
            if "p" in st:
                cv2.drawMarker(vis, st["p"], (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
            cv2.imshow(WIN, vis)
            k = cv2.waitKey(20) & 0xFF
            if k in (10, 13) and "p" in st:
                return np.array(st["p"], float)
            if k == ord("c"):
                return None
    finally:
        cv2.setMouseCallback(WIN, lambda *a: None)


def parse_points(text):
    """'200,0,40 180,60,30' or '200 0 40 180 60 30' -> [[x, y, z], ...] (numbers taken in groups of three)."""
    nums = [float(v) for v in re.findall(r"[-+]?\d+(?:\.\d*)?|[-+]?\.\d+", text)]
    if not nums or len(nums) % 3:
        raise SystemExit(f"--points needs x,y,z for every point (a multiple of 3 numbers), got {len(nums)}: {nums}")
    return [nums[i:i + 3] for i in range(0, len(nums), 3)]


def glue_negative_points(argv):
    """argparse reads '--points -190,-30,75' as a NEW option (it starts with '-'). Glue it to '--points=-190,...'."""
    out, i = [], 0
    while i < len(argv):
        if argv[i] == "--points" and i + 1 < len(argv) and re.match(r"^-[\d.]", argv[i + 1]):
            out.append("--points=" + argv[i + 1])
            i += 2
        else:
            out.append(argv[i])
            i += 1
    return out


def main():
    global MAX_BASE_TURN_DEG, JOINT1_NEGATIVE_ONLY, JOINT1_MIN_DEG, JOINT1_SOFT_LIMIT_DEG
    ap = argparse.ArgumentParser(description="Calibration check with final_run.py's motion",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("calib", help="calibration file from calib_camera_node.py (e.g. omx_calibration.json)")
    ap.add_argument("--goto", type=float, nargs="*", metavar="N",
                    help="MODE 4: --goto X Y Z [PITCH]  (mm, robot base frame; pitch in deg inside --pitch-range). "
                         "--goto alone = DEFAULT_POINTS[0]")
    ap.add_argument("--points", nargs="?", const="default", default=None,
                    help="MODE 2: fixed points --points \"x,y,z x,y,z\" (alone = DEFAULT_POINTS)")
    ap.add_argument("--check", action="store_true",
                    help="CHECK + OFFSET UPDATE: visit CHECK_POINTS (4 spread-out points; or --points), click the tip "
                         "at each, and at the end write the new x, y offset into the calibration file")
    ap.add_argument("--yes", action="store_true", help="write the offset update without asking y/N")
    ap.add_argument("--no-update", action="store_true", help="never change the calibration file, only report")
    ap.add_argument("--terminal", "-t", action="store_true", help="MODE 3: type points in the terminal")
    ap.add_argument("--return", dest="return_home", action="store_true",
                    help="with --goto: back to the start pose at the end")
    ap.add_argument("--wait", type=float, default=3.0, help="with --goto --return (no window): seconds at the point")
    ap.add_argument("--no-camera", action="store_true", help="with --goto / --terminal: robot only, no window")
    ap.add_argument("--snapshot", default="goto_view.png", help="camera picture saved at the point")
    ap.add_argument("--out", default="calib_check_results.json")
    ap.add_argument("--pitch", type=float, default=None,
                    help="gripper pitch in deg for every point (inside --pitch-range); default: current / closest")
    ap.add_argument("--pitch-range", type=float, nargs=2, default=[PITCH_MIN_DEG, PITCH_MAX_DEG],
                    metavar=("MIN", "MAX"),
                    help=f"allowed gripper pitch in deg (default {PITCH_MIN_DEG:.0f} {PITCH_MAX_DEG:.0f})")
    ap.add_argument("--hover", type=float, default=HOVER_MM, metavar="MM",
                    help="safe height above the target for the turn to face it (as final_run.py --hover)")
    ap.add_argument("--speed", type=float, default=CART_SPEED_MM, help="mm/s, lifts and straight moves")
    ap.add_argument("--slow", type=float, default=SLOW_SPEED_MM, help="mm/s, last descent and nudges")
    ap.add_argument("--allow-low", action="store_true", help=f"allow points below z={MIN_POINT_Z:.0f} mm")
    ap.add_argument("--fps", type=int, choices=[6, 15, 30], default=15)
    ap.add_argument("--ee-frame", default="end_effector_link", help="gripper frame (as in final_run.py)")
    ap.add_argument("--tip-offset", type=float, nargs=3, default=[0.0, 0.0, 0.0], metavar=("X", "Y", "Z"),
                    help="mm in the ee frame - the SAME as in the calibration and in final_run.py")
    ap.add_argument("--joint1-limit", type=float, default=JOINT1_SOFT_LIMIT_DEG, metavar="DEG",
                    help=f"multi-turn: joint1 stays on one side of 0 within this (final_run.py: "
                         f"{JOINT1_SOFT_LIMIT_DEG:.0f}, cables)")
    ap.add_argument("--max-base-turn", type=float, default=MAX_BASE_TURN_DEG, metavar="DEG")
    ap.add_argument("--keep-single-turn", action="store_true", help="do NOT switch joint1 to Operating Mode 4")
    ap.add_argument("--joint1-multiturn", action="store_true",
                    help="joint1 is multi-turn (only used if its Operating Mode cannot be read)")
    ap.add_argument("--joint1-mode4", action="store_true", help=argparse.SUPPRESS)     # old option, now the default
    args, unknown = ap.parse_known_args(glue_negative_points(sys.argv[1:]))
    if unknown:
        print(f"(ignored unknown arguments: {unknown})")
    MAX_BASE_TURN_DEG = float(args.max_base_turn)
    JOINT1_NEGATIVE_ONLY = True                       # final_run.py: joint1 on one side of 0 (cables)
    JOINT1_MIN_DEG = -abs(float(args.joint1_limit))
    JOINT1_SOFT_LIMIT_DEG = abs(float(args.joint1_limit))
    if args.no_camera and not args.terminal and args.goto is None:
        raise SystemExit("--no-camera only works together with --terminal or --goto")
    lo, hi = min(args.pitch_range), max(args.pitch_range)
    goto_pitch = None
    if args.goto is not None:                    # check the point BEFORE starting ROS / torque
        if len(args.goto) == 0:
            args.goto = list(DEFAULT_POINTS[0])
            print(f"--goto without numbers: default point {args.goto}")
        if len(args.goto) not in (3, 4):
            raise SystemExit(f"--goto needs X Y Z [PITCH], got {len(args.goto)} numbers: {args.goto}")
        if args.goto[2] < MIN_POINT_Z and not args.allow_low:
            raise SystemExit(f"z={args.goto[2]:.1f} is below {MIN_POINT_Z:.0f} mm - too close to the table "
                             f"(add --allow-low if you are sure)")
        if len(args.goto) == 4:
            goto_pitch = math.radians(args.goto[3])
    for what, v in (("--pitch", args.pitch), ("--goto pitch", None if goto_pitch is None else math.degrees(goto_pitch))):
        if v is not None and not lo - 1e-6 <= v <= hi + 1e-6:
            raise SystemExit(f"{what} {v:g} deg is outside the allowed gripper pitch {lo:g}..{hi:g} deg "
                             f"(--pitch-range MIN MAX to change it)")
    if args.check and (args.goto is not None or args.terminal):
        raise SystemExit("--check is its own mode: do not combine it with --goto / --terminal")
    pts = None
    if args.check:
        pts = CHECK_POINTS if args.points in (None, "default") else parse_points(args.points)
    elif args.goto is None and args.points is not None:
        pts = DEFAULT_POINTS if args.points == "default" else parse_points(args.points)
    if pts is not None:
        low = [p for p in pts if p[2] < MIN_POINT_Z]
        if low and not args.allow_low:
            raise SystemExit(f"points below z={MIN_POINT_Z:.0f} mm: {low} (add --allow-low if you are sure)")
        if not os.path.exists(args.calib):
            raise SystemExit(f"{args.calib} not found")
    c = CalibRun(args)
    if args.goto is not None:
        c.run_goto(args.goto[:3], goto_pitch)
    elif args.terminal:
        c.run_terminal([] if args.points in (None, "default") else parse_points(args.points))
    elif pts is not None:
        c.run_points(pts)
    else:
        c.run()


if __name__ == "__main__":
    main()
    # everything is saved and closed by now; skip the C++ destructors at interpreter exit (rclpy / librealsense
    # can abort there with 'terminate called without an active exception'). The controller keeps holding the arm.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
