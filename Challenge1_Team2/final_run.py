#!/usr/bin/env python3
"""
final_run.py  -  OpenMANIPULATOR-X peg sorting, FINAL RUN: pick a peg from ANYWHERE the camera sees (MODE 4)
==============================================================================================================

  1. HOLES  The camera finds the tray and its four holes (sorting_detector.py v7.3, metric tray frame from depth);
            omx_calibration.json puts them in the robot frame. Where the camera is more than --hole-warn off the
            taught "peg in hole" pose of peg_sort.py, the taught x, y is used for that hole (--holes auto).
  2. PEG    Place ONE peg anywhere in view (not in a hole). The camera gives its colour + shape (= which hole), its
            top centre and its height: START point. The hole is the END point.
  3. RUN    1 open  2 lift  3 ORIENT: at a safe height turn to face the peg with the taught pick pitch
            4 straight out to above the peg in that SAME orientation (base angle and pitch fixed)
            5 down to the grasp height (--grasp-z 70 + --pick-raise 20 = z 90 mm, kept 10 mm above the peg bottom /
              10 mm below its top)
            6 LOCK the gripper with a position command (no check whether the peg was caught)
            7 lift, turn to face the hole with the TAUGHT place pitch, over it, down until the peg is as deep in the
              hole as when taught, but --drop-raise (10 mm) higher, release (position command), straight up until
              clear of every peg (RETREAT_CLEAR_MM), home, back to the parking pose.
            If IK or a move cannot be solved from the current position, other start positions (reach-out 130-260 mm)
            are tried before giving up. A move that fails is re-planned and tried once more before asking.
  4. NEXT   The placed peg's shape is done: that peg (still visible at its END goal) is ignored from now on and
            only kept as an obstacle the arm moves over. The run ends when every hole is done.
  AUTO      Every planned peg runs at once (AUTO_RUN; --confirm asks with ENTER first). It still asks when the peg
            must first be turned by hand. Speeds: SPEED_FACTOR / SLOW_FACTOR at the top, or --speed / --slow.

PICK ZONE  measured at start (closed-form IK) and drawn light-blue in the camera window: for a peg standing on the
  table about 155-355 mm from the base (hexagon / square from ~100 mm), joint1 inside its range. A peg too close for
  z=90 is grasped higher (up to 10 mm under its top), where the arm reaches closer in.

GRIPPER  Direction set by GRIPPER_REVERSED at the top (False = normal, plain commands). If True (a gripper whose
  reading is reversed, bigger gripper_left_joint = more closed): gripper.py (the
  new one, next to this file) mirrors every position; at start the gripper closes onto its closed stop (sets the
  mirror) and then opens fully. Either way: nothing may be in the gripper at start. --grip-normal / --grip-reversed
  override GRIPPER_REVERSED for one run.
  MODE: at every start the script sets the gripper servo to Operating Mode 5 (current-based position, force capped
  by GRIP_CURRENT). If that fails, or with GRIP_MODE5 = False / --grip-mode 3, it uses position mode 3.

GRIPPER ORIENTATION PER SHAPE - taught in peg_sort.py (SAMPLES, averaged like peg_sort does):
  pick pitch  = average pitch of the shape's work-station samples   (0 = horizontal, + = tilted down)
  place pitch = average pitch of its "peg in hole" samples
  Both may be changed by at most --pitch-search (5 deg) where the taught pitch cannot reach. Yaw follows x, y on
  this 4-DOF arm (roll is always 0). Release depth = taught insert height + (grasp height above the peg bottom now
  - when taught; --taught-surface-z). The arm cannot turn the peg about the vertical: the camera compares the peg's
  rotation with its hole's and says how far to turn it by hand (the base rotation from pick to hole is included).

MOTION = calib_check.py MODE 4: straight lines at a fixed pitch, quintic joint moves, every move must really arrive,
  base turns over --max-base-turn refused. joint1 in Operating Mode 4 (multi-turn) for the session, kept on ONE side
  of 0 within --joint1-limit (210 deg, cables). Reach at z ~70 mm with a near-horizontal gripper: about 170-370 mm
  from the base.

START (ROS 2 sourced; the peg detector, teleop and calibration nodes must NOT run - they hold camera / arm):
  T1: ros2 launch open_manipulator_x_bringup hardware.launch.py
  T2: python3 final_run.py omx_calibration.json
  Before starting, jog the arm to a PARKING POSE where it covers neither the tray nor the pegs in the
  camera image. The torque is switched ON at start (the arm first holds its current pose) and the arm returns to
  that pose after every peg.

  python3 final_run.py omx_calibration.json --dry-run      detect + plan only: no torque, nothing moves
  python3 final_run.py omx_calibration.json --step         stops above the peg and above the hole (ENTER)
  python3 final_run.py omx_calibration.json --shape square --once
  python3 final_run.py omx_calibration.json --holes recorded       hole x, y from peg_sort.py, not the camera
  python3 final_run.py omx_calibration.json --exclude 35           drop a bad recording (like peg_sort)
  python3 final_run.py omx_calibration.json --confirm              ENTER before every peg
  python3 final_run.py omx_calibration.json --speed 80 --slow 20   faster lines / descents (mm/s)

KEYS (camera window)   ENTER = go on    r = detect again    q = quit    SPACE or q while moving = STOP (arm holds)
                       after a stop / failure:  r = resume (re-planned from where the arm is),
                                                o = open the gripper,  x = abandon this peg (arm holds)
  Ctrl+C (terminal) = EMERGENCY STOP: the arm holds where it is and the program exits without moving.
  Without a display the questions are asked in the terminal (ENTER / letter + ENTER).
"""
# =========================================================================== #
#  CONTENTS                                                                     #
#    Settings ........... constants at the top (HOLE_OFFSET_MM, SPEED, GRIP_*,  #
#                         PICK_RAISE_MM, JOINT1_SIDE, COLOR_SHAPE ...)          #
#    gripper_module ..... loads gripper.py from next to this file              #
#    Taught references .. load_samples, tip_samples, average_pose, ShapeRef,    #
#                         shape_refs, print_refs                                #
#    UI ................. window / terminal questions                           #
#    Vision ............. tray, holes, pegs in the robot base frame             #
#    Plan ............... Step, PegPlan, seg_dist, PlanarIK                     #
#    PegSortRun ......... start-up, gripper, servos, drawing, the 4 phases      #
#                         (holes, peg, plan, run) and the main loop             #
#    grip_test .......... gripper bench test (--grip-test)                      #
#    parse_args, main ... command line and entry point                          #
# =========================================================================== #
import argparse
import ast
import contextlib
import io
import itertools
import json
import math
import os
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import calib_check as cc          # noqa: E402  MODE 4 motion: URDF arm model, IK, quintic moves, joint1 mode 4
import sorting_detector as sd     # noqa: E402  tray, holes, pegs, metric tray frame (v7.3)

SHAPES = ("circle", "hexagon", "square", "triangle")
HOME = [math.pi, -1.0, 0.7, 0.3]   # peg_sort.py HOME: facing the work area, gripper horizontal, ~200 mm up
WIN = "Peg sort - final run"

# Peg rotation that sits FLAT in the parallel jaws (they close across the arm's direction, i.e. tangentially):
# rotation of the fitted outline (sorting_detector._yaw_of convention) = joint1 + offset, modulo period.
JAW = {"square": (90.0, 45.0), "hexagon": (60.0, 0.0), "triangle": (60.0, 30.0)}

STILL_N, STILL_MM = 6, 2.5       # peg must stay within STILL_MM over STILL_N frames before it is planned
YAW_WAIT_S = 4.0                 # wait this long for a rotation measurement (not needed for the circle)
GRASP_ABOVE_BOTTOM_MM = 10.0     # grasp point at least this far above the peg bottom (what it stands on) ...
GRASP_BELOW_TOP_MM = 10.0        # ... and this far below its top
PEG_LEN_OK = (85.0, 115.0)       # camera peg length outside this is not trusted (--peg-length is used instead)
TRAY_Z_FALLBACK = 65.0           # tray surface z if the camera found no hole
START_R = (130.0, 260.0)         # orient pose: reach-out (mm from the joint1 axis) of the current position, limited
START_R_TRY = (180.0, 220.0, 150.0, 250.0, 130.0)   # other start positions tried if IK / a move fails
# ---------------------------------------------------------------------------------------------------------- #
#  END-GOAL OFFSETS PER HOLE  (mm, robot base frame: x, y)  <-- EDIT THESE                                     #
#  Added to the hole position the peg is put into (camera or taught x, y). -x = towards the robot base side     #
#  that is more negative in x. --hole-dx / --hole-dy on the command line add to all four on top of this.        #
# ---------------------------------------------------------------------------------------------------------- #
HOLE_OFFSET_MM = {
    "circle":   (-20.0, -10.0),
    "hexagon":  (-23.0, -13.0),
    "square":   (-13.0, -11.0),
    "triangle": (-16.0, -12.0),
}
# GRIPPER DIRECTION  <-- the ONE place to change it
#   False = normal: plain open / close commands (open = bigger gripper_left_joint).
#   True  = reversed: every gripper command is mirrored (only for a gripper whose reading runs the other way).
#   Command line: --grip-normal / --grip-reversed override this for one run.
GRIPPER_REVERSED = True

# SPEED  <-- how fast the run goes
#   AUTO_RUN: True = no ENTER before each peg: a peg is picked as soon as it is found and planned (the plan is still
#     printed). It still asks when the peg must first be turned by hand or its rotation is unknown, and with --step.
#     --confirm asks before every peg for one run.
#   SPEED_FACTOR: straight-line speed (lifts, moves over the peg / hole) = calib_check CART_SPEED_MM x this.
#   SLOW_FACTOR: slow last descent onto the peg and into the hole = calib_check SLOW_SPEED_MM x this.
#     --speed / --slow (mm/s) set them directly for one run.
#   DIRECT_PARK: False = after the release, back to the parking pose via HOME (safe when the parking pose is low:
#     a direct joint move to a low parking pose sweeps low across the table and can hit pegs). True = one direct
#     joint move (only for a HIGH parking pose); --direct-park / --via-home choose for one run.
#   VERIFY_AFTER_PLACE: camera check after each peg (~3 s). False = skipped; --verify turns it on for one run.
#   RETRY_MOVES: a move that fails (not stopped by you, not the slow descent onto the peg / into the hole) is
#     re-planned from where the arm is and tried once more before the run asks you.
AUTO_RUN = True
SPEED_FACTOR = 1.5
SLOW_FACTOR = 1.5
DIRECT_PARK = False
VERIFY_AFTER_PLACE = False
RETRY_MOVES = True
# CAMERA  <-- RealSense start-up (works on USB 2)
#   If the stream does not start ("Frame didn't arrive"), the camera is reset (like unplugging it) and opened again;
#   if it still gets no frames, it is opened at the next rate in CAMERA_FALLBACK_FPS (less data per second - a USB 2
#   port cannot carry colour + depth at high frame rates). CAMERA_USB2_FPS: rate used from the start when the camera
#   is on USB 2 and --fps was not given (None = use --fps).
CAMERA_FALLBACK_FPS = (6,)
CAMERA_USB2_FPS = None
# AFTER THE RELEASE  <-- how high the empty gripper goes before it moves away
#   Straight up until it is RETREAT_CLEAR_MM above the top of the peg it just placed AND above the top of every
#   other peg the camera saw (never lower than the carry height; as high as the arm reaches there). Then it moves
#   away. Command line: --retreat-clear MM
RETREAT_CLEAR_MM = 40.0
# PLACED PEGS: once a peg has been placed, its shape is done and every peg seen within PLACED_NEAR_MM of a placed END
#   goal is ignored (it stays visible there, and is only kept as an obstacle the arm moves over).
PLACED_NEAR_MM = 40.0

# GRIP  <-- grip strength
#   GRIPPER MODE: at every start the script itself sets the gripper servo to Operating Mode 5 (current-based
#   position) with Goal Current GRIP_CURRENT - the grip force can never be more than that. To lock, the fingers close
#   until the peg stops them, then the command is held GRIP_PRESS_MM past that point.
#   If mode 5 cannot be set, the script switches the gripper to position mode 3 instead: the fingers then close in
#   1.5 mm steps until the peg stops them (the servo never gets a big position error, so never full power).
#   GRIP_MODE5 = False: always position mode 3. For one run: --grip-mode 3 / --grip-mode 5, --grip-current N.
#   If the gripper locks up in mode 5 (LED blinking red): power-cycle the arm and set GRIP_MODE5 = False.
#   At start the fingers go straight to the open position - they are never driven into the open stop.
#   GRIP_PRESS_MM: more = tighter. 2.0 is safe; max 2.5 (more trips the overload and the gripper locks up until
#   power-cycled). Command line: --grip-squeeze 0.0020 (in m)
GRIP_PRESS_MM = 2.0
#   Calm gripper: written to the gripper servo at every start, in mode 5 and mode 3 (they reset when the 12 V is
#   switched off).
#   Position P Gain 400 (servo default 800): half the force for the same press, softer moves.
#   Profile Velocity 60 (x0.229 rpm) / Acceleration 10: the fingers move at ~17 mm/s and start/stop smoothly
#   instead of snapping. 0 = leave the servo's own value.
GRIP_P_GAIN = 400
GRIP_PROFILE_VEL = 60
GRIP_PROFILE_ACC = 10
#   GRIP_MODE5: True = set the gripper to Operating Mode 5 at every start (see GRIPPER MODE above).
#   GRIP_CURRENT: mode 5 force limit in 2.69 mA units (150 = ~0.40 A; more = tighter). Capped at the servo's
#   Current Limit.
GRIP_MODE5 = True
GRIP_CURRENT = 150
GRIP_SQUEEZE_MM = GRIP_PRESS_MM

# GRASP HEIGHT  <-- how much higher than --grasp-z (70) the fingers close on the peg, kept at least 10 mm below
#   the peg top and 10 mm above its bottom. The peg still goes into the hole as deep as before (the release point
#   rises by the same amount). Command line: --pick-raise N
PICK_RAISE_MM = 60.0
GRIPPER_DXL_ID = 15

# JOINT1 SIDE  <-- which way the arm's reach faces. joint1 stays on ONE side of 0 (cables, as in peg_sort.py):
#   "negative" = [-210, 0] deg: the reach covers the camera / -y side of the table (as in peg_sort.py)
#   "positive" = [0, +210] deg: the +y side
#   "start"    = whichever side joint1 happens to read at start (it parks at ~180 deg, so this can be
#                either side)
#   With "negative" and joint1 reading e.g. +179.6 at start, the same pose is counted as -180.4: nothing moves.
JOINT1_SIDE = "negative"

# COLOUR -> SHAPE: a peg of this colour is ALWAYS this shape (no shape voting). Colours not listed here still
# get their shape from the camera.  <-- EDIT if your peg set changes
COLOR_SHAPE = {
    "orange": "hexagon",
    "green":  "circle",
    "blue":   "square",
    "yellow": "triangle",
}
REACH_LOW_MM = (150, 360)        # pick zone if the closed-form check is not usable (measured at start otherwise)
TOP_CLEAR_MM = 25.0              # gripper (ee point) at least this far above the peg top when moving over it
PRE_GRASP_MM = 20.0              # slow descent onto the peg starts this far above its top
BOTTOM_CLEAR_MM = 15.0           # carried peg's bottom stays this far above the tray / other pegs
PATH_NEAR_MM = 50.0              # pegs this close to the carry path count as obstacles
DIP_MM = 15.0                    # a joint move may not sag more than this below its lower end point
GRIP_RANGE_M = (-0.012, 0.020)   # gripper_left_joint readings outside this are impossible (SRDF -0.010 .. 0.019)
DXL_NAMES = {11: "joint1", 12: "joint2", 13: "joint3", 14: "joint4", 15: "gripper"}
DXL_HW_BITS = {1: "input voltage", 4: "overheating", 8: "motor encoder", 16: "electrical shock", 32: "OVERLOAD"}
FIX_SERVOS = ("Fix: Ctrl+C hardware.launch.py, switch the arm's 12 V power OFF and ON, start hardware.launch.py "
              "again (it reboots the servos), check with the teleop that the arm moves, then start this again.")


# --------------------------------------------------------------------------- #
#  Gripper driver: gripper.py next to this file (Gripper class, RAW_RANGE)      #
# --------------------------------------------------------------------------- #
def gripper_module():
    """The gripper.py module next to this file (imported on first use, so its ROS imports happen only then).

    Exits with a clear message if gripper.py is missing or is an older version without the reversed-gripper
    support this run needs (reverse / mirror settings, raw commands, RAW_RANGE)."""
    try:
        import gripper
    except ImportError as e:
        raise SystemExit(f"gripper.py not found next to final_run.py ({e})")
    needed = ("RAW_RANGE", "DEFAULTS", "Gripper")
    if (any(not hasattr(gripper, k) for k in needed) or "reverse" not in gripper.DEFAULTS
            or "mirror" not in gripper.DEFAULTS
            or any(not hasattr(gripper.Gripper, m) for m in ("_goal_raw", "to_raw", "from_raw", "_wait_still"))):
        raise SystemExit(f"{gripper.__file__} is an OLD gripper.py (no reversed-gripper support): replace it with "
                         f"the gripper.py that belongs to this final_run.py")
    return gripper


def gripper_mode_wanted(args):
    """Gripper Operating Mode the set-up will use: 5 (--grip-mode 5, the default when GRIP_MODE5 = True) or 3
    (--grip-mode 3, or --grip-current 0)."""
    mode = getattr(args, "grip_mode", 5 if GRIP_MODE5 else 3)
    return 5 if mode == 5 and int(getattr(args, "grip_current", GRIP_CURRENT)) > 0 else 3


WINDOW_TEST = """
import cv2, numpy as np
cv2.namedWindow("window test")
cv2.imshow("window test", np.zeros((60, 200, 3), np.uint8))
for _ in range(3):
    cv2.waitKey(50)
cv2.destroyAllWindows()
cv2.waitKey(1)
"""


def window_works(timeout=15.0, code=WINDOW_TEST):
    """Open and draw a small OpenCV window in a separate process. Inside Docker the X display can crash the whole
    program on the first picture (Gdk 'BadAccess ... MIT-SHM'), which Python cannot catch - here only the test
    process dies. -> (True, '') or (False, short reason)."""
    import subprocess
    if not os.environ.get("DISPLAY"):
        return False, "no DISPLAY set"
    try:
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, f"the window test did not finish within {timeout:.0f} s"
    if r.returncode == 0:
        return True, ""
    err = (r.stderr or "").strip().splitlines()
    key = next((ln.strip() for ln in err if "BadAccess" in ln or "cannot open display" in ln.lower()
                or "Error" in ln), err[-1].strip() if err else f"exit code {r.returncode}")
    return False, key[:120]


def camera_usb(rs):
    """USB type of the first RealSense camera, e.g. '3.2' or '2.1' ('' if not readable)."""
    try:
        devs = rs.context().query_devices()
        return devs[0].get_info(rs.camera_info.usb_type_descriptor) if len(devs) else ""
    except Exception:
        return ""


def reset_camera(rs, wait=15.0):
    """Hardware reset of the first RealSense camera (as if it were unplugged), then wait until it is back.
    -> True if it came back within wait seconds."""
    try:
        devs = rs.context().query_devices()
        if not len(devs):
            return False
        serial = devs[0].get_info(rs.camera_info.serial_number)
        devs[0].hardware_reset()
    except Exception as e:
        print(f"camera: reset failed ({e})")
        return False
    time.sleep(2.0)
    t0 = time.time()
    while time.time() - t0 < wait:
        try:
            if any(d.get_info(rs.camera_info.serial_number) == serial for d in rs.context().query_devices()):
                time.sleep(1.5)                         # let it finish starting up
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def wrap_deg(a, per=360.0):
    """Wrap angle a [deg] into [-per/2, per/2). per = symmetry period (360 by default, e.g. 90 for a square)."""
    return float((a + per / 2.0) % per - per / 2.0)


# =========================================================================== #
#  Per-shape gripper orientation and heights from peg_sort.py SAMPLES          #
# =========================================================================== #
def load_samples(path):
    """SAMPLES table of peg_sort.py, read as data (peg_sort.py itself is not imported: it needs MoveIt)."""
    try:
        with open(path) as f:
            tree = ast.parse(f.read(), path)
    except OSError as e:
        raise SystemExit(f"cannot read the recorded samples: {e} (--samples PATH_TO/peg_sort.py)")
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "SAMPLES" for t in node.targets):
            return ast.literal_eval(node.value)
    raise SystemExit(f"no SAMPLES table in {path}")


def tip_samples(samples, tip_offset_mm):
    """The SAMPLES poses are end_effector_link poses (tele_op.py). With --tip-offset the planner moves the TIP
    (end_effector_link + offset in its own frame), so move every taught position to the tip as well - otherwise
    each taught hole / grasp would be off by the offset (10 mm towards the base for --tip-offset 10 0 0)."""
    off = np.asarray(tip_offset_mm, float) / 1000.0
    if not np.any(off):
        return samples
    out = {}
    for shape, kinds in samples.items():
        out[shape] = {}
        for kind, d in kinds.items():
            out[shape][kind] = {}
            for k, v in d.items():
                r, pch, y = np.radians(v[3:6])
                Rz = np.array([[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]])
                Ry = np.array([[math.cos(pch), 0, math.sin(pch)], [0, 1, 0], [-math.sin(pch), 0, math.cos(pch)]])
                Rx = np.array([[1, 0, 0], [0, math.cos(r), -math.sin(r)], [0, math.sin(r), math.cos(r)]])
                p = np.asarray(v[:3], float) + Rz @ Ry @ Rx @ off
                out[shape][kind][k] = [float(p[0]), float(p[1]), float(p[2])] + list(v[3:6])
    return out


def parse_ids(text):
    """--exclude text such as '35', '35,22' or '35 22' -> set of int sample IDs (empty set for None / '')."""
    if text is None or str(text).strip() == "":
        return set()
    return {int(t) for t in str(text).replace(";", ",").replace(" ", ",").split(",") if t.strip()}


def average_pose(samples):
    """Same as peg_sort.average_pose: x, y, z, pitch arithmetic mean, yaw circular mean."""
    A = np.array(list(samples.values()), float)
    xyz = A[:, :3].mean(axis=0)
    yr = np.radians(A[:, 5])
    yaw = math.degrees(math.atan2(float(np.sin(yr).mean()), float(np.cos(yr).mean())))
    return [float(xyz[0]), float(xyz[1]), float(xyz[2]), 0.0, float(A[:, 4].mean()), yaw]


@dataclass
class ShapeRef:
    """Taught reference of one shape, from peg_sort.py SAMPLES.

    ws / hole: averaged work-station (pick) and peg-in-hole (place) poses [x, y, z (m), roll, pitch, yaw (deg)],
    with the sample IDs used and their spread. The properties give the values in rad / mm.
    """
    shape: str
    ws: list                      # averaged work-station pose  [x, y, z (m), roll, pitch, yaw (deg)]
    hole: list                    # averaged peg-in-hole pose
    ws_ids: list
    hole_ids: list
    ws_spread: float              # mm, worst sample distance from the average
    hole_spread: float
    hole_worst: int = None        # ID of that worst hole sample

    @property
    def pick_pitch(self):
        """Taught pick pitch [rad] (+ = gripper tilted down)."""
        return math.radians(self.ws[4])

    @property
    def place_pitch(self):
        """Taught place pitch [rad]."""
        return math.radians(self.hole[4])

    @property
    def pick_z(self):
        """Taught grasp height [mm, robot base frame]."""
        return self.ws[2] * 1000.0

    @property
    def place_z(self):
        """Taught insertion height [mm, robot base frame]."""
        return self.hole[2] * 1000.0

    @property
    def ws_xy(self):
        """Taught work-station x, y [mm]."""
        return np.array(self.ws[:2]) * 1000.0

    @property
    def hole_xy(self):
        """Taught hole x, y [mm]."""
        return np.array(self.hole[:2]) * 1000.0


def shape_refs(samples, exclude):
    """SAMPLES -> {shape: ShapeRef}, every pose averaged over its samples without the excluded IDs.

    SystemExit on unknown IDs in exclude, or when every sample of a shape / kind is excluded.
    """
    known = {k for s in samples.values() for kind in s.values() for k in kind}
    unknown = exclude - known
    if unknown:
        raise SystemExit(f"--exclude: unknown sample IDs {sorted(unknown)} (known: {sorted(known)})")
    refs = {}
    for shape in SHAPES:
        if shape not in samples:
            continue
        got = {}
        for kind in ("ws", "hole"):
            used = {k: v for k, v in samples[shape][kind].items() if k not in exclude}
            if not used:
                raise SystemExit(f"{shape} {kind}: every sample is excluded - nothing to average")
            avg = average_pose(used)
            dev = {k: 1000.0 * float(np.linalg.norm(np.subtract(v[:3], avg[:3]))) for k, v in used.items()}
            worst = max(dev, key=dev.get)
            got[kind] = (avg, sorted(used), dev[worst], worst)
        refs[shape] = ShapeRef(shape, got["ws"][0], got["hole"][0], got["ws"][1], got["hole"][1],
                               got["ws"][2], got["hole"][2], got["hole"][3])
    return refs


def print_refs(refs):
    """Print the per-shape table of taught pitches, heights and hole positions; flags a hole sample
    more than 8 mm from its average.
    """
    print("\nGripper orientation + heights per shape, inferred from peg_sort.py SAMPLES (averages):")
    print("  shape      pick pitch  grasp z | place pitch  insert z | recorded hole x, y      samples")
    for r in refs.values():
        print(f"  {r.shape:9s}  {r.ws[4]:+7.2f} deg {r.pick_z:6.1f} mm | {r.hole[4]:+7.2f} deg {r.place_z:6.1f} mm | "
              f"x={r.hole_xy[0]:+7.1f} y={r.hole_xy[1]:+7.1f}   ws {'+'.join(f'#{i}' for i in r.ws_ids)}, hole "
              f"{'+'.join(f'#{i}' for i in r.hole_ids)}" + (f"  (hole #{r.hole_worst} is {r.hole_spread:.1f} mm from "
                                                             f"the average!)" if r.hole_spread > 8 else ""))


# =========================================================================== #
#  Window or terminal questions                                               #
# =========================================================================== #
class UI:
    """Keyboard questions: in the OpenCV camera window, or in the terminal when there is no display (headless)."""
    KEYMAP = {10: "\n", 13: "\n", 32: " ", 27: "q"}

    def __init__(self, headless=False):
        """headless: True = no window, every question goes to the terminal."""
        self.headless = headless

    def show(self, img):
        """Show img in the camera window (nothing when headless or img is None)."""
        if not self.headless and img is not None:
            cv2.imshow(WIN, img)

    def poll(self, ms=1):
        """Key pressed in the camera window ('' if none)."""
        if self.headless:
            time.sleep(ms / 1000.0)
            return ""
        k = cv2.waitKey(ms) & 0xFF
        if k == 255:
            return ""
        return self.KEYMAP.get(k, chr(k).lower())

    def ask(self, prompt, keys, render=None):
        """Wait for one of `keys` ('\\n' = ENTER). Window: live image from render(); no window: terminal."""
        print(prompt, flush=True)
        if self.headless:
            while True:
                s = input("  > ").strip().lower()
                k = s[:1] if s else "\n"
                if k in keys:
                    return k
                print(f"  type one of: {', '.join('ENTER' if c == chr(10) else c for c in keys)}")
        while True:
            if render is not None:
                self.show(render())
            k = self.poll(30)
            if k and k in keys:
                return k


# =========================================================================== #
#  Camera side: tray, holes, pegs - all handed out in the robot base frame    #
# =========================================================================== #
# ---- tray colour ranges tried by the hole scan (HSV). The first is sorting_detector's own TRAY_LO / TRAY_HI.
#      Shade or a change of light can push part of the tray out of that range: the tray outline then comes out too
#      narrow (e.g. 326 x 83 mm instead of 330 x 100), the flattened tray view is stretched and holes are missed or
#      mixed up. The scan tries these ranges and keeps the outline whose measured size is closest to the real tray
#      (sorting_detector TRAY_MM_W x TRAY_MM_H).
TRAY_RANGES = [
    ("standard", sd.TRAY_LO.copy(), sd.TRAY_HI.copy()),
    ("wider", np.array((3, 20, 35)), np.array((26, 160, 255))),
    ("widest", np.array((0, 12, 25)), np.array((30, 185, 255))),
]
SCAN_FRAMES = 40          # colour frames for the tray outline
HOLE_FRAMES = 9           # colour frames the holes are voted over
QUAD_KEEP_PX = 6.0        # outline frames further than this from the median outline are dropped
TRAY_PLANE_TOL_MM = 12.0  # tray-coloured pixels further than this from the tray plane are not tray (table, black)
HOLE_SEARCH_MM = 15.0     # a hole searched at its taught position must be centred within this of it


# Red / dark-red rims around the holes (burnt laser-cut edges or marker lines) are not tray-coloured: the hole scan
# then sees each hole as hole + rim (too big, e.g. a 32 mm circle measured 45 mm) and a hole near the tray end can
# merge with the outside. The scan also tries with these red pixels counted as tray and keeps the better result.
RED_RANGES = [(np.array((0, 60, 35)), np.array((9, 255, 255))), (np.array((165, 60, 35)), np.array((180, 255, 255)))]


@contextlib.contextmanager
def red_as_tray(on):
    """While on: sorting_detector's tray-colour mask also contains red pixels (only used for the hole search)."""
    if not on:
        yield
        return
    orig = sd.tray_mask_of

    def mask(img):
        """sorting_detector's tray mask OR the red-rim pixels (RED_RANGES)."""
        m = orig(img)
        hsv = cv2.cvtColor(cv2.GaussianBlur(img, (5, 5), 0), cv2.COLOR_BGR2HSV)
        red = np.zeros(m.shape, np.uint8)
        for lo_, hi_ in RED_RANGES:
            red |= cv2.inRange(hsv, lo_, hi_)
        red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        return m | red
    sd.tray_mask_of = mask
    try:
        yield
    finally:
        sd.tray_mask_of = orig


@contextlib.contextmanager
def dark_is_hole(on):
    """While on: DARK pixels (below half the tray's own brightness) are never tray. A hole whose floor is dark brown
    (e.g. HSV ~ (16, 45, 68)) falls inside the tray colour range and would be taken for tray - no hole found.
    Only used for the hole search."""
    if not on:
        yield
        return
    orig = sd.tray_mask_of

    def mask(img):
        """sorting_detector's tray mask without the pixels darker than half the tray's median brightness."""
        m = orig(img)
        v = cv2.cvtColor(cv2.GaussianBlur(img, (5, 5), 0), cv2.COLOR_BGR2HSV)[:, :, 2]
        on_tray = v[m > 0]
        if on_tray.size < 500:
            return m
        thr = 0.5 * float(np.median(on_tray))
        m = m.copy()
        m[v < thr] = 0
        return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    sd.tray_mask_of = mask
    try:
        yield
    finally:
        sd.tray_mask_of = orig


def padded(img, H, pad=160):
    """The camera image grown by pad px on every side by repeating its edge pixels, and H for it: a tray that runs
    past the image edge then continues as tray colour in the flattened view instead of black (black = 'hole')."""
    big = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    return big, H @ np.array([[1.0, 0.0, -pad], [0.0, 1.0, -pad], [0.0, 0.0, 1.0]])


@contextlib.contextmanager
def tray_range(lo, hi):
    """Temporarily use another tray colour range in sorting_detector (find_tray / detect_slots read it)."""
    old = sd.TRAY_LO, sd.TRAY_HI
    sd.TRAY_LO, sd.TRAY_HI = lo, hi
    try:
        yield
    finally:
        sd.TRAY_LO, sd.TRAY_HI = old


def quiet(fn, *a, **k):
    """Call fn without its prints."""
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


def vote_slots(per_frame):
    """Holes found in several frames -> one hole per shape: the same hole (within 25 px of the flattened view) in
    different frames is one cluster; each cluster gets the shape most frames called it (one shape per cluster,
    best total). -> (slots, report lines)"""
    n = max(1, len(per_frame))
    clusters = []
    for slots in per_frame:
        for s_ in slots:
            for cl in clusters:
                if math.hypot(cl[0]["cf"][0] - s_["cf"][0], cl[0]["cf"][1] - s_["cf"][1]) < 25.0:
                    cl.append(s_)
                    break
            else:
                clusters.append([s_])
    clusters = sorted([cl for cl in clusters if len(cl) >= max(2, n // 3)], key=len, reverse=True)[:4]
    if not clusters:
        return [], []
    shapes = list(sd.EXPECTED_SLOTS)

    def votes(cl, sh):
        """Score for naming cluster cl as shape sh: frames that called it sh (+ a tiny IoU tie-break)."""
        return sum(1 for x in cl if x["shape"] == sh) + 1e-3 * sum(float(x.get("iou", 0)) for x in cl if x["shape"] == sh)
    best = max(itertools.permutations(shapes, len(clusters)),
               key=lambda perm: sum(votes(cl, sh) for cl, sh in zip(clusters, perm)))
    out, rep = [], []
    for cl, sh in zip(clusters, best):
        mx = float(np.median([x["cf"][0] for x in cl]))
        my = float(np.median([x["cf"][1] for x in cl]))
        same = [x for x in cl if x["shape"] == sh] or cl
        r = dict(min(same, key=lambda x: math.hypot(x["cf"][0] - mx, x["cf"][1] - my)))
        r["shape"] = sh
        k = sum(1 for x in cl if x["shape"] == sh)
        r["votes"] = k
        out.append(r)
        other = {}
        for x in cl:
            if x["shape"] != sh:
                other[x["shape"]] = other.get(x["shape"], 0) + 1
        rep.append(f"{sh} {k}/{n}" + ("" if not other else " (also read as " +
                                        ", ".join(f"{o} {c}x" for o, c in other.items()) + ")"))
    out.sort(key=lambda s_: s_["cf"][1])                 # same numbering as sorting_detector.detect_slots
    rows = [[out[0]]]
    for s_ in out[1:]:
        if abs(s_["cf"][1] - rows[-1][-1]["cf"][1]) > sd.ROW_TOL:
            rows.append([s_])
        else:
            rows[-1].append(s_)
    ordered = []
    for row in rows:
        for s_ in sorted(row, key=lambda x: x["cf"][0]):
            s_["id"] = len(ordered) + 1
            ordered.append(s_)
    return ordered, rep


class Vision:
    """Camera side of the run: finds the tray, its holes and the pegs with sorting_detector and hands every
    position out in the robot base frame [mm] through the calibration T (BASE_T_CAMERA_mm).
    """
    def __init__(self, cam, T):
        """Args:
            cam: open camera (calib_camera_node.Camera with .intr set).
            T:   4x4 transform robot base <- camera, in mm.
        """
        self.cam, self.intr = cam, cam.intr
        self.T = np.asarray(T, float)
        self.Ti = np.linalg.inv(self.T)
        self.corners = self.H = self.H_inv = self.bg = self.geo = None
        self.slots, self.yaw_off, self.frame_no = [], 0.0, 0
        self.tracker = sd.PegTracker()

    # ---- frames -------------------------------------------------------------
    def c2r(self, Pc):
        """Camera point(s) [mm] -> robot base frame."""
        return np.asarray(Pc, float) @ self.T[:3, :3].T + self.T[:3, 3]

    def r2c(self, Pb):
        """Robot base point(s) [mm] -> camera frame."""
        return np.asarray(Pb, float) @ self.Ti[:3, :3].T + self.Ti[:3, 3]

    def r2px(self, Pb):
        """Robot base point(s) [mm] -> image pixel(s); None if a point is not in front of the camera."""
        Pc = self.r2c(Pb).reshape(-1, 3)
        if (Pc[:, 2] <= 1.0).any():
            return None
        uv = self.cam.project(Pc)
        return uv[0] if np.ndim(Pb) == 1 else uv

    def t2r(self, Pt):
        """Tray-frame point(s) [mm] (sorting_detector.TrayFrame) -> robot base frame."""
        return self.c2r(self.geo.to_cam(np.asarray(Pt, float)))

    def hole_at_z(self, slot, z):
        """Robot point at height z on a detected hole's axis."""
        return self.axis_at_z(slot["tray"][:2], z)

    def axis_at_z(self, xy_tray, z):
        """Robot point at height z on the vertical (tray normal) through tray point xy: peg / hole axis."""
        P0 = self.t2r([xy_tray[0], xy_tray[1], 0.0])
        d = self.t2r([xy_tray[0], xy_tray[1], 100.0]) - P0
        return P0 + d * (z - P0[2]) / d[2]

    def surface_h_at_robot_xy(self, xy):
        """Height above the tray plane (mm) of the EMPTY scene under robot x, y (calibration depth), or None."""
        geo, h = self.geo, 0.0
        a = geo.height(self.r2c([xy[0], xy[1], 0.0]))
        b = geo.height(self.r2c([xy[0], xy[1], 1.0])) - a
        for _ in range(5):
            px = geo.to_px(self.r2c([xy[0], xy[1], (h - a) / b]))[0]
            hh = geo.surface_height_at(px)
            if hh is None:
                return None
            if abs(hh - h) < 0.2:
                return hh
            h = hh
        return h

    # ---- tray + holes (empty tray) ---------------------------------------------
    @staticmethod
    def tray_in(color, prev=None, exclude=None):
        """Tray outline -> (corners or None, info). A tray that runs past the image edge is ACCEPTED (unlike
        sorting_detector): the holes are measured in 3D from depth, so only the holes must be in view; the visible
        part of the tray is enough for the tray plane and the flattened view the holes are found in.
        exclude: pixels that cannot be tray (see tray_exclude)."""
        c, mask, info = sd.find_tray(color, prev, exclude)
        if c is None:
            return None, info
        if not sd.valid_quad(c):
            return None, "tray outline is not a clean quadrilateral"
        if sd.touches_border(c, color.shape):
            return c, "found, runs past the image edge (OK if all four holes are in view)"
        return c, "found"

    def tray_exclude(self, color, depth):
        """Pixels that cannot be tray top: not on the tray's plane (table, dark surroundings, arm, hand) or without
        depth. Dark / black surfaces flicker in and out of the tray colour range (camera noise), which made the
        outline jump into them; their depth shows they are not the tray. The plane is fitted to the biggest
        tray-coloured blob, outliers dropped. -> uint8 mask (255 = not tray) or None (no usable depth)."""
        if depth is None:
            return None
        m = sd.tray_mask_of(color)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
        if n <= 1:
            return None
        k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        v, u = np.nonzero((lab == k) & (depth > 0))
        if len(u) < 500:
            return None
        st = max(1, len(u) // 6000)
        P = np.asarray(self.cam.deproject(u[::st], v[::st], depth[v[::st], u[::st]]), np.float64).reshape(-1, 3)
        nrm, d0 = None, None
        for _ in range(3):
            c0 = P.mean(axis=0)
            nrm = np.linalg.svd(P - c0, full_matrices=False)[2][-1]
            d0 = -float(nrm @ c0)
            r = np.abs(P @ nrm + d0)
            keep = r < max(3.0, 3.0 * float(np.median(r)))
            if keep.sum() < 200:
                break
            P = P[keep]
        h_, w_ = depth.shape
        if getattr(self, "_grid", None) is None or self._grid[0].shape != depth.shape:
            self._grid = np.meshgrid(np.arange(w_, dtype=np.float64), np.arange(h_, dtype=np.float64))
        X = np.asarray(self.cam.deproject(self._grid[0], self._grid[1], depth.astype(np.float64)), np.float64)
        dist = np.abs(X.reshape(-1, 3) @ nrm + d0).reshape(h_, w_)
        return (((depth <= 0) | (dist > TRAY_PLANE_TOL_MM)) * 255).astype(np.uint8)

    def calibrate(self, need=8, frames=SCAN_FRAMES, expected=None):
        """Tray outline (median of many frames, best of TRAY_RANGES by real size; pixels off the tray plane left out)
        -> depth background -> tray frame -> holes voted over HOLE_FRAMES frames; a hole still missing is searched
        for where it was taught (expected = {shape: robot x, y}). -> (ok, message)"""
        grabs = [self.cam.grab() for _ in range(frames)]
        shots = [g_[0] for g_ in grabs]
        excl = [self.tray_exclude(col, dep) for col, dep in grabs]
        cands, why = [], "no frames"
        for name, lo, hi in TRAY_RANGES:
            quads, prev = [], None
            with tray_range(lo, hi):
                for col, ex in zip(shots, excl):
                    c, info = self.tray_in(col, prev, ex)
                    if c is None:
                        why = info
                        continue
                    prev = c if prev is None else prev
                    quads.append(np.asarray(c, np.float32))
            if len(quads) < max(5, need // 2):
                continue
            Q = np.stack(quads)
            med = np.median(Q, axis=0)
            keep = Q[np.abs(Q - med).reshape(len(Q), -1).max(axis=1) <= QUAD_KEEP_PX]
            if len(keep) < max(5, need // 2):
                why = f"tray outline jumps between frames ({name} colours)"
                continue
            corners = keep.mean(axis=0).astype(np.float32)
            if sd.valid_quad(corners):
                cands.append((name, lo, hi, corners))
        if not cands:
            return False, f"no steady tray outline ({why})"
        print("  capturing the depth background (keep everything still) ...", flush=True)
        color, bg = self.cam.capture(30)
        tried, best, bad_geo = [], None, None
        ex_bg = self.tray_exclude(color, bg)
        for name, lo, hi, corners in cands:
            with tray_range(lo, hi):
                _, mask, _ = sd.find_tray(color, corners, ex_bg)
            try:
                H, H_inv = sd.homographies(corners)
                geo = sd.TrayFrame(self.intr, bg, mask, corners)
            except (np.linalg.LinAlgError, ValueError):
                continue
            err = abs(geo.length - sd.TRAY_MM_W) / sd.TRAY_MM_W + abs(geo.width - sd.TRAY_MM_H) / sd.TRAY_MM_H
            tried.append(f"{name} {geo.length:.0f} x {geo.width:.0f} mm")
            if geo.plane_rms <= 4.0 and (best is None or err < best[0] - 0.01):
                best = (err, name, lo, hi, corners, H, H_inv, geo, mask)
            elif geo.plane_rms > 4.0:
                bad_geo = geo
        if best is None:
            g_ = bad_geo
            if g_ is not None:
                return False, (f"the tray scan is not flat (rms {g_.plane_rms:.1f} mm, camera {g_.d:.0f} mm above it) - "
                               f"something was in view or moving (hand, arm). Keep everything still and out of view, "
                               f"ENTER again")
            return False, "3D tray frame not built (too few tray pixels with depth)"
        err, name, lo, hi, corners, H, H_inv, geo, tmask = best
        print(f"  tray outline: {', '.join(tried)}  (real {sd.TRAY_MM_W:.0f} x {sd.TRAY_MM_H:.0f}) -> using '{name}'")
        clipped, completed = sd.touches_border(corners, color.shape), False
        if clipped:                                            # outline cut by the image edge: complete it
            c2, note = self.complete_outline(corners, geo, tmask, color.shape)
            if note:
                try:
                    H2, H_inv2 = sd.homographies(c2)
                    geo.set_corners(c2)
                    corners, H, H_inv, completed = c2, H2, H_inv2, True
                    print(f"  tray runs past the image edge: {note} -> {geo.length:.0f} x {geo.width:.0f} mm")
                except (np.linalg.LinAlgError, ValueError):
                    geo.set_corners(corners)
        # holes: several frames x (colour ranges, red rims as tray or not), majority vote per hole, best kept
        hole_shots = [color] + [self.cam.grab()[0] for _ in range(HOLE_FRAMES - 1)]
        pads = [padded(f, H) for f in hole_shots]
        order = [r for r in TRAY_RANGES if r[0] == name] + [r for r in TRAY_RANGES if r[0] != name]
        best_h = None
        for rn, rlo, rhi in order:
            for red, dark in ((False, False), (True, False), (False, True)):
                with tray_range(rlo, rhi), red_as_tray(red), dark_is_hole(dark):
                    per = [quiet(sd.detect_slots, f, Hp) for f, Hp in pads]
                got, r_ = vote_slots(per)
                if not got:
                    continue
                score = (len(got), sum(x.get("votes", 0) for x in got),
                         float(np.mean([float(x.get("iou", 0)) for x in got])))
                if best_h is None or score > best_h[0]:
                    best_h = (score, got, r_, rn + (", red rims counted as tray" if red else "")
                              + (", dark = hole" if dark else ""))
        if best_h is None:
            best_h = ((0, 0, 0.0), [], [], name)
        _, slots, rep, used = best_h
        missing_ = [x for x in sd.EXPECTED_SLOTS if x not in {s_["shape"] for s_ in slots}]
        shift = np.zeros(2)
        if missing_ and expected and slots:                    # where the camera sees the found holes vs taught:
            self.geo = geo                                     # look for the missing one with the same shift
            geo.annotate_slots(slots, H_inv)
            d_ = [self.t2r([s_["tray"][0], s_["tray"][1], 0.0])[:2] - np.asarray(expected[s_["shape"]], float)
                  for s_ in slots if s_["shape"] in expected]
            if len(d_) >= 2:
                shift = np.median(np.array(d_), axis=0)
        for sh_ in missing_:
            if expected and sh_ in expected:                   # look where it was taught (+ the camera's shift)
                got_ = self.find_hole_near(sh_, np.asarray(expected[sh_], float) + shift, geo, H, pads, order)
                if got_ is not None:
                    got_["id"] = 0
                    slots.append(got_)
                    rep.append(f"{sh_} {got_['votes']}/{len(pads)} (searched at its taught position)")
        if not slots:
            return False, "no holes found in the tray"
        slots.sort(key=lambda s_: s_["cf"][0])
        for i_, s_ in enumerate(slots):
            s_["id"] = i_ + 1
        missing = [x for x in sd.EXPECTED_SLOTS if x not in {s_["shape"] for s_ in slots}]
        print(f"  holes (voted over {len(hole_shots)} frames, '{used}' colours): " + "; ".join(rep)
              + (f"   NOT found: {missing}" if missing else ""))
        with tray_range(lo, hi):
            sd.check_tray_proportion(slots)
        geo.annotate_slots(slots, H_inv)
        self.corners, self.H, self.H_inv, self.bg, self.geo, self.slots = corners, H, H_inv, bg, geo, slots
        self.tray_lo_hi = (lo, hi)
        v = self.T[:3, :3] @ geo.R[:, 0]                       # tray x axis in the robot frame
        self.yaw_off = math.degrees(math.atan2(v[1], v[0]))
        for s in slots:
            s["r_centre"] = self.t2r([s["tray"][0], s["tray"][1], 0.0])   # hole centre on the tray surface
            s["r_yaw"] = s["yaw"] + self.yaw_off
            s["img_poly"] = sd.project(s["contour"], H_inv).reshape(-1, 2)
        self.tracker = sd.PegTracker()
        return True, geo.summary() + ("  [tray outline completed beyond the image edge]" if completed else
                                      "  [tray runs past the image edge: 'tray measured' is only the visible part, "
                                      "hole positions are not affected]" if clipped else "")

    def find_hole_near(self, shape, xy, geo, H, pads, ranges):
        """Search ONE hole in a 56 x 56 mm window around where it was taught (robot x, y on the tray plane), in the
        flattened tray view of every frame: the non-tray blob there, fitted with THIS shape. A hole near the tray
        end / image edge or with a dark rim fails the general scan's checks but is found this way.
        -> slot dict (like sorting_detector.detect_slots) or None."""
        Ti = self.Ti
        A = self.r2c([xy[0], xy[1], 0.0])
        B = Ti[:3, :3] @ np.array([0.0, 0.0, 1.0])
        den = float(geo.n @ B)
        if abs(den) < 1e-6:
            return None
        z = -(float(geo.n @ A) + geo.d) / den
        uv = self.r2px([xy[0], xy[1], z])
        if uv is None:
            return None
        cx, cy = cv2.perspectiveTransform(np.float32([[uv]]).reshape(1, 1, 2), H).reshape(2)
        k = sd.TRAY_W / sd.TRAY_MM_W                           # flattened px per mm
        if not (-20 <= cx <= sd.TRAY_W + 20 and -20 <= cy <= sd.TRAY_H + 20):
            return None
        win = np.zeros((sd.TRAY_H, sd.TRAY_W), np.uint8)
        cv2.rectangle(win, (int(cx - 28 * k), int(cy - 28 * k)), (int(cx + 28 * k), int(cy + 28 * k)), 255, -1)
        k7 = np.ones((7, 7), np.uint8)
        split = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (sd.SLOT_SPLIT_PX, sd.SLOT_SPLIT_PX))
        max_off = HOLE_SEARCH_MM * k                           # the hole centre must be this close to the taught one

        def hole_like(c_):
            """Same checks as sorting_detector's hole scan: size, solid (not a sliver / strip), not too thin."""
            a_ = cv2.contourArea(c_)
            if not (sd.SLOT_MIN_AREA <= a_ <= sd.SLOT_MAX_AREA):
                return False
            if a_ / (cv2.contourArea(cv2.convexHull(c_)) + 1e-6) < 0.85:
                return False
            (_, _), (w_, h_), _ = cv2.minAreaRect(c_)
            return min(w_, h_) / (max(w_, h_) + 1e-6) >= 0.5

        def nearest(cnts_):
            """The blob that CONTAINS the taught point, else the one whose centre is nearest - within max_off."""
            best_c, best_k = None, None
            for c_ in cnts_:
                if not hole_like(c_):
                    continue
                m_ = cv2.moments(c_)
                if m_["m00"] <= 0:
                    continue
                d_c = math.hypot(m_["m10"] / m_["m00"] - cx, m_["m01"] / m_["m00"] - cy)
                if d_c > max_off:
                    continue
                key = (0 if cv2.pointPolygonTest(c_, (float(cx), float(cy)), False) >= 0 else 1, d_c)
                if best_k is None or key < best_k:
                    best_c, best_k = c_, key
            return best_c

        best = None
        for rn, rlo, rhi in ranges:
            for red, dark in ((False, True), (True, False), (False, False)):
                hits = []
                with tray_range(rlo, rhi), red_as_tray(red), dark_is_hole(dark):
                    for f, Hp in pads:
                        rect = cv2.warpPerspective(f, Hp, (sd.TRAY_W, sd.TRAY_H))
                        holes = cv2.bitwise_and(win, cv2.bitwise_not(sd.tray_mask_of(rect)))
                        holes = cv2.morphologyEx(cv2.morphologyEx(holes, cv2.MORPH_OPEN, k7), cv2.MORPH_CLOSE, k7)
                        for attempt in (holes, cv2.morphologyEx(holes, cv2.MORPH_OPEN, split)):
                            cnts, _ = cv2.findContours(attempt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                            c = nearest(cnts)
                            if c is None:
                                continue
                            fs = sd.fit_shape(c.astype(np.float32), min_aspect=sd.SLOT_MIN_ASPECT)
                            sc = fs["scores"].get(shape, fs["scores"].get("rectangle") if shape == "square" else None)
                            poly = fs["polys"].get(shape) if fs.get("polys") else None
                            if poly is None and shape == "square" and fs.get("polys"):
                                poly = fs["polys"].get("rectangle")
                            if poly is not None and sc is not None and sc >= 0.78:
                                hits.append((float(sc), fs, c, poly))
                                break
                if len(hits) >= max(2, len(pads) // 2) and (best is None or len(hits) > best[0]):
                    best = (len(hits), hits)
        if best is None:
            return None
        n_, hits = best
        sc, fs, c, poly = max(hits, key=lambda h_: h_[0])
        cf = (float(fs["center"][0]), float(fs["center"][1]))
        return {"contour": np.asarray(poly, np.float32).reshape(-1, 1, 2), "shape": shape, "cf": cf,
                "center": sd.ip(cf), "yaw": sd._yaw_of(poly, shape), "iou": round(sc, 3), "raw": c, "votes": n_}

    def complete_outline(self, corners, geo, mask, img_shape):
        """A tray running past the image edge gives a cut-off, skewed outline: the flattened tray view is then
        distorted and the holes near that end are missed (the triangle). The tray is a rectangle of known size
        (sorting_detector TRAY_MM_W x TRAY_MM_H): fit it, in mm on the tray plane, to the tray's real edges (outline
        points away from the image border), extend it over the cut end, and use its corners.
        -> (corners, note) - note '' if nothing to do."""
        h, w = img_shape[:2]
        m = sd.EDGE_MARGIN + 3
        c = np.asarray(corners, np.float64).reshape(4, 2)
        edge = (c[:, 0] < m) | (c[:, 0] > w - 1 - m) | (c[:, 1] < m) | (c[:, 1] > h - 1 - m)
        if not edge.any() or mask is None:
            return corners, ""
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cnts:
            return corners, ""
        P = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)
        inside = (P[:, 0] >= m) & (P[:, 0] <= w - 1 - m) & (P[:, 1] >= m) & (P[:, 1] <= h - 1 - m)
        if inside.sum() < 60:
            return corners, ""
        T = geo.to_tray(geo.lift(P, 0.0))[:, :2]
        box = cv2.boxPoints(cv2.minAreaRect(T[inside].astype(np.float32))).astype(np.float64)
        e1, e2 = box[1] - box[0], box[2] - box[1]
        ux = e1 if np.linalg.norm(e1) >= np.linalg.norm(e2) else e2
        ux = ux / np.linalg.norm(ux)
        uy = np.array([-ux[1], ux[0]])
        c0 = T[inside].mean(axis=0)
        S, Tt = (T - c0) @ ux, (T - c0) @ uy
        s0, s1, t0, t1 = S[inside].min(), S[inside].max(), Tt[inside].min(), Tt[inside].max()
        notes = []
        for axis, full, lo_, hi_, val in (("length", sd.TRAY_MM_W, s0, s1, S), ("width", sd.TRAY_MM_H, t0, t1, Tt)):
            d = full - (hi_ - lo_)
            if d <= 5.0:
                continue
            cut = val[~inside]                                  # outline points on the image border
            n_lo = int(np.sum(np.abs(cut - lo_) < 15.0))
            n_hi = int(np.sum(np.abs(cut - hi_) < 15.0))
            if n_lo == n_hi:
                continue
            if axis == "length":
                s0, s1 = (s0, s1 + d) if n_hi > n_lo else (s0 - d, s1)
            else:
                t0, t1 = (t0, t1 + d) if n_hi > n_lo else (t0 - d, t1)
            notes.append(f"{axis} +{d:.0f} mm")
        if not notes:
            return corners, ""
        rect = [c0 + sa * ux + tb * uy for sa, tb in ((s0, t0), (s1, t0), (s1, t1), (s0, t1))]
        px = geo.tray_to_px(np.array([[q[0], q[1], 0.0] for q in rect])).astype(np.float32).reshape(4, 2)
        if not np.isfinite(px).all() or not cv2.isContourConvex(px.reshape(-1, 1, 2)):
            return corners, ""
        return sd.normalize_corners(px, np.asarray(corners, np.float32)), \
            "outline completed from the real tray size (" + ", ".join(notes) + ")"

    def slot_for(self, shape):
        """Detected hole of this shape (slot dict), or None."""
        return next((s for s in self.slots if s["shape"] == shape), None)

    def slot_at(self, xy_tray):
        """Hole whose outline contains tray point xy (a peg standing in it), or None."""
        for s in self.slots:
            if cv2.pointPolygonTest(s["tray_poly"].reshape(-1, 1, 2).astype(np.float32),
                                    (float(xy_tray[0]), float(xy_tray[1])), False) >= 0:
                return s
        return None

    # ---- pegs -------------------------------------------------------------------
    def detect(self, color, depth):
        """One frame of sorting_detector's peg pipeline (depth gate, hands ignored, top face, voting)."""
        self.frame_no += 1
        gate = sd.depth_gate(depth, self.bg, color)
        hand, _ = sd.scan_raised(color, gate[1])
        pegs = sd.detect_pegs(color, depth, self.bg, self.intr, self.H, gate,
                              do_fit=(self.frame_no % sd.FIT_EVERY == 0), ignore=hand, geo=self.geo, known=set())
        self.tracker.update(pegs)
        return pegs, hand

    def top_xy(self, p, depth):
        """Tray x, y of a peg's top-face centre, also for a peg standing IN a hole. The detector's top-face test
        measures height against the empty-scene depth, which inside a hole is partly floor, partly wall, so it
        often rejects such a peg; here the height above the tray PLANE is used (TrayFrame.blob_top/top_outline)."""
        if p.get("top_t") is not None:
            return np.asarray(p["top_t"][:2], float)
        h = self.geo.blob_top(p["silhouette"], depth)
        out = None if h is None else self.geo.top_outline(p["silhouette"], depth, h)
        if out is None or len(out) < 3:
            return None
        M = cv2.moments(np.asarray(out, np.float32).reshape(-1, 1, 2))
        return None if M["m00"] < 1 else np.array([M["m10"] / M["m00"], M["m01"] / M["m00"]])

    def peg_robot(self, p):
        """Detected peg (sorting_detector dict with top_t) -> dict in the robot base frame: colour, shape, votes,
        tray x, y, top point [mm], top height, height of what it stands on, length, rotation [deg, robot frame], pixel.
        """
        tt = np.asarray(p["top_t"], float)                     # tray frame: axis x, y and top height
        sup = self.geo.surface_height_at(self.geo.tray_to_px([tt[0], tt[1], 0.0])[0])
        sup = 0.0 if sup is None else float(sup)
        return {"color": p["color"], "shape": p["shape"], "votes": p["votes"], "tray_xy": tt[:2].copy(),
                "top": self.t2r(tt), "h_top": float(tt[2]), "support_h": sup, "length": float(tt[2]) - sup,
                "yaw": None if p.get("yaw") is None else float(p["yaw"]) + self.yaw_off, "px": p["center"]}


# =========================================================================== #
#  Plan                                                                        #
# =========================================================================== #
@dataclass
class Step:
    """One step of a peg plan.

    kind: open | close (gripper), lift | line | joint | carry (motion, planned by PegSortRun.plan_step),
    pause (wait for ENTER, only with --step).
    """
    name: str
    kind: str                      # open | close | lift | joint | line | carry | pause
    target: object = None          # robot point (mm)
    pitch: float = 0.0             # rad
    speed: float = None            # mm/s (lines)
    dip: str = "error"             # joint moves: 'error' | 'warn' if the tip sags (see DIP_MM)
    show: object = None            # robot point drawn as the GREEN cross while this step runs
    key: str = ""                  # 'pick' / 'place': remember the joints there (preview)
    hint: str = ""                 # extra advice if the step fails
    q: object = None               # joint vector (joint moves to HOME / park) - pass as q=


@dataclass
class PegPlan:
    """Everything planned for one peg: the detected peg and its hole, the taught ShapeRef, pick / place
    points [mm] and pitches [rad], the safe carry height, the steps with their trajectories, the joints at
    the keyed steps, the expected rotation error and notes for the user.
    """
    peg: dict
    slot: object
    ref: ShapeRef
    pick: np.ndarray
    place: np.ndarray
    pitch_pick: float
    pitch_place: float
    z_carry: float
    corr: float
    steps: list = field(default_factory=list)
    trajs: list = field(default_factory=list)
    total_s: float = 0.0
    q_at: dict = field(default_factory=dict)
    turn: float = None             # deg the peg must be turned by hand (+ = counter-clockwise from above)
    jaw: float = None              # deg between the peg faces and the jaws at the pick
    notes: list = field(default_factory=list)
    info: dict = field(default_factory=dict)


def seg_dist(p, a, b):
    """Distance from point p to the segment a-b (numpy vectors of the same size)."""
    ab = b - a
    t = 0.0 if float(ab @ ab) < 1e-9 else float(np.clip((p - a) @ ab / (ab @ ab), 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


class PlanarIK:
    """Closed-form IK of the OpenMANIPULATOR-X's joints 2-4 (one vertical plane, joint1 faces the target). Used to
    REJECT unreachable targets in microseconds (calib_check's numerical IK needs ~0.5 s to give up) and to measure
    the pick zone; the joints themselves still come from calib_check. Checked against the URDF model at start
    (random poses); if that check fails it is switched off and everything goes to the numerical IK."""

    def __init__(self, arm):
        """Read link lengths and joint signs from the arm model (calib_check.Arm) and self-check against it.
        self.ok = True only if the closed form reproduces 60 random poses of the model.
        """
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
            # (pitch is measured along joint1's direction, as calib_check.Arm.pitch does)
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
                c1 = self.phi0[0] - ph1                                  # cumulative joint sums (times sign)
                c2 = self.phi0[1] - ph2
                c3 = self.phi0[2] - ph3
                q = np.array([c1, c2 - c1, c3 - c2]) * np.array(self.sg)
                q = (q + math.pi) % (2 * math.pi) - math.pi
                if np.all(q >= self.lo - slack_rad) and np.all(q <= self.hi + slack_rad):
                    out.append(np.r_[cc.wrap(yaw + side), q])
        return out

    def maybe(self, P_mm, pitch):
        """False only if the target is surely unreachable (2 mm / 0.5 deg slack for the numerical IK)."""
        return not self.ok or bool(self.solutions(P_mm, pitch, slack_mm=2.0, slack_rad=math.radians(0.5)))


# =========================================================================== #
#  The run (calib_check.Checker = MODE 4 machinery)                           #
# =========================================================================== #
class PegSortRun(cc.Checker):
    """The final run: camera holes + camera peg -> pick & place with calib_check MODE 4 motion.

    Extends calib_check.Checker (IK, line / joint planning, execute, hold, E-stop, ROS handles).
    Flow, looped in run(): phase_holes -> wait_for_peg -> make_plan -> describe / confirm -> run_plan -> verify.
    """
    def __init__(self, args, refs):
        # same start-up as calib_check.Checker, plus: dry run touches neither torque nor the joint1 mode
        """Start-up.

        Loads the calibration, connects to ROS, builds the arm model from the URDF and checks it against TF,
        checks the servo modes, reads / switches joint1's Operating Mode, keeps joint1 on the JOINT1_SIDE side,
        switches the torque ON holding the current pose (not in --dry-run), remembers that pose as the parking
        pose, measures the pick zone, starts watching the servos and sets up the gripper.

        Args:
            args: parse_args() result.
            refs: {shape: ShapeRef}.
        """
        with open(args.calib) as f:
            self.cal = json.load(f)
        self.T = np.array(self.cal["BASE_T_CAMERA_mm"], float)       # includes the saved fine offset
        self.Ti = np.linalg.inv(self.T)
        self.base = self.cal.get("base_frame", "world")
        self.args, self.refs = args, refs
        self.inbox, self.view_target, self.headless = None, None, False
        self.stop_evt = threading.Event()
        self.grip_estop = threading.Event()
        self.multiturn, self.spin_th, self.results = False, None, []
        self.cam = self.vision = self.grip = None
        self.ui = UI(headless=True)
        self.phase, self.plan, self.filled, self.hole_src = "", None, set(), {}
        self.placed = {}                                  # shape -> robot x, y [mm] where it was put
        self.rclpy, self.node, self.exe, self.spin_th = cc.make_ros(args)
        urdf = cc.wait_for(lambda: self.node.urdf, 5, "/robot_description (is hardware.launch.py running?)")
        self.arm = cc.Arm(urdf, self.base, args.ee_frame, args.tip_offset)
        self.names = self.arm.names
        self.fastik = PlanarIK(self.arm)
        if not self.fastik.ok:
            print("note: closed-form reach check not usable with this URDF - planning uses the numerical IK only "
                  "(slower when a target is out of reach)")
        q = cc.wait_for(lambda: self.node.q(self.names), 10, "/joint_states (is hardware.launch.py running?)")
        T_tf = cc.wait_for(lambda: self.node.tf_ee(self.base, args.ee_frame), 5, f"TF {self.base}->{args.ee_frame}")
        d = np.linalg.norm(self.arm.fk(q)[:3, 3] - T_tf[:3, 3]) * 1000
        print(f"Arm model from the URDF: joints {self.names}; model vs live TF: {d:.2f} mm")
        if d > 2.0:
            raise SystemExit("The URDF model does not match the live TF - check --ee-frame / the base frame.")
        self._consensus_reads()
        self.check_modes()
        mode = self.node.read_dxl_mode(cc.JOINT1_DXL_ID)
        if mode is None:
            self.multiturn = bool(args.joint1_multiturn)
            print(f"joint1 Operating Mode: not readable - using {'multi' if self.multiturn else 'single'}-turn")
        else:
            self.multiturn = mode == 4
            print(f"joint1 Operating Mode: {mode} -> {'multi-turn' if self.multiturn else 'single-turn (+-180 deg)'}")
        q = self.current_q()
        self.window = cc.j1_window(self.multiturn, q[0])
        if args.dry_run:
            print("DRY RUN: torque, joint1 mode and gripper are not touched; nothing moves.")
            if not self.multiturn and not args.keep_single_turn:
                self.multiturn = True                           # the real run switches it: plan as it will be
                self.window = cc.j1_window(True, q[0])
                print("  (planning as if joint1 were multi-turn - the real run switches it to Operating Mode 4)")
        else:
            self.node.send(self.names, q, 0.5)                  # hold the CURRENT pose first: no jump
            time.sleep(0.6)
            msg = self.node.torque_on()
            print(msg if msg is not None else "torque service not found - make sure the torque is ON")
            time.sleep(0.5)
            cc.wait_for(lambda: self.node.q(self.names), 5, "/joint_states")
            if not self.multiturn and not args.keep_single_turn:
                self.switch_joint1_mode4()                       # calib_check: multi-turn for this session
            if not self.node.arm_ac.wait_for_server(timeout_sec=3.0):
                print("action /arm_controller/follow_joint_trajectory not found - trajectories go on the topic")
        q = self.current_q()
        if self.multiturn and JOINT1_SIDE in ("negative", "positive"):
            want_neg = JOINT1_SIDE == "negative"
            if want_neg and q[0] > math.radians(0.5):           # e.g. +179.6 -> -180.4: the same pose
                self.shift_joint1(-2 * math.pi)
            elif not want_neg and q[0] < -math.radians(0.5):
                self.shift_joint1(2 * math.pi)
            q = self.current_q()
            self.window = cc.j1_window(True, q[0])
        lo, hi = self.window
        turn_mode = ("multi-turn" if self.multiturn else
                     "SINGLE-TURN: holes on both sides of 180 deg cannot all be reached")
        print(f"joint1 {math.degrees(q[0]):+.1f} deg, allowed [{math.degrees(lo):+.0f}, {math.degrees(hi):+.0f}] deg "
              f"({turn_mode}); base turns over {cc.MAX_BASE_TURN_DEG:.0f} deg refused")
        if not lo - 0.01 <= q[0] <= hi + 0.01:
            print("WARNING: joint1 is outside its allowed range - moves will be refused. Restart hardware.launch.py "
                  "or change --joint1-limit.")
        self.park = q.copy()
        self.q_cmd = q.copy()
        tip = self.arm.tip(q) * 1000
        print(f"parking pose (returned to after every peg): tip x={tip[0]:.0f} y={tip[1]:.0f} z={tip[2]:.0f} mm")
        self.measure_zone()
        self.dxl, self.dxl_hist = None, deque(maxlen=50)
        self.watch_servos()
        if not args.dry_run:
            self.grip = self.make_gripper()
            self.servo_check()

    def shift_joint1(self, off):
        """Count joint1 'off' rad differently from the servo (one full turn): the same physical pose, so nothing
        moves. Joint readings get +off, every command sent to the arm gets -off. Used to keep joint1 on the
        JOINT1_SIDE side when it reads e.g. +179.6 deg at start instead of -180.4 deg."""
        n = self.node
        oq, osend, ohold, otraj = n.q, n.send, n.hold, n.send_traj
        j = 0                                                   # joint1 is the first of self.names

        def q(names):
            """Joint readings with joint1 shifted by off (None if not available)."""
            v = oq(names)
            if v is None:
                return None
            v = np.array(v, float)
            v[j] += off
            return v

        def to_hw(qq):
            """Joint vector in the shifted count -> the servo's own count."""
            qq = np.array(qq, float)
            qq[j] -= off
            return qq

        def send(names, qq, seconds):
            """Wrapped node.send: joint1 converted back to the hardware count."""
            return osend(names, to_hw(qq), seconds)

        def hold(names, qq):
            """Wrapped node.hold: joint1 converted back to the hardware count."""
            return ohold(names, to_hw(qq))

        def send_traj(names, tr):
            """Wrapped node.send_traj: joint1 of every trajectory point converted back to the hardware count."""
            Q = np.array(tr.Q, float).copy()
            Q[:, j] -= off
            return otraj(names, cc.Traj(t=tr.t, Q=Q, Qd=tr.Qd, Qdd=tr.Qdd))
        n.q, n.send, n.hold, n.send_traj = q, send, hold, send_traj
        raw = oq(self.names)
        if raw is not None:
            print(f"joint1 reads {math.degrees(raw[0]):+.1f} deg at start - counted as "
                  f"{math.degrees(raw[0] + off):+.1f} deg (the same pose, nothing moves) so the reach stays on the "
                  f"{JOINT1_SIDE} side (JOINT1_SIDE)")

    def solve(self, p_mm, pitch, q_ref, chained=False):
        """calib_check's IK, but targets the closed-form check rules out fail at once (not after ~0.5 s)."""
        fk = getattr(self, "fastik", None)
        if fk is not None and not fk.maybe(p_mm, pitch):
            raise cc.PlanError(f"x={p_mm[0]:.1f} y={p_mm[1]:.1f} z={p_mm[2]:.1f} mm with pitch "
                               f"{math.degrees(pitch):.1f} deg is out of reach (arm length / joint limits)")
        try:
            return super().solve(p_mm, pitch, q_ref, chained)
        except cc.PlanError as e:
            if fk is None or not fk.ok or ("out of reach" not in str(e) and "reachable only" not in str(e)):
                raise
            best = None                                   # numerical IK gave up, the closed form has a solution:
            for q in fk.solutions(p_mm, pitch):           # take the one nearest the current joints
                try:
                    q[0] = cc.unwrap_j1(q[0], q_ref[0], self.window)
                except cc.PlanError:
                    continue
                lo, hi = self.j1_lim()
                if lo <= q[0] <= hi:
                    dq = float(np.max(np.abs(q - np.asarray(q_ref, float))))
                    if best is None or dq < best[0]:
                        best = (dq, q)
            if best is None:
                raise
            return best[1]

    def measure_zone(self):
        """Radii (mm from the joint1 axis) where a peg standing on the table (--peg-length) can be picked with its
        shape's taught pitch (+- --pitch-search), grasp height --grasp-z or higher, and carried at the safe height.
        Closed-form, ~1 s. The light-blue zone in the camera window = the radii that work for EVERY shape."""
        a, fk = self.args, getattr(self, "fastik", None)
        self.zone_r, self.zone_shape = REACH_LOW_MM, {}
        if fk is None or not fk.ok:
            return
        top, j1 = float(a.peg_length), self.arm.j1_xy * 1000
        z_pref = min(min(max(a.grasp_z, GRASP_ABOVE_BOTTOM_MM), top - GRASP_BELOW_TOP_MM) + a.pick_raise,
                     top - GRASP_BELOW_TOP_MM)
        zs = [z_pref] + list(np.arange(z_pref + 5.0, top - GRASP_BELOW_TOP_MM, 5.0)) + [top - GRASP_BELOW_TOP_MM]
        ds = [0.0] + [k * x for x in np.arange(1.0, a.pitch_search + 0.01, 1.0) for k in (1, -1)]
        for shape, ref in self.refs.items():
            g_taught = ref.pick_z - a.taught_surface_z
            good = []
            for r in np.arange(60.0, 425.0, 5.0):
                P = np.r_[j1 + np.array([-r, 0.0])]
                ok = False
                for z in zs:
                    drop = ref.place_z + (z - g_taught)
                    zc = max(z + a.hover, top + TOP_CLEAR_MM, TRAY_Z_FALLBACK + z + BOTTOM_CLEAR_MM, drop + a.hover)
                    zp = min(zc, top + PRE_GRASP_MM)
                    for d in ds:
                        pa = ref.pick_pitch + math.radians(d)
                        if all(fk.solutions(np.r_[P, zz], pa) for zz in (z, zp, zc)):
                            ok = True
                            break
                    if ok:
                        break
                if ok:
                    good.append(r)
            self.zone_shape[shape] = (min(good), max(good)) if good else None
        rr = [v for v in self.zone_shape.values() if v is not None]
        if rr:
            self.zone_r = (max(v[0] for v in rr), min(v[1] for v in rr))
        print("pick zone for a peg standing on the table (mm from the base): " +
              ", ".join(f"{k} {v[0]:.0f}-{v[1]:.0f}" if v else f"{k} none" for k, v in self.zone_shape.items()) +
              f"  -> drawn: {self.zone_r[0]:.0f}-{self.zone_r[1]:.0f}")

    def check_modes(self):
        """Joints 2-4 must be in a POSITION mode. In velocity mode (1) a servo ignores position commands: the
        controller 'runs' every move and nothing moves. The gripper's mode is only reported here: the gripper set-up
        (gripper_current_mode) switches it to 5 (or 3) itself."""
        ok_modes = {12: (3, 4, 5), 13: (3, 4, 5), 14: (3, 4, 5)}
        names = {12: "joint2", 13: "joint3", 14: "joint4"}
        bad, seen = [], {}
        for i, okm in ok_modes.items():
            m = self.node.read_dxl_mode(i)
            seen[i] = m
            if m is not None and m not in okm:
                bad.append(f"{names[i]} (ID {i}) is in Operating Mode {m}" + (" = VELOCITY" if m == 1 else ""))
        if bad:
            raise SystemExit("SERVO MODE PROBLEM - these servos ignore position commands, the arm would not move:\n  "
                             + "\n  ".join(bad) + "\nFix:  python3 servo_info.py --fix   (rest the arm on the table "
                             "first), then power-cycle with the gripper half closed and restart hardware.launch.py.")
        g_mode = self.node.read_dxl_mode(GRIPPER_DXL_ID)
        if self.args.dry_run:
            g_txt = f"gripper {'?' if g_mode is None else g_mode} (not touched in the dry run)"
        else:
            want = gripper_mode_wanted(self.args)
            g_txt = f"gripper {'?' if g_mode is None else g_mode}" + ("" if g_mode == want else f" (set to {want} at the gripper set-up)")
        if all(v is not None for v in seen.values()):
            print("servo modes: joint2-4 " + "/".join(str(seen[i]) for i in (12, 13, 14)) + f" - OK, {g_txt}")

    def _consensus_reads(self):
        """The ROBOTIS driver can answer a get_dxl_data request with the value of an EARLIER request for the same
        item (a timed-out request stays in its buffer). Read until two answers in a row agree."""
        raw = getattr(self.node, "read_dxl_item", None)
        if raw is None:
            return

        def read(dxl_id, item, tries=6):
            """Read a Dynamixel item until two answers in a row agree (at most `tries` reads). -> value or None."""
            vals = []
            for _ in range(tries):
                v = raw(dxl_id, item, 1)
                if v is not None:
                    vals.append(v)
                    if len(vals) >= 2 and vals[-1] == vals[-2]:
                        return v
            return vals[-1] if vals else None
        self.node.read_dxl_item = read

    # ---- hardware handles (replaced in tests) -------------------------------------
    def make_gripper(self):
        """Create the gripper (built-in gripper.py) with the command-line settings.

        Checks that its reading is plausible, sets the servo mode / calm settings and finds the open (and,
        for a reversed gripper, the closed) stop. -> Gripper
        """
        gmod = gripper_module()
        a = self.args
        g = gmod.Gripper(self.node, estop=self.grip_estop, open_pos=a.grip_open, close_travel=a.grip_travel,
                    max_squeeze=a.grip_squeeze, effort=a.grip_effort, reverse=(GRIPPER_REVERSED or a.grip_reversed) and not a.grip_normal)
        if not g.wait_ready(5.0):
            raise SystemExit("gripper not ready (/gripper_controller/gripper_cmd or gripper_left_joint on /joint_states)")
        p = g.raw_position()
        print(f"gripper (gripper.py, {'REVERSED - positions mirrored' if g.c['reverse'] else 'normal'}): raw "
              f"gripper_left_joint {p:+.4f} m, light press {g.c['max_squeeze'] * 1000:.1f} mm")
        if p is not None and not GRIP_RANGE_M[0] <= p <= GRIP_RANGE_M[1]:
            raise SystemExit(f"The gripper reads {p:+.4f} m - impossible: the driver range is -0.010 .. +0.019 m. "
                             f"The gripper servo's reading is wrong (or it is in an error state), so open / close "
                             f"cannot be checked. {FIX_SERVOS}")
        self.grip_mode5 = self.gripper_current_mode(g)
        self.calibrate_gripper(g)
        return g

    def gripper_calm(self, n, gid):
        """Write the calm settings GRIP_P_GAIN, GRIP_PROFILE_ACC, GRIP_PROFILE_VEL to the gripper servo (0 = leave
        the servo's own value). -> text for the start-up line, e.g. 'Position P Gain 400, ...'"""
        calm = []
        for item, val in (("Position P Gain", GRIP_P_GAIN), ("Profile Acceleration", GRIP_PROFILE_ACC),
                          ("Profile Velocity", GRIP_PROFILE_VEL)):
            if val > 0:
                n.set_dxl_item(gid, item, int(val))
                time.sleep(0.15)
                calm.append((item, val))
        time.sleep(0.3)
        got = {item: n.read_dxl_item(gid, item) for item, _ in calm}
        return ", ".join(f"{k} {got.get(k)}" + ("" if got.get(k) == v else f" (wanted {v})") for k, v in calm)

    def gripper_torque_on(self, n, gid):
        """Switch the gripper's torque ON (the driver switches it off to write mode / settings). -> Torque Enable
        read back (1 = ON)."""
        tq = None
        for _ in range(2):
            n.set_dxl_item(gid, "Torque Enable", 1)
            time.sleep(0.5)
            tq = n.read_dxl_item(gid, "Torque Enable")
            if tq == 1:
                break
        return tq

    def gripper_position_mode(self, g):
        """Gripper servo -> position mode 3 + the calm settings (fallback when mode 5 is off or cannot be set).
        SystemExit if the gripper is left in a mode that ignores position commands. -> False (not mode 5)"""
        n, gid = self.node, GRIPPER_DXL_ID
        mode = n.read_dxl_mode(gid)
        hold = g.raw_position()
        if hold is not None:
            g._goal_raw(hold)                             # controller holds the fingers where they are
        if mode != 3:
            n.set_dxl_item(gid, "Operating Mode", 3)      # (driver: gripper torque OFF first)
            time.sleep(0.6)
            mode = n.read_dxl_mode(gid)
        calm = self.gripper_calm(n, gid)
        hold = g.raw_position()
        if hold is not None:
            g._goal_raw(hold)
        tq = self.gripper_torque_on(n, gid)
        print(f"gripper servo: mode {mode}" + ("" if mode == 3 else " (NOT 3 - restart hardware.launch.py)") + ", "
              + calm + f", torque {'ON' if tq == 1 else 'NOT confirmed'} - lock = close in steps until the peg stops "
              f"the fingers, hold {min(self.args.grip_squeeze, 0.0025) * 1000:.1f} mm past it")
        if mode is not None and mode not in (3, 4, 5):
            raise SystemExit(f"The gripper servo (ID {gid}) is in Operating Mode {mode} and could not be switched to "
                             f"a position mode - it would ignore every open / close command. {FIX_SERVOS}")
        return False

    def gripper_current_mode(self, g):
        """Gripper servo -> Operating Mode 5 (current-based position), done at every start: Goal Current =
        --grip-current (capped at the servo's Current Limit) + the calm settings, torque back ON.

        Falls back to position mode 3 (gripper_position_mode) when mode 5 or its Goal Current is not confirmed,
        and with --grip-mode 3 / --grip-current 0 / GRIP_MODE5 = False. -> True if mode 5 is set."""
        n, gid, want = self.node, GRIPPER_DXL_ID, int(self.args.grip_current)
        if not hasattr(n, "set_dxl_item"):
            return False
        if gripper_mode_wanted(self.args) != 5:
            return self.gripper_position_mode(g)
        mode0 = mode = n.read_dxl_mode(gid)
        hold = g.raw_position()
        if hold is not None:
            g._goal_raw(hold)                             # controller holds the fingers where they are
        if mode != 5:
            n.set_dxl_item(gid, "Operating Mode", 5)       # (driver: gripper torque OFF first)
            time.sleep(0.6)
            mode = n.read_dxl_mode(gid)
        cur = lim = None
        calm = ""
        if mode == 5:
            lim = n.read_dxl_item(gid, "Current Limit")
            if lim is not None and want > lim:
                want = int(lim)
            n.set_dxl_item(gid, "Goal Current", want)
            time.sleep(0.3)
            cur = n.read_dxl_item(gid, "Goal Current")
            calm = self.gripper_calm(n, gid)
        hold = g.raw_position()
        if hold is not None:
            g._goal_raw(hold)
        tq = self.gripper_torque_on(n, gid)
        if mode == 5:                                     # check again after torque ON
            c2 = n.read_dxl_item(gid, "Goal Current")
            cur = c2 if c2 is not None else cur
        ok = mode == 5 and cur is not None and abs(int(cur) - want) <= 2
        if not ok:
            self.grip_current_set = None
            print(f"gripper servo: could NOT set Operating Mode 5 / Goal Current (mode {mode}, Goal Current {cur}) - "
                  f"switching to position mode 3 instead")
            return self.gripper_position_mode(g)
        self.grip_current_set = cur
        print(f"gripper servo: Operating Mode 5 (current-based position)"
              + ("" if mode0 == 5 else f", switched from mode {mode0}")
              + f", Goal Current {cur} (~{cur * 2.69e-3:.2f} A, limit {lim})" + (f", {calm}" if calm else "")
              + f", torque {'ON' if tq == 1 else 'NOT confirmed'} - lock = close until the peg stops the fingers, "
              f"hold {min(self.args.grip_squeeze, 0.0025) * 1000:.1f} mm past it")
        if tq != 1:
            print("  WARNING: gripper torque not confirmed ON - restart hardware.launch.py if the gripper is limp")
        return True

    def calibrate_gripper(self, g):
        """Nothing is in the gripper at start. REVERSED gripper (gripper.py reverse=True, default): first close to
        the mechanical closed stop and set gripper.py's mirror from it, so 'closed' = -0.010 and commands never push
        past the stop. Then (both kinds) open as far as possible to find the open stop. All positions printed here
        are in gripper.py's normal sense: bigger = more open."""
        gmod = gripper_module()
        c = g.c
        if c.get("reverse"):
            lo, hi_raw = gmod.RAW_RANGE
            top = hi_raw - 0.0005                         # a reversed gripper CLOSES towards a bigger reading
            r0 = r = g.raw_position()
            if r is None:
                raise SystemExit(f"gripper: no position reading. {FIX_SERVOS}")
            note = "no closed stop found (closed fully to the driver limit)"
            while True:                                   # close in 1.5 mm steps: never a hard push on the stop
                cmd = min(top, r + 0.0015)
                st, _ = g._goal_raw(cmd)
                if st == "estop":
                    raise cc.StopMotion
                if st != "done":
                    raise SystemExit(f"gripper: could not close it to find the closed stop ({st}). {FIX_SERVOS}")
                g._wait_still(lambda v, cmd=cmd: abs(g.to_raw(v) - cmd) <= 0.0003)
                q = g.raw_position()
                if q is None:
                    break
                if q < cmd - 0.0005:                      # the fingers stopped short: closed stop
                    r, note = q, "closed stop"
                    break
                r = q
                if cmd >= top - 1e-9:
                    break
            c["mirror"] = r - 0.010
            g._goal_raw(r - 0.0005)                       # do not keep pushing on the stop
            c["close_limit"] = -0.010 + 0.0005            # clamp target: 0.5 mm before the closed stop
            print(f"gripper (REVERSED: a bigger raw reading = more closed): raw {r0:+.4f} -> {r:+.4f} m = {note}; "
                  f"positions are mirrored: position = {c['mirror']:+.4f} - raw")
        # open to the known open position, never pushed into the open stop
        u_max = max(g.from_raw(gmod.RAW_RANGE[0]), g.from_raw(gmod.RAW_RANGE[1]))
        c["open_pos"] = min(float(self.args.grip_open), u_max - 0.0005)
        st, _ = g._goal(c["open_pos"])
        if st == "estop":
            raise cc.StopMotion
        p = g._wait_still(lambda v: abs(v - c["open_pos"]) <= 0.0003)
        if p is not None and p < c["open_pos"] - 0.001:   # stopped short: the open stop is nearer - ease off it
            c["open_pos"] = p - 0.0005
            g._goal(c["open_pos"])
        c["open_ok"] = c["open_pos"] - 0.0025
        print(f"gripper: open {c['open_pos']:+.4f} m (fingers at {'n/a' if p is None else f'{p:+.4f}'} m), clamp "
              f"target {g.close_target:+.4f} m - not pushed into the open stop"
              + (f"  [raw: open {g.to_raw(c['open_pos']):+.4f}, clamp {g.to_raw(g.close_target):+.4f}]"
                 if c.get("reverse") else ""))

    def grip_move(self, pos, what):
        """Plain position command to the gripper; waits until the fingers stop. Never checks a peg, never fails
        (only an emergency stop interrupts it)."""
        g = self.grip
        if not hasattr(g, "_goal"):                       # simple gripper objects (tests): open() / close()
            (g.open if what == "open" else g.close)()
            return g.position()
        st, _ = g._goal(pos)
        if st == "estop":
            raise cc.StopMotion
        p = g._wait_still(lambda v: abs(v - pos) <= 0.0003)
        print(f"      gripper {what}: command {pos:+.4f} m, fingers at "
              f"{'n/a' if p is None else f'{p:+.4f}'} m" + ("" if st == "done" else f"  ({st})"))
        return p

    def grip_lock(self):
        """Lock: close until the peg stops the fingers (any size), then hold the command --grip-squeeze
        (GRIP_PRESS_MM) past the contact point - never 'fully closed' against a peg, so the servo is not driven
        hard for the whole carry. Mode 5: the force is also capped by Goal Current. Position mode (fallback): the
        fingers close in 1.5 mm steps, so the servo never sees a big error = never full power. No check whether a
        peg was caught."""
        g = self.grip
        target = g.close_target
        if not hasattr(g, "_goal"):                         # simple gripper objects (tests)
            self.grip_move(target, "lock")
            return
        press = max(0.0, min(self.args.grip_squeeze, 0.0025))
        mode5 = getattr(self, "grip_mode5", False)
        if mode5:
            p = self.grip_move(target, "lock")
        else:
            p, step = g.position(), 0.0015
            while True:
                cmd = max(target, (g.c["open_pos"] if p is None else p) - step)
                if g._goal(cmd)[0] == "estop":
                    raise cc.StopMotion
                q = g._wait_still(lambda v: abs(v - cmd) <= 0.0003)
                if q is None:
                    break
                blocked = q > cmd + 0.0005
                p = q
                if blocked or cmd <= target + 1e-9:
                    break
            print(f"      gripper lock: closed in steps (position mode), fingers at "
                  f"{'n/a' if p is None else f'{p:+.4f}'} m")
        if p is None:
            return
        if p <= target + 0.001:
            print("      gripper lock: closed fully - nothing between the fingers? (not checked, carrying on)")
            return
        hold = max(target, p - press)
        if g._goal(hold)[0] == "estop":
            raise cc.StopMotion
        q = g._wait_still(lambda v: False)
        cur = getattr(self, "grip_current_set", None)
        print(f"      gripper lock: fingers stopped on the peg at {p:+.4f} m, held {(p - hold) * 1000:.1f} mm past it"
              + (f", force capped by Goal Current {cur} (~{cur * 2.69e-3:.2f} A)" if mode5 and cur else
                 " (position mode)") + f", fingers now {'n/a' if q is None else f'{q:+.4f}'} m")

    # ---- servo status from the ROBOTIS driver (/dynamixel_hardware_interface/dxl_state) -------------------
    def watch_servos(self):
        """Passive: the driver publishes every servo's Hardware Error Status and the bus state. No bus traffic."""
        try:
            from dynamixel_interfaces.msg import DynamixelState
            self.node.create_subscription(DynamixelState, "/dynamixel_hardware_interface/dxl_state",
                                          self._on_dxl, 10)
        except Exception:
            pass                                     # older driver / not installed: the check is skipped

    def _on_dxl(self, m):
        """dxl_state callback: keep the latest (time, comm_state, ids, hw errors, torque) and the comm history."""
        self.dxl = (time.time(), int(m.comm_state), list(m.id), list(m.dxl_hw_state), list(m.torque_state))
        self.dxl_hist.append(int(m.comm_state))

    def servo_problems(self, wait=1.5):
        """-> list of problems ([] = all fine), or None if the driver's status topic is not available."""
        t0 = time.time()
        while (self.dxl is None or time.time() - self.dxl[0] > 1.0) and time.time() - t0 < wait:
            time.sleep(0.05)
        if self.dxl is None or time.time() - self.dxl[0] > 1.0:
            return None
        _, comm, ids, hw, _ = self.dxl
        out = []
        for i, h in zip(ids, hw):
            name = f"servo {i} ({DXL_NAMES.get(i, '?')})"
            if h:
                bits = [txt for b, txt in DXL_HW_BITS.items() if h & b] or [f"code {h}"]
                out.append(f"{name}: hardware error {' + '.join(bits)} - its torque is OFF and it ignores every "
                           f"command until rebooted (LED blinking red)")
        # torque: the topic's torque_state is never filled in by the ROBOTIS driver (always False) - read the servos
        read = getattr(self.node, "read_dxl_item", None)
        if read is not None:
            t_end = time.time() + 8.0                         # never hang here (slow driver / servo in error)
            for i in ids:
                if time.time() > t_end:
                    out.append(f"servo {i} ({DXL_NAMES.get(i, '?')}): torque not checked - the driver answers too "
                               f"slowly (a servo in an error state? see the hardware.launch.py terminal)")
                    continue
                if read(i, "Torque Enable", 2) == 0:
                    out.append(f"servo {i} ({DXL_NAMES.get(i, '?')}): torque OFF (read from the servo)")
        bad = sum(1 for c in self.dxl_hist if c != 0)
        if bad > len(self.dxl_hist) // 5:
            out.append(f"servo bus: {bad} of the last {len(self.dxl_hist)} driver cycles failed (code {comm}) - while "
                       f"reads fail the driver does NOT send the goal positions (U2D2 / cable / power)")
        return out

    def servo_check(self):
        """Start-up check: SystemExit with the fix if any servo reports a problem (see servo_problems)."""
        probs = self.servo_problems()
        if probs is None:
            print("servo status: /dynamixel_hardware_interface/dxl_state not available - not checked")
        elif probs:
            raise SystemExit("SERVO PROBLEM - nothing will move:\n  " + "\n  ".join(probs) + f"\n{FIX_SERVOS}")
        else:
            print("servo status: all servos OK (no hardware error, torque on - read from the servos, bus OK)")

    def make_camera(self):
        """Open the RealSense camera (calib_check.open_camera) and attach the colour intrinsics as cam.intr.

        Recovers on its own when the stream does not start (common on USB 2): first the requested rate again after a
        camera reset, then each rate of CAMERA_FALLBACK_FPS after a reset. SystemExit with what to check if none
        gives frames."""
        import pyrealsense2 as rs
        cc.preflight()
        usb = camera_usb(rs)
        fps = int(self.args.fps)
        if usb.startswith("2") and CAMERA_USB2_FPS and not self.args.fps_given:
            fps = int(CAMERA_USB2_FPS)
        tries = [(fps, False), (fps, True)] + [(f, True) for f in CAMERA_FALLBACK_FPS if f < fps]
        errors = []
        for k, (f, reset) in enumerate(tries):
            if reset:
                print(f"camera: resetting it (like unplugging it) and trying again at {f} fps ...", flush=True)
                if not reset_camera(rs):
                    errors.append("reset: the camera did not come back")
                    continue
            try:
                cam = cc.open_camera(f)                     # calib_camera_node.Camera (restarts after USB hiccups)
            except (SystemExit, RuntimeError) as e:
                msg = str(e.code if isinstance(e, SystemExit) else e).strip().splitlines()
                errors.append(f"{f} fps{' after a reset' if reset else ''}: {msg[0] if msg else type(e).__name__}")
                print(f"camera: no frames at {f} fps{' after a reset' if reset else ''}", flush=True)
                continue
            prof = cam.pipe.get_active_profile()
            cam.intr = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
            self.args.fps = f
            print(f"camera: streaming at {f} fps (USB {usb or '?'})" + ("" if k == 0 else " - recovered"))
            return cam
        raise SystemExit("The RealSense camera gives no frames:\n  " + "\n  ".join(errors) + "\n"
                         "Check: the USB cable and port (no hub; unplug and plug it in again), that no other program "
                         "uses the camera (realsense-viewer, a ROS realsense node, the peg detector), and that "
                         "realsense-viewer shows a picture at 640x480 6 fps.")

    def open_view(self):
        """Open the camera, create Vision and the window. Without a working window (no X display, --no-window, or
        the Docker X crash found by window_works) everything goes to the terminal and pictures to --snapshot."""
        self.cam = self.make_camera()
        self.vision = Vision(self.cam, self.T)
        headless, why = False, ""
        if self.args.no_window:
            headless, why = True, "--no-window"
        else:
            ok, why = window_works()
            headless = not ok
        if not headless:
            try:
                cv2.namedWindow(WIN)
            except Exception as e:                          # camera OK, but no window (X display)
                headless = True
                why = str(e).strip().splitlines()[-1] if str(e).strip() else type(e).__name__
        if headless:
            print(f"no camera WINDOW ({why}) - questions go to this terminal, pictures to {self.args.snapshot}; "
                  f"STOP while moving = Ctrl+C.")
            if "BadAccess" in why or "MIT-SHM" in why:
                print("  (Docker: the container may not share memory with the X display. For the window, start the "
                      "container with  --ipc=host  )")
            elif why != "--no-window":
                print("  (for the window: run   xhost +local:   on the HOST, then start again)")
        self.headless = headless
        self.ui = UI(headless)

    # ---- drawing (MODE 4 look: GREEN = target via calibration, RED = robot tip via calibration) --------------
    def draw(self, color, lines, pegs=None, plan=None):
        """Camera image with the overlay. -> image

        Draws the holes (or the pegs), the END goals (magenta), the pick zone (light blue, only without a plan),
        the pick -> hole arrow of a plan, the current target (green), the robot tip (red) and the text lines.
        """
        v, vis = self.vision, color.copy()
        if v.H is not None:
            if pegs is not None:
                sd.match_and_draw(vis, None, v.slots, pegs, v.H, v.H_inv, geo=v.geo)
            else:
                for s in v.slots:
                    cv2.polylines(vis, [np.round(s["img_poly"]).astype(np.int32)], True, (0, 200, 0), 2)
                    p = s["img_poly"]
                    sd.put_label(vis, f"S{s['id']} {s['shape']}", (int(p[:, 0].max()) + 4, int(p[:, 1].mean())),
                                 0.4, (0, 200, 0))
        for sh in SHAPES:                                   # END goals actually used (offsets included)
            G = self.end_goal(sh)
            uv = None if G is None else v.r2px(G)
            if uv is not None:
                cv2.drawMarker(vis, sd.ip(uv), (255, 0, 255), cv2.MARKER_CROSS, 16, 2)
                sd.put_label(vis, "goal", (int(uv[0]) - 12, int(uv[1]) + 18), 0.35, (255, 0, 255))
        zone = self.pick_zone()
        if zone is not None and plan is None:
            cv2.polylines(vis, [zone], True, (255, 200, 0), 1, cv2.LINE_AA)
            k = int(np.argmin(zone[:, 1]))
            sd.put_label(vis, "pick zone", (int(zone[k, 0]) + 4, int(zone[k, 1]) + 14), 0.4, (255, 200, 0))
        if plan is not None:
            a, b = v.r2px(plan.pick), v.r2px(plan.place)
            if a is not None and b is not None:
                cv2.arrowedLine(vis, sd.ip(a), sd.ip(b), (0, 255, 0), 1)
            for uv in (a, b):
                if uv is not None:
                    cv2.drawMarker(vis, sd.ip(uv), (0, 255, 0), cv2.MARKER_TILTED_CROSS, 22, 2)
        if self.view_target is not None and self.view_target[2] > 0:
            uv = self.cam.project(self.view_target)[0]
            cv2.drawMarker(vis, sd.ip(uv), (0, 255, 0), cv2.MARKER_TILTED_CROSS, 22, 2)
        tip = self.tip_now_mm()
        if tip is not None:
            uv = v.r2px(tip)
            if uv is not None:
                cv2.drawMarker(vis, sd.ip(uv), (0, 0, 255), cv2.MARKER_CROSS, 24, 2)
        return cc.overlay(vis, lines)

    def pick_zone(self):
        """Image outline (px, int32) of where a peg standing on the table can be grasped: REACH_LOW_MM from the
        joint1 axis at the grasp height, inside the joint1 range. Computed once."""
        if getattr(self, "_zone", False) is not False:
            return self._zone
        v = self.vision
        if v is None:
            return None
        self._zone = None
        lo, hi = self.window
        th = np.linspace(max(lo, -2 * math.pi), min(hi, 2 * math.pi), 60)
        j1, z = self.arm.j1_xy * 1000, self.args.grasp_z + self.args.pick_raise
        r_in, r_out = self.zone_r
        ring = [np.r_[j1 + r_out * np.array([math.cos(t), math.sin(t)]), z] for t in th] + \
               [np.r_[j1 + r_in * np.array([math.cos(t), math.sin(t)]), z] for t in th[::-1]]
        uv = [v.r2px(P) for P in ring]
        if any(u is None for u in uv):
            return None
        self._zone = np.round(np.array(uv)).astype(np.int32).reshape(-1, 1, 2)[:, 0]
        return self._zone

    def live(self, lines, plan=None):
        """-> function that draws a fresh camera frame with these lines (used as render for UI.ask)."""
        return lambda: self.draw(self.cam.grab()[0], lines, plan=plan)

    def show(self, cam, aruco, lines, target_cam=None):     # calib_check hook
        """calib_check hook: draw and show one frame while a move runs. -> image"""
        vis = self.draw(cam.grab()[0], [ln for ln in lines if "q = stop" not in ln] +
                        ["SPACE / q = STOP (arm holds),  Ctrl+C in the terminal = emergency stop"])
        self.ui.show(vis)
        return vis

    def _tick(self, cam, aruco, lines, target_cam=None):    # calib_check hook, called while a move runs
        """calib_check hook called during every move: refresh the window; SPACE / q or a stop request -> StopMotion."""
        if not self.ui.headless and cam is not None:
            self.show(cam, aruco, [self.phase] + list(lines))
            if self.ui.poll(1) in ("q", " "):
                raise cc.StopMotion
        else:
            time.sleep(0.02)
        if self.stop_evt.is_set():
            self.stop_evt.clear()
            raise cc.StopMotion

    def snapshot(self, img):
        """Save img to --snapshot (a failure is only printed)."""
        try:
            cv2.imwrite(self.args.snapshot, img)
        except Exception as e:
            print(f"  (could not save {self.args.snapshot}: {e})")

    # =========================== 1. HOLES ===========================
    def phase_holes(self):
        """Phase 1: the user shows the empty tray, the holes are found (Vision.calibrate) and confirmed.
        -> True to go on, False = quit.
        """
        def view():
            """Live picture for the question: the tray outline found in the current frame."""
            color, depth = self.cam.grab()
            c, info = Vision.tray_in(color, None, self.vision.tray_exclude(color, depth))
            vis = color.copy()
            if c is not None:
                cv2.polylines(vis, [np.round(c).astype(np.int32)], True, (0, 255, 0), 2)
            return cc.overlay(vis, ["HOLES: show the EMPTY tray - no pegs on the tray or anywhere in view,",
                                    "arm parked out of the camera's view of the table.   ENTER = find holes,  q = quit",
                                    "tray: " + info])
        while True:
            k = self.ui.ask("\nHOLES: show the EMPTY tray (no pegs on the tray or anywhere in view), arm parked "
                            "out of the camera's view of the table.  ENTER = find the holes, q = quit", "\nq", view)
            if k == "q":
                return False
            ok, msg = self.vision.calibrate(expected={sh: r.hole_xy for sh, r in self.refs.items()})
            print(("  " if ok else "  FAILED: ") + msg)
            if not ok:
                continue
            self.print_holes()
            self.check_park_view()
            keys = "\nrq" + ("h" if "camera" in self.hole_src.values() and self.refs else "")
            k = self.ui.ask("ENTER = use these holes,  r = find them again,  " +
                            ("h = use the RECORDED hole x, y (peg_sort.py) instead,  " if "h" in keys else "") +
                            "q = quit", keys,
                            self.live(["HOLES found (green). ENTER = use them,  r = again,  q = quit" +
                                       (",  h = recorded" if "h" in keys else "")]))
            if k == "h":
                self.args.holes = "recorded"
                self.hole_src = {sh: "recorded" for sh in self.refs}
                print("  using the RECORDED hole positions (x, y from peg_sort.py; camera only for rotation / check)")
            if k in "\nh":
                self.snapshot(self.draw(self.cam.grab()[0], ["holes"]))
                return True
            if k == "q":
                return False

    def fix_hole_labels(self):
        """The camera sometimes calls the circle hole a hexagon and the other way round. The taught holes say where
        each one is: if giving the detected holes the shapes of the nearest taught holes fits much better (30 mm or
        more in total), use those shapes."""
        import itertools
        v = self.vision
        slots = [s for s in v.slots if s["shape"] in self.refs]
        shapes = [sh for sh in SHAPES if sh in self.refs]
        if not slots or len(slots) > len(shapes):
            return

        def dist(s, sh):
            """mm between detected hole s (at its insertion height) and the taught hole of shape sh."""
            return float(np.linalg.norm(v.hole_at_z(s, self.refs[sh].place_z)[:2] - self.refs[sh].hole_xy))
        now = sum(dist(s, s["shape"]) for s in slots)
        best = min(itertools.permutations(shapes, len(slots)),
                   key=lambda perm: sum(dist(s, sh) for s, sh in zip(slots, perm)))
        cost = sum(dist(s, sh) for s, sh in zip(slots, best))
        if now - cost >= 30.0:
            for s, sh in zip(slots, best):
                if s["shape"] != sh:
                    print(f"  NOTE: hole S{s['id']} was classified '{s['shape']}' but sits at the taught {sh} hole -> "
                          f"treated as the {sh} hole (its rotation may be measured poorly)")
                    s["shape"] = sh

    def print_holes(self):
        """Print every hole, camera vs taught position, and decide per shape which x, y is used.

        Fills self.hole_src = {shape: 'camera' | 'recorded'} (from --holes / --hole-warn) and prints the END goals
        with their offsets.
        """
        v, j1 = self.vision, self.arm.j1_xy * 1000
        print(f"\nHOLES  camera -> robot base '{self.base}' (mm), at the insertion height    vs   recorded "
              f"'peg in hole' (peg_sort.py)")
        diffs, mode, self.hole_src = [], self.args.holes, {}
        self.fix_hole_labels()
        for shape in SHAPES:
            s, r = v.slot_for(shape), self.refs.get(shape)
            if s is None:
                if r is not None and mode != "camera":
                    self.hole_src[shape] = "recorded"
                print(f"  {shape:9s} NOT FOUND by the camera" + ("  -> using the RECORDED hole x, y"
                                                                  if shape in self.hole_src else ""))
                continue
            self.hole_src[shape] = "camera"
            line = (f"  S{s['id']} {shape:9s} centre x={s['r_centre'][0]:+7.1f} y={s['r_centre'][1]:+7.1f} "
                    f"surface z={s['r_centre'][2]:5.1f}  yaw {s['r_yaw']:+6.1f}  size {s['size_mm']:.0f} mm")
            if r is not None:
                c = v.hole_at_z(s, r.place_z)
                d = c[:2] - r.hole_xy
                u = (c[:2] - j1) / max(np.linalg.norm(c[:2] - j1), 1e-6)
                diffs.append(d)
                line += (f" | recorded x={r.hole_xy[0]:+7.1f} y={r.hole_xy[1]:+7.1f} -> camera - recorded "
                         f"{np.linalg.norm(d):4.1f} mm (outward {d @ u:+5.1f}, sideways {d @ np.array([-u[1], u[0]]):+5.1f})")
                if r.hole_spread > 8:
                    line += f"  [recording #{r.hole_worst} is {r.hole_spread:.0f} mm off its average: --exclude {r.hole_worst}?]"
                if mode == "recorded" or (mode == "auto" and np.linalg.norm(d) > self.args.hole_warn):
                    self.hole_src[shape] = "recorded"
            line += f"  -> USING {self.hole_src[shape].upper()}"
            print(line)
        if diffs:
            D = np.array(diffs)
            m, worst = D.mean(0), float(np.linalg.norm(D, axis=1).max())
            print(f"  camera - recorded: mean dx={m[0]:+.1f} dy={m[1]:+.1f} mm, worst {worst:.1f} mm")
            print("  END-goal offsets (HOLE_OFFSET_MM + --hole-dx/-dy): " +
                  ", ".join(f"{sh} x {self.hole_offset(sh)[0]:+.1f} y {self.hole_offset(sh)[1]:+.1f}" for sh in SHAPES))
            for sh in SHAPES:
                G = self.end_goal(sh)
                if G is not None:
                    dx, dy = self.hole_offset(sh)
                    print(f"  END GOAL {sh:9s} x={G[0]:+7.1f} y={G[1]:+7.1f}  = {self.hole_src[sh]} x, y "
                          f"{G[0] - dx:+7.1f}, {G[1] - dy:+7.1f}  + offset {dx:+.1f}, {dy:+.1f}   (magenta + in the "
                          f"window)")
            if worst > self.args.hole_warn and mode == "auto":
                print(f"  Holes where camera and recording differ by more than {self.args.hole_warn:.0f} mm use the "
                      f"RECORDED x, y (the robot really put pegs there). The camera is off at the tray because the "
                      f"calibration points only cover the work-station; pegs picked far from it can be off by about as much "
                      f"(recalibrate with points spread over the table and the tray for better picks).")

    def check_park_view(self):
        """Warn if, seen from the camera, the parked arm hides part of the tray or of the pick zone (it is in the
        depth background there, so a peg under it would not be seen). Each point of the arm is followed along its
        camera ray down to the table and to the grasp height."""
        v = self.vision
        _, frames = self.arm.fk(self.park, frames=True)
        pts = [f * 1000 for f in frames] + [self.arm.tip(self.park) * 1000]
        dense = [a + (b - a) * t for a, b in zip(pts[:-1], pts[1:]) for t in np.linspace(0, 1, 8)]
        C = self.T[:3, 3]                                   # camera centre in the robot frame
        lo, hi = self.window
        j1 = self.arm.j1_xy * 1000
        r_in, r_out = self.zone_r
        tray = None if v.corners is None else np.asarray(v.corners, np.float32).reshape(-1, 1, 2)
        W, H = getattr(v.intr, "width", 640), getattr(v.intr, "height", 480)
        for P in dense:
            if P[2] >= C[2] - 1.0:
                continue
            for z0 in (0.0, float(self.args.grasp_z + self.args.pick_raise)):
                if P[2] <= z0:
                    continue
                Q = C + (P - C) * (C[2] - z0) / (C[2] - P[2])
                uv = v.r2px(Q)
                if uv is None or not (0 <= uv[0] < W and 0 <= uv[1] < H):
                    continue                                # outside the camera image: hides nothing it sees
                d = Q[:2] - j1
                r, th = float(np.hypot(*d)), math.atan2(d[1], d[0])
                in_zone = r_in - 20 <= r <= r_out + 20 and any(lo - 0.05 <= th + k * 2 * math.pi <= hi + 0.05
                                                               for k in (-2, -1, 0, 1, 2))
                in_tray = tray is not None and uv is not None and \
                    cv2.pointPolygonTest(tray, (float(uv[0]), float(uv[1])), True) > -10
                if in_zone or in_tray:
                    print(f"  WARNING: seen from the camera, the PARKED arm hides part of the "
                          f"{'tray' if in_tray else 'pick zone'} (around x={Q[0]:.0f} y={Q[1]:.0f}). A peg there would "
                          f"not be seen. Better: quit, jog the arm so the camera sees past it, restart.")
                    return

    # =========================== 2. PEG ===========================
    def wait_for_peg(self):
        """Detect until one peg is ready: known shape, standing still, rotation measured. -> (peg, slot, others)"""
        v = self.vision
        v.tracker = sd.PegTracker()
        hist, seen, said = {}, {}, None
        print("\nPEG: place ONE peg anywhere the camera sees it (not in a hole), take your hand away.  q = quit\n"
              f"     It can be grasped inside the light-blue 'pick zone' in the camera window: about {self.zone_r[0]:.0f}-"
              f"{self.zone_r[1]:.0f} mm from the base, joint1 within [{math.degrees(self.window[0]):+.0f}, "
              f"{math.degrees(self.window[1]):+.0f}] deg.")
        while True:
            color, depth = self.cam.grab()
            pegs, hand = v.detect(color, depth)
            got, status, infos, obst = self.choose(pegs, hand, hist, seen, depth)
            if status != said:
                print(f"  {status}", flush=True)
                said = status
            self.view_target = None
            self.ui.show(self.draw(color, ["PEG: place ONE peg anywhere in view (not in a hole), hand away.",
                                           status, "q = quit"], pegs=pegs))
            if self.ui.poll(1) == "q":
                return None
            if got is not None:
                peg, slot = got
                return peg, slot, [i for i in infos if i is not peg] + obst

    def choose(self, pegs, hand, hist, seen, depth):
        """-> ((peg, slot) or None, status text, free pegs, obstacles = tops of pegs already in holes)

        Placed pegs are never candidates: a peg whose shape is done, or one standing within PLACED_NEAR_MM of a
        placed END goal, is skipped silently (it stays in the returned list, so the arm still moves over it)."""
        v, now, infos, obst, full = self.vision, time.time(), [], [], set()
        for p in pegs:
            xy = v.top_xy(p, depth)
            s_in = None if xy is None else v.slot_at(xy)
            if s_in is not None:                               # already in a hole: not a candidate, an obstacle
                full.add(id(s_in))
                h = v.geo.blob_top(p["silhouette"], depth)
                if h is not None:
                    obst.append({"color": p["color"], "top": v.t2r([xy[0], xy[1], h])})
            elif p["top_ok"] and p.get("top_t") is not None:
                info = v.peg_robot(p)
                info["p"] = p
                infos.append(info)
        status, ready = "waiting for a peg", []
        for i in infos:
            p, col, shape = i["p"], i["color"], i["shape"]
            if any(float(np.linalg.norm(i["top"][:2] - P)) < PLACED_NEAR_MM for P in self.placed.values()):
                continue                               # a peg already placed at its END goal: ignore it
            forced = COLOR_SHAPE.get(col)
            if forced is not None:                     # colour decides the shape at once (COLOR_SHAPE)
                if shape != forced:
                    i["yaw"] = None                    # that rotation was measured for another shape
                i["shape"] = shape = forced
            seen.setdefault(col, now)
            h = hist.setdefault(col, deque(maxlen=STILL_N))
            h.append(i["top"].copy())
            if (forced is None and not p["stable"]) or shape not in SHAPES:
                status = f"{col} peg: deciding the shape (votes {p['votes']})"
                continue
            if self.args.shape and shape != self.args.shape:
                status = f"{col} {shape}: ignored (--shape {self.args.shape})"
                continue
            slot = v.slot_for(shape)
            if slot is None and self.hole_src.get(shape) != "recorded":
                status = f"{col} {shape}: the camera found no {shape} hole (try --holes recorded)"
                continue
            if shape in self.filled:
                continue                               # this shape is done: its peg is ignored
            if slot is not None and id(slot) in full:
                status = f"{col} {shape}: its hole is already filled"
                continue
            H = np.array(h)
            if len(h) < STILL_N or float(np.linalg.norm(H - H.mean(0), axis=1).max()) > STILL_MM:
                status = f"{col} {shape}: hold still ..."
                continue
            if shape != "circle" and i["yaw"] is None and now - seen[col] < YAW_WAIT_S:
                status = f"{col} {shape}: measuring its rotation ..."
                continue
            status = f"{col} {shape} found"
            ready.append((i, slot))
        if hand is not None and hand.any():
            return None, "hand in view - waiting until it is gone", infos, obst
        if not ready:
            return None, status, infos, obst
        order = {c: k for k, c in enumerate(sd.PICK_ORDER)}
        return min(ready, key=lambda r: order.get(r[0]["color"], 99)), status, infos, obst

    # =========================== 3. PLAN ===========================
    def fit_pitch(self, points, pref, q_ref, what):
        """Recorded pitch if every point is reachable with it, else the closest one within --pitch-search."""
        last = None
        steps = np.arange(0.0, self.args.pitch_search + 0.01, 0.5)
        for d in [0.0] + [s * k for s in steps[1:] for k in (1, -1)]:
            a = pref + math.radians(d)
            try:
                for P in points:
                    self.solve(P, a, q_ref)
                return a, d
            except cc.PlanError as e:
                if "seam" in str(e):
                    raise
                last = e
        raise cc.PlanError(f"{what} is not reachable with the gripper pitch within {self.args.pitch_search:.0f} deg of "
                           f"the recorded {math.degrees(pref):+.1f} deg ({last})")

    def peg_height(self, peg):
        """-> (bottom z, grasp z, how) from the camera: the peg top is measured, its length too unless implausible."""
        a = self.args
        top = float(peg["top"][2])
        L, how = float(peg["length"]), "camera"
        if not PEG_LEN_OK[0] <= L <= PEG_LEN_OK[1]:
            L, how = a.peg_length, f"assumed {a.peg_length:.0f} mm (camera said {peg['length']:.0f})"
        bottom = top - L
        lo, hi = bottom + GRASP_ABOVE_BOTTOM_MM, top - GRASP_BELOW_TOP_MM
        z = a.grasp_z if lo <= hi else (lo + hi) / 2.0
        z = min(max(z, lo), hi) if lo <= hi else z
        z = min(z + a.pick_raise, hi) if lo <= hi else z      # --pick-raise: grasp higher on the peg
        return bottom, z, L, how

    def ray_point(self, xy, r, z):
        """Point at distance r from the joint1 axis, on the vertical plane through xy (the arm faces xy)."""
        j1 = self.arm.j1_xy * 1000
        u = np.asarray(xy[:2], float) - j1
        n = float(np.linalg.norm(u))
        u = u / n if n > 1e-6 else np.array([-1.0, 0.0])
        return np.r_[j1 + r * u, z]

    def end_goal(self, shape):
        """Where the peg's axis goes for this hole, on the tray surface: camera or taught x, y + its offset. None if
        the holes are not known yet."""
        v, ref = self.vision, self.refs.get(shape)
        if ref is None or v is None or shape not in getattr(self, "hole_src", {}):
            return None
        slot = v.slot_for(shape)
        z = slot["r_centre"][2] if slot is not None else TRAY_Z_FALLBACK
        if self.hole_src.get(shape) == "recorded" or slot is None:
            P = np.r_[ref.hole_xy, z]
        else:
            P = v.hole_at_z(slot, z)
        dx, dy = self.hole_offset(shape)
        return P + np.array([dx, dy, 0.0])

    def hole_offset(self, shape):
        """(dx, dy) mm added to this shape's END goal: HOLE_OFFSET_MM + --hole-dx / --hole-dy."""
        dx, dy = HOLE_OFFSET_MM.get(shape, (0.0, 0.0))
        return dx + self.args.hole_dx, dy + self.args.hole_dy

    def radius(self, P):
        """Horizontal distance [mm] of point P from the joint1 axis."""
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
        raise cc.PlanError(
            f"{what} is at a base angle of {math.degrees(th):+.0f} deg, outside the range joint1 may turn "
            f"[{math.degrees(lo):+.0f}, {math.degrees(hi):+.0f}] deg (kept on one side of 0 for the cables, as in "
            f"peg_sort.py; --joint1-limit). Move it {math.degrees(gap) + 3:.0f} deg further round the base, e.g. to "
            f"x={sx:+.0f} y={sy:+.0f} mm (same distance from the base)")

    def preview_steps(self, steps, q0):
        """Plan every motion step from q0 -> (trajs, q_end, seconds, joints at keyed steps). Raises PlanError."""
        q, total, trajs, q_at = np.asarray(q0, float).copy(), 0.0, [], {}
        for st in steps:
            tr = None
            if st.kind in ("lift", "line", "joint", "carry"):
                try:
                    tr = self.plan_step(st, q)
                except cc.PlanError as e:
                    raise cc.PlanError(f"step '{st.name}': {e}")
                if tr is not None:
                    q, total = tr.q_end, total + tr.T
            if st.key:
                q_at[st.key] = q.copy()
            trajs.append(tr)
        return trajs, q, total, q_at

    def make_plan(self, peg, slot, others):
        """1-2 camera: start = peg, end = hole.  3 orientation first (face the peg, taught pitch) at a safe height.
        4 straight out to above the peg in that orientation.  5 down to the grasp height (~--grasp-z).
        6 lock (position command).  7 face the hole with the taught place pitch, above it, down, release."""
        a, v, ref = self.args, self.vision, self.refs[peg["shape"]]
        notes, S, slow, name = [], Step, a.slow, f"{peg['color']} {peg['shape']}"
        q0 = self.current_q()
        # --- 1/2: start point = the peg (camera) ---
        top_z = float(peg["top"][2])
        bottom, z_pref, L, how = self.peg_height(peg)
        z_hi = max(z_pref, top_z - GRASP_BELOW_TOP_MM)
        g_taught = ref.pick_z - a.taught_surface_z             # grasp height above the peg bottom when taught
        tray_z = (slot["r_centre"][2] if slot is not None else
                  max((s_["r_centre"][2] for s_ in v.slots), default=TRAY_Z_FALLBACK))
        # grasp height: --grasp-z (clamped to the peg) first; a peg too close to the base for it is grasped
        # higher, up to GRASP_BELOW_TOP_MM under its top (a higher wrist reaches closer in)
        zs = [z_pref] + [z for z in np.arange(z_pref + 5.0, z_hi, 5.0)] + ([z_hi] if z_hi > z_pref + 1 else [])
        first_err, got_h = None, None
        for z_g in zs:
            pick = v.axis_at_z(peg["tray_xy"], z_g)
            g = z_g - bottom                                   # grasp point above the peg bottom
            # --- end point = the hole (camera; taught x, y where the camera is off) ---
            drop_z = ref.place_z + (g - g_taught) + a.drop_raise   # peg bottom as taught (+ --drop-raise)
            if self.hole_src.get(peg["shape"]) == "recorded" or slot is None:
                place = np.r_[ref.hole_xy, drop_z]
            else:
                place = v.hole_at_z(slot, drop_z)
            off = self.hole_offset(peg["shape"])
            place = place + np.array([off[0], off[1], 0.0])     # END-goal offset (HOLE_OFFSET_MM + --hole-dx/-dy)
            # --- safe height: over the peg top, the carried peg's bottom over the tray and other pegs ---
            obst, obst_what = max((s_["r_centre"][2] for s_ in v.slots), default=tray_z), "tray"
            for o in others:
                if seg_dist(o["top"][:2], pick[:2], place[:2]) < PATH_NEAR_MM and o["top"][2] > obst:
                    obst, obst_what = float(o["top"][2]), f"{o['color']} peg"
            z_carry = max(z_g + a.hover, top_z + TOP_CLEAR_MM, obst + g + BOTTOM_CLEAR_MM, drop_z + a.hover)
            z_pre_pick = min(z_carry, max(z_g, top_z + PRE_GRASP_MM))
            z_pre_drop = min(z_carry, max(drop_z + 20.0, tray_z + g + BOTTOM_CLEAR_MM))
            above_pick, pre_pick = np.r_[pick[:2], z_carry], np.r_[pick[:2], z_pre_pick]
            above_hole, pre_drop = np.r_[place[:2], z_carry], np.r_[place[:2], z_pre_drop]
            r_pick = self.radius(pick)
            self.check_base_angle(pick, f"the {name}")
            self.check_base_angle(place, f"the {peg['shape']} hole")
            # --- taught orientation per shape, at most --pitch-search deg off, the same at every stop ---
            try:
                pp, dp = self.fit_pitch([above_pick, pre_pick, pick], ref.pick_pitch, q0, f"the {name}")
                got_h = True
                break
            except cc.PlanError as e:
                if "seam" in str(e) or "no equivalent" in str(e):
                    raise
                if first_err is None:
                    first_err = e
        if got_h is None:
            r0_ = self.radius(v.axis_at_z(peg["tray_xy"], z_pref))
            raise cc.PlanError(f"{first_err}. The peg is {r0_:.0f} mm from the robot base; the arm reaches pegs "
                               f"standing on the table about {self.zone_r[0]:.0f}-{self.zone_r[1]:.0f} mm from it (light-blue "
                               f"pick zone)" + (f" - also tried grasping higher, up to z={z_hi:.0f}" if len(zs) > 1
                                                else "") + " - move the peg into the zone")
        if abs(z_g - (a.grasp_z + a.pick_raise)) > 1.0:
            why = ("the peg is close to the base, grasped higher so the arm reaches it" if z_g > z_pref + 1 else
                   f"the peg stands from z={bottom:.0f} to {top_z:.0f} (fingers stay {GRASP_ABOVE_BOTTOM_MM:.0f} mm "
                   f"above its bottom, {GRASP_BELOW_TOP_MM:.0f} mm below its top)")
            notes.append(f"grasp at z={z_g:.0f} instead of {a.grasp_z + a.pick_raise:.0f}: {why}")
        if obst_what != "tray" and z_carry == obst + g + BOTTOM_CLEAR_MM:
            notes.append(f"safe height raised to z={z_carry:.0f} so the peg clears the {obst_what}")
        pl, dl = self.fit_pitch([above_hole, pre_drop, place], ref.place_pitch, q0, f"the {peg['shape']} hole")
        # --- after the release: straight up until clear of the placed peg and of every other peg seen ---
        placed_top = drop_z - g + L                           # top of the peg once it is released
        tops = [float(o["top"][2]) for o in others] + [placed_top]
        z_ret_want = max(z_carry, max(tops) + a.retreat_clear)
        z_ret = z_carry
        for z in sorted(set([z_ret_want] + list(np.arange(z_carry, z_ret_want, 10.0))), reverse=True):
            try:
                self.solve(np.r_[place[:2], z], pl, q0)       # highest height the arm reaches above the hole
                z_ret = float(z)
                break
            except cc.PlanError:
                continue
        retreat_pt = np.r_[place[:2], z_ret]
        if z_ret < z_ret_want - 1.0:
            notes.append(f"after the release the gripper can only rise to z={z_ret:.0f} here (wanted {z_ret_want:.0f}: "
                         f"{a.retreat_clear:.0f} mm above the highest peg top {max(tops):.0f})")
        for d, w in ((dp, "pick"), (dl, "drop")):
            if d:
                notes.append(f"{w}: taught pitch not reachable here, using {d:+.1f} deg from it")
        tgt = f"S{slot['id']} {slot['shape']}" if slot is not None else f"{peg['shape']} hole (taught)"
        # --- 3/4/5/6 pick, tried from different start positions until IK and the moves work ---
        tip0 = self.arm.tip(q0) * 1000
        r_list = []
        for r in (min(max(self.radius(tip0), START_R[0]), START_R[1]),) + START_R_TRY:
            if all(abs(r - x) > 15 for x in r_list):
                r_list.append(r)
        errs, got = [], None
        for k, r in enumerate(r_list):
            O = self.ray_point(pick, r, z_carry)
            steps_a = [
                S("open the gripper", "open", show=pick),
                S("lift", "lift", O, pp, show=pick),
                S(f"orient: face the {name}, gripper pitch {math.degrees(pp):+.1f} deg", "joint", O, pp,
                  dip="warn", show=pick),
                S(f"over the {name} (same orientation)", "line", above_pick, pp, show=pick),
                S("down to just above the peg", "line", pre_pick, pp, show=pick),
                S("check: above the peg", "pause", show=pick),
                S(f"down to the grasp height z={z_g:.0f}", "line", pick, pp, slow, show=pick, key="pick",
                  hint=" - something is in the way (finger on the peg top?)"),
                S("lock the gripper (position command)", "close", show=pick),
                S("lift the peg", "line", above_pick, pp, show=place),
            ]
            if not a.step:
                steps_a = [x for x in steps_a if x.kind != "pause"]
            try:
                tr_a, q_a, t_a, at_a = self.preview_steps(steps_a, q0)
                got = (r, steps_a, tr_a, q_a, t_a, at_a)
                break
            except cc.PlanError as e:
                errs.append((r, e))
        if got is None:
            raise cc.PlanError(f"no start position works for the {name}: from the current position "
                               f"({errs[0][1]})" + (f"; also tried {len(errs) - 1} other start positions "
                                                    f"(reach-out {', '.join(f'{r:.0f}' for r, _ in errs[1:])} mm)"
                                                    if len(errs) > 1 else ""))
        r0, steps_a, tr_a, q_a, t_a, at_a = got
        if r0 != r_list[0]:
            notes.append(f"start position: the current one could not be solved ({errs[0][1]}), using a reach-out "
                         f"of {r0:.0f} mm instead")
        # --- 7: face the hole with the taught place pitch, above it, down, release ---
        r2_list = []
        for r in (min(max(r_pick, START_R[0]), START_R[1]),) + START_R_TRY:
            if all(abs(r - x) > 15 for x in r2_list):
                r2_list.append(r)
        park_q = self.park.copy()
        tails = ([[S("back to the parking pose (direct)", "joint", q=park_q, dip="error")]] if a.direct_park else []) + \
            [[S("home", "joint", q=np.array(HOME, float), dip="warn"),
              S("back to the parking pose", "joint", q=park_q.copy(), dip="warn")]]
        errs2, got2 = [], None
        for r2 in r2_list:
            O2 = self.ray_point(place, r2, z_carry)
            head = [
                S(f"orient: face {tgt}, taught pitch {math.degrees(pl):+.1f} deg", "joint", O2, pl, show=place),
                S(f"over {tgt} (same orientation)", "line", above_hole, pl, show=place),
                S("down to just above the hole", "line", pre_drop, pl, show=place),
                S("check: above the hole", "pause", show=place),
                S(f"down into {tgt} (slow)", "line", place, pl, slow, show=place, key="place",
                  hint=" - the peg is probably on the edge of the hole (o = let it go here, r = try again)"),
                S("release (gripper open, position command)", "open", show=place),
                S(f"retreat straight up to z={z_ret:.0f} (clear of the pegs)", "line", retreat_pt, pl, show=place),
            ]
            if not a.step:
                head = [x for x in head if x.kind != "pause"]
            for tail in tails:                            # direct to the parking pose first, else via HOME
                steps_b = head + tail
                try:
                    tr_b, q_b, t_b, at_b = self.preview_steps(steps_b, q_a)
                    got2 = (steps_b, tr_b, t_b, at_b)
                    break
                except cc.PlanError as e:
                    errs2.append((r2, e))
            if got2 is not None:
                if r2 != r2_list[0]:
                    notes.append(f"turn to the hole: from the pick it could not be solved ({errs2[0][1]}), going "
                                 f"via a reach-out of {r2:.0f} mm instead")
                break
        if got2 is None:
            raise cc.PlanError(f"the {name} can be picked, but not placed: {errs2[0][1]}")
        steps_b, tr_b, t_b, at_b = got2
        plan = PegPlan(peg, slot, ref, pick, place, pp, pl, z_carry, 0.0, notes=notes)
        plan.steps, plan.trajs, plan.total_s = steps_a + steps_b, tr_a + tr_b, t_a + t_b
        plan.q_at = {**at_a, **at_b}
        plan.info = {"bottom": bottom, "length": L, "len_how": how, "grasp_z": z_g, "g": g, "g_taught": g_taught,
                     "drop_z": drop_z, "tray_z": tray_z, "r0": r0, "r_pick": r_pick}
        # --- rotation: the arm cannot turn the peg; it turns with the base from pick to hole ---
        sym = sd.SYMMETRY_DEG.get(peg["shape"], 360.0)
        if peg["shape"] != "circle" and peg["yaw"] is not None and "pick" in plan.q_at and "place" in plan.q_at:
            dq1 = math.degrees(plan.q_at["place"][0] - plan.q_at["pick"][0])
            if slot is not None:
                plan.turn = wrap_deg(slot["r_yaw"] - (peg["yaw"] + dq1), sym)
            per, off = JAW[peg["shape"]]
            plan.jaw = wrap_deg(peg["yaw"] - (math.degrees(plan.q_at["pick"][0]) + off), per)
        elif peg["shape"] != "circle":
            notes.append("rotation not measured - check by eye that the peg is turned like its hole")
        if peg["shape"] != "circle" and slot is None:
            notes.append(f"the camera did not see the {peg['shape']} hole, so its rotation is unknown - turn the peg "
                         f"like the hole by eye")
        return plan

    def plan_step(self, s, q0):
        """One motion step from joints q0 (used for the preview AND, re-planned from the real joints, for running)."""
        sp = s.speed or self.args.speed
        if s.kind == "lift":                                  # MODE 4: straight up first
            tip = self.arm.tip(q0) * 1000
            if s.target[2] - tip[2] < 1.0:
                return None
            p = np.r_[tip[:2], s.target[2]]
            try:
                return self.plan_line(q0, p, s.pitch, sp, cc.MIN_SEGMENT_T)
            except cc.PlanError:
                try:
                    return self.plan_joint_to(q0, self.solve(p, s.pitch, q0))
                except cc.PlanError as e:
                    if "base would turn" in str(e) or "seam" in str(e):
                        raise
                    return None                               # no lift from this pose: the joint move goes directly
        if s.kind == "line":
            tip = self.arm.tip(q0) * 1000
            if (float(np.linalg.norm(np.asarray(s.target, float) - tip)) < 1.0
                    and abs(self.arm.pitch(q0) - s.pitch) < math.radians(0.5)):
                return None                                   # already there: no move (saves a minimum segment)
            return self.plan_line(q0, s.target, s.pitch, sp, cc.MIN_SEGMENT_T)
        if s.kind == "joint":
            tr = self.plan_joint_to(q0, s.q if s.q is not None else self.solve(s.target, s.pitch, q0))
            self._dip(tr, s)
            return tr
        if s.kind == "carry":                                 # straight line at the carry height ...
            try:
                return self.plan_line(q0, s.target, s.pitch, sp, cc.MIN_SEGMENT_T)
            except cc.PlanError as e:
                if "seam" in str(e):
                    raise
                print(f"  carry: no straight line ({e}) - joint move at the carry height instead")
                tr = self.plan_joint_to(q0, self.solve(s.target, s.pitch, q0))   # ... or a joint move that
                self._dip(tr, s)                                                 # does not sag
                return tr
        return None

    def _dip(self, tr, s):
        """Check a joint-move trajectory: a tip sagging more than DIP_MM below its lower end point raises
        PlanError (s.dip == 'error') or prints a note (s.dip == 'warn').
        """
        if tr is None:
            return
        z = np.array([self.arm.tip(q)[2] for q in list(tr.Q[::4]) + [tr.Q[-1]]]) * 1000
        floor = min(z[0], z[-1]) - DIP_MM
        if z.min() < floor:
            msg = f"'{s.name}': the joint move would sag {min(z[0], z[-1]) - z.min():.0f} mm below its lower end"
            if s.dip == "error":
                raise cc.PlanError(msg)
            print(f"  note: {msg}")

    def preview(self, plan, q0):
        """Plan every step of plan from q0 again; fills plan.trajs, plan.q_at and plan.total_s. Raises PlanError."""
        q, total, plan.trajs = np.asarray(q0, float).copy(), 0.0, []
        for s in plan.steps:
            tr = None
            if s.kind in ("lift", "line", "joint", "carry"):
                try:
                    tr = self.plan_step(s, q)
                except cc.PlanError as e:
                    raise cc.PlanError(f"step '{s.name}': {e}")
                if tr is not None:
                    q, total = tr.q_end, total + tr.T
            if s.key:
                plan.q_at[s.key] = q.copy()
            plan.trajs.append(tr)
        plan.total_s = total

    def describe(self, plan):
        """Print the plan: start / end, grasp and drop points, pitches, joint1 angles, rotation advice, notes
        and the step list with durations.
        """
        p, r, s_, i = plan.peg, plan.ref, plan.slot, plan.info
        print(f"\n==================  {p['color'].upper()} {p['shape'].upper()}  ==================")
        print(f"  START (camera) peg top x={p['top'][0]:+7.1f} y={p['top'][1]:+7.1f} z={p['top'][2]:6.1f} mm, bottom "
              f"z={i['bottom']:5.1f} (length {i['len_how']})" + ("" if p["yaw"] is None else
                                                                   f", turned {p['yaw']:+.1f} deg") +
              f"  (votes {p['votes']}), {i['r_pick']:.0f} mm from the base")
        print("  END   " + (f"S{s_['id']} {s_['shape']}" if s_ is not None else "(hole not seen)") +
              f"  x, y from the {'camera' if self.hole_src.get(p['shape']) == 'camera' and s_ is not None else 'taught pose'}"
              f", tray surface z={i['tray_z']:.0f}"
              + (lambda o: f", offset x {o[0]:+.1f} y {o[1]:+.1f} mm (HOLE_OFFSET_MM)" if abs(o[0]) + abs(o[1]) > 0.05
                 else "")(self.hole_offset(p["shape"])))
        print(f"  grasp x={plan.pick[0]:+7.1f} y={plan.pick[1]:+7.1f} z={plan.pick[2]:6.1f}  pitch "
              f"{math.degrees(plan.pitch_pick):+.1f} deg (taught {r.ws[4]:+.1f})  - {i['g']:.0f} mm above the peg bottom")
        print(f"  drop  x={plan.place[0]:+7.1f} y={plan.place[1]:+7.1f} z={plan.place[2]:6.1f}  pitch "
              f"{math.degrees(plan.pitch_place):+.1f} deg (taught {r.hole[4]:+.1f})  - peg bottom at "
              f"z={plan.place[2] - i['g']:.0f}, "
              + (f"{self.args.drop_raise:+.0f} mm vs taught" if abs(self.args.drop_raise) > 0.5 else "as deep as taught")
              + f"; safe height z={plan.z_carry:.0f}")
        if "pick" in plan.q_at and "place" in plan.q_at:
            print(f"  joint1 {math.degrees(plan.q_at['pick'][0]):+.1f} deg at the peg -> "
                  f"{math.degrees(plan.q_at['place'][0]):+.1f} deg at the hole")
        if plan.jaw is not None:
            print(f"  jaws vs peg faces at the pick: {plan.jaw:+.1f} deg" +
                  ("  (the jaws may turn the peg while closing)" if abs(plan.jaw) > 15 else ""))
        if plan.turn is not None:
            if abs(plan.turn) <= self.args.yaw_tol:
                print(f"  rotation: the peg arrives {plan.turn:+.1f} deg from its hole - OK")
            else:
                print(f"  ROTATION: the peg would arrive {abs(plan.turn):.1f} deg off its hole. Turn it "
                      f"{abs(plan.turn):.0f} deg {'COUNTER-CLOCKWISE' if plan.turn > 0 else 'CLOCKWISE'} (seen from "
                      f"above, as in the camera window), then r")
        for n in plan.notes:
            print(f"  note: {n}")
        print(f"Plan (MODE 4 moves), ~{plan.total_s:.1f} s of motion:")
        for k, (st, tr) in enumerate(zip(plan.steps, plan.trajs), 1):
            extra = f"{tr.T:5.2f} s" if tr is not None else ("" if st.kind in ("open", "close", "pause") else "(there)")
            print(f"  {k:2d}. {st.name:<52s} {extra}")

    def confirm(self, plan):
        """Show the plan in the window (and save a snapshot) and ask what to do.
        -> ENTER = run, 'r' = detect again, 'q' = quit (dry run: only 'r' / 'q').
        """
        p = plan.peg
        lines = [f"{p['color']} {p['shape']} -> " + (f"S{plan.slot['id']}" if plan.slot is not None else "hole") +
                 "   GREEN = pick / hole,  RED = robot tip"]
        if plan.turn is not None and abs(plan.turn) > self.args.yaw_tol:
            lines.append(f"turn the peg {abs(plan.turn):.0f} deg {'counter-clockwise' if plan.turn > 0 else 'clockwise'}"
                         f" (as seen here), then r")
        if self.args.dry_run:
            lines.append("DRY RUN: r = detect again,  q = quit")
            keys = "rq"
        else:
            lines.append("ENTER = RUN,  r = detect again,  q = quit")
            keys = "\nrq"
        self.snapshot(self.draw(self.cam.grab()[0], lines, plan=plan))
        return self.ui.ask("ENTER = run,  r = detect again,  q = quit" if not self.args.dry_run else
                           "DRY RUN - nothing moves.  r = detect again,  q = quit", keys, self.live(lines, plan))

    def auto_ok(self, plan):
        """True if this plan may run without ENTER (AUTO_RUN / --confirm): not a dry run, no --step, and the peg needs
        no turning by hand (rotation known and within --yaw-tol, or a circle)."""
        a, p = self.args, plan.peg
        if a.dry_run or a.step or not a.auto:
            return False
        if p["shape"] == "circle":
            return True
        return plan.turn is not None and abs(plan.turn) <= a.yaw_tol

    # =========================== 4. RUN ===========================
    def run_plan(self, plan):
        """Run plan.steps one by one. -> 'ok' or 'abandoned'

        A failed move (not stopped by the user, not a slow descent with key 'pick' / 'place') is re-planned from
        where the arm is and tried once more (RETRY_MOVES). After a failure or stop the arm holds and the user picks:
        r = resume (re-planned from where the arm is), n = next step, o = open the gripper, x = abandon this peg.
        """
        self.plan, i, n = plan, 0, len(plan.steps)
        retried = set()
        while i < n:
            s = plan.steps[i]
            self.phase = f"[{i + 1}/{n}] {s.name}"
            print(self.phase, flush=True)
            self.view_target = None if s.show is None else self.vision.r2c(s.show)
            user_stop = False
            try:
                ok, why = self.do_step(s)
            except cc.StopMotion:
                ok, why, user_stop = False, "STOPPED - arm holding", True
            if ok:
                i += 1
                continue
            if (RETRY_MOVES and not user_stop and i not in retried and not s.key
                    and s.kind in ("lift", "line", "joint", "carry")):
                retried.add(i)
                self.hold()
                print(f"      {why} - trying once more, re-planned from where the arm is")
                time.sleep(0.3)
                continue
            self.hold()
            print(f"      {why}")
            while True:
                k = self.ui.ask("      r = resume this step (re-planned from where the arm is),  n = carry on with the "
                                "NEXT step (if this was a false alarm),  o = open the gripper,  x = abandon this peg "
                                "(arm holds)", "rnox",
                                self.live([f"{self.phase}: {why}"[:110],
                                           "r = resume,  n = next step,  o = open gripper,  x = abandon"]))
                if k == "o":
                    self.grip_move(self.grip.c["open_pos"], "open")
                    continue
                break
            if k == "x":
                return "abandoned"
            if k == "n":
                print(f"      carrying on after '{s.name}' as if it had succeeded")
                i += 1
            self.stop_evt.clear()
        self.view_target = None
        return "ok"

    def do_step(self, s):
        """Execute one step: gripper, pause, or a motion re-planned from the current joints.
        -> (ok, reason if not ok). Also detects a move the servos did not execute (no joint moved).
        """
        if s.kind == "pause":
            k = self.ui.ask(f"      {s.name}: ENTER = go down,  x = stop here", "\nx",
                            self.live([f"{self.phase}: is the RED cross on the GREEN one?", "ENTER = go down,  x = stop"]))
            return (True, "") if k == "\n" else (False, "stopped before going down")
        if s.kind == "open":
            self.grip_move(self.grip.c["open_pos"], "open")
            return True, ""
        if s.kind == "close":
            self.grip_lock()
            return True, ""
        try:
            tr = self.plan_step(s, self.current_q())
        except cc.PlanError as e:
            return False, f"cannot plan '{s.name}': {e}"
        if tr is None:
            return True, ""                                   # nothing to move (already there)
        q0 = self.current_q()
        ok = self.execute(tr, self.cam, None, s.name)
        if ok or tr is None:
            return ok, ""
        q1 = self.current_q()
        need = float(np.max(np.abs(tr.q_end - q0)))
        moved = float(np.max(np.abs(q1 - q0)))
        if need > math.radians(5) and moved < math.radians(1):
            probs = self.servo_problems(0.5)
            return False, (f"'{s.name}': NO joint moved at all (largest change {math.degrees(moved):.1f} deg of "
                           f"{math.degrees(need):.0f} deg) although the controller ran the move. The servos are not "
                           f"executing commands: torque really off, a servo in error, or bus errors. " +
                           ("Driver says: " + "; ".join(probs) + ". " if probs else
                            "Driver reports no servo error - look at the hardware.launch.py terminal for 'Write Fail' "
                            "/ 'Hardware Error'. " if probs is not None else "") + FIX_SERVOS)
        return False, f"'{s.name}' did not finish{s.hint}"

    def verify(self, plan, frames=45):
        """Camera check after the arm is back in its parking pose: is the peg in its hole?"""
        v, col, pts = self.vision, plan.peg["color"], []
        v.tracker = sd.PegTracker()
        for _ in range(frames):
            color, depth = self.cam.grab()
            pegs, _ = v.detect(color, depth)
            for p in pegs:
                xy = v.top_xy(p, depth) if p["color"] == col else None
                if xy is not None:
                    pts.append(xy)
            if not self.ui.headless:
                self.ui.show(self.draw(color, ["checking the result ..."], pegs=pegs))
                self.ui.poll(1)
        if not pts:
            print(f"  CHECK: the {col} peg is not visible (still in the gripper? fallen?)")
            return False
        xy = np.median(pts, axis=0)
        s = plan.slot if plan.slot is not None else v.slot_at(xy)
        if s is None:
            print(f"  CHECK: {col} peg now at tray x={xy[0]:.0f} y={xy[1]:.0f} mm - no hole there")
            return False
        inside = v.slot_at(xy) is s
        off = float(np.linalg.norm(xy - s["tray"][:2]))
        print(f"  CHECK: {col} {plan.peg['shape']} {'IS IN' if inside else 'is NOT in'} hole S{s['id']} {s['shape']} "
              f"(axis {off:.1f} mm from the hole centre)")
        return inside

    def recover(self):
        """After an abandoned peg: the arm holds where it stopped."""
        while True:
            k = self.ui.ask("Arm holding (torque ON).  ENTER = lift 30 mm, home, back to the parking pose,  "
                            "o = open the gripper,  q = quit (arm keeps holding)", "\noq",
                            self.live(["arm holding.  ENTER = lift + park,  o = open gripper,  q = quit"]))
            if k == "o":
                self.grip_move(self.grip.c["open_pos"], "open")
                continue
            if k == "q":
                return False
            q = self.current_q()
            tip, pitch = self.arm.tip(q) * 1000, self.arm.pitch(q)
            steps = [Step("lift 30 mm", "lift", np.r_[tip[:2], tip[2] + 30.0], pitch),
                     Step("home", "joint", q=np.array(HOME, float), dip="warn"),
                     Step("back to the parking pose", "joint", q=self.park.copy(), dip="warn")]
            res = self.run_plan(PegPlan(None, None, None, tip, tip, pitch, pitch, tip[2], 0.0, steps=steps))
            if res == "ok":
                return True

    # =========================== main loop ===========================
    def run(self):
        """Main loop: holes, then peg after peg (detect, plan, confirm, run, verify) until every hole is filled,
        --once, or quit. -> exit code (0 = done / quit, 1 = cannot plan + quit, 2 = quit while the arm holds).
        """
        self.open_view()
        if not self.phase_holes():
            return 0
        wanted = [s for s in SHAPES if s in self.refs and (not self.args.shape or s == self.args.shape)]
        while True:
            left = [s for s in wanted if s not in self.filled]
            if not left:
                print("\nAll holes are filled. Done.")
                return 0
            got = self.wait_for_peg()
            if got is None:
                return 0
            peg, slot, others = got
            try:
                plan = self.make_plan(peg, slot, others)
            except cc.PlanError as e:
                print(f"\nCANNOT PLAN the {peg['color']} {peg['shape']}: {e}\nNothing moved.")
                k = self.ui.ask("r = detect again (move the peg first),  q = quit", "rq",
                                self.live(["cannot plan - see the terminal.  r = detect again,  q = quit"]))
                if k == "q":
                    return 1
                continue
            self.describe(plan)
            if self.auto_ok(plan):
                k = "\n"
                print("AUTO: running (SPACE / q in the camera window = stop)")
            else:
                k = self.confirm(plan)
            if k == "q":
                return 0
            if k == "r":
                continue
            res = self.run_plan(plan)
            if res == "ok":
                self.filled.add(peg["shape"])                 # done: this shape and its placed peg are ignored now
                self.placed[peg["shape"]] = np.asarray(plan.place[:2], float).copy()
                left_n = len([s for s in wanted if s not in self.filled])
                print(f"\n{peg['color'].capitalize()} {peg['shape']} placed - arm parked. {left_n} to go.")
                if self.args.verify:
                    self.verify(plan)
            elif not self.recover():
                print("Quit. Torque is still ON, the arm is holding.")
                return 2
            if self.args.once:
                return 0

    def emergency_stop(self):
        """Ctrl+C: cancel any gripper goal and stop the arm where it is (calib_check _estop)."""
        self.grip_estop.set()
        try:
            if self.grip is not None:
                self.grip.cancel()
        except Exception:
            pass
        self._estop()

    def shutdown(self):
        """Close the window, the camera and ROS."""
        if self.cam is not None:
            if not self.ui.headless:
                try:
                    cv2.destroyAllWindows()
                except Exception:
                    pass
            try:
                self.cam.stop()
            except Exception:
                pass
        self.close_ros()


# =========================================================================== #
#  gripper test (--grip-test): no arm motion, no camera                        #
# =========================================================================== #
def _s16(v):
    """Unsigned 16-bit servo value -> signed int (None stays None)."""
    if v is None:
        return None
    v = int(v) & 0xFFFF
    return v - 0x10000 if v & 0x8000 else v


def grip_test(args):
    """Gripper bench test with EXACTLY the gripper code of the run. Put a peg between the fingers by hand.
    Reads the gripper servo while it holds: how hard it works (PWM = share of full power) and whether it heads for
    an overload - and finds the strongest press that stays safe."""
    import types
    ns = types.SimpleNamespace(args=args, grip_estop=threading.Event())
    ns.rclpy, ns.node, ns.exe, ns.spin_th = cc.make_ros(args)
    for name in ("_consensus_reads", "gripper_current_mode", "gripper_position_mode", "gripper_calm",
                 "gripper_torque_on", "calibrate_gripper", "grip_move", "grip_lock"):
        setattr(ns, name, types.MethodType(getattr(PegSortRun, name), ns))
    n, gid = ns.node, GRIPPER_DXL_ID

    def rd(item):
        """Read one item of the gripper servo (2 tries)."""
        return n.read_dxl_item(gid, item, 2)

    def status(tag=""):
        """Print one line of gripper servo state (position, PWM share, current, temperature, torque, errors).
        -> (PWM share in % or None, Hardware Error Status).
        """
        hw, pwm, lim = rd("Hardware Error Status"), _s16(rd("Present PWM")), rd("PWM Limit")
        cur, tmp, tq = _s16(rd("Present Current")), rd("Present Temperature"), rd("Torque Enable")
        p = ns.grip.position() if getattr(ns, "grip", None) is not None else None
        share = None if pwm is None or not lim else 100.0 * abs(pwm) / lim
        bits = [t for b, t in DXL_HW_BITS.items() if hw and hw & b]
        print(f"  {tag:8s} fingers {'n/a' if p is None else f'{p:+.4f}'} m | power {'?' if share is None else f'{share:3.0f}%'}"
              f" (PWM {pwm}/{lim}) | current {'?' if cur is None else f'{cur * 2.69:5.0f} mA'} | "
              f"{tmp} C | torque {tq} | error {'none' if not hw else ' + '.join(bits) or hw}")
        return share, hw

    try:
        ns._consensus_reads()
        print("\nGRIPPER TEST - the arm does not move, no camera. Gripper servo ID", gid)
        mode, pg, lim, cl = n.read_dxl_mode(gid), rd("Position P Gain"), rd("PWM Limit"), rd("Current Limit")
        print(f"  servo: Operating Mode {mode}, Position P Gain {pg}, PWM Limit {lim}, Current Limit {cl}")
        share, hw = status("now")
        if hw:
            print("  The gripper servo is in an ERROR state (it ignores every command until power-cycled).\n  " + FIX_SERVOS)
            return 1
        ns.grip = PegSortRun.make_gripper(ns)
        help_ = ("\nkeys + ENTER:  o = open | c = close on the peg exactly like the run | h [s] = hold test, watch "
                 "the servo for s seconds (default 30 = one carry) | p <mm> = press for this test (now "
                 "{:.1f}) | a = auto: find the strongest safe press | s = status | q = quit (opens first)")
        while True:
            print(help_.format(args.grip_squeeze * 1000))
            try:
                cmd = input("grip-test> ").strip().split()
            except EOFError:
                cmd = ["q"]
            if not cmd:
                continue
            k = cmd[0].lower()
            if k == "q":
                break
            if k == "o":
                ns.grip_move(ns.grip.c["open_pos"], "open")
                status("open")
            elif k == "c":
                ns.grip_lock()
                time.sleep(1.0)
                status("holding")
            elif k == "s":
                status("now")
            elif k == "p" and len(cmd) > 1:
                try:
                    args.grip_squeeze = min(0.0025, max(0.0, float(cmd[1]) / 1000.0))
                except ValueError:
                    pass
                print(f"  press for this test: {args.grip_squeeze * 1000:.1f} mm (to keep it: GRIP_PRESS_MM at the top)")
            elif k == "h":
                secs = float(cmd[1]) if len(cmd) > 1 else 30.0
                print(f"  holding for {secs:.0f} s - watch power: below ~60 % is safe for a carry, near 100 % the "
                      f"servo overloads and locks up")
                t0, worst = time.time(), 0.0
                while time.time() - t0 < secs:
                    sh, hw = status(f"{time.time() - t0:4.0f} s")
                    worst = max(worst, sh or 0.0)
                    if hw:
                        print("  OVERLOAD / ERROR - the servo has stopped. " + FIX_SERVOS)
                        return 1
                    time.sleep(1.0)
                print(f"  highest power while holding: {worst:.0f} %  -> "
                      + ("OK for the run" if worst < 60 else "TOO HIGH: lower the press (p 1.5, p 1.0 ...)"))
            elif k == "a":
                print("  peg between the fingers? Trying presses 0.5 .. 2.5 mm, 4 s each (opens in between).")
                best = None
                for mm in (0.5, 1.0, 1.5, 2.0, 2.5):
                    args.grip_squeeze = mm / 1000.0
                    ns.grip_move(ns.grip.c["open_pos"], "open")
                    ns.grip_lock()
                    shares = []
                    for _ in range(4):
                        time.sleep(1.0)
                        sh, hw = status(f"{mm:.1f} mm")
                        if hw:
                            print("  OVERLOAD / ERROR - stopped. " + FIX_SERVOS)
                            return 1
                        shares.append(sh or 0.0)
                    m = max(shares)
                    print(f"  press {mm:.1f} mm -> power up to {m:.0f} %")
                    if m < 60:
                        best = mm
                    else:
                        break
                ns.grip_move(ns.grip.c["open_pos"], "open")
                if best is None:
                    print("  even 0.5 mm drives the servo hard - check the gripper mechanism (binding?) and P gain")
                else:
                    args.grip_squeeze = best / 1000.0
                    print(f"  STRONGEST SAFE PRESS: {best:.1f} mm  ->  set GRIP_PRESS_MM = {best:.1f} at the top of "
                          f"final_run.py (or run with --grip-squeeze {best / 1000:.4f})")
        ns.grip_move(ns.grip.c["open_pos"], "open")
        return 0
    except KeyboardInterrupt:
        print("\nCtrl+C")
        return 130
    except SystemExit as e:
        print(e.code if isinstance(e.code, str) else f"stopped ({e.code})")
        return 1
    finally:
        cc.Checker.close_ros(ns)


# =========================================================================== #
#  main                                                                        #
# =========================================================================== #
def parse_args(argv=None):
    """Command-line options (see --help); unknown arguments are printed and ignored.
    Also sets the fields calib_check.Checker expects (pitch, pitch_range, no_camera). -> argparse.Namespace
    """
    ap = argparse.ArgumentParser(description="Peg sorting final run: camera holes + camera peg -> pick & place "
                                             "(calib_check MODE 4 motion)",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("calib", nargs="?", default=os.path.join(HERE, "omx_calibration.json"),
                    help="calibration file from calib_camera_node.py (BASE_T_CAMERA_mm)")
    ap.add_argument("--samples", default=os.path.join(HERE, "peg_sort.py"),
                    help="peg_sort.py whose SAMPLES table gives the gripper pitch / heights per shape")
    ap.add_argument("--exclude", default="", help="sample IDs left out of the averages, e.g. \"35\" or \"35,22\"")
    ap.add_argument("--shape", choices=SHAPES, default=None, help="only handle pegs of this shape")
    ap.add_argument("--once", action="store_true", help="stop after one peg")
    ap.add_argument("--holes", choices=["auto", "camera", "recorded"], default="recorded",
                    help="hole x, y: recorded (default; the taught peg_sort.py positions - the same every run, so "
                         "HOLE_OFFSET_MM stays valid); auto = camera unless it differs more than --hole-warn from the "
                         "recording (can switch between runs!); camera")
    ap.add_argument("--hole-dx", type=float, default=0.0, metavar="MM",
                    help="added to the x of EVERY end goal, on top of the per-hole HOLE_OFFSET_MM table at the top "
                         "of this file")
    ap.add_argument("--hole-dy", type=float, default=0.0, metavar="MM",
                    help="added to the y of EVERY end goal, on top of HOLE_OFFSET_MM")
    ap.add_argument("--hole-warn", type=float, default=8.0, metavar="MM",
                    help="warn when camera and recorded holes differ more than this")
    ap.add_argument("--dry-run", action="store_true", help="detect and plan only: no torque, nothing moves")
    ap.add_argument("--step", action="store_true", help="stop above the peg and above the hole until ENTER")
    ap.add_argument("--hover", type=float, default=cc.HOVER_MM, metavar="MM",
                    help="carry height above the grasp / insert height (MODE 4 hover)")
    ap.add_argument("--speed", type=float, default=cc.CART_SPEED_MM * SPEED_FACTOR,
                    help="mm/s, lifts and moves over the peg / hole (default calib_check CART_SPEED_MM x SPEED_FACTOR)")
    ap.add_argument("--slow", type=float, default=cc.SLOW_SPEED_MM * SLOW_FACTOR,
                    help="mm/s, last descent onto the peg / into the hole (default SLOW_SPEED_MM x SLOW_FACTOR)")
    ap.add_argument("--confirm", dest="auto", action="store_false", default=AUTO_RUN,
                    help="ask (ENTER) before every peg (default AUTO_RUN at the top: run at once)")
    ap.add_argument("--auto", dest="auto", action="store_true", help="run every planned peg at once (no ENTER)")
    ap.add_argument("--via-home", dest="direct_park", action="store_false", default=DIRECT_PARK,
                    help="go via HOME back to the parking pose after every peg (default DIRECT_PARK at the top)")
    ap.add_argument("--direct-park", dest="direct_park", action="store_true",
                    help="straight back to the parking pose in one joint move (only for a HIGH parking pose)")
    ap.add_argument("--retreat-clear", type=float, default=RETREAT_CLEAR_MM, metavar="MM",
                    help="after the release, rise until this far above the highest peg top (RETREAT_CLEAR_MM)")
    ap.add_argument("--verify", action="store_true", default=VERIFY_AFTER_PLACE,
                    help="camera check after every peg (default VERIFY_AFTER_PLACE at the top)")
    ap.add_argument("--pitch-search", type=float, default=5.0, metavar="DEG",
                    help="gripper pitch may differ this much from the taught one where the taught one cannot reach")
    ap.add_argument("--grasp-z", type=float, default=70.0, metavar="MM",
                    help="grasp height (robot base frame) before --pick-raise; kept 10 mm above the peg bottom")
    ap.add_argument("--pick-raise", type=float, default=PICK_RAISE_MM, metavar="MM",
                    help="grasp this much higher than --grasp-z (default PICK_RAISE_MM at the top of this file), "
                         "but at least 10 mm below the peg top")
    ap.add_argument("--drop-raise", type=float, default=10.0, metavar="MM",
                    help="release the peg this much HIGHER than taught (peg bottom 10 mm less deep in the hole); "
                         "negative = deeper")
    ap.add_argument("--peg-length", type=float, default=100.0, metavar="MM",
                    help="peg length, used when the camera's measurement is not plausible")
    ap.add_argument("--taught-surface-z", type=float, default=81.0, metavar="MM",
                    help="z of the surface the pegs stood on when the peg_sort.py samples were taught (work-station "
                         "block: peg top 180 - 99 mm peg); gives the taught insertion depth")
    ap.add_argument("--yaw-tol", type=float, default=sd.ALIGN_TOL_DEG, metavar="DEG",
                    help="peg rotation error at the hole that is accepted without asking you to turn the peg")
    ap.add_argument("--snapshot", default="peg_plan.png", help="camera picture of the plan / holes")
    ap.add_argument("--no-window", action="store_true",
                    help="no camera window: questions in the terminal, pictures to --snapshot (STOP = Ctrl+C)")
    ap.add_argument("--fps", type=int, choices=[6, 15, 30], default=None,
                    help="camera frame rate (default 15; on USB 2, 6 is the safe choice)")
    ap.add_argument("--ee-frame", default="end_effector_link")
    ap.add_argument("--tip-offset", type=float, nargs=3, default=[0.0, 0.0, 0.0], metavar=("X", "Y", "Z"),
                    help="mm in the ee frame - the SAME as during the calibration and the peg_sort recordings")
    ap.add_argument("--grip-open", type=float, default=0.0118, help="m, gripper.py open position")
    ap.add_argument("--grip-travel", type=float, default=0.030,
                    help="m, finger travel from open when clamping; default = all the way to 0.5 mm before the "
                         "closed stop (the peg stops the fingers first, then the command is eased to it)")
    ap.add_argument("--grip-reversed", action="store_true",
                    help="the gripper reading is REVERSED (bigger reading = more closed): mirror the commands. "
                         "Default: GRIPPER_REVERSED at the top of this file")
    ap.add_argument("--grip-normal", action="store_true",
                    help="the gripper reading is NOT reversed (bigger reading = more open): plain commands. "
                         "Default: GRIPPER_REVERSED at the top of this file")
    ap.add_argument("--grip-squeeze", type=float, default=GRIP_SQUEEZE_MM / 1000.0,
                    help="m, hold the fingers this far past the peg contact (default GRIP_PRESS_MM; max 0.0025)")
    ap.add_argument("--grip-test", action="store_true",
                    help="gripper bench test only (no arm motion, no camera): open / close on a peg by hand, watch "
                         "how hard the servo works while holding, find the strongest safe press")
    ap.add_argument("--grip-mode", type=int, choices=[3, 5], default=5 if GRIP_MODE5 else 3,
                    help="gripper Operating Mode set at start: 5 = current-based position (force capped by "
                         "--grip-current), 3 = position (closes in steps). Default from GRIP_MODE5 at the top")
    ap.add_argument("--grip-current", type=int, default=GRIP_CURRENT,
                    help="gripper Goal Current in mode 5 (2.69 mA units; default GRIP_CURRENT at the top); 0 = use "
                         "position mode 3")
    ap.add_argument("--grip-effort", type=float, default=10.0)
    ap.add_argument("--joint1-limit", type=float, default=210.0, metavar="DEG",
                    help="multi-turn: joint1 stays on one side of 0 within this (peg_sort: -210, cables)")
    ap.add_argument("--max-base-turn", type=float, default=cc.MAX_BASE_TURN_DEG, metavar="DEG")
    ap.add_argument("--keep-single-turn", action="store_true", help="do NOT switch joint1 to Operating Mode 4")
    ap.add_argument("--joint1-multiturn", action="store_true",
                    help="joint1 is multi-turn (only used if its Operating Mode cannot be read)")
    args, unknown = ap.parse_known_args(argv)
    if unknown:
        print(f"(ignored unknown arguments: {unknown})")
    args.fps_given = args.fps is not None
    if args.fps is None:
        args.fps = 15
    args.pitch, args.pitch_range, args.no_camera = None, [-30.0, 30.0], False   # calib_check fields
    return args


VERSION = "2026-11-13e (after the release: up clear of all pegs, back via HOME; window / camera fixes)"


def main(argv=None):
    """Entry point: parse the options, set calib_check's joint1 limits, load the taught references, start the
    run and always shut down cleanly. -> exit code (130 = Ctrl+C).
    """
    args = parse_args(argv)
    print(f"final_run.py version {VERSION}")
    if args.grip_test:
        return grip_test(args)
    cc.MAX_BASE_TURN_DEG = float(args.max_base_turn)
    cc.JOINT1_NEGATIVE_ONLY = True                     # peg_sort: joint1 on one side of 0 (cables)
    cc.JOINT1_MIN_DEG = -abs(float(args.joint1_limit))
    cc.JOINT1_SOFT_LIMIT_DEG = abs(float(args.joint1_limit))
    print(f"speed: lines {args.speed:.0f} mm/s, slow descents {args.slow:.0f} mm/s, "
          f"{'AUTO (no ENTER per peg)' if args.auto else 'ENTER before every peg'}, "
          f"{'direct' if args.direct_park else 'via HOME'} back to the parking pose"
          + (", camera check after each peg" if args.verify else ""))
    refs = shape_refs(tip_samples(load_samples(args.samples), args.tip_offset), parse_ids(args.exclude))
    if any(args.tip_offset):
        print(f"--tip-offset {args.tip_offset}: the taught poses (end_effector_link) are moved to the tip as well, "
              f"so taught holes stay where they physically were")
    if args.exclude:
        print(f"samples left out of the averages: {sorted(parse_ids(args.exclude))}")
    print_refs(refs)
    try:
        run = PegSortRun(args, refs)
    except KeyboardInterrupt:
        print("Ctrl+C during start-up - exiting (if the torque was already on, the arm holds its pose).")
        return 130
    except SystemExit as e:
        print(e.code if isinstance(e.code, str) else f"stopped during start-up ({e.code})")
        return 1
    code = 0
    try:
        code = run.run()
    except KeyboardInterrupt:
        run.emergency_stop()
        print("Ctrl+C - EMERGENCY STOP, arm holding where it is, exiting.")
        code = 130
    except SystemExit as e:                     # camera gone, ... : the arm holds where it is
        run.hold()
        print(e.code if isinstance(e.code, str) else f"stopped ({e.code})")
        code = 1
    finally:
        run.shutdown()
    return code


if __name__ == "__main__":
    rc = main()
    # everything is closed by now; skip the C++ destructors at interpreter exit (rclpy / librealsense can abort
    # there with 'terminate called without an active exception'). The controller keeps holding the arm.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)
