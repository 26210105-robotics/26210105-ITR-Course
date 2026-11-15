#!/usr/bin/env python3
"""
peg_sort.py  -  OpenMANIPULATOR-X, peg & sort, ALL SHAPES  (MoveIt IK)
=====================================================================

Type a shape (circle / hexagon / square / triangle) and the arm takes that peg from
the work-station (WS) to its hole, releases it and returns home.

Targets       Each shape has 3 recorded WS samples and 3 recorded "peg in hole"
              samples (SAMPLES table below, from tele_op.py). The point used is the
              AVERAGE of the three: x, y, z and pitch arithmetic mean, yaw circular
              mean (so +179.9 and -179.1 average to ~180, not 0).
              Leave bad samples out with  -p exclude:="35"  or  -p exclude:="35,22".
              The averaged WS point is where you place the peg before a run.
IK            MoveIt  /compute_ik  (move_group's kinematics plugin, KDL by default),
              collision-aware, called at every waypoint of every Cartesian line.
Checks        MoveIt  /compute_fk  (model vs TF), /check_state_validity (collisions).
Trajectory    rest-to-rest QUINTIC time scaling on every segment, sent straight to
              arm_controller (so SPACE can stop it instantly).

Sequence for one shape (clearance:=0.03):
  1. open gripper
  2. current pose  --joint-space quintic-->  above WS
  3. above WS      --Cartesian quintic-->    WS
  4. close gripper on the peg - gripper.py moves the fingers only gripper_close_travel
     (not all the way) and eases to a light press on the peg, so it never locks up
  5. WS            --Cartesian quintic-->    above WS              (lift)
  6. above WS      --Cartesian quintic-->    above hole            (transfer)
  7. above hole    --Cartesian quintic-->    hole                  (slow insertion)
  8. open gripper (gripper.py, stops just inside the open end stop), CHECKED on
     /joint_states - if the fingers did not open the arm stays in the hole and asks
     (r = retry), it never pulls the peg out

Gripper: gripper.py must sit next to this file. Tune with
  -p gripper_close_travel:=0.010   (finger travel when clamping)
  -p gripper_open_pos:=0.0118      -p gripper_max_squeeze:=0.002
  9. hole          --Cartesian quintic-->    above hole            (retreat)
 10. above hole    --joint-space quintic-->  home

Menu (type, then ENTER):
  circle | hexagon | square | triangle   (or c / h / s / t)   pick & place that peg
  open | close                                                  open / close gripper
  home                                                          arm to home
  list                                                          show averaged targets
  q                                                             quit (torque stays ON)

JOINT1 (base) MODE - read from the joint1 Dynamixel at start-up:
  Operating Mode 4 (multi-turn, recommended): joint1 is kept NEGATIVE, in
      [joint1_min_deg (-210), 0]. Home = -180 deg. Holes are reached at about
      circle -201, hexagon -187, square -173, triangle -159 deg: short base turns.
  Operating Mode 3 (single-turn): joint1 cannot pass +-180 deg, so holes on the
      y>0 side (circle, hexagon) and the y<0 side (square, triangle) cannot be
      served one after the other without turning the base by hand.
No single move may turn the base more than max_base_turn_deg (200).

  SPACE  = EMERGENCY STOP (any time). Arm holds where it is, torque stays ON.
           Then  r = resume (re-plans the interrupted step from where the arm is)
                 x = abandon this run, back to the menu (arm keeps holding)
  Ctrl+C = E-stop + quit

Requires, in two other terminals:
  ros2 launch open_manipulator_x_bringup hardware.launch.py     (controllers, TF)
  ros2 launch ~/colcon_ws/square_moveit.launch.py                (move_group, ROS 2 Humble)

Usage
  python3 peg_sort.py                                         # menu, plan, confirm, run
  python3 peg_sort.py --ros-args -p dry_run:=true             # menu, plan only, no motion
  python3 peg_sort.py --ros-args -p shape:=square             # one square run, then exit
  python3 peg_sort.py --ros-args -p exclude:="35"             # drop sample #35 from the average

ROBOTIS ships kinematics.yaml with  position_only_ik: True, so MoveIt's IK ignores
the gripper pitch. The script detects this and then steers MoveIt to the right
pitch by re-seeding it (pitch = joint2+joint3+joint4 on this arm). For exact pitch
straight from MoveIt, set  position_only_ik: false  in kinematics.yaml.
"""

import math
import os
import queue
import select
import sys
import termios
import threading
import time
import tty
from dataclasses import dataclass, field

import numpy as np

import rclpy
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from rclpy.time import Time

from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import PoseStamped
from moveit_msgs.msg import RobotState
from moveit_msgs.srv import GetPositionFK, GetPositionIK, GetStateValidity
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from rcl_interfaces.srv import GetParameters
from sensor_msgs.msg import JointState
from std_srvs.srv import SetBool
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

import tf2_ros

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gripper import Gripper            # noqa: E402  gripper.py, next to this file


# --------------------------------------------------------------------------- #
#  Recorded samples (tele_op.py)   [x, y, z] m  +  [roll, pitch, yaw] deg      #
#  'ws'   = peg standing on the work-station (grasp pose)                      #
#  'hole' = peg inserted in the hole                                           #
# --------------------------------------------------------------------------- #
SAMPLES = {
    'circle': {
        'ws': {
            16: [-0.1760, +0.0009, +0.1001, 0.00, -0.79, +179.74],
            18: [-0.1705, +0.0067, +0.1011, 0.00, +1.67, +177.89],
            20: [-0.1732, +0.0043, +0.0953, 0.00, +3.08, +178.68],
        },
        'hole': {
            17: [-0.2910, +0.1132, +0.0663, 0.00, +6.42, +159.52],
            19: [-0.2894, +0.1157, +0.0686, 0.00, +4.57, +158.99],
            21: [-0.2917, +0.1161, +0.0673, 0.00, +5.89, +159.08],
        },
    },
    'hexagon': {
        'ws': {
            22: [-0.2032, +0.0013, +0.1009, 0.00, +4.92, +179.65],
            24: [-0.1834, +0.0132, +0.1014, 0.00, +7.38, +176.13],
            26: [-0.1763, +0.0107, +0.0963, 0.00, +11.07, +176.75],
        },
        'hole': {
            23: [-0.2913, +0.0393, +0.0629, 0.00, +10.46, +172.62],
            25: [-0.2847, +0.0384, +0.0750, 0.00, +4.66, +172.62],
            27: [-0.2889, +0.0390, +0.0694, 0.00, +8.35, +172.62],
        },
    },
    'square': {
        'ws': {
            28: [-0.1497, +0.0002, +0.0947, 0.00, -0.62, +179.91],
            30: [-0.1421, +0.0012, +0.0884, 0.00, +9.93, +179.56],
            32: [-0.1544, -0.0026, +0.0910, 0.00, +9.67, -179.12],
        },
        'hole': {
            29: [-0.2971, -0.0386, +0.0671, 0.00, +7.47, -172.88],
            31: [-0.2917, -0.0379, +0.0672, 0.00, +11.87, -172.88],
            33: [-0.2923, -0.0394, +0.0651, 0.00, +13.18, -172.62],
        },
    },
    'triangle': {
        'ws': {
            34: [-0.1658, -0.0022, +0.0939, 0.00, -2.20, -179.30],
            38: [-0.1940, -0.0009, +0.0930, 0.00, -10.02, -179.74],
            40: [-0.1486, -0.0143, +0.0920, 0.00, -6.42, -174.90],
        },
        'hole': {
            35: [-0.3065, -0.1201, +0.0789, 0.00, -3.16, -159.35],
            39: [-0.2839, -0.1152, +0.0733, 0.00, -1.76, -158.73],
            41: [-0.2792, -0.1159, +0.0764, 0.00, -5.19, -158.29],
        },
    },
}
SHAPE_ORDER = ['circle', 'hexagon', 'square', 'triangle']
SHAPE_ALIASES = {
    'c': 'circle', 'circ': 'circle', 'circle': 'circle', 'round': 'circle',
    'h': 'hexagon', 'hex': 'hexagon', 'hexagon': 'hexagon',
    's': 'square', 'sq': 'square', 'square': 'square',
    't': 'triangle', 'tri': 'triangle', 'triangle': 'triangle',
}

MENU = """
Commands (type, then ENTER):
  circle | hexagon | square | triangle   (or c / h / s / t)   pick & place that peg
  open | close                                                  gripper
  home                                                          arm to home
  list                                                          averaged targets
  q                                                             quit
  SPACE = EMERGENCY STOP at any time"""

HOME = [math.pi, -1.0, 0.7, 0.3]      # home facing the work area (joint1 = 180 deg); SRDF shape [rad]
ARM_JOINTS = ['joint1', 'joint2', 'joint3', 'joint4']
GRIPPER_JOINT = 'gripper_left_joint'

def wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def deg(v):
    return np.degrees(np.asarray(v, dtype=float))


def fmt_q(q):
    return '[' + ', '.join(f'{x:+7.2f}' for x in deg(q)) + '] deg'


class PlanError(RuntimeError):
    pass


def quat_from_rpy(r, p, y):
    cr, sr = math.cos(r / 2), math.sin(r / 2)
    cp, sp = math.cos(p / 2), math.sin(p / 2)
    cy, sy = math.cos(y / 2), math.sin(y / 2)
    return (sr * cp * cy - cr * sp * sy,      # x
            cr * sp * cy + sr * cp * sy,      # y
            cr * cp * sy - sr * sp * cy,      # z
            cr * cp * cy + sr * sp * sy)      # w


def rpy_from_quat(x, y, z, w):
    R00 = 1 - 2 * (y * y + z * z)
    R10 = 2 * (x * y + w * z)
    R20 = 2 * (x * z - w * y)
    R21 = 2 * (y * z + w * x)
    R22 = 1 - 2 * (x * x + y * y)
    return np.array([math.atan2(R21, R22), math.atan2(-R20, math.hypot(R00, R10)),
                     math.atan2(R10, R00)])


def pitch_of(q):
    """EE pitch of OpenMANIPULATOR-X: joints 2-4 share the same y axis."""
    return float(q[1] + q[2] + q[3])


def j1_window(P):
    """Allowed joint1 range [lo, hi] in rad."""
    if not P['joint1_multiturn']:
        return -math.pi - 1e-9, math.pi + 1e-9
    if P['joint1_negative_only']:
        return math.radians(P['joint1_min_deg']), 1e-9
    lim = math.radians(P['joint1_soft_limit_deg'])
    return -lim, lim


def unwrap_j1(target, ref, window):
    """joint1 = target + k*360 deg that lies inside window and is closest to ref."""
    lo, hi = window
    cands = [target + k * 2 * math.pi for k in (-2, -1, 0, 1, 2)]
    cands = [c for c in cands if lo - 1e-9 <= c <= hi + 1e-9]
    if not cands:
        raise PlanError(f'joint1 {math.degrees(target):+.1f} deg has no equivalent inside '
                        f'[{math.degrees(lo):+.0f}, {math.degrees(hi):+.0f}] deg '
                        f'(raise joint1_min_deg if the cables allow it)')
    return min(cands, key=lambda c: abs(c - ref))


def pose_to_task(p):
    return np.array([p[0], p[1], p[2], math.radians(p[4])])


# =========================================================================== #
#  Averaging the recorded samples                                              #
# =========================================================================== #
def parse_shape(text):
    return SHAPE_ALIASES.get(str(text).strip().lower())


def parse_ids(v):
    """exclude parameter -> set of sample IDs (accepts 35, "35,22", [35, 22])."""
    if v is None or v == '':
        return set()
    if isinstance(v, bool):
        return set()
    if isinstance(v, int):
        return {v}
    if hasattr(v, '__iter__') and not isinstance(v, str):
        return {int(x) for x in v}
    return {int(t) for t in str(v).replace(';', ',').replace(' ', ',').split(',') if t.strip()}


def average_pose(samples):
    """Best point = mean of the samples. x, y, z, pitch: arithmetic. yaw: circular."""
    A = np.array(list(samples.values()), float)
    xyz = A[:, :3].mean(axis=0)
    pitch = float(A[:, 4].mean())
    yr = np.radians(A[:, 5])
    yaw = math.degrees(math.atan2(float(np.sin(yr).mean()), float(np.cos(yr).mean())))
    return [float(xyz[0]), float(xyz[1]), float(xyz[2]), 0.0, pitch, yaw]


def averaged(P, shape, kind):
    """(average pose, samples used, distance of each sample from the average in mm)."""
    used = {k: v for k, v in SAMPLES[shape][kind].items() if k not in P['exclude_ids']}
    if not used:
        raise PlanError(f'{shape} {kind}: every sample is excluded - nothing to average')
    avg = average_pose(used)
    dev = {k: 1000.0 * float(np.linalg.norm(np.subtract(v[:3], avg[:3]))) for k, v in used.items()}
    return avg, used, dev


def print_targets(P, shapes):
    print('\nAveraged targets  (x, y, z [m], pitch / yaw [deg]):')
    for shape in shapes:
        for kind, label in (('ws', 'WS  '), ('hole', 'hole')):
            avg, used, dev = averaged(P, shape, kind)
            worst = max(dev, key=dev.get)
            ids = '+'.join(f'#{k}' for k in used)
            print(f'  {shape:<8s} {label}  x={avg[0]:+.4f} y={avg[1]:+.4f} z={avg[2]:+.4f}  '
                  f'p={avg[4]:+6.2f} y={avg[5]:+7.2f}   avg of {ids:<14s} '
                  f'spread {dev[worst]:4.1f} mm')
            if kind == 'hole' and len(used) > 1 and dev[worst] > P['spread_warn_mm']:
                print(f'      WARNING: hole sample #{worst} is {dev[worst]:.1f} mm from the average. '
                      f'If it was a bad recording, run with  -p exclude:="{worst}"')


# =========================================================================== #
#  Quintic trajectories                                                        #
# =========================================================================== #
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


# =========================================================================== #
#  Sequence description                                                        #
# =========================================================================== #
@dataclass
class Step:
    name: str
    kind: str                       # 'joint' | 'cart' | 'grip'
    target: object = field(default=None)   # joints / task vector / 'open' | 'close'
    speed: float = 0.0              # Cartesian linear speed [m/s]


# =========================================================================== #
#  Keyboard (E-stop + menu) thread                                             #
# =========================================================================== #
class KeyReader(threading.Thread):
    def __init__(self, on_space):
        super().__init__(daemon=True)
        self.on_space = on_space
        self.keys = queue.Queue()
        self._halt = threading.Event()
        self._fd = sys.stdin.fileno()
        self._old = termios.tcgetattr(self._fd)

    def run(self):
        tty.setcbreak(self._fd)          # keeps Ctrl+C working
        try:
            while not self._halt.is_set():
                r, _, _ = select.select([sys.stdin], [], [], 0.02)
                if not r:
                    continue
                ch = os.read(self._fd, 1).decode(errors='ignore')
                if ch == ' ':
                    self.on_space()
                else:
                    self.keys.put(ch)
        finally:
            self.restore()

    def restore(self):
        try:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)
        except Exception:
            pass

    def stop(self):
        self._halt.set()

    def flush(self):
        while not self.keys.empty():
            self.keys.get_nowait()

    def _get(self):
        while True:                      # short timeouts keep Ctrl+C responsive
            try:
                return self.keys.get(timeout=0.1)
            except queue.Empty:
                continue

    def wait_for(self, allowed):
        self.flush()
        while True:
            ch = self._get()
            if ch in ('\r', '\n'):
                ch = '\n'
            if ch.lower() in allowed:
                return ch.lower()

    def read_line(self, prompt):
        """Line input with echo (the terminal is in cbreak mode; SPACE stays the E-stop)."""
        self.flush()
        sys.stdout.write(prompt)
        sys.stdout.flush()
        buf = ''
        while True:
            ch = self._get()
            if ch in ('\r', '\n'):
                sys.stdout.write('\n')
                sys.stdout.flush()
                return buf.strip()
            if ch in ('\x7f', '\b'):
                if buf:
                    buf = buf[:-1]
                    sys.stdout.write('\b \b')
                    sys.stdout.flush()
                continue
            if ch.isprintable():
                buf += ch
                sys.stdout.write(ch)
                sys.stdout.flush()


# =========================================================================== #
#  ROS 2 node                                                                  #
# =========================================================================== #
class PegSort(Node):
    def __init__(self):
        super().__init__('peg_sort')
        self.P = {}

        def p(name, default):
            self.P[name] = self.declare_parameter(name, default).value
        p('shape', '')                   # circle | hexagon | square | triangle -> one run, no menu
        self.P['exclude'] = self.declare_parameter(
            'exclude', '', ParameterDescriptor(dynamic_typing=True)).value
        p('spread_warn_mm', 8.0)         # warn when a hole sample is this far from the average
        p('seam_turn_deg', 5.0)          # base turn pick->hole when they straddle +-180 deg
        p('max_base_turn_deg', 200.0)    # refuse any single move that turns the base more
        p('joint1_multiturn', False)     # used only if the motor's mode cannot be read
        p('joint1_dxl_id', 11)           # Dynamixel ID of joint1 (read its Operating Mode)
        p('dxl_get_service', '/dynamixel_hardware_interface/get_dxl_data')
        p('joint1_soft_limit_deg', 200.0)  # multiturn, negative_only off: |joint1| <= this
        p('joint1_negative_only', True)  # multiturn: keep joint1 in [joint1_min_deg, 0] always
        p('joint1_min_deg', -210.0)      # most negative joint1 allowed (wires). circle hole ~ -201
        p('home_joints', HOME)
        p('clearance', 0.03)             # m above WS / hole for approach & retreat (0 = off)
        p('grasp_at_start', True)        # close gripper at WS before leaving it
        p('gripper_open_pos', 0.0118)    # m  open, just inside the mechanical open stop (~0.0123)
        p('gripper_close_travel', 0.010) # m  finger travel from open when clamping (not all the way)
        p('gripper_max_squeeze', 0.002)  # m  light press past the peg contact
        p('gripper_effort', 10.0)
        p('max_joint_vel', 0.6)          # rad/s
        p('max_joint_acc', 1.5)          # rad/s^2
        p('cart_speed', 0.04)            # m/s  (peak of the quintic)
        p('insert_speed', 0.015)         # m/s  (descent into the hole)
        p('ang_speed', 0.5)              # rad/s pitch
        p('min_segment_time', 1.0)       # s
        p('dt', 0.02)                    # s between trajectory points
        p('waypoint_step', 0.004)        # m between MoveIt IK calls on Cartesian lines
        p('pitch_tol_deg', 1.0)          # accepted EE pitch error
        p('yaw_warn_deg', 1.5)
        p('tf_check_tol', 0.005)         # m, MoveIt-FK vs TF disagreement that aborts
        p('correct_model_offset', True)  # shift targets if MoveIt's URDF is offset from TF
        p('ik_timeout', 0.05)            # s per MoveIt IK call
        p('check_collisions', True)
        p('dry_run', False)
        p('group', 'arm')
        p('base_frame', 'world')
        p('ee_frame', 'end_effector_link')
        p('joint1_link', 'link2')        # link whose origin lies on the joint1 axis
        p('move_group_node', '/move_group')
        p('arm_action', '/arm_controller/follow_joint_trajectory')
        p('arm_topic', '/arm_controller/joint_trajectory')
        p('gripper_action', '/gripper_controller/gripper_cmd')
        p('torque_service', '/dynamixel_hardware_interface/set_dxl_torque')
        p('joint_states_topic', '/joint_states')

        try:
            self.P['exclude_ids'] = parse_ids(self.P['exclude'])
        except ValueError:
            raise SystemExit(f"exclude:={self.P['exclude']!r} - give sample IDs, e.g. exclude:=\"35,22\"")
        known = {k for s in SAMPLES.values() for kind in s.values() for k in kind}
        unknown = self.P['exclude_ids'] - known
        if unknown:
            raise SystemExit(f'exclude: unknown sample IDs {sorted(unknown)} (known: {sorted(known)})')
        if self.P['shape'] and parse_shape(self.P['shape']) is None:
            raise SystemExit(f"shape:={self.P['shape']} - use one of {SHAPE_ORDER}")

        self.torque_ready = False

        self._js_lock = threading.Lock()
        self._js = {}
        self.create_subscription(JointState, self.P['joint_states_topic'], self._on_js, 20)

        self.arm = ActionClient(self, FollowJointTrajectory, self.P['arm_action'])
        self.arm_pub = self.create_publisher(JointTrajectory, self.P['arm_topic'], 10)
        self.torque_cli = self.create_client(SetBool, self.P['torque_service'])

        self.ik_cli = self.create_client(GetPositionIK, '/compute_ik')
        self.fk_cli = self.create_client(GetPositionFK, '/compute_fk')
        self.valid_cli = self.create_client(GetStateValidity, '/check_state_validity')
        self.param_cli = self.create_client(GetParameters,
                                            self.P['move_group_node'] + '/get_parameters')

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.estop = threading.Event()
        self._goal_lock = threading.Lock()
        self._active = []                 # live arm goal handles

        # gripper driver (gripper.py): open() / close(), shares this node and the E-stop
        self.gripper_dev = Gripper(
            self, estop=self.estop,
            action=self.P['gripper_action'], joint_states=self.P['joint_states_topic'],
            joint=GRIPPER_JOINT, open_pos=float(self.P['gripper_open_pos']),
            close_travel=float(self.P['gripper_close_travel']),
            max_squeeze=float(self.P['gripper_max_squeeze']),
            effort=float(self.P['gripper_effort']))

    # ---------- state -----------------------------------------------------------
    def _on_js(self, msg):
        with self._js_lock:
            for n, v in zip(msg.name, msg.position):
                self._js[n] = v

    def raw_q(self):
        with self._js_lock:
            return np.array([self._js[n] for n in ARM_JOINTS])

    def current_q(self):
        """Joint angles with joint1 wrapped to +-180 deg in single-turn mode. With torque OFF a
        Dynamixel keeps counting past one turn (e.g. -223 deg); turning torque ON resets it to the
        single-turn value (+137 deg). Planning and hold commands always use that value."""
        q = self.raw_q()
        if not self.P['joint1_multiturn']:
            q[0] = wrap(q[0])
        return q

    def current_gripper(self):
        with self._js_lock:
            return self._js.get(GRIPPER_JOINT)

    def wait_joint_states(self, timeout=10.0):
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            with self._js_lock:
                if all(n in self._js for n in ARM_JOINTS):
                    return True
            time.sleep(0.05)
        return False

    def tf_ee(self, timeout=2.0):
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            try:
                tr = self.tf_buffer.lookup_transform(self.P['base_frame'], self.P['ee_frame'], Time())
                v = tr.transform.translation
                return np.array([v.x, v.y, v.z])
            except Exception:
                time.sleep(0.05)
        return None

    def read_dxl(self, dxl_id, item):
        """Read one control-table item of a Dynamixel through the hardware interface, or None."""
        try:
            from dynamixel_interfaces.srv import GetDataFromDxl
        except ImportError:
            return None
        if getattr(self, '_dxl_get_cli', None) is None:
            self._dxl_get_cli = self.create_client(GetDataFromDxl, self.P['dxl_get_service'])
        cli = self._dxl_get_cli
        if not cli.wait_for_service(timeout_sec=2.0):
            return None
        req = GetDataFromDxl.Request()
        req.id = int(dxl_id)
        req.item_name = item
        req.timeout_sec = 1.0
        res = self.call(cli, req, 3.0)
        if res is None or not res.result:
            return None
        return int(res.item_data)

    def read_joint1_mode(self):
        """Operating Mode of the joint1 Dynamixel (3 single-turn, 4 multi-turn) or None."""
        return self.read_dxl(self.P['joint1_dxl_id'], 'Operating Mode')

    # ---------- generic blocking service call (executor spins in another thread) --
    @staticmethod
    def call(cli, req, timeout=2.0):
        fut = cli.call_async(req)
        t0 = time.monotonic()
        while not fut.done():
            if time.monotonic() - t0 > timeout:
                cli.remove_pending_request(fut)
                return None
            time.sleep(0.001)
        return fut.result()

    # ---------- emergency stop -----------------------------------------------
    def _hold_msg(self):
        q = self.current_q()
        pt = JointTrajectoryPoint()
        pt.positions = [float(v) for v in q]
        pt.velocities = [0.0] * 4
        pt.time_from_start = Duration(sec=0, nanosec=100_000_000)
        jt = JointTrajectory()
        jt.joint_names = ARM_JOINTS
        jt.points = [pt]
        return jt

    def send_hold(self):
        """Command the arm to stay exactly where it is now (preempts any running goal)."""
        try:
            jt = self._hold_msg()
            goal = FollowJointTrajectory.Goal()
            goal.trajectory = jt
            self.arm.send_goal_async(goal)
            self.arm_pub.publish(jt)              # backup path through the topic interface
        except Exception as e:                    # never let the E-stop path throw
            self.get_logger().error(f'hold command failed: {e}')

    def trigger_estop(self):
        first = not self.estop.is_set()
        self.estop.set()
        self.send_hold()
        with self._goal_lock:
            for gh in list(self._active):
                try:
                    gh.cancel_goal_async()
                except Exception:
                    pass
        try:
            self.gripper_dev.cancel()
        except Exception:
            pass
        if first:
            print('\n\n  ###########  EMERGENCY STOP  -  arm holding position  ###########\n', flush=True)

    def _wait(self, fut, timeout):
        t0 = time.monotonic()
        while not fut.done():
            if self.estop.is_set() or time.monotonic() - t0 > timeout:
                return False
            time.sleep(0.005)
        return True

    def _track(self, gh, add):
        with self._goal_lock:
            if add:
                self._active.append(gh)
            elif gh in self._active:
                self._active.remove(gh)

    # ---------- torque ------------------------------------------------------------
    def enable_torque(self):
        # Make the controller's command equal to where the arm is now, so the
        # motors do not jump to an old goal when torque comes on.
        self.send_hold()
        time.sleep(0.4)
        if not self.torque_cli.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn(f"torque service {self.P['torque_service']} not found - "
                                   'make sure torque is ON before continuing')
            return
        res = self.call(self.torque_cli, SetBool.Request(data=True), 5.0)
        if res is not None:
            print(f'Torque ON: {res.message or "ok"}', flush=True)
        else:
            self.get_logger().warn('torque service did not answer')
        time.sleep(0.3)

    # ---------- execution ---------------------------------------------------------
    def run_traj(self, traj):
        if self.estop.is_set():
            return 'estop'
        jt = JointTrajectory()
        jt.joint_names = ARM_JOINTS
        for k in range(1, len(traj.t)):              # skip t=0: JTC blends from current state
            pt = JointTrajectoryPoint()
            pt.positions = [float(v) for v in traj.Q[k]]
            pt.velocities = [float(v) for v in traj.Qd[k]]
            pt.accelerations = [float(v) for v in traj.Qdd[k]]
            ns = int(round(float(traj.t[k]) * 1e9))
            pt.time_from_start = Duration(sec=ns // 1_000_000_000, nanosec=ns % 1_000_000_000)
            jt.points.append(pt)
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = jt
        goal.goal_time_tolerance = Duration(sec=1)

        fut = self.arm.send_goal_async(goal)
        if not self._wait(fut, 5.0):
            return 'estop' if self.estop.is_set() else 'error: arm goal not acknowledged'
        gh = fut.result()
        if not gh.accepted:
            return 'error: arm_controller rejected the trajectory'
        self._track(gh, True)
        try:
            if self.estop.is_set():                    # E-stop raced with the send
                gh.cancel_goal_async()
                self.send_hold()
                return 'estop'
            rfut = gh.get_result_async()
            if not self._wait(rfut, traj.T + 10.0):
                if self.estop.is_set():
                    return 'estop'
                self.send_hold()
                return 'error: trajectory timed out'
            res = rfut.result()
        finally:
            self._track(gh, False)
        if self.estop.is_set():
            return 'estop'
        if res.status == GoalStatus.STATUS_SUCCEEDED and res.result.error_code == 0:
            return 'ok'
        return f'error: status={res.status} code={res.result.error_code} {res.result.error_string}'

    # ---------- gripper (gripper.py) ----------------------------------------------
    def gripper_open(self, grasp=False):
        """Open via gripper.py (verified, never pushes on the open stop)."""
        res = self.gripper_dev.open()
        if res.startswith('error') and grasp:
            res += ('\n      The arm stays where it is - retreating now would pull the peg back out. '
                    'r = retry the open.')
        return res

    def gripper_close(self, grasp=False):
        """Clamp via gripper.py (moves only close_travel, presses lightly on the peg)."""
        return self.gripper_dev.close()

    def gripper(self, which, grasp=False):
        return self.gripper_open(grasp) if which == 'open' else self.gripper_close(grasp)


# =========================================================================== #
#  MoveIt kinematics (IK / FK / collision checks through move_group)          #
# =========================================================================== #
IK_ERRORS = {-31: 'NO_IK_SOLUTION', -6: 'TIMED_OUT', -12: 'GOAL_IN_COLLISION',
             -10: 'START_STATE_IN_COLLISION', -21: 'FRAME_TRANSFORM_FAILURE',
             -15: 'INVALID_GROUP_NAME', -16: 'INVALID_GOAL_CONSTRAINTS'}


class MoveItKinematics:
    def __init__(self, node):
        self.node = node
        self.P = node.P
        self.position_only = True
        self.offset = np.zeros(3)         # TF position - MoveIt FK position
        self.j1_xy = np.array([0.012, 0.0])
        self.last_code = None

    # ---------- setup -------------------------------------------------------------
    def wait_ready(self, timeout=15.0):
        for cli, name in ((self.node.ik_cli, '/compute_ik'), (self.node.fk_cli, '/compute_fk')):
            if not cli.wait_for_service(timeout_sec=timeout):
                raise PlanError(
                    f'MoveIt service {name} not available. Start move_group in another terminal:\n'
                    f'  ros2 launch ~/colcon_ws/square_moveit.launch.py')
        if self.P['check_collisions'] and not self.node.valid_cli.wait_for_service(timeout_sec=2.0):
            print('  /check_state_validity not available - joint moves not collision-checked')
            self.P['check_collisions'] = False

    def read_ik_mode(self):
        name = f"robot_description_kinematics.{self.P['group']}.position_only_ik"
        val = None
        if self.node.param_cli.wait_for_service(timeout_sec=2.0):
            res = self.node.call(self.node.param_cli, GetParameters.Request(names=[name]))
            if res is not None and res.values and res.values[0].type == ParameterType.PARAMETER_BOOL:
                val = res.values[0].bool_value
        self.position_only = True if val is None else bool(val)
        src = 'move_group parameter' if val is not None else 'assumed (parameter not readable)'
        return self.position_only, src

    def calibrate(self, q_now, tf_p):
        """Compare MoveIt's robot model with the live TF, find the joint1 axis."""
        p_m, rpy_m = self.fk(q_now)
        p2, _ = self.fk(q_now, self.P['joint1_link'])
        self.j1_xy = p2[:2].copy()
        p_fk = rpy_m[1] if abs(wrap(rpy_m[0])) < math.pi / 2 else wrap(math.pi - rpy_m[1])
        pitch_err = math.degrees(abs(wrap(p_fk - pitch_of(q_now))))
        if pitch_err > 0.5:
            raise PlanError(f'MoveIt FK pitch differs from joint2+joint3+joint4 by {pitch_err:.2f} deg '
                            '- this script assumes the OpenMANIPULATOR-X joint layout')
        if tf_p is None:
            return None
        self.offset = tf_p - p_m
        return float(np.linalg.norm(self.offset))

    # ---------- raw MoveIt calls ------------------------------------------------
    def _state(self, q):
        rs = RobotState()
        rs.joint_state.name = list(ARM_JOINTS)
        q = np.array(q, float)
        q[0] = wrap(q[0])                      # MoveIt's model is single-turn (+-180 deg)
        rs.joint_state.position = [float(v) for v in q]
        rs.is_diff = True
        return rs

    def fk(self, q, link=None):
        req = GetPositionFK.Request()
        req.header.frame_id = self.P['base_frame']
        req.fk_link_names = [link or self.P['ee_frame']]
        req.robot_state = self._state(q)
        res = self.node.call(self.node.fk_cli, req)
        if res is None or res.error_code.val != 1 or not res.pose_stamped:
            raise PlanError(f'MoveIt /compute_fk failed for {req.fk_link_names[0]}')
        ps = res.pose_stamped[0].pose
        o = ps.orientation
        return (np.array([ps.position.x, ps.position.y, ps.position.z]),
                rpy_from_quat(o.x, o.y, o.z, o.w))

    def ik_raw(self, xyz, pitch, yaw, seed):
        req = GetPositionIK.Request()
        r = req.ik_request
        r.group_name = self.P['group']
        r.robot_state = self._state(seed)
        r.avoid_collisions = bool(self.P['check_collisions'])
        r.ik_link_name = self.P['ee_frame']
        ps = PoseStamped()
        ps.header.frame_id = self.P['base_frame']
        ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = (float(v) for v in xyz)
        qx, qy, qz, qw = quat_from_rpy(0.0, pitch, yaw)
        ps.pose.orientation.x, ps.pose.orientation.y = qx, qy
        ps.pose.orientation.z, ps.pose.orientation.w = qz, qw
        r.pose_stamped = ps
        to = float(self.P['ik_timeout'])
        r.timeout = Duration(sec=int(to), nanosec=int((to % 1) * 1e9))
        res = self.node.call(self.node.ik_cli, req, timeout=to + 2.0)
        if res is None:
            self.last_code = 'no answer from move_group'
            return None
        if res.error_code.val != 1:
            self.last_code = IK_ERRORS.get(res.error_code.val, str(res.error_code.val))
            return None
        js = res.solution.joint_state
        d = dict(zip(js.name, js.position))
        try:
            return np.array([d[n] for n in ARM_JOINTS])
        except KeyError:
            self.last_code = 'solution without arm joints'
            return None

    def valid(self, q):
        req = GetStateValidity.Request()
        req.robot_state = self._state(q)
        req.group_name = self.P['group']
        res = self.node.call(self.node.valid_cli, req)
        if res is None:
            return True, ''
        names = sorted({f'{c.contact_body_1}<->{c.contact_body_2}' for c in res.contacts})
        return bool(res.valid), ', '.join(names)

    # ---------- task-level IK ---------------------------------------------------
    def heading(self, xyz):
        return math.atan2(xyz[1] - self.j1_xy[1], xyz[0] - self.j1_xy[0])

    def solve(self, task_tf, q_ref, chained=False):
        """MoveIt IK for [x, y, z, pitch] given in the robot's TF frame."""
        xyz = np.asarray(task_tf[:3], float) - self.offset
        pitch = float(task_tf[3])
        yaw = self.heading(xyz)               # the only yaw a 4-DOF arm can have there
        tol = math.radians(self.P['pitch_tol_deg'])
        q_ref = np.asarray(q_ref, float)

        seeds = [np.r_[yaw, q_ref[1:]]]
        if not chained:
            seeds += [np.r_[yaw, self.P['home_joints'][1:]], np.array([yaw, 0.0, 0.0, 0.0]),
                      np.array([yaw, 0.3, 0.3, 0.0]), np.array([yaw, -0.5, 1.0, -0.4]),
                      np.array([yaw, 0.6, -0.6, 0.8]), np.array([yaw, -0.3, 1.2, -0.9])]
        best, best_err = None, None
        self.last_code = None
        for s in seeds:
            q = self.ik_raw(xyz, pitch, yaw, s)
            if q is None:
                continue
            if self.position_only:
                q = self._steer_pitch(q, xyz, pitch, yaw)
            err = abs(wrap(pitch - pitch_of(q)))
            outward = abs(wrap(q[0] - yaw)) < math.radians(10)
            if not outward:
                continue
            if best_err is None or err < best_err - 1e-4 or (
                    err < tol and best_err < tol and np.linalg.norm(q - q_ref) < np.linalg.norm(best - q_ref)):
                best, best_err = q, err
            if chained and err < tol:
                break
        if best is None:
            raise PlanError(f'MoveIt IK found no solution ({self.last_code}) for '
                            f'x={task_tf[0]:+.4f} y={task_tf[1]:+.4f} z={task_tf[2]:+.4f} '
                            f'pitch={math.degrees(pitch):+.2f} deg')
        if best_err > tol:
            raise PlanError(f'MoveIt IK reaches x={task_tf[0]:+.4f} y={task_tf[1]:+.4f} '
                            f'z={task_tf[2]:+.4f} only with pitch {math.degrees(pitch_of(best)):+.2f} '
                            f'deg instead of {math.degrees(pitch):+.2f} deg (joint limits). '
                            f'Closest: {fmt_q(best)}\n  -> accept that tilt with  -p pitch_tol_deg:='
                            f'{math.ceil(math.degrees(best_err) + 0.5):.0f}.0   or raise the joint limit '
                            f'in MoveIt\'s joint_limits.yaml, or re-record this pose ~1 cm further out')
        return best

    def _steer_pitch(self, q, xyz, pitch, yaw, iters=12):
        """position_only_ik: walk MoveIt's 1-D solution family to the wanted pitch."""
        for _ in range(iters):
            err = wrap(pitch - pitch_of(q))
            if abs(err) < math.radians(0.05):
                break
            seed = q.copy()
            seed[3] += err
            q2 = self.ik_raw(xyz, pitch, yaw, seed)
            if q2 is None or abs(wrap(pitch - pitch_of(q2))) >= abs(err):
                seed = q.copy()                     # retry with the correction spread over 2-4
                seed[1:] += err / 3.0
                q2 = self.ik_raw(xyz, pitch, yaw, seed)
                if q2 is None or abs(wrap(pitch - pitch_of(q2))) >= abs(err):
                    break
            q = q2
        return q

    def task_of(self, q):
        """[x, y, z, pitch] of joint vector q in the robot's TF frame (MoveIt FK)."""
        p, _ = self.fk(q)
        return np.r_[p + self.offset, pitch_of(q)]


# =========================================================================== #
#  Planning                                                                    #
# =========================================================================== #
def plan_cartesian(kin, q0, x_goal, lin_speed, P):
    """Straight line in [x,y,z,pitch]: MoveIt IK at every waypoint, quintic in time."""
    q0 = np.asarray(q0, float)
    x0 = kin.task_of(q0)
    dx = np.asarray(x_goal, float) - x0
    dx[3] = wrap(dx[3])
    L = float(np.linalg.norm(dx[:3]))
    if L < 2e-4 and abs(dx[3]) < math.radians(0.1):
        return None
    n = max(2, int(math.ceil(max(L / P['waypoint_step'], abs(dx[3]) / math.radians(1.0)))))
    Q = [q0]
    q = q0
    for i in range(1, n + 1):
        try:
            q = kin.solve(x0 + dx * (i / n), q, chained=True)
        except PlanError:
            q = kin.solve(x0 + dx * (i / n), q, chained=False)
        if P['joint1_multiturn']:
            q = q.copy()
            q[0] = unwrap_j1(q[0], Q[-1][0], j1_window(P))
        step = np.abs(q - Q[-1])
        if step[0] > math.pi:
            raise PlanError('straight line crosses the joint1 +-180 deg seam behind the robot '
                            '(one pose has y<0, the other y>0) - path refused')
        if float(np.max(step)) > 0.3:
            raise PlanError(f'MoveIt IK jumped {math.degrees(float(np.max(step))):.0f} deg between '
                            f'two waypoints {P["waypoint_step"] * 1000:.0f} mm apart (elbow flip / '
                            f'joint limit) - path not executable as a straight line')
        Q.append(q)
    T0 = max(P['min_segment_time'], 1.875 * L / lin_speed, 1.875 * abs(dx[3]) / P['ang_speed'])
    return retime_quintic(np.array(Q), T0, P['max_joint_vel'], P['max_joint_acc'], P['dt'])


def check_traj(kin, tr, name, every=0.25):
    if not kin.P['check_collisions'] or tr is None:
        return
    k = max(1, int(round(every / kin.P['dt'])))
    for i in list(range(0, len(tr.t), k)) + [len(tr.t) - 1]:
        ok, what = kin.valid(tr.Q[i])
        if not ok:
            raise PlanError(f'{name}: MoveIt reports a collision at t={tr.t[i]:.2f} s '
                            f'{fmt_q(tr.Q[i])}  {what}')


def build_steps(node, kin, q_now):
    """Steps for P['start_pose'] (WS) -> P['end_pose'] (hole), labels in P['start/end_label']."""
    P = node.P
    a, b = P['start_label'], P['end_label']
    c = float(P['clearance'])
    xs, xe = pose_to_task(P['start_pose']), pose_to_task(P['end_pose'])
    up = np.array([0.0, 0.0, c, 0.0])

    q_s = kin.solve(xs, q_now)
    q_e = kin.solve(xe, q_s)
    if P['joint1_multiturn'] and abs(q_e[0] - q_s[0]) > math.pi:
        print(f'\njoint1 multi-turn: {a} -> {b} crosses 180 deg the short way '
              f'({math.degrees(abs(wrap(q_e[0] - q_s[0]))):.1f} deg base turn).')
    elif abs(q_e[0] - q_s[0]) > math.pi:
        # WS and hole straddle the joint1 +-180 deg seam. Never swing round (wires):
        # rotate the pick point about the base axis so it sits seam_turn_deg from the hole,
        # on the hole's side of the seam.
        th_e = q_e[0]
        d = wrap(q_s[0] - th_e)                          # short-way base angle WS - hole
        th_new = th_e + math.copysign(math.radians(P['seam_turn_deg']), d)
        lim = math.pi - math.radians(0.5)
        th_new = max(-lim, min(lim, th_new))
        rot = th_new - (th_e + d)
        c0 = kin.j1_xy + kin.offset[:2]                  # joint1 axis in the TF frame
        v = xs[:2] - c0
        xs = xs.copy()
        xs[:2] = c0 + np.array([math.cos(rot) * v[0] - math.sin(rot) * v[1],
                                math.sin(rot) * v[0] + math.cos(rot) * v[1]])
        shift = float(np.linalg.norm(xs[:2] - np.asarray(P['start_pose'][:2])))
        print(f'\nJOINT1 SEAM: {a} (joint1 {math.degrees(q_s[0]):+.2f}) and {b} (joint1 '
              f'{math.degrees(th_e):+.2f}) are on opposite sides of +-180 deg.')
        print(f'  -> pick point rotated about the base by {math.degrees(rot):+.2f} deg so the base turns '
              f'only {abs(math.degrees(th_new - th_e)):.1f} deg (no long swing).')
        print(f'  -> PICK AT  x={xs[0]:+.4f}  y={xs[1]:+.4f}  z={xs[2]:+.4f}   '
              f'({shift * 1000:.1f} mm from the averaged {a}) - put the peg there.')
        q_s = kin.solve(xs, q_e)
        q_e = kin.solve(xe, q_s)
    P['start_task'] = xs

    steps = []
    if P['grasp_at_start']:
        steps.append(Step('open gripper (ready to grasp)', 'grip', 'open'))
    q_first = kin.solve(xs + up, q_s) if c > 0 else q_s
    if (not P['joint1_multiturn']
            and abs(q_first[0] - q_now[0]) > math.radians(P['max_base_turn_deg'])):
        raise PlanError(
            f'the base is at joint1 {math.degrees(q_now[0]):+.1f} deg, on the other side of the '
            f'+-180 deg seam from {a} ({math.degrees(q_first[0]):+.1f} deg). Reaching it would turn '
            f'the base {abs(math.degrees(q_first[0] - q_now[0])):.0f} deg. Turn torque off, rotate the '
            f'base BY HAND to the {b} side, then run again - or set joint1 to Operating Mode 4 '
            f'(multi-turn) so the script can cross 180 deg directly.')
    if c > 0:
        steps.append(Step(f'approach: current -> above {a}', 'joint', q_first))
        steps.append(Step(f'descend to {a}', 'cart', xs, P['cart_speed']))
    else:
        steps.append(Step(f'current -> {a}', 'joint', q_s))
    if P['grasp_at_start']:
        steps.append(Step('close gripper on peg', 'grip', 'close'))

    if c > 0:
        steps.append(Step(f'lift from {a}', 'cart', xs + up, P['cart_speed']))
        steps.append(Step(f'transfer: above {a} -> above {b}', 'cart', xe + up, P['cart_speed']))
        steps.append(Step(f'insert: descend into {b}', 'cart', xe, P['insert_speed']))
    else:
        steps.append(Step(f'{a} -> {b}', 'cart', xe, P['cart_speed']))
    steps.append(Step('open gripper (release peg)', 'grip', 'open'))
    if c > 0:
        steps.append(Step(f'retreat above {b}', 'cart', xe + up, P['cart_speed']))
    steps.append(Step('go home', 'joint', np.array(P['home_joints'], float)))
    return steps


def plan_step(node, kin, step, q0):
    P = node.P
    if step.kind == 'joint':
        tgt = np.array(step.target, float)
        if P['joint1_multiturn']:
            tgt[0] = unwrap_j1(tgt[0], q0[0], j1_window(P))
        else:
            # +180 and -180 deg are the same direction: take the one on the arm's side, and stay
            # 0.5 deg inside the single-turn range (a goal of exactly 180 deg is out of range)
            tgt[0] = unwrap_j1(tgt[0], q0[0], j1_window(P))
            lim = math.pi - math.radians(0.5)
            tgt[0] = max(-lim, min(lim, tgt[0]))
        turn = abs(float(tgt[0]) - float(q0[0]))
        if turn > math.radians(P['max_base_turn_deg']):
            raise PlanError(f'{step.name}: base would turn {math.degrees(turn):.0f} deg '
                            f'(> max_base_turn_deg {P["max_base_turn_deg"]:.0f}) - refused')
        tr = plan_joint(q0, tgt, P['max_joint_vel'], P['max_joint_acc'],
                        P['min_segment_time'], P['dt'])
        check_traj(kin, tr, step.name)
        return tr
    if step.kind == 'cart':
        return plan_cartesian(kin, q0, step.target, step.speed, P)
    return None


def preview(node, kin, steps, q_now):
    """Plan the whole sequence once (chained) so every IK problem is found before moving."""
    P = node.P
    print('\nTargets (MoveIt IK):')
    xs = P.get('start_task', pose_to_task(P['start_pose']))
    rotated = not np.allclose(xs, pose_to_task(P['start_pose']))
    for name, pose, task in ((P['start_label'] + (' (rotated pick)' if rotated else ''),
                              P['start_pose'], xs),
                             (P['end_label'], P['end_pose'], pose_to_task(P['end_pose']))):
        q = kin.solve(task, q_now)
        p, rpy = kin.fk(q)
        p = p + kin.offset
        r = deg(rpy)
        perr = np.linalg.norm(p - np.asarray(task[:3])) * 1000
        print(f'  {name}: q = {fmt_q(q)}')
        print(f'         FK  x={p[0]:+.4f} y={p[1]:+.4f} z={p[2]:+.4f} ({perr:.2f} mm) | '
              f'rpy r={r[0]:+.2f} p={r[1]:+.2f} y={r[2]:+.2f}')
        dyaw = math.degrees(wrap(math.radians(r[2] - pose[5])))
        if abs(dyaw) > P['yaw_warn_deg'] and not (rotated and task is xs):
            print(f'         WARNING: averaged yaw {pose[5]:+.2f} differs by {dyaw:+.2f} deg '
                  f'(yaw is fixed by x,y on this 4-DOF arm)')

    print('\nPlan:')
    plans = []
    q = q_now.copy()
    total = 0.0
    for i, s in enumerate(steps, 1):
        if s.kind == 'grip':
            g = node.gripper_dev
            pos = g.c['open_pos'] if s.target == 'open' else g.close_target
            print(f'  {i}. {s.name:<46s} gripper -> {pos:+.4f} m', flush=True)
            plans.append(None)
            continue
        tr = plan_step(node, kin, s, q)
        plans.append(tr)
        if tr is None:
            print(f'  {i}. {s.name:<46s} (already there)')
            continue
        total += tr.T
        vpk = deg(np.max(np.abs(tr.Qd)))
        print(f'  {i}. {s.name:<46s} {tr.T:5.2f} s  end {fmt_q(tr.q_end)}  '
              f'peak |qd| {vpk:.0f} deg/s', flush=True)
        dq1 = math.degrees(abs(tr.q_end[0] - q[0]))
        if dq1 > 180:
            print(f'      WARNING: joint1 turns {dq1:.0f} deg in this step - keep the area clear')
        q = tr.q_end
    print(f'  total motion time ~ {total:.1f} s\n')
    return plans


# =========================================================================== #
#  Running                                                                     #
# =========================================================================== #
def ensure_torque(node):
    """Torque ON once per session, then check joint1 is on the expected branch."""
    if node.torque_ready:
        return True
    node.estop.clear()
    node.enable_torque()
    time.sleep(0.3)
    q_on = node.raw_q()
    lo, hi = j1_window(node.P)
    if not (lo - 0.01 <= q_on[0] <= hi + 0.01):
        node.send_hold()
        print(f'joint1 reads {math.degrees(q_on[0]):+.1f} deg after torque ON (outside '
              f'[{math.degrees(lo):+.0f}, {math.degrees(hi):+.0f}]). Not moving.')
        return False
    node.torque_ready = True
    return True


def execute(node, kin, keys, steps, plans):
    """Run the steps. Returns 'ok', or 'abandoned' if the user gave up after a stop/error."""
    i = 0
    while i < len(steps):
        s = steps[i]
        print(f'[{i + 1}/{len(steps)}] {s.name}', flush=True)
        if s.kind == 'grip':
            res = node.gripper(s.target, grasp=True)
        else:
            q_cur = node.current_q()
            tr = plans[i]
            try:
                if tr is None or float(np.max(np.abs(tr.Q[0] - q_cur))) > 0.02:
                    tr = plan_step(node, kin, s, q_cur)     # re-plan from where the arm IS
            except PlanError as e:
                res = f'error: {e}'
            else:
                res = 'ok' if tr is None else node.run_traj(tr)
                if tr is not None and res == 'ok':
                    print(f'      done ({tr.T:.2f} s)', flush=True)
            plans[i] = None                                 # a retry always re-plans

        if res == 'ok':
            i += 1
            continue

        if res != 'estop':
            node.send_hold()
            print(f'      {res}')
        print('      r = resume this step from the current position,  '
              'x = abandon this run (arm keeps holding)', flush=True)
        if keys.wait_for({'r', 'x'}) == 'x':
            print('Run abandoned. Torque is still ON and the arm is holding. '
                  '(Type "home" to return home, "open" to release the peg.)')
            return 'abandoned'
        node.estop.clear()
        print('      resuming ...', flush=True)
    return 'ok'


def confirm(keys, what):
    print(f'Press ENTER to {what}, x to cancel.   (SPACE = EMERGENCY STOP)', flush=True)
    return keys.wait_for({'\n', 'x'}) == '\n'


def run_shape(node, kin, keys, shape):
    """Plan and (after ENTER) execute WS -> hole for one shape, using the averaged points."""
    P = node.P
    ws, _, _ = averaged(P, shape, 'ws')
    hole, _, _ = averaged(P, shape, 'hole')
    P['start_pose'], P['end_pose'] = ws, hole
    P['start_label'], P['end_label'] = f'{shape} WS', f'{shape} hole'
    P.pop('start_task', None)

    print(f'\n==================  {shape.upper()}  ==================')
    print_targets(P, [shape])
    q_now = node.current_q()
    steps = build_steps(node, kin, q_now)
    plans = preview(node, kin, steps, q_now)
    xs = P['start_task']
    print(f'>>> Place the {shape} peg on the work-station at  x={xs[0]:+.4f}  y={xs[1]:+.4f}  '
          f'(grasp height z={xs[2]:+.4f})')
    if P['dry_run']:
        print('dry_run:=true - nothing was sent to the robot.')
        return 'ok'
    if not confirm(keys, f'run the {shape} peg'):
        print('Cancelled, nothing moved.')
        return 'cancelled'
    if not ensure_torque(node):
        return 'error'
    node.estop.clear()
    res = execute(node, kin, keys, steps, plans)
    if res == 'ok':
        print(f'\n{shape.capitalize()} peg placed - arm is home.')
    return res


def go_home(node, kin, keys):
    step = Step('go home', 'joint', np.array(node.P['home_joints'], float))
    tr = plan_step(node, kin, step, node.current_q())
    if tr is None:
        print('  already home.')
        return 'ok'
    print(f'  go home: {tr.T:.2f} s -> {fmt_q(tr.q_end)}')
    if node.P['dry_run']:
        print('dry_run:=true - nothing was sent to the robot.')
        return 'ok'
    if not confirm(keys, 'go home'):
        print('Cancelled, nothing moved.')
        return 'cancelled'
    if not ensure_torque(node):
        return 'error'
    node.estop.clear()
    return execute(node, kin, keys, [step], [tr])


def gripper_cmd(node, which):
    if node.P['dry_run']:
        print(f'dry_run:=true - gripper {which} not sent.')
        return
    if not ensure_torque(node):
        return
    node.estop.clear()
    res = node.gripper(which)
    print(f'  gripper {which}: {res}')


# =========================================================================== #
#  main                                                                        #
# =========================================================================== #
def main(args=None):
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = PegSort()
    ex = MultiThreadedExecutor(num_threads=4)
    ex.add_node(node)
    spin = threading.Thread(target=ex.spin, daemon=True)
    spin.start()

    keys = None
    exit_code = 0
    try:
        if not sys.stdin.isatty():
            print('This script needs an interactive terminal (SPACE is the E-stop).')
            return 1

        if __doc__:                       # None if the docstring was stripped/moved
            print(__doc__.split('JOINT1 (base)')[0])
        else:
            print('peg_sort.py  -  SPACE = EMERGENCY STOP, r = resume, x = abandon run')
        if node.P['exclude_ids']:
            print(f"Samples left out of the averages: {sorted(node.P['exclude_ids'])}")
        print_targets(node.P, SHAPE_ORDER)

        print('\nWaiting for /joint_states ...', flush=True)
        if not node.wait_joint_states():
            print('No /joint_states with joint1..joint4 - is hardware.launch.py running?')
            return 1

        kin = MoveItKinematics(node)
        print('Waiting for MoveIt (move_group) ...', flush=True)
        kin.wait_ready()
        pos_only, src = kin.read_ik_mode()
        if pos_only:
            print(f'MoveIt IK: position_only_ik = True ({src}) -> pitch is steered by re-seeding MoveIt')
        else:
            print(f'MoveIt IK: full pose (position_only_ik = False, {src})')

        mode = node.read_joint1_mode()
        if mode is None:
            print(f"joint1 Operating Mode: not readable - using joint1_multiturn:="
                  f"{str(node.P['joint1_multiturn']).lower()}")
        else:
            node.P['joint1_multiturn'] = (mode == 4)
            print(f"joint1 Operating Mode: {mode} -> "
                  f"{'multi-turn (negative joint1)' if mode == 4 else 'single-turn (+-180 deg)'}")
        g = node.gripper_dev
        print(f'gripper (gripper.py): open {g.c["open_pos"]:+.4f} m, clamp target {g.close_target:+.4f} m '
              f'(travel {g.c["close_travel"] * 1000:.1f} mm), light press {g.c["max_squeeze"] * 1000:.1f} mm')
        if not node.P['joint1_multiturn']:
            print('  NOTE: in single-turn mode the base cannot pass 180 deg. circle/hexagon holes '
                  '(y>0) and square/triangle holes (y<0) are on opposite sides - after one side,\n'
                  '  the other side may be refused. Operating Mode 4 (multi-turn) avoids this.')

        q_now = node.current_q()
        q_raw = node.raw_q()
        if node.P['joint1_multiturn']:
            lo, hi = j1_window(node.P)
            print(f'joint1 multi-turn mode: using the motor reading {math.degrees(q_raw[0]):+.2f} deg '
                  f'as is; joint1 kept in [{math.degrees(lo):+.0f}, {math.degrees(hi):+.0f}] deg')
            if not (lo - math.radians(0.5) <= q_raw[0] <= hi + math.radians(0.5)):
                print(f'\njoint1 reads {math.degrees(q_raw[0]):+.1f} deg, outside that range. The motor '
                      f'counts from where it was at power-on, so it is on the PLUS branch.\n'
                      f'Fix (once per power-on):\n'
                      f'  1. torque off, turn the base by hand a few deg so the gripper is on the\n'
                      f'     y < 0 side (just past 180 deg, toward the square/triangle holes)\n'
                      f'  2. power-cycle the robot (12 V) - the reading restarts negative (~ -175)\n'
                      f'  3. restart hardware.launch.py and run again\n'
                      f'Every run ends at home = -180 deg, so later runs stay negative.')
                return 1
        elif abs(q_raw[0] - q_now[0]) > 1e-6:
            print(f'NOTE: joint1 reads {math.degrees(q_raw[0]):+.2f} deg (multi-turn count, torque off). '
                  f'The motor resets it to {math.degrees(q_now[0]):+.2f} deg when torque comes on - '
                  f'planning with that.')
        tf_p = node.tf_ee()
        off = kin.calibrate(q_now, tf_p)
        if off is None:
            node.get_logger().warn('TF not available - MoveIt model not cross-checked')
        else:
            print(f'MoveIt FK vs TF {node.P["base_frame"]}->{node.P["ee_frame"]}: {off * 1000:.2f} mm '
                  f'(dx={kin.offset[0] * 1000:+.1f} dy={kin.offset[1] * 1000:+.1f} '
                  f'dz={kin.offset[2] * 1000:+.1f} mm)')
            if off > node.P['tf_check_tol']:
                if node.P['correct_model_offset'] and off < 0.03:
                    print('  MoveIt\'s URDF is offset from the robot\'s TF - targets are shifted by '
                          'this amount (correct_model_offset:=true)')
                else:
                    print('MoveIt\'s robot model does not match TF - refusing to move. Make sure '
                          'move_group uses the same URDF as hardware.launch.py.')
                    return 1
            else:
                kin.offset[:] = 0.0
        print(f'Current joints {fmt_q(q_now)}   gripper {node.current_gripper()}')

        if not node.P['dry_run']:
            if not node.arm.wait_for_server(timeout_sec=5.0):
                print(f"action server {node.P['arm_action']} not available - aborting.")
                return 1
            if not node.gripper_dev.wait_ready(5.0):
                print(f"gripper not ready ({node.P['gripper_action']} or {GRIPPER_JOINT} on "
                      f"{node.P['joint_states_topic']}) - aborting.")
                return 1

        keys = KeyReader(node.trigger_estop)
        keys.start()

        # ---- one-shot mode:  -p shape:=square ----------------------------------
        if node.P['shape']:
            shape = parse_shape(node.P['shape'])
            try:
                res = run_shape(node, kin, keys, shape)
            except PlanError as e:
                print(f'\nPLANNING FAILED: {e}\nNothing was moved.')
                return 1
            return 0 if res in ('ok', 'cancelled') else 2

        # ---- interactive menu ----------------------------------------------------
        print(MENU)
        while True:
            cmd = keys.read_line('\npeg-sort> ').lower()
            if not cmd:
                continue
            if cmd in ('q', 'quit', 'exit'):
                print('Bye. ' + ('Torque is still ON, arm holding.' if node.torque_ready else ''))
                return 0
            if cmd in ('help', '?', 'menu'):
                print(MENU)
                continue
            if cmd in ('list', 'l'):
                print_targets(node.P, SHAPE_ORDER)
                continue
            if cmd in ('open', 'o'):
                gripper_cmd(node, 'open')
                continue
            if cmd in ('close', 'cl'):
                gripper_cmd(node, 'close')
                continue
            try:
                if cmd == 'home':
                    go_home(node, kin, keys)
                    continue
                shape = parse_shape(cmd)
                if shape is None:
                    print(f'  unknown command "{cmd}" - type help')
                    continue
                run_shape(node, kin, keys, shape)
            except PlanError as e:
                print(f'\nPLANNING FAILED: {e}\nNothing was moved for the failing step.')

    except KeyboardInterrupt:
        node.trigger_estop()
        print('Ctrl+C - arm holding, exiting.')
        exit_code = 130
    except PlanError as e:
        print(f'\nPLANNING FAILED: {e}\nNothing was moved for the failing step.')
        exit_code = 1
    finally:
        if keys is not None:
            keys.stop()
            keys.join(timeout=0.5)
            keys.restore()
        time.sleep(0.2)
        try:
            ex.shutdown(timeout_sec=2.0)
            spin.join(timeout=2.0)
            node.destroy_node()
        except Exception:
            pass
        finally:
            if rclpy.ok():
                rclpy.shutdown()
    return exit_code


if __name__ == '__main__':
    sys.exit(main(sys.argv) or 0)
