#!/usr/bin/env python3
"""
Addverb Heal - BUZZ WIRE path teaching (ROS 2)

Built on your pygame keyboard teleop (same keys, same DLS Jacobian IK, same
velocity controller). Teach the path in two ways, at the same time if you like:

  1. KEYBOARD: hold keys as before.
  2. HAND-GUIDE: hold the gripper and push / twist it. The arm follows
     (admittance: hand force -> Cartesian velocity -> DLS IK -> joint velocities).

While recording, a waypoint (tool pose + joint angles) is saved every
`waypoint_period_s` seconds (default 5 s, set 10.0 for 10 s) and published on
ROS 2 topics. Press P at any time to drop an extra waypoint (sharp bends).

KEYS (click the pygame window first)
  q/a w/s e/d   +/- X Y Z            (hold)
  r/f t/g y/h   +/- Roll Pitch Yaw   (hold)
  SHIFT (hold)  slow / fine motion
  SPACE         hand-guide ON / OFF  (hands OFF the robot ~1 s while it zeroes)
  Z             re-zero hand-guide force (hands off)
  O             start / stop recording (auto waypoint every waypoint_period_s)
  P             save a waypoint now
  U             undo last waypoint
  C             gripper CLOSE (grab the buzz-wire loop)
  V , V         gripper OPEN: press V twice within 1.5 s (so the loop is never dropped by accident)
  ESC           stop all motion (also switches hand-guide off)
  X             exit: stops robot, writes the CSV

GRIPPER (set gripper_mode + gripper_name to what your Heal exposes; find it with
         `ros2 action list -t`, `ros2 topic list -t`, `ros2 service list -t`, `ros2 control list_controllers`)
  gripper_mode:=action       control_msgs/action/GripperCommand   (default, name /gripper_controller/gripper_cmd)
  gripper_mode:=float_array  std_msgs/Float64MultiArray           (forward position controller, e.g. /gripper_controller/commands)
  gripper_mode:=float        std_msgs/Float64
  gripper_mode:=bool         std_msgs/Bool                        (gripper_close_is_true sets which value closes)
  gripper_mode:=setbool      std_srvs/srv/SetBool service         (same)
  gripper_mode:=none         gripper keys disabled
  Positions: gripper_open_position / gripper_close_position (units = whatever your gripper uses),
  gripper_max_effort for the action. The last gripper command is logged in the CSV 'gripper' column.

TOPICS (prefix = output_ns, default /buzzwire)
  /buzzwire/waypoint          geometry_msgs/PoseStamped        each new waypoint (tip pose in base_link)
  /buzzwire/waypoint_joints   sensor_msgs/JointState           joint angles of each new waypoint
  /buzzwire/path              nav_msgs/Path                    all waypoints so far (latched, show in RViz)
  /buzzwire/joint_path        trajectory_msgs/JointTrajectory  all waypoints in joint space (latched),
                                                               time_from_start = when it was taught
  /buzzwire/handguide_wrench  geometry_msgs/WrenchStamped      estimated hand force (for checking/plotting)
  CSV file: waypoint_file (default ./buzzwire_waypoints_<date>_<time>.csv), rewritten on every change.

HAND-GUIDE: where the force comes from
  Default: joint torques in /joint_states 'effort'
      tau_ext = effort_sign * (effort - gravity_model(q) - offset)
      F_tip   = (J J^T + mu I)^-1 J tau_ext        (force/torque at the tip, base_link frame)
  'offset' is measured when you switch hand-guide on, so keep hands off for that second.
  If your Heal has a wrist F/T sensor, use it instead: -p wrench_topic:=/your/wrench_topic

FIRST RUN CHECK (robot clear of the wire, e-stop in reach)
  1. Run with  -p handguide_dry_run:=true   (force is shown, but hand-guide never moves the arm)
  2. Press SPACE with hands off, wait for "ACTIVE".
  3. Push the gripper gently toward +X of base_link: F x in the window must go POSITIVE.
     If it goes negative, run with -p effort_sign:=1.0
  4. Hands off, jog with the keyboard to a different pose. F should stay small (< ~5 N).
     If it drifts a lot, try -p use_gravity_model:=false (some drivers already remove gravity
     from 'effort', or report current instead of N*m).
  5. Run normally. Default gains are deliberately low; raise handguide_lin_gain slowly.

RUN
  python3 heal_buzzwire_teach.py --ros-args -p tip_link:=end-effector -p waypoint_period_s:=5.0
"""

from __future__ import annotations

import csv
import math
import os
import threading
import time
from datetime import datetime

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy

try:
    from rclpy.executors import ExternalShutdownException
except ImportError:  # older rclpy
    class ExternalShutdownException(Exception):
        pass

import PyKDL as kdl
from urdf_parser_py.urdf import URDF
from kdl_parser_py.urdf import treeFromUrdfModel

from builtin_interfaces.msg import Duration
from geometry_msgs.msg import PoseStamped, WrenchStamped
from nav_msgs.msg import Path
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

import pygame


def _vec_deadband(v: np.ndarray, db: float) -> np.ndarray:
    """Remove `db` from the length of v (smooth start, no jump at the threshold)."""
    n = float(np.linalg.norm(v))
    if n <= db or n == 0.0:
        return np.zeros_like(v)
    return v * ((n - db) / n)


def _clamp_norm(v: np.ndarray, vmax: float) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if vmax > 0.0 and n > vmax:
        return v * (vmax / n)
    return v


class BuzzWireTeach(Node):
    MOVE_KEYS = "qawsedrftgyh"
    COMMAND_KEYS = "opuzcv"

    def __init__(self):
        super().__init__("buzzwire_teach")

        # -------------------------
        # Params (override with --ros-args -p name:=value)
        # -------------------------
        defaults = {
            # same as your teleop
            "joint_states_topic": "/joint_states",
            "robot_description_topic": "/robot_description",
            "cmd_topic": "/velocity_controller/commands",
            "base_link": "base_link",
            "tip_link": "end-effector",
            "rate_hz": 200.0,
            "v": 0.075,                    # keyboard linear m/s (and angular rad/s)
            "damping": 0.05,               # lambda in your DLS
            "keyboard_hz": 200.0,
            "pygame_window_w": 600,
            "pygame_window_h": 330,
            "pygame_grab_input": True,
            "exit_key": "x",
            "exit_on_window_close": True,
            # safety
            "fine_scale": 0.25,            # speed factor while SHIFT is held
            "max_joint_vel": 0.6,          # rad/s, whole qdot is scaled down if any joint exceeds (0 = off)
            "joint_state_timeout_s": 0.5,  # stop if /joint_states goes silent
            # waypoint recording
            "waypoint_period_s": 5.0,      # auto waypoint every N s while recording
            "waypoint_min_dist_m": 0.002,  # auto waypoint skipped if moved less than this ...
            "waypoint_min_angle_deg": 2.0,  # ... and rotated less than this (P always saves)
            "record_on_start": False,
            "output_ns": "/buzzwire",
            "waypoint_file": "",           # "" = auto name in current folder, "none" = no file
            # hand-guide
            "wrench_topic": "",            # WrenchStamped from a wrist F/T sensor; "" = use joint efforts
            "wrench_in_tip_frame": True,   # sensor axes = tip_link axes (rotated to base_link here)
            "wrench_sign": 1.0,
            "effort_sign": -1.0,           # tau_ext = effort_sign * (effort - gravity - offset)
            "use_gravity_model": True,     # subtract URDF gravity torques from effort
            "gravity": [0.0, 0.0, -9.81],  # in base_link frame
            "handguide_calib_s": 0.6,      # zeroing time when hand-guide switches on
            "handguide_filter_hz": 5.0,    # low-pass on the force estimate
            "handguide_force_deadband": 5.0,   # N
            "handguide_torque_deadband": 0.6,  # N*m
            "handguide_lin_gain": 0.006,   # (m/s) per N above deadband
            "handguide_ang_gain": 0.08,    # (rad/s) per N*m above deadband
            "handguide_max_lin": 0.08,     # m/s
            "handguide_max_ang": 0.35,     # rad/s
            "handguide_translation_only": False,  # True = hand-guide never rotates the tool
            "handguide_force_limit": 80.0,  # N; above this hand-guide switches off (collision / bad estimate)
            "handguide_mu": 0.01,          # damping in the joint-torque -> tip-force solve
            "handguide_dry_run": False,    # True = show force, never move from hand-guide
            "block_if_other_publishers": False,  # True = send nothing while another program publishes to cmd_topic
            # gripper (keys C / V V)
            "gripper_mode": "action",      # action | float_array | float | bool | setbool | none
            "gripper_name": "/gripper_controller/gripper_cmd",  # action / topic / service name
            "gripper_open_position": 0.0,
            "gripper_close_position": 0.8,
            "gripper_max_effort": 20.0,    # action only
            "gripper_array_size": 1,       # float_array only: how many values to send
            "gripper_close_is_true": True,  # bool / setbool: True closes
            "gripper_open_confirm": True,  # V must be pressed twice to open
        }
        for name, val in defaults.items():
            self.declare_parameter(name, val)

        def P(name):
            return self.get_parameter(name).value

        self.joint_states_topic = P("joint_states_topic")
        self.robot_description_topic = P("robot_description_topic")
        self.cmd_topic = P("cmd_topic")
        self.base_link = P("base_link")
        self.tip_link = P("tip_link")
        self.rate_hz = float(P("rate_hz"))
        self.v = float(P("v"))
        self.damping = float(P("damping"))

        self.keyboard_hz = float(P("keyboard_hz"))
        self.pygame_w = int(P("pygame_window_w"))
        self.pygame_h = int(P("pygame_window_h"))
        self.pygame_grab_input = bool(P("pygame_grab_input"))
        self.exit_on_window_close = bool(P("exit_on_window_close"))

        self.fine_scale = float(P("fine_scale"))
        self.max_joint_vel = float(P("max_joint_vel"))
        self.js_timeout = float(P("joint_state_timeout_s"))

        self.wp_period = max(0.1, float(P("waypoint_period_s")))
        self.wp_min_dist = float(P("waypoint_min_dist_m"))
        self.wp_min_angle = float(P("waypoint_min_angle_deg"))
        ns = "/" + str(P("output_ns")).strip("/")
        wp_file = str(P("waypoint_file")).strip()
        if wp_file == "":
            wp_file = f"buzzwire_waypoints_{datetime.now():%Y%m%d_%H%M%S}.csv"
        self.wp_file = None if wp_file.lower() == "none" else os.path.abspath(wp_file)

        self.wrench_topic = str(P("wrench_topic")).strip()
        self.wrench_in_tip_frame = bool(P("wrench_in_tip_frame"))
        self.wrench_sign = float(P("wrench_sign"))
        self.effort_sign = float(P("effort_sign"))
        self.use_gravity_model = bool(P("use_gravity_model"))
        self.gravity = [float(x) for x in P("gravity")]
        self.hg_calib_s = float(P("handguide_calib_s"))
        fc = float(P("handguide_filter_hz"))
        self.hg_alpha = 1.0 if fc <= 0.0 else 1.0 - math.exp(-2.0 * math.pi * fc / self.rate_hz)
        self.hg_f_db = float(P("handguide_force_deadband"))
        self.hg_t_db = float(P("handguide_torque_deadband"))
        self.hg_lin_gain = float(P("handguide_lin_gain"))
        self.hg_ang_gain = float(P("handguide_ang_gain"))
        self.hg_max_lin = float(P("handguide_max_lin"))
        self.hg_max_ang = float(P("handguide_max_ang"))
        self.hg_translation_only = bool(P("handguide_translation_only"))
        self.hg_force_limit = float(P("handguide_force_limit"))
        self.hg_mu = float(P("handguide_mu"))
        self.hg_dry_run = bool(P("handguide_dry_run"))
        self.block_if_other_pubs = bool(P("block_if_other_publishers"))

        self.grip_mode = str(P("gripper_mode")).strip().lower()
        self.grip_name = str(P("gripper_name")).strip()
        self.grip_open_pos = float(P("gripper_open_position"))
        self.grip_close_pos = float(P("gripper_close_position"))
        self.grip_max_effort = float(P("gripper_max_effort"))
        self.grip_array_size = max(1, int(P("gripper_array_size")))
        self.grip_close_is_true = bool(P("gripper_close_is_true"))
        self.grip_open_confirm = bool(P("gripper_open_confirm"))

        # -------------------------
        # Keyboard mapping (same semantics as your teleop)
        # -------------------------
        v = self.v
        self.key_mapping = {
            "q": np.array([ v, 0, 0, 0, 0, 0], dtype=float),
            "a": np.array([-v, 0, 0, 0, 0, 0], dtype=float),
            "w": np.array([0,  v, 0, 0, 0, 0], dtype=float),
            "s": np.array([0, -v, 0, 0, 0, 0], dtype=float),
            "e": np.array([0, 0,  v, 0, 0, 0], dtype=float),
            "d": np.array([0, 0, -v, 0, 0, 0], dtype=float),
            "r": np.array([0, 0, 0,  v, 0, 0], dtype=float),
            "f": np.array([0, 0, 0, -v, 0, 0], dtype=float),
            "t": np.array([0, 0, 0, 0,  v, 0], dtype=float),
            "g": np.array([0, 0, 0, 0, -v, 0], dtype=float),
            "y": np.array([0, 0, 0, 0, 0,  v], dtype=float),
            "h": np.array([0, 0, 0, 0, 0, -v], dtype=float),
        }
        self._pg_key_to_char = {getattr(pygame, f"K_{c}"): c for c in self.MOVE_KEYS}

        exit_char = str(P("exit_key")).strip().lower()[:1] or "x"
        if exit_char in self.MOVE_KEYS or exit_char in self.COMMAND_KEYS:
            self.get_logger().warn(f"exit_key '{exit_char}' is already used; using 'x'.")
            exit_char = "x"
        self.exit_key_char = exit_char
        self._exit_pg_key = self._pygame_key_from_char(exit_char) or pygame.K_x

        # -------------------------
        # State shared with the pygame thread (guarded by _lock)
        # -------------------------
        self._lock = threading.Lock()
        self.pressed_keys: set[str] = set()
        self._fine = False
        self.active_twist = np.zeros(6, dtype=float)
        self._requests: list[str] = []
        self._ui = {
            "hg": "OFF", "F": np.zeros(6), "rec": False, "n_wp": 0, "next_s": None,
            "pos": None, "msg": "Waiting for robot_description and joint_states...",
            "grip": "not used yet",
        }

        # -------------------------
        # ROS pub/sub
        # -------------------------
        self.cmd_pub = self.create_publisher(Float64MultiArray, self.cmd_topic, 10)

        latched = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.wp_pub = self.create_publisher(PoseStamped, f"{ns}/waypoint", 10)
        self.wpj_pub = self.create_publisher(JointState, f"{ns}/waypoint_joints", 10)
        self.path_pub = self.create_publisher(Path, f"{ns}/path", latched)
        self.jpath_pub = self.create_publisher(JointTrajectory, f"{ns}/joint_path", latched)
        self.wrench_pub = self.create_publisher(WrenchStamped, f"{ns}/handguide_wrench", 10)

        self.current_joint_state: JointState | None = None
        self._js_time = 0.0
        self._js_names: list[str] = []
        self._js_name_to_index: dict[str, int] = {}
        self.create_subscription(JointState, self.joint_states_topic, self._joint_state_cb, 50)

        self._wrench = None
        self._wrench_time = 0.0
        if self.wrench_topic:
            self.create_subscription(WrenchStamped, self.wrench_topic, self._wrench_cb, 50)

        self._robot_desc_sub = self.create_subscription(
            String, self.robot_description_topic, self._robot_description_cb, latched
        )

        # Gripper interface (keys C / V V)
        self._grip_error = None
        self._grip_client = None
        self._grip_pub = None
        self._grip_msg_type = None
        self._grip_cmd = ""            # last command sent: "open" / "closed" (logged in CSV)
        self._grip_open_armed_t = -1e9
        self._setup_gripper()

        # -------------------------
        # KDL (built once from URDF)
        # -------------------------
        self._kdl_ready = False
        self.chain = None
        self.n_joints = 0
        self.chain_joint_names: list[str] = []
        self.jac_solver = None
        self.fk_solver = None
        self.dyn_solver = None

        # -------------------------
        # Hand-guide state (ROS thread only)
        # -------------------------
        self._hg_state = "OFF"          # OFF / CALIB / ACTIVE
        self._hg_t0 = 0.0
        self._hg_samples: list[np.ndarray] = []
        self._hg_offset = None
        self._hg_F = np.zeros(6)
        self._hg_pub_count = 0

        # -------------------------
        # Waypoints (ROS thread only)
        # -------------------------
        self._wps: list[dict] = []
        self._recording = bool(P("record_on_start"))
        self._rec_t0 = None
        self._next_auto_t = 0.0
        self._file_error_logged = False

        # Addverb's velocity_controller stops (and has crashed) when 2+ programs publish to it
        self._blocked = False
        self._next_pub_check = 0.0
        self._other_pubs = 0

        # -------------------------
        # Exit flags
        # -------------------------
        self._exit_requested = False
        self.finished = False
        self._stopped_and_saved = False

        # -------------------------
        # Pygame keyboard thread + control timer
        # -------------------------
        self._kb_running = True
        self._kb_thread = threading.Thread(target=self._pygame_loop, daemon=True)
        self._kb_thread.start()
        self._timer = self.create_timer(1.0 / self.rate_hz, self._on_timer)

        self.get_logger().info(
            f"Started. Waiting for {self.robot_description_topic} (URDF) and {self.joint_states_topic}..."
        )
        self.get_logger().info(
            f"Commands -> {self.cmd_topic} at {self.rate_hz:.0f} Hz. Waypoints -> {ns}/... "
            f"every {self.wp_period:g} s while recording. File: {self.wp_file or 'disabled'}"
        )
        self.get_logger().info(
            "Keys: qa ws ed (XYZ), rf tg yh (RPY), SHIFT slow, SPACE hand-guide, Z re-zero, "
            f"O record, P save, U undo, C grip close, V V grip open, ESC stop, {self.exit_key_char.upper()} exit."
        )

    # =====================================================================
    # Pygame thread
    # =====================================================================
    @staticmethod
    def _pygame_key_from_char(ch: str) -> int | None:
        if ch and ("a" <= ch <= "z" or "0" <= ch <= "9"):
            return getattr(pygame, f"K_{ch}", None)
        return None

    def _recompute_twist(self):
        tw = np.zeros(6, dtype=float)
        for k in self.pressed_keys:
            vec = self.key_mapping.get(k)
            if vec is not None:
                tw += vec
        self.active_twist = tw * (self.fine_scale if self._fine else 1.0)

    def _clear_keys(self):
        with self._lock:
            self.pressed_keys = set()
            self._recompute_twist()

    def _request(self, what: str):
        with self._lock:
            self._requests.append(what)

    def request_exit(self, reason: str = ""):
        if self._exit_requested:
            return
        self._exit_requested = True
        self.get_logger().warn(f"Exit requested{': ' + reason if reason else '.'}")
        self._kb_running = False
        self._clear_keys()
        self._publish_zero()  # stop right away; the timer stops again and saves

    def _pygame_loop(self):
        try:
            pygame.init()
            screen = pygame.display.set_mode((self.pygame_w, self.pygame_h))
            pygame.display.set_caption("Buzz-wire teach (CLICK HERE)")
            clock = pygame.time.Clock()
            try:
                pygame.event.set_grab(self.pygame_grab_input)
            except Exception:
                pass
            font = pygame.font.Font(None, 22)
            header = [
                "qa ws ed: XYZ   rf tg yh: RPY   hold SHIFT: slow",
                f"SPACE hand-guide  Z re-zero  O record  P save  U undo  ESC stop  {self.exit_key_char.upper()} exit",
                "C gripper close   V V (twice) gripper open",
            ]
            had_focus = True

            while rclpy.ok() and self._kb_running and not self._exit_requested:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        if self.exit_on_window_close:
                            self.request_exit("Pygame window closed.")
                        else:
                            self.get_logger().warn("Pygame window closed -> keyboard off, hand-guide off.")
                            self._kb_running = False
                            self._clear_keys()
                            self._request("stop")
                        break
                    if event.type == pygame.KEYDOWN:
                        k = event.key
                        if k == self._exit_pg_key:
                            self.request_exit(f"Exit key '{self.exit_key_char}' pressed.")
                        elif k == pygame.K_ESCAPE:
                            self._clear_keys()
                            self._request("stop")
                        elif k == pygame.K_SPACE:
                            self._request("toggle_hg")
                        elif k == pygame.K_z:
                            self._request("zero")
                        elif k == pygame.K_o:
                            self._request("toggle_rec")
                        elif k == pygame.K_p:
                            self._request("save")
                        elif k == pygame.K_u:
                            self._request("undo")
                        elif k == pygame.K_c:
                            self._request("grip_close")
                        elif k == pygame.K_v:
                            self._request("grip_open")

                if (not self._kb_running) or self._exit_requested:
                    break

                focused = pygame.key.get_focused()
                if not focused:
                    self._clear_keys()
                    if had_focus:
                        self._request("focus_lost")
                else:
                    keys = pygame.key.get_pressed()
                    if keys[pygame.K_ESCAPE]:
                        self._clear_keys()
                    else:
                        held = {ch for pgk, ch in self._pg_key_to_char.items() if keys[pgk]}
                        fine = bool(pygame.key.get_mods() & pygame.KMOD_SHIFT)
                        with self._lock:
                            self.pressed_keys = held
                            self._fine = fine
                            self._recompute_twist()
                had_focus = focused

                self._draw(screen, font, header)
                clock.tick(max(60.0, self.keyboard_hz))

        except Exception as e:
            self.get_logger().error(f"Pygame loop crashed: {e}")
            self._clear_keys()
            self._request("stop")
        finally:
            try:
                pygame.quit()
            except Exception:
                pass

    def _draw(self, screen, font, header):
        with self._lock:
            ui = dict(self._ui)
            ui["F"] = ui["F"].copy()
            keys_str = "".join(sorted(self.pressed_keys)) or "(none)"
            fine = self._fine

        white, grey = (230, 230, 230), (160, 160, 160)
        green, amber, red = (120, 230, 120), (255, 200, 80), (255, 110, 110)
        screen.fill((25, 25, 25))
        y = 8

        def line(text, color=white, dy=24):
            nonlocal y
            screen.blit(font.render(text, True, color), (10, y))
            y += dy

        for h in header:
            line(h, grey, 22)
        y += 6

        hg = ui["hg"]
        if hg == "OFF":
            line("Hand-guide: OFF", grey)
        elif hg == "CALIB":
            line("Hand-guide: HANDS OFF - zeroing...", amber)
        else:
            line("Hand-guide: ACTIVE" + ("  (DRY RUN - no motion)" if self.hg_dry_run else "  - hold the gripper"), green)
        F = ui["F"]
        line(f"F [N]  x {F[0]:+6.1f}  y {F[1]:+6.1f}  z {F[2]:+6.1f}   |F| {np.linalg.norm(F[:3]):5.1f}",
             white if hg == "ACTIVE" else grey)
        line(f"T [Nm] x {F[3]:+6.2f}  y {F[4]:+6.2f}  z {F[5]:+6.2f}", white if hg == "ACTIVE" else grey)
        line(f"Gripper ({self.grip_mode}): {ui['grip']}", (150, 200, 255))

        if ui["rec"]:
            nxt = ui["next_s"]
            line(f"REC ON   waypoints: {ui['n_wp']}   next in {nxt:4.1f} s" if nxt is not None
                 else f"REC ON   waypoints: {ui['n_wp']}", red)
        else:
            line(f"REC OFF  waypoints: {ui['n_wp']}   (O to start)", grey)

        pos = ui["pos"]
        if pos is not None:
            line(f"Tip [m]: x {pos[0]:+.3f}  y {pos[1]:+.3f}  z {pos[2]:+.3f}")
        line(f"Pressed: {keys_str}{'  (SLOW)' if fine else ''}", (180, 220, 180))
        line(ui["msg"], amber)
        pygame.display.flip()

    # =====================================================================
    # ROS callbacks
    # =====================================================================
    def _joint_state_cb(self, msg: JointState):
        if list(msg.name) != self._js_names:
            self._js_names = list(msg.name)
            self._js_name_to_index = {n: i for i, n in enumerate(msg.name)}
        self.current_joint_state = msg
        self._js_time = time.monotonic()

    def _wrench_cb(self, msg: WrenchStamped):
        f, t = msg.wrench.force, msg.wrench.torque
        self._wrench = np.array([f.x, f.y, f.z, t.x, t.y, t.z], dtype=float)
        self._wrench_time = time.monotonic()

    def _robot_description_cb(self, msg: String):
        if self._kdl_ready:
            return
        urdf_xml = msg.data.strip()
        if not urdf_xml or "<robot" not in urdf_xml:
            self.get_logger().error("robot_description received but doesn't look like URDF XML.")
            return
        try:
            robot = URDF.from_xml_string(urdf_xml)
            ok, tree = treeFromUrdfModel(robot)
            if not ok:
                self.get_logger().error("Failed to construct KDL tree from URDF.")
                return
            chain = tree.getChain(self.base_link, self.tip_link)
            n = int(chain.getNrOfJoints())
            if n == 0:
                self.get_logger().error(
                    f"KDL chain has 0 joints for {self.base_link} -> {self.tip_link}. "
                    "Check base_link/tip_link names."
                )
                return

            # movable joints only ("None" = fixed in KDL < 1.5, "Fixed" in KDL >= 1.5)
            names = []
            for i in range(chain.getNrOfSegments()):
                j = chain.getSegment(i).getJoint()
                if j.getTypeName() not in ("None", "Fixed"):
                    names.append(j.getName())
            if len(names) != n:
                self.get_logger().error(f"Joint-name extraction mismatch: n={n}, got {names}")
                return

            self.chain = chain
            self.n_joints = n
            self.chain_joint_names = names
            self.jac_solver = kdl.ChainJntToJacSolver(chain)
            self.fk_solver = kdl.ChainFkSolverPos_recursive(chain)
            self.dyn_solver = kdl.ChainDynParam(chain, kdl.Vector(*self.gravity))
            self._kdl_ready = True
            self.get_logger().info(f"KDL ready: n_joints={n}, chain joint order={names}")
            self._status("Ready. Click here, then use keys / SPACE for hand-guide.")
            self.destroy_subscription(self._robot_desc_sub)
        except Exception as e:
            self.get_logger().error(f"Exception creating KDL from URDF: {e}")

    # =====================================================================
    # Kinematics
    # =====================================================================
    def _to_jnt(self, q: np.ndarray) -> kdl.JntArray:
        a = kdl.JntArray(self.n_joints)
        for i in range(self.n_joints):
            a[i] = float(q[i])
        return a

    def _compute_jacobian(self, q: np.ndarray) -> np.ndarray:
        """J (6 x n), chain joint order. Reference point = tip, expressed in base_link."""
        jac = kdl.Jacobian(self.n_joints)
        self.jac_solver.JntToJac(self._to_jnt(q), jac)
        return np.array([[jac[i, j] for j in range(self.n_joints)] for i in range(6)], dtype=float)

    def _compute_fk(self, q: np.ndarray):
        frame = kdl.Frame()
        if self.fk_solver.JntToCart(self._to_jnt(q), frame) < 0:
            raise RuntimeError("FK failed")
        pos = np.array([frame.p[0], frame.p[1], frame.p[2]], dtype=float)
        quat = np.array(frame.M.GetQuaternion(), dtype=float)  # x, y, z, w
        R = np.array([[frame.M[i, j] for j in range(3)] for i in range(3)], dtype=float)
        return pos, quat, R

    def _gravity_torque(self, q: np.ndarray) -> np.ndarray:
        g = kdl.JntArray(self.n_joints)
        self.dyn_solver.JntToGravity(self._to_jnt(q), g)
        return np.array([g[i] for i in range(self.n_joints)], dtype=float)

    def _dls_pinv_times(self, J: np.ndarray, v6: np.ndarray) -> np.ndarray:
        """Your DLS: qdot = J^T (J J^T + lambda I)^-1 v  (lambda, not lambda^2, as in your draft)."""
        A = J @ J.T + float(self.damping) * np.eye(6)
        return J.T @ np.linalg.solve(A, v6)

    def _limit_joint_vel(self, qdot: np.ndarray) -> np.ndarray:
        m = float(np.max(np.abs(qdot))) if qdot.size else 0.0
        if self.max_joint_vel > 0.0 and m > self.max_joint_vel:
            qdot = qdot * (self.max_joint_vel / m)  # keep direction, slow everything down
        return qdot

    # =====================================================================
    # Control loop
    # =====================================================================
    def _on_timer(self):
        try:
            self._step()
        except Exception as e:
            # never leave the velocity controller holding an old command
            self._publish_zero()
            self.get_logger().error(f"Control step failed: {e!r} -> zero velocity", throttle_duration_sec=1.0)

    def _step(self):
        now = time.monotonic()

        if self._exit_requested:
            if not self.finished:
                self.stop_and_save()
                self.finished = True
            return

        with self._lock:
            reqs, self._requests = self._requests, []

        # gripper keys work even before the arm is ready
        for r in reqs:
            if r.startswith("grip_"):
                self._grip_request(r, now)
        reqs = [r for r in reqs if not r.startswith("grip_")]

        js = self.current_joint_state
        if not self._kdl_ready or js is None:
            if reqs:
                self._status("Not ready yet: waiting for URDF / joint_states.")
            return

        if now - self._js_time > self.js_timeout:
            self._publish_zero()
            if self._hg_state != "OFF":
                self._hg_off("joint_states stopped")
            self._status("NO /joint_states - is the robot launch (ros2_control_node) still running?")
            self.get_logger().warn(
                f"No {self.joint_states_topic} for > {self.js_timeout:g} s -> not moving. "
                "Check the launch terminal: ros2_control_node may have died.",
                throttle_duration_sec=2.0,
            )
            return

        q, effort, missing = self._read_joint_state(js)
        if missing:
            self._publish_zero()
            self.get_logger().warn(
                f"Chain joints not in JointState: {missing} -> not moving. Check URDF vs /joint_states names.",
                throttle_duration_sec=2.0,
            )
            return

        if now >= self._next_pub_check:
            self._next_pub_check = now + 0.5
            others = self.count_publishers(self.cmd_topic) - 1  # minus this node
            self._other_pubs = others
            if others > 0 and not self.block_if_other_pubs:
                self.get_logger().warn(
                    f"{others} OTHER publisher(s) on {self.cmd_topic}: Addverb's velocity_controller stops "
                    "when 2+ programs publish, so the robot may not move. "
                    f"Find it: ros2 topic info -v {self.cmd_topic}", throttle_duration_sec=5.0)
        if self.block_if_other_pubs and now >= self._next_pub_check - 0.5:
            others = self._other_pubs
            if others > 0 and not self._blocked:
                self._blocked = True
                self._clear_keys()
                if self._hg_state != "OFF":
                    self._hg_off("another publisher")
                self.get_logger().error(
                    f"{others} other publisher(s) on {self.cmd_topic} (heal_init.py / IK script still running?). "
                    "Not sending anything until it is closed - two publishers make velocity_controller stop."
                )
            elif others <= 0 and self._blocked:
                self._blocked = False
                self.get_logger().info(f"{self.cmd_topic} is free again - teleop active.")
                self._status("Other publisher gone - teleop active again.")
        if self._blocked:
            self._status(f"BLOCKED: another program publishes to {self.cmd_topic}. Close it.")
            return

        J = self._compute_jacobian(q)
        pos, quat, R = self._compute_fk(q)

        for r in reqs:
            self._handle_request(r, q, pos, quat, effort, now)

        hg_twist = self._handguide_update(q, J, R, effort, now)

        with self._lock:
            kb_twist = self.active_twist.copy()
        if self._hg_state == "CALIB":
            kb_twist[:] = 0.0  # hold still while zeroing the force

        twist = kb_twist + hg_twist
        if np.allclose(twist, 0.0):
            self._publish_zero()
        else:
            qdot = self._dls_pinv_times(J, twist)
            qdot[np.abs(qdot) < 1e-3] = 0.0
            self._publish_cmd(self._limit_joint_vel(qdot))

        if self._recording and now >= self._next_auto_t:
            nxt = self._next_auto_t + self.wp_period
            self._next_auto_t = nxt if nxt > now else now + self.wp_period
            self._save_waypoint(q, pos, quat, now, "auto")

        if self._other_pubs > 0:
            self._status(f"WARNING: {self._other_pubs} other publisher(s) on {self.cmd_topic} - robot may not move")
        with self._lock:
            self._ui["hg"] = self._hg_state
            self._ui["F"] = self._hg_F.copy()
            self._ui["rec"] = self._recording
            self._ui["n_wp"] = len(self._wps)
            self._ui["next_s"] = max(0.0, self._next_auto_t - now) if self._recording else None
            self._ui["pos"] = pos

    def _read_joint_state(self, js: JointState):
        q = np.zeros(self.n_joints, dtype=float)
        effort = np.zeros(self.n_joints, dtype=float) if len(js.effort) else None
        missing = []
        for i, jn in enumerate(self.chain_joint_names):
            idx = self._js_name_to_index.get(jn)
            if idx is None or idx >= len(js.position):
                missing.append(jn)
                continue
            q[i] = float(js.position[idx])
            if effort is not None:
                if idx < len(js.effort):
                    effort[i] = float(js.effort[idx])
                else:
                    effort = None
        return q, effort, missing

    def _handle_request(self, r, q, pos, quat, effort, now):
        if r == "toggle_hg":
            if self._hg_state == "OFF":
                self._hg_start(q, effort, now)
            else:
                self._hg_off("switched off")
        elif r == "zero":
            if self._hg_state != "OFF":
                self._hg_start(q, effort, now)
            else:
                self._status("Hand-guide is off (SPACE to switch on).")
        elif r == "stop":
            if self._hg_state != "OFF":
                self._hg_off("ESC")
            self._status("STOP: all motion cleared.")
        elif r == "focus_lost":
            if self._hg_state != "OFF":
                self._hg_off("window lost focus")
        elif r == "toggle_rec":
            self._recording = not self._recording
            if self._recording:
                self._next_auto_t = now  # first waypoint right away
                self._status(f"REC ON: waypoint every {self.wp_period:g} s (P = extra one).")
                self.get_logger().info("Recording started.")
            else:
                self._status(f"REC OFF: {len(self._wps)} waypoints.")
                self.get_logger().info(f"Recording stopped: {len(self._wps)} waypoints.")
        elif r == "save":
            self._save_waypoint(q, pos, quat, now, "manual")
        elif r == "undo":
            self._undo_waypoint()

    # =====================================================================
    # Hand-guide (admittance)
    # =====================================================================
    def _hg_signal(self, q, effort, now):
        """Raw signal the offset is taken from: joint-torque residual (n,) or sensor wrench (6,)."""
        if self.wrench_topic:
            if self._wrench is None or now - self._wrench_time > 0.2:
                return None
            return self._wrench.copy()
        if effort is None:
            return None
        s = effort.copy()
        if self.use_gravity_model:
            s -= self._gravity_torque(q)
        return s

    def _hg_wrench(self, s, J, R):
        """(signal - offset) -> wrench at the tip, base_link frame: [Fx Fy Fz Tx Ty Tz]."""
        d = s - self._hg_offset
        if self.wrench_topic:
            w = self.wrench_sign * d
            if self.wrench_in_tip_frame:
                w = np.concatenate([R @ w[:3], R @ w[3:]])
            return w
        tau_ext = self.effort_sign * d
        A = J @ J.T + self.hg_mu * np.eye(6)
        return np.linalg.solve(A, J @ tau_ext)

    def _hg_start(self, q, effort, now):
        if self._hg_signal(q, effort, now) is None:
            src = f"wrench topic {self.wrench_topic}" if self.wrench_topic else "'effort' in /joint_states"
            self._status(f"Hand-guide unavailable: no data from {src}.")
            self.get_logger().error(f"Hand-guide needs {src}; none received.")
            return
        self._hg_state = "CALIB"
        self._hg_t0 = now
        self._hg_samples = []
        self._hg_F = np.zeros(6)
        self._status("HANDS OFF the robot - zeroing force...")

    def _hg_off(self, reason: str):
        self._hg_state = "OFF"
        self._hg_samples = []
        self._hg_F = np.zeros(6)
        self._status(f"Hand-guide OFF ({reason}).")
        self.get_logger().info(f"Hand-guide OFF ({reason}).")

    def _handguide_update(self, q, J, R, effort, now) -> np.ndarray:
        twist = np.zeros(6, dtype=float)
        if self._hg_state == "OFF":
            return twist

        s = self._hg_signal(q, effort, now)
        if s is None:
            self._hg_off("force data stopped")
            return twist

        if self._hg_state == "CALIB":
            self._hg_samples.append(s)
            if now - self._hg_t0 >= self.hg_calib_s and len(self._hg_samples) >= 5:
                self._hg_offset = np.mean(self._hg_samples, axis=0)
                self._hg_samples = []
                self._hg_F = np.zeros(6)
                self._hg_state = "ACTIVE"
                msg = "Hand-guide ACTIVE" + (" (DRY RUN: no motion)." if self.hg_dry_run else ": hold the gripper and push.")
                self._status(msg)
                self.get_logger().info(msg)
            return twist

        # ACTIVE
        F_raw = self._hg_wrench(s, J, R)
        self._hg_F = self._hg_F + self.hg_alpha * (F_raw - self._hg_F)
        F = self._hg_F

        self._hg_pub_count += 1
        if self._hg_pub_count % max(1, int(self.rate_hz / 20.0)) == 0:  # ~20 Hz
            w = WrenchStamped()
            w.header.stamp = self.get_clock().now().to_msg()
            w.header.frame_id = self.base_link
            w.wrench.force.x, w.wrench.force.y, w.wrench.force.z = (float(x) for x in F[:3])
            w.wrench.torque.x, w.wrench.torque.y, w.wrench.torque.z = (float(x) for x in F[3:])
            self.wrench_pub.publish(w)

        fmag = float(np.linalg.norm(F[:3]))
        if fmag > self.hg_force_limit:
            self._hg_off(f"|F| {fmag:.0f} N > handguide_force_limit {self.hg_force_limit:.0f} N")
            return twist

        v = _clamp_norm(self.hg_lin_gain * _vec_deadband(F[:3], self.hg_f_db), self.hg_max_lin)
        w = _clamp_norm(self.hg_ang_gain * _vec_deadband(F[3:], self.hg_t_db), self.hg_max_ang)
        if self.hg_translation_only:
            w = np.zeros(3)
        if self.hg_dry_run:
            return twist
        twist[:3] = v
        twist[3:] = w
        return twist

    # =====================================================================
    # Gripper (keys C / V V)
    # =====================================================================
    def _setup_gripper(self):
        m, name = self.grip_mode, self.grip_name
        try:
            if m == "none":
                self._grip_set("disabled (gripper_mode:=none)")
            elif m == "action":
                from rclpy.action import ActionClient
                from control_msgs.action import GripperCommand
                self._grip_msg_type = GripperCommand
                self._grip_client = ActionClient(self, GripperCommand, name)
            elif m == "float_array":
                self._grip_pub = self.create_publisher(Float64MultiArray, name, 10)
            elif m == "float":
                from std_msgs.msg import Float64
                self._grip_msg_type = Float64
                self._grip_pub = self.create_publisher(Float64, name, 10)
            elif m == "bool":
                from std_msgs.msg import Bool
                self._grip_msg_type = Bool
                self._grip_pub = self.create_publisher(Bool, name, 10)
            elif m == "setbool":
                from std_srvs.srv import SetBool
                self._grip_msg_type = SetBool
                self._grip_client = self.create_client(SetBool, name)
            else:
                self._grip_error = (f"unknown gripper_mode '{m}' "
                                    "(use action, float_array, float, bool, setbool or none)")
        except ImportError as e:
            self._grip_error = f"{e.name or e} not installed - change gripper_mode or install it"
        if self._grip_error:
            self._grip_set(f"OFF: {self._grip_error}")
            self.get_logger().error(f"Gripper keys disabled: {self._grip_error}")
        elif m != "none":
            self.get_logger().info(f"Gripper: mode={m}, name={name}, open={self.grip_open_pos}, "
                                   f"close={self.grip_close_pos}")

    def _grip_set(self, text: str):
        with self._lock:
            self._ui["grip"] = text

    def _grip_request(self, r: str, now: float):
        if self.grip_mode == "none" or self._grip_error:
            self._status("Gripper keys are off: " + (self._grip_error or "gripper_mode is 'none'"))
            return
        if r == "grip_close":
            self._grip_open_armed_t = -1e9
            self._grip_send(close=True)
        elif r == "grip_open":
            if self.grip_open_confirm and now - self._grip_open_armed_t > 1.5:
                self._grip_open_armed_t = now
                self._status("Press V again (within 1.5 s) to OPEN the gripper - the loop will drop!")
                return
            self._grip_open_armed_t = -1e9
            self._grip_send(close=False)

    def _grip_send(self, close: bool):
        word = "CLOSE" if close else "OPEN"
        pos = self.grip_close_pos if close else self.grip_open_pos
        flag = (close == self.grip_close_is_true)
        m, name = self.grip_mode, self.grip_name

        # check the server / build the message first
        if m == "action" and not self._grip_client.server_is_ready():
            self._grip_set(f"{word} not sent: action server {name} not found")
            self._status(f"Gripper action server '{name}' not found - check `ros2 action list -t`.")
            return
        if m == "setbool" and not self._grip_client.service_is_ready():
            self._grip_set(f"{word} not sent: service {name} not found")
            self._status(f"Gripper service '{name}' not found - check `ros2 service list -t`.")
            return
        if m == "action":
            sent = f"{word} sent (position {pos:g})"
        elif m == "setbool":
            sent = f"{word} sent (data={flag})"
        elif m in ("float_array", "float"):
            sent = f"{word} published {pos:g}" + (f" x{self.grip_array_size}" if m == "float_array" else "")
        else:  # bool
            sent = f"{word} published data={flag}"

        # show "sent" BEFORE sending, so a fast reply from the gripper is not overwritten
        self._grip_cmd = "closed" if close else "open"
        self._grip_set(sent)
        self._status(f"Gripper {sent}")
        self.get_logger().info(f"Gripper {sent} -> {name}")

        if m == "action":
            goal = self._grip_msg_type.Goal()
            goal.command.position = float(pos)
            goal.command.max_effort = float(self.grip_max_effort)
            fut = self._grip_client.send_goal_async(goal)
            fut.add_done_callback(lambda f, w=word: self._grip_goal_response(f, w))
        elif m == "setbool":
            req = self._grip_msg_type.Request()
            req.data = flag
            fut = self._grip_client.call_async(req)
            fut.add_done_callback(lambda f, w=word: self._grip_service_done(f, w))
        elif m == "float_array":
            msg = Float64MultiArray()
            msg.data = [float(pos)] * self.grip_array_size
            self._grip_pub.publish(msg)
        elif m == "float":
            self._grip_pub.publish(self._grip_msg_type(data=float(pos)))
        else:  # bool
            self._grip_pub.publish(self._grip_msg_type(data=flag))

    def _grip_goal_response(self, fut, word: str):
        try:
            handle = fut.result()
        except Exception as e:
            self._grip_set(f"{word} failed: {e}")
            return
        if not handle.accepted:
            self._grip_set(f"{word} rejected by the gripper controller")
            self.get_logger().warn(f"Gripper {word} goal rejected.")
            return
        handle.get_result_async().add_done_callback(lambda f, w=word: self._grip_result(f, w))

    def _grip_result(self, fut, word: str):
        try:
            res = fut.result().result
        except Exception as e:
            self._grip_set(f"{word} failed: {e}")
            return
        if res.reached_goal:
            how = "done"
        elif res.stalled:
            how = "stalled - holding the tool" if word == "CLOSE" else "stalled"
        else:
            how = "finished without reaching the goal"
        self._grip_set(f"{word}: {how} (position {res.position:.3f})")
        self.get_logger().info(f"Gripper {word}: {how}, position {res.position:.3f}, effort {res.effort:.1f}")

    def _grip_service_done(self, fut, word: str):
        try:
            res = fut.result()
            self._grip_set(f"{word}: {'ok' if res.success else 'FAILED'} {res.message}".strip())
        except Exception as e:
            self._grip_set(f"{word} failed: {e}")

    # =====================================================================
    # Waypoints
    # =====================================================================
    def _save_waypoint(self, q, pos, quat, now, source: str):
        quat = quat.copy()
        if self._wps:
            last = self._wps[-1]
            dot = float(np.dot(quat, last["quat"]))
            if dot < 0.0:  # keep quaternion sign continuous along the path
                quat = -quat
                dot = -dot
            if source == "auto":
                dist = float(np.linalg.norm(pos - last["pos"]))
                ang = math.degrees(2.0 * math.acos(min(1.0, dot)))
                if dist < self.wp_min_dist and ang < self.wp_min_angle:
                    self._status(f"Auto waypoint skipped: moved {dist * 1000:.1f} mm, {ang:.1f} deg.")
                    return

        if self._rec_t0 is None:
            self._rec_t0 = now
        wp = {"t": now - self._rec_t0, "pos": pos.copy(), "quat": quat, "q": q.copy(), "source": source,
              "gripper": self._grip_cmd}
        self._wps.append(wp)
        k = len(self._wps)

        stamp = self.get_clock().now().to_msg()
        self.wp_pub.publish(self._pose_msg(wp, stamp))
        js = JointState()
        js.header.stamp = stamp
        js.name = list(self.chain_joint_names)
        js.position = [float(x) for x in wp["q"]]
        self.wpj_pub.publish(js)
        self._publish_paths(stamp)
        self._write_file()

        txt = f"Waypoint #{k} ({source}) at [{pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f}] m"
        self._status(txt)
        self.get_logger().info(txt)

    def _undo_waypoint(self):
        if not self._wps:
            self._status("No waypoint to undo.")
            return
        self._wps.pop()
        self._publish_paths(self.get_clock().now().to_msg())
        self._write_file()
        self._status(f"Removed last waypoint, {len(self._wps)} left.")
        self.get_logger().info(f"Undo: {len(self._wps)} waypoints left.")

    def _pose_msg(self, wp, stamp) -> PoseStamped:
        ps = PoseStamped()
        ps.header.stamp = stamp
        ps.header.frame_id = self.base_link
        ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = (float(x) for x in wp["pos"])
        (ps.pose.orientation.x, ps.pose.orientation.y,
         ps.pose.orientation.z, ps.pose.orientation.w) = (float(x) for x in wp["quat"])
        return ps

    def _publish_paths(self, stamp):
        path = Path()
        path.header.stamp = stamp
        path.header.frame_id = self.base_link
        path.poses = [self._pose_msg(wp, stamp) for wp in self._wps]
        self.path_pub.publish(path)

        jt = JointTrajectory()  # header.stamp left at 0 = "start when received" if ever replayed
        jt.header.frame_id = self.base_link
        jt.joint_names = list(self.chain_joint_names)
        for wp in self._wps:
            pt = JointTrajectoryPoint()
            pt.positions = [float(x) for x in wp["q"]]
            sec = int(wp["t"])
            pt.time_from_start = Duration(sec=sec, nanosec=int((wp["t"] - sec) * 1e9))
            jt.points.append(pt)
        self.jpath_pub.publish(jt)

    def _write_file(self):
        if not self.wp_file:
            return
        try:
            tmp = self.wp_file + ".tmp"
            with open(tmp, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["index", "t_s", "source", "x", "y", "z", "qx", "qy", "qz", "qw"]
                           + self.chain_joint_names + ["gripper"])
                for i, wp in enumerate(self._wps):
                    w.writerow([i, f"{wp['t']:.3f}", wp["source"]]
                               + [f"{x:.6f}" for x in wp["pos"]]
                               + [f"{x:.6f}" for x in wp["quat"]]
                               + [f"{x:.6f}" for x in wp["q"]]
                               + [wp["gripper"]])
            os.replace(tmp, self.wp_file)
        except Exception as e:
            if not self._file_error_logged:
                self._file_error_logged = True
                self.get_logger().error(f"Could not write {self.wp_file}: {e}")

    # =====================================================================
    # Output helpers
    # =====================================================================
    def _status(self, text: str):
        with self._lock:
            self._ui["msg"] = text

    def _publish_cmd(self, qdot: np.ndarray):
        if self._blocked:
            return
        msg = Float64MultiArray()
        msg.data = [float(x) for x in qdot]
        self.cmd_pub.publish(msg)

    def _publish_zero(self):
        if self.n_joints > 0:
            try:
                self._publish_cmd(np.zeros(self.n_joints))
            except Exception:
                pass

    def stop_and_save(self):
        """Zero velocity + final file write. Safe to call more than once."""
        for _ in range(3):
            self._publish_zero()
        if self._stopped_and_saved:
            return
        self._stopped_and_saved = True
        self._kb_running = False
        self._clear_keys()
        if self._wps:
            self._write_file()
            self.get_logger().info(f"Saved {len(self._wps)} waypoints to {self.wp_file or '(file disabled)'}")

    def destroy_node(self):
        self.stop_and_save()
        if self._kb_thread.is_alive():
            self._kb_thread.join(timeout=1.0)
        super().destroy_node()


def main():
    # Keep the ROS context alive on Ctrl+C so the zero-velocity command still goes out
    # (a forward velocity controller keeps the LAST command it got).
    try:
        from rclpy.signals import SignalHandlerOptions
        rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    except (ImportError, TypeError):
        rclpy.init()

    node = BuzzWireTeach()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    try:
        while rclpy.ok() and not node.finished:
            executor.spin_once(timeout_sec=0.05)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
            node.stop_and_save()
            time.sleep(0.05)
            executor.remove_node(node)
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
