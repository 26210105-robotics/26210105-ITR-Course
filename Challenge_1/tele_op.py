#!/usr/bin/env python3
"""
OpenMANIPULATOR-X keyboard teleop + end-effector pose recorder.

Move the arm with the keyboard (or by hand with torque off), then press ENTER
to append the current end-effector position + orientation to a CSV file.

Requires hardware.launch.py to be running (arm_controller, joint_state_broadcaster,
robot_state_publisher).

Keys
  q / a   joint1 + / -        (base yaw)
  w / s   joint2 + / -        (shoulder)
  e / d   joint3 + / -        (elbow)
  r / f   joint4 + / -        (wrist pitch)
  [ / ]   halve / double the jog step
  t       toggle torque (OFF = move the arm by hand, ON = hold & jog)
  p       print current EE pose (does not save)
  ENTER   save current EE pose to CSV
  x       quit (Ctrl+C also works)

Parameters (ros2 run style: --ros-args -p name:=value)
  base_frame    (default: world)
  ee_frame      (default: end_effector_link)
  csv_path      (default: ee_poses.csv)
  step_deg      (default: 2.0)
  move_time     (default: 0.3)  seconds per jog move
  start_manual  (default: false) start with torque OFF (move the arm by hand)
"""

import csv
import math
import os
import select
import sys
import termios
import threading
import time
import tty
from datetime import datetime

import rclpy
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_srvs.srv import SetBool
from tf2_ros import Buffer, TransformException, TransformListener
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

ARM_JOINTS = ['joint1', 'joint2', 'joint3', 'joint4']

# Approximate soft limits (rad). Moves that would go further OUTSIDE these are
# blocked; if a joint already starts outside, you can still move it back inside.
# Check your URDF if you need exact values.
SOFT_LIMITS = {
    'joint1': (-math.pi, math.pi),
    'joint2': (-2.05, 1.57),
    'joint3': (-1.55, 1.53),
    'joint4': (-1.80, 2.00),
}

KEYMAP = {
    'q': ('joint1', +1), 'a': ('joint1', -1),
    'w': ('joint2', +1), 's': ('joint2', -1),
    'e': ('joint3', +1), 'd': ('joint3', -1),
    'r': ('joint4', +1), 'f': ('joint4', -1),
}

CSV_HEADER = [
    'sample', 'timestamp',
    'x_m', 'y_m', 'z_m',
    'qx', 'qy', 'qz', 'qw',
    'roll_deg', 'pitch_deg', 'yaw_deg',
] + [f'{j}_rad' for j in ARM_JOINTS]


def quat_to_rpy(x, y, z, w):
    """Quaternion -> roll, pitch, yaw (radians, ZYX convention)."""
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    s = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    pitch = math.asin(s)
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


class EEPoseTeleop(Node):
    def __init__(self):
        super().__init__('omx_ee_teleop')

        self.base_frame = self.declare_parameter('base_frame', 'world').value
        self.ee_frame = self.declare_parameter('ee_frame', 'end_effector_link').value
        self.csv_path = os.path.abspath(
            self.declare_parameter('csv_path', 'ee_poses.csv').value)
        self.step = math.radians(float(self.declare_parameter('step_deg', 2.0).value))
        self.move_time = float(self.declare_parameter('move_time', 0.3).value)
        self.start_manual = bool(self.declare_parameter('start_manual', False).value)

        self.lock = threading.Lock()
        self.joint_pos = {}
        self.target = None          # commanded arm joint positions
        self.torque_on = True       # hardware.launch.py enables torque at start
        self.sample = self._count_existing_samples()

        self.create_subscription(JointState, '/joint_states', self._js_cb, 10)
        self.traj_pub = self.create_publisher(
            JointTrajectory, '/arm_controller/joint_trajectory', 10)
        self.torque_cli = self.create_client(
            SetBool, '/dynamixel_hardware_interface/set_dxl_torque')

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

    # ---------- state ----------
    def _js_cb(self, msg):
        with self.lock:
            for name, pos in zip(msg.name, msg.position):
                self.joint_pos[name] = pos

    def current_arm(self):
        with self.lock:
            if all(j in self.joint_pos for j in ARM_JOINTS):
                return [self.joint_pos[j] for j in ARM_JOINTS]
        return None

    # ---------- motion ----------
    def _send_target(self):
        msg = JointTrajectory()
        msg.joint_names = ARM_JOINTS
        pt = JointTrajectoryPoint()
        pt.positions = list(self.target)
        pt.time_from_start = Duration(seconds=self.move_time).to_msg()
        msg.points = [pt]
        self.traj_pub.publish(msg)

    def jog(self, joint, sign):
        if not self.torque_on:
            return 'Torque is OFF - press t to turn it on before jogging.'
        if self.target is None:
            cur = self.current_arm()
            if cur is None:
                return 'No joint states yet.'
            self.target = cur
        i = ARM_JOINTS.index(joint)
        old = self.target[i]
        new = old + sign * self.step
        lo, hi = SOFT_LIMITS[joint]
        if (new > hi and new > old) or (new < lo and new < old):
            return f'{joint} soft limit reached ({math.degrees(old):.1f} deg).'
        self.target[i] = new
        self._send_target()
        return f'{joint} -> {math.degrees(new):7.2f} deg'

    def set_torque(self, on):
        if not self.torque_cli.wait_for_service(timeout_sec=1.0):
            return 'Torque service not available.'
        if on:
            # Command the CURRENT position first so the arm doesn't snap back
            # to an old target when torque comes on.
            cur = self.current_arm()
            if cur is None:
                return 'No joint states yet.'
            self.target = cur
            self._send_target()
            time.sleep(0.2)
        future = self.torque_cli.call_async(SetBool.Request(data=on))
        t0 = time.time()
        while not future.done() and time.time() - t0 < 3.0:
            time.sleep(0.02)
        if not future.done():
            return 'Torque service call timed out.'
        res = future.result()
        if res.success:
            self.torque_on = on
            if not on:
                self.target = None  # re-sync from measured pose next time
        return f'Torque {"ON" if on else "OFF"}: {res.message or ("ok" if res.success else "failed")}'

    # ---------- pose + CSV ----------
    def get_ee_pose(self):
        try:
            tf = self.tf_buffer.lookup_transform(
                self.base_frame, self.ee_frame, rclpy.time.Time())
        except TransformException as ex:
            self.get_logger().warn(f'TF lookup failed: {ex}')
            return None
        t, q = tf.transform.translation, tf.transform.rotation
        rpy = quat_to_rpy(q.x, q.y, q.z, q.w)
        return (t.x, t.y, t.z), (q.x, q.y, q.z, q.w), rpy

    def _count_existing_samples(self):
        if not os.path.exists(self.csv_path):
            return 0
        with open(self.csv_path) as f:
            return max(0, sum(1 for _ in f) - 1)

    def record(self):
        pose = self.get_ee_pose()
        joints = self.current_arm()
        if pose is None or joints is None:
            return 'Could not read pose - nothing saved.'
        (x, y, z), (qx, qy, qz, qw), (r, p, yw) = pose
        self.sample += 1
        new_file = not os.path.exists(self.csv_path)
        with open(self.csv_path, 'a', newline='') as f:
            w = csv.writer(f)
            if new_file:
                w.writerow(CSV_HEADER)
            w.writerow([
                self.sample, datetime.now().isoformat(timespec='milliseconds'),
                f'{x:.5f}', f'{y:.5f}', f'{z:.5f}',
                f'{qx:.5f}', f'{qy:.5f}', f'{qz:.5f}', f'{qw:.5f}',
                f'{math.degrees(r):.3f}', f'{math.degrees(p):.3f}', f'{math.degrees(yw):.3f}',
            ] + [f'{j:.5f}' for j in joints])
        return f'Saved sample #{self.sample}  ' + self.format_pose(pose)

    @staticmethod
    def format_pose(pose):
        (x, y, z), _, (r, p, yw) = pose
        return (f'pos [m] x={x:+.4f} y={y:+.4f} z={z:+.4f} | '
                f'rpy [deg] r={math.degrees(r):+7.2f} p={math.degrees(p):+7.2f} '
                f'y={math.degrees(yw):+7.2f}')


def main():
    rclpy.init()
    node = EEPoseTeleop()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()

    print(__doc__.split('Parameters')[0])
    print(f'Frames: {node.base_frame} -> {node.ee_frame}')
    print(f'CSV:    {node.csv_path}')
    print(f'Step:   {math.degrees(node.step):.2f} deg\n')

    print('Waiting for /joint_states ...')
    t0 = time.time()
    while node.current_arm() is None:
        if time.time() - t0 > 10.0:
            print('No /joint_states after 10 s. Is hardware.launch.py running?')
            break
        time.sleep(0.1)

    if node.start_manual:
        print('Starting in MANUAL mode - support the arm! Torque off in 2 s ...')
        time.sleep(2.0)   # time to grab the arm
        print(node.set_torque(False))
        print('Move the arm by hand, press ENTER to save a pose, t to re-enable torque.')

    settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())
        while rclpy.ok():
            ready, _, _ = select.select([sys.stdin], [], [], 0.1)
            if not ready:
                continue
            key = sys.stdin.read(1)
            if key in ('\n', '\r'):
                print(node.record())
            elif key in KEYMAP:
                print(node.jog(*KEYMAP[key]))
            elif key == '[':
                node.step = max(math.radians(0.1), node.step / 2)
                print(f'Step: {math.degrees(node.step):.2f} deg')
            elif key == ']':
                node.step = min(math.radians(20.0), node.step * 2)
                print(f'Step: {math.degrees(node.step):.2f} deg')
            elif key == 't':
                if node.torque_on:
                    print('Turning torque OFF - support the arm!')
                print(node.set_torque(not node.torque_on))
            elif key == 'p':
                pose = node.get_ee_pose()
                print(node.format_pose(pose) if pose else 'Pose not available yet.')
            elif key == 'x':
                break
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        print(f'\nDone. {node.sample} sample(s) in {node.csv_path}')
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
