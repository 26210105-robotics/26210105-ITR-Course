#!/usr/bin/env python3
"""
gripper.py  -  OpenMANIPULATOR-X gripper: open() and close() that never lock up
===============================================================================

REVERSED GRIPPER (reverse=True, the default here): on this robot a BIGGER gripper_left_joint reading means
MORE CLOSED (the ROBOTIS driver assumes the opposite), so the plain controller commands open and close the
wrong way round. gripper.py mirrors every position:  sent to the controller = mirror - value,  and
position() = mirror - reading. Everything below (open_pos, close_travel, position(), ...) is therefore in
the NORMAL sense: bigger = more open, open stop ~ +0.0123, closing towards -0.010. mirror = 0.0026 m puts
this gripper's closed stop (reading ~ +0.0126) at -0.010. Commands sent are always kept inside the driver
range -0.010 .. +0.019 m. A gripper that is not reversed: reverse=False (terminal: --normal).

Why a separate driver: commanding the SRDF limits (open 0.019 m, close -0.010 m) makes the
gripper Dynamixel push forever - against the mechanical open stop (~0.0123 m) or against the
peg - until it trips its Overload error (red LED blinking) and ignores every command.

  open()   goes to open_pos (0.0118 m, just inside the open stop) and CHECKS on /joint_states
           that the fingers really opened. If the stop is lower than open_pos, the command is
           eased back to where the fingers stopped, so the motor never keeps pushing.
  close()  moves the fingers only close_travel from open (target = open_pos - close_travel),
           never all the way. If the peg stops the fingers earlier, the command is eased to
           (contact - max_squeeze) so the motor presses lightly instead of at full force.

Both return 'ok', 'estop' or 'error: <reason>'.

From Python (the node must be spinning in an executor, as in peg_sort.py):
    from gripper import Gripper
    g = Gripper(node, estop=threading.Event(), close_travel=0.010)
    g.wait_ready()
    g.open()
    g.close()

From a terminal (hardware.launch.py running):
    python3 gripper.py open
    python3 gripper.py close
    python3 gripper.py close --travel 0.008      # clamp less / more
    python3 gripper.py status                    # print finger position (and the raw reading)
    python3 gripper.py open --normal             # for a gripper whose reading is NOT reversed
"""

import argparse
import threading
import time

from action_msgs.msg import GoalStatus
from control_msgs.action import GripperCommand
from rclpy.action import ActionClient
from sensor_msgs.msg import JointState

DEFAULTS = dict(
    action='/gripper_controller/gripper_cmd',
    joint_states='/joint_states',
    joint='gripper_left_joint',
    open_pos=0.0118,      # m  commanded open: just inside the mechanical open stop (~0.0123)
    close_travel=0.010,   # m  how far the fingers move from open_pos when clamping
    close_limit=-0.010,   # m  never command below this (SRDF "close")
    max_squeeze=0.002,    # m  pressed past the contact point when the peg stops the fingers
    effort=10.0,          #    GripperCommand max_effort
    open_ok=0.010,        # m  open() succeeds when measured >= this ...
    min_move=0.002,       # m  ... and the fingers moved at least this (or were already open)
    timeout=3.0,          # s  per motion
    retries=1,            #    extra open attempts
    verbose=True,
    reverse=True,         #    True: this gripper's reading grows when it CLOSES - mirror all positions
    mirror=0.0026,        # m  reversed: sent = mirror - position, position = mirror - reading
)
RAW_RANGE = (-0.010, 0.019)   # m  driver / SRDF range of gripper_left_joint: commands are clipped to it

STATUS = {1: 'ACCEPTED', 2: 'EXECUTING', 3: 'CANCELING', 4: 'SUCCEEDED', 5: 'CANCELED', 6: 'ABORTED'}


def _f(v):
    """Format a position [m] as '+0.0123' for log lines ('n/a' for None)."""
    return 'n/a' if v is None else f'{v:+.4f}'


class Gripper:
    """Open / close the OpenMANIPULATOR-X gripper via the GripperCommand action, with feedback from /joint_states.

    All positions are in the NORMAL sense (bigger = more open). For a reversed gripper they are mirrored
    on the way to and from the controller (see to_raw / from_raw).
    """
    def __init__(self, node, estop=None, **cfg):
        """Args:
            node:  rclpy node; it must be spinning in an executor.
            estop: threading.Event that aborts every wait when set (a new one if None).
            **cfg: overrides of DEFAULTS; unknown keys raise ValueError.
        """
        unknown = set(cfg) - set(DEFAULTS)
        if unknown:
            raise ValueError(f'unknown gripper settings: {sorted(unknown)}')
        self.c = dict(DEFAULTS, **cfg)
        self.node = node
        self.estop = estop if estop is not None else threading.Event()
        self.client = ActionClient(node, GripperCommand, self.c['action'])
        self._lock = threading.Lock()
        self._pos = None
        self._active = []
        node.create_subscription(JointState, self.c['joint_states'], self._on_js, 20)

    # ---------- state -----------------------------------------------------------
    def _on_js(self, msg):
        """/joint_states callback: store the raw gripper joint reading."""
        if self.c['joint'] in msg.name:
            with self._lock:
                self._pos = msg.position[msg.name.index(self.c['joint'])]

    def to_raw(self, position):
        """Normal-sense position -> value sent to the controller (mirrored if reverse), inside RAW_RANGE."""
        r = self.c['mirror'] - position if self.c['reverse'] else position
        return min(max(r, RAW_RANGE[0]), RAW_RANGE[1])

    def from_raw(self, reading):
        """Driver reading -> normal-sense position (mirrored if reverse); None stays None."""
        return None if reading is None else (self.c['mirror'] - reading if self.c['reverse'] else reading)

    def raw_position(self):
        """gripper_left_joint as the driver reports it [m] (not mirrored) or None."""
        with self._lock:
            return self._pos

    def position(self):
        """Measured finger position [m] in the normal sense (bigger = more open) or None."""
        return self.from_raw(self.raw_position())

    @property
    def close_target(self):
        """Clamp target [m, normal sense]: open_pos - close_travel, never below close_limit."""
        return max(self.c['close_limit'], self.c['open_pos'] - self.c['close_travel'])

    def wait_ready(self, timeout=5.0):
        """True when the action server and the finger position are available."""
        if not self.client.wait_for_server(timeout_sec=timeout):
            return False
        t0 = time.monotonic()
        while self.position() is None and time.monotonic() - t0 < timeout:
            time.sleep(0.05)
        return self.position() is not None

    def cancel(self):
        """Cancel any running gripper goal (call from an E-stop)."""
        with self._lock:
            for gh in list(self._active):
                try:
                    gh.cancel_goal_async()
                except Exception:
                    pass

    def _say(self, text):
        """Print text if the 'verbose' setting is on."""
        if self.c['verbose']:
            print(text, flush=True)

    # ---------- low level -------------------------------------------------------
    def _wait(self, fut, timeout):
        """Wait for an rclpy future. -> True when done, False on E-stop or after timeout seconds."""
        t0 = time.monotonic()
        while not fut.done():
            if self.estop.is_set() or time.monotonic() - t0 > timeout:
                return False
            time.sleep(0.005)
        return True

    def _goal(self, position):
        """Send one goal (normal-sense position) and wait for its result -> ('done', result) | ('estop'|'error: ..', None)."""
        return self._goal_raw(self.to_raw(position))

    def _goal_raw(self, raw):
        """Send one goal exactly as given to the controller (no mirroring)."""
        if self.estop.is_set():
            return 'estop', None
        goal = GripperCommand.Goal()
        goal.command.position = float(raw)
        goal.command.max_effort = float(self.c['effort'])
        fut = self.client.send_goal_async(goal)
        if not self._wait(fut, 5.0):
            return ('estop' if self.estop.is_set() else 'error: gripper goal not acknowledged'), None
        gh = fut.result()
        if not gh.accepted:
            return 'error: gripper_controller rejected the goal', None
        with self._lock:
            self._active.append(gh)
        try:
            rfut = gh.get_result_async()
            if not self._wait(rfut, self.c['timeout'] + 5.0):
                return ('estop' if self.estop.is_set() else 'error: gripper result timed out'), None
            res = rfut.result()
        finally:
            with self._lock:
                if gh in self._active:
                    self._active.remove(gh)
        if self.estop.is_set():
            return 'estop', None
        return 'done', res

    @staticmethod
    def _desc(res):
        """Short text of a GripperCommand result: controller status + stalled / reached flags."""
        r = res.result
        return (f'controller {STATUS.get(res.status, res.status)}'
                + (', stalled' if r.stalled else '') + (', reached' if r.reached_goal else ''))

    def _wait_still(self, done, still=0.0002, settle=0.2):
        """Wait until done(pos), or the fingers stop moving (stop / peg), or timeout."""
        t0 = time.monotonic()
        last, t_last = self.position(), time.monotonic()
        while time.monotonic() - t0 < self.c['timeout'] and not self.estop.is_set():
            time.sleep(0.02)
            p = self.position()
            if p is None:
                continue
            if done(p):
                return p
            if last is None or abs(p - last) > still:
                last, t_last = p, time.monotonic()
            elif time.monotonic() - t_last > settle and time.monotonic() - t0 > 0.4:
                return p
        return self.position()

    # ---------- the two functions -----------------------------------------------
    def open(self):
        """Open to open_pos, verified. Never leaves the motor pushing on the open stop."""
        c = self.c
        target = c['open_pos']
        p0 = self.position()
        p = p0
        for k in range(1 + max(0, int(c['retries']))):
            st, res = self._goal(target)
            if st != 'done':
                return st
            p = self._wait_still(lambda v: v >= target - 0.0003)
            if self.estop.is_set():
                return 'estop'
            self._say(f'      gripper open : {_f(p0)} -> {_f(p)} m   ({self._desc(res)})')
            if p is None:                           # no feedback: trust the controller
                if res.status == GoalStatus.STATUS_SUCCEEDED and not res.result.stalled:
                    return 'ok'
            elif p >= c['open_ok'] and (p0 is None or p0 >= c['open_ok'] or p - p0 >= c['min_move']):
                if p < target - 0.0005:             # stopped on a lower end stop: stop pushing
                    rest = p - 0.0005
                    if self._goal(rest)[0] == 'done':
                        self._say(f'      gripper rest : end stop at {p:+.4f} m, command eased to {rest:+.4f} m')
                return 'ok'
            if k < int(c['retries']):
                self._say('      fingers did not open - sending open again ...')
        return (f'error: gripper did NOT open (fingers at {_f(p)} m, need >= {c["open_ok"]:+.4f} m). '
                f'If the gripper LED blinks red the motor tripped (overload): restart '
                f'hardware.launch.py (it reboots the motor) or power-cycle.')

    def close(self):
        """Clamp: move only close_travel from open; press lightly if the peg stops the fingers."""
        c = self.c
        target = self.close_target
        p0 = self.position()
        st, res = self._goal(target)
        if st != 'done':
            return st
        p = self._wait_still(lambda v: v <= target + 0.0003)
        if self.estop.is_set():
            return 'estop'
        self._say(f'      gripper close: {_f(p0)} -> {_f(p)} m   (target {target:+.4f}, {self._desc(res)})')
        if p is None:
            r = res.result
            ok = res.status == GoalStatus.STATUS_SUCCEEDED or r.reached_goal or r.stalled
            return 'ok' if ok else f'error: gripper did not close ({self._desc(res)})'
        if p > target + 0.0005:                     # stopped early: on the peg (or not moving at all)
            if p0 is not None and p0 - p < c['min_move']:
                return (f'error: gripper did not move (fingers at {p:+.4f} m). If its LED blinks red '
                        f'the motor tripped (overload): restart hardware.launch.py or power-cycle.')
            hold = max(target, p - c['max_squeeze'])
            if hold > target + 1e-4:
                st, _ = self._goal(hold)
                if st != 'done':
                    return st
                self._say(f'      gripper hold : peg at {p:+.4f} m, command eased to {hold:+.4f} m '
                          f'(light press {c["max_squeeze"] * 1000:.1f} mm)')
        else:
            self._say('      gripper close: reached the clamp target with no peg contact - '
                      'if a peg should be there, increase close_travel')
        return 'ok'


# =========================================================================== #
#  Stand-alone use:  python3 gripper.py open | close | status                  #
# =========================================================================== #
def main():
    """Command-line entry: python3 gripper.py open | close | status. -> process exit code."""
    import rclpy
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.node import Node

    ap = argparse.ArgumentParser(description='OpenMANIPULATOR-X gripper open / close')
    ap.add_argument('command', choices=['open', 'close', 'status'])
    ap.add_argument('--travel', type=float, default=DEFAULTS['close_travel'],
                    help='close: finger travel from open [m] (default %(default)s)')
    ap.add_argument('--open-pos', type=float, default=DEFAULTS['open_pos'],
                    help='open position [m] (default %(default)s)')
    ap.add_argument('--squeeze', type=float, default=DEFAULTS['max_squeeze'],
                    help='press past peg contact [m] (default %(default)s)')
    ap.add_argument('--normal', action='store_true',
                    help='the gripper reading is NOT reversed (bigger reading = more open): no mirroring')
    ap.add_argument('--mirror', type=float, default=DEFAULTS['mirror'],
                    help='reversed gripper: sent = mirror - position [m] (default %(default)s)')
    a, ros_args = ap.parse_known_args()

    rclpy.init(args=ros_args)
    node = Node('gripper_cli')
    ex = MultiThreadedExecutor(num_threads=2)
    ex.add_node(node)
    threading.Thread(target=ex.spin, daemon=True).start()
    try:
        g = Gripper(node, open_pos=a.open_pos, close_travel=a.travel, max_squeeze=a.squeeze,
                    reverse=not a.normal, mirror=a.mirror)
        if not g.wait_ready():
            print('gripper action server or /joint_states not available - is hardware.launch.py running?')
            return 1
        if a.command == 'status':
            print(f'gripper = {_f(g.position())} m   (open {g.c["open_pos"]:+.4f}, clamp target '
                  f'{g.close_target:+.4f})   raw gripper_left_joint = {_f(g.raw_position())} m'
                  + (f'  [reversed: position = {g.c["mirror"]:+.4f} - raw]' if g.c['reverse'] else ''))
            return 0
        res = g.open() if a.command == 'open' else g.close()
        print(f'{a.command}: {res}')
        return 0 if res == 'ok' else 1
    except KeyboardInterrupt:
        return 130
    finally:
        ex.shutdown(timeout_sec=1.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    raise SystemExit(main())
