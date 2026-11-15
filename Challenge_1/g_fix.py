#!/usr/bin/env python3
"""
gripper_fix.py  -  put the OpenMANIPULATOR-X gripper reading back where gripper.py / final_run.py expect it
=========================================================================================================

Problem: gripper_left_joint reads far outside the fingers' range (e.g. +0.042 or -0.032 m when fully open instead
of about +0.012 m), and the value jumps by a whole motor turn between restarts. The driver turns the gripper motor
angle into metres with a fixed line from the robot config (revolute_min/max -> prismatic_min/max). If the gripper
mechanism sits at a different motor angle than that line assumes, every reading - and every command - is off.

Fix: the gripper servo's Homing Offset (EEPROM) shifts the motor angle the servo reports. This script
  1. switches ONLY the gripper torque off (the arm keeps holding),
  2. asks you to hold the fingers fully CLOSED, then fully OPEN, and measures both,
  3. computes the Homing Offset that makes "fully open" read --open-target (gripper.py: open stop ~0.0123 m),
  4. writes it to servo 15 after you type y (the old value is printed so you can restore it).
Then: Ctrl+C hardware.launch.py, power OFF, fingers about half closed, power ON, start hardware.launch.py,
and check with   python3 gripper.py status   (fingers open by hand: about +0.012 m). Follow the printed POWER-ON RULE.

  python3 gripper_fix.py                    measure, show the proposal, write after 'y'
  python3 gripper_fix.py --measure-only     measure and show, write nothing
  python3 gripper_fix.py --restore 0        write Homing Offset 0 back (or any old value printed before)

Needs hardware.launch.py running. Nothing else may command the gripper meanwhile.
"""
import argparse
import math
import os
import re
import sys
import threading
import time

TICK_RAD = 0.0015339807878856412          # X-series: rad per position tick, zero at tick 2048 (driver model file)
TURN = 4096
ROBOTIS_DEFAULT = {"revolute_min": 0.92, "revolute_max": -1.52, "prismatic_min": -0.01, "prismatic_max": 0.019}


# ---------------- pure maths (tested offline) ----------------
def line(p):
    """slope (m/rad) and intercept (m) of the driver's revolute -> prismatic conversion"""
    slope = (p["prismatic_max"] - p["prismatic_min"]) / (p["revolute_max"] - p["revolute_min"])
    return slope, p["prismatic_min"] - slope * p["revolute_min"]


def m_to_ticks(m, p):
    slope, b = line(p)
    return ((m - b) / slope + math.pi) / TICK_RAD


def ticks_to_m(t, p):
    slope, b = line(p)
    return slope * (t * TICK_RAD - math.pi) + b


def to_int32(u):
    u = int(u) & 0xFFFFFFFF
    return u - (1 << 32) if u & 0x80000000 else u


def plan_offset(p_closed, p_open, old_offset, p, open_target):
    """-> dict with the new Homing Offset and the power-on rule.
    At power-on an X-series servo reports its motor angle within ONE turn (0..4095) + Homing Offset. If the finger
    range crosses the motor's 0/360 deg point, a power-on on the smaller side of that point comes up one turn off;
    the offset is chosen for the LARGER side and the rule says which finger position to avoid at power-on."""
    t_open, t_closed = m_to_ticks(p_open, p), m_to_ticks(p_closed, p)
    span = t_closed - t_open                              # ticks from open to closed (> 0 on the OMX)
    lo = (t_open - old_offset) % TURN                     # motor angle of "fully open" within one turn
    target = m_to_ticks(open_target, p)
    rule, wraps = "any finger position is fine at power-on", False
    ends = sorted([lo, lo + span])
    if ends[1] >= TURN or ends[0] < 0:                    # range crosses 0/360 deg
        wraps = True
        below, above = TURN - ends[0], ends[1] - TURN     # ticks of the range on each side of the crossing
        mm = lambda t: abs(t * TICK_RAD * line(p)[0]) * 1000
        if above > below:                                 # most of the range is past the crossing
            lo -= TURN
            avoid = "fully OPEN" if span > 0 else "fully CLOSED"
            rule = (f"switch the arm on with the fingers NOT {avoid}: close them a little (about half way). "
                    f"Only the last {mm(below):.1f} mm at {avoid.split()[1].lower()} comes up one turn off.")
        else:
            avoid = "fully CLOSED" if span > 0 else "fully OPEN"
            rule = (f"switch the arm on with the fingers NOT {avoid}: open them a little (about half way). "
                    f"Only the last {mm(above):.1f} mm at {avoid.split()[1].lower()} comes up one turn off.")
    new_offset = int(round(target - lo))
    return {"new_offset": new_offset, "span_ticks": span, "raw_open": lo, "wraps": wraps, "rule": rule,
            "open_after": ticks_to_m(lo + new_offset, p), "closed_after": ticks_to_m(lo + new_offset + span, p)}


# ---------------- ROS side ----------------
def make_node():
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, QoSProfile
    from sensor_msgs.msg import JointState
    from std_msgs.msg import String
    from dynamixel_interfaces.srv import GetDataFromDxl, SetDataToDxl

    class FixNode(Node):
        def __init__(self, joint):
            super().__init__("gripper_fix")
            self.joint, self.lock, self.samples, self.urdf = joint, threading.Lock(), [], None
            self.create_subscription(JointState, "/joint_states", self._js, 50)
            qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
            self.create_subscription(String, "/robot_description", lambda m: setattr(self, "urdf", m.data), qos)
            self.get_cli = self.create_client(GetDataFromDxl, "/dynamixel_hardware_interface/get_dxl_data")
            self.set_cli = self.create_client(SetDataToDxl, "/dynamixel_hardware_interface/set_dxl_data")
            self.Get, self.Set = GetDataFromDxl, SetDataToDxl

        def _js(self, m):
            if self.joint in m.name:
                with self.lock:
                    self.samples = (self.samples + [(time.time(), m.position[m.name.index(self.joint)])])[-400:]

        def mean_pos(self, seconds=1.0):
            time.sleep(seconds)
            with self.lock:
                v = [x for t, x in self.samples if time.time() - t < seconds]
            if not v:
                return None, None
            mu = sum(v) / len(v)
            return mu, max(abs(x - mu) for x in v)

        def _call(self, cli, req, timeout=3.0):
            fut = cli.call_async(req)
            t0 = time.time()
            while not fut.done() and time.time() - t0 < timeout:
                time.sleep(0.01)
            return fut.result() if fut.done() else None

        def get(self, dxl_id, item, tries=6):
            """Read one item. The driver can hand back the answer of an EARLIER request for the same item (a
            timed-out request stays in its buffer), so read until two answers in a row agree."""
            vals = []
            for _ in range(tries):
                req = self.Get.Request()
                req.id, req.item_name, req.timeout_sec = int(dxl_id), item, 1.0
                r = self._call(self.get_cli, req)
                if r is not None and r.result:
                    vals.append(int(r.item_data))
                    if len(vals) >= 2 and vals[-1] == vals[-2]:
                        return vals[-1]
                else:
                    time.sleep(0.3)
            return vals[-1] if vals else None

        def set(self, dxl_id, item, value):
            req = self.Set.Request()
            req.id, req.item_name, req.item_data = int(dxl_id), item, int(value) & 0xFFFFFFFF
            r = self._call(self.set_cli, req)
            return r is not None and bool(r.result)

    rclpy.init()
    return rclpy, FixNode


def conv_params(urdf):
    found = {}
    for k in ROBOTIS_DEFAULT:
        m = re.search(r'<param\s+name="%s"\s*>\s*([-+0-9.eE]+)\s*<' % k, urdf or "")
        if m:
            found[k] = float(m.group(1))
    if len(found) == len(ROBOTIS_DEFAULT):
        return found, "from /robot_description"
    return dict(ROBOTIS_DEFAULT), "ROBOTIS defaults (not found in /robot_description)"


def main():
    ap = argparse.ArgumentParser(description="Fix the OMX gripper position reading (servo Homing Offset)",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--id", type=int, default=15, help="gripper Dynamixel ID (default 15)")
    ap.add_argument("--joint", default="gripper_left_joint")
    ap.add_argument("--open-target", type=float, default=0.0123,
                    help="reading wanted with the fingers fully open (m), gripper.py's open stop")
    ap.add_argument("--measure-only", action="store_true", help="measure and show, write nothing")
    ap.add_argument("--restore", type=int, default=None, metavar="OFFSET", help="write this Homing Offset and exit")
    a, _ = ap.parse_known_args()

    rclpy, FixNode = make_node()
    from rclpy.executors import SingleThreadedExecutor
    node = FixNode(a.joint)
    ex = SingleThreadedExecutor()
    ex.add_node(node)
    threading.Thread(target=ex.spin, daemon=True).start()
    try:
        if not (node.get_cli.wait_for_service(timeout_sec=5.0) and node.set_cli.wait_for_service(timeout_sec=5.0)):
            print("driver services not found - is hardware.launch.py running?")
            return 1
        if a.restore is not None:
            node.set(a.id, "Torque Enable", 0)
            time.sleep(0.5)
            ok = node.set(a.id, "Homing Offset", a.restore)
            time.sleep(0.8)
            back = node.get(a.id, "Homing Offset")
            time.sleep(1.0)
            print(f"Homing Offset of servo {a.id} -> {a.restore}: {'written' if ok else 'write FAILED'}, reads "
                  f"{None if back is None else to_int32(back)}. Power-cycle the arm and restart hardware.launch.py.")
            return 0 if ok else 1

        t0 = time.time()
        while (node.urdf is None or node.mean_pos(0.2)[0] is None) and time.time() - t0 < 5.0:
            time.sleep(0.1)
        if node.mean_pos(0.3)[0] is None:
            print(f"no {a.joint} on /joint_states - is hardware.launch.py running?")
            return 1
        p, src = conv_params(node.urdf)
        slope, b = line(p)
        print(f"driver conversion ({src}): motor {p['revolute_min']:+.2f} rad -> {p['prismatic_min']:+.3f} m, "
              f"{p['revolute_max']:+.2f} rad -> {p['prismatic_max']:+.3f} m  ({slope * 1000:+.2f} mm/rad)")
        mode = node.get(a.id, "Operating Mode")
        old = node.get(a.id, "Homing Offset")
        if old is None:
            print("could not read the Homing Offset of the gripper servo - try again (the driver answers from its "
                  "read cycle)")
            return 1
        old = to_int32(old)
        print(f"servo {a.id}: Operating Mode {mode}, Homing Offset {old}   <- write this down to undo: "
              f"python3 gripper_fix.py --restore {old}")
        if mode not in (3, 5):
            print(f"The gripper servo is in Operating Mode {mode}" + (" (VELOCITY)" if mode == 1 else "") +
                  " - it ignores position commands and its position counts whole turns, so the offset cannot be "
                  "set reliably. Fix the modes first:  python3 servo_info.py --fix   (then power-cycle with the "
                  "gripper open, relaunch, and run this again).")
            return 1

        print("\nSwitching the GRIPPER torque off (only the gripper - the arm keeps holding) ...")
        if not node.set(a.id, "Torque Enable", 0):
            print("could not switch the gripper torque off")
            return 1
        time.sleep(0.8)

        def measure(what):
            while True:
                input(f"\nHold the fingers FULLY {what} by hand (gently, to the stop), keep holding, press ENTER ...")
                mu, dev = node.mean_pos(1.0)
                if mu is None:
                    print("  no reading")
                    continue
                print(f"  {what.lower():6s}: {mu:+.4f} m" + (f"  (moved {dev * 1000:.1f} mm while measuring - "
                                                              f"hold still, measuring again)" if dev > 0.001 else ""))
                if dev <= 0.001:
                    return mu

        pc = measure("CLOSED")
        po = measure("OPEN")
        travel = po - pc
        print(f"\nfinger travel closed -> open: {travel * 1000:+.1f} mm (reading units)")
        if travel <= 0:
            print("The reading goes DOWN when the fingers open - the direction is reversed (servo Drive Mode, or the "
                  "fingers were held the other way round). Nothing written. Send me these two numbers.")
            return 1
        if not 0.008 <= travel <= 0.045:
            print("That travel is not plausible for the OMX gripper (expected about 15-25 mm). Nothing written. "
                  "Measure again, holding the fingers right at their stops.")
            return 1
        r = plan_offset(pc, po, old, p, a.open_target)
        print(f"proposal: Homing Offset {old} -> {r['new_offset']}  (servo {a.id})")
        print(f"  after a power-on with the fingers OPEN:  open reads {r['open_after']:+.4f} m, closed reads "
              f"{r['closed_after']:+.4f} m   (driver range {p['prismatic_min']:+.3f} .. {p['prismatic_max']:+.3f})")
        print(f"  POWER-ON RULE: {r['rule']}")
        if a.measure_only:
            print("\n--measure-only: nothing written.")
            return 0
        if input("\nWrite this Homing Offset to the gripper servo? (y/N) ").strip().lower() != "y":
            print("nothing written.")
            return 0
        ok = node.set(a.id, "Homing Offset", r["new_offset"])
        back, t0 = None, time.time()
        while time.time() - t0 < 4.0:                    # the driver writes it in its next cycle
            time.sleep(0.5)
            back = node.get(a.id, "Homing Offset")
            back = None if back is None else to_int32(back)
            if back == r["new_offset"]:
                break
        print(f"written: {ok}, servo now reports Homing Offset {back}")
        if back != r["new_offset"]:
            print("The servo did NOT take the new offset (look for write errors in the hardware.launch.py terminal). "
                  "Check with  python3 servo_info.py  after a power-cycle.")
            return 1
        print("\nNOW:  1) Ctrl+C hardware.launch.py   2) switch the 12 V power OFF   3) fingers about HALF closed\n"
              "      4) power ON   5) start hardware.launch.py   6) python3 gripper.py status  -> about "
              f"{a.open_target:+.4f} m with the fingers open\n"
              "      7) python3 gripper.py close ; python3 gripper.py open   - the fingers must move\n"
              f"To undo: python3 gripper_fix.py --restore {old}")
        return 0 if ok else 1
    finally:
        ex.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    os._exit(rc)        # skip rclpy's C++ destructors ('terminate called without an active exception')
