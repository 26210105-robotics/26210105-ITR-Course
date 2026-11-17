#!/usr/bin/env python3
"""
servo_info.py  -  what the five OpenMANIPULATOR-X servos really are set to (read straight from each servo)

  python3 servo_info.py          table: Operating Mode, Homing Offset, Torque, Hardware Error, position   (read-only)
  python3 servo_info.py --fix    puts a WRONG Operating Mode back to the standard one, after you type y:
                                 joints 11-14 -> 3 (position; joint1 may stay 4 = multi-turn, final_run uses it)
                                 gripper 15   -> 5 (current-based position)

Operating Mode 1 (velocity) or 0 / 16 make a servo IGNORE position commands: ROS thinks the move ran, nothing moves.
Writing a mode switches that servo's torque OFF (the driver does it first): REST THE ARM ON THE TABLE before --fix.
Needs hardware.launch.py running.
"""
import argparse
import os
import sys
import threading
import time

IDS = {11: "joint1", 12: "joint2", 13: "joint3", 14: "joint4", 15: "gripper"}
MODES = {0: "current", 1: "VELOCITY", 3: "position", 4: "multi-turn position", 5: "current-based position",
         16: "PWM"}
OK_MODES = {11: (3, 4), 12: (3,), 13: (3,), 14: (3,), 15: (5, 3)}
FIX_TO = {11: 3, 12: 3, 13: 3, 14: 3, 15: 5}
HW_BITS = {1: "input voltage", 4: "overheating", 8: "encoder", 16: "electrical shock", 32: "OVERLOAD"}


def to_int32(u):
    u = int(u) & 0xFFFFFFFF
    return u - (1 << 32) if u & 0x80000000 else u


def make_node():
    import rclpy
    from rclpy.node import Node
    from dynamixel_interfaces.srv import GetDataFromDxl, SetDataToDxl

    class N(Node):
        def __init__(self):
            super().__init__("servo_info")
            self.g = self.create_client(GetDataFromDxl, "/dynamixel_hardware_interface/get_dxl_data")
            self.s = self.create_client(SetDataToDxl, "/dynamixel_hardware_interface/set_dxl_data")
            self.Get, self.Set = GetDataFromDxl, SetDataToDxl

        def _call(self, cli, req, timeout=3.0):
            fut = cli.call_async(req)
            t0 = time.time()
            while not fut.done() and time.time() - t0 < timeout:
                time.sleep(0.01)
            return fut.result() if fut.done() else None

        def get(self, i, item, tries=4):
            for _ in range(tries):
                req = self.Get.Request()
                req.id, req.item_name, req.timeout_sec = int(i), item, 1.0
                r = self._call(self.g, req)
                if r is not None and r.result:
                    return int(r.item_data)
                time.sleep(0.3)
            return None

        def set(self, i, item, value):
            req = self.Set.Request()
            req.id, req.item_name, req.item_data = int(i), item, int(value) & 0xFFFFFFFF
            r = self._call(self.s, req)
            return r is not None and bool(r.result)

    rclpy.init()
    return rclpy, N()


def table(n):
    rows = {}
    print("\n ID  servo    Operating Mode               Homing Offset  Torque  HW error        position (raw)")
    for i, name in IDS.items():
        mode = n.get(i, "Operating Mode")
        off = n.get(i, "Homing Offset")
        tq = n.get(i, "Torque Enable")
        hw = n.get(i, "Hardware Error Status")
        pos = n.get(i, "Present Position")
        off = None if off is None else to_int32(off)
        pos = None if pos is None else to_int32(pos)
        rows[i] = mode
        mtxt = "no answer" if mode is None else f"{mode} {MODES.get(mode, '?')}"
        flag = "" if mode is None or mode in OK_MODES[i] else "  <-- WRONG"
        hwt = "-" if hw is None else ("none" if hw == 0 else "+".join(t for b, t in HW_BITS.items() if hw & b) or str(hw))
        ptxt = "-" if pos is None else f"{pos:+7d}  ({pos / 4096.0:+.2f} turns)"
        print(f" {i}  {name:8s} {mtxt:28s} {str(off):>13s}  {str(tq):>6s}  {hwt:14s}  {ptxt}{flag}")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fix", action="store_true", help="put wrong Operating Modes back to standard (asks first)")
    a, _ = ap.parse_known_args()
    rclpy, n = make_node()
    from rclpy.executors import SingleThreadedExecutor
    ex = SingleThreadedExecutor()
    ex.add_node(n)
    threading.Thread(target=ex.spin, daemon=True).start()
    code = 0
    try:
        if not (n.g.wait_for_service(timeout_sec=5.0) and n.s.wait_for_service(timeout_sec=5.0)):
            print("driver services not found - is hardware.launch.py running?")
            return 1
        rows = table(n)
        wrong = {i: m for i, m in rows.items() if m is not None and m not in OK_MODES[i]}
        print("\nexpected: joints 11-14 mode 3 (joint1 may be 4), gripper 15 mode 5 (or 3); Homing Offset 0 unless "
              "you set one; HW error none")
        if not wrong:
            print("Operating Modes are fine.")
            return 0
        print("WRONG modes: " + ", ".join(f"{IDS[i]} (ID {i}) = {m} {MODES.get(m, '?')}" for i, m in wrong.items())
              + "\n -> these servos ignore position commands. Fix with:  python3 servo_info.py --fix")
        if not a.fix:
            return 2
        print("\nWill write: " + ", ".join(f"ID {i} -> {FIX_TO[i]} ({MODES[FIX_TO[i]]})" for i in wrong) +
              "\nEach of these servos loses torque while its mode is written. REST THE ARM ON THE TABLE (or hold it).")
        if input("Write these Operating Modes? (y/N) ").strip().lower() != "y":
            print("nothing written.")
            return 0
        for i in wrong:
            ok = n.set(i, "Operating Mode", FIX_TO[i])
            print(f"  ID {i}: write {'queued' if ok else 'REFUSED'}")
            time.sleep(0.5)
        time.sleep(1.0)
        rows = table(n)
        still = [i for i in wrong if rows.get(i) != FIX_TO[i]]
        if still:
            print(f"\nNOT changed: IDs {still}. Look at the hardware.launch.py terminal for write errors; the modes "
                  f"can also be set with DYNAMIXEL Wizard 2.0.")
            code = 1
        else:
            print("\nModes fixed (stored in the servos).")
        print("NOW: 1) Ctrl+C hardware.launch.py  2) power OFF  3) open the gripper fully by hand  4) power ON\n"
              "     5) start hardware.launch.py  6) python3 servo_info.py   (check)  7) python3 gripper.py status")
        return code
    finally:
        ex.shutdown()
        n.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    os._exit(rc)        # skip rclpy's C++ destructors ('terminate called without an active exception')
