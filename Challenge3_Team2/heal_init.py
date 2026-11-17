#!/usr/bin/env python3
"""
heal_init.py - bring the Heal end-effector to the INIT pose (from wherever it is now)

Same format as the pick & place script: it runs heal_dls_ik.py as a subprocess
through run_ik(). After the move it reads /end_effector_pose and /joint_states once
and reports how close it got. If it is still off, it tries again (MAX_ATTEMPTS).

The gripper is NOT touched (so a held buzz-wire loop is never dropped).

Run (with the velocity controller + kdl_fk_node launched):
    python3 heal_init.py
"""

import math
import os
import signal
import subprocess
import time
from pathlib import Path

import yaml


IK_SCRIPT_PATH = Path(
    "/home/iitgn-robotics/Debojit_WS/addverb_ws/src/iitgn_robotics_lab_wd/heal_dls_ik.py"
)
VELOCITY_TOPICS = ["/velocity_controller/commands", "/velocity/command"]
POSE_TOPIC = "/end_effector_pose"
JOINT_STATES_TOPIC = "/joint_states"

# =========================
# INIT pose (recorded from /end_effector_pose, frame base_link)
# =========================
INIT_POS = [0.2817030190556593, 0.5971603072677625, 0.2694676920097109]
# quaternion in the order the topic prints it: x, y, z, w (same as your pick & place poses)
INIT_QUAT = [-0.06393936413803344, 0.7621708406457605, 0.6363503084967654, -0.10032772427913529]

# Joint angles recorded at the same pose (/joint_states), used only to CHECK the arm
# ended in the same configuration (IK can reach the same pose with a flipped elbow/wrist).
INIT_JOINTS = {
    "joint1": -0.855372,
    "joint2": 0.721719,
    "joint3": -0.039995,
    "joint4": -0.328716,
    "joint5": 2.04995,
    "joint6": -0.107696,
}

MAX_WAIT_S = 30          # per IK attempt
MAX_ATTEMPTS = 2         # retry once if not close enough
POS_TOL_M = 0.003        # 3 mm
ANG_TOL_DEG = 2.0
JOINT_TOL_RAD = 0.15     # ~8.6 deg: larger means a different arm configuration


# =========================
# Helpers (same as pick & place)
# =========================

def cleanup_stale_ik_processes():
    # Ensure no previous IK publishers are still alive before starting.
    subprocess.run(
        ["pkill", "-f", str(IK_SCRIPT_PATH)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    time.sleep(0.5)


def get_topic_publisher_count(topic: str):
    try:
        result = subprocess.run(
            ["ros2", "topic", "info", topic],
            capture_output=True, text=True, timeout=2, check=False,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    for line in result.stdout.splitlines():
        line = line.strip()
        if line.startswith("Publisher count:"):
            try:
                return int(line.split(":", 1)[1].strip())
            except ValueError:
                return None
    return None


def wait_for_topic_publishers_to_clear(timeout_s=5.0):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        counts = [get_topic_publisher_count(topic) for topic in VELOCITY_TOPICS]
        active_counts = [c for c in counts if c is not None]
        if not active_counts or all(c == 0 for c in active_counts):
            return True
        time.sleep(0.2)
    return False


def run_ik(target_pos, target_quat, max_wait=20):
    cleanup_stale_ik_processes()
    if not wait_for_topic_publishers_to_clear(timeout_s=6.0):
        print("Warning: existing velocity publishers still detected before IK start.")

    cmd = [
        "python3", str(IK_SCRIPT_PATH),
        str(target_pos[0]), str(target_pos[1]), str(target_pos[2]),
        str(target_quat[0]), str(target_quat[1]), str(target_quat[2]), str(target_quat[3]),
        str(max_wait),
    ]
    print(f"Moving to: {target_pos}")
    process = subprocess.Popen(cmd, start_new_session=True)

    try:
        process.wait(timeout=max_wait + 5)   # IK node exits when it converges
        print(f"IK process finished for target: {target_pos}")
    except KeyboardInterrupt:
        print("Interrupted by user, stopping IK process...")
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=2)
        raise
    except subprocess.TimeoutExpired:
        print(f"No convergence after {max_wait:.1f}s, stopping IK process for target: {target_pos}")
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()

    if not wait_for_topic_publishers_to_clear(timeout_s=6.0):
        print("Warning: velocity publisher endpoint still visible after IK stop.")
    time.sleep(1)


# =========================
# Check where we ended up
# =========================

def echo_once(topic, timeout_s=5.0):
    """One message from `ros2 topic echo --once`, as a dict (None if nothing arrived)."""
    try:
        r = subprocess.run(["ros2", "topic", "echo", "--once", topic],
                           capture_output=True, text=True, timeout=timeout_s, check=False)
    except subprocess.TimeoutExpired:
        return None
    if r.returncode != 0 or not r.stdout.strip():
        return None
    return yaml.safe_load(r.stdout.split("\n---")[0])


def angle_between_quats_deg(q1, q2):
    dot = abs(sum(a * b for a, b in zip(q1, q2)))
    n = math.sqrt(sum(a * a for a in q1)) * math.sqrt(sum(b * b for b in q2))
    return math.degrees(2.0 * math.acos(min(1.0, dot / n)))


def check_pose():
    """Returns (pos_err_m, ang_err_deg) or None if the pose topic is not available."""
    msg = echo_once(POSE_TOPIC)
    if msg is None:
        print(f"Could not read {POSE_TOPIC} (is kdl_fk_node running?) - skipping pose check.")
        return None
    p, o = msg["pose"]["position"], msg["pose"]["orientation"]
    pos = [p["x"], p["y"], p["z"]]
    quat = [o["x"], o["y"], o["z"], o["w"]]
    pos_err = math.dist(pos, INIT_POS)
    ang_err = angle_between_quats_deg(quat, INIT_QUAT)
    print(f"Now at   : [{pos[0]:+.4f}, {pos[1]:+.4f}, {pos[2]:+.4f}] m")
    print(f"Target   : [{INIT_POS[0]:+.4f}, {INIT_POS[1]:+.4f}, {INIT_POS[2]:+.4f}] m")
    print(f"Error    : {pos_err * 1000:.1f} mm, {ang_err:.2f} deg")
    return pos_err, ang_err


def check_joints():
    msg = echo_once(JOINT_STATES_TOPIC)
    if msg is None:
        print(f"Could not read {JOINT_STATES_TOPIC} - skipping joint check.")
        return
    now = dict(zip(msg["name"], msg["position"]))
    worst = 0.0
    for name, target in INIT_JOINTS.items():
        if name not in now:
            continue
        d = math.atan2(math.sin(now[name] - target), math.cos(now[name] - target))
        worst = max(worst, abs(d))
        print(f"  {name}: {now[name]:+.4f}  (recorded {target:+.4f}, diff {math.degrees(d):+6.1f} deg)")
    if worst > JOINT_TOL_RAD:
        print("WARNING: pose reached, but the arm is in a DIFFERENT joint configuration "
              "than the recorded one (largest joint difference "
              f"{math.degrees(worst):.0f} deg). Check before starting the buzz-wire run.")
    else:
        print("Joint configuration matches the recorded one.")


# =========================
# MAIN
# =========================

def main():
    print("Bringing end-effector to INIT pose...")
    for attempt in range(1, MAX_ATTEMPTS + 1):
        print(f"\n--- Attempt {attempt}/{MAX_ATTEMPTS} ---")
        run_ik(INIT_POS, INIT_QUAT, max_wait=MAX_WAIT_S)
        err = check_pose()
        if err is None:
            break
        if err[0] <= POS_TOL_M and err[1] <= ANG_TOL_DEG:
            print("INIT pose reached.")
            break
        if attempt < MAX_ATTEMPTS:
            print("Not close enough yet, running IK again from here...")
    else:
        print(f"WARNING: still {err[0] * 1000:.1f} mm / {err[1]:.1f} deg away after "
              f"{MAX_ATTEMPTS} attempts.")

    print("\nJoint check:")
    check_joints()
    print("Done!")


if __name__ == "__main__":
    main()
