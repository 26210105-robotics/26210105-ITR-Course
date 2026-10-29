import time
import mujoco
import mujoco.viewer
import numpy as np

MODEL_PATH = "FrPos.xml"
ARM_DOF = 7

# Designated Home Position
HOME_QPOS = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])

# --- Scene names (from Pos1.xml) ---
CUBE_BODY = "cube"
CUBE_GEOM = "cube_geom"
TRAY_BODY = "tray"
TRAY_FLOOR_GEOM = "tray_bottom"

HOVER_OFFSET = np.array([0.0, 0.0, 0.25])  # 25 cm directly above the target
PLACE_CLEARANCE = 0.005   # release the cube 5 mm above the tray floor

# Gripper States (velocity actuator on finger_joint1: + opens, - closes)
GRIPPER_OPEN = 0.1
GRIPPER_CLOSED = -0.2     # ctrlrange limit; with kv=20 this squeezes the cube with ~4 N
FINGER_OPEN_QPOS = 0.04   # finger joint range is 0..0.04 (fully open)

# Velocity control: joint velocity = quintic feed-forward + KP_VEL * position error
KP_VEL = 10.0

# Sequence execution parameters (no fixed timings)
MAX_JOINT_SPEED = 0.8    # rad/s, peak joint speed of each quintic segment
REACH_TOL = 0.01         # rad, max joint error to count a waypoint as reached
VEL_TOL = 0.05           # rad/s, arm must also be nearly still
GRIP_VEL_TOL = 1e-3      # finger speed below which the gripper is considered settled
GRIP_SETTLE_STEPS = 100  # gripper must stay settled this many steps

# DLS-IK parameters
DLS_DAMPING = 0.05    # lambda
DLS_MAX_STEP = 0.2    # max joint step per iteration (rad)
DLS_NULL_GAIN = 0.1   # pull toward HOME_QPOS in the null space


def orientation_error(target_quat, current_xmat):
    """World-frame rotation error (axis * angle) from current to target."""
    cur_quat = np.zeros(4)
    mujoco.mju_mat2Quat(cur_quat, current_xmat)
    cur_conj = np.zeros(4)
    mujoco.mju_negQuat(cur_conj, cur_quat)
    err_quat = np.zeros(4)
    mujoco.mju_mulQuat(err_quat, target_quat, cur_conj)
    err = np.zeros(3)
    mujoco.mju_quat2Vel(err, err_quat, 1.0)
    return err


def solve_ik_dls(model, ik_data, target_pos, target_quat, site_id, q_init,
                 q_lo, q_hi, max_iters=1000, pos_tol=1e-4, rot_tol=1e-3):
    """Damped Least Squares IK: dq = J^T (J J^T + lambda^2 I)^-1 e"""
    q = np.clip(np.copy(q_init), q_lo, q_hi)
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    I7 = np.eye(ARM_DOF)

    for _ in range(max_iters):
        ik_data.qpos[ARM_QPOS] = q
        mujoco.mj_kinematics(model, ik_data)
        mujoco.mj_comPos(model, ik_data)

        err_p = target_pos - ik_data.site_xpos[site_id]
        err_r = orientation_error(target_quat, ik_data.site_xmat[site_id])

        if np.linalg.norm(err_p) < pos_tol and np.linalg.norm(err_r) < rot_tol:
            break

        mujoco.mj_jacSite(model, ik_data, jacp, jacr, site_id)
        J = np.vstack([jacp[:, ARM_DOFS], jacr[:, ARM_DOFS]])   # 6x7
        e = np.concatenate([err_p, err_r])                        # 6

        # DLS step
        J_dls = J.T @ np.linalg.inv(J @ J.T + DLS_DAMPING**2 * np.eye(6))
        dq = J_dls @ e

        # Null-space posture term (keeps elbow near home)
        N = I7 - np.linalg.pinv(J, rcond=1e-4) @ J
        dq += N @ (DLS_NULL_GAIN * (HOME_QPOS - q))

        # Limit step size, then clip to joint limits
        n = np.linalg.norm(dq)
        if n > DLS_MAX_STEP:
            dq *= DLS_MAX_STEP / n
        q = np.clip(q + dq, q_lo, q_hi)

    pe = np.linalg.norm(target_pos - ik_data.site_xpos[site_id])
    print(f"   pos err = {pe * 1000:.3f} mm")
    return q


def timed_ik(label, *args):
    """Run solve_ik_dls and record how long it took."""
    print(f"Solving DLS-IK for {label}...")
    t0 = time.perf_counter()
    q = solve_ik_dls(*args)
    dt = time.perf_counter() - t0
    IK_TIMES[label] = dt
    print(f"   solve time = {dt * 1000:.2f} ms")
    return q


def quintic(q0, qf, s, T):
    """Quintic (minimum-jerk) polynomial: zero velocity and acceleration at both ends.
    s is the normalized progress in [0, 1], T the segment duration.
    Returns (position, velocity)."""
    s = min(max(s, 0.0), 1.0)
    scale = 10 * s**3 - 15 * s**4 + 6 * s**5
    dscale = (30 * s**2 - 60 * s**3 + 30 * s**4) / T
    return q0 + (qf - q0) * scale, (qf - q0) * dscale


def segment_duration(q0, qf):
    """Duration so the quintic's peak joint speed (1.875 * dq / T) stays at MAX_JOINT_SPEED."""
    return max(1.875 * np.max(np.abs(qf - q0)) / MAX_JOINT_SPEED, 0.2)


def print_summary(completed):
    """Print the two outcomes: IK solve time and overall simulation time."""
    sim_elapsed = data.time - SIM_T0
    wall_elapsed = time.perf_counter() - WALL_T0
    print("\n" + "=" * 52)
    print(" RESULTS")
    print("=" * 52)
    print(" 1) Path solving time (DLS-IK, offline)")
    for label, dt in IK_TIMES.items():
        print(f"      {label:<20s} {dt * 1000:9.2f} ms")
    print(f"      {'Total':<20s} {IK_TOTAL * 1000:9.2f} ms")
    status = "sequence complete" if completed else "viewer closed before completion"
    print(f" 2) Overall simulation time ({status})")
    print(f"      Simulated time     {sim_elapsed:9.3f} s")
    print(f"      Wall-clock time    {wall_elapsed:9.3f} s")
    if STEP_TIMES:
        print("      Per step (simulated):")
        for name, dt in STEP_TIMES:
            print(f"        {name:<18s} {dt:7.3f} s")
    print("=" * 52 + "\n")


# ------------------------------------------------------------
# Initialization & Offline Planning
# ------------------------------------------------------------
model = mujoco.MjModel.from_xml_path(MODEL_PATH)

# Velocity actuators (kv up to 500, armature 0.1) are unstable with the default
# explicit Euler integrator at dt = 0.002: kv*dt/inertia ~ 10 > 2, so the joints
# chatter in place. implicitfast integrates the actuator damping stably.
# (Same as adding <option integrator="implicitfast"/> to the XML.)
model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST

data = mujoco.MjData(model)
ik_data = mujoco.MjData(model)   # separate data so IK doesn't disturb the sim

site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripper")

# --- Arm joint indices, taken from the first 7 actuators ---
# (the arm is NOT assumed to be qpos[0:7]; a cube freejoint may come first)
ARM_JNT = np.array([model.actuator_trnid[i, 0] for i in range(ARM_DOF)])
ARM_QPOS = model.jnt_qposadr[ARM_JNT]     # indices into qpos
ARM_DOFS = model.jnt_dofadr[ARM_JNT]      # indices into qvel / qfrc / Jacobian columns

# --- Finger joints (joints with "finger" in their name) ---
FINGER_JNT = [j for j in range(model.njnt)
              if "finger" in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or "").lower()]
FINGER_QPOS = model.jnt_qposadr[FINGER_JNT]
FINGER_DOFS = model.jnt_dofadr[FINGER_JNT]

# Gripper actuator = the actuator right after the 7 arm actuators
GRIP_ACT = ARM_DOF

# Joint limits (unlimited joints are not clipped)
q_lo = np.where(model.jnt_limited[ARM_JNT], model.jnt_range[ARM_JNT, 0], -np.inf)
q_hi = np.where(model.jnt_limited[ARM_JNT], model.jnt_range[ARM_JNT, 1], np.inf)

# --- Read cube and tray poses from the XML ---
cube_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, CUBE_BODY)
cube_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, CUBE_GEOM)
tray_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, TRAY_BODY)
tray_floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, TRAY_FLOOR_GEOM)

mujoco.mj_forward(model, ik_data)                 # qpos0 = poses written in the XML
cube_half = model.geom_size[cube_geom, 2]         # 0.025 for the 5 cm cube

# Pick: grasp at the cube's centre
PICK_TARGET = ik_data.xpos[cube_id].copy()
PRE_PICK_TARGET = PICK_TARGET + HOVER_OFFSET

# Place: tray centre, cube resting just above the tray floor
tray_floor_top = ik_data.geom_xpos[tray_floor][2] + model.geom_size[tray_floor, 2]
PLACE_TARGET = np.array([ik_data.xpos[tray_id][0], ik_data.xpos[tray_id][1],
                         tray_floor_top + cube_half + PLACE_CLEARANCE])
PRE_PLACE_TARGET = PLACE_TARGET + HOVER_OFFSET

# Cube yaw from its quat; the cube is symmetric every 90 deg, so use the
# equivalent yaw in [-45, 45] deg (smallest wrist rotation)
cube_R = ik_data.xmat[cube_id].reshape(3, 3)
cube_yaw = np.arctan2(cube_R[1, 0], cube_R[0, 0])
grasp_yaw = (cube_yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4

print(f"Cube at {PICK_TARGET.round(3)}, yaw {np.degrees(cube_yaw):.1f} deg "
      f"(grasp yaw {np.degrees(grasp_yaw):.1f} deg)")
print(f"Tray place target: {PLACE_TARGET.round(3)}\n")

# Gripper orientation at HOME (pointing straight down)
ik_data.qpos[ARM_QPOS] = HOME_QPOS
mujoco.mj_kinematics(model, ik_data)
home_quat = np.zeros(4)
mujoco.mju_mat2Quat(home_quat, ik_data.site_xmat[site_id])

# Pick orientation = HOME orientation rotated by the cube's yaw about world z
yaw_quat = np.array([np.cos(grasp_yaw / 2), 0.0, 0.0, np.sin(grasp_yaw / 2)])
pick_quat = np.zeros(4)
mujoco.mju_mulQuat(pick_quat, yaw_quat, home_quat)

# Place orientation = HOME orientation, so the cube ends up square with the tray
place_quat = home_quat

# --- OUTCOME 1: time taken by the solver to plan the whole path ---
IK_TIMES = {}
ik_t0 = time.perf_counter()

pre_pick_qpos = timed_ik("Pre-Pick Hover", model, ik_data, PRE_PICK_TARGET, pick_quat,
                         site_id, HOME_QPOS, q_lo, q_hi)
pick_qpos = timed_ik("Pick", model, ik_data, PICK_TARGET, pick_quat,
                     site_id, pre_pick_qpos, q_lo, q_hi)
pre_place_qpos = timed_ik("Pre-Place Hover", model, ik_data, PRE_PLACE_TARGET, place_quat,
                          site_id, pick_qpos, q_lo, q_hi)
place_qpos = timed_ik("Place", model, ik_data, PLACE_TARGET, place_quat,
                      site_id, pre_place_qpos, q_lo, q_hi)

IK_TOTAL = time.perf_counter() - ik_t0
print(f"\nTotal path solving time: {IK_TOTAL * 1000:.2f} ms")

# Reset to home before simulating (cube stays where the XML put it)
data.qpos[ARM_QPOS] = HOME_QPOS
data.qpos[FINGER_QPOS] = FINGER_OPEN_QPOS
mujoco.mj_forward(model, data)

# ------------------------------------------------------------
# Execution sequence (all IK already solved above)
# Each step: (name, arm target qpos, gripper command, kind)
#   kind "move" -> done when the arm reaches the target
#   kind "grip" -> done when the fingers stop moving
# ------------------------------------------------------------
SEQUENCE = [
    ("Approach Pre-Pick", pre_pick_qpos,  GRIPPER_OPEN,   "move"),
    ("Descend to Pick",   pick_qpos,      GRIPPER_OPEN,   "move"),
    ("Grasp",             pick_qpos,      GRIPPER_CLOSED, "grip"),
    ("Lift",              pre_pick_qpos,  GRIPPER_CLOSED, "move"),
    ("Transport",         pre_place_qpos, GRIPPER_CLOSED, "move"),
    ("Descend to Place",  place_qpos,     GRIPPER_CLOSED, "move"),
    ("Release",           place_qpos,     GRIPPER_OPEN,   "grip"),
    ("Retract",           pre_place_qpos, GRIPPER_OPEN,   "move"),
    ("Return Home",       HOME_QPOS,      GRIPPER_OPEN,   "move"),
]

step_idx = 0
qcmd = HOME_QPOS.copy()          # commanded arm position
qd_ff = np.zeros(ARM_DOF)        # commanded arm velocity (quintic derivative)
seg_start = HOME_QPOS.copy()     # where the current quintic segment starts
seg_T = segment_duration(seg_start, SEQUENCE[0][1])
seg_s = 0.0                      # progress through the current segment (0 -> 1)
settled_steps = 0
print(f"\n[1/{len(SEQUENCE)}] {SEQUENCE[0][0]}")

# --- OUTCOME 2: overall simulation time ---
STEP_TIMES = []                  # (step name, simulated seconds)
summary_printed = False

# ------------------------------------------------------------
# Main Simulation Loop
# ------------------------------------------------------------
with mujoco.viewer.launch_passive(model, data) as viewer:
    SIM_T0 = data.time               # simulated clock at start of the sequence
    WALL_T0 = time.perf_counter()    # real clock at start of the sequence
    step_t0 = data.time

    while viewer.is_running():
        step_start = time.time()

        if step_idx < len(SEQUENCE):
            name, q_target, gripper_cmd, kind = SEQUENCE[step_idx]

            # Advance along the quintic from the segment start to the target
            seg_s = min(seg_s + model.opt.timestep / seg_T, 1.0)
            qcmd, qd_ff = quintic(seg_start, q_target, seg_s, seg_T)
            qcmd = np.clip(qcmd, q_lo, q_hi)

            # Check whether this step is complete
            if kind == "move":
                q_err = np.max(np.abs(q_target - data.qpos[ARM_QPOS]))
                q_vel = np.max(np.abs(data.qvel[ARM_DOFS]))
                done = seg_s >= 1.0 and q_err < REACH_TOL and q_vel < VEL_TOL
            else:  # "grip"
                finger_vel = np.max(np.abs(data.qvel[FINGER_DOFS])) if len(FINGER_DOFS) else 0.0
                settled_steps = settled_steps + 1 if finger_vel < GRIP_VEL_TOL else 0
                done = settled_steps >= GRIP_SETTLE_STEPS

            if done:
                STEP_TIMES.append((name, data.time - step_t0))
                step_t0 = data.time
                step_idx += 1
                settled_steps = 0
                if step_idx < len(SEQUENCE):
                    # Start the next quintic segment from the current command
                    seg_start = qcmd.copy()
                    seg_T = segment_duration(seg_start, SEQUENCE[step_idx][1])
                    seg_s = 0.0
                    print(f"[{step_idx + 1}/{len(SEQUENCE)}] {SEQUENCE[step_idx][0]}")
                else:
                    print("Sequence complete. Holding home.")
                    print_summary(completed=True)
                    summary_printed = True
        else:
            gripper_cmd = GRIPPER_OPEN   # finished: hold home
            qd_ff = np.zeros(ARM_DOF)

        # Velocity actuators take joint velocities, not angles:
        # feed-forward from the quintic + P correction toward the commanded position
        v_cmd = qd_ff + KP_VEL * (qcmd - data.qpos[ARM_QPOS])
        for i in range(ARM_DOF):
            if model.actuator_ctrllimited[i]:
                v_cmd[i] = np.clip(v_cmd[i], *model.actuator_ctrlrange[i])

        data.ctrl[:ARM_DOF] = v_cmd
        data.ctrl[GRIP_ACT] = gripper_cmd

        # Gravity compensation on the ARM joints only (never on the cube's freejoint)
        data.qfrc_applied[ARM_DOFS] = data.qfrc_bias[ARM_DOFS]

        mujoco.mj_step(model, data)
        viewer.sync()

        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

# Viewer closed before the sequence finished: still report what we have
if not summary_printed:
    print_summary(completed=False)
