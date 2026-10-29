import time
import mujoco
import mujoco.viewer
import numpy as np

MODEL_PATH = "Pos1.xml"
ARM_DOF = 7

# Designated Home Position
HOME_QPOS = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])

# --- Cartesian Waypoints ---
HOVER_OFFSET = np.array([0.0, 0.0, 0.25])  # 25 cm directly above the target

PICK_TARGET = np.array([0.4, 0.4, 0.02])
PRE_PICK_TARGET = PICK_TARGET + HOVER_OFFSET

PLACE_TARGET = np.array([0.4, -0.3, 0.30])
PRE_PLACE_TARGET = PLACE_TARGET + HOVER_OFFSET

# Gripper States (velocity actuator: + opens, - closes)
GRIPPER_OPEN = 0.1
GRIPPER_CLOSED = -0.1
FINGER_OPEN_QPOS = 0.04   # initial finger joint position (fully open)

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
        ik_data.qpos[:ARM_DOF] = q
        mujoco.mj_kinematics(model, ik_data)
        mujoco.mj_comPos(model, ik_data)

        err_p = target_pos - ik_data.site_xpos[site_id]
        err_r = orientation_error(target_quat, ik_data.site_xmat[site_id])

        if np.linalg.norm(err_p) < pos_tol and np.linalg.norm(err_r) < rot_tol:
            break

        mujoco.mj_jacSite(model, ik_data, jacp, jacr, site_id)
        J = np.vstack([jacp[:, :ARM_DOF], jacr[:, :ARM_DOF]])   # 6x7
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


# ------------------------------------------------------------
# Initialization & Offline Planning
# ------------------------------------------------------------
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)
ik_data = mujoco.MjData(model)   # separate data so IK doesn't disturb the sim

site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripper")

# Joint limits (unlimited joints are not clipped)
q_lo = np.where(model.jnt_limited[:ARM_DOF], model.jnt_range[:ARM_DOF, 0], -np.inf)
q_hi = np.where(model.jnt_limited[:ARM_DOF], model.jnt_range[:ARM_DOF, 1], np.inf)

# Target orientation = gripper orientation at HOME (pointing down)
ik_data.qpos[:ARM_DOF] = HOME_QPOS
mujoco.mj_kinematics(model, ik_data)
target_quat = np.zeros(4)
mujoco.mju_mat2Quat(target_quat, ik_data.site_xmat[site_id])

print("Solving DLS-IK for Pre-Pick Hover...")
pre_pick_qpos = solve_ik_dls(model, ik_data, PRE_PICK_TARGET, target_quat, site_id, HOME_QPOS, q_lo, q_hi)

print("Solving DLS-IK for Pick...")
pick_qpos = solve_ik_dls(model, ik_data, PICK_TARGET, target_quat, site_id, pre_pick_qpos, q_lo, q_hi)

print("Solving DLS-IK for Pre-Place Hover...")
pre_place_qpos = solve_ik_dls(model, ik_data, PRE_PLACE_TARGET, target_quat, site_id, pick_qpos, q_lo, q_hi)

print("Solving DLS-IK for Place...")
place_qpos = solve_ik_dls(model, ik_data, PLACE_TARGET, target_quat, site_id, pre_place_qpos, q_lo, q_hi)

# Reset to home before simulating
data.qpos[:ARM_DOF] = HOME_QPOS
data.qpos[ARM_DOF] = FINGER_OPEN_QPOS
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

# ------------------------------------------------------------
# Main Simulation Loop
# ------------------------------------------------------------
with mujoco.viewer.launch_passive(model, data) as viewer:
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
                q_err = np.max(np.abs(q_target - data.qpos[:ARM_DOF]))
                q_vel = np.max(np.abs(data.qvel[:ARM_DOF]))
                done = seg_s >= 1.0 and q_err < REACH_TOL and q_vel < VEL_TOL
            else:  # "grip"
                finger_vel = abs(data.qvel[ARM_DOF])
                settled_steps = settled_steps + 1 if finger_vel < GRIP_VEL_TOL else 0
                done = settled_steps >= GRIP_SETTLE_STEPS

            if done:
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
        else:
            gripper_cmd = GRIPPER_OPEN   # finished: hold home
            qd_ff = np.zeros(ARM_DOF)

        # Velocity actuators take joint velocities, not angles:
        # feed-forward from the quintic + P correction toward the commanded position
        v_cmd = qd_ff + KP_VEL * (qcmd - data.qpos[:ARM_DOF])
        for i in range(ARM_DOF):
            if model.actuator_ctrllimited[i]:
                v_cmd[i] = np.clip(v_cmd[i], *model.actuator_ctrlrange[i])

        data.ctrl[:ARM_DOF] = v_cmd
        data.ctrl[ARM_DOF] = gripper_cmd

        mujoco.mj_step(model, data)
        viewer.sync()

        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
