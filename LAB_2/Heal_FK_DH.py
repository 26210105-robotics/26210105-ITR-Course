import time
import numpy as np
import mujoco
import mujoco.viewer

# ==============================================================================
# Helper Functions for 3D Geometry Transformation
# ==============================================================================

def euler_to_matrix(euler: np.ndarray) -> np.ndarray:
    """Computes 3x3 rotation matrix from Euler angles [roll, pitch, yaw] in radians (extrinsic XYZ)."""
    rx, ry, rz = euler

    # Rotation about X
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(rx), -np.sin(rx)],
        [0, np.sin(rx), np.cos(rx)]
    ])
    # Rotation about Y
    Ry = np.array([
        [np.cos(ry), 0, np.sin(ry)],
        [0, 1, 0],
        [-np.sin(ry), 0, np.sin(ry)]
    ])
    # Rotation about Z
    Rz = np.array([
        [np.cos(rz), -np.sin(rz), 0],
        [np.sin(rz), np.cos(rz), 0],
        [0, 0, 1]
    ])

    return Rz @ Ry @ Rx

def quat_to_matrix(quat: np.ndarray) -> np.ndarray:
    """Converts MuJoCo quaternion [w, x, y, z] to a 3x3 rotation matrix."""
    w, x, y, z = quat
    return np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x**2 + y**2)]
    ])

def make_transform(pos: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Builds a 4x4 Homogeneous Transformation Matrix from position vector and 3x3 rotation matrix."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = pos
    return T

# ==============================================================================
# Forward Kinematics for Addverb Heal Robot (6-DOF)
# ==============================================================================

def forward_kinematics_heal(q: np.ndarray) -> np.ndarray:
    """
    Computes the 4x4 end-effector transform relative to world frame for
    the `addverb_heal` 6-DOF arm given joint angles q = [q1, q2, q3, q4, q5, q6].
    """
    # 1. Base to Link 1 (joint_1 along +Z)
    T_base_l1 = make_transform(np.array([0, 0, 0.171]), np.eye(3))
    R_j1 = euler_to_matrix([0, 0, q[0]])  # Joint axis [0, 0, 1]
    T_j1 = make_transform(np.zeros(3), R_j1)

    # 2. Link 1 to Link 2 (joint_2 along -Z in local frame)
    R_l2_offset = quat_to_matrix(np.array([0.707105, 0.707108, 0, 0]))
    T_l1_l2 = make_transform(np.array([0, 0.0875, 0.1498]), R_l2_offset)
    R_j2 = euler_to_matrix([0, 0, -q[1]])  # Joint axis [0, 0, -1]
    T_j2 = make_transform(np.zeros(3), R_j2)

    # 3. Link 2 to Link 3 (joint_3 along +Z)
    R_l3_offset = euler_to_matrix([0, 0, -1.57])
    T_l2_l3 = make_transform(np.array([0, 0.3, 0]), R_l3_offset)
    R_j3 = euler_to_matrix([0, 0, q[2]])  # Joint axis [0, 0, 1]
    T_j3 = make_transform(np.zeros(3), R_j3)

    # 4. Link 3 to Link 4 (joint_4 along +Z)
    R_l4_offset = euler_to_matrix([-1.57, 0, 0])
    T_l3_l4 = make_transform(np.array([0, 0.1593, 0.0875]), R_l4_offset)
    R_j4 = euler_to_matrix([0, 0, q[3]])  # Joint axis [0, 0, 1]
    T_j4 = make_transform(np.zeros(3), R_j4)

    # 5. Link 4 to Link 5 (joint_5 along +Z)
    R_l5_offset = euler_to_matrix([0.50951, 0, 1.57])
    T_l4_l5 = make_transform(np.array([0, 0.03185, 0.16105]), R_l5_offset)
    R_j5 = euler_to_matrix([0, 0, q[4]])  # Joint axis [0, 0, 1]
    T_j5 = make_transform(np.zeros(3), R_j5)

    # 6. Link 5 to End Effector (joint_6 along -Z in local frame)
    R_ee_offset = quat_to_matrix(np.array([0.707105, 0.707108, 0, 0]))
    T_l5_ee = make_transform(np.array([0, -0.1227, 0.0654]), R_ee_offset)
    R_j6 = euler_to_matrix([0, 0, -q[5]])  # Joint axis [0, 0, -1]
    T_j6 = make_transform(np.zeros(3), R_j6)

    # Forward Kinematics Chain
    T = T_base_l1 @ T_j1 @ T_l1_l2 @ T_j2 @ T_l2_l3 @ T_j3 @ T_l3_l4 @ T_j4 @ T_l4_l5 @ T_j5 @ T_l5_ee @ T_j6
    return T


# ==============================================================================
# Simulation Setup
# ==============================================================================
def main():
    model = mujoco.MjModel.from_xml_path("mj_heal_torque.xml")
    data = mujoco.MjData(model)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()
            t = data.time

            # Step the simulation forward
            mujoco.mj_step(model, data)

            # Apply control inputs for 6 motors
            if t < 2.0:
                data.ctrl[:] = data.qfrc_bias[:model.nu]+[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            elif t < 4.0:
                data.ctrl[:] = data.qfrc_bias[:model.nu]+[1.5, -1.5, 1.5, 0.0, 0.0, 0.0]
            else:
                data.ctrl[:] = data.qfrc_bias[:model.nu]+[0.5, -1.0, 1.0, -0.5, 0.5, 0.0]

            # Extract 6 joint positions
            q_current = data.qpos[:6]

            # Compute EE Pose via Forward Kinematics
            T_fk = forward_kinematics_heal(q_current)
            pos_fk = T_fk[:3, 3]

            # Get MuJoCo ground truth site/body position for verification
            ee_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "end_effector")
            pos_mujoco = data.xpos[ee_body_id]

            print(f"[Time {t:4.2f}s] FK EE Pos: ({pos_fk[0]:.4f}, {pos_fk[1]:.4f}, {pos_fk[2]:.4f}) | MuJoCo EE Pos: ({pos_mujoco[0]:.4f}, {pos_mujoco[1]:.4f}, {pos_mujoco[2]:.4f})")

            viewer.sync()

            # Maintain real-time execution timing
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()
