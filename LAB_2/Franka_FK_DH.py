import time
import numpy as np
import mujoco
import mujoco.viewer

# ==============================================================================
# Denavit-Hartenberg (DH) Forward Kinematics for Franka Emika Panda
# ==============================================================================
# Standard DH Parameters for Franka Panda:
# Format per row: [a, alpha, d, theta_offset]
#
# Joint 1: a=0,       alpha=-pi/2, d=0.333,  theta_offset=0
# Joint 2: a=0,       alpha=pi/2,  d=0,      theta_offset=0
# Joint 3: a=0.0825,  alpha=pi/2,  d=0.316,  theta_offset=0
# Joint 4: a=-0.0825, alpha=-pi/2, d=0,      theta_offset=0
# Joint 5: a=0,       alpha=pi/2,  d=0.384,  theta_offset=0
# Joint 6: a=0.088,   alpha=pi/2,  d=0,      theta_offset=0
# Joint 7: a=0,       alpha=0,     d=0.107,  theta_offset=0
# Hand/Flange TCP offset along Z-axis: d = 0.103 (distance to gripper site)
# ==============================================================================

DH_PARAMS = np.array([
    [0.0,     -np.pi / 2, 0.333, 0.0],
    [0.0,      np.pi / 2, 0.0,   0.0],
    [0.0825,   np.pi / 2, 0.316, 0.0],
    [-0.0825, -np.pi / 2, 0.0,   0.0],
    [0.0,      np.pi / 2, 0.384, 0.0],
    [0.088,    np.pi / 2, 0.0,   0.0],
    [0.0,      0.0,       0.207, -np.pi / 4]  # Combined Joint 7 d=0.107 + hand site offset ~0.1
])


def dh_transform(a: float, alpha: float, d: float, theta: float) -> np.ndarray:
    """Computes the standard 4x4 DH transformation matrix[cite: 1]."""
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    cos_a, sin_a = np.cos(alpha), np.sin(alpha)

    return np.array([
        [cos_t, -sin_t * cos_a,  sin_t * sin_a, a * cos_t],
        [sin_t,  cos_t * cos_a, -cos_t * sin_a, a * sin_t],
        [0.0,    sin_a,          cos_a,         d],
        [0.0,    0.0,            0.0,           1.0]
    ])


def forward_kinematics_dh(q: np.ndarray) -> np.ndarray:
    """
    Computes the 4x4 Homogeneous Transformation Matrix T for the end-effector
    given joint angles q (7-DOF) using Product of DH Transformations[cite: 3].
    """
    T = np.eye(4)
    for i in range(7):
        a, alpha, d, theta_offset = DH_PARAMS[i]
        theta = q[i] + theta_offset
        T_i = dh_transform(a, alpha, d, theta)
        T = T @ T_i  # Chain multiplication T_0^i = T_0^{i-1} * T_{i-1}^i
    return T



# ==============================================================================
# Simulation Setup
# ==============================================================================
def main():
    model = mujoco.MjModel.from_xml_path("mjPanda.xml")
    data = mujoco.MjData(model)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()
            t = data.time

            # Step the simulation forward
            mujoco.mj_step(model, data)

            # Apply Phase Control
            if t < 2.0:
                # 0s - 2s: Pose 1 (Open Gripper)
                data.ctrl[:] = [0.0, -0.78, 0.0, -2.35, 0.0, 1.57, 0.78, 0.04]
            elif t < 4.0:
                # 2s - 4s: Pose 2 (Close Gripper)
                data.ctrl[:] = [0.0, -0.78, 0.0, -2.35, 0.0, 1.57, 0.78, 0.0]
            else:
                # After 4s: Lift Joint 2
                data.ctrl[:] = [0.0, -1.20, 0.0, -2.35, 0.0, 1.57, 0.78, 0.0]

            # Extract 7 joint angles from qpos (indices 0 to 6)
            q_current = data.qpos[:7]

            # Compute EE Pose via DH Forward Kinematics
            T_dh = forward_kinematics_dh(q_current)
            pos_dh = T_dh[:3, 3]

            # Output calculated DH end-effector position and orientation
            print(f"[Time {t:4.2f}s] DH EE Position (XYZ): {pos_dh[0]:.4f}, {pos_dh[1]:.4f}, {pos_dh[2]:.4f}")
            # Sync renderer display window
            viewer.sync()

            # Align simulation timing with real time
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()
