import mujoco  # noqa: I001
import mujoco.viewer
import numpy as np
import os
import time

# 1. Load your XML model and create the data structure
model = mujoco.MjModel.from_xml_path("mjPanda.xml")# type: ignore
gripper_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripper")# type: ignore
#print([model.geom(i).name for i in range(model.ngeom)])
data = mujoco.MjData(model)# type: ignore
quat = np.zeros(4)
#print('Total number of DoFs in the model:', model.nv)

# 2. Launch the passive viewer window automatically when the script runs
with mujoco.viewer.launch_passive(model, data) as viewer:
    # 3. Physics loop
    while viewer.is_running():
        step_start = time.time()
        t = data.time
        # Step the physics forward by one time step
        mujoco.mj_step(model, data)# type: ignore
        if t < 2.0:
                    # 0s - 2s: Pose 1 (Open Gripper)
                data.ctrl[:] = [0.0, -0.78, 0.0, -2.35, 0.0, 1.57, 0.78, 0.04]
                mujoco.mju_mat2Quat(quat, data.site_xmat[gripper_id])
                pos = data.site_xpos[gripper_id]
                print("Position (XYZ):", pos)
                print("Orientation (Quaternion w,x,y,z):", quat)
        elif t < 4.0:
                    # 2s - 4s: Pose 2 (Close Gripper)
            data.ctrl[:] = [0.0, -0.78, 0.0, -2.35, 0.0, 1.57, 0.78, 0.0]
            mujoco.mju_mat2Quat(quat, data.site_xmat[gripper_id])
            pos = data.site_xpos[gripper_id]
            print("Position (XYZ):", pos)
            print("Orientation (Quaternion w,x,y,z):", quat)
        else:
                    # After 4s: Lift Joint 2
            data.ctrl[:] = [0.0, -1.20, 0.0, -2.35, 0.0, 1.57, 0.78, 0.0]
            mujoco.mju_mat2Quat(quat, data.site_xmat[gripper_id])
            pos = data.site_xpos[gripper_id]
            print("Position (XYZ):", pos)
            print("Orientation (Quaternion w,x,y,z):", quat)
        # Update the 3D display window with the new physics state
        viewer.sync()

        # Keep simulation timing aligned with real time
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
