import time
import numpy as np
import mujoco
import mujoco.viewer
from pathlib import Path

from utils import Rx, Ry, Rz, ELEMENTARY_ROTATIONS, set_body_orientation

MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "asymmetric_body.xml"


rotation_sequence = [
    ("x", np.deg2rad(45), "fixed"),
    ("y", np.deg2rad(135), "fixed"),
    ("z", np.deg2rad(-120), "fixed"),
]


def compose_sequence(sequence):
    R = np.eye(3)
    for axis, angle, frame in sequence:
        R_step = ELEMENTARY_ROTATIONS[axis](angle)
        for axis, angle, frame in sequence:
            R_step = ELEMENTARY_ROTATIONS[axis](angle)
            if frame == "current":
                R = R @ R_step
            elif frame == "fixed":
                R = R_step @ R
            else:
                raise ValueError(f"frame must be 'current' or 'fixed', got {frame!r}")

    return R


def main():
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    R_final = compose_sequence(rotation_sequence)
    set_body_orientation(data, R_final)
    mujoco.mj_forward(model, data)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("Viewer open. Animating sequence...")
        print(f"Applied sequence: {rotation_sequence}")

        R_current = np.eye(3)
        fps = 60
        animation_duration = 1.0
        frames_per_step = int(animation_duration * fps)


        time.sleep(1.0)

        for axis, final_angle, frame in rotation_sequence:
            for step in range(frames_per_step + 1):
                if not viewer.is_running():
                    return

                        # Interpolate the angle from 0 to final_angle
                current_angle = final_angle * (step / frames_per_step)
                R_step = ELEMENTARY_ROTATIONS[axis](current_angle)

                if frame == "current":
                    R_render = R_current @ R_step
                elif frame == "fixed":
                    R_render = R_step @ R_current

                set_body_orientation(data, R_render)
                mujoco.mj_forward(model, data)
                viewer.sync()
                time.sleep(1.0 / fps)

            R_current = R_render

        time.sleep(0.5)

        print("Animation complete. Close the window to exit.")
                # Keep viewer open dynamically after the animation completes
        while viewer.is_running():
            viewer.sync()
            time.sleep(1.0 / 60)


if __name__ == "__main__":
    main()
