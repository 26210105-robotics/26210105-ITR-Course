"""
01_rotation_sandbox.py -- HW1 Part 2, Task 1: does rotation order matter?

STARTER CODE. The model loading, viewer, and simulation loop below
are complete and working -- run this file as-is and you should see
the asymmetric dart sitting in the viewer. Your job is to fill in
the TODOs so that:

  1. The user can queue up a sequence of elemental rotations
     (about x, y, or z), each one EITHER about the current body
     frame OR about the fixed space frame (their choice).
  2. The dart's orientation updates to reflect that sequence.
  3. You can run the SAME sequence of angles twice -- once
     "current frame" and once "fixed frame" -- and see (and
     screen-record) that the final orientation is visibly
     different, exactly as you proved symbolically in HW1
     Problem 3 (Lynch & Park Ch.3 Ex.3.4-style reasoning) and
     the "Composition of Rotations" lecture derivation.

This is intentionally a plain script, not a GUI app -- editing the
`rotation_sequence` list below and re-running is a perfectly good
"sandbox." A slider UI is a nice-to-have, not a requirement. Use AI
freely here; document what you asked it for in your AI Use Note.
"""

import time
import numpy as np
import mujoco
import mujoco.viewer

from utils import Rx, Ry, Rz, ELEMENTARY_ROTATIONS, set_body_orientation

MODEL_PATH = "/home/pammu/Documents/Mtech/ME639/hw01-mujoco-rotations/model/asymmetric_body.xml"


# ---------------------------------------------------------------
# TODO(student): edit this sequence to try different experiments.
# Each entry is (axis, angle_radians, frame) where frame is either
# "current" (compose on the RIGHT: R_new = R_old @ R_step) or
# "fixed" (compose on the LEFT: R_new = R_step @ R_old).
# Compare the two frame choices for the SAME list of (axis, angle).
# ---------------------------------------------------------------
rotation_sequence = [ #Recreating 3rd problem
    ("x", np.deg2rad(45), "fixed"),
    ("y", np.deg2rad(135), "fixed"),
    ("z", np.deg2rad(-120), "fixed"),
]


def compose_sequence(sequence):
    """TODO(student): implement this.

    Given a list of (axis, angle, frame) tuples, return the final
    3x3 rotation matrix R obtained by applying them in order,
    starting from R = identity.

    Hint: ELEMENTARY_ROTATIONS["x"](angle) gives you R_x(angle) etc.
    Hint: think about *why* current-frame composition is a right-
    multiply and fixed-frame composition is a left-multiply -- this
    is exactly the "Current Frame" vs. "Fixed Frame" distinction
    from the Composition-of-Rotations lecture. Don't just guess;
    check it against your Problem 3/4 reasoning.
    """
    R = np.eye(3)
    for axis, angle, frame in sequence:
        R_step = ELEMENTARY_ROTATIONS[axis](angle)
        # TODO: replace the next line with the correct composition
        # rule depending on `frame`.
        for axis, angle, frame in sequence:
            R_step = ELEMENTARY_ROTATIONS[axis](angle)
            if frame == "current":
                R = R @ R_step          # body-frame rotation: compose on the right
            elif frame == "fixed":
                R = R_step @ R          # space-frame rotation: compose on the left
            else:
                raise ValueError(f"frame must be 'current' or 'fixed', got {frame!r}")
        #raise NotImplementedError("Implement current- vs fixed-frame composition")
    return R


def main():
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    R_final = compose_sequence(rotation_sequence)
    set_body_orientation(data, R_final)
    mujoco.mj_forward(model, data)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("Viewer open. Animating sequence...")
        print(f"Applied sequence: {rotation_sequence}")

        R_current = np.eye(3)
        fps = 60
        animation_duration = 1.0 # 1 second per rotation step
        frames_per_step = int(animation_duration * fps)

            # Pause briefly before starting the animation
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

                    # Update the base rotation state for the next sequence item
            R_current = R_render

                    # Short pause between individual sequence steps for visual clarity
        time.sleep(0.5)

        print("Animation complete. Close the window to exit.")
                # Keep viewer open dynamically after the animation completes
        while viewer.is_running():
            viewer.sync()
            time.sleep(1.0 / 60)


if __name__ == "__main__":
    main()
