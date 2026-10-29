import itertools

import mujoco
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

# ---------------------------------------------------------------
# 1. Load model and data
# ---------------------------------------------------------------
model = mujoco.MjModel.from_xml_path("mj_heal_torque.xml")
data = mujoco.MjData(model)

site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripper")

ARM_JOINTS = 7
q_min = model.jnt_range[:ARM_JOINTS, 0]
q_max = model.jnt_range[:ARM_JOINTS, 1]
qpos_idx = model.jnt_qposadr[:ARM_JOINTS]  # correct qpos slots even if other joints exist

# ---------------------------------------------------------------
# 2. Sample random joint configurations
# ---------------------------------------------------------------
NUM_SAMPLES = 50000
workspace_points = np.empty((NUM_SAMPLES, 3))

print(f"Sampling {NUM_SAMPLES} joint configurations...")
for k in range(NUM_SAMPLES):
    data.qpos[qpos_idx] = np.random.uniform(q_min, q_max)
    mujoco.mj_kinematics(model, data)  # FK only, faster than mj_forward
    workspace_points[k] = data.site_xpos[site_id]

# ---------------------------------------------------------------
# 3. Bounding box
# ---------------------------------------------------------------
mins = workspace_points.min(axis=0)
maxs = workspace_points.max(axis=0)

print("\n--- Robot Workspace Bounding Box (Meters) ---")
for axis, lo, hi in zip("XYZ", mins, maxs):
    print(f"{axis} range: [{lo:.3f}, {hi:.3f}]  (size {hi - lo:.3f})")

# ---------------------------------------------------------------
# 4. 3D plot: bounding box wireframe + convex hull
# ---------------------------------------------------------------
fig = plt.figure(figsize=(9, 8))
ax = fig.add_subplot(111, projection="3d")

# Bounding box: connect corners that differ in exactly one coordinate
corners = np.array(list(itertools.product(*zip(mins, maxs))))
for a, b in itertools.combinations(corners, 2):
    if np.sum(a != b) == 1:
        ax.plot(*zip(a, b), color="red", linewidth=1.5)

# Convex hull: the actual outer envelope of reachable points
hull = ConvexHull(workspace_points)
ax.plot_trisurf(*workspace_points.T, triangles=hull.simplices,
                alpha=0.15, color="tab:blue", edgecolor="none")

# Light point sample for context
ax.scatter(*workspace_points[::50].T, s=1, alpha=0.3, color="gray")

ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")
ax.set_title("Reachable Workspace (Heal Arm)")
ax.set_box_aspect(maxs - mins)  # true-to-scale proportions

# ---------------------------------------------------------------
# 5. 2D projections with bounding rectangles
# ---------------------------------------------------------------
fig2, axes = plt.subplots(1, 3, figsize=(15, 4.5))
views = [((0, 1), "XY (top)"), ((0, 2), "XZ (side)"), ((1, 2), "YZ (front)")]
labels = "XYZ"

for ax2, ((i, j), title) in zip(axes, views):
    ax2.scatter(workspace_points[::10, i], workspace_points[::10, j],
                s=1, alpha=0.2)
    ax2.add_patch(plt.Rectangle((mins[i], mins[j]),
                                maxs[i] - mins[i], maxs[j] - mins[j],
                                fill=False, color="red", linewidth=2))
    ax2.set_xlabel(f"{labels[i]} (m)")
    ax2.set_ylabel(f"{labels[j]} (m)")
    ax2.set_title(title)
    ax2.set_aspect("equal")

plt.tight_layout()
plt.show()
