import mujoco
import numpy as np
import glfw
import time

# ---------------------------------------------------------------
# 1. Load the scene
# ---------------------------------------------------------------
xml_path = "scene.xml"
model = mujoco.MjModel.from_xml_path(xml_path)
data  = mujoco.MjData(model)

# ---------------------------------------------------------------
# 2. Resolve wheel actuators
# ---------------------------------------------------------------
def find_actuator(candidates):
    for name in candidates:
        i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if i != -1:
            return i
    return -1

left_id  = find_actuator(["left_wheel_motor", "wheel_left",  "left_wheel",  "left_wheel_joint"])
right_id = find_actuator(["right_wheel_motor","wheel_right", "right_wheel", "right_wheel_joint"])

if left_id == -1 or right_id == -1:
    print("[WARN] Wheel actuators not found by name; using actuators 0 and 1.")
    left_id, right_id = 0, 1

print(f"[INFO] Using wheel actuators: left={left_id}, right={right_id}")

# ---------------------------------------------------------------
# 3. Resolve chassis body
# ---------------------------------------------------------------
body_id = -1
for cand in ("base_link", "base_footprint", "base", "chassis",
             "turtlebot3_waffle", "buddy", "waffle"):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, cand)
    if body_id != -1:
        print(f"[INFO] Chassis body found by name: '{cand}' (id={body_id})")
        break

if body_id == -1:
    # auto-detect: movable body with most geoms
    best_id, best_count = -1, -1
    for i in range(1, model.nbody):
        c = int(np.sum(model.geom_bodyid == i))
        if c > best_count:
            best_count, best_id = c, i
    body_id = best_id
    print(f"[INFO] Chassis body auto-detected: id={body_id}")

# ---------------------------------------------------------------
# 4. Key state
# ---------------------------------------------------------------
key_state = {'w': False, 'a': False, 's': False, 'd': False}

def key_callback(window, key, scancode, action, mods):
    if action in (glfw.PRESS, glfw.RELEASE):
        p = (action == glfw.PRESS)
        if   key == glfw.KEY_W: key_state['w'] = p
        elif key == glfw.KEY_S: key_state['s'] = p
        elif key == glfw.KEY_A: key_state['a'] = p
        elif key == glfw.KEY_D: key_state['d'] = p
        elif key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)

# ---------------------------------------------------------------
# 5. GLFW window
# ---------------------------------------------------------------
if not glfw.init():
    raise Exception("GLFW init failed")
window = glfw.create_window(1200, 900, "TurtleBot - WASD Viewer", None, None)
if not window:
    glfw.terminate(); raise Exception("GLFW window failed")
glfw.make_context_current(window)
glfw.set_key_callback(window, key_callback)
glfw.swap_interval(1)

# ---------------------------------------------------------------
# 6. MuJoCo rendering
# ---------------------------------------------------------------
cam = mujoco.MjvCamera()
mujoco.mjv_defaultCamera(cam)
cam.distance, cam.elevation, cam.azimuth = 2.0, -25, 90

opt = mujoco.MjvOption()
mujoco.mjv_defaultOption(opt)
opt.frame = mujoco.mjtFrame.mjFRAME_BODY
opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False

scene   = mujoco.MjvScene(model, maxgeom=20000)
context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)

# ---------------------------------------------------------------
# 7. Controller
# ---------------------------------------------------------------
class TurtleBotController:
    MAX_WHEEL_VEL = 6.0
    LINEAR_SPEED  = 0.6
    ANGULAR_SPEED = 1.5
    def __init__(self, l, r): self.l, self.r = l, r
    def step(self, data):
        fwd  = (1.0 if key_state['w'] else 0.0) - (1.0 if key_state['s'] else 0.0)
        turn = (1.0 if key_state['d'] else 0.0) - (1.0 if key_state['a'] else 0.0)
        v, w = fwd * self.LINEAR_SPEED, turn * self.ANGULAR_SPEED
        left, right = v - w, v + w
        m = max(abs(left), abs(right), 1e-6)
        if m > 1.0:
            left /= m; right /= m
        data.ctrl[self.l] = left  * self.MAX_WHEEL_VEL
        data.ctrl[self.r] = right * self.MAX_WHEEL_VEL

ctrl = TurtleBotController(left_id, right_id)
mujoco.mj_forward(model, data)

# ---------------------------------------------------------------
# 8. Visible rotation-matrix triad on the chassis
# ---------------------------------------------------------------
def draw_triad(data, body_id, scene, scale=0.2):
    if body_id <= 0:
        return
    pos = data.xpos[body_id].copy()
    mat = data.xmat[body_id].reshape(3, 3)
    colors = [(1,0,0,1), (0,1,0,1), (0,0,1,1)]   # X=red, Y=green, Z=blue

    for axis in range(3):
        if scene.ngeom >= scene.maxgeom:
            return
        g = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            g,
            mujoco.mjtGeom.mjGEOM_ARROW,
            np.zeros(3), np.zeros(3), np.zeros(9),
            np.array(colors[axis], dtype=np.float32),
        )
        start = pos
        end   = pos + mat[:, axis] * scale
        mujoco.mjv_connector(
            g, mujoco.mjtGeom.mjGEOM_ARROW,
            0.01, start, end,
        )
        scene.ngeom += 1

# ---------------------------------------------------------------
# 9. Main loop
# ---------------------------------------------------------------
last_time = time.time()

while not glfw.window_should_close(window):
    ctrl.step(data)
    mujoco.mj_step(model, data)

    now = time.time()
    if now - last_time > (1.0 / 60.0):
        width, height = glfw.get_framebuffer_size(window)
        viewport = mujoco.MjrRect(0, 0, width, height)

        mujoco.mjv_updateScene(
            model, data, opt, None, cam,
            mujoco.mjtCatBit.mjCAT_ALL, scene
        )
        draw_triad(data, body_id, scene, scale=0.2)
        mujoco.mjr_render(viewport, scene, context)

        # --- Extract and format the Rotation Matrix ---
        if body_id > 0:
            mat = data.xmat[body_id].reshape(3, 3)
            # Format to 3 decimal places for clean UI alignment
            mat_str = (
                f"Chassis Rotation Matrix:\n"
                f"[{mat[0,0]: 5.3f}, {mat[0,1]: 5.3f}, {mat[0,2]: 5.3f}]\n"
                f"[{mat[1,0]: 5.3f}, {mat[1,1]: 5.3f}, {mat[1,2]: 5.3f}]\n"
                f"[{mat[2,0]: 5.3f}, {mat[2,1]: 5.3f}, {mat[2,2]: 5.3f}]"
            )
        else:
            mat_str = "Rotation Matrix: N/A"

        # --- Append Matrix to the UI Overlay ---
        overlay = (
            f"W={'^' if key_state['w'] else '-'} "
            f"S={'v' if key_state['s'] else '-'} "
            f"A={'<' if key_state['a'] else '-'} "
            f"D={'>' if key_state['d'] else '-'}  "
            f"body_id={body_id}\n\n"
            f"{mat_str}"
        )

        mujoco.mjr_overlay(
            mujoco.mjtFont.mjFONT_NORMAL,
            mujoco.mjtGridPos.mjGRID_TOPLEFT,
            viewport, "TurtleBot", overlay, context,
        )
        glfw.swap_buffers(window)
        glfw.poll_events()
        last_time = now

glfw.terminate()
