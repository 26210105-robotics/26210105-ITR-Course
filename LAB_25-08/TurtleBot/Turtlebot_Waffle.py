import mujoco
import numpy as np
import glfw
import time

# --- Setup MuJoCo ---
xml_path = "scene.xml"
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# Actuator IDs (Verify these match the motor names in your XML)
actuator_names = ["thrust1", "thrust2", "thrust3", "thrust4"]
motor_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in actuator_names]

# Body whose frame we want to visualize (Verify "x2" is your drone's chassis body name)
body_name = "x2"
body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
if body_id == -1:
    body_id = 1 # Fallback if "x2" is not found

# State dictionary for "Hold-to-Move" keys and Spacebar trigger
key_state = {'w': False, 'a': False, 's': False, 'd': False}
space_event = False

def key_callback(window, key, scancode, action, mods):
    """Detects WASD holds, Spacebar presses for takeoff, and ESC to quit."""
    global space_event
    if action == glfw.PRESS or action == glfw.RELEASE:
        is_pressed = (action == glfw.PRESS)
        if key == glfw.KEY_W: key_state['w'] = is_pressed
        elif key == glfw.KEY_S: key_state['s'] = is_pressed
        elif key == glfw.KEY_A: key_state['a'] = is_pressed
        elif key == glfw.KEY_D: key_state['d'] = is_pressed
        elif key == glfw.KEY_SPACE:
            if is_pressed:
                space_event = True
        elif key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)

# --- Initialize GLFW ---
if not glfw.init():
    raise Exception("GLFW initialization failed")

window = glfw.create_window(1200, 900, "X2 Drone - Custom WASD Viewer", None, None)
if not window:
    glfw.terminate()
    raise Exception("GLFW window creation failed")

glfw.make_context_current(window)
glfw.set_key_callback(window, key_callback)
glfw.swap_interval(1)

# --- Initialize MuJoCo Rendering Components ---
cam = mujoco.MjvCamera()
mujoco.mjv_defaultCamera(cam)
cam.distance = 2.0
cam.elevation = -20 # Angle slightly down to see the drone better

opt = mujoco.MjvOption()
mujoco.mjv_defaultOption(opt)
opt.frame = mujoco.mjtFrame.mjFRAME_NONE

scene = mujoco.MjvScene(model, maxgeom=10000)
context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)

AXIS_COLORS = [
    np.array([1, 0, 0, 1], dtype=np.float32),
    np.array([0, 1, 0, 1], dtype=np.float32),
    np.array([0, 0, 1, 1], dtype=np.float32),
]

def draw_frame(scene, origin, rotmat, length=0.2, width=0.008):
    for axis in range(3):
        if scene.ngeom >= scene.maxgeom:
            break
        direction = rotmat[:, axis]
        end = origin + direction * length
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            geom, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3), np.eye(3).flatten(), AXIS_COLORS[axis]
        )
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_ARROW, width, origin, end)
        scene.ngeom += 1

# --- Flight Controller ---
class X2Controller:
    HOVER_THRUST = 3.2495625
    CTRL_MAX = 13.0
    MOTOR_POS = np.array([
        [-0.14, -0.18, 0.05],
        [-0.14,  0.18, 0.05],
        [ 0.14,  0.18, 0.08],
        [ 0.14, -0.18, 0.08],
    ])
    YAW_SIGN = np.array([-1.0, 1.0, -1.0, 1.0])

    TARGET_ALT = 0.3
    GROUND_ALT = 0.1
    TAKEOFF_TIME = 2.5
    TILT_CMD = 0.14

    KP_ALT, KD_ALT = 40.0, 14.0
    KP_ATT, KD_ATT = 6.0, 1.1
    KD_YAW = 0.6

    STABLE_ALT_TOL = 0.03
    STABLE_VEL_TOL = 0.08
    STABLE_TILT_TOL = 0.06

    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.roll_sign = -np.sign(self.MOTOR_POS[:, 1])
        self.pitch_sign = -np.sign(self.MOTOR_POS[:, 0])
        self.phase = "ground"
        self.takeoff_t0 = None
        self.movement_enabled = False

    @staticmethod
    def _quat_to_roll_pitch(quat_wxyz):
        w, x, y, z = quat_wxyz
        roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        sinp = np.clip(2 * (w * y - z * x), -1.0, 1.0)
        pitch = np.arcsin(sinp)
        return roll, pitch

    def begin_takeoff(self):
        if self.phase == "ground":
            self.phase = "takeoff"
            self.takeoff_t0 = time.time()

    def step(self):
        d = self.data

        if self.phase == "ground":
            for act_id in motor_ids:
                d.ctrl[act_id] = 0.0
            return

        z = d.qpos[2]
        vz = d.qvel[2]
        quat = d.sensor("body_quat").data
        gyro = d.sensor("body_gyro").data
        roll, pitch = self._quat_to_roll_pitch(quat)

        if self.phase == "takeoff":
            elapsed = time.time() - self.takeoff_t0
            frac = min(1.0, elapsed / self.TAKEOFF_TIME)
            target_alt = self.GROUND_ALT + frac * (self.TARGET_ALT - self.GROUND_ALT)

            stable = (
                frac >= 1.0
                and abs(self.TARGET_ALT - z) < self.STABLE_ALT_TOL
                and abs(vz) < self.STABLE_VEL_TOL
                and abs(roll) < self.STABLE_TILT_TOL
                and abs(pitch) < self.STABLE_TILT_TOL
            )
            if stable:
                self.phase = "flying"
                self.movement_enabled = True
        else:
            target_alt = self.TARGET_ALT

        target_roll = 0.0
        target_pitch = 0.0

        # Read WASD states directly from our global key_state dictionary
        if self.movement_enabled:
            if key_state['d']: target_roll += self.TILT_CMD
            if key_state['a']: target_roll -= self.TILT_CMD
            if key_state['w']: target_pitch -= self.TILT_CMD
            if key_state['s']: target_pitch += self.TILT_CMD

        alt_err = target_alt - z
        alt_cmd = self.KP_ALT * alt_err - self.KD_ALT * vz

        roll_err = target_roll - roll
        pitch_err = target_pitch - pitch
        roll_cmd = self.KP_ATT * roll_err - self.KD_ATT * gyro[0]
        pitch_cmd = self.KP_ATT * pitch_err - self.KD_ATT * gyro[1]
        yaw_cmd = -self.KD_YAW * gyro[2]

        base = self.HOVER_THRUST + alt_cmd / 4.0
        motor_cmd = base + roll_cmd * self.roll_sign + pitch_cmd * self.pitch_sign + yaw_cmd * self.YAW_SIGN
        motor_cmd = np.clip(motor_cmd, 0.0, self.CTRL_MAX)

        for i, act_id in enumerate(motor_ids):
            d.ctrl[act_id] = motor_cmd[i]

ctrl = X2Controller(model, data)
data.qpos[2] = 0.1
mujoco.mj_forward(model, data)

# --- Main Single-Threaded Loop ---
last_time = time.time()

while not glfw.window_should_close(window):
    # Process spacebar takeoff trigger
    if space_event:
        ctrl.begin_takeoff()
        space_event = False

    # 1. Calculate and Apply Flight Control
    ctrl.step()

    # 2. Step the Physics Simulation
    mujoco.mj_step(model, data)

    # 3. Render Graphics at ~60Hz
    current_time = time.time()
    if current_time - last_time > (1.0 / 60.0):
        width, height = glfw.get_framebuffer_size(window)
        viewport = mujoco.MjrRect(0, 0, width, height)

        mujoco.mjv_updateScene(model, data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scene)

        # Draw world and body frames
        draw_frame(scene, np.zeros(3), np.eye(3))
        if body_id != -1:
            body_pos = data.xpos[body_id].copy()
            body_mat = data.xmat[body_id].reshape(3, 3).copy()
            draw_frame(scene, body_pos, body_mat)

        mujoco.mjr_render(viewport, scene, context)

        # On-screen overlay showing flight phase
        overlay_text = f"Phase: {ctrl.phase.upper()}"
        if ctrl.phase == "flying":
            overlay_text += " (WASD active)"

        mujoco.mjr_overlay(
            mujoco.mjtFont.mjFONT_NORMAL,
            mujoco.mjtGridPos.mjGRID_TOPLEFT,
            viewport,
            "Drone Status",
            overlay_text,
            context,
        )

        glfw.swap_buffers(window)
        glfw.poll_events()
        last_time = current_time

glfw.terminate()
