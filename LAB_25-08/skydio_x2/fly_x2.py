import mujoco
import numpy as np
import glfw
import time

# --- Setup MuJoCo ---
xml_path = "scene.xml"
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

actuator_names = ["thrust1", "thrust2", "thrust3", "thrust4"]
motor_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in actuator_names]

body_name = "x2"
body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
if body_id == -1: body_id = 1

# --- STEP 1: HELD-KEY STATE TRACKER ---
# Fix #1: keys are now HELD state, not one-shot events, so input becomes a
# continuous velocity command integrated into the waypoint instead of an
# instantaneous 0.5 m jump. Space (takeoff) and Escape remain one-shot.
key_held = {'w': False, 'a': False, 's': False, 'd': False, 'q': False, 'e': False}
takeoff_requested = False

def key_callback(window, key, scancode, action, mods):
    global takeoff_requested
    if action == glfw.PRESS:
        if key == glfw.KEY_W: key_held['w'] = True
        elif key == glfw.KEY_S: key_held['s'] = True
        elif key == glfw.KEY_A: key_held['a'] = True
        elif key == glfw.KEY_D: key_held['d'] = True
        elif key == glfw.KEY_Q: key_held['q'] = True
        elif key == glfw.KEY_E: key_held['e'] = True
        elif key == glfw.KEY_SPACE: takeoff_requested = True
        elif key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)
    elif action == glfw.RELEASE:
        if key == glfw.KEY_W: key_held['w'] = False
        elif key == glfw.KEY_S: key_held['s'] = False
        elif key == glfw.KEY_A: key_held['a'] = False
        elif key == glfw.KEY_D: key_held['d'] = False
        elif key == glfw.KEY_Q: key_held['q'] = False
        elif key == glfw.KEY_E: key_held['e'] = False

# --- Initialize GLFW ---
if not glfw.init(): raise Exception("GLFW initialization failed")
window = glfw.create_window(1200, 900, "X2 Drone - Autonomous DS Planner", None, None)
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
cam.elevation = -20

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
        if scene.ngeom >= scene.maxgeom: break
        direction = rotmat[:, axis]
        end = origin + direction * length
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            geom, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3), np.eye(3).flatten(), AXIS_COLORS[axis]
        )
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_ARROW, width, origin, end)
        scene.ngeom += 1

# --- STEP 2: AUTONOMOUS CASCADED CONTROLLER ---
class X2Controller:
    HOVER_THRUST = 3.27
    CTRL_MAX = 10.0
    MOTOR_POS = np.array([
        [-0.14, -0.18, 0.05],
        [-0.14,  0.18, 0.05],
        [ 0.14,  0.18, 0.08],
        [ 0.14, -0.18, 0.08],
    ])
    YAW_SIGN = np.array([-1.0, 1.0, -1.0, 1.0])

    TARGET_ALT = 0.5
    GROUND_ALT = 0.1
    TAKEOFF_TIME = 4

    # Fix #1: velocity of the waypoint under held-key input, and yaw rate
    # under held q/e, replacing the old one-shot 0.5 m / 0.26 rad jumps.
    POS_CMD_SPEED = 5     # m/s the target waypoint moves while a key is held
    YAW_CMD_RATE = 0.6      # rad/s the target yaw rotates while q/e is held

    KP_POS = 0.8
    KP_VEL = 0.3
    MAX_TILT = 0.15

    # Fix #4: widen the gap between outer (position/altitude) and inner
    # (attitude) loop stiffness so attitude settles well ahead of the loops
    # that depend on it, instead of all three fighting on similar timescales.
    KP_ALT, KD_ALT = 8.0, 3.0
    KI_ALT = 1.5              # Fix #3: integral term removes steady-state hover offset
    ALT_I_CLAMP = 1.0          # Fix #5: anti-windup clamp on the altitude integrator

    KP_ATT, KD_ATT = 7.0, 1.6
    KP_YAW, KD_YAW = 1.0, 0.3

    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.roll_sign = -np.sign(self.MOTOR_POS[:, 1])
        self.pitch_sign = -np.sign(self.MOTOR_POS[:, 0])

        self.phase = "ground"
        self.takeoff_t0 = None

        self.target_pos = np.array([0.0, 0.0, self.GROUND_ALT])
        self.target_yaw = 0.0

        self.alt_int = 0.0
        self.last_t = time.time()

    @staticmethod
    def _quat_to_euler(quat_wxyz):
        w, x, y, z = quat_wxyz
        roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        sinp = np.clip(2 * (w * y - z * x), -1.0, 1.0)
        pitch = np.arcsin(sinp)
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        return roll, pitch, yaw

    def request_takeoff(self):
        if self.phase == "ground":
            self.phase = "takeoff"
            self.takeoff_t0 = time.time()
            self.target_pos[0:2] = self.data.qpos[0:2].copy()
            self.target_pos[2] = self.TARGET_ALT
            _, _, self.target_yaw = self._quat_to_euler(self.data.sensor("body_quat").data)

    def process_high_level_commands(self, held, dt):
        """Maps held-key intents to a smoothly ramped waypoint/yaw, using the
        drone's MEASURED heading (Fix #2) rather than the setpoint yaw, so the
        commanded direction always matches what the attitude controller
        actually treats as 'forward' -- even mid-turn."""
        if self.phase != "flying":
            return

        _, _, yaw_meas = self._quat_to_euler(self.data.sensor("body_quat").data)

        step = self.POS_CMD_SPEED * dt
        dx = np.cos(yaw_meas) * step
        dy = np.sin(yaw_meas) * step

        if held['w']:
            self.target_pos[0] += dx
            self.target_pos[1] += dy
        if held['s']:
            self.target_pos[0] -= dx
            self.target_pos[1] -= dy
        if held['a']:
            self.target_pos[0] -= dy
            self.target_pos[1] += dx
        if held['d']:
            self.target_pos[0] += dy
            self.target_pos[1] -= dx

        if held['q']: self.target_yaw += self.YAW_CMD_RATE * dt
        if held['e']: self.target_yaw -= self.YAW_CMD_RATE * dt

    def step(self, dt):
        d = self.data
        if self.phase == "ground":
            for act_id in motor_ids: d.ctrl[act_id] = 0.0
            return

        pos = d.qpos[0:3]
        vel = d.qvel[0:3]
        quat = d.sensor("body_quat").data
        gyro = d.sensor("body_gyro").data
        roll, pitch, yaw = self._quat_to_euler(quat)

        # Altitude
        if self.phase == "takeoff":
            elapsed = time.time() - self.takeoff_t0
            frac = min(1.0, elapsed / self.TAKEOFF_TIME)
            current_target_z = self.GROUND_ALT + frac * (self.TARGET_ALT - self.GROUND_ALT)
            if frac >= 1.0 and abs(self.TARGET_ALT - pos[2]) < 0.05 and abs(vel[2]) < 0.05:
                self.phase = "flying"
        else:
            current_target_z = self.target_pos[2]

        alt_err = current_target_z - pos[2]

        # Fix #3 + #5: clamped integral term on altitude to remove steady-state
        # offset from HOVER_THRUST not perfectly matching true hover thrust,
        # with anti-windup so it can't accumulate while other loops saturate.
        self.alt_int = np.clip(self.alt_int + alt_err * dt, -self.ALT_I_CLAMP, self.ALT_I_CLAMP)

        alt_cmd = self.KP_ALT * alt_err + self.KI_ALT * self.alt_int - self.KD_ALT * vel[2]
        base_thrust = self.HOVER_THRUST + alt_cmd / 4.0

        # DS Navigation (Global to Local Frame mapping)
        target_roll, target_pitch = 0.0, 0.0
        if self.phase == "flying":
            pos_err = self.target_pos[0:2] - pos[0:2]
            MAX_CMD_VEL = 1.5  # m/s -- defense in depth against any large pos_err spike
            target_vel_global = np.clip(self.KP_POS * pos_err, -MAX_CMD_VEL, MAX_CMD_VEL)
            vel_err_global = target_vel_global - vel[0:2]

            c, s = np.cos(yaw), np.sin(yaw)
            local_vel_err_x = c * vel_err_global[0] + s * vel_err_global[1]
            local_vel_err_y = -s * vel_err_global[0] + c * vel_err_global[1]

            target_pitch = np.clip(-self.KP_VEL * local_vel_err_x, -self.MAX_TILT, self.MAX_TILT)
            target_roll  = np.clip( self.KP_VEL * local_vel_err_y, -self.MAX_TILT, self.MAX_TILT)

        # Attitude & Yaw Control
        yaw_err = self.target_yaw - yaw
        yaw_err = (yaw_err + np.pi) % (2 * np.pi) - np.pi

        roll_cmd = self.KP_ATT * (target_roll - roll) - self.KD_ATT * gyro[0]
        pitch_cmd = self.KP_ATT * (target_pitch - pitch) - self.KD_ATT * gyro[1]
        yaw_cmd = self.KP_YAW * yaw_err - self.KD_YAW * gyro[2]

        motor_cmd = base_thrust + roll_cmd * self.roll_sign + pitch_cmd * self.pitch_sign + yaw_cmd * self.YAW_SIGN
        motor_cmd = np.clip(motor_cmd, 0.0, self.CTRL_MAX)
        for i, act_id in enumerate(motor_ids): d.ctrl[act_id] = motor_cmd[i]

ctrl = X2Controller(model, data)
data.qpos[2] = 0.1
mujoco.mj_forward(model, data)
last_time = time.time()
last_ctrl_time = time.time()

# --- STEP 3: MAIN LOOP ---
while not glfw.window_should_close(window):

    now = time.time()
    # Fix: clamp control-loop dt. mj_step always advances physics by the
    # model's fixed internal timestep regardless of wall-clock time, but the
    # waypoint ramp / yaw ramp / altitude integrator above use this dt. An
    # unclamped stall (window drag, GC pause, alt-tab) inflates dt for one
    # iteration, injecting a huge one-shot command into a single physics
    # step -- which is exactly what blows up QACC. 0.05s caps that at a
    # worst case of ~20Hz, well below any real render/control stall.
    DT_MAX = 0.05
    dt = min(DT_MAX, max(1e-4, now - last_ctrl_time))
    last_ctrl_time = now

    if takeoff_requested:
        ctrl.request_takeoff()
        takeoff_requested = False

    # Planner: translate held keys into a smoothly ramped waypoint (Fix #1, #2)
    ctrl.process_high_level_commands(key_held, dt)

    # Controller: compute the physics required to reach that waypoint
    ctrl.step(dt)

    mujoco.mj_step(model, data)

    current_time = time.time()
    if current_time - last_time > (1.0 / 60.0):
        width, height = glfw.get_framebuffer_size(window)
        viewport = mujoco.MjrRect(0, 0, width, height)
        mujoco.mjv_updateScene(model, data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scene)

        draw_frame(scene, np.zeros(3), np.eye(3))
        if body_id != -1:
            body_pos = data.xpos[body_id].copy()
            body_mat = data.xmat[body_id].reshape(3, 3).copy()
            draw_frame(scene, body_pos, body_mat)

        mujoco.mjr_render(viewport, scene, context)

        overlay_text = f"Phase: {ctrl.phase.upper()}"
        if ctrl.phase == "flying":
            overlay_text += f"\nTarget X/Y: {ctrl.target_pos[0]:.2f}, {ctrl.target_pos[1]:.2f}\nTarget Yaw: {np.degrees(ctrl.target_yaw):.1f} deg"

        mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT, viewport, "Planner Status", overlay_text, context)

        glfw.swap_buffers(window)
        glfw.poll_events()
        last_time = current_time

glfw.terminate()
