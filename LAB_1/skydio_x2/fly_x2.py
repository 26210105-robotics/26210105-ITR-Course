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

print("Model timestep:", model.opt.timestep)
print("Bodies:")
for i in range(model.nbody):
    print(" ", i, mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i))

# --- STEP 1: HELD-KEY STATE TRACKER ---
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

    POS_CMD_SPEED = 1.5       # m/s the target waypoint moves while a key is held (was 5)
    YAW_CMD_RATE = 0.4        # rad/s yaw ramp while q/e held (was 0.6)
    POS_BOUND = 3.0           # keep target within this box around origin (m)
    ALT_BOUND_HI = 2.0        # max target altitude (m)

    KP_POS = 0.8
    KP_VEL = 0.3
    MAX_TILT = 0.15
    MAX_CMD_VEL = 1.5

    KP_ALT, KD_ALT = 8.0, 3.0
    KI_ALT = 1.5
    ALT_I_CLAMP = 1.0

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
            self.takeoff_t0 = self.data.time       # simulation time, not wall clock
            self.target_pos[0:2] = self.data.qpos[0:2].copy()
            self.target_pos[2] = self.TARGET_ALT
            _, _, self.target_yaw = self._quat_to_euler(self.data.sensor("body_quat").data)

    def process_high_level_commands(self, held, dt):
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

        # Clamp waypoint to a safe operating box so held keys can't run it away
        self.target_pos[0] = np.clip(self.target_pos[0], -self.POS_BOUND, self.POS_BOUND)
        self.target_pos[1] = np.clip(self.target_pos[1], -self.POS_BOUND, self.POS_BOUND)
        self.target_pos[2] = np.clip(self.target_pos[2], self.GROUND_ALT, self.ALT_BOUND_HI)

        if held['q']: self.target_yaw += self.YAW_CMD_RATE * dt
        if held['e']: self.target_yaw -= self.YAW_CMD_RATE * dt

    def step(self, dt):
        d = self.data

        # Safety guard: if the sim already went non-finite, stop commanding
        if not np.isfinite(d.qvel).all() or not np.isfinite(d.qpos).all():
            self.alt_int = 0.0
            return

        if self.phase == "ground":
            for act_id in motor_ids: d.ctrl[act_id] = 0.0
            return

        pos = d.qpos[0:3]
        vel = d.qvel[0:3]
        quat = d.sensor("body_quat").data
        gyro = d.sensor("body_gyro").data
        roll, pitch, yaw = self._quat_to_euler(quat)

        # If the drone somehow drifted outside a larger safety radius, pull
        # the waypoint home and reset the integrator (does not teleport, just
        # commands recovery).
        if np.linalg.norm(pos[0:2]) > (self.POS_BOUND + 1.5):
            self.target_pos[0:2] = np.zeros(2)
            self.alt_int = 0.0

        # Altitude
        if self.phase == "takeoff":
            elapsed = d.time - self.takeoff_t0     # sim time delta
            frac = min(1.0, elapsed / self.TAKEOFF_TIME)
            current_target_z = self.GROUND_ALT + frac * (self.TARGET_ALT - self.GROUND_ALT)
            if frac >= 1.0 and abs(self.TARGET_ALT - pos[2]) < 0.05 and abs(vel[2]) < 0.05:
                self.phase = "flying"
        else:
            current_target_z = self.target_pos[2]

        alt_err = current_target_z - pos[2]
        self.alt_int = np.clip(self.alt_int + alt_err * dt, -self.ALT_I_CLAMP, self.ALT_I_CLAMP)

        alt_cmd = self.KP_ALT * alt_err + self.KI_ALT * self.alt_int - self.KD_ALT * vel[2]
        base_thrust = self.HOVER_THRUST + alt_cmd / 4.0

        # Position / velocity -> tilt
        target_roll, target_pitch = 0.0, 0.0
        if self.phase == "flying":
            pos_err = self.target_pos[0:2] - pos[0:2]
            target_vel_global = np.clip(self.KP_POS * pos_err, -self.MAX_CMD_VEL, self.MAX_CMD_VEL)
            vel_err_global = target_vel_global - vel[0:2]

            c, s = np.cos(yaw), np.sin(yaw)
            local_vel_err_x =  c * vel_err_global[0] + s * vel_err_global[1]
            local_vel_err_y = -s * vel_err_global[0] + c * vel_err_global[1]

            target_pitch = np.clip(-self.KP_VEL * local_vel_err_x, -self.MAX_TILT, self.MAX_TILT)
            target_roll  = np.clip( self.KP_VEL * local_vel_err_y, -self.MAX_TILT, self.MAX_TILT)

        # Attitude & yaw
        yaw_err = self.target_yaw - yaw
        yaw_err = (yaw_err + np.pi) % (2 * np.pi) - np.pi

        roll_cmd  = self.KP_ATT * (target_roll  - roll)  - self.KD_ATT * gyro[0]
        pitch_cmd = self.KP_ATT * (target_pitch - pitch) - self.KD_ATT * gyro[1]
        yaw_cmd   = self.KP_YAW * yaw_err - self.KD_YAW * gyro[2]

        motor_cmd = base_thrust + roll_cmd * self.roll_sign + pitch_cmd * self.pitch_sign + yaw_cmd * self.YAW_SIGN
        motor_cmd = np.clip(motor_cmd, 0.0, self.CTRL_MAX)
        for i, act_id in enumerate(motor_ids):
            d.ctrl[act_id] = motor_cmd[i]

ctrl = X2Controller(model, data)
data.qpos[2] = 0.1
mujoco.mj_forward(model, data)

# --- STEP 3: MAIN LOOP ---
# Physics runs at the model's own fixed timestep; controller is driven by the
# same dt. Rendering is throttled to ~60 Hz separately.
sim_dt = model.opt.timestep
last_render = time.time()

while not glfw.window_should_close(window):

    if takeoff_requested:
        ctrl.request_takeoff()
        takeoff_requested = False

    # Planner (waypoint / yaw ramp) and controller both use the sim dt.
    ctrl.process_high_level_commands(key_held, sim_dt)
    ctrl.step(sim_dt)

    mujoco.mj_step(model, data)

    # Render throttled to ~60 Hz
    now = time.time()
    if now - last_render > (1.0 / 60.0):
        width, height = glfw.get_framebuffer_size(window)
        viewport = mujoco.MjrRect(0, 0, width, height)
        mujoco.mjv_updateScene(model, data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scene)

        # Draw world and body frames (rotation matrix arrows)
        draw_frame(scene, np.zeros(3), np.eye(3))
        if body_id != -1:
            body_pos = data.xpos[body_id].copy()
            body_mat = data.xmat[body_id].reshape(3, 3).copy()
            draw_frame(scene, body_pos, body_mat)

        mujoco.mjr_render(viewport, scene, context)

        # Top-left status
        overlay_text = f"Phase: {ctrl.phase.upper()}"
        if ctrl.phase == "flying":
            overlay_text += (
                f"\nTarget X/Y: {ctrl.target_pos[0]:.2f}, {ctrl.target_pos[1]:.2f}"
                f"\nTarget Yaw: {np.degrees(ctrl.target_yaw):.1f} deg"
                f"\nSim time: {data.time:.2f} s"
            )

        mujoco.mjr_overlay(
            mujoco.mjtFont.mjFONT_NORMAL,
            mujoco.mjtGridPos.mjGRID_TOPLEFT,
            viewport,
            "Planner Status",
            overlay_text,
            context,
        )

        # Top-right numeric rotation matrix (World <- Body)
        if body_id != -1:
            R = data.xmat[body_id].reshape(3, 3)
            mat_text = (
                f"[{R[0,0]:+.2f} {R[0,1]:+.2f} {R[0,2]:+.2f}]\n"
                f"[{R[1,0]:+.2f} {R[1,1]:+.2f} {R[1,2]:+.2f}]\n"
                f"[{R[2,0]:+.2f} {R[2,1]:+.2f} {R[2,2]:+.2f}]"
            )
            mujoco.mjr_overlay(
                mujoco.mjtFont.mjFONT_NORMAL,
                mujoco.mjtGridPos.mjGRID_TOPRIGHT,
                viewport,
                "Body Rotation Matrix (World <- Body)",
                mat_text,
                context,
            )

        glfw.swap_buffers(window)
        glfw.poll_events()
        last_render = now

glfw.terminate()
