#!/usr/bin/env python3
"""
batch_mink.py - batch-run the Fr_Mink.py pick-and-place over many cube / tray poses.

Same controller as Fr_Mink.py (Mink IK offline -> quintic segments -> velocity
actuators + gravity comp), but:
  * the cube and tray (frame) poses come from .txt files instead of the XML,
  * opens the MuJoCo viewer in real time by default (--headless to skip it),
  * every trial is scored and written to a CSV.

Pose file format: a dict  { 0: [x, y, z, qw, qx, qy, qz], 1: [...], ... }  (sorted by
index), a list of lists, or one pose per line (space/comma separated, one header line
allowed). '#' comments are ignored. Each pose is:
    x y z                      -> no rotation
    x y z yaw                  -> yaw about world z, degrees (use --yaw-rad for radians)
    x y z qw qx qy qz          -> full quaternion, MuJoCo (w, x, y, z) order

Usage:
    python batch_mink.py        # Pos3.xml + cube_poses.txt + frame_poses.txt from this folder
    python batch_mink.py --xml Pos3.xml --cubes cube_poses.txt --frames frame_poses.txt
    python batch_mink.py --xml Pos3.xml --cubes cube_poses.txt                 # tray from XML
    python batch_mink.py --xml Pos3.xml --cubes c.txt --frames t.txt --cross    # every combination
    python batch_mink.py --headless      # no viewer, much faster (sim_time_s is identical)
    python batch_mink.py --fast          # viewer on, but not throttled to real time
"""
import argparse
import ast
import csv
import itertools
import os
import time

import mujoco
import numpy as np
import mink

ARM_DOF = 7
HOME_QPOS = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])

# --- Scene names ---
CUBE_BODY = "cube"
CUBE_GEOM = "cube_geom"
TRAY_BODY = "tray"
TRAY_FLOOR_GEOM = "tray_bottom"
EE_SITE = "gripper"
TRAY_INNER_HALF = 0.07    # inner face of the tray walls (wall centre 0.075 - half thickness 0.005)

HOVER_OFFSET = np.array([0.0, 0.0, 0.25])
PLACE_CLEARANCE = 0.005

GRIPPER_OPEN = 0.1
GRIPPER_CLOSED = -0.2
FINGER_OPEN_QPOS = 0.04

KP_VEL = 10.0
MAX_JOINT_SPEED = 0.8
REACH_TOL = 0.01
VEL_TOL = 0.05
GRIP_VEL_TOL = 1e-3
GRIP_SETTLE_STEPS = 100

IK_SOLVER = "daqp"
IK_DT = 0.01
IK_DAMPING = 1e-3
IK_MAX_ITERS = 1000
POSTURE_COST = 1e-2

# --- Batch-only parameters ---
MOVE_TIMEOUT_PAD = 10.0   # s of sim time allowed beyond a move's planned duration
GRIP_TIMEOUT = 5.0        # s of sim time allowed for a grip/release to settle
FINAL_SETTLE = 1.0        # s held at home before scoring
IK_FAIL_MM = 1.0          # IK position error above this = target unreachable
LIFT_MIN = 0.10           # cube must rise at least this much during "Lift"
PLACE_Z_TOL = 0.01        # cube centre must end within this of its resting height in the tray


# ------------------------------------------------------------
# Helpers (unchanged from Fr_Mink.py)
# ------------------------------------------------------------
def orientation_error(target_quat, current_xmat):
    cur_quat = np.zeros(4)
    mujoco.mju_mat2Quat(cur_quat, current_xmat)
    cur_conj = np.zeros(4)
    mujoco.mju_negQuat(cur_conj, cur_quat)
    err_quat = np.zeros(4)
    mujoco.mju_mulQuat(err_quat, target_quat, cur_conj)
    err = np.zeros(3)
    mujoco.mju_quat2Vel(err, err_quat, 1.0)
    return err


def quintic(q0, qf, s, T):
    s = min(max(s, 0.0), 1.0)
    scale = 10 * s**3 - 15 * s**4 + 6 * s**5
    dscale = (30 * s**2 - 60 * s**3 + 30 * s**4) / T
    return q0 + (qf - q0) * scale, (qf - q0) * dscale


def segment_duration(q0, qf):
    return max(1.875 * np.max(np.abs(qf - q0)) / MAX_JOINT_SPEED, 0.2)


def yaw_quat(yaw):
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def quat_yaw(q):
    w, x, y, z = q
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def square_yaw(yaw):
    """Equivalent yaw in [-45, 45) deg for a part that is symmetric every 90 deg."""
    return (yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4


# ------------------------------------------------------------
# Pose files
# ------------------------------------------------------------
def _to_pose(vals, where, yaw_in_rad):
    vals = [float(v) for v in vals]
    if len(vals) == 3:
        quat = np.array([1.0, 0.0, 0.0, 0.0])
    elif len(vals) == 4:
        yaw = vals[3] if yaw_in_rad else np.radians(vals[3])
        quat = yaw_quat(yaw)
    elif len(vals) == 7:
        quat = np.array(vals[3:])
    else:
        raise ValueError(f"{where}: expected 3, 4 or 7 numbers, got {len(vals)}")
    return np.array(vals[:3]), quat / np.linalg.norm(quat)


def load_poses(path, yaw_in_rad=False):
    """Accepts either
         a dict   { 0: [x, y, z, qw, qx, qy, qz], 1: [...], ... }   (sorted by index)
         a list   [ [x, y, z, ...], [...], ... ]
         plain lines  'x y z ...'  (space/comma separated, one header line allowed)
       '#' comments are ignored in all formats."""
    with open(path) as f:
        text = "\n".join(line.split("#")[0] for line in f)

    try:
        obj = ast.literal_eval(text.strip())
    except (ValueError, SyntaxError):
        obj = None

    if isinstance(obj, dict):
        poses = [_to_pose(obj[k], f"{path} index {k}", yaw_in_rad) for k in sorted(obj)]
    elif isinstance(obj, (list, tuple)) and obj and isinstance(obj[0], (list, tuple)):
        poses = [_to_pose(v, f"{path} entry {i}", yaw_in_rad) for i, v in enumerate(obj)]
    else:
        poses, header_skipped = [], False
        for ln, raw in enumerate(text.splitlines(), 1):
            line = raw.replace(",", " ").strip()
            if not line:
                continue
            try:
                vals = [float(v) for v in line.split()]
            except ValueError:
                if not poses and not header_skipped:
                    header_skipped = True
                    continue
                raise ValueError(f"{path}:{ln}: cannot parse {raw.strip()!r}")
            poses.append(_to_pose(vals, f"{path}:{ln}", yaw_in_rad))

    if not poses:
        raise ValueError(f"{path}: no poses found")
    return poses


# ------------------------------------------------------------
# Pick-and-place runner (one model, reused for every trial)
# ------------------------------------------------------------
class PickPlace:
    def __init__(self, xml_path):
        m = mujoco.MjModel.from_xml_path(xml_path)
        m.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        self.model = m
        self.data = mujoco.MjData(m)
        self.ik_data = mujoco.MjData(m)

        self.site_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, EE_SITE)
        self.cube_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, CUBE_BODY)
        self.cube_geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, CUBE_GEOM)
        self.tray_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, TRAY_BODY)
        self.tray_floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, TRAY_FLOOR_GEOM)
        self.cube_qadr = m.jnt_qposadr[m.body_jntadr[self.cube_id]]   # freejoint: pos(3) + quat(4)
        self.cube_half = m.geom_size[self.cube_geom, 2]

        arm_jnt = np.array([m.actuator_trnid[i, 0] for i in range(ARM_DOF)])
        self.arm_q = m.jnt_qposadr[arm_jnt]
        self.arm_v = m.jnt_dofadr[arm_jnt]
        self.frozen = np.setdiff1d(np.arange(m.nv), self.arm_v)
        finger_jnt = [j for j in range(m.njnt)
                      if "finger" in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or "").lower()]
        self.finger_q = m.jnt_qposadr[finger_jnt]
        self.finger_v = m.jnt_dofadr[finger_jnt]
        self.grip_act = ARM_DOF
        self.q_lo = np.where(m.jnt_limited[arm_jnt], m.jnt_range[arm_jnt, 0], -np.inf)
        self.q_hi = np.where(m.jnt_limited[arm_jnt], m.jnt_range[arm_jnt, 1], np.inf)

        # XML poses, used when a pose file does not supply one
        self.xml_cube = (m.qpos0[self.cube_qadr:self.cube_qadr + 3].copy(),
                         m.qpos0[self.cube_qadr + 3:self.cube_qadr + 7].copy())
        self.xml_tray = (m.body_pos[self.tray_id].copy(), m.body_quat[self.tray_id].copy())

        # Mink
        self.configuration = mink.Configuration(m)
        self.ee_task = mink.FrameTask(frame_name=EE_SITE, frame_type="site",
                                      position_cost=1.0, orientation_cost=1.0, lm_damping=1.0)
        self.posture_task = mink.PostureTask(m, cost=POSTURE_COST)
        self.ik_tasks = [self.ee_task, self.posture_task]
        self.ik_limits = [mink.ConfigurationLimit(m)]

        # Gripper orientation at HOME (pointing down); independent of cube/tray
        self.ik_data.qpos[:] = m.qpos0
        self.ik_data.qpos[self.arm_q] = HOME_QPOS
        mujoco.mj_kinematics(m, self.ik_data)
        self.home_quat = np.zeros(4)
        mujoco.mju_mat2Quat(self.home_quat, self.ik_data.site_xmat[self.site_id])

    # -- scene ----------------------------------------------------------
    def set_scene(self, cube_pose, tray_pose):
        m = self.model
        cube_pos, cube_quat = cube_pose if cube_pose is not None else self.xml_cube
        tray_pos, tray_quat = tray_pose if tray_pose is not None else self.xml_tray
        m.qpos0[self.cube_qadr:self.cube_qadr + 3] = cube_pos      # freejoint start pose
        m.qpos0[self.cube_qadr + 3:self.cube_qadr + 7] = cube_quat
        m.body_pos[self.tray_id] = tray_pos                        # tray is a static body
        m.body_quat[self.tray_id] = tray_quat

        q_home_full = m.qpos0.copy()
        q_home_full[self.arm_q] = HOME_QPOS
        self.posture_task.set_target(q_home_full)

    # -- IK ----------------------------------------------------------------
    def solve_ik(self, target_pos, target_quat, q_init, pos_tol=1e-4, rot_tol=1e-3):
        t0 = time.perf_counter()
        cfg, sid = self.configuration, self.site_id
        q = self.model.qpos0.copy()
        q[self.arm_q] = q_init
        cfg.update(q)
        self.ee_task.set_target(mink.SE3.from_rotation_and_translation(
            rotation=mink.SO3(wxyz=target_quat), translation=target_pos))

        for _ in range(IK_MAX_ITERS):
            err_p = np.linalg.norm(target_pos - cfg.data.site_xpos[sid])
            err_r = np.linalg.norm(orientation_error(target_quat, cfg.data.site_xmat[sid]))
            if err_p < pos_tol and err_r < rot_tol:
                break
            vel = mink.solve_ik(cfg, self.ik_tasks, IK_DT, IK_SOLVER,
                                damping=IK_DAMPING, limits=self.ik_limits)
            vel[self.frozen] = 0.0
            cfg.integrate_inplace(vel, IK_DT)

        pe = np.linalg.norm(target_pos - cfg.data.site_xpos[sid]) * 1000
        re = np.degrees(np.linalg.norm(orientation_error(target_quat, cfg.data.site_xmat[sid])))
        return cfg.q[self.arm_q].copy(), pe, re, (time.perf_counter() - t0) * 1000

    def plan(self, log=print):
        m, d = self.model, self.ik_data
        d.qpos[:] = m.qpos0
        mujoco.mj_forward(m, d)

        pick = d.xpos[self.cube_id].copy()
        floor_top = d.geom_xpos[self.tray_floor][2] + m.geom_size[self.tray_floor, 2]
        place = np.array([d.geom_xpos[self.tray_floor][0], d.geom_xpos[self.tray_floor][1],
                          floor_top + self.cube_half + PLACE_CLEARANCE])

        cube_yaw = quat_yaw(m.qpos0[self.cube_qadr + 3:self.cube_qadr + 7])
        tray_yaw = quat_yaw(m.body_quat[self.tray_id])
        pick_quat, place_quat = np.zeros(4), np.zeros(4)
        mujoco.mju_mulQuat(pick_quat, yaw_quat(square_yaw(cube_yaw)), self.home_quat)
        # place square with the tray (same as Fr_Mink.py when the tray has zero yaw)
        mujoco.mju_mulQuat(place_quat, yaw_quat(square_yaw(tray_yaw)), self.home_quat)

        log(f"Cube at {np.round(pick, 3)}, yaw {np.degrees(cube_yaw):.1f} deg "
            f"(grasp yaw {np.degrees(square_yaw(cube_yaw)):.1f} deg)")
        log(f"Tray place target: {np.round(place, 3)}")
        log("")

        targets = [("Pre-Pick Hover", pick + HOVER_OFFSET, pick_quat),
                   ("Pick", pick, pick_quat),
                   ("Pre-Place Hover", place + HOVER_OFFSET, place_quat),
                   ("Place", place, place_quat)]
        qs, errs, rots, ms = [], [], [], []
        q_prev = HOME_QPOS
        for name, pos, quat in targets:
            log(f"Solving Mink IK for {name}...")
            q, pe, re, t = self.solve_ik(pos, quat, q_prev)
            log(f"   pos err = {pe:.3f} mm")
            log(f"   solve time = {t:.2f} ms")
            qs.append(q); errs.append(pe); rots.append(re); ms.append(t)
            q_prev = q
        log("")
        log(f"Total path solving time: {sum(ms):.2f} ms")
        log("")

        return dict(pre_pick=qs[0], pick=qs[1], pre_place=qs[2], place=qs[3],
                    ik_names=[t[0] for t in targets],
                    ik_pos_err_mm=errs, ik_rot_err_deg=rots, ik_ms=ms,
                    cube_yaw=cube_yaw, tray_yaw=tray_yaw)

    # -- execution -------------------------------------------------------
    def execute(self, p, viewer=None, realtime=False, log=print):
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)                      # cube back to this trial's qpos0
        d.qpos[self.arm_q] = HOME_QPOS
        d.qpos[self.finger_q] = FINGER_OPEN_QPOS
        mujoco.mj_forward(m, d)

        seq = [
            ("Approach Pre-Pick", p["pre_pick"],  GRIPPER_OPEN,   "move"),
            ("Descend to Pick",   p["pick"],      GRIPPER_OPEN,   "move"),
            ("Grasp",             p["pick"],      GRIPPER_CLOSED, "grip"),
            ("Lift",              p["pre_pick"],  GRIPPER_CLOSED, "move"),
            ("Transport",         p["pre_place"], GRIPPER_CLOSED, "move"),
            ("Descend to Place",  p["place"],     GRIPPER_CLOSED, "move"),
            ("Release",           p["place"],     GRIPPER_OPEN,   "grip"),
            ("Retract",           p["pre_place"], GRIPPER_OPEN,   "move"),
            ("Return Home",       HOME_QPOS,      GRIPPER_OPEN,   "move"),
        ]

        dt = m.opt.timestep
        cube_z0 = d.xpos[self.cube_id][2]
        step_idx, settled, step_t, hold_t = 0, 0, 0.0, 0.0
        qcmd, qd_ff = HOME_QPOS.copy(), np.zeros(ARM_DOF)
        seg_start = HOME_QPOS.copy()
        seg_T, seg_s = segment_duration(seg_start, seq[0][1]), 0.0
        status, lift_dz = "completed", np.nan
        step_times = []                      # (step name, simulated seconds)
        wall0 = time.time()
        res = dict(step_times=step_times, sim_time=np.nan, wall_time=np.nan)

        def finish(st):
            if np.isnan(res["sim_time"]):    # sequence never finished: time up to now
                res.update(sim_time=d.time, wall_time=time.time() - wall0)
            res.update(status=st, lift_dz=lift_dz)
            return res

        log(f"[1/{len(seq)}] {seq[0][0]}")

        while True:
            if viewer is not None and not viewer.is_running():
                return finish("viewer_closed")
            t0 = time.time()

            if step_idx < len(seq):
                name, q_target, gripper_cmd, kind = seq[step_idx]
                seg_s = min(seg_s + dt / seg_T, 1.0)
                qcmd, qd_ff = quintic(seg_start, q_target, seg_s, seg_T)
                qcmd = np.clip(qcmd, self.q_lo, self.q_hi)
                step_t += dt

                if kind == "move":
                    q_err = np.max(np.abs(q_target - d.qpos[self.arm_q]))
                    q_vel = np.max(np.abs(d.qvel[self.arm_v]))
                    done = seg_s >= 1.0 and q_err < REACH_TOL and q_vel < VEL_TOL
                    limit = seg_T + MOVE_TIMEOUT_PAD
                else:
                    fv = np.max(np.abs(d.qvel[self.finger_v])) if len(self.finger_v) else 0.0
                    settled = settled + 1 if fv < GRIP_VEL_TOL else 0
                    done = settled >= GRIP_SETTLE_STEPS
                    limit = GRIP_TIMEOUT

                if not done and step_t > limit:
                    step_times.append((name, step_t))
                    log(f"TIMEOUT in {name} after {step_t:.2f} s")
                    return finish(f"timeout:{name}")

                if done:
                    if name == "Lift":
                        lift_dz = d.xpos[self.cube_id][2] - cube_z0
                    step_times.append((name, step_t))
                    step_idx += 1
                    settled, step_t = 0, 0.0
                    if step_idx < len(seq):
                        seg_start = qcmd.copy()
                        seg_T = segment_duration(seg_start, seq[step_idx][1])
                        seg_s = 0.0
                        log(f"[{step_idx + 1}/{len(seq)}] {seq[step_idx][0]}")
                    else:
                        res.update(sim_time=d.time, wall_time=time.time() - wall0)
                        log("Sequence complete. Holding home.")
            else:
                gripper_cmd = GRIPPER_OPEN
                qd_ff = np.zeros(ARM_DOF)
                hold_t += dt
                if hold_t >= FINAL_SETTLE:
                    return finish(status)

            v_cmd = qd_ff + KP_VEL * (qcmd - d.qpos[self.arm_q])
            for i in range(ARM_DOF):
                if m.actuator_ctrllimited[i]:
                    v_cmd[i] = np.clip(v_cmd[i], *m.actuator_ctrlrange[i])
            d.ctrl[:ARM_DOF] = v_cmd
            d.ctrl[self.grip_act] = gripper_cmd
            d.qfrc_applied[self.arm_v] = d.qfrc_bias[self.arm_v]

            mujoco.mj_step(m, d)
            if viewer is not None:
                viewer.sync()
                if realtime:
                    rest = dt - (time.time() - t0)
                    if rest > 0:
                        time.sleep(rest)

    def score(self):
        """Final cube pose expressed in the tray frame."""
        m, d = self.model, self.data
        R = d.xmat[self.tray_id].reshape(3, 3)
        rel = R.T @ (d.xpos[self.cube_id] - d.xpos[self.tray_id])
        floor_top_local = m.geom_pos[self.tray_floor][2] + m.geom_size[self.tray_floor, 2]
        rest_z = floor_top_local + self.cube_half
        in_xy = max(abs(rel[0]), abs(rel[1])) <= TRAY_INNER_HALF - self.cube_half + 0.005
        in_z = abs(rel[2] - rest_z) < PLACE_Z_TOL
        return bool(in_xy and in_z), rel, d.xpos[self.cube_id].copy()


# ------------------------------------------------------------
# Batch driver
# ------------------------------------------------------------
def build_pairs(cubes, trays, cross):
    if trays is None:
        return [(c, None) for c in cubes]
    if cross:
        return list(itertools.product(cubes, trays))
    if len(trays) == 1:
        return [(c, trays[0]) for c in cubes]
    if len(cubes) == 1:
        return [(cubes[0], t) for t in trays]
    if len(cubes) != len(trays):
        raise SystemExit(f"{len(cubes)} cube poses vs {len(trays)} tray poses: "
                         "make them equal length, or pass --cross for every combination")
    return list(zip(cubes, trays))


class TrialLog:
    """Prints to the terminal (indented with '   | ') and writes the same lines to a file."""
    def __init__(self, path):
        self.f = open(path, "w")

    def __call__(self, msg=""):
        print(f"   | {msg}" if msg else "   |")
        self.f.write(msg + "\n")

    def raw(self, msg=""):                # header / footer lines, no indent
        print(msg)
        self.f.write(msg + "\n")

    def close(self):
        self.f.close()


def results_block(log, plan, res, rel, placed):
    bar = "=" * 52
    log(bar)
    log(" RESULTS")
    log(bar)
    if plan is not None:
        log(" 1) Path solving time (Mink IK, offline)")
        for name, t in zip(plan["ik_names"], plan["ik_ms"]):
            log(f"      {name:<22}{t:8.2f} ms")
        log(f"      {'Total':<22}{sum(plan['ik_ms']):8.2f} ms")
    if res is not None:
        done = res["status"] == "completed"
        log(f" 2) Overall simulation time ({'sequence complete' if done else 'stopped: ' + res['status']})")
        log(f"      Simulated time      {res['sim_time']:8.3f} s")
        log(f"      Wall-clock time     {res['wall_time']:8.3f} s")
        log("      Per step (simulated):")
        for name, t in res["step_times"]:
            log(f"        {name:<20}{t:6.3f} s")
        lift = res["lift_dz"]
        log(" 3) Outcome")
        log(f"      Cube lift during Lift  {'n/a' if np.isnan(lift) else f'{lift:.3f} m'}")
        log(f"      Final offset in tray   x {rel[0]*1000:+.1f} mm, y {rel[1]*1000:+.1f} mm "
            f"-> {'in tray' if placed else 'NOT in tray'}")
    log(bar)


def stats_line(name, vals, unit):
    vals = [v for v in vals if v != "" and v is not None and not np.isnan(v)]
    if not vals:
        return f" {name:<14} no data"
    return (f" {name:<14} mean {np.mean(vals):9.3f} {unit:<2}  min {np.min(vals):9.3f}"
            f"   max {np.max(vals):9.3f}")


def run_batch(args, pp, pairs, viewer=None):
    fields = ["index", "status", "flag",
              "cube_x", "cube_y", "cube_z", "cube_qw", "cube_qx", "cube_qy", "cube_qz",
              "frame_x", "frame_y", "frame_z",
              "pos_err_pre_pick_hover_mm", "pos_err_pick_mm",
              "pos_err_pre_place_hover_mm", "pos_err_place_mm",
              "ik_total_ms", "sim_time_s", "wall_time_s"]
    os.makedirs(args.log_dir, exist_ok=True)
    rows = []
    batch_t0 = time.time()

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i, (cube, tray) in enumerate(pairs):
            pp.set_scene(cube, tray)
            cpos, cquat = cube if cube is not None else pp.xml_cube
            tpos, _ = tray if tray is not None else pp.xml_tray
            row = dict(index=i, status="", flag="",
                       cube_x=round(cpos[0], 4), cube_y=round(cpos[1], 4), cube_z=round(cpos[2], 4),
                       cube_qw=round(cquat[0], 4), cube_qx=round(cquat[1], 4),
                       cube_qy=round(cquat[2], 4), cube_qz=round(cquat[3], 4),
                       frame_x=round(tpos[0], 4), frame_y=round(tpos[1], 4), frame_z=round(tpos[2], 4))

            log = TrialLog(os.path.join(args.log_dir, f"run_{i:03d}.log"))
            log.raw(f"=== Run {i + 1}/{len(pairs)}  (index {i}) ===")
            log.raw(f"   cube  pos {[round(float(v), 4) for v in cpos]}  "
                    f"quat {[round(float(v), 4) for v in cquat]}")
            log.raw(f"   frame pos {[round(float(v), 4) for v in tpos]}")

            flags, plan, res, rel, placed = [], None, None, None, False
            try:
                plan = pp.plan(log)
                errs = plan["ik_pos_err_mm"]
                row.update(pos_err_pre_pick_hover_mm=round(errs[0], 3),
                           pos_err_pick_mm=round(errs[1], 3),
                           pos_err_pre_place_hover_mm=round(errs[2], 3),
                           pos_err_place_mm=round(errs[3], 3),
                           ik_total_ms=round(sum(plan["ik_ms"]), 2))
                if max(errs) >= IK_FAIL_MM:
                    flags.append("ik_unreachable")

                if flags and args.skip_unreachable:
                    row["status"] = "skipped"
                    log("IK unreachable - skipping simulation")
                else:
                    res = pp.execute(plan, viewer, args.realtime, log)
                    placed, rel, _ = pp.score()
                    row["status"] = "complete" if res["status"] == "completed" else res["status"]
                    row["sim_time_s"] = round(res["sim_time"], 3)
                    row["wall_time_s"] = round(res["wall_time"], 3)
                    if not res["lift_dz"] > LIFT_MIN:   # also catches NaN (never reached Lift)
                        flags.append("not_lifted")
                    if not placed:
                        flags.append("not_in_tray")
                log("")
                results_block(log, plan, res, rel, placed)
            except Exception as e:  # keep the batch going
                row["status"] = f"error: {type(e).__name__}: {e}"
                log(row["status"])

            row["flag"] = ";".join(flags)
            log.raw(f"   -> {row['status']}" + (f"  [{row['flag']}]" if row["flag"] else ""))
            log.raw("")
            log.close()

            w.writerow(row)
            f.flush()
            rows.append(row)
            if row["status"] == "viewer_closed":
                print("Viewer closed - stopping batch.")
                break

    # ---------------- batch summary ----------------
    n_done = sum(r["status"] == "complete" for r in rows)
    n_flag = sum(bool(r["flag"]) for r in rows)
    ok = [r for r in rows if r["status"] == "complete"]
    bar = "=" * 52
    lines = [bar,
             f" BATCH SUMMARY   {n_done}/{len(pairs)} runs completed"
             f"{f'   ({n_flag} flagged)' if n_flag else ''}",
             bar,
             stats_line("ik_total_ms", [r.get("ik_total_ms") for r in rows], "ms"),
             stats_line("sim_time_s", [r.get("sim_time_s") for r in ok], "s"),
             stats_line("wall_time_s", [r.get("wall_time_s") for r in ok], "s"),
             f" Batch wall time {time.time() - batch_t0:.1f} s"]
    bad = [r for r in rows if r["status"] != "complete" or r["flag"]]
    if bad:
        lines.append(" Problem runs:")
        lines += [f"   index {r['index']:>3}: {r['status']}" + (f" [{r['flag']}]" if r["flag"] else "")
                  for r in bad]
    lines += [f" Results: {args.out}   Logs: {args.log_dir}/", bar]
    summary = "\n".join(lines)
    print(summary)
    with open(os.path.join(args.log_dir, "batch_summary.log"), "w") as sf:
        sf.write(summary + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--xml", default="FrPos.xml")
    ap.add_argument("--cubes", default="cube_poses.txt", help="cube pose file")
    ap.add_argument("--frames", default="frame_poses.txt",
                    help="frame (tray) pose file; pass --frames \"\" to keep the XML tray")
    ap.add_argument("--cross", action="store_true", help="run every cube x tray combination")
    ap.add_argument("--yaw-rad", action="store_true", help="yaw column is in radians")
    ap.add_argument("--out", default="batch_results.csv")
    ap.add_argument("--log-dir", default="batch_logs", help="per-run logs + batch_summary.log")
    ap.add_argument("--skip-unreachable", action="store_true",
                    help="don't simulate trials whose IK misses by more than 1 mm")
    ap.add_argument("--headless", action="store_true",
                    help="no viewer window (fast); the viewer is on by default")
    ap.add_argument("--fast", action="store_true",
                    help="with the viewer, run as fast as possible instead of real time")
    args = ap.parse_args()

    cubes = load_poses(args.cubes, args.yaw_rad)
    trays = load_poses(args.frames, args.yaw_rad) if args.frames else None
    pairs = build_pairs(cubes, trays, args.cross)
    print(f"{len(pairs)} trials ({len(cubes)} cube poses"
          f"{f', {len(trays)} tray poses' if trays else ''})\n")

    args.view = not args.headless
    args.realtime = args.view and not args.fast

    mode = ("ON (real time)" if args.realtime else "ON (fast)") if args.view else "OFF (--headless)"
    print(f"Viewer: {mode}   scene: {args.xml}\n")

    pp = PickPlace(args.xml)
    if args.view:
        import mujoco.viewer
        with mujoco.viewer.launch_passive(pp.model, pp.data) as viewer:
            run_batch(args, pp, pairs, viewer)
    else:
        run_batch(args, pp, pairs)


if __name__ == "__main__":
    main()
