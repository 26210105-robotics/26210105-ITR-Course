#!/usr/bin/env python3
"""
prmth_run.py - replay the cleaned buzz-wire path on the Addverb Heal

INPUT      waypoint CSV (columns x, y, z, qx, qy, qz, qw: end-effector pose in base_link,
           quaternion x,y,z,w). Extra columns (joint1..joint6, ...) are ignored.

TRAJECTORY Quintic (5th-order) polynomial per segment between waypoints (--order 7 for septic),
           for position and orientation (rotation vector relative to the first waypoint).
           Position, velocity and acceleration match at both ends of every segment.
             - corners (direction change more than VIA_ANGLE) and in-place rotations: the arm
               stops there -> it never swings wide around a corner near the wire
             - waypoints on a straight stretch: it passes through them without stopping

IK         Damped least-squares velocity IK with feed-forward:
               twist = trajectory velocity * k (feed-forward) + K * pose error (feedback)
               qdot  = (WJ)^T ((WJ)(WJ)^T + lambda I)^-1 * W twist
           Away from singularities W = I (plain DLS, exact pose).
           NEAR A SINGULARITY (smallest singular value of J below SIGMA_RELAX) the orientation rows
           get a small weight, so the loop POSITION stays exact and the gripper is allowed to tilt
           a little (at most --max-tilt-deg) instead. That is what lets the arm get through
           stretches it could not reach with the exact taught orientation.
           k (0..1) slows the path down whenever a joint would exceed --max-joint-vel, so the arm
           never falls behind the path because of joint-speed saturation.

FK CHECKS  Before anything moves:
             1. every waypoint: IK walked there in small steps, FK(q) must hit the waypoint
                position (< 0.5 mm); orientation exact, or tilted <= --max-tilt-deg near a
                singularity; joint limits; joint jumps
             2. dry run: the whole trajectory simulated with the exact controller -> max FK error
           While moving:
             3. FK of the measured joints vs the trajectory every cycle -> abort above
                ABORT_MM / ABORT_DEG, or near a joint limit

START      If the arm is not already at the first waypoint it moves there in a straight line
           (no lift; --approach-up-mm N for the old "N mm above the start, then down").

LAG        The real arm lags its velocity commands. When the loop falls more than --wait-mm behind
           the path, the path slows down; at --hold-mm it waits for the arm. After a run, the script
           prints how each joint followed its commands (gain, delay).

SAFETY     Prints the plan and waits for Enter. Zero velocity at the end, on abort, on Ctrl+C and
           if /joint_states stops. Refuses to start if another program publishes to the velocity
           topic (Addverb's controller stops with two publishers).

RUN        (robot launch up, teleop closed)
  python3 prmth_run.py wp.csv --dry-run --save-urdf heal.urdf   # checks only + save URDF
  python3 prmth_run.py wp.csv                                   # checks, asks, moves
  python3 prmth_run.py wp.csv --offline heal.urdf --start-joints="j1,j2,j3,j4,j5,j6"
                                                                # checks without the robot
Writes replay_log_<date>_<time>.csv (desired vs FK-measured pose) after a run.
"""

import argparse
import csv
import math
import os
import time
from datetime import datetime

import numpy as np
from scipy.spatial.transform import Rotation

import PyKDL as kdl
from urdf_parser_py.urdf import URDF
from kdl_parser_py.urdf import treeFromUrdfModel

CMD_TOPIC = "/velocity_controller/commands"
CMD_JOINT_ORDER = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]   # as the teleop
JOINT_STATES_TOPIC = "/joint_states"
ROBOT_DESCRIPTION_TOPIC = "/robot_description"
BASE_LINK = "base_link"
TIP_LINK = "end-effector"

PEAK = {5: 15.0 / 8.0, 7: 35.0 / 16.0}   # peak / average speed of a rest-to-rest segment


# =============================================================================================
# Kinematics (KDL, from the URDF)
# =============================================================================================
class Kinematics:
    def __init__(self, urdf_xml, base=BASE_LINK, tip=TIP_LINK):
        robot = URDF.from_xml_string(urdf_xml)
        ok, tree = treeFromUrdfModel(robot)
        if not ok:
            raise RuntimeError("could not build a KDL tree from the URDF")
        self.chain = tree.getChain(base, tip)
        segs = [self.chain.getSegment(i) for i in range(self.chain.getNrOfSegments())]
        self.names = [s.getJoint().getName() for s in segs if s.getJoint().getTypeName() not in ("None", "Fixed")]
        self.n = len(self.names)
        if self.n == 0:
            raise RuntimeError(f"no movable joints between {base} and {tip}")
        self.fk_solver = kdl.ChainFkSolverPos_recursive(self.chain)
        self.jac_solver = kdl.ChainJntToJacSolver(self.chain)
        jm = {j.name: j for j in robot.joints}
        lo, hi = [], []
        for n in self.names:
            j = jm[n]
            if j.type == "continuous" or j.limit is None:
                lo.append(-np.inf)
                hi.append(np.inf)
            else:
                lo.append(j.limit.lower)
                hi.append(j.limit.upper)
        self.lower, self.upper = np.array(lo, float), np.array(hi, float)

    def _arr(self, q):
        a = kdl.JntArray(self.n)
        for i in range(self.n):
            a[i] = float(q[i])
        return a

    def fk(self, q):
        f = kdl.Frame()
        self.fk_solver.JntToCart(self._arr(q), f)
        p = np.array([f.p[0], f.p[1], f.p[2]])
        R = np.array([[f.M[i, j] for j in range(3)] for i in range(3)])
        return p, R

    def jac(self, q):
        J = kdl.Jacobian(self.n)
        self.jac_solver.JntToJac(self._arr(q), J)
        return np.array([[J[i, j] for j in range(self.n)] for i in range(6)])

    def limit_margin(self, q):
        """Smallest distance (rad) to a joint limit."""
        return float(np.min(np.minimum(q - self.lower, self.upper - q)))


def rot_err(R_d, R):
    """Orientation error as a rotation vector in the base frame (R_d * R^T)."""
    return Rotation.from_matrix(R_d @ R.T).as_rotvec()


def sigma_min(J):
    return float(np.linalg.svd(J, compute_uv=False)[-1])


# =============================================================================================
# Singularity handling, shared by the IK check, the dry run and the live run
# =============================================================================================
class SingularityPolicy:
    """Away from a singularity: W = I, tiny damping (exact pose).
    Near one (sigma_min(J) < sigma_relax): orientation rows weighted down to w_ori_min -> position
    kept, orientation allowed to give a little (optional extra damping up to lam_max, default off:
    the weighting already keeps the joint speeds bounded, extra damping only made the loop lag)."""

    def __init__(self, sigma_relax=0.04, w_ori_min=0.02, lam_min=1e-4, lam_max=0.0, sigma_floor=0.015):
        self.sigma_relax, self.w_ori_min = sigma_relax, w_ori_min
        self.lam_min, self.lam_max = lam_min, lam_max
        self.sigma_floor = sigma_floor

    def escape(self, J, smin, q, q_entry):
        """Right at a singularity the arm can get trapped on it (and, from there, slide into the
        other arm configuration, e.g. elbow up <-> down). Below sigma_floor this returns a joint
        direction (0..1 in size) along the singular motion, pointing back to the configuration the
        arm had when it entered the singular stretch. Moving along it barely changes the pose."""
        if q_entry is None or smin >= self.sigma_floor:
            return np.zeros(len(q))
        v = np.linalg.svd(J)[2][-1]
        s = float(v @ (np.asarray(q_entry) - q))
        if abs(s) < 1e-9:
            return np.zeros(len(q))
        return (1.0 - smin / self.sigma_floor) * math.copysign(1.0, s) * v

    def weight(self, smin):
        """Orientation weight from sigma_min of the plain Jacobian."""
        if smin >= self.sigma_relax:
            return 1.0
        return self.w_ori_min + (1.0 - self.w_ori_min) * (smin / self.sigma_relax) ** 2

    def operator(self, J, smin):
        """Matrix A with qdot = A @ twist (weighted DLS)."""
        w = self.weight(smin)
        W = np.diag([1.0, 1.0, 1.0, w, w, w])
        Jw = W @ J
        lam = self.lam_min
        if smin < self.sigma_relax:
            lam += self.lam_max * (1.0 - (smin / self.sigma_relax) ** 2)
        return Jw.T @ np.linalg.solve(Jw @ Jw.T + lam * np.eye(6), W), w


# =============================================================================================
# Polynomial trajectory (quintic by default, septic with --order 7)
# =============================================================================================
def hermite_coeffs(T, b0, b1):
    """Coefficients c[0..2m-1] (each a D-vector) of x(tau) = sum c_k tau^k on [0, T] that match the
    m boundary rows b0, b1 = [position, velocity, acceleration(, jerk)] at tau = 0 and tau = T.
    m = 3 -> quintic, m = 4 -> septic."""
    b0, b1 = np.asarray(b0, float), np.asarray(b1, float)
    m = b0.shape[0]
    c = np.zeros((2 * m, b0.shape[1]))
    for d in range(m):
        c[d] = b0[d] / math.factorial(d)
    M = np.zeros((m, m))
    rhs = np.zeros((m, b0.shape[1]))
    for d in range(m):
        known = sum(c[k] * (math.factorial(k) / math.factorial(k - d)) * T ** (k - d) for k in range(d, m))
        rhs[d] = b1[d] - known
        for j, k in enumerate(range(m, 2 * m)):
            M[d, j] = math.factorial(k) / math.factorial(k - d) * T ** (k - d)
    c[m:] = np.linalg.solve(M, rhs)
    return c


def poly(c, tau, d=0):
    """d-th derivative of the polynomial at tau."""
    out = np.zeros(c.shape[1])
    for k in range(d, len(c)):
        out += c[k] * (math.factorial(k) / math.factorial(k - d)) * tau ** (k - d)
    return out


def skew(w):
    return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])


def right_jacobian(phi):
    th = np.linalg.norm(phi)
    K = skew(phi)
    if th < 1e-6:
        return np.eye(3) - 0.5 * K + K @ K / 6.0
    return np.eye(3) - (1 - math.cos(th)) / th ** 2 * K + (th - math.sin(th)) / th ** 3 * K @ K


class PolyTrajectory:
    """Polynomial segments through poses P (N,3) / R (Rotation, N). stop[i] forces a stop at pose i."""

    def __init__(self, P, R, v_max, w_max, rot_weight=0.03, via_angle_deg=20.0, stop=None, t_min=0.3,
                 order=5):
        self.order = order
        self.m = 3 if order == 5 else 4
        self.P = np.asarray(P, float)
        self.R_ref = R[0]
        rv = (self.R_ref.inv() * R).as_rotvec()
        self.X = np.hstack([self.P, rv])
        N = len(self.P)
        self.stop = np.zeros(N, bool) if stop is None else np.asarray(stop, bool).copy()
        self.stop[0] = self.stop[-1] = True
        self.rot_weight, self.via_cos = rot_weight, math.cos(math.radians(via_angle_deg))
        dist = np.linalg.norm(np.diff(self.P, axis=0), axis=1)
        ang = np.array([(R[i].inv() * R[i + 1]).magnitude() for i in range(N - 1)])
        pk = PEAK[order]
        self.T = np.maximum.reduce([pk * dist / v_max, pk * ang / w_max, np.full(N - 1, t_min)])
        for _ in range(8):                       # build, check peak speeds, stretch slow segments
            self._build()
            ratio = self._peak_ratio(v_max, w_max)
            if ratio.max() <= 1.0 + 1e-3:
                break
            self.T *= np.maximum(1.0, ratio * 1.02)
        self.duration = float(self.t0[-1] + self.T[-1])

    def _via_velocities(self):
        X, T, N = self.X, self.T, len(self.X)
        V = np.zeros_like(X)
        for i in range(1, N - 1):
            if self.stop[i]:
                continue
            u_in, u_out = (X[i] - X[i - 1]) / T[i - 1], (X[i + 1] - X[i]) / T[i]
            w_in = np.concatenate([u_in[:3], self.rot_weight * u_in[3:]])
            w_out = np.concatenate([u_out[:3], self.rot_weight * u_out[3:]])
            ni, no = np.linalg.norm(w_in), np.linalg.norm(w_out)
            if ni > 1e-6 and no > 1e-6 and (w_in @ w_out) / (ni * no) > self.via_cos:
                V[i] = 0.5 * (u_in + u_out)       # gentle bend: pass through
            else:
                self.stop[i] = True               # corner / in-place turn: stop
        return V

    def _build(self):
        V = self._via_velocities()
        Z = [np.zeros(6)] * (self.m - 2)
        self.coef = [hermite_coeffs(self.T[i], [self.X[i], V[i], *Z], [self.X[i + 1], V[i + 1], *Z])
                     for i in range(len(self.T))]
        self.t0 = np.concatenate([[0.0], np.cumsum(self.T)[:-1]])

    def _peak_ratio(self, v_max, w_max):
        r = np.zeros(len(self.T))
        for i, c in enumerate(self.coef):
            for tau in np.linspace(0, self.T[i], 40):
                xd = poly(c, tau, 1)
                Jr = right_jacobian(poly(c, tau)[3:])
                r[i] = max(r[i], np.linalg.norm(xd[:3]) / v_max, np.linalg.norm(Jr @ xd[3:]) / w_max)
        return r

    def sample(self, t):
        """Desired position, rotation matrix, linear velocity, angular velocity (base frame)."""
        t = min(max(t, 0.0), self.duration)
        k = self.segment(t)
        tau = min(t - self.t0[k], self.T[k])
        x, xd = poly(self.coef[k], tau), poly(self.coef[k], tau, 1)
        Rm = (self.R_ref * Rotation.from_rotvec(x[3:])).as_matrix()
        w = Rm @ (right_jacobian(x[3:]) @ xd[3:])
        return x[:3], Rm, xd[:3], w

    def segment(self, t):
        return int(np.clip(np.searchsorted(self.t0, t, side="right") - 1, 0, len(self.T) - 1))

    def waypoint_times(self):
        return np.concatenate([self.t0, [self.duration]])


# =============================================================================================
# DLS IK controller (shared by the dry run and the live run)
# =============================================================================================
class DLSController:
    def __init__(self, kin, policy, kp=2.0, ko=2.0, v_lin_max=0.06, v_ang_max=0.6, qdot_max=0.25,
                 escape_vel=0.15, wait_mm=0.5, hold_mm=2.0, wait_deg=0.75, hold_deg=2.5):
        self.kin, self.policy, self.kp, self.ko = kin, policy, kp, ko
        self.v_lin_max, self.v_ang_max, self.qdot_max = v_lin_max, v_ang_max, qdot_max
        self.escape_vel = escape_vel
        self.wait_mm, self.hold_mm, self.wait_deg, self.hold_deg = wait_mm, hold_mm, wait_deg, hold_deg
        self.q_entry = None

    def lag_factor(self, ep_mm, eo_deg, w_ori):
        """Path speed allowed by the tracking error: 1 below wait_*, 0 (path waits) above hold_*.
        Where the orientation is deliberately relaxed (near a singularity) only position counts."""
        def ramp(e, lo, hi):
            return float(np.clip((hi - e) / max(hi - lo, 1e-9), 0.0, 1.0))
        f = ramp(ep_mm, self.wait_mm, self.hold_mm)
        if w_ori > 0.99:
            f = min(f, ramp(eo_deg, self.wait_deg, self.hold_deg))
        return f

    def reset(self, q):
        self.q_entry = np.array(q, float)

    def step(self, q, p_d, R_d, v_d, w_d, k_cap=1.0):
        """Returns qdot, k (path speed factor actually used), pos err (m), ori err (deg), sigma_min,
        orientation weight, FK position."""
        p, R = self.kin.fk(q)
        J = self.kin.jac(q)
        e_p = p_d - p
        e_o = rot_err(R_d, R)
        v_fb, w_fb = self.kp * e_p, self.ko * e_o
        nv, nw = np.linalg.norm(v_fb), np.linalg.norm(w_fb)
        if nv > self.v_lin_max:
            v_fb *= self.v_lin_max / nv
        if nw > self.v_ang_max:
            w_fb *= self.v_ang_max / nw
        smin = sigma_min(J)
        A, w_ori = self.policy.operator(J, smin)
        q_ff = A @ np.concatenate([v_d, w_d])
        q_fb = A @ np.concatenate([v_fb, w_fb])
        if smin >= self.policy.sigma_relax:
            self.q_entry = np.array(q, float)          # last configuration clear of singularities
        q_fb = q_fb + self.escape_vel * self.policy.escape(J, smin, q, self.q_entry)
        ep_mm, eo_deg = float(np.linalg.norm(e_p)) * 1000, float(np.degrees(np.linalg.norm(e_o)))
        # largest k <= k_cap with |k*q_ff + q_fb| <= qdot_max for every joint, and the path waits
        # for the arm when it falls behind
        m = self.qdot_max
        k = float(min(1.0, k_cap, self.lag_factor(ep_mm, eo_deg, w_ori)))
        if np.any(np.abs(q_fb) > m):
            k = 0.0
        else:
            for a, b in zip(q_ff, q_fb):
                if a > 1e-12:
                    k = min(k, (m - b) / a)
                elif a < -1e-12:
                    k = min(k, (-m - b) / a)
            k = max(0.0, k)
        qdot = k * q_ff + q_fb
        mx = float(np.max(np.abs(qdot)))
        if mx > m:
            qdot *= m / mx
        return qdot, k, float(np.linalg.norm(e_p)), float(np.degrees(np.linalg.norm(e_o))), smin, w_ori, p


class PathClock:
    """Path time that advances at k * real time; k may drop at once but recovers over K_RAMP s."""
    K_RAMP = 0.5

    def __init__(self):
        self.t, self.k = 0.0, 1.0

    def cap(self, dt):
        return min(1.0, self.k + dt / self.K_RAMP)

    def advance(self, k, dt):
        self.k = k
        self.t += k * dt


# =============================================================================================
# Checks
# =============================================================================================
def ik_pose(kin, q0, p_t, R_t, policy, iters=400, q_entry=None, escape_step=0.01):
    """Position-level weighted DLS IK from q0 (same singularity policy as the controller).
    Returns q, position error (m), angle error (deg)."""
    q = np.array(q0, float)
    for _ in range(iters):
        p, R = kin.fk(q)
        e = np.concatenate([p_t - p, rot_err(R_t, R)])
        J = kin.jac(q)
        smin = sigma_min(J)
        if np.linalg.norm(e[:3]) < 1e-5 and np.linalg.norm(e[3:]) < 1e-4 and smin >= policy.sigma_floor:
            break
        A, _ = policy.operator(J, smin)
        dq = A @ e + escape_step * policy.escape(J, smin, q, q_entry)
        m = np.max(np.abs(dq))
        q = q + (dq * 0.2 / m if m > 0.2 else dq)
    p, R = kin.fk(q)
    return q, float(np.linalg.norm(p_t - p)), float(np.degrees(np.linalg.norm(rot_err(R_t, R))))


def check_waypoints(kin, q_start, P, R, labels, policy, max_tilt_deg, verbose=True, substeps=8):
    """Check 1: IK to every waypoint (walked from the previous one in small steps, so it stays on
    the same arm configuration), verified with FK; joint limits; singularity. Returns (ok, q_list)."""
    ok, q, qs = True, np.array(q_start, float), []
    q_entry = q.copy()
    n_tilt, max_tilt, worst = 0, 0.0, (np.inf, 0)
    if verbose:
        print("\n  #  waypoint            x      y      z [cm]   FK err [mm / deg]  "
              "max joint move [deg]  limit margin [deg]  sigma_min")
    for i in range(len(P)):
        q_new = q
        if i:
            for k in range(1, substeps):
                f = k / substeps
                Rk = (R[i - 1] * Rotation.from_rotvec(f * (R[i - 1].inv() * R[i]).as_rotvec())).as_matrix()
                q_new = ik_pose(kin, q_new, (1 - f) * P[i - 1] + f * P[i], Rk, policy, iters=60,
                                q_entry=q_entry)[0]
        q_new, ep, eo = ik_pose(kin, q_new, P[i], R[i].as_matrix(), policy, q_entry=q_entry)
        jump = float(np.degrees(np.max(np.abs(q_new - q)))) if i else 0.0
        margin = float(np.degrees(kin.limit_margin(q_new)))
        smin = sigma_min(kin.jac(q_new))
        if smin >= policy.sigma_relax:
            q_entry = q_new.copy()
        if smin < worst[0]:
            worst = (smin, i)
        bad, note = [], ""
        if ep > 0.0005:
            bad.append("position not reachable")
        if eo > max_tilt_deg:
            bad.append(f"needs more than {max_tilt_deg:g} deg tilt")
        elif eo > 0.5:
            n_tilt += 1
            max_tilt = max(max_tilt, eo)
            note = f"~ tilted {eo:.1f} deg (singularity)"
        elif smin < policy.sigma_relax:
            note = "~ near singularity, ok"
        if margin < 3.0:
            bad.append("near joint limit")
        if i and jump > 90.0:
            bad.append("huge joint move for a small pose change - flip")
        ok &= not bad
        if verbose:
            tag = ("!! " + ", ".join(bad)) if bad else note
            print(f"{i:3d}  {labels[i]:14s} {P[i][0] * 100:6.2f} {P[i][1] * 100:6.2f} {P[i][2] * 100:6.2f}"
                  f"      {ep * 1000:6.3f} / {eo:5.2f}      {jump:8.2f}          {margin:8.1f}"
                  f"          {smin:6.3f}  {tag}")
        q = q_new
        qs.append(q_new)
    if verbose:
        if n_tilt:
            print(f"\n  {n_tilt} waypoints near a singularity: position exact, gripper tilted up to "
                  f"{max_tilt:.1f} deg there (limit {max_tilt_deg:g} deg, --max-tilt-deg).")
        s, i = worst
        if s < policy.sigma_relax:
            _, _, Vt = np.linalg.svd(kin.jac(qs[i]))
            v = Vt[-1] / np.max(np.abs(Vt[-1]))
            js = ", ".join(f"{kin.names[j]} {v[j]:+.2f}" for j in np.argsort(-np.abs(v))[:3])
            print(f"  closest to singular: {labels[i]} (sigma {s:.3f}), joints there [deg] "
                  f"{np.round(np.degrees(qs[i]), 1).tolist()}; singular joint motion: {js}")
    return ok, qs


def dry_run(kin, ctrl, traj, q_start, dt, labels, settle_s=2.0):
    """Check 2: simulate the whole run with the exact controller (perfect velocity tracking)."""
    q = np.array(q_start, float)
    ctrl.reset(q)
    clock = PathClock()
    T = traj.duration
    r = dict(max_mm=0.0, max_deg=0.0, at_mm=0.0, at_deg=0.0, min_margin=np.inf, min_sigma=np.inf,
             slowed=0, min_k=1.0, at_k=0.0, stalled=False)
    t_real, t_done = 0.0, None
    while True:
        p_d, R_d, v_d, w_d = traj.sample(clock.t)
        if clock.t >= T:
            v_d, w_d = np.zeros(3), np.zeros(3)
            t_done = t_done if t_done is not None else t_real
            if t_real - t_done > settle_s:
                break
        qdot, k, ep, eo, smin, _, _ = ctrl.step(q, p_d, R_d, v_d, w_d, clock.cap(dt))
        if clock.t < T and k < 0.999:
            r["slowed"] += 1
            if k < r["min_k"]:
                r["min_k"], r["at_k"] = k, clock.t
        if ep * 1000 > r["max_mm"]:
            r["max_mm"], r["at_mm"] = ep * 1000, clock.t
        if eo > r["max_deg"]:
            r["max_deg"], r["at_deg"] = eo, clock.t
        r["min_margin"] = min(r["min_margin"], kin.limit_margin(q))
        r["min_sigma"] = min(r["min_sigma"], smin)
        q = q + qdot * dt
        clock.advance(k, dt)
        t_real += dt
        if t_real > 3.0 * T + 30.0:
            r["stalled"] = True
            break
    p_d, R_d, _, _ = traj.sample(T)
    p, R = kin.fk(q)
    r.update(final_mm=np.linalg.norm(p_d - p) * 1000,
             final_deg=float(np.degrees(np.linalg.norm(rot_err(R_d, R)))),
             min_margin_deg=float(np.degrees(r["min_margin"])), real_s=t_real)
    where = lambda t: f"near {labels[min(traj.segment(t), len(labels) - 2)]}"
    r["where_mm"], r["where_deg"], r["where_k"] = where(r["at_mm"]), where(r["at_deg"]), where(r["at_k"])
    return r


# =============================================================================================
# Plan
# =============================================================================================
def load_waypoints(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    P = np.array([[float(r["x"]), float(r["y"]), float(r["z"])] for r in rows])
    Q = np.array([[float(r["qx"]), float(r["qy"]), float(r["qz"]), float(r["qw"])] for r in rows])
    return P, Rotation.from_quat(Q)


def build_plan(kin, q_now, P, R, a):
    """Prepend the approach when needed: by default a straight line from the current pose to the
    first waypoint (no lift, it stays clear of the wire setup). --approach-up-mm > 0 restores the
    old current -> above first waypoint -> first waypoint."""
    p_now, R_now = kin.fk(q_now)
    d0 = float(np.linalg.norm(P[0] - p_now))
    a0 = float(np.degrees((R[0].inv() * Rotation.from_matrix(R_now)).magnitude()))
    labels = [f"path {i}" for i in range(len(P))]
    stop = np.zeros(len(P), bool)
    if a.no_approach or (d0 < 0.003 and a0 < 3.0):
        return P, R, labels, stop, d0, a0
    if d0 > a.max_approach_m:
        raise SystemExit(f"ERROR: the arm is {d0 * 100:.1f} cm from the first waypoint (more than "
                         f"{a.max_approach_m * 100:.0f} cm). Bring it closer first (e.g. with the teleop).")
    if a.approach_up_mm <= 0:
        P2 = np.vstack([p_now, P])
        R2 = Rotation.from_quat(np.vstack([Rotation.from_matrix(R_now).as_quat(), R.as_quat()]))
        stop = np.concatenate([[True], stop])
        stop[1] = True                                      # stop on the first waypoint
        return P2, R2, ["current"] + labels, stop, d0, a0
    pre = P[0] + np.array([0.0, 0.0, a.approach_up_mm / 1000.0])
    P2 = np.vstack([p_now, pre, P])
    R2 = Rotation.from_quat(np.vstack([Rotation.from_matrix(R_now).as_quat(), R[0].as_quat(), R.as_quat()]))
    labels = ["current", "above start"] + labels
    stop = np.concatenate([[True, True], stop])
    stop[2] = True                                          # stop on the first waypoint
    return P2, R2, labels, stop, d0, a0


def analyze_log(path):
    with open(path) as f:
        rd = csv.reader(f)
        head = next(rd)
        rows = [[float(v) for v in r] for r in rd if r]
    names = [h[2:] for h in head if h.startswith("q_")]
    e = np.array([r[head.index("err_mm")] for r in rows])
    print(f"{path}: {len(rows)} cycles, {rows[-1][0]:.1f} s, off path max {e.max():.2f} mm, "
          f"RMS {np.sqrt(np.mean(e ** 2)):.2f} mm")
    return joint_tracking_report(rows, names)


def joint_tracking_report(log_rows, names, win=20, max_delay_s=1.5):
    """How each joint really followed the velocity commands: for each joint, the measured joint
    motion over short windows (win cycles) is fitted to the commanded motion shifted by a delay of
    0..max_delay_s. Prints the gain (1.0 = moves at the commanded speed), the delay and the fit
    quality (1.0 = the gain/delay model explains the motion perfectly). Returns {name: (g, d_s, fit)}."""
    L = np.array(log_rows, float)
    n = len(names)
    t, q, qc = L[:, 0], L[:, 12:12 + n], L[:, 12 + n:12 + 2 * n]
    dt = np.diff(t, prepend=t[0])
    step = float(np.median(dt[1:])) if len(dt) > 1 else 0.01
    max_delay = int(round(max_delay_s / max(step, 1e-3)))
    max_delay = min(max_delay, max(0, len(L) - win - 40))
    if len(L) < win + 40:
        return {}
    fresh = float(np.mean(np.any(np.abs(np.diff(q, axis=0)) > 1e-9, axis=1)))
    print(f"\njoint tracking (measured joint motion vs commanded), {len(L)} cycles of {step * 1000:.0f} ms;"
          f" joint angles changed in {fresh * 100:.0f}% of cycles:")
    out = {}
    for j in range(n):
        if np.sum(np.abs(qc[:, j] * dt)) < 0.01:
            print(f"  {names[j]:8s} barely commanded")
            continue
        cum = np.concatenate([[0.0], np.cumsum(qc[:, j] * dt)])     # integral of the command
        meas = q[max_delay + win:, j] - q[max_delay:-win, j]
        best = None
        for d in range(max_delay + 1):
            i0 = np.arange(max_delay, len(L) - win) - d
            cmd = cum[i0 + win] - cum[i0]
            den = float(cmd @ cmd)
            if den < 1e-12:
                continue
            g = float(cmd @ meas) / den
            res = float(np.sum((meas - g * cmd) ** 2))
            if best is None or res < best[0]:
                best = (res, g, d)
        if best is None:
            continue
        res, g, d = best
        fit = 1.0 - res / max(float(meas @ meas), 1e-15)
        d_s = d * step
        flag = ""
        if fit < 0.5:
            flag = "?? poor fit - this joint does not behave like 'delayed command'"
        elif g < 0:
            flag = "!! moves the WRONG WAY"
        elif g < 0.7 or g > 1.3:
            flag = "!! does not move at the commanded speed"
        elif d >= max_delay:
            flag = "!! delay at the end of the search range"
        elif d_s > 0.2:
            flag = "!! large delay"
        print(f"  {names[j]:8s} gain {g:5.2f}   delay {d_s * 1000:5.0f} ms   fit {fit:4.2f}  {flag}")
        out[names[j]] = (g, d_s, fit)
    return out


# =============================================================================================
# ROS node (live run)
# =============================================================================================
def run_live(a, P, R):
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Float64MultiArray, String

    class ReplayNode(Node):
        def __init__(self):
            super().__init__("heal_buzzwire_replay")
            self.js, self.js_time, self.urdf, self.pub = None, 0.0, None, None
            self.create_subscription(JointState, JOINT_STATES_TOPIC, self._js, 50)
            qos = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE,
                             durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                             history=QoSHistoryPolicy.KEEP_LAST, depth=1)
            self.create_subscription(String, ROBOT_DESCRIPTION_TOPIC, self._urdf, qos)

        def _js(self, msg):
            if set(CMD_JOINT_ORDER).issubset(msg.name):      # only the arm's full messages
                self.js = dict(zip(msg.name, msg.position))
                self.js_time = time.monotonic()

        def _urdf(self, msg):
            self.urdf = msg.data

        def q(self, names):
            return np.array([self.js[n] for n in names], float)

        def send(self, names, qdot):
            by_name = dict(zip(names, qdot))
            m = Float64MultiArray()
            m.data = [float(by_name.get(j, 0.0)) for j in CMD_JOINT_ORDER]
            self.pub.publish(m)

        def zero(self):
            if self.pub is None:
                return
            for _ in range(3):
                try:
                    m = Float64MultiArray()
                    m.data = [0.0] * len(CMD_JOINT_ORDER)
                    self.pub.publish(m)
                except Exception:
                    pass
                time.sleep(0.01)

    try:  # keep ROS alive on Ctrl+C so the zero velocity still goes out
        from rclpy.signals import SignalHandlerOptions
        rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    except (ImportError, TypeError):
        rclpy.init()
    node = ReplayNode()
    log_rows, result = [], None
    try:
        t0 = time.monotonic()
        while (node.js is None or node.urdf is None) and time.monotonic() - t0 < 10.0:
            rclpy.spin_once(node, timeout_sec=0.1)
        if node.urdf is None:
            print(f"ERROR: no URDF on {ROBOT_DESCRIPTION_TOPIC} - is the robot launch running?")
            return 1
        if node.js is None:
            print(f"ERROR: no full arm message on {JOINT_STATES_TOPIC}.")
            return 1
        kin = Kinematics(node.urdf, a.base_link, a.tip_link)
        q_now = node.q(kin.names)
        if a.save_urdf:
            with open(a.save_urdf, "w") as f:
                f.write(node.urdf)
            print(f"URDF saved: {os.path.abspath(a.save_urdf)}")
            print(f"current joints: --start-joints=\"{','.join(f'{v:.6f}' for v in q_now)}\"")
        plan = prepare(a, kin, q_now, P, R)
        if plan is None:
            return 1
        traj, ctrl = plan
        if a.dry_run:
            print("\n--dry-run: nothing moved.")
            return 0

        others = node.get_publishers_info_by_topic(a.cmd_topic)
        if others:
            print(f"ERROR: {sorted({o.node_name for o in others})} already publish(es) to {a.cmd_topic}. "
                  "Close it first. Not moving.")
            return 1
        node.pub = node.create_publisher(Float64MultiArray, a.cmd_topic, 10)
        input("\nPress Enter to start the replay (Ctrl+C to cancel) ")

        # ---------------- live loop
        dt = 1.0 / a.rate_hz
        for _ in range(5):
            rclpy.spin_once(node, timeout_sec=0.0)
        q0 = node.q(kin.names)
        p0, R0 = kin.fk(q0)
        p_s, R_s, _, _ = traj.sample(0.0)
        if np.linalg.norm(p0 - p_s) > 0.005:
            print("ERROR: the arm moved since the checks. Run again. Not moving.")
            return 1
        clock = PathClock()
        ctrl.reset(q0)
        comp = a.comp                       # delay / joint-gain compensation (from --calib-from)
        delay, jg = comp["delay"], np.array([comp["gain"].get(n, 1.0) for n in kin.names])
        sent = []                           # (time, intended qdot) still "in flight"
        hist_t, hist_path = [0.0], [0.0]    # path time history, to know where the arm should be now
        t_start = t_prev = time.monotonic()
        next_print, settled_since, t_fin, last_progress = 0.0, None, None, 0.0
        max_ep = max_eo = 0.0
        w_ori = 1.0
        result = "done"
        while True:
            t_loop = time.monotonic()
            rclpy.spin_once(node, timeout_sec=0.0)
            t = t_loop - t_start
            step_dt = min(t_loop - t_prev, 5 * dt) if t > 0 else dt
            t_prev = t_loop
            if t_loop - node.js_time > a.js_timeout:      # never steer on old joint angles
                result = f"ABORT: /joint_states stopped (no data for {t_loop - node.js_time:.2f} s)"
                break
            q = node.q(kin.names)
            # where the arm will be once the commands already sent have acted (Smith predictor)
            sent = [(ts, v) for ts, v in sent if ts > t - delay - 0.05]
            q_pred = q.copy()
            for idx, (ts, v) in enumerate(sent):
                t_next = sent[idx + 1][0] if idx + 1 < len(sent) else t
                q_pred += v * max(0.0, min(t_next, t) - max(ts, t - delay))
            # where the arm should be NOW (the path point from `delay` ago) -> real deviation
            path_then = float(np.interp(t - delay, hist_t, hist_path))
            p_then, R_then, _, _ = traj.sample(path_then)
            p, R = kin.fk(q)
            ep = float(np.linalg.norm(p_then - p))
            eo = float(np.degrees(np.linalg.norm(rot_err(R_then, R))))
            max_ep, max_eo = max(max_ep, ep), max(max_eo, eo)

            p_d, R_d, v_d, w_d = traj.sample(clock.t)
            finishing = clock.t >= traj.duration
            if finishing:
                v_d, w_d = np.zeros(3), np.zeros(3)
            k_cap = min(clock.cap(step_dt), ctrl.lag_factor(ep * 1000, eo, w_ori))
            qdot, k, _, _, smin, w_ori, _ = ctrl.step(q_pred, p_d, R_d, v_d, w_d, k_cap)
            # FK check 3: the measured pose must be on the path
            where = traj.labels[min(traj.segment(path_then) + 1, len(traj.labels) - 1)]
            if ep * 1000 > a.abort_mm or eo > a.abort_deg:
                result = (f"ABORT: off the path by {ep * 1000:.1f} mm / {eo:.1f} deg at t = {path_then:.1f} s "
                          f"(on the way to {where})")
                break
            if np.degrees(kin.limit_margin(q)) < 2.0:
                result = f"ABORT: joint limit within 2 deg at t = {path_then:.1f} s (on the way to {where})"
                break
            if not finishing:
                if k > 0.05:
                    last_progress = t
                elif t - last_progress > a.stall_s:
                    result = (f"ABORT: the arm is not catching up with the path ({ep * 1000:.1f} mm / "
                              f"{eo:.1f} deg off for {a.stall_s:g} s, on the way to {where})")
                    break
            q_cmd = qdot / jg                                   # slow joints get a bigger command
            mx = float(np.max(np.abs(q_cmd)))
            if mx > a.max_cmd_vel:
                q_cmd *= a.max_cmd_vel / mx
                qdot = q_cmd * jg
            node.send(kin.names, q_cmd)
            sent.append((t, qdot))
            clock.advance(k, step_dt)
            hist_t.append(t + 1e-9)
            hist_path.append(clock.t)
            log_rows.append([t, clock.t, k, *p_then, *p, ep * 1000, eo, smin, *q, *q_cmd])
            if t_loop >= next_print:
                next_print = t_loop + 0.5
                print(f"\r  {min(path_then, traj.duration):6.1f} / {traj.duration:.1f} s   "
                      f"off path {ep * 1000:5.2f} mm {eo:5.2f} deg   speed {k * 100:3.0f}%   ",
                      end="", flush=True)
            if finishing:
                t_fin = t_fin if t_fin is not None else t
                if path_then >= traj.duration and ep < 0.0005 and eo < max(0.5, a.max_tilt_deg):
                    settled_since = settled_since or t
                    if t - settled_since > 0.2:
                        break
                if t - t_fin > 3.0 + delay:
                    result = "done (final pose not fully settled)"
                    break
            time.sleep(max(0.0, dt - (time.monotonic() - t_loop)))
        node.zero()
        print(f"\n\n{result}")
        print(f"tracking (FK of measured joints vs trajectory): max {max_ep * 1000:.2f} mm, {max_eo:.2f} deg")
        if log_rows:
            e = np.array([r[9] for r in log_rows])
            print(f"                                               RMS {np.sqrt(np.mean(e ** 2)):.2f} mm")
            joint_tracking_report(log_rows, kin.names)
        return 0 if result.startswith("done") else 2
    except (KeyboardInterrupt, EOFError):
        print("\nCancelled.")
        result = "cancelled"
        return 130
    finally:
        node.zero()
        if log_rows:
            name = f"replay_log_{datetime.now():%Y%m%d_%H%M%S}.csv"
            with open(name, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["t", "path_t", "speed", "x_d", "y_d", "z_d", "x", "y", "z", "err_mm", "err_deg",
                            "sigma_min"] + [f"q_{n}" for n in CMD_JOINT_ORDER]
                           + [f"qdot_{n}" for n in CMD_JOINT_ORDER])
                w.writerows([[f"{v:.6f}" for v in r] for r in log_rows])
            print(f"log: {os.path.abspath(name)}  ({result})")
        time.sleep(0.05)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


# =============================================================================================
# Checks + plan printout (used live and offline)
# =============================================================================================
def prepare(a, kin, q_now, P, R):
    p_now, R_now = kin.fk(q_now)
    print(f"\nFK now     : [{p_now[0] * 100:+.2f}, {p_now[1] * 100:+.2f}, {p_now[2] * 100:+.2f}] cm")
    P2, R2, labels, stop, d0, a0 = build_plan(kin, q_now, P, R, a)
    print(f"path start : [{P[0][0] * 100:+.2f}, {P[0][1] * 100:+.2f}, {P[0][2] * 100:+.2f}] cm  "
          f"({d0 * 100:.1f} cm, {a0:.1f} deg away)")
    if len(P2) > len(P) and a.approach_up_mm > 0:
        print(f"approach   : current -> {a.approach_up_mm:.0f} mm above the start -> start")
    elif len(P2) > len(P):
        print(f"approach   : straight line to the start ({d0 * 1000:.0f} mm, no lift)")

    policy = SingularityPolicy(sigma_relax=a.sigma_relax, w_ori_min=a.w_ori_min)
    traj = PolyTrajectory(P2, R2, a.max_speed_mm_s / 1000.0, math.radians(a.max_rot_deg_s),
                          a.rot_weight, a.via_angle_deg, stop, order=a.order)
    traj.labels = labels
    ctrl = DLSController(kin, policy, kp=a.kp, ko=a.ko, qdot_max=a.max_joint_vel,
                         wait_mm=a.wait_mm, hold_mm=a.hold_mm, wait_deg=a.wait_deg, hold_deg=a.hold_deg)

    print("\nCHECK 1 - IK to every waypoint, verified with FK:")
    ok1, _ = check_waypoints(kin, q_now, P2, R2, labels, policy, a.max_tilt_deg)
    print("\nCHECK 2 - dry run of the whole trajectory with the DLS controller:")
    d = dry_run(kin, ctrl, traj, q_now, 1.0 / a.rate_hz, labels)
    deg_tol = max(a.dry_tol_deg, a.max_tilt_deg)
    ok2 = (d["max_mm"] <= a.dry_tol_mm and d["max_deg"] <= deg_tol and d["min_margin_deg"] > 3.0
           and not d["stalled"])
    print(f"  max position error {d['max_mm']:.2f} mm ({d['where_mm']}), max orientation error "
          f"{d['max_deg']:.2f} deg ({d['where_deg']})")
    print(f"  final {d['final_mm']:.3f} mm / {d['final_deg']:.3f} deg, closest joint limit "
          f"{d['min_margin_deg']:.1f} deg, smallest singular value {d['min_sigma']:.3f}")
    if d["slowed"]:
        print(f"  slowed down for joint speed in {d['slowed']} cycles, slowest {d['min_k'] * 100:.0f}% "
              f"({d['where_k']}); total about {d['real_s']:.0f} s (more on the real arm if it lags)")
    if d["stalled"]:
        print("  !! the path stalled (joint speed limit) - it would never finish")
    n_stops = int(traj.stop.sum())
    print(f"\nPLAN: {len(P2)} waypoints, {len(P2) - 1} {'quintic' if a.order == 5 else 'septic'} segments, "
          f"stops at {n_stops} (corners / turns), {traj.duration:.0f} s, peak {a.max_speed_mm_s:g} mm/s, "
          f"{a.max_rot_deg_s:g} deg/s")
    if not (ok1 and ok2):
        print("\nCHECKS FAILED (see !! above).", "Continuing because of --force." if a.force else
              "Not moving. Use --force only if you understand why.")
        if not a.force:
            return None
    else:
        print("All checks passed.")
    return traj, ctrl


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="?", help="waypoint CSV (x, y, z, qx, qy, qz, qw)")
    ap.add_argument("--analyze-log", metavar="LOG",
                    help="only print how each joint followed its commands in a replay_log_*.csv")
    ap.add_argument("--dry-run", action="store_true", help="run all checks, do not move")
    ap.add_argument("--save-urdf", metavar="FILE", help="save the robot's URDF and print the current joints")
    ap.add_argument("--offline", metavar="URDF", help="checks only, with this URDF file (no robot)")
    ap.add_argument("--start-joints", help='with --offline: start joint angles in rad, write --start-joints="j1,...,j6"')
    ap.add_argument("--order", type=int, choices=(5, 7), default=5, help="5 = quintic, 7 = septic")
    ap.add_argument("--max-speed-mm-s", type=float, default=20.0, help="peak tool speed")
    ap.add_argument("--max-rot-deg-s", type=float, default=20.0, help="peak tool rotation speed")
    ap.add_argument("--via-angle-deg", type=float, default=20.0, help="stop at bends sharper than this")
    ap.add_argument("--rot-weight", type=float, default=0.03, help="m per rad, for the bend test")
    ap.add_argument("--approach-up-mm", type=float, default=0.0,
                    help="0 = straight line to the start (default); >0 = lift this much above it first")
    ap.add_argument("--max-approach-m", type=float, default=0.40)
    ap.add_argument("--no-approach", action="store_true")
    ap.add_argument("--kp", type=float, default=2.0, help="position feedback gain (1/s)")
    ap.add_argument("--ko", type=float, default=2.0, help="orientation feedback gain (1/s)")
    ap.add_argument("--max-joint-vel", type=float, default=0.25,
                    help="rad/s; the path slows down where a joint would go faster")
    ap.add_argument("--wait-mm", type=float, default=0.5, help="path slows when the loop lags more than this")
    ap.add_argument("--hold-mm", type=float, default=2.0, help="path waits for the arm at this lag")
    ap.add_argument("--wait-deg", type=float, default=0.75)
    ap.add_argument("--hold-deg", type=float, default=2.5)
    ap.add_argument("--stall-s", type=float, default=5.0, help="abort if the path waits this long")
    ap.add_argument("--calib-from", metavar="LOG",
                    help="measure the arm's command delay and per-joint speed gain from a replay_log_*.csv "
                         "and compensate for them")
    ap.add_argument("--delay-ms", type=float, help="command delay to compensate (overrides --calib-from)")
    ap.add_argument("--max-cmd-vel", type=float, default=0.6,
                    help="rad/s, hard limit on the velocity actually sent to a joint")
    ap.add_argument("--max-tilt-deg", type=float, default=4.0,
                    help="how far the gripper may tilt from the taught orientation near a singularity")
    ap.add_argument("--sigma-relax", type=float, default=0.04,
                    help="below this sigma_min the orientation is relaxed (0 = never)")
    ap.add_argument("--w-ori-min", type=float, default=0.02, help="orientation weight at the singularity")
    ap.add_argument("--rate-hz", type=float, default=100.0)
    ap.add_argument("--js-timeout", type=float, default=0.1, help="s without /joint_states -> abort")
    ap.add_argument("--abort-mm", type=float, default=8.0, help="live: stop if this far off the path")
    ap.add_argument("--abort-deg", type=float, default=8.0)
    ap.add_argument("--dry-tol-mm", type=float, default=2.0)
    ap.add_argument("--dry-tol-deg", type=float, default=2.0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--cmd-topic", default=CMD_TOPIC)
    ap.add_argument("--base-link", default=BASE_LINK)
    ap.add_argument("--tip-link", default=TIP_LINK)
    a = ap.parse_args()

    if a.analyze_log:
        analyze_log(a.analyze_log)
        return 0
    a.comp = {"delay": 0.0, "gain": {}}
    if a.calib_from:
        rep = analyze_log(a.calib_from)
        good = {n: v for n, v in rep.items() if v[2] >= 0.7}
        if good:
            a.comp["delay"] = float(np.median([v[1] for v in good.values()]))
            a.comp["gain"] = {n: float(np.clip(v[0], 0.3, 1.5)) for n, v in good.items()}
    if a.delay_ms is not None:
        a.comp["delay"] = a.delay_ms / 1000.0
    if a.calib_from or a.delay_ms is not None:
        gs = ", ".join(f"{n} x{1 / g:.2f}" for n, g in a.comp["gain"].items() if abs(g - 1) > 0.05) or "none"
        print(f"\ncompensation: command delay {a.comp['delay'] * 1000:.0f} ms; joint commands scaled: {gs}\n")
    if not a.csv:
        ap.error("the waypoint CSV is required")
    P, R = load_waypoints(a.csv)
    print(f"{len(P)} waypoints from {a.csv}")
    if a.offline:
        kin = Kinematics(open(a.offline).read(), a.base_link, a.tip_link)
        policy = SingularityPolicy(sigma_relax=a.sigma_relax, w_ori_min=a.w_ori_min)
        q = (np.array([float(v) for v in a.start_joints.split(",")]) if a.start_joints
             else ik_pose(kin, np.zeros(kin.n), P[0], R[0].as_matrix(), policy)[0])
        return 0 if prepare(a, kin, q, P, R) is not None else 1
    return run_live(a, P, R)


if __name__ == "__main__":
    raise SystemExit(main())
