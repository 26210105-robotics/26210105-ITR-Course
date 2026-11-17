#!/usr/bin/env python3
"""
buzzwire_path_fit.py - clean, smooth and resample a taught buzz-wire path

Input : the waypoint CSV written by the teleop (pramth_tele_op.py):
        index,t_s,source,x,y,z,qx,qy,qz,qw,joint1..joint6,gripper   (pose of end-effector in base_link)

Output (written next to the input file):
  <name>_clean_waypoints.csv  compact, corrected waypoint set (key poses: corners, turns, rotations)
  <name>_smooth_path.csv      dense smooth path (about every 5 mm / 2 deg) with suggested timing
     columns: index, t_s, x, y, z, qx, qy, qz, qw   (end-effector pose in base_link, quaternion x,y,z,w
     like /end_effector_pose; t_s = time at SPEED_MM_S / ROT_SPEED_DEG_S). No joint angles: the
     corrections are in Cartesian space, so replay through IK (e.g. your DLS), not the taught joints.
  <name>_fit_3d.png           3D view, same viewpoint as the photo (from the front of the board)
  <name>_fit_profile.png      front view (x-z), raw vs cleaned vs smoothed
  <name>_fit_orientation.png  gripper twist and depth (y) along the path

What it does
  1. Stops: while the tool stays inside STOP_RADIUS, only the first and last waypoint of the
     stop are kept. The in-place rotation you made at a stop is kept; the back-and-forth
     adjusting (re-calibration) in between is dropped. Exact duplicates go the same way.
  2. Legs: the wire's up/down legs are vertical. Each leg (a run going up or down by more than
     3 cm) is straightened to the median x of your waypoints on it - your sideways drift while
     moving up/down is removed, the distance between neighbouring legs is kept.
  3. Backtracking: the wire is crossed once from one post to the other, so the position along
     the travel axis (x here) must never go back. Overshoot-and-return between the legs is
     removed with a weighted isotonic (monotone least-squares) fit; the straightened legs are
     held fixed, so narrow arches are never merged.
  4. Top arches: the height of each top arch (between an up leg and the next down leg) is the
     median of your waypoints there. Dips more than TOP_DIP below it (sagging while you rotated)
     are deleted, the arch is made flat at that height and the legs end exactly there (no
     overshoot above the arch).
  5. Noisy points deleted: after a first smoothing pass, waypoints that sit far from the curve
     (more than OUTLIER_MM, default 2.5 mm, in the wire plane) are deleted and the path is
     fitted again. Corners (leg ends) are never deleted.
  6. Smoothing spline (scipy UnivariateSpline) over path length, done PIECE BY PIECE: each leg
     and each stretch between two legs is smoothed on its own, with the corner points (leg ends)
     pinned - so smoothing never cuts a corner or lowers a top arch. Depth (y, the direction
     across the wire plane) is smoothed over the whole path (SIGMA_Y). Orientation (rotation vector
     relative to the first pose) is smoothed over the whole path. Rotation counts as path length
     too (ROT_WEIGHT m per rad), so in-place turns are part of the path.
  7. Rounded corners: every direction change (the top and bottom of each leg, and every other
     turn sharper than TURN_DEG) is replaced by a smooth curve
     (quadratic Bezier tangent to both sides) of radius CORNER_RADIUS. Where two corners are
     close (the narrow top arches) the radius shrinks to fit, so the arch becomes a round cap.
     A gripper turn made at a corner is spread along the curve instead of happening in place.
  8. Shape-preserving resampling: PCHIP for position, SLERP for orientation - no overshoot
     between waypoints, so the corners are not swung wider than you taught them.
  9. Key waypoints: Ramer-Douglas-Peucker simplification of the smooth path (KEY_TOL).

Run
  python3 buzzwire_path_fit.py buzzwire_waypoints_20261114_xxxxxx.csv
  python3 buzzwire_path_fit.py file.csv --sigma-mm 1.5 --spacing-mm 5 --monotonic none
Needs numpy, scipy, pandas, matplotlib (all in a normal ROS 2 Humble python).
"""

import argparse
import os

import math
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator, UnivariateSpline
from scipy.spatial.transform import Rotation, Slerp



# --------------------------------------------------------------------------- helpers
def quat_continuous(q):
    """Flip quaternion signs so neighbours are on the same side (q and -q are the same pose)."""
    q = q.copy()
    for i in range(1, len(q)):
        if np.dot(q[i], q[i - 1]) < 0:
            q[i] = -q[i]
    return q


def angle_between(r_a, r_b):
    return (r_a.inv() * r_b).magnitude()


def pava_decreasing(y, w=None):
    """Weighted isotonic regression: closest (weighted least squares) non-increasing sequence to y."""
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    blocks = []                                    # [mean, weight, count]
    for v, wi in zip(-np.asarray(y, float), w):    # non-decreasing on -y
        blocks.append([v, wi, 1])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            v2, w2, c2 = blocks.pop()
            v1, w1, c1 = blocks.pop()
            blocks.append([(v1 * w1 + v2 * w2) / (w1 + w2), w1 + w2, c1 + c2])
    out = np.concatenate([[v] * c for v, _, c in blocks])
    return -out


def find_legs(x, z, min_rise=0.03):
    """Runs of mostly-vertical motion in one z direction (up or down) that rise/fall >= min_rise.
    Returns a list of (first, last, 'up'|'down') point indices."""
    dz, dx = np.diff(z), np.diff(x)
    vert = np.abs(dz) > 1.5 * np.abs(dx)
    legs, i = [], 0
    while i < len(dz):
        if vert[i] and abs(dz[i]) > 0.002:
            sgn, j = np.sign(dz[i]), i
            while (j + 1 < len(dz) and (np.sign(dz[j + 1]) == sgn or abs(dz[j + 1]) < 0.002)
                   and (vert[j + 1] or abs(dx[j + 1]) < 0.003)):
                j += 1
            legs.append([i, j + 1, "up" if sgn > 0 else "down"])
            i = j + 1
        else:
            i += 1
    merged = []                                    # join pieces of one leg split by a tiny wiggle
    for leg in legs:
        if (merged and leg[2] == merged[-1][2] and leg[0] - merged[-1][1] <= 1
                and abs(np.median(x[leg[0]:leg[1] + 1])
                        - np.median(x[merged[-1][0]:merged[-1][1] + 1])) < 0.004):
            merged[-1][1] = leg[1]
        else:
            merged.append(leg)
    return [tuple(l) for l in merged if abs(z[l[1]] - z[l[0]]) >= min_rise]


def drop_rows(c, drop):
    """Delete rows (bool mask) and remap the leg indices stored in c.attrs['legs']."""
    keep = ~np.asarray(drop, bool)
    new_idx = np.cumsum(keep) - 1
    legs = [(int(new_idx[a]), int(new_idx[b]), k) for a, b, k in c.attrs.get("legs", [])]
    out = c[keep].reset_index(drop=True).copy()
    out.attrs["legs"] = legs
    return out


def flatten_tops(c, dip_m):
    """Flat, dip-free top arches between an up leg and the following down leg."""
    legs = c.attrs.get("legs", [])
    z = c["z"].to_numpy(float).copy()
    drop = np.zeros(len(c), bool)
    n_dips, report = 0, []
    for (a1, b1, k1), (a2, b2, k2) in zip(legs[:-1], legs[1:]):
        if not (k1 == "up" and k2 == "down"):
            continue
        sec = np.arange(b1, a2 + 1)                     # top corner of up leg .. top corner of down leg
        h = float(np.median(z[sec]))
        dips = [i for i in sec[1:-1] if z[i] < h - dip_m]
        drop[dips] = True
        n_dips += len(dips)
        rest = [i for i in sec if not drop[i]]
        h = float(np.median(z[rest]))
        z[rest] = h                                     # flat top
        up = np.arange(a1, b1)                          # up-leg points above the arch -> clip
        z[up] = np.minimum(z[up], h)
        dn = np.arange(a2 + 1, b2 + 1)
        z[dn] = np.minimum(z[dn], h)
        report.append(f"{h * 100:.1f}")
    c = c.copy()
    c["z"] = z
    c.attrs["legs"] = legs
    return drop_rows(c, drop), n_dips, report


def rdp(points, eps):
    """Ramer-Douglas-Peucker on an (N, D) polyline. Returns kept indices."""
    keep = np.zeros(len(points), bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(points) - 1)]
    while stack:
        a, b = stack.pop()
        if b <= a + 1:
            continue
        seg = points[b] - points[a]
        L2 = float(seg @ seg)
        rel = points[a + 1:b] - points[a]
        if L2 == 0:
            d = np.linalg.norm(rel, axis=1)
        else:
            t = np.clip(rel @ seg / L2, 0, 1)
            d = np.linalg.norm(rel - t[:, None] * seg, axis=1)
        k = int(np.argmax(d))
        if d[k] > eps:
            m = a + 1 + k
            keep[m] = True
            stack += [(a, m), (m, b)]
    return np.flatnonzero(keep)


def pose_param(P, rots, rot_weight):
    """Cumulative path parameter: translation (m) + rot_weight * rotation (rad)."""
    dp = np.linalg.norm(np.diff(P, axis=0), axis=1)
    da = np.array([angle_between(rots[i], rots[i + 1]) for i in range(len(rots) - 1)])
    return np.concatenate([[0.0], np.cumsum(np.sqrt(dp ** 2 + (rot_weight * da) ** 2))])


# --------------------------------------------------------------------------- pipeline
def clean(df, stop_radius, keep_rot_deg, monotonic, straighten_legs=True):
    log = []
    P = df[["x", "y", "z"]].to_numpy(float)
    Q = quat_continuous(df[["qx", "qy", "qz", "qw"]].to_numpy(float))
    rots = Rotation.from_quat(Q)
    n0 = len(df)

    # 1. stops -> first + last (+ any point that rotated > keep_rot_deg from the last kept one)
    keep = []
    i = 0
    while i < len(P):
        j = i
        while j + 1 < len(P) and np.linalg.norm(P[j + 1] - P[i]) < stop_radius:
            j += 1
        keep.append(i)
        if j > i:
            last = i
            for k in range(i + 1, j):
                if np.degrees(angle_between(rots[last], rots[k])) > keep_rot_deg:
                    keep.append(k)
                    last = k
            if np.degrees(angle_between(rots[i], rots[j])) > 1.0 or j == len(P) - 1:
                keep.append(j)
        i = j + 1
    keep = sorted(set(keep))
    # drop exact repeats left over (same pose)
    filt = [keep[0]]
    for k in keep[1:]:
        if (np.linalg.norm(P[k] - P[filt[-1]]) > 1e-4
                or np.degrees(angle_between(rots[filt[-1]], rots[k])) > 0.5):
            filt.append(k)
    keep = filt
    c = df.iloc[keep].reset_index(drop=True).copy()
    c[["qx", "qy", "qz", "qw"]] = quat_continuous(c[["qx", "qy", "qz", "qw"]].to_numpy(float))
    log.append(f"{'stops / re-adjusting collapsed':34s}: {n0} -> {len(c)} waypoints")

    # travel axis (the one the wire is crossed along)
    axis, sign = None, 0
    if monotonic != "none":
        if monotonic == "auto":
            span = {a: c[a].iloc[-1] - c[a].iloc[0] for a in ("x", "y")}
            axis = max(span, key=lambda a: abs(span[a]))
            sign = -1 if span[axis] < 0 else 1
        else:
            axis, sign = monotonic.lstrip("+-"), (-1 if monotonic.startswith("-") else 1)

    # 2. straighten the vertical legs (x along the travel axis = median of the leg)
    weight = np.ones(len(c))
    legs = []
    if straighten_legs and axis:
        v = c[axis].to_numpy(float).copy()
        legs = find_legs(v, c["z"].to_numpy(float))
        shift = 0.0
        for a_, b_, kind in legs:
            med = float(np.median(v[a_:b_ + 1]))
            shift = max(shift, float(np.max(np.abs(v[a_:b_ + 1] - med))))
            v[a_:b_ + 1] = med
            weight[a_:b_ + 1] = 1000.0               # hold legs fixed in the next step
        c[axis] = v
        log.append(f"{'vertical legs straightened':34s}: {len(legs)} legs "
                   f"({sum(k == 'down' for *_, k in legs)} down, {sum(k == 'up' for *_, k in legs)} up), "
                   f"max sideways correction {shift * 1000:.1f} mm")

    # 3. backtracking along the travel axis, legs held fixed
    if axis:
        v = c[axis].to_numpy(float).copy()
        fixed = pava_decreasing(v, weight) if sign < 0 else -pava_decreasing(-v, weight)
        moved = np.abs(fixed - v)
        c[axis] = fixed
        log.append(f"{'backtracking along ' + ('+' if sign > 0 else '-') + axis + ' removed':34s}: "
                   f"{int((moved > 1e-3).sum())} waypoints shifted, max {moved.max() * 1000:.1f} mm")
    c.attrs["legs"] = legs
    return c, log, (axis, sign)


def piecewise_smooth(s, P, legs, ai, sigma_m, sigma_y_m):
    """x, z smoothed piece by piece between corners (leg ends, pinned); y over the whole path."""
    n = len(P)
    corners = sorted({0, n - 1} | {a for a, _, _ in legs} | {b for _, b, _ in legs})
    leg_spans = {(a, b) for a, b, _ in legs}
    Ps = P.copy()
    for a, b in zip(corners[:-1], corners[1:]):
        idx = np.arange(a, b + 1)
        if len(idx) < 5:
            continue                                # too short to smooth: keep as taught
        w = np.ones(len(idx))
        w[0] = w[-1] = 100.0                        # pin the corners
        for k in (0, 2):
            if (a, b) in leg_spans and k == ai:
                continue                            # a straightened leg stays straight
            spl = UnivariateSpline(s[idx], P[idx, k], w=w, k=3, s=len(idx) * sigma_m ** 2)
            Ps[idx, k] = spl(s[idx])
        Ps[a, [0, 2]], Ps[b, [0, 2]] = P[a, [0, 2]], P[b, [0, 2]]
    Ps[:, 1] = UnivariateSpline(s, P[:, 1], k=3, s=n * sigma_y_m ** 2)(s)
    return Ps


def delete_outliers(c, travel, sigma_m, sigma_y_m, rot_weight, min_m):
    """First smoothing pass, then delete waypoints far from the curve (never corners)."""
    legs = c.attrs.get("legs", [])
    P = c[["x", "y", "z"]].to_numpy(float)
    rots = Rotation.from_quat(c[["qx", "qy", "qz", "qw"]].to_numpy(float))
    s = pose_param(P, rots, rot_weight)
    ai = "xyz".index(travel[0]) if travel[0] else -1
    Ps = piecewise_smooth(s, P, legs, ai, sigma_m, sigma_y_m)
    res = np.linalg.norm((Ps - P)[:, [0, 2]], axis=1)
    thr = min_m
    corners = {a for a, _, _ in legs} | {b for _, b, _ in legs} | {0, len(P) - 1}
    drop = np.array([res[i] > thr and i not in corners for i in range(len(P))])
    return drop_rows(c, drop), int(drop.sum()), thr


def _slerp2(Ra, Rb, f):
    return Slerp([0.0, 1.0], Rotation.from_quat(np.vstack([Ra.as_quat(), Rb.as_quat()])))([f])[0]


def round_corners(Pd, Rd, corner_pos, radius, step=0.001):
    """Replace each corner of the dense path by a quadratic Bezier curve tangent to both sides.
    Returns new (P, R) polyline (dense, ~step spacing on the curves) and the radii used."""
    cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(Pd, axis=0), axis=1))])
    ci = sorted({int(np.argmin(np.linalg.norm(Pd - c, axis=1))) for c in corner_pos})

    def point_at(dist, before):
        dist = float(np.clip(dist, 0.0, cum[-1]))
        if before:
            i = int(np.searchsorted(cum, dist, side="right") - 1)
            i = min(max(i, 0), len(cum) - 2)
            j = i + 1
        else:
            j = int(np.searchsorted(cum, dist, side="left"))
            j = min(max(j, 1), len(cum) - 1)
            i = j - 1
        L = cum[j] - cum[i]
        f = 0.0 if L <= 0 else (dist - cum[i]) / L
        return Pd[i] + f * (Pd[j] - Pd[i]), _slerp2(Rd[i], Rd[j], f), i, j

    outP, outQ, radii = [], [], []
    keep_from = 0
    for k, c in enumerate(ci):
        gap_prev = cum[c] - (cum[ci[k - 1]] if k else 0.0)
        gap_next = (cum[ci[k + 1]] if k + 1 < len(ci) else cum[-1]) - cum[c]
        rr = min(radius, 0.45 * gap_prev if k else 0.9 * gap_prev,
                 0.45 * gap_next if k + 1 < len(ci) else 0.9 * gap_next)
        if rr < 0.0005:
            continue
        E, RE, iE, _ = point_at(cum[c] - rr, True)
        X, RX, _, jX = point_at(cum[c] + rr, False)
        if iE < keep_from:
            continue
        for m in range(keep_from, iE + 1):
            outP.append(Pd[m]); outQ.append(Rd[m].as_quat())
        C = Pd[c]
        n_arc = max(8, int(np.ceil(2 * rr / step)))
        for u in np.linspace(0.0, 1.0, n_arc + 1):
            outP.append((1 - u) ** 2 * E + 2 * u * (1 - u) * C + u ** 2 * X)
            outQ.append(_slerp2(RE, RX, u).as_quat())
        keep_from = jX if cum[jX] > cum[c] + rr + 1e-9 else jX + 1
        radii.append(rr)
    for m in range(keep_from, len(Pd)):
        outP.append(Pd[m]); outQ.append(Rd[m].as_quat())
    P2 = np.array(outP)
    Q2 = np.array(outQ)
    for i in range(1, len(Q2)):                      # sign continuity
        if np.dot(Q2[i], Q2[i - 1]) < 0:
            Q2[i] = -Q2[i]
    return P2, Rotation.from_quat(Q2), radii


def densify_bends(P, keyi, max_bend_deg=15.0):
    """Add dense-path points to the key waypoints until the direction changes by at most
    max_bend_deg at every key waypoint, so each curve is described by enough points."""
    keyi = sorted(set(int(k) for k in keyi))
    for _ in range(50):
        added = False
        new = set(keyi)
        for n in range(1, len(keyi) - 1):
            a, b, c = keyi[n - 1], keyi[n], keyi[n + 1]
            u, w = P[b] - P[a], P[c] - P[b]
            nu, nw = np.linalg.norm(u), np.linalg.norm(w)
            if nu < 1e-6 or nw < 1e-6:
                continue
            if math.degrees(math.acos(np.clip(u @ w / (nu * nw), -1, 1))) > max_bend_deg:
                for lo, hi in ((a, b), (b, c)):
                    if hi - lo > 1:
                        new.add((lo + hi) // 2)
                        added = True
        keyi = sorted(new)
        if not added:
            break
    return np.array(keyi)


def find_turns(P, look=0.004, min_deg=12.0):
    """Every direction change on a dense curve: turning angle between the chord `look` behind and the
    chord `look` ahead; local maxima above min_deg (at most one per 2*look of path)."""
    cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    ib = np.searchsorted(cum, cum - look, side="left")
    ia = np.minimum(np.searchsorted(cum, cum + look, side="left"), len(P) - 1)
    u = P - P[ib]
    w = P[ia] - P
    nu, nw = np.linalg.norm(u, axis=1), np.linalg.norm(w, axis=1)
    ok = (nu > 0.5 * look) & (nw > 0.5 * look)
    cosang = np.einsum("ij,ij->i", u, w) / np.maximum(nu * nw, 1e-12)
    ang = np.where(ok, np.degrees(np.arccos(np.clip(cosang, -1, 1))), 0.0)
    turns = []
    for i in np.argsort(-ang):
        if ang[i] < min_deg:
            break
        if all(abs(cum[i] - cum[j]) > 2 * look for j in turns):
            turns.append(int(i))
    return sorted(turns), ang


def resample_uniform(P, R, spacing, rot_weight):
    """Equal steps along a dense polyline (position + rot_weight * rotation)."""
    keep = np.concatenate([[True], (np.linalg.norm(np.diff(P, axis=0), axis=1) > 1e-7)
                           | (np.abs(np.sum(R[1:].as_quat() * R[:-1].as_quat(), axis=1)) < 1 - 1e-12)])
    P, R = P[keep], R[np.flatnonzero(keep)]
    s = pose_param(P, R, rot_weight)
    n = max(2, int(np.ceil(s[-1] / spacing)) + 1)
    st = np.linspace(0.0, s[-1], n)
    Pn = np.column_stack([np.interp(st, s, P[:, k]) for k in range(3)])
    Rn = Slerp(s, R)(st)
    return Pn, Rn


def smooth_and_resample(c, sigma_m, sigma_rot_deg, rot_weight, spacing_m, key_tol_m,
                        v_lin, v_rot, travel, sigma_y_m=0.003, corner_radius_m=0.008, turn_deg=12.0,
                        max_bend_deg=15.0):
    P = c[["x", "y", "z"]].to_numpy(float)
    rots = Rotation.from_quat(c[["qx", "qy", "qz", "qw"]].to_numpy(float))
    s = pose_param(P, rots, rot_weight)
    n = len(s)

    # 3. smoothing splines over path length (s = n*sigma^2 -> residual RMS ~ sigma)
    def sspline(y, sig):
        return UnivariateSpline(s, y, k=3, s=n * sig ** 2)(s)

    # position: piece by piece between corners (leg ends), corners pinned
    legs = c.attrs.get("legs", [])
    axis, sign = travel
    Ps = piecewise_smooth(s, P, legs, "xyz".index(axis) if axis else -1, sigma_m, sigma_y_m)
    R0 = rots[0]
    rv = (R0.inv() * rots).as_rotvec()              # rotation relative to the first pose (tool frame)
    rvs = np.column_stack([sspline(rv[:, i], np.radians(sigma_rot_deg)) for i in range(3)])
    Rs = R0 * Rotation.from_rotvec(rvs)
    if axis:                                        # keep the travel axis monotone after smoothing
        a = "xyz".index(axis)
        Ps[:, a] = (np.minimum.accumulate if sign < 0 else np.maximum.accumulate)(Ps[:, a])

    corner_idx = sorted({a for a, _, _ in legs} | {b for _, b, _ in legs})
    corner_pos = Ps[corner_idx] if corner_idx else np.zeros((0, 3))

    # 4. shape-preserving dense resampling
    s_s = pose_param(Ps, Rs, rot_weight)            # parameter of the smoothed waypoints
    good = np.concatenate([[True], np.diff(s_s) > 1e-9])
    s_s, Ps, Rs = s_s[good], Ps[good], Rs[np.flatnonzero(good)]
    pchip, slerp = PchipInterpolator(s_s, Ps, axis=0), Slerp(s_s, Rs)
    # equal spacing measured along the actual curve (not the straight chords between waypoints)
    fine = np.linspace(0, s_s[-1], 20 * (len(s_s) + int(s_s[-1] / spacing_m)))
    arc = pose_param(pchip(fine), slerp(fine), rot_weight)
    n_out = max(2, int(np.ceil(arc[-1] / spacing_m)) + 1)
    sd = np.interp(np.linspace(0, arc[-1], n_out), arc, fine)
    Pd = pchip(sd)
    Rd = slerp(sd)

    # rounded corners, then equal spacing again
    radii, n_turns = [], 0
    if corner_radius_m > 0:
        fine_n = max(len(Pd) * 10, 2000)
        sf = np.linspace(0, s_s[-1], fine_n)
        Pf0, Rf0 = pchip(sf), slerp(sf)
        # every other direction change on the curve (not only the leg ends) is rounded as well
        turns, _ = find_turns(Pf0, look=corner_radius_m / 2, min_deg=turn_deg)
        extra = [Pf0[i] for i in turns
                 if not len(corner_pos) or np.min(np.linalg.norm(corner_pos - Pf0[i], axis=1)) > corner_radius_m]
        n_turns = len(extra)
        allc = np.vstack([corner_pos] + ([np.array(extra)] if extra else [])) if (len(corner_pos) or extra) \
            else np.zeros((0, 3))
        if len(allc):
            Pf, Rf, radii = round_corners(Pf0, Rf0, allc, corner_radius_m)
            Pd, Rd = resample_uniform(Pf, Rf, spacing_m, rot_weight)

    # suggested timing: constant tool speed, slower of translation / rotation per step
    dp = np.linalg.norm(np.diff(Pd, axis=0), axis=1)
    da = np.array([angle_between(Rd[i], Rd[i + 1]) for i in range(len(Rd) - 1)])
    t = np.concatenate([[0.0], np.cumsum(np.maximum(dp / v_lin, da / v_rot))])

    # 5. key waypoints (RDP in position + weighted rotation space)
    rvd = (R0.inv() * Rd).as_rotvec()
    keyi = rdp(np.column_stack([Pd, rot_weight * rvd]), key_tol_m)
    keyi = densify_bends(Pd, keyi, max_bend_deg)
    sd = pose_param(Pd, Rd, rot_weight)
    return dict(s=s, P=P, rots=rots, Ps=Ps, Rs=Rs, s_s=s_s, sd=sd, Pd=Pd, Rd=Rd, t=t,
                keyi=keyi, R0=R0, rv=rv, radii=radii, n_turns=n_turns)


def table(P, R, t):
    q = quat_continuous(R.as_quat())
    d = pd.DataFrame({"index": np.arange(len(P)), "t_s": np.round(t, 3)})
    for i, a in enumerate("xyz"):
        d[a] = np.round(P[:, i], 6)
    for i, a in enumerate(("qx", "qy", "qz", "qw")):
        d[a] = np.round(q[:, i], 6)
    return d


# --------------------------------------------------------------------------- plots
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
C_SMOOTH, C_KEY, C_RAW = "#2a78d6", "#eb6834", "#a3a29c"


def style(ax):
    ax.set_facecolor(SURF)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color=GRID, lw=0.8)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)


def plot_all(raw, c, r, out_prefix, travel):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = 100.0
    Praw = raw[["x", "y", "z"]].to_numpy(float) * cm
    Pd, Pk = r["Pd"] * cm, r["Pd"][r["keyi"]] * cm
    files = []

    # --- 3D, seen from the front of the board (camera side), like the photo
    fig = plt.figure(figsize=(12, 7), facecolor=SURF)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(SURF)
    floor = min(Praw[:, 2].min(), Pd[:, 2].min()) - 2.0
    ax.plot(Pd[:, 0], Pd[:, 1], np.full(len(Pd), floor), color=GRID, lw=2.0)        # shadow on the table
    ax.plot(Praw[:, 0], Praw[:, 1], Praw[:, 2], color=C_RAW, lw=1.0, label="taught (raw)")
    ax.scatter(Praw[:, 0], Praw[:, 1], Praw[:, 2], s=6, color=C_RAW, depthshade=False)
    ax.plot(Pd[:, 0], Pd[:, 1], Pd[:, 2], color=C_SMOOTH, lw=2.4, label="smoothed path")
    ax.scatter(Pk[:, 0], Pk[:, 1], Pk[:, 2], s=36, color=C_KEY, edgecolor=SURF, linewidth=1.2,
               depthshade=False, label=f"key waypoints ({len(Pk)})")
    for p_, txt in ((Pd[0], "start"), (Pd[-1], "end")):
        ax.scatter(*p_, s=80, facecolor="none", edgecolor=INK, linewidth=1.5, depthshade=False)
        ax.text(p_[0], p_[1], p_[2] + 1.0, txt, color=INK, fontsize=10)
    rng = np.ptp(np.vstack([Praw, Pd]), axis=0)
    ax.set_box_aspect((rng[0], max(rng[1], 4.0) * 2.0, rng[2] + 2.0))   # depth drawn x2 to be readable
    ax.view_init(elev=22, azim=112)          # from the front of the board, a little from the side
    ax.set_xlabel("x (cm)", color=INK2, labelpad=10)
    ax.set_ylabel("y (cm)\n(depth, x2)", color=INK2, labelpad=12)
    ax.set_zlabel("z (cm)", color=INK2, labelpad=6)
    ax.tick_params(colors=INK2, labelsize=8)
    from matplotlib.ticker import MaxNLocator
    ax.yaxis.set_major_locator(MaxNLocator(3))
    for a_ in (ax.xaxis, ax.yaxis, ax.zaxis):
        a_.pane.set_facecolor(SURF)
        a_.pane.set_edgecolor(GRID)
        a_._axinfo["grid"]["color"] = GRID
    ax.set_title("Buzz-wire path in 3D: taught vs smoothed (end-effector, base_link frame)",
                 color=INK, fontsize=12, loc="left")
    ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.92), frameon=False, labelcolor=INK2, fontsize=9)
    f = out_prefix + "_fit_3d.png"
    fig.savefig(f, dpi=150, bbox_inches="tight")
    plt.close(fig)
    files.append(f)

    # --- front view x-z (the wire's shape)
    fig, ax = plt.subplots(figsize=(12, 5), facecolor=SURF)
    style(ax)
    ax.plot(Praw[:, 0], Praw[:, 2], color=C_RAW, lw=1.0, marker="o", ms=3, label="taught (raw, all 157)")
    ax.plot(Pd[:, 0], Pd[:, 2], color=C_SMOOTH, lw=2.2, label="smoothed path")
    ax.scatter(Pk[:, 0], Pk[:, 2], s=40, color=C_KEY, edgecolor=SURF, linewidth=1.2, zorder=5,
               label=f"key waypoints ({len(Pk)})")
    ax.annotate("start", Pd[0, [0, 2]], xytext=(6, 6), textcoords="offset points", color=INK, fontsize=9)
    ax.annotate("end", Pd[-1, [0, 2]], xytext=(6, 6), textcoords="offset points", color=INK, fontsize=9)
    ax.set_aspect("equal")
    ax.invert_xaxis()                        # as seen from the front of the board (photo view)
    ax.set_xlabel("x (cm)  - axis reversed to match the camera view")
    ax.set_ylabel("z (cm)")
    ax.set_title("Front view: the wire's shape", color=INK, fontsize=12, loc="left")
    ax.legend(loc="lower left", frameon=False, labelcolor=INK2, fontsize=9, ncol=3)
    f = out_prefix + "_fit_profile.png"
    fig.savefig(f, dpi=150, bbox_inches="tight")
    plt.close(fig)
    files.append(f)

    # --- orientation (twist about the tool axis) and depth along the path
    R0 = r["R0"]
    tw_c = np.degrees(r["rv"][:, 2])
    tw_d = np.degrees((R0.inv() * r["Rd"]).as_rotvec()[:, 2])
    s_c = r["s"] / r["s"][-1] * 100
    s_d = r["sd"] / r["sd"][-1] * 100
    fig, axs = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True, facecolor=SURF)
    for a in axs:
        style(a)
    axs[0].plot(s_c, tw_c, color=C_RAW, marker="o", ms=3, lw=1.0, label="cleaned waypoints")
    axs[0].plot(s_d, tw_d, color=C_SMOOTH, lw=2.2, label="smoothed")
    axs[0].set_ylabel("gripper twist (deg)")
    axs[0].set_title("Gripper rotation about its own axis, relative to the start", color=INK,
                     fontsize=12, loc="left")
    axs[0].legend(loc="upper right", frameon=False, labelcolor=INK2, fontsize=9)
    axs[1].plot(s_c, c["y"].to_numpy() * cm, color=C_RAW, marker="o", ms=3, lw=1.0, label="cleaned waypoints")
    axs[1].plot(s_d, Pd[:, 1], color=C_SMOOTH, lw=2.2, label="smoothed")
    axs[1].set_ylabel("depth y (cm)")
    axs[1].set_xlabel("progress along the path (%)")
    axs[1].set_title("Depth y: the wire plane is slightly tilted toward the robot", color=INK,
                     fontsize=12, loc="left")
    axs[1].legend(loc="upper right", frameon=False, labelcolor=INK2, fontsize=9)
    fig.tight_layout()
    f = out_prefix + "_fit_orientation.png"
    fig.savefig(f, dpi=150, bbox_inches="tight")
    plt.close(fig)
    files.append(f)
    return files


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", help="waypoint CSV from the teleop")
    ap.add_argument("--stop-radius-mm", type=float, default=3.0, help="a 'stop' = tool stays inside this")
    ap.add_argument("--sigma-mm", type=float, default=1.5, help="position smoothing (expected noise)")
    ap.add_argument("--sigma-y-mm", type=float, default=3.0, help="depth (y) smoothing, whole path")
    ap.add_argument("--sigma-rot-deg", type=float, default=2.0, help="orientation smoothing")
    ap.add_argument("--rot-weight", type=float, default=0.03, help="m of path per rad of rotation")
    ap.add_argument("--spacing-mm", type=float, default=2.0, help="dense path spacing")
    ap.add_argument("--max-bend-deg", type=float, default=15.0,
                    help="key waypoints: at most this direction change at each (more points on curves)")
    ap.add_argument("--key-tol-mm", type=float, default=1.0, help="key waypoint tolerance")
    ap.add_argument("--corner-radius-mm", type=float, default=8.0, help="rounding of the corners (0 = sharp)")
    ap.add_argument("--turn-deg", type=float, default=12.0,
                    help="any direction change sharper than this (over one corner radius) is rounded too")
    ap.add_argument("--speed-mm-s", type=float, default=10.0, help="suggested replay tool speed")
    ap.add_argument("--rot-speed-deg-s", type=float, default=10.0, help="suggested replay turn speed")
    ap.add_argument("--monotonic", default="auto",
                    help="travel axis that must never go back: auto, -x, +x, -y, +y or none")
    ap.add_argument("--top-dip-mm", type=float, default=3.0, help="delete top-arch points this far below it")
    ap.add_argument("--outlier-mm", type=float, default=2.5, help="min distance off the curve to delete")
    ap.add_argument("--no-straighten-legs", action="store_true", help="keep the legs as taught")
    ap.add_argument("--no-plots", action="store_true")
    a = ap.parse_args()

    raw = pd.read_csv(a.csv)
    c, log, travel = clean(raw, a.stop_radius_mm / 1000, 60.0, a.monotonic, not a.no_straighten_legs)
    if c.attrs.get("legs"):
        c, n_dips, heights = flatten_tops(c, a.top_dip_mm / 1000)
        log.append(f"{'top arches made flat':34s}: {len(heights)} arches at z = {', '.join(heights)} cm, "
                   f"{n_dips} dip waypoints deleted")
    n_before = len(c)
    c, n_out, thr = delete_outliers(c, travel, a.sigma_mm / 1000, a.sigma_y_mm / 1000, a.rot_weight,
                                    a.outlier_mm / 1000)
    log.append(f"{'noisy waypoints deleted':34s}: {n_out} (more than {thr * 1000:.1f} mm off the curve), "
               f"{n_before} -> {len(c)} waypoints")
    r = smooth_and_resample(c, a.sigma_mm / 1000, a.sigma_rot_deg, a.rot_weight, a.spacing_mm / 1000,
                            a.key_tol_mm / 1000, a.speed_mm_s / 1000, np.radians(a.rot_speed_deg_s), travel,
                            a.sigma_y_mm / 1000, a.corner_radius_mm / 1000, a.turn_deg,
                            a.max_bend_deg)

    base = os.path.splitext(a.csv)[0]
    dense = table(r["Pd"], r["Rd"], r["t"])
    key = dense.iloc[r["keyi"]].copy()
    key["index"] = np.arange(len(key))
    dense.to_csv(base + "_smooth_path.csv", index=False)
    key.to_csv(base + "_clean_waypoints.csv", index=False)

    # checks
    D = (r["Ps"] - c[["x", "y", "z"]].to_numpy(float)) * 1000
    dev = np.linalg.norm(D[:, [0, 2]], axis=1)      # in the wire plane (x-z)
    devy = np.abs(D[:, 1])                          # depth
    Pd = r["Pd"]
    seg = np.linalg.norm(np.diff(Pd, axis=0), axis=1)
    back = 0
    if travel[0]:
        ai = "xyz".index(travel[0])
        back = int((np.diff(Pd[:, ai]) * travel[1] < -1e-4).sum())
    tot_rot = sum(angle_between(r["Rd"][i], r["Rd"][i + 1]) for i in range(len(Pd) - 1))
    print("\n".join(log))
    print(f"{'smoothing, in the wire plane (x-z)':34s}: RMS {np.sqrt((dev ** 2).mean()):.2f} mm, "
          f"max {dev.max():.2f} mm from the cleaned waypoints")
    print(f"{'smoothing, depth (y)':34s}: RMS {np.sqrt((devy ** 2).mean()):.2f} mm, max {devy.max():.2f} mm")
    print(f"{'smooth path':34s}: {len(Pd)} poses, length {seg.sum() * 100:.1f} cm, "
          f"total rotation {np.degrees(tot_rot):.0f} deg, backtracking steps {back}")
    if r["radii"]:
        rr = np.array(r["radii"]) * 1000
        print(f"{'direction changes rounded':34s}: {len(rr)} ({len(rr) - r['n_turns']} leg corners + "
              f"{r['n_turns']} other turns), radius {rr.min():.1f}-{rr.max():.1f} mm "
              f"(smaller only where turns are close)")
    Pk = r["Pd"][r["keyi"]]
    uk = np.diff(Pk, axis=0)
    uk /= np.maximum(np.linalg.norm(uk, axis=1), 1e-12)[:, None]
    bend = np.degrees(np.arccos(np.clip((uk[:-1] * uk[1:]).sum(1), -1, 1)))
    print(f"{'key waypoints':34s}: {len(key)} (within {a.key_tol_mm} mm of the smooth path, "
          f"largest direction change at a waypoint {bend.max():.0f} deg)")
    print(f"{'suggested replay time':34s}: {r['t'][-1]:.0f} s at {a.speed_mm_s:g} mm/s, "
          f"{a.rot_speed_deg_s:g} deg/s")
    print(f"wrote {base}_clean_waypoints.csv\nwrote {base}_smooth_path.csv")
    if not a.no_plots:
        for f in plot_all(raw, c, r, base, travel):
            print("wrote", f)


if __name__ == "__main__":
    main()
