"""Peg sorting detector v7.3

New in v7.3 (shape detection, slots and hand ignoring are unchanged from v7.2):
  * Depth for the robot. At 'b' the tray plane is fitted from depth and a metric TRAY FRAME is built (x along the
    tray's long edge, y across, z up, mm). Every peg gets its real top height, the position of its axis on the tray
    plane (parallax-corrected: the top face is above the tray, so its image position is shifted on a tilted camera),
    its rotation, and ready-to-use gripper targets: pick, pick-approach, place, place-approach.
  * Pick / carry / place. Each peg is remembered by its colour (one peg per colour). Its shape and target slot are kept
    while the gripper covers it, lifts it and carries it; states: on_table -> occluded -> carried -> placed.
    The terminal prints each state change, and whether the placed peg is aligned.
  * ROS 2. JSON state + next pick / place poses + TF camera -> tray (see ROS section in the settings).

v7.2 notes:

Changes from v7.1:
  * Hexagon slot fixed. The real hexagon hole (rounded laser-cut corners) overlaps an ideal circle almost as well
    as an ideal hexagon, and the old circle/hexagon tie-break then called it 'circle'; the real circle hole won the
    'one slot per shape' contest and the hexagon slot was dropped. Now: (1) the tie is decided by the 6-fold ripple
    of the outline radius (a hexagon has it, a circle does not; same 4-fold test for a rounded square peg), (2) holes are assigned to the expected shapes
    one-to-one, so two holes can no longer compete for the same name.
  * Human-intervention warnings / pause removed. Hands are ignored silently instead: skin-coloured raised blobs
    entering from the image edge are masked out of peg detection, raised pixels can no longer be taken for tray,
    and the tray pose is simply kept while an arm or hand is over the tray.
  * Pegs: squash correction for every shape (an oblique circle top face is no longer a 'rectangle'), ambiguous fits
    do not vote, the shape label has hysteresis (no square/unknown flicker), labels are readable and do not overlap.
"""
import itertools
import json
import time
from collections import Counter, deque

import cv2
import numpy as np
import pyrealsense2 as rs

# ======================= Settings (tune in your lab) =======================
COLORS = {
    "orange": [((8, 170, 120), (20, 255, 255))],
    "yellow": [((21, 120, 120), (35, 255, 255))],
    "green":  [((36, 60, 60), (65, 255, 255))],   # was up to H=85: the teal mat (H~81) was inside this range
    "blue":   [((90, 100, 50), (130, 255, 255))],
    "red":    [((0, 150, 70), (7, 255, 255)), ((170, 150, 70), (179, 255, 255))],
}

# MDF tray colour (tan, low saturation). H capped at 22 so yellow pegs are not tray.
TRAY_LO = np.array((5, 30, 50))
TRAY_HI = np.array((22, 140, 255))
MIN_TRAY_AREA = 15000          # camera px
EDGE_MARGIN = 5                # tray corners closer than this to the image edge = clipped

# Rectified (top-down) tray size in px. Keep the real tray's aspect ratio (~3.3 : 1).
TRAY_W, TRAY_H = 660, 200
SLOT_MIN_AREA, SLOT_MAX_AREA = 1200, 12000   # in rectified px
SLOT_SPLIT_PX = 17              # a hole merged with a thin strip (mat along the tray edge) is cut apart with this
                                # opening; strips thinner than this go, a ~95 px hole only gets its corners rounded
ROW_TOL = 60                                  # rectified px

# Pegs
MIN_AREA, MAX_AREA, MIN_SOLIDITY = 800, 40000, 0.85
MIN_H, MAX_H = 10, 220          # mm above calibration surface. MAX_H must be ABOVE your tallest peg:
                                # a taller peg loses its top face to the 'hand / arm' gate (was 130)
TOP_FACE_TOL = 6                # mm: pixels within this of the peg's highest point = TOP FACE
# A tall peg seen obliquely shows mostly side wall, so its top face can be a small part of the colour blob.
# Accept it down to TOP_FACE_MIN_RATIO of the blob if it is also big, compact and solid (not a wall strip).
TOP_FACE_MIN_RATIO = 0.08
TOP_FACE_MIN_PX = 400
TOP_FACE_MIN_ASPECT = 0.30
TOP_FACE_MIN_SOLIDITY = 0.80
# NOTE: every peg measurement (shape, centre, xyz) refers to the TOP FACE only, because the
# robot grasps and aligns from the top view. Side walls / body volume are never used for the centre.

# Real tray size in mm (MEASURE YOURS) -> converts rectified px offsets to mm
TRAY_MM_W, TRAY_MM_H = 330.0, 100.0
MM_PER_PX_X, MM_PER_PX_Y = TRAY_MM_W / TRAY_W, TRAY_MM_H / TRAY_H
ALIGN_TOL_MM = 3.0              # peg centroid within this of slot centroid = ALIGNED

ALIGN_TOL_DEG = 5.0             # peg rotation within this of the slot rotation (mod shape symmetry)

# The four holes of YOUR tray. Exactly one slot per name is kept; spurious ones are dropped.
EXPECTED_SLOTS = ("triangle", "square", "hexagon", "circle")

# Template fitting (ideal circle / triangle / hexagon / rectangle fitted to an outline)
FIT_RES = 64.0                  # outlines are resampled to ~this size so every shape gets the same precision
                                # (64 px: ~1.5% of the size per pixel; raise to 100 for more precision, ~2.5x slower)
FIT_MIN_IOU = 0.80              # overlap with the best ideal model below this = 'unknown'
FIT_EVERY = 2                   # run the template fit on every Nth frame (~3.4 ms per peg); in between the
                                # last fit is reused. Set 1 for every frame if your loop has time to spare.
FIT_MIN_ASPECT = 0.55           # outlines squashed down to this (minor/major axis of the 2nd moment) are un-squashed
                                # before fitting: circle/triangle/square/hexagon are all 'round' in that sense, so any
                                # squash is perspective (oblique top face, tray-plane parallax). Below = really elongated.
                                # Trade-off: a peg top face that is really a 0.55-0.7 rectangle now reads 'square'
                                # (you have no rectangular pegs). Set 0.75 for the v7.1 behaviour.
SLOT_MIN_ASPECT = 0.70          # same, for tray holes (they lie on the tray plane, so only mild squash is expected)
HEX_CIRCLE_TIE = 0.04           # circle and hexagon overlaps closer than this -> decided by the 6-fold radius ripple
HEX_RIPPLE_MIN = 0.024          # relative 6th harmonic of the radius. Measured on your recording: hexagon hole 0.035,
                                # circle hole 0.015. Sharp hexagon ~0.055, noisy circle < 0.013.
# Same idea for square and triangle vs circle. Soft depth edges round the corners of a peg's TOP FACE (the 5x5
# median on the height map + the mask morphology), so a square top face can overlap a circle better than a square
# (your blue peg: circle 0.90 vs square 0.86 -> 'circle'). Its radius still rises and falls 4 times per turn.
# Rounded squares measured 0.063-0.089 (sharp 0.12-0.14); circles <= 0.022 even ragged / with chunks missing.
POLY_CIRCLE_TIE = {"triangle": 0.08, "square": 0.08, "hexagon": HEX_CIRCLE_TIE}     # overlap window for the test
POLY_RIPPLE_MIN = {"triangle": 0.06, "square": 0.04, "hexagon": HEX_RIPPLE_MIN}     # n-fold ripple threshold
POLY_SIDES = {"triangle": 3, "square": 4, "hexagon": 6}
SQUARE_DOMINANCE = 1.3          # square only: the 4-fold ripple must also be this many times stronger than the 1..3-fold
                                # ripple. A circle top face with one side cut flat (missing depth, seen on your green peg)
                                # also has some 4-fold ripple, but more 1..3-fold: measured <= 0.90. Rounded squares,
                                # incl. chipped / squashed / residual keystone: >= 1.65.
SLOT_ASSIGN_SLACK = 0.03        # a hole may take a shape whose overlap is within this of its own best overlap
VOTE_MIN_MARGIN = 0.03          # a peg fit votes only if its best shape beats the runner-up by this much

# Hand / skin ignoring (silent: no warnings, nothing paused; needs the 'b' depth calibration)
HAND_MIN_H = 20                 # mm above the calibration surface: a hand or arm is at least this high
HAND_MIN_PX = 2500              # raised blob size; it must also touch the image border (an arm always enters from an edge)
SKIN_MIN_FRAC = 0.25            # skin-coloured share of that blob -> a human hand/arm, masked out of peg detection.
                                # The black robot arm is not skin, so it is NOT masked (a peg it carries stays visible).
SKIN_CR, SKIN_CB = (135, 180), (85, 135)     # YCrCb skin range
HAND_PAD_PX = 15                # grow the ignored hand region by this (fingertips, motion blur)
TRAY_RAISED_MM = 25             # tray-coloured pixels higher than this above the calibration surface are NOT tray
                                # (skin, hands, pegs). Keep it above the tray's thickness if the tray may move after 'b'.

# Tray pose stabilisation (fixed camera + fixed tray => the pose should not move)
POSE_DEADBAND_PX = 3.0          # new tray corners closer than this to the locked pose are ignored (= noise)
POSE_MOVE_FRAMES = 8            # a real tray move must persist this many frames ...
POSE_MOVE_SPREAD_PX = 2.0       # ... and be consistent (corner std below this) before the pose is updated

# Peg tracking / voting
TRACK_RADIUS_PX = 40            # same colour within this distance = same peg
VOTE_N = 9                      # shape label = majority of the last VOTE_N frames
TRACK_MAX_MISS = 10             # frames a peg may vanish before its track is dropped
EMA_ALPHA = 0.35                # position smoothing (higher = faster, noisier)

LOCK_EXPOSURE = False           # set True only after the image looks well exposed
SHOW_REJECTED = False           # draw the colour blobs that were thrown away ('rejected blue 414px sol=0.61') - a
                                # diagnostic only, they never become pegs. Toggle with key 'x'.

# ============================ v7.3: depth, robot targets, pick/carry/place ============================
# All robot numbers are in the TRAY FRAME, built from depth at 'b': origin = a tray corner on the tray's top surface,
# x = along the long edge, y = across, z = up (towards the camera), mm; yaw = degrees about z from x towards y.
GRASP_BELOW_TOP_MM = 15         # gripper TCP (centre between the fingers) this far below the peg's top face
APPROACH_MM = 80                # pre-grasp / pre-place point this far above the pick / place point
PLACE_CLEARANCE_MM = 5          # extra lift so the peg's bottom clears the tray surface before it is lowered in
LIFT_MM = 12                    # top face this much higher than at rest = lifted by the gripper
MOVE_MM = 15                    # peg axis this far from its rest position = being moved
STILL_MM, STILL_FRAMES = 2.5, 8 # axis spread below STILL_MM over STILL_FRAMES frames = standing still
LOST_FRAMES = 45                # unseen this long with no arm near it = 'missing'
ARM_NEAR_PX = 45                # robot arm within this of a peg's last image position = it may be covering the peg
PEG_COLORS_UNIQUE = True        # one peg per colour: identity, shape and target are remembered by colour
JUMP_PX, JUMP_MM = 20, 10       # a jump larger than this resets the smoothing (fast motion, no lag)
ARM_DARK_V = 80                 # the robot arm is black (HSV V below this): no safety margin is cut around it in the
                                # depth gate, so a peg held in the gripper next to it stays visible
TOP_BAND_MM = 4.0               # top face + this much of the upper side wall form the peg's 3D outline (centre, yaw)
CARRIED_MIN_SOLIDITY, CARRIED_MIN_AREA = 0.55, 300    # relaxed blob rules for a remembered peg (partly hidden by fingers)
PICK_ORDER = ("yellow", "blue", "orange", "green", "red")   # order in which pegs are offered as the next pick

# ROS 2 (needs 'source /opt/ros/<distro>/setup.bash' before starting; without ROS the detector runs as before)
ROS_ENABLE = True
ROS_NODE = "peg_sorting_detector"
ROS_STATE_TOPIC = "/peg_sorting/state"          # std_msgs/String, JSON: tray, slots, every peg, next pick / place
ROS_PICK_TOPIC = "/peg_sorting/pick_pose"       # geometry_msgs/PoseStamped (tray frame): gripper TCP to grasp the next peg
ROS_PLACE_TOPIC = "/peg_sorting/place_pose"     # geometry_msgs/PoseStamped (tray frame): gripper TCP to release it in its slot
CAMERA_FRAME_ID = "camera_color_optical_frame"  # camera coordinates here are the colour optical frame (x right, y down)
TRAY_FRAME_ID = "peg_tray"                      # published as TF  CAMERA_FRAME_ID -> TRAY_FRAME_ID
# Your camera is static, so ONE fixed transform links it to the robot. Either publish it yourself (e.g.
#   ros2 run tf2_ros static_transform_publisher --x .. --y .. --z .. --qx .. --qy .. --qz .. --qw ..
#       --frame-id base_link --child-frame-id camera_color_optical_frame)
# or paste the camera pose in the robot base frame here as a 4x4 matrix (mm); then this node publishes it.
BASE_FRAME_ID = "base_link"
BASE_T_CAMERA = None
# Fine correction AFTER calibration, in the robot base frame (mm). It is added to every position the robot gets:
# if the gripper lands 2 mm short in +x, set x to +2; 1.5 mm too low -> z = +1.5. (Find it live with
# 'omx_camera_calibration.py --verify', which prints the value to paste here.)
CALIB_OFFSET_MM = [0.0, 0.0, 0.0]


def base_T_camera():
    """BASE_T_CAMERA with CALIB_OFFSET_MM applied (None if not calibrated)."""
    if BASE_T_CAMERA is None:
        return None
    T = np.array(BASE_T_CAMERA, np.float64)
    T[:3, 3] += np.asarray(CALIB_OFFSET_MM, np.float64)
    return T


# ============================== Geometry helpers ==============================
def centroid(c):
    M = cv2.moments(c)
    return (int(M["m10"] / (M["m00"] + 1e-6)), int(M["m01"] / (M["m00"] + 1e-6)))


def filled(c, shape):
    m = np.zeros(shape, np.uint8)
    cv2.drawContours(m, [c.astype(np.int32)], -1, 255, -1)
    return m


def project(c, M):
    return cv2.perspectiveTransform(np.asarray(c, np.float32).reshape(-1, 1, 2), M)


def classify_shape_legacy(c):
    c = np.asarray(c, np.float32).reshape(-1, 1, 2)
    area = cv2.contourArea(c)
    if area < 1:
        return "unknown"
    peri = cv2.arcLength(c, True)
    n = len(cv2.approxPolyDP(c, 0.03 * peri, True))
    n2 = len(cv2.approxPolyDP(c, 0.02 * peri, True))
    (_, _), r = cv2.minEnclosingCircle(c)
    fill = area / (np.pi * r * r + 1e-6)     # circle ~0.95, hexagon ~0.83, square ~0.64, triangle ~0.41

    if n == 3:
        return "triangle"
    if n == 4:
        (_, _), (w, h), _ = cv2.minAreaRect(c)
        if area / (w * h + 1e-6) < 0.75:     # a noisy triangle fills only ~50% of its rectangle
            return "triangle"
        return "square" if min(w, h) / (max(w, h) + 1e-6) >= 0.85 else "rectangle"
    if fill >= 0.90:
        return "circle"
    if (n == 6 or n2 == 6) and 0.75 <= fill < 0.90:
        return "hexagon"
    return "unknown"


SYMMETRY_DEG = {"triangle": 120.0, "square": 90.0, "rectangle": 180.0, "hexagon": 60.0, "circle": 360.0}


def ip(p):
    """float point -> int tuple for OpenCV drawing."""
    return (int(round(float(p[0]))), int(round(float(p[1]))))


def put_label(img, text, org, scale=0.45, color=(255, 255, 255), placed=None, bg=(35, 35, 35)):
    """Text on a dark box (readable over any background). With a 'placed' list the label is moved up (then down)
    until it no longer overlaps an earlier label of the same frame. Returns the box (x, y, w, h)."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), base = cv2.getTextSize(text, font, scale, 1)
    H_, W_ = img.shape[:2]
    bw, bh = tw + 6, th + base + 6
    x = int(min(max(0, org[0]), max(0, W_ - bw)))
    y0 = int(org[1])                                # baseline of the text

    def box(y):
        return (x, y - th - 3, bw, bh)

    def free(b):
        return all(b[0] + b[2] <= q[0] or q[0] + q[2] <= b[0] or b[1] + b[3] <= q[1] or q[1] + q[3] <= b[1]
                   for q in (placed or []))

    tries = [y0 - k * (bh + 2) for k in range(5)] + [y0 + k * (bh + 2) for k in range(1, 6)]
    tries = [y for y in tries if th + 3 <= y <= H_ - base - 3] or [min(max(y0, th + 3), H_ - base - 3)]
    y = next((y for y in tries if free(box(y))), tries[0])
    b = box(y)
    cv2.rectangle(img, (b[0], b[1]), (b[0] + b[2], b[1] + b[3]), bg, -1)
    cv2.putText(img, text, (x + 3, y), font, scale, color, 1, cv2.LINE_AA)
    if placed is not None:
        placed.append(b)
    return b


def _poly_mask(pts, size):
    m = np.zeros(size, np.uint8)
    cv2.fillPoly(m, [np.round(pts).astype(np.int32)], 1)
    return m


# Unit polygons for every whole-degree rotation (pre-computed once: the sweep below runs every frame)
_UNIT = {n: np.stack([np.stack([np.cos(np.deg2rad(d + np.arange(n) * 360.0 / n)),
                                np.sin(np.deg2rad(d + np.arange(n) * 360.0 / n))], 1)
                      for d in range(int(360 / n))]) for n in (3, 6)}


def _regular_polygon(center, R, n, ang_deg):
    return _UNIT[n][int(round(ang_deg)) % int(360 / n)] * R + np.asarray(center)


def _yaw_of(poly, name):
    """Rotation of a fitted model in degrees, modulo the shape's symmetry (circle: no rotation)."""
    if name == "circle":
        return 0.0
    if name == "rectangle":
        d = poly[1] - poly[0]
        return float(np.degrees(np.arctan2(d[1], d[0])) % 180.0)
    n = {"triangle": 3, "square": 4, "hexagon": 6}[name]
    v = poly[0] - poly.mean(0)
    return float(np.degrees(np.arctan2(v[1], v[0])) % (360.0 / n))


def fit_shape(c, min_aspect=None):
    """Fit ideal models (circle / equilateral triangle / hexagon / rectangle-square) to an outline and
    keep the one that overlaps it best (IoU). Replaces vertex counting, which breaks on ragged outlines.
    Returns dict(shape, iou, poly, center, yaw, scores); poly (the EXACT vertices of the fitted model)
    and center are in the SAME coordinates as c.
    Anisotropy (perspective squash) is removed first: triangle, square, hexagon and circle all have an
    isotropic second moment, so ANY squash is distortion, not shape. Un-squashing is an affine map, and an
    affine-squashed regular shape un-squashes back to the same regular shape (only rotated), so it is exact.
    v7.1 only did this for mild squash (>= 0.75); an oblique circle top face (~0.6) then became a 'rectangle'.
    Extra keys: 'polys' (fitted model per candidate shape), 'margin' (best minus runner-up), 'ripple', 'aspect'."""
    c = np.asarray(c, np.float32).reshape(-1, 2)
    if len(c) >= 3:             # every peg / slot shape is convex: the hull heals chunks lost to highlights,
        c = cv2.convexHull(c.reshape(-1, 1, 2)).reshape(-1, 2)      # occlusions and colour-mask dropouts
    M0 = cv2.moments(c.reshape(-1, 1, 2))
    if M0["m00"] > 1:
        mean = np.array([M0["m10"] / M0["m00"], M0["m01"] / M0["m00"]])
        cov = np.array([[M0["mu20"], M0["mu11"]], [M0["mu11"], M0["mu02"]]]) / M0["m00"]
        w, V = np.linalg.eigh(cov)
        ratio = np.sqrt(max(w[0], 1e-9) / max(w[1], 1e-9))
        lim = FIT_MIN_ASPECT if min_aspect is None else min_aspect
        if lim <= ratio < 0.995:             # below the limit = genuinely elongated (a strip): keep as is
            W = V @ np.diag(np.sqrt(w.mean() / np.maximum(w, 1e-9))) @ V.T
            out = _fit_iso(((c - mean) @ W.T + mean).astype(np.float32))
            if out["poly"] is not None:
                Wi = np.linalg.inv(W)
                unw = lambda p: ((np.asarray(p, np.float64) - mean) @ Wi.T + mean).astype(np.float32)
                out["poly"], out["center"] = unw(out["poly"]), unw(out["center"])
                out["polys"] = {k: unw(p) for k, p in out["polys"].items()}
                out["yaw"] = _yaw_of(out["poly"], out["shape"]) if out["shape"] != "unknown" else 0.0
            out["aspect"] = round(float(ratio), 3)
            return out
        out = _fit_iso(c)
        out["aspect"] = round(float(ratio), 3)
        return out
    return _fit_iso(c)


def _ripple(T, n, spectrum=False):
    """Relative amplitude of the n-th harmonic of the outline radius r(theta) of a filled mask.
    An n-sided polygon's radius rises and falls n times per turn, even with rounded corners; a circle's does not.
    hexagon (n=6): ~0.055 sharp, ~0.035 rounded; square (n=4): ~0.13 sharp, ~0.07 rounded; circle: < 0.023.
    Far more robust than overlap or radius spread once corners are rounded."""
    cnts, _ = cv2.findContours(T, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return np.zeros(181) if spectrum else 0.0
    d = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)
    if len(d) < 24:
        return np.zeros(181) if spectrum else 0.0
    M = cv2.moments(T, binaryImage=True)
    ctr = np.array([M["m10"], M["m01"]]) / max(M["m00"], 1e-9)
    v = d - ctr
    th, r = np.arctan2(v[:, 1], v[:, 0]), np.hypot(v[:, 0], v[:, 1])
    o = np.argsort(th)
    g = np.linspace(-np.pi, np.pi, 360, endpoint=False)
    rr = np.interp(g, th[o], r[o], period=2 * np.pi)
    F = np.abs(np.fft.rfft(rr)) / len(rr) * 2 / max(rr.mean(), 1e-9)
    return F if spectrum else float(F[n])


def _square_dominant(T):
    """True if the outline's 4-fold ripple dominates its 1..3-fold ripple (a square, not a circle with a flat cut)."""
    F = _ripple(T, 4, spectrum=True)
    return bool(F[4] >= SQUARE_DOMINANCE * max(F[1], F[2], F[3], 1e-4))


def _fit_iso(c):
    c = np.asarray(c, np.float32).reshape(-1, 2)
    unk = {"shape": "unknown", "iou": 0.0, "poly": None, "center": None, "yaw": 0.0, "scores": {}, "best": "",
           "polys": {}, "margin": 0.0, "ripple": 0.0, "aspect": 1.0}
    if len(c) < 3 or cv2.contourArea(c.reshape(-1, 1, 2)) < 25:
        return unk
    x0, y0 = c.min(0)
    x1, y1 = c.max(0)
    s = FIT_RES / max(float(x1 - x0), float(y1 - y0), 1e-6)
    pad = 8
    q = (c - [x0, y0]) * s + pad
    size = (int((y1 - y0) * s) + 2 * pad + 1, int((x1 - x0) * s) + 2 * pad + 1)
    T = _poly_mask(q, size)
    A = float(cv2.countNonZero(T))
    M = cv2.moments(T, binaryImage=True)
    if M["m00"] < 1:
        return unk
    cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]

    def iou(m):
        i = cv2.countNonZero(cv2.bitwise_and(T, m))
        u = cv2.countNonZero(cv2.bitwise_or(T, m))
        return i / (u + 1e-9)

    cand = {}
    r = np.sqrt(A / np.pi)
    th = np.linspace(0, 2 * np.pi, 72, endpoint=False)
    circ = np.stack([cx + r * np.cos(th), cy + r * np.sin(th)], 1)
    cand["circle"] = (iou(_poly_mask(circ, size)), circ)

    for name, n in (("triangle", 3), ("hexagon", 6)):          # regular polygon, rotation swept
        R = np.sqrt(2 * A / (n * np.sin(2 * np.pi / n)))
        best = (-1.0, None, 0.0)

        def try_angle(ang, best):
            poly = _regular_polygon((cx, cy), R, n, ang)
            v = iou(_poly_mask(poly, size))
            return (v, poly, ang) if v > best[0] else best

        for ang in np.arange(0.0, 360.0 / n, 6.0):             # coarse sweep, then refine to 1 degree
            best = try_angle(ang, best)
        for step, span in ((2.0, 4.0), (1.0, 1.0)):
            for ang in np.arange(best[2] - span, best[2] + span + 0.1, step):
                best = try_angle(ang, best)
        cand[name] = (best[0], best[1])
    try:                                                       # non-regular triangle
        _, tri = cv2.minEnclosingTriangle(q.reshape(-1, 1, 2).astype(np.float32))
        tri = tri.reshape(3, 2)
        v = iou(_poly_mask(tri, size))
        if v > cand["triangle"][0]:
            cand["triangle"] = (v, tri)
    except cv2.error:
        pass
    rect = cv2.minAreaRect(q.astype(np.float32))
    box = cv2.boxPoints(rect)
    w_, h_ = rect[1]
    rname = "square" if min(w_, h_) / (max(w_, h_) + 1e-6) >= 0.85 else "rectangle"
    cand[rname] = (iou(_poly_mask(box, size)), box)

    best = max(cand, key=lambda k: cand[k][0])
    scores = {k: round(float(v[0]), 3) for k, v in cand.items()}
    # Circle vs polygon with rounded corners: the overlap is nearly tied, the n-fold ripple of the radius decides.
    # (v7.1 used the spread of the HULL VERTEX radii for the hexagon only; v7.2 first version: hexagon only.)
    ripple, pair, ruled_out = 0.0, None, set()
    if best == "circle":
        close = [k for k in POLY_SIDES if k in cand and cand[k][0] >= cand["circle"][0] - POLY_CIRCLE_TIE[k]]
        rips = {k: _ripple(T, POLY_SIDES[k]) for k in close}
        hits = [k for k in close if rips[k] >= POLY_RIPPLE_MIN[k] and (k != "square" or _square_dominant(T))]
        if close:
            k = max(hits, key=lambda k: rips[k] / POLY_RIPPLE_MIN[k]) if hits else \
                max(close, key=lambda k: rips[k] / POLY_RIPPLE_MIN[k])
            pair, ripple = k, rips[k]
            ruled_out = {j for j in close if j not in hits}     # in the window but no ripple: not that polygon
            if hits:
                best = k
    elif best in POLY_SIDES and cand["circle"][0] >= cand[best][0] - POLY_CIRCLE_TIE[best]:
        pair, ripple = best, _ripple(T, POLY_SIDES[best])
        if ripple < POLY_RIPPLE_MIN[best]:
            best = "circle"
    v, poly = cand[best]
    rivals = [cand[k][0] for k in cand if k != best and k not in ruled_out and not (pair and k in ("circle", pair))]
    margin = float(v - max(rivals)) if rivals else 1.0
    if pair:                                                   # confidence of the ripple decision itself
        margin = min(margin, abs(ripple / POLY_RIPPLE_MIN[pair] - 1) * 0.1)
    back = lambda p: ((np.asarray(p, np.float32) - pad) / s + [x0, y0]).astype(np.float32)
    poly_o = back(poly)
    return {"shape": best if v >= FIT_MIN_IOU else "unknown", "iou": round(float(v), 3),
            "poly": poly_o, "center": back(np.array([[cx, cy]]))[0], "yaw": _yaw_of(poly_o, best),
            "scores": scores, "best": best, "polys": {k: back(p[1]) for k, p in cand.items()},
            "margin": round(margin, 3), "ripple": round(ripple, 4), "aspect": 1.0}


def classify_shape(c):
    """Shape label from template fitting (see fit_shape)."""
    return fit_shape(c)["shape"]


def is_hand_like(c):
    area = cv2.contourArea(c)
    if area < MIN_AREA or area > MAX_AREA:
        return True
    return area / (cv2.contourArea(cv2.convexHull(c)) + 1e-6) < MIN_SOLIDITY


def depth_at(img, px, k=5):
    x, y = px
    h, w = img.shape[:2]
    patch = img[max(0, y - k):min(h, y + k + 1), max(0, x - k):min(w, x + k + 1)]
    v = patch[patch > 0]
    return float(np.median(v)) if v.size else 0.0


def to_xyz(intr, px, z_mm):
    if z_mm <= 0:
        return None
    return rs.rs2_deproject_pixel_to_point(intr, [float(px[0]), float(px[1])], float(z_mm))


# ============================== Tray tracking ==============================
def tray_mask_of(img):
    hsv = cv2.cvtColor(cv2.GaussianBlur(img, (7, 7), 0), cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, TRAY_LO, TRAY_HI)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))


def order_corners(pts):
    pts = np.asarray(pts, np.float32).reshape(4, 2)
    s, d = pts.sum(1), np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)],
                     pts[np.argmax(s)], pts[np.argmax(d)]], np.float32)   # tl, tr, br, bl


def normalize_corners(c, prev=None):
    c = order_corners(c)
    if np.linalg.norm(c[1] - c[0]) < np.linalg.norm(c[2] - c[1]):   # long edge first
        c = np.roll(c, -1, axis=0)
    if prev is not None:                                           # keep 180-degree orientation stable
        flipped = np.roll(c, 2, axis=0)
        if np.linalg.norm(flipped - prev) < np.linalg.norm(c - prev):
            c = flipped
    return c


def find_tray(img, prev=None, exclude=None):
    """Return (ordered 4 corners or None, tray colour mask, info text).
    exclude: pixels that cannot be tray (raised above the calibration surface, or an ignored hand). Skin and
    trousers fall inside the tan tray-colour range; without this they merged into the tray and skewed its corners."""
    m = tray_mask_of(img)
    if exclude is not None:
        m[exclude > 0] = 0
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, m, "tray-colour mask is empty"
    big = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(big) < MIN_TRAY_AREA:
        return None, m, f"largest tray-colour blob only {int(cv2.contourArea(big))}px (< MIN_TRAY_AREA)"
    hull = cv2.convexHull(big)
    peri = cv2.arcLength(hull, True)
    quad = None
    for eps in (0.02, 0.03, 0.04, 0.05):
        a = cv2.approxPolyDP(hull, eps * peri, True)
        if len(a) == 4:
            quad = a
            break
    if quad is None:
        quad = cv2.boxPoints(cv2.minAreaRect(hull))
    info = f"tray-colour blob {int(cv2.contourArea(big))}px, hull {int(cv2.contourArea(hull))}px"
    return normalize_corners(quad, prev), m, info


def touches_border(c, shape):
    h, w = shape[:2]
    return bool((c[:, 0] < EDGE_MARGIN).any() or (c[:, 0] > w - 1 - EDGE_MARGIN).any() or
                (c[:, 1] < EDGE_MARGIN).any() or (c[:, 1] > h - 1 - EDGE_MARGIN).any())


def valid_quad(c):
    """A usable tray quad is finite, convex and big enough. Degenerate quads (collapsed or
    self-crossing hulls) give a singular homography and used to crash np.linalg.inv."""
    c = np.asarray(c, np.float32).reshape(4, 2)
    if not np.isfinite(c).all():
        return False
    q = c.reshape(-1, 1, 2)
    return bool(cv2.isContourConvex(q)) and cv2.contourArea(q) >= MIN_TRAY_AREA


def homographies(corners):
    dst = np.float32([[0, 0], [TRAY_W, 0], [TRAY_W, TRAY_H], [0, TRAY_H]])
    H = cv2.getPerspectiveTransform(corners, dst)
    return H, np.linalg.inv(H)


# ============================== Slot detection (rectified) ==============================
def _slot_score(f, name):
    s = f["scores"].get(name)
    if s is None and name == "square":
        s = f["scores"].get("rectangle")
    return s


def assign_slots(cands):
    """One hole per expected shape and one shape per hole, maximising the total fit.
    v7.1 kept 'the best hole per label', so when the hexagon hole was labelled 'circle' it competed with the real
    circle hole, lost, and was thrown away. Here a hole may also take a shape within SLOT_ASSIGN_SLACK of its own
    best overlap (circle <-> hexagon on a rounded hole), and its own label earns a small bonus."""
    opts = []
    for f in cands:
        top = max(f["scores"].values()) if f["scores"] else 0.0
        o = {}
        for name in EXPECTED_SLOTS:
            sc = _slot_score(f, name)
            if sc is not None and sc >= FIT_MIN_IOU and (f["shape"] == name or sc >= top - SLOT_ASSIGN_SLACK):
                o[name] = sc + (0.05 if f["shape"] == name else 0.0)
        opts.append(o)
    best_total, best_pick = -1.0, ()
    for pick in itertools.product(range(-1, len(cands)), repeat=len(EXPECTED_SLOTS)):
        used = [i for i in pick if i >= 0]
        if len(used) != len(set(used)):
            continue
        if any(i >= 0 and name not in opts[i] for name, i in zip(EXPECTED_SLOTS, pick)):
            continue
        total = sum(opts[i][name] for name, i in zip(EXPECTED_SLOTS, pick) if i >= 0)
        if total > best_total:
            best_total, best_pick = total, pick
    return {name: cands[i] for name, i in zip(EXPECTED_SLOTS, best_pick) if i >= 0}


def check_tray_proportion(slots):
    """A round hole must look round in the flattened view. If it is stretched, TRAY_W:TRAY_H does not match the real
    tray's length:width (and TRAY_MM_W/TRAY_MM_H, i.e. every mm value, are off by the same factor)."""
    circ = [s for s in slots if s["shape"] == "circle"]
    if not circ:
        return
    _, _, w, h = cv2.boundingRect(circ[0]["raw"])     # a round hole: width/height = x/y stretch
    sx_over_sy = w / max(h, 1e-6)                      # >1: tray x stretched relative to y
    if abs(sx_over_sy - 1) > 0.06:
        est = (TRAY_W / TRAY_H) / sx_over_sy
        print(f"NOTE: the circle slot is {w:.0f}x{h:.0f} px in the flattened view (should be round). The real tray "
              f"length:width is probably ~{est:.2f}:1, not {TRAY_MM_W / TRAY_MM_H:.2f}:1. Measure the tray and set "
              f"TRAY_MM_W / TRAY_MM_H (and TRAY_W:TRAY_H to the same ratio).")


def detect_slots(img, H):
    """Run on the EMPTY tray. Holes are found in the flattened tray view, each fitted with an ideal
    model, and the best fit per expected shape is kept. Stored in rectified tray coordinates."""
    rect = cv2.warpPerspective(img, H, (TRAY_W, TRAY_H))
    tray = tray_mask_of(rect)
    region = np.zeros(tray.shape, np.uint8)
    region[12:-12, 12:-12] = 255
    holes = cv2.bitwise_and(region, cv2.bitwise_not(tray))
    k7 = np.ones((7, 7), np.uint8)
    holes = cv2.morphologyEx(holes, cv2.MORPH_OPEN, k7)
    holes = cv2.morphologyEx(holes, cv2.MORPH_CLOSE, k7)

    def hole_ok(c):
        a = cv2.contourArea(c)
        if not (SLOT_MIN_AREA <= a <= SLOT_MAX_AREA):
            return False
        if a / (cv2.contourArea(cv2.convexHull(c)) + 1e-6) < 0.9:
            return False
        (_, _), (w, h), _ = cv2.minAreaRect(c)
        return min(w, h) / (max(w, h) + 1e-6) >= 0.5   # thin strip (tray edge / mat gap), not a hole

    cnts, _ = cv2.findContours(holes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs = []
    for c in cnts:
        if hole_ok(c):
            blobs.append(c)
        elif cv2.contourArea(c) >= SLOT_MIN_AREA:
            # Square-slot fix: a hole near the tray's long edge can touch the strip of mat seen along that edge
            # (your setup: the teal mat overlaps the tray edge). Hole + strip form one blob that fails the checks
            # above, so the square slot was lost. Cut thin attachments off this blob only and re-check the pieces.
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (SLOT_SPLIT_PX, SLOT_SPLIT_PX))
            part = cv2.morphologyEx(filled(c, holes.shape), cv2.MORPH_OPEN, k)
            pcs, _ = cv2.findContours(part, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            blobs += [p for p in pcs if hole_ok(p)]
    cands = []
    for c in blobs:
        f = fit_shape(c.astype(np.float32), min_aspect=SLOT_MIN_ASPECT)   # stricter: a 0.6 gap strip must
        if f["poly"] is not None:                                         # not become a 'square' slot
            f["raw"] = c
            cands.append(f)

    print("Hole candidates (flattened tray view):")
    for i, f in enumerate(cands):
        x, y, w, h = cv2.boundingRect(f["raw"])
        print(f"  #{i} at x={x + w // 2:3d} y={y + h // 2:3d}  -> {f['shape']:9s} scores={f['scores']}"
              + (f"  ripple={f['ripple']}" if f["ripple"] else ""))

    best = assign_slots(cands)
    missing = [n for n in EXPECTED_SLOTS if n not in best]
    if missing:
        print("WARNING: slots NOT found:", missing, "- check TRAY_LO/TRAY_HI and the 'Tray (rectified)' window.")
    if len(cands) > len(best):
        print(f"INFO: {len(cands) - len(best)} hole candidate(s) not used (not an expected shape / duplicate).")
    if not best:
        return []

    slots = []
    for name, f in best.items():
        poly = f["polys"].get(name)
        if poly is None and name == "square":
            poly = f["polys"].get("rectangle")
        cf = (float(f["center"][0]), float(f["center"][1]))
        slots.append({"contour": poly.reshape(-1, 1, 2).astype(np.float32), "shape": name,
                      "cf": cf, "center": ip(cf), "yaw": _yaw_of(poly, name),
                      "iou": f["scores"].get(name, f["scores"].get("rectangle", 0.0)), "raw": f["raw"]})
    check_tray_proportion(slots)

    slots.sort(key=lambda s: s["cf"][1])               # grid order: rows by y, then left -> right
    rows = [[slots[0]]]
    for s_ in slots[1:]:
        if abs(s_["cf"][1] - rows[-1][-1]["cf"][1]) > ROW_TOL:
            rows.append([s_])
        else:
            rows[-1].append(s_)
    ordered = []
    for row in rows:
        for s_ in sorted(row, key=lambda s: s["cf"][0]):
            s_["id"] = len(ordered) + 1
            ordered.append(s_)
    print("Slots:", [(s_["id"], s_["shape"], f"fit={s_['iou']}") for s_ in ordered])
    return ordered


# ============================== v7.3: 3D tray frame from depth ==============================
class TrayFrame:
    """Metric tray frame from the depth calibration ('b').
    1. The tray's top surface is fitted as a plane  n.X + d = 0  (camera coordinates, mm, n towards the camera),
       using the depth of the tray-coloured pixels of the empty tray.
    2. The 4 tray corners (colour outline) are intersected with that plane -> real 3D corners -> axes and real size.
    3. Any pixel can then be 'lifted' to a plane parallel to the tray at height h (the ray through the pixel meets it),
       which gives exact tray-frame positions for slots (h = 0) and for peg top faces (h = measured top height).
    Heights are perpendicular to the tray (true vertical), not along the camera axis as in the depth gate."""

    def __init__(self, intr, bg_mm, tray_mask, corners):
        self.fx, self.fy, self.ppx, self.ppy = float(intr.fx), float(intr.fy), float(intr.ppx), float(intr.ppy)
        self.bg = bg_mm
        m = cv2.erode(tray_mask, np.ones((9, 9), np.uint8)) > 0       # stay away from tray / hole edges
        v, u = np.nonzero(m & (bg_mm > 0))
        if len(u) < 500:
            raise ValueError("too few tray pixels with depth to fit the tray plane")
        step = max(1, len(u) // 20000)
        P = self.deproject(u[::step], v[::step], bg_mm[v[::step], u[::step]])
        n, d = self._fit_plane(P)
        for _ in range(2):                                             # drop outliers (holes, edges), refit
            r = np.abs(P @ n + d)
            P = P[r < max(3.0, 3 * np.median(r))]
            n, d = self._fit_plane(P)
        if d < 0:                                                      # normal towards the camera = 'up'
            n, d = -n, -d
        self.n, self.d = n, d
        self.plane_rms = float(np.sqrt(np.mean((P @ n + d) ** 2)))
        self.set_corners(corners)

    @staticmethod
    def _fit_plane(P):
        c = P.mean(0)
        n = np.linalg.svd(P - c, full_matrices=False)[2][-1]
        return n, -float(n @ c)

    def deproject(self, u, v, z):
        z = np.asarray(z, np.float64)
        return np.stack([(np.asarray(u, np.float64) - self.ppx) / self.fx * z,
                         (np.asarray(v, np.float64) - self.ppy) / self.fy * z, z], -1)

    def ray(self, px):
        px = np.asarray(px, np.float64).reshape(-1, 2)
        return np.stack([(px[:, 0] - self.ppx) / self.fx, (px[:, 1] - self.ppy) / self.fy, np.ones(len(px))], -1)

    def lift(self, px, h=0.0):
        """pixel(s) -> camera-frame point(s) on the plane parallel to the tray at height h (mm)."""
        r = self.ray(px)
        t = (h - self.d) / (r @ self.n)
        out = r * t[:, None]
        return out[0] if np.ndim(px) == 1 else out

    def height(self, Xc):
        return np.asarray(Xc) @ self.n + self.d

    def to_px(self, Xc):
        Xc = np.asarray(Xc, np.float64).reshape(-1, 3)
        out = np.stack([Xc[:, 0] / Xc[:, 2] * self.fx + self.ppx, Xc[:, 1] / Xc[:, 2] * self.fy + self.ppy], -1)
        return out

    def set_corners(self, corners):
        C = self.lift(np.asarray(corners, np.float64).reshape(4, 2), 0.0)
        x = C[1] - C[0]
        x -= (x @ self.n) * self.n
        x /= np.linalg.norm(x)
        z = self.n
        y = np.cross(z, x)
        self.R = np.stack([x, y, z], 1)              # columns = tray axes in camera coordinates
        # origin on the corner that puts the whole tray at x >= 0 and y >= 0 (z up fixes y by the right-hand rule)
        self.o = C[0] if (C[3] - C[0]) @ y >= 0 else C[3]
        self.corners_tray = self.to_tray(C)
        self.length = float(np.linalg.norm(C[1] - C[0]))
        self.width = float(abs((C[3] - C[0]) @ y))

    def to_tray(self, Xc):
        return (np.asarray(Xc, np.float64) - self.o) @ self.R

    def to_cam(self, Xt):
        return np.asarray(Xt, np.float64) @ self.R.T + self.o

    def tray_to_px(self, Xt):
        return self.to_px(self.to_cam(Xt))

    def surface_height_at(self, px, k=4):
        """Height of the EMPTY scene (calibration depth) at a pixel: slot floor, table under a peg ..."""
        x, y = ip(px)
        h_, w_ = self.bg.shape
        if not (0 <= x < w_ and 0 <= y < h_):
            return None
        patch = self.bg[max(0, y - k):y + k + 1, max(0, x - k):x + k + 1]
        vv, uu = np.nonzero(patch > 0)
        if not len(vv):
            return None
        P = self.deproject(uu + max(0, x - k), vv + max(0, y - k), patch[vv, uu])
        return float(np.median(self.height(P)))

    @staticmethod
    def _pix(contour, depth_mm, erode=0):
        """(u, v, depth) of the valid depth pixels inside a contour, computed on its bounding box only (fast)."""
        c = np.asarray(contour, np.float32).reshape(-1, 2)
        x, y, w, h = cv2.boundingRect(c.astype(np.int32).reshape(-1, 1, 2))
        H_, W_ = depth_mm.shape
        x0, y0, x1, y1 = max(0, x), max(0, y), min(W_, x + w + 1), min(H_, y + h + 1)
        if x1 <= x0 or y1 <= y0:
            return np.zeros(0, int), np.zeros(0, int), np.zeros(0)
        m = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.drawContours(m, [(c - [x0, y0]).astype(np.int32).reshape(-1, 1, 2)], -1, 255, -1)
        if erode:
            m = cv2.erode(m, np.ones((erode, erode), np.uint8))
        d = depth_mm[y0:y1, x0:x1]
        v, u = np.nonzero((m > 0) & (d > 0))
        return u + x0, v + y0, d[v, u]

    def blob_top(self, contour, depth_mm):
        """Highest part of a colour blob (90th percentile height): tells 'lifted' even with the top face hidden."""
        u, v, z = self._pix(contour, depth_mm)
        if len(u) < 30:
            return None
        return float(np.percentile(self.height(self.deproject(u, v, z)), 90))

    def top_outline(self, silhouette, depth_mm, h_top):
        """3D outline of a peg's top: every blob pixel within TOP_BAND_MM of the top height, deprojected with ITS OWN
        depth (not projected onto one plane). Side-wall pixels then land on the peg's real outline (the wall is
        vertical), so the outline is unbiased. 3D test: centre error 0.5-1.3 mm (plane projection: 1.7-3.5 mm)."""
        u, v, z = self._pix(silhouette, depth_mm)
        if len(u) < 30:
            return None
        P = self.deproject(u, v, z)
        P = P[self.height(P) >= h_top - TOP_BAND_MM]
        if len(P) < 30:
            return None
        xy = self.to_tray(P)[:, :2].astype(np.float32)
        return cv2.convexHull(xy.reshape(-1, 1, 2)).reshape(-1, 2)

    def face_height(self, contour, depth_mm):
        """Robust metric height of a peg's top face: median over its valid depth pixels (edges eroded off)."""
        u, v, z = self._pix(contour, depth_mm, erode=3)
        if len(u) < 30:
            return None
        hs = self.height(self.deproject(u, v, z))
        lo, hi = np.percentile(hs, (15, 85))
        sel = hs[(hs >= lo) & (hs <= hi)]
        return float(np.median(sel if sel.size else hs))

    def annotate_slots(self, slots, H_inv):
        """Metric pose of every slot: centre on the tray surface, outline, rotation and the hole floor height.
        Seen at an angle, a hole's far inner wall hides part of its floor, so part of the visible outline is the
        BOTTOM edge of the hole, not the top edge (in the 3D test that shifted slot centres by 2.5-3 mm). Each outline
        point is therefore lifted to its own height: tray surface (0) where the outside is tray surface, the hole floor
        where the outside is inner wall (found from the calibration depth). Then the ideal shape is fitted in mm."""
        for s in slots:
            c_img = project([s["cf"]], H_inv).reshape(2)
            poly_img = project(s["contour"], H_inv).reshape(-1, 2)
            s["img_center"] = ip(c_img)
            s["yaw_rect"] = s.get("yaw_rect", s["yaw"])
            fl = self.surface_height_at(c_img)
            s["floor_h"] = fl if fl is not None else 0.0
            s["tray"] = self.to_tray(self.lift(c_img, 0.0))
            s["tray_poly"] = self.to_tray(self.lift(poly_img, 0.0))[:, :2].astype(np.float32)
            s["yaw"] = _yaw_of(s["tray_poly"], s["shape"])
            s["wall_px"] = 0
            if s.get("raw") is not None and s["floor_h"] < -3.0:
                self._refine_slot(s, H_inv, c_img)
            s["size_mm"] = float(2 * np.sqrt(cv2.contourArea(s["tray_poly"].reshape(-1, 1, 2)) / np.pi))

    def _refine_slot(self, s, H_inv, c_img):
        raw = np.asarray(s["raw"], np.float64).reshape(-1, 2)
        dense = []
        for a, b in zip(raw, np.roll(raw, -1, 0)):             # densify the hole outline (every ~2 px)
            k = max(1, int(np.linalg.norm(b - a) / 2))
            dense += [a + (b - a) * t for t in np.arange(k) / k]
        pts = project(np.asarray(dense), H_inv).reshape(-1, 2)
        dirs = pts - c_img
        dirs /= np.maximum(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-6)
        h_out = np.array([self.surface_height_at(q, k=1) for q in pts + 3 * dirs], dtype=object)
        wall = np.array([h is not None and h < 0.4 * s["floor_h"] for h in h_out])   # outside is below the surface
        hs = np.where(wall, s["floor_h"], 0.0)
        P = np.array([self.lift(q, h) for q, h in zip(pts, hs)])
        xy = self.to_tray(P)[:, :2].astype(np.float32)
        f = fit_shape(xy, min_aspect=SLOT_MIN_ASPECT)
        poly = f["polys"].get(s["shape"], f["polys"].get("rectangle") if s["shape"] == "square" else None)
        if poly is None or f["center"] is None:
            return
        s["tray"] = np.array([float(f["center"][0]), float(f["center"][1]), 0.0])
        s["tray_poly"] = np.asarray(poly, np.float32)
        s["yaw"] = _yaw_of(s["tray_poly"], s["shape"])
        s["wall_px"] = int(wall.sum())

    def summary(self):
        return (f"tray plane fitted (rms {self.plane_rms:.1f} mm), camera {self.d:.0f} mm above the tray surface; "
                f"tray measured {self.length:.0f} x {self.width:.0f} mm")


def grip_yaw(shape, yaw):
    """Gripper closing direction for a parallel gripper: across two flat faces (circle: any, 0)."""
    if yaw is None or shape not in POLY_SIDES:
        return 0.0
    n = POLY_SIDES[shape]
    return float((yaw + 180.0 / n) % (360.0 / n))


def wrap(a, per):
    return float((a + per / 2) % per - per / 2)


# ============================== Peg detection ==============================
def depth_gate(depth_mm, bg_mm, img=None):
    valid = (depth_mm > 0) & (bg_mm > 0)
    height = np.where(valid, bg_mm - depth_mm, 0).astype(np.float32)
    height = cv2.medianBlur(height, 5)
    too_tall = valid & (height > MAX_H)
    seed = too_tall
    if img is not None:      # v7.3: no safety margin around the BLACK robot arm, so a peg in its gripper survives
        seed = too_tall & (cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 2] > ARM_DARK_V)
    too_tall = too_tall | (cv2.dilate(seed.astype(np.uint8) * 255, np.ones((15, 15), np.uint8), iterations=2) > 0)
    in_band = (height >= MIN_H) & (height <= MAX_H)
    return (((in_band | ~valid) & ~too_tall).astype(np.uint8) * 255), height


def top_face(c, height, shape):
    """Isolate the TOP FACE of a peg from its colour blob using the height map.
    Returns (contour, found, info).
      found=True  -> contour is the top face only (side walls removed).
      found=False -> contour is the whole colour silhouette (top + side walls); its centroid is a
                     VOLUME-like centre and must NOT be used as the grasp/alignment reference.
      info        -> why it succeeded/failed ('short' is shown on screen, the rest by the 'd' dump)."""
    m = filled(c, shape)
    vals = height[(m > 0) & (height > 0)]
    info = {"blob_px": int(cv2.countNonZero(m)), "valid_height_px": int(vals.size),
            "top_height_mm": None, "top_face_ratio": None, "reason": "", "short": ""}
    if vals.size < 50:
        info["reason"], info["short"] = "fewer than 50 valid depth pixels on the blob", "no depth"
        return c, False, info
    top = np.percentile(vals, 90)                                  # height of the top surface
    info["top_height_mm"] = round(float(top), 1)
    face = ((m > 0) & (height >= top - TOP_FACE_TOL)).astype(np.uint8) * 255
    face = cv2.morphologyEx(face, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(face, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        info["reason"], info["short"] = "no pixels within TOP_FACE_TOL of the top height", "no plateau"
        return c, False, info
    best = max(cnts, key=cv2.contourArea)
    fa = cv2.contourArea(best)
    ratio = fa / (cv2.contourArea(c) + 1e-6)
    (_, _), (w_, h_), _ = cv2.minAreaRect(best)
    aspect = min(w_, h_) / (max(w_, h_) + 1e-6)
    solidity = fa / (cv2.contourArea(cv2.convexHull(best)) + 1e-6)
    info.update(top_face_ratio=round(float(ratio), 2), top_face_px=int(fa),
                aspect=round(float(aspect), 2), solidity=round(float(solidity), 2))
    if ratio >= 0.3:                                               # as before: clearly the top face
        info["reason"] = "ok"
        return best, True, info
    if ratio >= TOP_FACE_MIN_RATIO and fa >= TOP_FACE_MIN_PX and aspect >= TOP_FACE_MIN_ASPECT \
            and solidity >= TOP_FACE_MIN_SOLIDITY:                 # small but compact: tall peg, oblique view
        info["reason"] = "ok (small top face on a tall peg)"
        return best, True, info
    if ratio < TOP_FACE_MIN_RATIO:
        info["short"] = f"top={ratio:.0%}"
    elif fa < TOP_FACE_MIN_PX:
        info["short"] = f"{int(fa)}px"
    elif aspect < TOP_FACE_MIN_ASPECT:
        info["short"] = "thin strip"
    else:
        info["short"] = "ragged"
    info["reason"] = "top face too small / thin / ragged to trust (" + info["short"] + ")"
    return c, False, info


REJECTED, PEG_MASK = [], None          # filled by detect_pegs; drawn / shown by main() for diagnosis


def detect_pegs(img, depth_mm, bg_mm, intr, H, gate=None, do_fit=True, ignore=None, geo=None, known=()):
    """ignore: mask of an ignored human hand/arm (scan_raised). Nothing inside it can become a peg.
    geo:   TrayFrame (v7.3) -> metric top height, axis position and rotation of every peg with a top face.
    known: colours of remembered pegs: their blobs pass with relaxed rules (partly hidden by the gripper fingers)."""
    global REJECTED, PEG_MASK
    REJECTED = []
    hsv = cv2.cvtColor(cv2.GaussianBlur(img, (7, 7), 0), cv2.COLOR_BGR2HSV)
    allowed = np.full(hsv.shape[:2], 255, np.uint8)
    height = None
    if bg_mm is not None:
        allowed, height = gate if gate is not None else depth_gate(depth_mm, bg_mm)
    if ignore is not None and ignore.any():
        allowed = cv2.bitwise_and(allowed, cv2.bitwise_not(ignore))

    kernel = np.ones((5, 5), np.uint8)
    pegs = []
    combined = np.zeros(hsv.shape[:2], np.uint8)
    for name, ranges in COLORS.items():
        mask = np.zeros(hsv.shape[:2], np.uint8)
        for lo, hi in ranges:
            mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
        mask = cv2.bitwise_and(mask, allowed)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        combined |= mask

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            a0 = cv2.contourArea(c)
            if name in known and CARRIED_MIN_AREA <= a0 <= MAX_AREA and \
                    a0 / (cv2.contourArea(cv2.convexHull(c)) + 1e-6) >= CARRIED_MIN_SOLIDITY:
                pass                              # remembered peg, maybe partly covered by the gripper: keep
            elif is_hand_like(c):
                a_ = cv2.contourArea(c)
                if a_ >= 400:                     # remember WHY a sizeable colour blob was thrown away
                    REJECTED.append((c, name, a_, a_ / (cv2.contourArea(cv2.convexHull(c)) + 1e-6)))
                continue
            silhouette = c                       # whole colour blob (top + side walls), display only
            top_ok, tinfo = False, {"reason": "no height map (press 'b' first)"}
            if height is not None:               # needs 'b' calibration (height map)
                c, top_ok, tinfo = top_face(c, height, mask.shape)
            # From here on c is the TOP FACE (when top_ok). Everything below uses it.
            M = cv2.moments(c)
            cf = (M["m10"] / (M["m00"] + 1e-6), M["m01"] / (M["m00"] + 1e-6))   # centre of the TOP FACE
            rc = project(c, H) if H is not None else None      # top-face outline in the flat tray view
            fit, rcf = None, None
            if top_ok and do_fit:                # no top face => no shape: the silhouette would lie
                fit = fit_shape(rc if rc is not None else c)
            if rc is not None:
                if fit is not None and fit["center"] is not None:
                    rcf = (float(fit["center"][0]), float(fit["center"][1]))
                else:
                    Mr = cv2.moments(rc)
                    rcf = (Mr["m10"] / (Mr["m00"] + 1e-6), Mr["m01"] / (Mr["m00"] + 1e-6))
            ctr = ip(cf)
            # ---- v7.3: metric 3D from depth (tray frame) ----
            h_top, top_t, yaw_t = None, None, None
            if geo is not None and top_ok:
                h_top = geo.face_height(c, depth_mm)          # true height of the top face above the tray surface
                if h_top is not None:
                    top_t = geo.to_tray(geo.lift(cf, h_top))  # top-face centre; (x, y) = the peg's AXIS on the tray
                    out3 = geo.top_outline(silhouette, depth_mm, h_top)
                    f3 = fit_shape(out3) if (out3 is not None and fit is not None) else None
                    if out3 is not None:                      # better: centre of the real 3D outline (see top_outline)
                        M3 = cv2.moments(out3.reshape(-1, 1, 2))
                        if M3["m00"] > 1:
                            top_t = np.array([M3["m10"] / M3["m00"], M3["m01"] / M3["m00"], h_top])
                    if fit is not None and fit["poly"] is not None and H is not None and fit["shape"] != "unknown":
                        fit["yaw_rect"] = fit["yaw"]
                        poly3 = None if f3 is None else f3["polys"].get(fit["shape"], f3["polys"].get("rectangle"))
                        if poly3 is not None:             # real rotation, from the metric outline
                            fit["yaw"] = yaw_t = _yaw_of(poly3, fit["shape"])
                        else:
                            pimg = project(fit["poly"], np.linalg.inv(H)).reshape(-1, 2)
                            pt = geo.to_tray(geo.lift(pimg, h_top))[:, :2]
                            fit["yaw"] = yaw_t = _yaw_of(pt, fit["shape"])
            if geo is not None and fit is not None and yaw_t is None and "yaw_rect" not in fit:
                fit["yaw_rect"], fit["yaw"] = fit["yaw"], None   # never mix rect-space and tray-frame angles
            sil_c = centroid(silhouette)
            blob_h = geo.blob_top(silhouette, depth_mm) if geo is not None else None
            pegs.append({"color": name, "shape": fit["shape"] if fit else "unknown", "area": float(a0),
                         "h_top": h_top, "top_t": top_t, "sil_center": sil_c, "blob_h": blob_h,
                         "contour": c, "silhouette": silhouette, "top_ok": top_ok, "top_info": tinfo,
                         "fit": fit, "cf": cf, "center": ctr,
                         "rcontour": rc, "rcf": rcf, "rcenter": ip(rcf) if rcf else None,
                         "yaw": fit["yaw"] if fit else None, "stable": False, "votes": "0/0",
                         # depth at the top-face centre = distance to the top surface (grasp point)
                         "xyz": to_xyz(intr, ctr, depth_at(depth_mm, ctr))})
    PEG_MASK = combined
    return pegs


class PegTracker:
    """Gives each peg a persistent identity so its shape label is a VOTE over the last VOTE_N frames and
    its position/rotation are smoothed. Removes the per-frame flicker (square / rectangle / unknown)."""

    def __init__(self):
        self.tracks, self.next_id = [], 1
        self.frozen = set()        # v7.3: colours being handled by the gripper: no shape votes (top face hidden)

    def update(self, pegs):
        if PEG_COLORS_UNIQUE:      # v7.3: one peg per colour -> keep the best blob of each colour (top face, size)
            best_of = {}
            for p in pegs:
                k = (p["top_ok"], p["area"])
                if p["color"] not in best_of or k > best_of[p["color"]][0]:
                    best_of[p["color"]] = (k, p)
            pegs[:] = [v[1] for v in best_of.values()]
        used = set()
        for p in pegs:
            best, bd = None, (1e9 if PEG_COLORS_UNIQUE else TRACK_RADIUS_PX)   # unique colour: any distance
            for t in self.tracks:
                if t["color"] != p["color"] or t["id"] in used:
                    continue
                d = float(np.hypot(t["cf"][0] - p["cf"][0], t["cf"][1] - p["cf"][1]))
                if d < bd:
                    best, bd = t, d
            if best is None:
                best = {"id": self.next_id, "color": p["color"], "hist": deque(maxlen=VOTE_N),
                        "yaws": deque(maxlen=VOTE_N), "cf": p["cf"], "rcf": p["rcf"], "miss": 0}
                self.next_id += 1
                self.tracks.append(best)
            used.add(best["id"])
            best["miss"] = 0
            a = EMA_ALPHA
            if np.hypot(best["cf"][0] - p["cf"][0], best["cf"][1] - p["cf"][1]) > JUMP_PX:
                best["cf"], best["rcf"] = p["cf"], p["rcf"]     # fast motion (gripper): follow without lag
            best["cf"] = ((1 - a) * best["cf"][0] + a * p["cf"][0], (1 - a) * best["cf"][1] + a * p["cf"][1])
            if p["top_t"] is not None:                         # v7.3: smoothed metric top-face centre
                tt = best.get("top_t")
                best["top_t"] = p["top_t"] if tt is None or np.linalg.norm(tt - p["top_t"]) > JUMP_MM \
                    else (1 - a) * tt + a * p["top_t"]
                p["top_t"], p["h_top"] = best["top_t"], float(best["top_t"][2])
            if p["rcf"] is not None:
                if best["rcf"] is None:
                    best["rcf"] = p["rcf"]
                else:
                    best["rcf"] = ((1 - a) * best["rcf"][0] + a * p["rcf"][0],
                                   (1 - a) * best["rcf"][1] + a * p["rcf"][1])
            new_fit = p["fit"] is not None                     # a fresh fit exists only on fit frames
            if new_fit:
                best["fit"] = p["fit"]
            elif p["top_ok"] and best.get("fit") is not None:
                p["fit"] = best["fit"]                         # reuse the last fit (drawing / matching)
            if new_fit and p["top_ok"] and p["fit"]["shape"] != "unknown" and p["color"] not in self.frozen \
                    and p["fit"].get("margin", 1.0) >= VOTE_MIN_MARGIN:
                best["hist"].append(p["fit"]["shape"])         # one vote per FRESH, UNAMBIGUOUS fit only
                best["yaws"].append((p["fit"]["shape"], p["fit"]["yaw"]))
                best["yaw_fresh"] = best.get("yaw_fresh", 0) + 1

            # Label with hysteresis: once a shape has a 60% majority it stays until ANOTHER shape reaches 60%.
            # v7.1 dropped back to 'unknown' whenever the majority dipped (blue peg: square / unknown flicker).
            cnt, total = Counter(best["hist"]), len(best["hist"])
            if cnt:
                top, n_top = cnt.most_common(1)[0]
                if total >= 4 and n_top / total >= 0.6:
                    best["label"] = top
            label = best.get("label")
            stable = label is not None
            n = cnt[label] if stable else (cnt.most_common(1)[0][1] if cnt else 0)
            label = label or "unknown"
            p["id"], p["stable"], p["votes"] = best["id"], stable, f"{n}/{total}"
            p["yaw_fresh"] = best.get("yaw_fresh", 0)
            # v7.3: a remembered peg keeps its shape while its top face is hidden (gripper); it is marked 'remembered'
            p["remembered"] = PEG_COLORS_UNIQUE and stable and not p["top_ok"]
            p["shape"] = label if (stable and (p["top_ok"] or p["remembered"])) else "unknown"
            p["cf"], p["center"] = best["cf"], ip(best["cf"])
            if best["rcf"] is not None:
                p["rcf"], p["rcenter"] = best["rcf"], ip(best["rcf"])
            p["yaw"] = None
            if p["shape"] not in ("unknown", "circle"):        # circular mean, period = shape symmetry
                per = SYMMETRY_DEG[p["shape"]]
                ys = [y for s_, y in best["yaws"] if s_ == p["shape"] and y is not None]
                if ys:
                    z = np.exp(1j * 2 * np.pi * np.array(ys) / per).mean()
                    p["yaw"] = float((np.angle(z) * per / (2 * np.pi)) % per)
        for t in self.tracks:
            if t["id"] not in used:
                t["miss"] += 1
        if not PEG_COLORS_UNIQUE:        # unique colours: a track (= the peg's memory) is never dropped
            self.tracks = [t for t in self.tracks if t["miss"] <= TRACK_MAX_MISS]

    def labels(self):
        return {t["color"]: t.get("label") for t in self.tracks}

    def reset_yaw(self, color):
        """v7.3: forget the rotation history (the gripper may have turned the peg); the shape label is kept."""
        for t in self.tracks:
            if t["color"] == color:
                t["yaws"].clear()
                t["yaw_fresh"] = 0


def scan_raised(img, height):
    """Silent replacement of the v7/v7.1 'human intervention' detector (no warnings, nothing paused).
    Finds sizeable raised objects that enter from the image border (an arm always does; pegs sit inside the
    workspace) and returns two masks:
      hand -> those that are mostly skin-coloured: a human hand/arm. Masked out of peg detection and tray mask.
      arms -> all of them (human or the black robot arm). Only used to keep the tray pose while one is over the tray.
    Together with the depth gate (anything taller than MAX_H) and is_hand_like() this is the hand ignoring."""
    fg = (height >= HAND_MIN_H).astype(np.uint8) * 255
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fg)
    h, w = fg.shape
    hand, arms = np.zeros((h, w), np.uint8), np.zeros((h, w), np.uint8)
    skin = None
    for i in range(1, n):
        x, y, bw, bh, a = [int(v) for v in stats[i]]
        if a < HAND_MIN_PX or not (x <= 2 or y <= 2 or x + bw >= w - 2 or y + bh >= h - 2):
            continue
        comp = lab == i
        arms[comp] = 255
        if skin is None:
            ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
            skin = cv2.inRange(ycc, (0, SKIN_CR[0], SKIN_CB[0]), (255, SKIN_CR[1], SKIN_CB[1])) > 0
        if np.count_nonzero(skin & comp) / a >= SKIN_MIN_FRAC:
            hand[comp] = 255
    if hand.any():
        k = 2 * HAND_PAD_PX + 1
        hand = cv2.dilate(hand, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    return hand, arms


# ============================== v7.3: pick / carry / place monitor ==============================
def _r(v, n=1):
    return None if v is None else round(float(v), n)


class PickPlaceMonitor:
    """Remembers every peg by colour and follows it through pick and place:
        on_table  -> resting, top face visible. Its REST pose gives the pick target (frozen while it is handled).
        occluded  -> not visible and the robot arm is over its last position (gripper closing on it).
        carried   -> lifted (top face higher than at rest), moving, or gone together with the arm.
        placed    -> standing still inside its target slot (ALIGNED or not is reported).
        missing   -> not seen for a long time and no arm near it.
    The shape and the target slot never change while the peg is occluded or carried."""

    def __init__(self):
        self.mem, self.events, self.reset = {}, [], set()

    def frozen(self):
        return {c for c, m in self.mem.items() if m["state"] in ("occluded", "carried") and not m["settle"]}

    def _arm_near(self, arms, px):
        if arms is None or px is None:
            return False
        disk = np.zeros(arms.shape, np.uint8)
        cv2.circle(disk, px, ARM_NEAR_PX, 255, -1)
        return bool(np.count_nonzero(disk & arms))

    def _event(self, now, m, old, new, extra=""):
        msg = f"[{now:7.1f}s] {m['color']} {m['shape'] or '?'}: {old} -> {new}{extra}"
        print(msg)
        self.events.append(msg)

    def update(self, pegs, slots, geo, arms, now, labels):
        out = []
        if geo is None:
            return out
        seen = {p["color"]: p for p in pegs}
        for col in sorted(set(self.mem) | set(seen)):
            m = self.mem.setdefault(col, {"color": col, "state": "new", "rest": None, "rest_h": None,
                                          "support_h": None, "length": None, "rest_yaw": None, "shape": None,
                                          "hist": deque(maxlen=STILL_FRAMES), "miss": 0, "last_px": None,
                                          "pos": None, "h": None, "yaw": None, "approx": False, "carry_n": 0,
                                          "settle": False})
            p = seen.get(col)
            if labels.get(col):
                m["shape"] = labels[col]
            target = next((s for s in slots if s["shape"] == m["shape"] and "tray" in s), None)
            old, new = m["state"], m["state"]
            if p is not None:
                m["miss"], m["last_px"] = 0, p["center"]
                exact = p["top_t"] is not None
                if exact:
                    m["pos"], m["h"], m["approx"] = np.asarray(p["top_t"][:2]), float(p["top_t"][2]), False
                    m["yaw"] = p["yaw"]
                else:                               # top face hidden (fingers): blob centre lifted at the last height
                    h_est = m["h"] if m["h"] is not None else (m["rest_h"] or 0.0)
                    m["pos"], m["approx"] = geo.to_tray(geo.lift(p["sil_center"], h_est))[:2], True
                m["hist"].append(m["pos"])
                H_ = np.array(m["hist"])
                still = exact and len(H_) == STILL_FRAMES and \
                    float(np.max(np.linalg.norm(H_ - H_.mean(0), axis=1))) < STILL_MM
                in_slot = target is not None and \
                    cv2.pointPolygonTest(target["tray_poly"].reshape(-1, 1, 2), tuple(map(float, m["pos"])), False) >= 0
                top_now = m["h"] if exact else p.get("blob_h")       # depth: highest point even if the top is hidden
                lifted = m["rest_h"] is not None and top_now is not None and top_now > m["rest_h"] + LIFT_MM
                moved = exact and m["rest"] is not None and float(np.linalg.norm(m["pos"] - m["rest"])) > MOVE_MM
                m["carry_n"] = m["carry_n"] + 1 if (lifted or (moved and not still)) else 0
                want = None
                if exact and still and in_slot:
                    want = "placed"
                elif m["carry_n"] >= 3:
                    want = "carried"
                elif exact and still:
                    want = "on_table"
                elif not exact and old in ("on_table", "placed", "new") and self._arm_near(arms, p["center"]):
                    want = "occluded"               # gripper over it: top face hidden, not lifted (yet)
                if want in ("placed", "on_table") and old in ("carried", "occluded") and not m["settle"]:
                    m["settle"] = True              # the gripper may have turned it: re-measure the rotation first
                    self.reset.add(col)
                    want = None
                if m["settle"]:
                    if want in ("placed", "on_table") and (p.get("yaw_fresh", 0) >= 3 or m["shape"] == "circle"):
                        m["settle"], m["yaw"] = False, p["yaw"]
                    elif want == "carried":
                        m["settle"] = False
                    else:
                        want = None
                if want is not None:
                    new = want
                if new == "on_table" and exact and still:
                    m["rest"], m["rest_h"], m["rest_yaw"] = m["pos"].copy(), m["h"], m["yaw"]
                    sup = geo.surface_height_at(geo.tray_to_px([m["pos"][0], m["pos"][1], 0.0])[0])
                    m["support_h"] = sup if sup is not None else 0.0
                    m["length"] = m["rest_h"] - m["support_h"]
            else:
                m["miss"] += 1
                m["hist"].clear()
                near = self._arm_near(arms, m["last_px"])
                if old in ("on_table", "placed", "new"):
                    new = "occluded" if near else ("missing" if m["miss"] > LOST_FRAMES else old)
                elif old == "occluded" and not near and m["miss"] > 5:
                    new = "carried"                 # the arm left and took the peg with it (hidden in the gripper)
                elif old == "carried" and m["miss"] > 4 * LOST_FRAMES and (arms is None or not arms.any()):
                    new = "missing"

            o = self._targets(m, target)
            if new != old and not (old == "new" and new in ("occluded", "missing")):
                extra = ""
                if new == "placed" and target is not None:
                    extra = (f" in S{target['id']}: " + ("ALIGNED" if o["aligned"] else "NOT aligned") +
                             f" (off {o['dist_mm']:.1f} mm" + ("" if o["rot_deg"] is None else
                                                              f", rot {o['rot_deg']:+.1f} deg") + ")")
                elif new == "carried" and target is not None:
                    extra = f" (target S{target['id']} {target['shape']})"
                elif new == "on_table" and m["rest"] is not None:
                    extra = f" at x={m['rest'][0]:.0f} y={m['rest'][1]:.0f} mm, top {m['rest_h']:.0f} mm"
                self._event(now, m, old, new, extra)
            m["state"] = new
            o["state"] = new
            if p is not None:
                p["m3d"] = o
            out.append(o)
        return out

    def _targets(self, m, target):
        """Everything the robot needs for this peg, in the tray frame (mm, deg)."""
        sh = m["shape"]
        per = SYMMETRY_DEG.get(sh, 360.0)
        o = {"color": m["color"], "shape": sh, "state": m["state"], "visible": m["miss"] == 0,
             "approx": bool(m["approx"]), "target_slot": None if target is None else target["id"],
             "pos": None if m["pos"] is None else [_r(m["pos"][0]), _r(m["pos"][1]), _r(m["h"])],
             "yaw_deg": _r(m["yaw"]), "length_mm": _r(m["length"]),
             "dx_mm": None, "dy_mm": None, "dist_mm": None, "rot_deg": None, "aligned": False,
             "pick": None, "place": None, "last_px": m["last_px"]}
        if target is not None and m["pos"] is not None:
            d = np.asarray(target["tray"][:2]) - m["pos"]
            o["dx_mm"], o["dy_mm"], o["dist_mm"] = _r(d[0]), _r(d[1]), _r(np.linalg.norm(d))
            if m["yaw"] is not None and sh not in (None, "circle"):
                o["rot_deg"] = _r(wrap(target["yaw"] - m["yaw"], per))
            o["aligned"] = bool(o["dist_mm"] <= ALIGN_TOL_MM and (o["rot_deg"] is None or abs(o["rot_deg"]) <= ALIGN_TOL_DEG))
        if m["rest"] is not None:
            gy = grip_yaw(sh, m["rest_yaw"])
            z = m["rest_h"] - GRASP_BELOW_TOP_MM
            o["pick"] = {"tcp": [_r(m["rest"][0]), _r(m["rest"][1]), _r(z)],
                         "approach": [_r(m["rest"][0]), _r(m["rest"][1]), _r(z + APPROACH_MM)], "yaw_deg": _r(gy)}
            if target is not None:
                rot = 0.0 if (m["rest_yaw"] is None or sh == "circle") else wrap(target["yaw"] - m["rest_yaw"], per)
                L = m["length"] if m["length"] is not None else m["rest_h"]
                zp = target["floor_h"] + L - GRASP_BELOW_TOP_MM
                za = L - GRASP_BELOW_TOP_MM + PLACE_CLEARANCE_MM + APPROACH_MM
                tx, ty = float(target["tray"][0]), float(target["tray"][1])
                o["place"] = {"tcp": [_r(tx), _r(ty), _r(zp)], "approach": [_r(tx), _r(ty), _r(za)],
                              "yaw_deg": _r(gy + rot), "rotate_by_deg": _r(rot)}
        return o


def next_peg(views):
    """The peg the robot should handle now: the one already in the gripper, else the next one on the table."""
    for st in ("carried", "occluded"):
        for v in views:
            if v["state"] == st:
                return v
    order = {c: i for i, c in enumerate(PICK_ORDER)}
    cands = [v for v in views if v["state"] == "on_table" and v["pick"] is not None and v["target_slot"] is not None]
    return min(cands, key=lambda v: order.get(v["color"], 99)) if cands else None


# ============================== v7.3: ROS 2 output ==============================
def _quat_from_R(R):
    R = np.asarray(R, np.float64)
    w = np.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = np.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2
    y = np.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2
    z = np.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2
    x = np.copysign(x, R[2, 1] - R[1, 2])
    y = np.copysign(y, R[0, 2] - R[2, 0])
    z = np.copysign(z, R[1, 0] - R[0, 1])
    q = np.array([x, y, z, w])
    return q / np.linalg.norm(q)


class RosLink:
    """Publishes on ROS 2 if rclpy is available (source your ROS setup first). Positions in metres (ROS units).
      ROS_STATE_TOPIC  std_msgs/String       JSON with tray, slots, all pegs (mm / deg, tray frame), next peg
      ROS_PICK_TOPIC   geometry_msgs/PoseStamped  frame TRAY_FRAME_ID: TCP grasp point of the next peg, yaw about z
      ROS_PLACE_TOPIC  geometry_msgs/PoseStamped  frame TRAY_FRAME_ID: TCP release point in its slot, yaw about z
      TF (static)      CAMERA_FRAME_ID -> TRAY_FRAME_ID  (and BASE_FRAME_ID -> CAMERA_FRAME_ID if BASE_T_CAMERA)
    The poses give the gripper YAW only (rotation about the tray's up axis); the gripper points straight down."""

    def __init__(self):
        self.ok, self.node = False, None
        if not ROS_ENABLE:
            return
        try:
            import rclpy
            from std_msgs.msg import String
            from geometry_msgs.msg import PoseStamped, TransformStamped
            from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
        except Exception as e:                       # no ROS in this shell: run without it
            print(f"ROS 2 not available ({e.__class__.__name__}: {e}) - running without ROS output.")
            return
        self.rclpy, self.String, self.PoseStamped, self.TransformStamped = rclpy, String, PoseStamped, TransformStamped
        rclpy.init()
        self.node = rclpy.create_node(ROS_NODE)
        self.pub_state = self.node.create_publisher(String, ROS_STATE_TOPIC, 10)
        self.pub_pick = self.node.create_publisher(PoseStamped, ROS_PICK_TOPIC, 10)
        self.pub_place = self.node.create_publisher(PoseStamped, ROS_PLACE_TOPIC, 10)
        self.tf = StaticTransformBroadcaster(self.node)
        self.ok, self.tf_key = True, None
        print(f"ROS 2: publishing {ROS_STATE_TOPIC}, {ROS_PICK_TOPIC}, {ROS_PLACE_TOPIC}, TF {CAMERA_FRAME_ID} -> "
              f"{TRAY_FRAME_ID}")

    def _stamp(self):
        return self.node.get_clock().now().to_msg()

    def _tf_msg(self, parent, child, R, t_mm):
        m = self.TransformStamped()
        m.header.stamp, m.header.frame_id, m.child_frame_id = self._stamp(), parent, child
        m.transform.translation.x, m.transform.translation.y, m.transform.translation.z = [float(v) / 1000 for v in t_mm]
        q = _quat_from_R(R)
        m.transform.rotation.x, m.transform.rotation.y, m.transform.rotation.z, m.transform.rotation.w = map(float, q)
        return m

    def publish_tf(self, geo):
        if not self.ok or geo is None:
            return
        key = tuple(np.round(np.r_[geo.R.ravel(), geo.o], 3))
        if key == self.tf_key:
            return
        self.tf_key = key
        msgs = [self._tf_msg(CAMERA_FRAME_ID, TRAY_FRAME_ID, geo.R, geo.o)]
        if BASE_T_CAMERA is not None:
            T = base_T_camera()                      # includes CALIB_OFFSET_MM
            msgs.append(self._tf_msg(BASE_FRAME_ID, CAMERA_FRAME_ID, T[:3, :3], T[:3, 3]))
        self.tf.sendTransform(msgs)

    def _pose(self, xyz_mm, yaw_deg):
        m = self.PoseStamped()
        m.header.stamp, m.header.frame_id = self._stamp(), TRAY_FRAME_ID
        m.pose.position.x, m.pose.position.y, m.pose.position.z = [float(v) / 1000 for v in xyz_mm]
        a = np.deg2rad(yaw_deg or 0.0) / 2
        m.pose.orientation.z, m.pose.orientation.w = float(np.sin(a)), float(np.cos(a))
        return m

    def publish(self, state, nxt):
        if not self.ok:
            return
        self.pub_state.publish(self.String(data=json.dumps(state)))
        if nxt is not None and nxt.get("pick"):
            self.pub_pick.publish(self._pose(nxt["pick"]["tcp"], nxt["pick"]["yaw_deg"]))
        if nxt is not None and nxt.get("place"):
            self.pub_place.publish(self._pose(nxt["place"]["tcp"], nxt["place"]["yaw_deg"]))
        self.rclpy.spin_once(self.node, timeout_sec=0.0)

    def close(self):
        if self.ok:
            self.node.destroy_node()
            self.rclpy.shutdown()


def build_state(now, geo, slots, views, nxt):
    st = {"t": round(now, 3), "frame": TRAY_FRAME_ID, "units": "mm, deg", "calibrated": geo is not None}
    if geo is not None:
        cam = lambda xyz: [_r(v) for v in geo.to_cam(xyz)]
        st["tray"] = {"length_mm": _r(geo.length), "width_mm": _r(geo.width), "camera_height_mm": _r(geo.d)}
        st["slots"] = [{"id": s["id"], "shape": s["shape"], "x": _r(s["tray"][0]), "y": _r(s["tray"][1]),
                        "yaw_deg": _r(s["yaw"]), "floor_mm": _r(s["floor_h"]), "cam_xyz": cam(s["tray"])}
                       for s in slots if "tray" in s]
        st["pegs"] = []
        for v in views:
            d = {k: v[k] for k in v if k != "last_px"}
            if v["pos"] is not None and v["pos"][2] is not None:
                d["cam_xyz"] = cam(v["pos"])
            st["pegs"].append(d)
    st["next"] = None if nxt is None else nxt["color"]
    return st


# ============================== Matching / drawing ==============================
def match_and_draw(vis, rect_view, slots, pegs, H, H_inv, stale=False, geo=None):
    # Slots drawn ORANGE when the tray itself is not detected: overlays then show the last good pose.
    col = (0, 140, 255) if stale else (0, 200, 0)
    placed, placed_r = [], []               # label boxes already drawn this frame (labels never overlap)
    for s in slots:
        pc = project(s["contour"], H_inv)
        cv2.polylines(vis, [pc.astype(np.int32)], True, col, 2)
        if len(pc) <= 6:                                       # exact vertices of the ideal slot polygon
            for v in pc.reshape(-1, 2):
                cv2.circle(vis, ip(v), 3, col, -1)
        cx, cy = project([s["cf"]], H_inv)[0, 0]
        s["img_center"] = ip((cx, cy))
        cv2.drawMarker(vis, s["img_center"], col, cv2.MARKER_CROSS, 14, 2)        # slot centroid
    for s in slots:                         # slot labels beside the slot (v7.1 drew them over the hole)
        pc = project(s["contour"], H_inv).reshape(-1, 2)
        put_label(vis, f"S{s['id']} {s['shape']}", (int(pc[:, 0].max()) + 4, int(pc[:, 1].mean()) + 5), 0.4,
                  col, placed)
        if rect_view is not None:
            cv2.polylines(rect_view, [s["contour"].astype(np.int32)], True, col, 2)
            cv2.drawMarker(rect_view, s["center"], col, cv2.MARKER_CROSS, 14, 2)
            r = s["contour"].reshape(-1, 2)
            put_label(rect_view, f"S{s['id']} {s['shape']}", (int(r[:, 0].min()), int(r[:, 1].max()) + 16), 0.4,
                      col, placed_r)

    for p in pegs:
        p.update(target=None, placed=False, aligned=False, dx_mm=None, dy_mm=None, dist_mm=None, dyaw=None)
        label = f"{p['color']} {p['shape']}"
        # grey = whole colour silhouette (reference), cyan = TOP FACE outline,
        # magenta = fitted ideal shape with its exact vertices, red = TOP-FACE centre (grasp reference)
        cv2.polylines(vis, [p["silhouette"].astype(np.int32)], True, (160, 160, 160), 1)
        cv2.polylines(vis, [p["contour"].astype(np.int32)], True, (255, 255, 0), 1)
        if p["top_ok"]:
            if p["fit"] is not None and p["fit"]["poly"] is not None and H_inv is not None and H is not None \
                    and p["rcontour"] is not None and p["shape"] != "unknown":
                fp = project(p["fit"]["poly"], H_inv)
                cv2.polylines(vis, [fp.astype(np.int32)], True, (255, 0, 255), 2)
                if len(fp) <= 6:
                    for v in fp.reshape(-1, 2):
                        cv2.circle(vis, ip(v), 3, (255, 0, 255), -1)
                if rect_view is not None:
                    cv2.polylines(rect_view, [p["fit"]["poly"].astype(np.int32)], True, (255, 0, 255), 2)
            cv2.circle(vis, p["center"], 5, (0, 0, 255), -1)
            cv2.drawMarker(vis, p["center"], (0, 0, 255), cv2.MARKER_CROSS, 16, 1)
        else:
            cv2.circle(vis, p["center"], 7, (0, 0, 255), 1)    # hollow = centre NOT reliable
        if geo is not None and p.get("top_t") is not None and H is not None:
            # v7.3: the peg's AXIS on the tray plane (top-face centre dropped vertically): parallax-free position.
            foot = geo.tray_to_px([p["top_t"][0], p["top_t"][1], 0.0])[0]
            cv2.circle(vis, ip(foot), 4, (0, 255, 255), 1)
            p["rcenter"] = ip(project([foot], H).reshape(2))
        if rect_view is not None and p["rcenter"] is not None:
            cv2.circle(rect_view, p["rcenter"], 5, (0, 0, 255), -1)

        if H is not None and (p["top_ok"] or p.get("remembered")) and p["shape"] in EXPECTED_SLOTS \
                and p["rcf"] is not None:
            same = [s for s in slots if s["shape"] == p["shape"]]
            if same:
                p["target"] = min(same, key=lambda s: np.hypot(s["cf"][0] - p["rcf"][0], s["cf"][1] - p["rcf"][1]))
        t = p["target"]
        if t is not None:
            dx, dy = t["cf"][0] - p["rcf"][0], t["cf"][1] - p["rcf"][1]       # slot - peg, tray px
            p["dx_mm"], p["dy_mm"] = dx * MM_PER_PX_X, dy * MM_PER_PX_Y
            p["dist_mm"] = float(np.hypot(p["dx_mm"], p["dy_mm"]))
            p["placed"] = cv2.pointPolygonTest(t["contour"], (float(p["rcf"][0]), float(p["rcf"][1])), False) >= 0
            if p["yaw"] is not None and p["shape"] in SYMMETRY_DEG and p["shape"] != "circle":
                per = SYMMETRY_DEG[p["shape"]]
                p["dyaw"] = float((t["yaw"] - p["yaw"] + per / 2) % per - per / 2)   # rotation still needed
            p["aligned"] = p["dist_mm"] <= ALIGN_TOL_MM and (p["dyaw"] is None or abs(p["dyaw"]) <= ALIGN_TOL_DEG)
            m3 = p.get("m3d")
            if m3 is not None and m3["dist_mm"] is not None:  # v7.3: metric, parallax-corrected (depth) values
                p["dx_mm"], p["dy_mm"], p["dist_mm"] = m3["dx_mm"], m3["dy_mm"], m3["dist_mm"]
                p["dyaw"], p["aligned"], p["placed"] = m3["rot_deg"], m3["aligned"], m3["state"] == "placed"
            if p["aligned"]:
                label += " ALIGNED"
            else:
                label += f" > S{t['id']} {p['dist_mm']:.1f}mm"
                if p["dyaw"] is not None:
                    label += f" rot={p['dyaw']:+.0f}deg"
                cv2.arrowedLine(vis, p["center"], t["img_center"], (0, 255, 255), 2)
                if rect_view is not None:
                    cv2.line(rect_view, p["rcenter"], t["center"], (0, 255, 255), 2)
        elif not p["top_ok"]:
            label += f" [no top face: {p['top_info'].get('short', '?')}]"   # no centre, no alignment
        elif p["shape"] == "unknown":
            guess = ""
            if p["fit"] is not None and p["fit"].get("best"):      # what it was closest to, and how well
                guess = f" ~{p['fit']['best']} {p['fit']['iou']:.2f}"
            label += (" ?" if p["stable"] or p["votes"] == "0/0" else " (voting)") + guess
        else:
            label += " (no slot)"

        m3 = p.get("m3d")
        if m3 is not None:
            if m3["state"] in ("occluded", "carried", "placed"):
                label = f"[{m3['state']}] " + label
            if m3["pos"] is not None and m3["pos"][2] is not None and not m3["approx"]:
                label += f" h={m3['pos'][2]:.0f}"
        x, y, w, h = cv2.boundingRect(p["contour"])
        cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 255, 255), 2)
        txt_col = (120, 255, 120) if p["aligned"] else (255, 255, 255)
        put_label(vis, label, (x, max(15, y - 8)), 0.45, txt_col, placed)
        if rect_view is not None and p["rcontour"] is not None:
            cv2.polylines(rect_view, [p["rcontour"].astype(np.int32)], True, (255, 255, 255), 1)
    return placed


def draw_memory(vis, views, slots, placed):
    """v7.3: pegs that are NOT visible right now (under / in the gripper) at their last position, with their target."""
    for v in views:
        if v["visible"] or v["last_px"] is None or v["state"] in ("missing", "new"):
            continue
        cv2.circle(vis, v["last_px"], 14, (0, 165, 255), 2)
        t = next((s for s in slots if s["id"] == v["target_slot"]), None)
        if t is not None and "img_center" in t:
            cv2.arrowedLine(vis, v["last_px"], t["img_center"], (0, 165, 255), 1)
        put_label(vis, f"[{v['state']}] {v['color']} {v['shape']}" + (f" > S{t['id']}" if t else ""),
                  (v["last_px"][0] - 20, v["last_px"][1] - 18), 0.45, (0, 200, 255), placed)


def shape_features(c):
    """Numbers behind a shape label: the new template fit and, for comparison, the old vertex counting."""
    c = np.asarray(c, np.float32).reshape(-1, 1, 2)
    area = cv2.contourArea(c)
    if area < 1:
        return {"area": 0, "shape": "unknown"}
    peri = cv2.arcLength(c, True)
    hull = cv2.convexHull(c)
    f = fit_shape(c)
    return {"area": int(area),
            "solidity": round(area / (cv2.contourArea(hull) + 1e-6), 2),
            "n@0.03": len(cv2.approxPolyDP(c, 0.03 * peri, True)),
            "fit_shape": f["shape"], "fit_iou": f["iou"], "scores": f["scores"], "margin": f["margin"],
            "squash": f["aspect"], "ripple": f["ripple"],
            "yaw_deg": round(f["yaw"], 1), "legacy_shape": classify_shape_legacy(c)}


def debug_dump(slots, pegs, vis, rect_view, tray_ok, raw=None):
    """Press 'd'. Prints classifier inputs for every slot and peg and saves both views."""
    print("\n===== DEBUG =====  tray pose:", "LIVE" if tray_ok else "STALE (rect-space values unreliable)")
    for s_ in slots:
        print(f" slot S{s_['id']}", shape_features(s_["contour"]))
    for p_ in pegs:
        print(f" peg {p_['color']}  top_face_found={p_['top_ok']}  label_shape={p_['shape']}")
        print("   top-face check:", p_["top_info"])
        if raw is not None:      # HSV of the pixels inside the colour blob: are they near a COLORS boundary?
            hsv_ = cv2.cvtColor(raw, cv2.COLOR_BGR2HSV)
            mk = filled(p_["silhouette"], raw.shape[:2])
            px = hsv_[mk > 0]
            if px.size:
                q = lambda v: [int(np.percentile(px[:, v], t)) for t in (5, 50, 95)]
                print(f"   HSV 5/50/95 pct  H={q(0)}  S={q(1)}  V={q(2)}")
        print("   image-space:", shape_features(p_["contour"]))
        if p_["rcontour"] is not None:
            print("   rect-space :", shape_features(p_["rcontour"]))
    cv2.imwrite("debug_view.png", vis)
    if rect_view is not None:
        cv2.imwrite("debug_rect.png", rect_view)
    print("saved debug_view.png / debug_rect.png\n")


def print_slot_coords(slots, geo, H_inv, why=""):
    """Slot centroid coordinates in the terminal: printed automatically when the slots are found ('b' / 'r') or the
    tray pose changes, and on key 's'. Centre = centre of the hole opening on the tray's top surface."""
    print(f"==== SLOT COORDINATES{' (' + why + ')' if why else ''} ====")
    if not slots:
        print("  no slots found yet (press 'b' with the empty tray in view)")
        return
    if geo is None:
        print("  slot  shape      pixel (u, v)      [press 'b' for mm coordinates]")
    else:
        print("  frames: tray = x along tray, y across, z up (mm, origin = tray corner on its top surface); "
              f"camera = {CAMERA_FRAME_ID} (mm)" + (f"; base = {BASE_FRAME_ID} (mm)" if BASE_T_CAMERA is not None else ""))
        print("  slot  shape      pixel (u, v)    tray x      y      z   |  camera X      Y      Z"
              + ("   |  base X      Y      Z" if BASE_T_CAMERA is not None else "") + "   |  yaw   floor")
    for s_ in slots:
        if geo is not None and "tray" in s_:
            Pt = np.asarray(s_["tray"], np.float64)
            uv = geo.tray_to_px(Pt)[0]
            Pc = geo.to_cam(Pt)
            line = (f"  S{s_['id']:<3d} {s_['shape']:9s} ({uv[0]:6.1f}, {uv[1]:6.1f})   {Pt[0]:7.1f} {Pt[1]:6.1f} {Pt[2]:6.1f}"
                    f"   | {Pc[0]:7.1f} {Pc[1]:6.1f} {Pc[2]:6.1f}")
            if BASE_T_CAMERA is not None:
                T = base_T_camera()                  # includes CALIB_OFFSET_MM
                Pb = T[:3, :3] @ Pc + T[:3, 3]
                line += f"   | {Pb[0]:7.1f} {Pb[1]:6.1f} {Pb[2]:6.1f}"
            line += f"   | {s_['yaw']:5.1f} {s_['floor_h']:+6.1f}"
        else:
            uv = project([s_["cf"]], H_inv).reshape(2)
            line = f"  S{s_['id']:<3d} {s_['shape']:9s} ({uv[0]:6.1f}, {uv[1]:6.1f})"
        print(line)


def print_report(slots, pegs, tray_z, intr, geo=None, views=()):
    fmt = lambda v: None if v is None else [round(a, 1) for a in v]
    if geo is not None:                       # v7.3: metric report in the tray frame (what the robot gets)
        print("==== ROBOT VIEW (tray frame: x along tray, y across, z up; mm, deg) ====")
        print(" ", geo.summary())
        for s_ in slots:
            if "tray" in s_:
                print(f"  S{s_['id']} {s_['shape']:9s} x={s_['tray'][0]:7.1f} y={s_['tray'][1]:7.1f}  "
                      f"yaw={s_['yaw']:6.1f}  floor={s_['floor_h']:+.1f}")
        for v in views:
            print(f"  {v['color']:7s} {str(v['shape']):9s} [{v['state']}] pos={v['pos']} yaw={v['yaw_deg']} "
                  f"len={v['length_mm']} -> S{v['target_slot']} dx={v['dx_mm']} dy={v['dy_mm']} "
                  f"dist={v['dist_mm']} rot={v['rot_deg']}{' ALIGNED' if v['aligned'] else ''}")
            if v["pick"]:
                print(f"           pick  tcp={v['pick']['tcp']} approach={v['pick']['approach']} "
                      f"gripper_yaw={v['pick']['yaw_deg']}")
            if v["place"]:
                print(f"           place tcp={v['place']['tcp']} approach={v['place']['approach']} "
                      f"gripper_yaw={v['place']['yaw_deg']} (rotate {v['place']['rotate_by_deg']:+.1f})")
    print("---- slots (camera frame, mm) ----")
    for s in slots:
        print(f"S{s['id']} {s['shape']:9s} fit_iou={s['iou']} yaw={s['yaw']:.0f}deg "
              f"xyz={fmt(to_xyz(intr, s.get('img_center', (0, 0)), tray_z))}")
    print("---- pegs (top face) ----")
    for p in pegs:
        t = p["target"]
        base = f"#{p.get('id', '?')} {p['color']:7s} {p['shape']:10s} votes={p['votes']} xyz={fmt(p['xyz'])}"
        if t is None:
            print(base + f" -> no target  top_face={p['top_ok']}")
            continue
        rot = "" if p["dyaw"] is None else f" rot={p['dyaw']:+.1f}deg"
        print(base + f" -> S{t['id']}  dx={p['dx_mm']:+.1f} dy={p['dy_mm']:+.1f} mm (tray axes) "
              f"dist={p['dist_mm']:.1f} mm{rot}  {'[ALIGNED]' if p['aligned'] else ''}")


# ============================== Camera ==============================
def capture_background(pipe, align, scale_mm, n=30):
    acc = cnt = None
    for _ in range(n):
        fr = align.process(pipe.wait_for_frames())
        d = fr.get_depth_frame()
        if not d:
            continue
        a = np.asanyarray(d.get_data()).astype(np.float32) * scale_mm
        v = (a > 0).astype(np.float32)
        acc = a if acc is None else acc + a
        cnt = v if cnt is None else cnt + v
    return acc / np.maximum(cnt, 1)


def try_homographies(c):
    try:
        return homographies(c)
    except np.linalg.LinAlgError:
        print("tray pose: singular homography, ignored")
        return None


def main():
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    profile = pipe.start(cfg)
    dev = profile.get_device()
    scale_mm = dev.first_depth_sensor().get_depth_scale() * 1000.0
    align = rs.align(rs.stream.color)
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

    for _ in range(90):                       # let auto-exposure settle
        pipe.wait_for_frames()
    if LOCK_EXPOSURE:
        try:
            cs = dev.first_color_sensor()
            cs.set_option(rs.option.enable_auto_exposure, 0)
            cs.set_option(rs.option.enable_auto_white_balance, 0)
        except Exception as e:
            print("Could not lock exposure:", e)

    bg_mm, slots, pegs = None, [], []
    geo, views, arms, monitor, ros = None, [], None, PickPlaceMonitor(), RosLink()   # v7.3
    corners_prev, H, H_inv, tray_z = None, None, None, 0.0
    move_buf, last_why, show_mask, show_pegmask = [], None, False, False
    show_rej = SHOW_REJECTED
    tracker = PegTracker()
    t0 = time.time()
    frame_no = 0
    print("Show the WHOLE empty tray, nobody in view, robot arm parked out of view -> 'b'.  'r' re-detect slots, 's' slot coordinates, 'p' report, "
          "'d' debug dump, 'm' tray-mask window, 'k' peg-colour-mask window, 'x' show rejected blobs, 'q' quit.")
    try:
        while True:
            fr = align.process(pipe.wait_for_frames())
            color, depth = fr.get_color_frame(), fr.get_depth_frame()
            if not color or not depth:
                continue
            raw = np.asanyarray(color.get_data()).copy()
            depth_mm = np.asanyarray(depth.get_data()).astype(np.float32) * scale_mm
            vis = raw.copy()
            now = time.time() - t0
            frame_no += 1

            # --- 1) hands: ignored silently (depth height map, same gate as the pegs use) ---
            # No warnings and nothing is paused. A skin-coloured raised blob entering from the image edge is masked
            # out of the pegs and the tray; raised pixels are never tray; the tray pose is kept while anything
            # (hand or robot arm) is over the tray. Hands above MAX_H are removed by the depth gate as before.
            gate, hand, tray_excl, arm_on_tray = None, None, None, False
            if bg_mm is not None:
                gate = depth_gate(depth_mm, bg_mm, raw)
                hand, arms = scan_raised(raw, gate[1])
                tray_excl = ((gate[1] > TRAY_RAISED_MM).astype(np.uint8) * 255) | hand
                if corners_prev is not None and arms.any():
                    zone = np.zeros(arms.shape, np.uint8)
                    cv2.fillConvexPoly(zone, np.round(corners_prev).astype(np.int32), 255)
                    zone = cv2.dilate(zone, np.ones((21, 21), np.uint8))
                    arm_on_tray = bool(np.count_nonzero(zone & arms))

            # --- 2) tray pose: stabilised (fixed camera => pose should not move) ---
            corners, tmask, tinfo = find_tray(raw, corners_prev, tray_excl)
            if corners is None:
                why = "tray not found"
            elif not valid_quad(corners):
                why = "tray quad degenerate (not convex / too small)"
            elif touches_border(corners, raw.shape):
                why = "tray corners touch image edge"
            else:
                why = ""
            if why != last_why and not arm_on_tray:  # print only when the CATEGORY changes; an arm/hand over
                print(f"[{now:7.1f}s] tray detection:", why or "OK", "|", tinfo)   # the tray explains a miss:
                last_why = why                                                    # stay silent
            pose_ok_now = (why == "")
            if show_mask:                        # key 'm': what the tray-colour mask really contains
                mv = cv2.cvtColor(tmask, cv2.COLOR_GRAY2BGR)
                if corners is not None:
                    cv2.polylines(mv, [corners.astype(np.int32)], True, (0, 0, 255), 2)
                cv2.imshow("Tray mask", mv)

            if pose_ok_now and not arm_on_tray:     # an arm over the tray hides part of it: keep the locked pose
                if corners_prev is None:                                  # first pose: adopt
                    hh = try_homographies(corners)
                    if hh:
                        corners_prev, (H, H_inv) = corners, hh
                else:
                    d = float(np.linalg.norm(corners - corners_prev, axis=1).mean())
                    if d <= POSE_DEADBAND_PX:                              # noise: keep the locked pose
                        move_buf = []
                    else:                                                  # maybe the tray really moved
                        move_buf = (move_buf + [corners])[-POSE_MOVE_FRAMES:]
                        if len(move_buf) == POSE_MOVE_FRAMES and \
                                float(np.stack(move_buf).std(axis=0).max()) <= POSE_MOVE_SPREAD_PX:
                            new_c = np.mean(move_buf, axis=0).astype(np.float32)
                            hh = try_homographies(new_c)
                            if hh:
                                corners_prev, (H, H_inv) = new_c, hh
                                print(f"[{now:7.1f}s] INFO: tray moved ({d:.1f}px) - pose updated")
                                if geo is not None:                   # v7.3: move the tray frame too
                                    geo.set_corners(corners_prev)
                                    geo.annotate_slots(slots, H_inv)
                                print_slot_coords(slots, geo, H_inv, "tray moved")
                            move_buf = []
                zs = depth_mm[(tmask > 0) & (depth_mm > 0)]
                tray_z = float(np.median(zs)) if zs.size else tray_z
            if corners_prev is not None:
                cv2.polylines(vis, [corners_prev.astype(np.int32)], True, (255, 255, 0), 2)
            tray_ok = H is not None and pose_ok_now and not arm_on_tray
            # orange slots only when the tray is lost for no visible reason; while an arm/hand covers it the locked
            # pose is still valid (fixed camera, fixed tray), so the slots stay green
            stale = not (H is not None and (pose_ok_now or arm_on_tray))

            # --- 3) pegs: detect, track, vote (keeps running while a hand is in view; the hand is just masked) ---
            rect_view = cv2.warpPerspective(raw, H, (TRAY_W, TRAY_H)) if H is not None else None
            known = {c for c, l in tracker.labels().items() if l} if PEG_COLORS_UNIQUE else set()
            pegs = detect_pegs(raw, depth_mm, bg_mm, intr, H, gate, do_fit=(frame_no % FIT_EVERY == 0),
                               ignore=hand, geo=geo, known=known)
            tracker.frozen = monitor.frozen()    # v7.3: no shape votes while the gripper handles a peg
            tracker.update(pegs)
            views = monitor.update(pegs, slots, geo, arms if bg_mm is not None else None, now, tracker.labels())
            for c_ in monitor.reset:             # peg released by the gripper: re-measure its rotation
                tracker.reset_yaw(c_)
            monitor.reset.clear()
            nxt = next_peg(views)
            if hand is not None and hand.any():  # thin grey outline: where a hand is being ignored (no warning)
                hc, _ = cv2.findContours(hand, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(vis, hc, -1, (150, 150, 150), 1)
            if H is not None:
                placed_lbl = match_and_draw(vis, rect_view, slots, pegs, H, H_inv, stale=stale, geo=geo)
                draw_memory(vis, views, slots, placed_lbl)
            ros.publish_tf(geo)                  # v7.3: ROS 2 (no-op without ROS)
            ros.publish(build_state(now, geo, slots, views, nxt), nxt)

            for rc_, nm_, a_, sol_ in (REJECTED if show_rej else []):  # discarded colour blobs, and why (key 'x')
                cv2.polylines(vis, [rc_.astype(np.int32)], True, (0, 0, 255), 1)
                bx, by, _, _ = cv2.boundingRect(rc_)
                put_label(vis, f"rejected {nm_} {int(a_)}px sol={sol_:.2f}", (bx, by + 12), 0.4, (110, 110, 255))
            if show_pegmask and PEG_MASK is not None:
                cv2.imshow("Peg colour masks", PEG_MASK)
            status = ("calibrated" if bg_mm is not None else "press 'b' (empty tray, nobody in view)")
            if arm_on_tray:
                status += " | pose kept (arm over tray)"
            elif not pose_ok_now:
                status += " | pose held (tray not detected)"
            if hand is not None and hand.any():
                status += " | hand ignored"
            if nxt is not None:
                status += f" | next: {nxt['color']} ({nxt['state']})"
            if ros.ok:
                status += " | ROS"
            cv2.putText(vis, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imshow("Peg sorting", vis)
            if rect_view is not None:
                cv2.imshow("Tray (rectified)", rect_view)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("b"), ord("r")):
                if not tray_ok:
                    print("Need a live tray pose and nobody in view (no corner on the image edge).")
                    continue
                if key == ord("b"):
                    bg_mm = capture_background(pipe, align, scale_mm)
                    tracker, monitor = PegTracker(), PickPlaceMonitor()
                    try:                                       # v7.3: metric tray frame from depth
                        geo = TrayFrame(intr, bg_mm, tmask, corners_prev)
                        print(geo.summary())
                    except ValueError as e:
                        geo = None
                        print("3D tray frame NOT built:", e)
                slots = detect_slots(raw, H)
                if geo is not None:
                    geo.annotate_slots(slots, H_inv)
                print_slot_coords(slots, geo, H_inv, "slots detected")
            elif key == ord("p"):
                print_report(slots, pegs, tray_z, intr, geo, views)
            elif key == ord("s"):
                print_slot_coords(slots, geo, H_inv)
            elif key == ord("d"):
                debug_dump(slots, pegs, vis, rect_view, tray_ok, raw)
            elif key == ord("x"):
                show_rej = not show_rej
            elif key == ord("k"):
                show_pegmask = not show_pegmask
                if not show_pegmask:
                    cv2.destroyWindow("Peg colour masks")
            elif key == ord("m"):
                show_mask = not show_mask
                if not show_mask:
                    cv2.destroyWindow("Tray mask")
            elif key == ord("q"):
                break
    finally:
        pipe.stop()
        cv2.destroyAllWindows()
        ros.close()


if __name__ == "__main__":
    main()
