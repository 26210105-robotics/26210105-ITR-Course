#!/usr/bin/env python3
"""
NODE 1 - camera side of the camera <-> OpenMANIPULATOR-X calibration (RealSense + ArUco board, eye-to-hand).
Works together with NODE 2 = omx_teleop_calib.py (your keyboard teleop, extended).

START (3 terminals, ROS 2 sourced in each; the peg detector must NOT run - it would hold the camera):
  T1: ros2 launch open_manipulator_x_bringup hardware.launch.py          (as you do now)
  T2: python3 calib_camera_node.py                                        (node 1: camera window)
  T3: python3 omx_teleop_calib.py --ros-args -p tip_offset_mm:="[0.0, 0.0, 0.0]"   (node 2: keyboard)

SEQUENCE for every point - all ENTERs can be pressed in the ROBOT terminal (node 2):
  1. jog the gripper TIP onto the CENTRE of a marker            -> ENTER  (node 2 sends the tip position)
  2. jog the robot back OUT OF THE CAMERA VIEW                   -> ENTER  (node 1 captures the board)
  3. in the CAMERA window: drag a box around the marker you touched, ENTER/SPACE   (manual crop)
     the marker centre becomes a 3D camera point (depth)         -> ENTER = keep (r = select again, s = skip)
  robot terminal: u = undo last point, c = compute & finish.   camera window: same keys, q = quit.
  Points are saved after each one (--out); the solution and its error are printed from 3 points on.

TOPICS (standard messages, no extra package):
  /calib/state        std_msgs/String             1 -> 2   current step (tells node 2 what ENTER does)
  /calib/robot_point  geometry_msgs/PointStamped  2 -> 1   gripper tip, robot base frame (m), sent on ENTER
  /calib/capture      std_msgs/Empty              2 -> 1   robot is out of view: capture now
  /calib/command      std_msgs/String             2 -> 1   keep / undo / compute / measure / quit
  /calib/tip          geometry_msgs/PointStamped  2 -> 1   live tip at 10 Hz (used by --verify)

RESULT: BASE_T_CAMERA (paste into peg_sorting_detector_v7_3.py, with BASE_FRAME_ID = your base frame, 'world') or a
static_transform_publisher command (use one of the two). Camera frame = colour optical frame, 640x480, mm - the
same as the peg detector.

OTHER MODES
  python3 calib_camera_node.py --verify omx_calibration.json   live check + fine offset (node 2 running):
        red cross = robot tip via calibration; x/X y/Y z/Z nudge, [ ] step, 0 reset, S save, ENTER (or ENTER in the
        robot terminal with the tip on a marker centre) = measured error + suggested correction, q quit
  python3 calib_camera_node.py --solve omx_calibration.json [--offset DX DY DZ]   recompute (no ROS needed)
"""
import argparse
import json
import threading
import time

import cv2
import numpy as np

CAMERA_FRAME_ID = "camera_color_optical_frame"
WIN = "Calibration camera (node 1)"
DICTS = ["DICT_4X4_50", "DICT_5X5_100", "DICT_6X6_250", "DICT_7X7_250", "DICT_ARUCO_ORIGINAL",
         "DICT_APRILTAG_36h11", "DICT_APRILTAG_25h9"]


# ============================== camera ==============================
class Camera:
    """RealSense colour + depth aligned to colour, same settings as the peg detector."""

    def __init__(self):
        import pyrealsense2 as rs
        self.rs = rs
        self.pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.cfg = cfg
        prof = self.pipe.start(cfg)
        self.scale_mm = prof.get_device().first_depth_sensor().get_depth_scale() * 1000.0
        self.align = rs.align(rs.stream.color)
        i = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.fx, self.fy, self.ppx, self.ppy = float(i.fx), float(i.fy), float(i.ppx), float(i.ppy)
        for _ in range(60):                       # auto exposure settles
            self.pipe.wait_for_frames()

    def grab(self):
        """Next colour + depth frame. If the camera stops sending (USB hiccup), the stream is restarted."""
        for attempt in range(1, 4):
            try:
                while True:
                    fr = self.align.process(self.pipe.wait_for_frames(3000))
                    c, d = fr.get_color_frame(), fr.get_depth_frame()
                    if c and d:
                        return (np.asanyarray(c.get_data()).copy(),
                                np.asanyarray(d.get_data()).astype(np.float32) * self.scale_mm)
            except RuntimeError as e:
                print(f"  camera: no frame for 3 s ({e}) - restarting the stream (attempt {attempt}/3) ...")
                self.restart()
        raise SystemExit("The RealSense keeps stopping. Unplug and re-plug it into a USB 3 port (blue) directly on "
                         "the PC, with the original or a short USB 3 cable, then start again.")

    def restart(self):
        try:
            self.pipe.stop()
        except RuntimeError:
            pass
        time.sleep(1.0)
        try:
            self.pipe.start(self.cfg)
            for _ in range(15):
                self.pipe.wait_for_frames(5000)
            print("  camera: stream restarted.")
        except RuntimeError as e:
            print(f"  camera: restart failed ({e})")

    def capture(self, n=15):
        """Last colour frame + depth averaged over n frames (zeros = no reading, ignored)."""
        acc = cnt = None
        for _ in range(n):
            color, d = self.grab()
            v = (d > 0).astype(np.float32)
            acc = d if acc is None else acc + d
            cnt = v if cnt is None else cnt + v
        return color, np.where(cnt > 0, acc / np.maximum(cnt, 1), 0).astype(np.float32)

    def deproject(self, u, v, z):
        z = np.asarray(z, np.float64)
        return np.stack([(np.asarray(u, np.float64) - self.ppx) / self.fx * z,
                         (np.asarray(v, np.float64) - self.ppy) / self.fy * z, z], -1)

    def project(self, P):
        P = np.asarray(P, np.float64).reshape(-1, 3)
        return np.stack([P[:, 0] / P[:, 2] * self.fx + self.ppx, P[:, 1] / P[:, 2] * self.fy + self.ppy], -1)

    def stop(self):
        self.pipe.stop()


# ============================== ArUco ==============================
class Aruco:
    def __init__(self, name):
        self.ok = hasattr(cv2, "aruco")
        self.name = name
        if not self.ok:
            print("cv2.aruco not available (pip install opencv-contrib-python) - you will click the marker centre.")
            return
        # works with the old ArUco API (OpenCV <= 4.6, e.g. Ubuntu 22.04 / ROS Humble) and the new one (>= 4.7)
        if hasattr(cv2.aruco, "getPredefinedDictionary"):
            d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))
        else:
            d = cv2.aruco.Dictionary_get(getattr(cv2.aruco, name))
        if hasattr(cv2.aruco, "DetectorParameters_create"):     # old API
            p = cv2.aruco.DetectorParameters_create()
        else:
            p = cv2.aruco.DetectorParameters()
        p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        if hasattr(cv2.aruco, "ArucoDetector"):           # OpenCV >= 4.7
            det = cv2.aruco.ArucoDetector(d, p)
            self._detect = lambda g: det.detectMarkers(g)
        else:
            self._detect = lambda g: cv2.aruco.detectMarkers(g, d, parameters=p)

    def detect(self, img):
        """-> list of (id, 4x2 corners)"""
        if not self.ok:
            return []
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        corners, ids, _ = self._detect(g)
        if ids is None:
            return []
        return [(int(i), np.asarray(c, np.float64).reshape(4, 2)) for i, c in zip(ids.ravel(), corners)]


def choose_dictionary(cam, forced):
    if forced:
        return Aruco(forced)
    if not hasattr(cv2, "aruco"):
        return Aruco("DICT_4X4_50")
    imgs = [cam.grab()[0] for _ in range(5)]
    best, best_n = "DICT_4X4_50", 0
    for name in DICTS:
        if not hasattr(cv2.aruco, name):
            continue
        a = Aruco(name)
        n = max(len(a.detect(im)) for im in imgs)
        if n > best_n:
            best, best_n = name, n
    print(f"ArUco dictionary: {best} ({best_n} markers visible)" if best_n else
          "No ArUco markers seen yet - using DICT_4X4_50 (force another with --dict). You can still click points.")
    return Aruco(best)



def quat_to_R(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def R_to_quat(R):
    w = np.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = np.copysign(np.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = np.copysign(np.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = np.copysign(np.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    q = np.array([x, y, z, w])
    return q / np.linalg.norm(q)


# ============================== 3D point of the touched marker ==============================
def point_3d(cam, depth, px, region_poly):
    """Camera point (mm) at pixel px on the surface: plane fitted to the depth inside region_poly, then the ray
    through px meets that plane (sub-mm; one noisy depth pixel cannot spoil it)."""
    h, w = depth.shape
    m = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m, [np.round(region_poly).astype(np.int32)], 1)
    v, u = np.nonzero((m > 0) & (depth > 0))
    if len(u) < 20:
        return None, "too few depth pixels on the marker"
    P = cam.deproject(u, v, depth[v, u])
    keep = np.ones(len(P), bool)
    for _ in range(3):
        c = P[keep].mean(0)
        n = np.linalg.svd(P[keep] - c, full_matrices=False)[2][-1]
        r = np.abs((P - c) @ n)
        keep = r < max(1.5, 3 * np.median(r[keep]))
    c = P[keep].mean(0)
    n = np.linalg.svd(P[keep] - c, full_matrices=False)[2][-1]
    ray = np.array([(px[0] - cam.ppx) / cam.fx, (px[1] - cam.ppy) / cam.fy, 1.0])
    t = (n @ c) / (n @ ray)
    rms = float(np.sqrt(np.mean(((P[keep] - c) @ n) ** 2)))
    return ray * t, f"plane from {int(keep.sum())} depth px, flatness {rms:.1f} mm"


# ============================== solving ==============================
def rigid_fit(A, B):
    """R, t with B ~ R A + t (least squares, no scale)."""
    A, B = np.asarray(A, np.float64), np.asarray(B, np.float64)
    ca, cb = A.mean(0), B.mean(0)
    U, S, Vt = np.linalg.svd((A - ca).T @ (B - cb))
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, cb - R @ ca


def solve(pairs, base_frame, verbose=True):
    cam = np.array([p["camera_mm"] for p in pairs])
    rob = np.array([p["robot_mm"] for p in pairs])
    R, t = rigid_fit(cam, rob)
    res = np.linalg.norm(cam @ R.T + t - rob, axis=1)
    loo = []
    if len(pairs) >= 4:
        for i in range(len(pairs)):
            k = np.arange(len(pairs)) != i
            Ri, ti = rigid_fit(cam[k], rob[k])
            loo.append(float(np.linalg.norm(Ri @ cam[i] + ti - rob[i])))
    sv = np.linalg.svd(cam - cam.mean(0), compute_uv=False)
    out = {"R": R.tolist(), "t_mm": t.tolist(), "rms_mm": float(np.sqrt(np.mean(res ** 2))),
           "max_mm": float(res.max()), "loo_rms_mm": float(np.sqrt(np.mean(np.square(loo)))) if loo else None,
           "residuals_mm": res.tolist(), "spread_mm": sv.tolist()}
    if verbose:
        print(f"\n---- solution from {len(pairs)} points ----")
        for p, r in zip(pairs, res):
            print(f"  point {p['n']:2d} (marker {p['marker']}): error {r:5.1f} mm" + ("   <-- check this one" if r > max(
                3 * np.median(res), 5) else ""))
        print(f"  RMS {out['rms_mm']:.1f} mm, max {out['max_mm']:.1f} mm" +
              (f", leave-one-out RMS {out['loo_rms_mm']:.1f} mm (realistic accuracy)" if loo else ""))
        if sv[1] < 0.2 * sv[0]:
            print("  WARNING: the points are almost on a LINE - spread them over the workspace.")
        elif sv[2] < 0.05 * sv[0]:
            print("  note: all points lie in one plane (OK). Height (z) accuracy improves with 1-2 points at another "
                  "height (board on a block).")
    return R, t, out


def report(R, t, base_frame, camera_frame, offset=(0.0, 0.0, 0.0)):
    """offset: fine correction in the robot base frame (mm), added to every position (see --offset / --verify)."""
    offset = np.asarray(offset, np.float64)
    t = np.asarray(t, np.float64) + offset
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    q = R_to_quat(R)
    print("\n==================== CALIBRATION RESULT ====================")
    if np.any(offset != 0):
        print(f"(includes the fine offset x={offset[0]:+.1f} y={offset[1]:+.1f} z={offset[2]:+.1f} mm, robot base frame)")
    print(f"Camera pose in the robot base frame '{base_frame}':")
    print(f"  camera position: x={t[0]:.1f} y={t[1]:.1f} z={t[2]:.1f} mm")
    look = R[:, 2]
    print(f"  camera looks along ({look[0]:+.2f}, {look[1]:+.2f}, {look[2]:+.2f}) "
          f"= {np.degrees(np.arcsin(-look[2])):.0f} deg below horizontal (if the base z axis is up)")
    print("\n1) Paste into peg_sorting_detector_v7_3.py (replace the line 'BASE_T_CAMERA = None'):")
    rows = ",\n                 ".join("[" + ", ".join(f"{v:.6f}" if j < 3 else f"{v:.2f}" for j, v in enumerate(r)) + "]"
                                       for r in T)
    print(f"BASE_T_CAMERA = [{rows}]")
    print(f'BASE_FRAME_ID = "{base_frame}"')
    print("CALIB_OFFSET_MM = [0.0, 0.0, 0.0]   # already contained in BASE_T_CAMERA above; use it for later tweaks")
    print("\n2) OR publish it yourself (do not do both - two publishers of one TF conflict):")
    print(f"ros2 run tf2_ros static_transform_publisher --x {t[0] / 1000:.5f} --y {t[1] / 1000:.5f} "
          f"--z {t[2] / 1000:.5f} --qx {q[0]:.6f} --qy {q[1]:.6f} --qz {q[2]:.6f} --qw {q[3]:.6f} "
          f"--frame-id {base_frame} --child-frame-id {camera_frame}")
    print("=============================================================")
    return T


def save(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ============================== drawing / waiting ==============================
def overlay(img, lines, markers=(), pts=(), cam=None):
    vis = img.copy()
    for mid, c in markers:
        cv2.polylines(vis, [np.round(c).astype(np.int32)], True, (0, 255, 0), 2)
        ctr = c.mean(0)
        cv2.drawMarker(vis, (int(ctr[0]), int(ctr[1])), (0, 255, 0), cv2.MARKER_CROSS, 12, 1)
        cv2.putText(vis, str(mid), (int(c[0, 0]), int(c[0, 1]) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    if cam is not None:
        for p in pts:                                      # already recorded points
            uv = cam.project(p["camera_mm"])[0]
            cv2.circle(vis, (int(uv[0]), int(uv[1])), 6, (0, 0, 255), 2)
            cv2.putText(vis, f"P{p['n']}", (int(uv[0]) + 7, int(uv[1]) + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (0, 0, 255), 1)
    y = 22
    for ln in lines:
        (tw, th), _ = cv2.getTextSize(ln, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(vis, (6, y - th - 4), (14 + tw, y + 5), (30, 30, 30), -1)
        cv2.putText(vis, ln, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
        y += th + 12
    return vis



def select_marker(cam, aruco, color, depth):
    """Manual crop -> marker in the crop (or a clicked point) -> 3D camera point. Returns dict or None."""
    shown = overlay(color, ["Drag a box around the marker you touched, then ENTER/SPACE (c = cancel)"],
                    aruco.detect(color))
    roi = cv2.selectROI(WIN, shown, showCrosshair=True, fromCenter=False)
    x, y, w, h = [int(v) for v in roi]
    H_, W_ = color.shape[:2]                               # keep the box inside the image
    x1, y1 = min(W_, x + w), min(H_, y + h)
    x, y = max(0, x), max(0, y)
    w, h = x1 - x, y1 - y
    if w < 5 or h < 5:
        return None
    pad = 10
    x0, y0 = max(0, x - pad), max(0, y - pad)
    crop = color[y0:min(color.shape[0], y + h + pad), x0:min(color.shape[1], x + w + pad)]
    found = [(i, c + [x0, y0]) for i, c in aruco.detect(crop)]
    if not found:                                          # try the full image, keep markers inside the box
        found = [(i, c) for i, c in aruco.detect(color)
                 if x <= c[:, 0].mean() <= x + w and y <= c[:, 1].mean() <= y + h]
    if found:
        ctr_box = np.array([x + w / 2, y + h / 2])
        mid, c = min(found, key=lambda f: np.linalg.norm(f[1].mean(0) - ctr_box))
        if len(found) > 1:
            print(f"  {len(found)} markers in the box - taking marker {mid} (closest to the box centre).")
        px = c.mean(0)                                     # marker centre, sub-pixel
        P, info = point_3d(cam, depth, px, c)
        return {"marker": mid, "pixel": px.tolist(), "corners": c.tolist(), "camera_mm": None if P is None else
                P.tolist(), "info": info}
    # ---- no marker found: click the touched point in a zoomed crop ----
    print("  no ArUco marker found in the box - click the exact point you touched (zoomed window), ENTER to accept.")
    zoom = max(1, int(400 / max(w, h)))
    big = cv2.resize(color[y:y + h, x:x + w], None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST)
    click = {}
    cv2.namedWindow("click the point")
    cv2.setMouseCallback("click the point", lambda e, u, v, *a: click.update(p=(u, v)) if e == cv2.EVENT_LBUTTONDOWN else None)
    while True:
        b = big.copy()
        if "p" in click:
            cv2.drawMarker(b, click["p"], (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
        cv2.imshow("click the point", b)
        k = cv2.waitKey(20) & 0xFF
        if k in (10, 13) and "p" in click:
            break
        if k == ord("c"):
            cv2.destroyWindow("click the point")
            return None
    cv2.destroyWindow("click the point")
    px = np.array([x + (click["p"][0] + 0.5) / zoom - 0.5, y + (click["p"][1] + 0.5) / zoom - 0.5])
    r = 12
    disk = np.array([[px[0] + r * np.cos(a), px[1] + r * np.sin(a)] for a in np.linspace(0, 2 * np.pi, 24)])
    P, info = point_3d(cam, depth, px, disk)
    return {"marker": "clicked", "pixel": px.tolist(), "corners": None,
            "camera_mm": None if P is None else P.tolist(), "info": info}



def marker_size_check(cam, sel, size_mm):
    """Cross-check depth against the printed marker size (both ways should agree within ~1-2 %)."""
    c = np.asarray(sel["corners"])
    K = np.array([[cam.fx, 0, cam.ppx], [0, cam.fy, cam.ppy], [0, 0, 1]])
    s = size_mm / 2
    obj = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], np.float64)
    flag = getattr(cv2, "SOLVEPNP_IPPE_SQUARE", cv2.SOLVEPNP_ITERATIVE)
    ok, rvec, tvec = cv2.solvePnP(obj, c, K, None, flags=flag)
    if not ok:
        return "PnP failed"
    d = np.linalg.norm(tvec.ravel()) - np.linalg.norm(sel["camera_mm"])
    return f"distance from marker size vs depth differs by {d:+.1f} mm"



def solve_file(path, offset=None):
    """Recompute from saved points. offset None = keep the offset stored in the file."""
    with open(path) as f:
        data = json.load(f)
    if offset is None:
        offset = data.get("offset_mm", [0.0, 0.0, 0.0])
    R, t, out = solve(data["pairs"], data["base_frame"])
    T = report(R, t, data["base_frame"], data.get("camera_frame", CAMERA_FRAME_ID), offset)
    data["result"], data["BASE_T_CAMERA_mm"], data["offset_mm"] = out, T.tolist(), list(map(float, offset))
    data["BASE_T_CAMERA_raw_mm"] = np.vstack([np.column_stack([R, t]), [0, 0, 0, 1]]).tolist()
    save(path, data)


# ============================== ROS 2 node ==============================
def make_node():
    import rclpy
    from geometry_msgs.msg import PointStamped
    from rclpy.node import Node
    from std_msgs.msg import Empty, String

    class CameraCalibNode(Node):
        """Only messaging lives here; the camera window runs in the main thread."""

        def __init__(self):
            super().__init__("calib_camera")
            self.lock = threading.Lock()
            self.events, self.state = [], "STARTING"
            self.tip, self.tip_frame, self.tip_t = None, None, 0.0
            self.tips = []                                    # last ~1 s of tips (averaged for measurements)
            self.String = String
            self.pub_state = self.create_publisher(String, "/calib/state", 10)
            self.create_subscription(PointStamped, "/calib/robot_point", self._point, 10)
            self.create_subscription(Empty, "/calib/capture", lambda m: self._ev("capture", None), 10)
            self.create_subscription(String, "/calib/command", lambda m: self._ev("cmd", m.data.strip().lower()), 10)
            self.create_subscription(PointStamped, "/calib/tip", self._tip, 10)
            self.create_timer(0.5, self._pub_state)

        def _point(self, m):
            self._ev("point", (np.array([m.point.x, m.point.y, m.point.z]) * 1000.0, m.header.frame_id))

        def _tip(self, m):
            with self.lock:
                self.tip = np.array([m.point.x, m.point.y, m.point.z]) * 1000.0
                self.tip_frame, self.tip_t = m.header.frame_id, time.time()
                self.tips = (self.tips + [(self.tip_t, self.tip)])[-10:]

        def _ev(self, kind, val):
            with self.lock:
                self.events.append((kind, val))

        def pop(self):
            with self.lock:
                return self.events.pop(0) if self.events else None

        def live_tip(self):
            with self.lock:
                return (self.tip.copy(), self.tip_frame) if self.tip is not None and time.time() - self.tip_t < 1.0 \
                    else (None, None)

        def mean_tip(self):
            """Average of the tips received in the last second (the arm is held still while measuring)."""
            with self.lock:
                P = [p for t, p in self.tips if time.time() - t < 1.2]
            return np.mean(P, axis=0) if P else None

        def set_state(self, s):
            if s != self.state:
                self.state = s
                self._pub_state()

        def _pub_state(self):
            self.pub_state.publish(self.String(data=self.state))

    rclpy.init()
    return rclpy, CameraCalibNode()


KEYS = {10: "enter", 13: "enter", ord("u"): "undo", ord("c"): "compute", ord("q"): "quit", ord("r"): "reselect",
        ord("s"): "skip"}


def preflight():
    """Clear messages for the usual start-up problems (instead of a long traceback)."""
    import os
    print(f"OpenCV {cv2.__version__}, aruco module: {'yes' if hasattr(cv2, 'aruco') else 'NO'}, numpy {np.__version__}")
    if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        print("WARNING: no DISPLAY - the camera window cannot open. In Docker: run with -e DISPLAY=$DISPLAY "
              "-v /tmp/.X11-unix:/tmp/.X11-unix and 'xhost +local:' on the host.")
    try:
        import pyrealsense2 as rs
    except ImportError as e:
        raise SystemExit(f"pyrealsense2 not importable ({e}) - pip install pyrealsense2")
    devs = rs.context().query_devices()
    if len(devs) == 0:
        raise SystemExit("No RealSense camera found. Check the USB cable (USB 3 port); in Docker pass the camera in "
                         "(--privileged -v /dev:/dev). Also close realsense-viewer / the peg detector.")
    d = devs[0]
    usb = ""
    if hasattr(rs.camera_info, "usb_type_descriptor") and d.supports(rs.camera_info.usb_type_descriptor):
        usb = d.get_info(rs.camera_info.usb_type_descriptor)
    print(f"RealSense: {d.get_info(rs.camera_info.name)}  serial {d.get_info(rs.camera_info.serial_number)}"
          + (f"  USB {usb}" if usb else ""))
    if usb.startswith("2"):
        print("WARNING: the camera runs on USB 2 - frames can stop arriving. Use a USB 3 (blue) port and a USB 3 "
              "cable, no hub.")


def open_camera():
    try:
        return Camera()
    except RuntimeError as e:
        raise SystemExit(f"Could not start the RealSense stream: {e}\n -> is another program using the camera "
                         f"(peg detector, realsense-viewer, a ROS realsense node)? Close it and retry.")


def calibrate(node, args):
    preflight()
    cam = open_camera()
    aruco = choose_dictionary(cam, args.dict)
    data = {"base_frame": args.base_frame, "camera_frame": CAMERA_FRAME_ID, "dictionary": aruco.name,
            "camera_intrinsics": [cam.fx, cam.fy, cam.ppx, cam.ppy], "offset_mm": list(map(float, args.offset)),
            "pairs": []}
    if args.resume:
        with open(args.resume) as f:
            data["pairs"] = json.load(f)["pairs"]
        print(f"Resumed {len(data['pairs'])} points from {args.resume}")
    pairs = data["pairs"]
    state, p_rob, color, depth, sel, conf_vis = "WAIT_POINT", None, None, None, None, None
    shown_state = None
    cv2.namedWindow(WIN)
    print("\nTape the board down. 4+ points over the board; better 8-12 with the board at 2-3 places (one raised).")
    try:
        while True:
            n = (max(p["n"] for p in pairs) + 1) if pairs else 1
            node.set_state(f"{state} n={n}")
            if state != shown_state:
                shown_state = state
                print({"WAIT_POINT": f"\n[point {n}] STEP 1: jog the gripper TIP onto the CENTRE of a marker, "
                                     f"then ENTER in the ROBOT terminal.",
                       "WAIT_CAPTURE": f"[point {n}] STEP 2: jog the robot OUT OF VIEW, then ENTER (robot terminal "
                                       f"or this window).",
                       "SELECT": f"[point {n}] STEP 3: in the camera window drag a box around the marker you "
                                 f"touched, ENTER/SPACE.",
                       "CONFIRM": f"[point {n}] ENTER = keep, r = select again, s = skip this point."}[state])
            ev = node.pop()
            k = 255
            if state in ("WAIT_POINT", "WAIT_CAPTURE"):
                img, _ = cam.grab()
                lines = ([f"POINT {n} - STEP 1/3: tip on a marker centre, ENTER in the ROBOT terminal"]
                         if state == "WAIT_POINT" else
                         [f"POINT {n} - STEP 2/3: robot OUT OF VIEW, then ENTER",
                          f"tip: {p_rob[0]:.1f} {p_rob[1]:.1f} {p_rob[2]:.1f} mm"])
                lines.append(f"{len(pairs)} points  |  u undo, c compute & finish, q quit")
                cv2.imshow(WIN, overlay(img, lines, aruco.detect(img), pairs, cam))
                k = cv2.waitKey(20) & 0xFF
            elif state == "CONFIRM":
                cv2.imshow(WIN, conf_vis)
                k = cv2.waitKey(20) & 0xFF
            else:
                time.sleep(0.01)
            cmd = KEYS.get(k) if k != 255 else None
            if ev is not None and ev[0] == "cmd":
                cmd = {"keep": "enter", "measure": "enter"}.get(ev[1], ev[1])
            if cmd == "quit":
                break
            if cmd == "compute":
                if len(pairs) < 3:
                    print("  need at least 3 points (4+ recommended).")
                else:
                    finish(pairs, data, args)
                    node.set_state("DONE")
                    time.sleep(1.0)
                    break
                continue
            if cmd == "undo" and state in ("WAIT_POINT", "WAIT_CAPTURE"):
                if state == "WAIT_CAPTURE":
                    print("  tip position discarded.")
                elif pairs:
                    print(f"  removed point {pairs.pop()['n']}")
                    save(args.out, data)
                state = "WAIT_POINT"
                continue

            if state == "WAIT_POINT":
                if ev is not None and ev[0] == "point":
                    p_rob, frame = ev[1]
                    if frame and frame != data["base_frame"]:
                        print(f"  note: robot points are in '{frame}' - using that as the base frame.")
                        data["base_frame"] = frame
                    print(f"  tip received: x={p_rob[0]:.1f} y={p_rob[1]:.1f} z={p_rob[2]:.1f} mm ({frame})")
                    state = "WAIT_CAPTURE"
                elif cmd == "enter":
                    print("  press ENTER in the ROBOT terminal (node 2) - it sends the tip position.")
            elif state == "WAIT_CAPTURE":
                if ev is not None and ev[0] == "point":
                    p_rob = ev[1][0]
                    print(f"  tip position updated: {p_rob[0]:.1f} {p_rob[1]:.1f} {p_rob[2]:.1f} mm")
                elif (ev is not None and ev[0] == "capture") or cmd == "enter":
                    color, depth = cam.capture(args.frames)
                    print("  board captured.")
                    state = "SELECT"
            elif state == "SELECT":
                node._pub_state()
                sel = select_marker(cam, aruco, color, depth)            # blocks until the box is drawn
                while node.pop() is not None:                             # drop ENTERs pressed meanwhile
                    pass
                if sel is not None and sel["camera_mm"] is not None:
                    P = np.array(sel["camera_mm"])
                    print(f"  marker {sel['marker']}: pixel ({sel['pixel'][0]:.1f}, {sel['pixel'][1]:.1f}) -> camera "
                          f"x={P[0]:.1f} y={P[1]:.1f} z={P[2]:.1f} mm ({sel['info']})")
                    if args.marker_mm and sel["corners"] is not None:
                        print("  size check:", marker_size_check(cam, sel, args.marker_mm))
                    conf_vis = overlay(color, [f"POINT {n} - marker {sel['marker']}",
                                               "ENTER = keep, r = select again, s = skip"])
                    cv2.circle(conf_vis, (int(sel["pixel"][0]), int(sel["pixel"][1])), 6, (0, 0, 255), 2)
                else:
                    if sel is not None:
                        print("  no depth there:", sel["info"])
                    sel = None
                    conf_vis = overlay(color, ["nothing selected: ENTER or r = select again, s = skip this point"])
                state = "CONFIRM"
            elif state == "CONFIRM":
                if cmd == "enter":
                    if sel is None:
                        state = "SELECT"
                        continue
                    pairs.append({"n": n, "marker": sel["marker"], "robot_mm": [float(v) for v in p_rob],
                                  "camera_mm": sel["camera_mm"], "pixel": sel["pixel"], "t": time.time()})
                    save(args.out, data)
                    print(f"  saved point {n} ({len(pairs)} total) -> {args.out}")
                    if len(pairs) >= 3:
                        _, _, out = solve(pairs, data["base_frame"])
                        data["result"] = out
                        save(args.out, data)
                    else:
                        print(f"  {3 - len(pairs)} more point(s) before the first solution (4+ recommended).")
                    state = "WAIT_POINT"
                elif cmd == "reselect":
                    state = "SELECT"
                elif cmd == "skip":
                    print("  point skipped.")
                    state = "WAIT_POINT"
    finally:
        cv2.destroyAllWindows()
        cam.stop()


def finish(pairs, data, args):
    R, t, out = solve(pairs, data["base_frame"])
    data["offset_mm"] = list(map(float, args.offset))
    T = report(R, t, data["base_frame"], CAMERA_FRAME_ID, args.offset)
    data["result"], data["BASE_T_CAMERA_mm"] = out, T.tolist()
    data["BASE_T_CAMERA_raw_mm"] = np.vstack([np.column_stack([R, t]), [0, 0, 0, 1]]).tolist()
    save(args.out, data)
    print(f"saved to {args.out}")


def verify(node, args):
    """Live check + fine offset, using the live tip from node 2 (/calib/tip)."""
    with open(args.verify) as f:
        data = json.load(f)
    raw = data.get("BASE_T_CAMERA_raw_mm")
    if raw is None:
        raw = np.array(data["BASE_T_CAMERA_mm"])
        raw[:3, 3] -= np.asarray(data.get("offset_mm", [0, 0, 0]))
    raw = np.array(raw, np.float64)
    off = np.array(data.get("offset_mm", [0.0, 0.0, 0.0]), np.float64)
    steps, si = [0.5, 1.0, 2.0, 5.0], 1
    preflight()
    cam = open_camera()
    aruco = Aruco(data.get("dictionary", "DICT_4X4_50"))
    print("VERIFY - keep the robot out of view for 1 s (board reference) ...")
    color0, depth0 = cam.capture(args.frames)
    ref = [(i, c, point_3d(cam, depth0, c.mean(0), c)[0]) for i, c in aruco.detect(color0)]
    print(f"{len(ref)} markers as reference. Red cross = robot tip via the calibration.\n"
          "keys (camera window): x/X y/Y z/Z nudge offset, [ ] step, 0 reset, ENTER error, S save, q quit\n"
          "robot terminal: ENTER (tip on a marker centre) = error + suggested correction")
    node.set_state("VERIFY")

    def T_now():
        T = raw.copy()
        T[:3, 3] += off
        return T

    try:
        while True:
            img, _ = cam.grab()
            tip_b, _ = node.live_tip()
            lines = ["VERIFY: red cross = robot tip via calibration" if tip_b is not None else
                     "VERIFY: no live tip from node 2 (/calib/tip) - is omx_teleop_calib.py running?",
                     f"offset x={off[0]:+.1f} y={off[1]:+.1f} z={off[2]:+.1f} mm  step {steps[si]} mm",
                     "x/X y/Y z/Z nudge, [ ] step, 0 reset, ENTER error, S save, q quit"]
            vis = overlay(img, lines, aruco.detect(img))
            tip_c = None
            if tip_b is not None:
                Ti = np.linalg.inv(T_now())
                tip_c = Ti[:3, :3] @ tip_b + Ti[:3, 3]
                if tip_c[2] > 0:
                    uv = cam.project(tip_c)[0]
                    cv2.drawMarker(vis, (int(uv[0]), int(uv[1])), (0, 0, 255), cv2.MARKER_CROSS, 24, 2)
            cv2.imshow(WIN, vis)
            k = cv2.waitKey(20) & 0xFF
            ch = chr(k) if k != 255 else ""
            ev = node.pop()
            cmd = ev[1] if ev is not None and ev[0] == "cmd" else None
            if ch == "q" or cmd == "quit":
                break
            axis = {"x": 0, "y": 1, "z": 2}
            if ch and ch.lower() in axis:
                off[axis[ch.lower()]] += steps[si] if ch.isupper() else -steps[si]
                print(f"  offset now x={off[0]:+.1f} y={off[1]:+.1f} z={off[2]:+.1f} mm")
            elif ch and ch in "[]":
                si = max(0, si - 1) if ch == "[" else min(len(steps) - 1, si + 1)
            elif ch == "0":
                off[:] = 0
                print("  offset reset to 0")
            elif ch == "S" or cmd == "save":
                T = report(raw[:3, :3], raw[:3, 3], data["base_frame"], data.get("camera_frame", CAMERA_FRAME_ID), off)
                data["offset_mm"], data["BASE_T_CAMERA_mm"], data["BASE_T_CAMERA_raw_mm"] = off.tolist(), T.tolist(), \
                    raw.tolist()
                save(args.verify, data)
                print(f"saved to {args.verify}. For a detector that already has the old BASE_T_CAMERA, instead set:\n"
                      f"CALIB_OFFSET_MM = [{off[0]:.1f}, {off[1]:.1f}, {off[2]:.1f}]")
            elif k in (10, 13) or cmd in ("measure", "keep", "enter"):
                good = [r for r in ref if r[2] is not None]
                if tip_c is None or not good:
                    print("  no live tip or no reference markers.")
                    continue
                i, c, P = min(good, key=lambda r: np.linalg.norm(r[2] - tip_c))
                T = T_now()
                tip_m = node.mean_tip()                             # averaged over ~1 s
                d = (tip_m if tip_m is not None else tip_b) - (T[:3, :3] @ P + T[:3, 3])
                print(f"  tip vs marker {i}: {np.linalg.norm(d):.1f} mm (base dx={d[0]:+.1f} dy={d[1]:+.1f} "
                      f"dz={d[2]:+.1f}); to cancel it press/type: x {d[0]:+.1f}, y {d[1]:+.1f}, z {d[2]:+.1f}")
    finally:
        cv2.destroyAllWindows()
        cam.stop()


def main():
    ap = argparse.ArgumentParser(description="Node 1: camera side of the camera <-> OMX calibration",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--base-frame", default="world", help="robot base frame (taken from node 2's messages anyway)")
    ap.add_argument("--dict", default=None, help="ArUco dictionary, e.g. DICT_4X4_50 (default: auto-detect)")
    ap.add_argument("--marker-mm", type=float, default=None, help="printed marker side (mm): extra depth check")
    ap.add_argument("--frames", type=int, default=15, help="depth frames averaged per capture")
    ap.add_argument("--out", default="omx_calibration.json", help="points + result are saved here")
    ap.add_argument("--resume", default=None, help="continue from a saved file")
    ap.add_argument("--offset", type=float, nargs=3, default=None, metavar=("DX", "DY", "DZ"),
                    help="fine correction, robot base frame, mm (added to every position)")
    ap.add_argument("--solve", default=None, help="only recompute the result from a saved file (no ROS needed)")
    ap.add_argument("--verify", default=None, help="live check + fine offset of a saved calibration")
    args, _ = ap.parse_known_args()
    if args.solve:
        return solve_file(args.solve, args.offset)
    if args.offset is None:
        args.offset = [0.0, 0.0, 0.0]
    rclpy, node = make_node()
    from rclpy.executors import SingleThreadedExecutor
    exe = SingleThreadedExecutor()
    exe.add_node(node)
    threading.Thread(target=exe.spin, daemon=True).start()
    try:
        (verify if args.verify else calibrate)(node, args)
    finally:
        exe.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
