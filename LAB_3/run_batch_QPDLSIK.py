import argparse
import ast
import csv
import logging
import os
import sys
import time
from datetime import datetime

import mujoco
import mujoco.viewer
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # find Fr_QPDLSIK.py next to this file
import Fr_QPDLSIK as sim

# Outcome thresholds
LIFT_MIN = 0.05            # m, cube must rise at least this much to count as lifted
IK_POS_ACCEPT = 1e-3       # m, waypoint IK position error above this is flagged
IK_ROT_ACCEPT = np.radians(1.0)
TRAY_REST_TOL = 0.01       # m, cube centre must be within this of its resting height in the tray
UPRIGHT_MIN = 0.95         # cos of the cube's tilt from vertical (~18 deg)

# Columns of the original batch_results.csv, followed by a few extra diagnostics
CSV_COLUMNS = [
    "index", "status", "flag",
    "cube_x", "cube_y", "cube_z", "cube_qw", "cube_qx", "cube_qy", "cube_qz",
    "frame_x", "frame_y", "frame_z",
    "pos_err_pre_pick_hover_mm", "pos_err_pick_mm",
    "pos_err_pre_place_hover_mm", "pos_err_place_mm",
    "ik_total_ms", "sim_time_s", "wall_time_s",
    # extras
    "final_cube_x", "final_cube_y", "final_cube_z", "place_xy_err_mm", "max_lift_mm",
]
IK_COLS = {"Pre-Pick Hover": "pos_err_pre_pick_hover_mm", "Pick": "pos_err_pick_mm",
           "Pre-Place Hover": "pos_err_pre_place_hover_mm", "Place": "pos_err_place_mm"}


def load_pose_file(path):
    """Read a '{index: [values...]}' file, ignoring '#' comments."""
    with open(sim.resolve_path(path)) as f:
        text = "\n".join(line.split("#", 1)[0] for line in f)
    return {int(k): [float(x) for x in v] for k, v in ast.literal_eval(text).items()}


def setup_logging(log_path):
    logger = logging.getLogger("batch")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s  %(message)s", "%H:%M:%S")
    for h in (logging.FileHandler(log_path, mode="w"), logging.StreamHandler(sys.stdout)):
        h.setFormatter(fmt)
        logger.addHandler(h)
    return logger


def evaluate(robot, plan, result):
    """Return (status, flags) for one episode."""
    flags = []
    for label, info in plan["ik"].items():
        if info["pos_err"] > IK_POS_ACCEPT or info["rot_err"] > IK_ROT_ACCEPT:
            flags.append(f"ik_not_converged_{label.lower().replace(' ', '_').replace('-', '_')}")

    if result["max_lift"] < LIFT_MIN:
        flags.append("not_lifted")

    pos, quat = result["cube_pos"], result["cube_quat"]
    tray = robot.model.body_pos[robot.tray_id]
    rest_z = plan["tray_floor_top"] + robot.cube_half
    in_tray = (abs(pos[0] - tray[0]) < robot.tray_inner_half
               and abs(pos[1] - tray[1]) < robot.tray_inner_half
               and abs(pos[2] - rest_z) < TRAY_REST_TOL)
    if not in_tray:
        flags.append("not_in_tray")
    else:
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, quat)
        if R[8] < UPRIGHT_MIN:          # cube's local z vs world z
            flags.append("tilted")

    if result["status"] == "complete":
        status = "success" if not flags else "failed"
    else:
        status = result["status"]        # "viewer_closed" or "timeout"
        flags.append("incomplete")
    return status, flags


def make_row(idx, cube, frame, plan, result, status, flags):
    tray = np.asarray(frame)
    pos = result["cube_pos"]
    row = {
        "index": idx, "status": status, "flag": ";".join(flags) if flags else "none",
        "cube_x": cube[0], "cube_y": cube[1], "cube_z": cube[2],
        "cube_qw": cube[3], "cube_qx": cube[4], "cube_qy": cube[5], "cube_qz": cube[6],
        "frame_x": frame[0], "frame_y": frame[1], "frame_z": frame[2],
        "ik_total_ms": round(plan["ik_total"] * 1000, 2),
        "sim_time_s": round(result["sim_time"], 3),
        "wall_time_s": round(result["wall_time"], 3),
        "final_cube_x": round(pos[0], 4), "final_cube_y": round(pos[1], 4),
        "final_cube_z": round(pos[2], 4),
        "place_xy_err_mm": round(np.linalg.norm(pos[:2] - tray[:2]) * 1000, 2),
        "max_lift_mm": round(result["max_lift"] * 1000, 1),
    }
    for label, col in IK_COLS.items():
        row[col] = round(plan["ik"][label]["pos_err"] * 1000, 3)
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=sim.MODEL_PATH)
    ap.add_argument("--cubes", default="cube_poses.txt")
    ap.add_argument("--frames", default="frame_poses.txt")
    ap.add_argument("--out", default="batch_runs", help="parent folder for this run's outputs")
    ap.add_argument("--indices", type=int, nargs="*", help="run only these indices")
    ap.add_argument("--fast", action="store_true", help="don't throttle the viewer to real time")
    ap.add_argument("--hold", type=float, default=1.0,
                    help="seconds to stay at home after each episode before the next one")
    ap.add_argument("--timeout", type=float, default=sim.EPISODE_TIMEOUT_S,
                    help="simulated seconds before an episode counts as stuck")
    args = ap.parse_args()

    run_dir = os.path.join(args.out, "batch_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    csv_path = os.path.join(run_dir, "batch_results.csv")
    log = setup_logging(os.path.join(run_dir, "batch.log")).info

    cubes = load_pose_file(args.cubes)
    frames = load_pose_file(args.frames)
    common = sorted(set(cubes) & set(frames))
    if set(cubes) != set(frames):
        log(f"WARNING: indices only in one file are skipped: {sorted(set(cubes) ^ set(frames))}")
    indices = [i for i in (args.indices if args.indices else common) if i in common]

    robot = sim.Robot(args.model)
    log(f"Model:  {os.path.abspath(robot.model_path)}")
    log(f"Poses:  {args.cubes} ({len(cubes)}), {args.frames} ({len(frames)})")
    log(f"Running {len(indices)} episodes: {indices}")
    log(f"Mode:   {'fast' if args.fast else 'real-time'} viewer, timeout {args.timeout:.0f} s")
    log(f"Output: {os.path.abspath(run_dir)}")

    rows = []
    batch_t0 = time.perf_counter()
    with open(csv_path, "w", newline="") as fcsv, \
            mujoco.viewer.launch_passive(robot.model, robot.data) as viewer:
        writer = csv.DictWriter(fcsv, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        fcsv.flush()

        for n, idx in enumerate(indices, 1):
            if not viewer.is_running():
                log(f"Viewer closed - skipping remaining episodes {indices[n - 1:]}")
                break
            cube, frame = cubes[idx], frames[idx]
            log("")
            log("-" * 70)
            log(f"Episode {n}/{len(indices)}  (index {idx})")
            log(f"Cube  pose : {cube}")
            log(f"Frame pos  : {frame}")

            with viewer.lock():
                robot.set_scene(cube, frame)
            plan = sim.plan_path(robot, log=log)
            result = sim.run_sequence(robot, viewer, plan, realtime=not args.fast,
                                      hold=args.hold, timeout=args.timeout, log=log)
            status, flags = evaluate(robot, plan, result)

            for name, dt in result["step_times"]:
                log(f"   step {name:<18s} {dt:7.3f} s")
            row = make_row(idx, cube, frame, plan, result, status, flags)
            log(f"Result: {status.upper()}  flags={row['flag']}  "
                f"sim={row['sim_time_s']} s  wall={row['wall_time_s']} s  "
                f"IK={row['ik_total_ms']} ms  final cube={result['cube_pos'].round(4)}  "
                f"xy err to tray centre={row['place_xy_err_mm']} mm")
            writer.writerow(row)
            fcsv.flush()
            rows.append(row)

            if status == "viewer_closed":
                log(f"Viewer closed - skipping remaining episodes {indices[n:]}")
                break

    # ---- Summary ----
    log("")
    log("=" * 70)
    log(" BATCH SUMMARY")
    log("=" * 70)
    ok = [r for r in rows if r["status"] == "success"]
    log(f" Episodes run      : {len(rows)} / {len(indices)}")
    log(f" Success           : {len(ok)}")
    for st in ("failed", "timeout", "viewer_closed"):
        bad = [r["index"] for r in rows if r["status"] == st]
        if bad:
            log(f" {st:<18s}: {len(bad)}  -> indices {bad}")
    if rows:
        ik = np.array([r["ik_total_ms"] for r in rows])
        st = np.array([r["sim_time_s"] for r in rows])
        xy = np.array([r["place_xy_err_mm"] for r in ok]) if ok else None
        log(f" IK total (ms)     : mean {ik.mean():.2f}  min {ik.min():.2f}  max {ik.max():.2f}")
        log(f" Sim time (s)      : mean {st.mean():.3f}  min {st.min():.3f}  max {st.max():.3f}")
        if xy is not None:
            log(f" Place xy err (mm) : mean {xy.mean():.2f}  max {xy.max():.2f}  (successful episodes)")
    log(f" Batch wall time   : {time.perf_counter() - batch_t0:.1f} s")
    log(f" CSV               : {os.path.abspath(csv_path)}")
    log("=" * 70)


if __name__ == "__main__":
    main()
