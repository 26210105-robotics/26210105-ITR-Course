import argparse
import csv
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time


# ------------------------------------------------------------------
# Pose files
# ------------------------------------------------------------------
def read_pose_file(path, expected_len):
    """Parse 'index: [v0, v1, ...],' lines into {index: [floats]}.
    '#' comments (full-line or inline) are ignored."""
    poses = {}
    with open(path) as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.split("#", 1)[0].strip()
            if not line or line in "{}":
                continue
            m = re.match(r"^(\d+)\s*:\s*\[([^\]]*)\]\s*,?$", line)
            if not m:
                raise ValueError(f"{path}:{lineno}: cannot parse line: {raw.rstrip()}")
            vec = [float(v) for v in m.group(2).split(",") if v.strip()]
            if len(vec) != expected_len:
                raise ValueError(f"{path}:{lineno}: expected {expected_len} values, got {len(vec)}")
            poses[int(m.group(1))] = vec
    return poses


# ------------------------------------------------------------------
# XML editing (text-level, so comments and formatting are preserved)
# ------------------------------------------------------------------
def _set_attr(tag, attr, value):
    """Replace (or add) attr="..." inside a single opening tag."""
    pat = re.compile(rf'(\s{attr}=")[^"]*(")')
    if pat.search(tag):
        return pat.sub(lambda m: m.group(1) + value + m.group(2), tag, count=1)
    return tag.replace(">", f' {attr}="{value}">', 1)


def _edit_body(xml, body_name, **attrs):
    pat = re.compile(rf'<body\s[^>]*name="{body_name}"[^>]*>')
    matches = pat.findall(xml)
    if len(matches) != 1:
        raise ValueError(f'expected exactly one <body name="{body_name}">, found {len(matches)}')
    tag = matches[0]
    new_tag = tag
    for attr, value in attrs.items():
        new_tag = _set_attr(new_tag, attr, value)
    return xml.replace(tag, new_tag, 1)


def fmt(vec):
    return " ".join(f"{v:.6g}" for v in vec)


def write_scene(xml_text, cube_pose, frame_pos):
    xml_text = _edit_body(xml_text, "cube", pos=fmt(cube_pose[:3]), quat=fmt(cube_pose[3:7]))
    xml_text = _edit_body(xml_text, "tray", pos=fmt(frame_pos[:3]))
    return xml_text


# ------------------------------------------------------------------
# Running Fr_DLSIK.py and parsing its output
# ------------------------------------------------------------------
NUM = r"([-+]?\d+(?:\.\d+)?)"
RE_POS_ERR = re.compile(rf"pos err = {NUM} mm")
RE_IK_TOTAL = re.compile(rf"Total path solving time: {NUM} ms")
RE_SIM = re.compile(rf"Simulated time\s+{NUM} s")
RE_WALL = re.compile(rf"Wall-clock time\s+{NUM} s")
IK_LABELS = ["Pre-Pick Hover", "Pick", "Pre-Place Hover", "Place"]


def _pump(stream, q):
    for line in iter(stream.readline, ""):
        q.put(line)
    q.put(None)


def run_once(python, script, timeout, log_path):
    """Run the script; stop it once its RESULTS block has been printed.
    Returns (status, parsed_metrics)."""
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    proc = subprocess.Popen([python, "-u", script], stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
    q = queue.Queue()
    threading.Thread(target=_pump, args=(proc.stdout, q), daemon=True).start()

    m = {"pos_err_mm": [], "ik_total_ms": None, "sim_time_s": None, "wall_time_s": None}
    complete = False
    rule_lines = 0           # '====' lines seen after "Sequence complete"
    status = None
    deadline = time.time() + timeout

    with open(log_path, "w") as log:
        while True:
            try:
                line = q.get(timeout=max(0.05, deadline - time.time()))
            except queue.Empty:
                status = "timeout"
                break
            if line is None:                       # process exited on its own
                proc.wait()
                status = "complete" if complete else f"exited ({proc.returncode})"
                break
            log.write(line)
            print("   | " + line, end="")

            if (r := RE_POS_ERR.search(line)):
                m["pos_err_mm"].append(float(r.group(1)))
            elif (r := RE_IK_TOTAL.search(line)):
                m["ik_total_ms"] = float(r.group(1))
            elif (r := RE_SIM.search(line)):
                m["sim_time_s"] = float(r.group(1))
            elif (r := RE_WALL.search(line)):
                m["wall_time_s"] = float(r.group(1))
            elif "Sequence complete" in line:
                complete = True
            elif complete and line.strip() and set(line.strip()) == {"="}:
                rule_lines += 1
                if rule_lines == 3:                # RESULTS block fully printed
                    status = "complete"
                    break
            if time.time() > deadline:
                status = "timeout"
                break

    if proc.poll() is None:                        # close the viewer / process
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    return status, m


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--xml", default="FrPos.xml", help="scene XML that Fr_DLSIK.py loads")
    ap.add_argument("--script", default="Fr_DLSIK.py")
    ap.add_argument("--cube-file", default="cube_poses.txt")
    ap.add_argument("--frame-file", default="frame_poses.txt")
    ap.add_argument("--indices", type=int, nargs="*", help="pose indices to run (default: all)")
    ap.add_argument("--timeout", type=float, default=120.0, help="wall-clock seconds per run")
    ap.add_argument("--max-pos-err", type=float, default=1.0,
                    help="IK position error (mm) above which a run is flagged")
    ap.add_argument("--results", default="batch_results.csv")
    ap.add_argument("--log-dir", default="batch_logs")
    ap.add_argument("--python", default=sys.executable, help="interpreter (use mjpython on macOS)")
    ap.add_argument("--keep-xml", action="store_true", help="leave the last pose in the XML")
    args = ap.parse_args()

    cubes = read_pose_file(args.cube_file, 7)
    frames = read_pose_file(args.frame_file, 3)
    common = sorted(set(cubes) & set(frames))
    missing = sorted(set(cubes) ^ set(frames))
    if missing:
        print(f"Warning: indices {missing} are in only one pose file and will be skipped")
    indices = common if not args.indices else [i for i in args.indices if i in common]
    if not indices:
        sys.exit("No pose indices to run.")

    os.makedirs(args.log_dir, exist_ok=True)
    with open(args.xml) as f:
        original_xml = f.read()
    backup = args.xml + ".bak"
    shutil.copyfile(args.xml, backup)

    fields = ["index", "status", "flag",
              "cube_x", "cube_y", "cube_z", "cube_qw", "cube_qx", "cube_qy", "cube_qz",
              "frame_x", "frame_y", "frame_z"] + \
             [f"pos_err_{l.lower().replace(' ', '_').replace('-', '_')}_mm" for l in IK_LABELS] + \
             ["ik_total_ms", "sim_time_s", "wall_time_s"]
    rows = []
    batch_t0 = time.perf_counter()

    try:
        with open(args.results, "w", newline="") as fcsv:
            writer = csv.DictWriter(fcsv, fieldnames=fields)
            writer.writeheader()

            for n, i in enumerate(indices, 1):
                cube, frame = cubes[i], frames[i]
                print(f"\n=== Run {n}/{len(indices)}  (index {i}) ===")
                print(f"   cube  pos {cube[:3]}  quat {cube[3:]}")
                print(f"   frame pos {frame}")

                with open(args.xml, "w") as f:
                    f.write(write_scene(original_xml, cube, frame))

                status, m = run_once(args.python, args.script, args.timeout,
                                     os.path.join(args.log_dir, f"run_{i:02d}.log"))
                errs = (m["pos_err_mm"] + [None] * 4)[:4]
                flag = ""
                if any(e is not None and e > args.max_pos_err for e in errs):
                    flag = "IK_ERROR"
                row = {"index": i, "status": status, "flag": flag,
                       "cube_x": cube[0], "cube_y": cube[1], "cube_z": cube[2],
                       "cube_qw": cube[3], "cube_qx": cube[4], "cube_qy": cube[5], "cube_qz": cube[6],
                       "frame_x": frame[0], "frame_y": frame[1], "frame_z": frame[2],
                       "ik_total_ms": m["ik_total_ms"], "sim_time_s": m["sim_time_s"],
                       "wall_time_s": m["wall_time_s"]}
                for label, e in zip(IK_LABELS, errs):
                    row[f"pos_err_{label.lower().replace(' ', '_').replace('-', '_')}_mm"] = e
                writer.writerow(row)
                fcsv.flush()
                rows.append(row)
                print(f"   -> {status}{'  [' + flag + ']' if flag else ''}")
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        if not args.keep_xml:
            with open(args.xml, "w") as f:
                f.write(original_xml)
            os.remove(backup)
            print(f"\nRestored original {args.xml}")
        else:
            print(f"\nLeft last pose in {args.xml}; original saved as {backup}")

    # --- Batch summary ---
    done = [r for r in rows if r["status"] == "complete"]
    print("\n" + "=" * 52)
    print(f" BATCH SUMMARY   {len(done)}/{len(rows)} runs completed")
    print("=" * 52)
    for key, unit, scale in [("ik_total_ms", "ms", 1), ("sim_time_s", "s", 1), ("wall_time_s", "s", 1)]:
        vals = [r[key] for r in done if r[key] is not None]
        if vals:
            print(f" {key:<14s} mean {sum(vals)/len(vals):9.3f} {unit}   "
                  f"min {min(vals):9.3f}   max {max(vals):9.3f}")
    bad = [r["index"] for r in rows if r["status"] != "complete" or r["flag"]]
    if bad:
        print(f" Needs a look: indices {bad}")
    print(f" Batch wall time {time.perf_counter() - batch_t0:.1f} s")
    print(f" Results: {args.results}   Logs: {args.log_dir}/")
    print("=" * 52)


if __name__ == "__main__":
    main()
