import csv
import os

import numpy as np
import mujoco
from pathlib import Path

from utils import hat, get_body_orientation, is_close_to_identity

MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "asymmetric_body.xml"
RESULTS_CSV = "skew_identity_residuals.csv"

N_CHECKS_PER_STEP = 5
N_LOGGED_STEPS = 5
STEPS_BETWEEN_LOGS = 200


def random_unit_angular_velocity(rng):
    w = rng.normal(size=3)
    return 2.0 * w / np.linalg.norm(w)


def check_identities(R, rng):
    R = np.asarray(R, dtype=float)

    max_residual_cross = 0.0
    max_residual_skew = 0.0

    for _ in range(N_CHECKS_PER_STEP):
        v = rng.normal(size=3)
        w = rng.normal(size=3)
        omega = rng.normal(size=3)

        lhs_cross = R @ np.cross(v, w)
        rhs_cross = np.cross(R @ v, R @ w)
        resid_cross = np.max(np.abs(lhs_cross - rhs_cross))

        lhs_skew = R @ hat(omega) @ R.T
        rhs_skew = hat(R @ omega)
        resid_skew = np.max(np.abs(lhs_skew - rhs_skew))

        max_residual_cross = max(max_residual_cross, resid_cross)
        max_residual_skew = max(max_residual_skew, resid_skew)

    return max_residual_cross, max_residual_skew


def save_residuals(rows, path=RESULTS_CSV):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "t_s", "max_resid_cross", "max_resid_skew"])
        for step, t, rc, rs in rows:
            writer.writerow([step, f"{t:.6f}", f"{rc:.6e}", f"{rs:.6e}"])
    return os.path.abspath(path)


def main():
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    rng = np.random.default_rng(seed=0)

    data.qvel[3:6] = random_unit_angular_velocity(rng)
    mujoco.mj_forward(model, data)

    results = []

    print(f"{'step':>5} {'t (s)':>8} {'max resid: R(vxw)=(Rv)x(Rw)':>28} {'max resid: RwR^T=(Rw)^':>24}")
    for log_i in range(N_LOGGED_STEPS):
        for _ in range(STEPS_BETWEEN_LOGS):
            mujoco.mj_step(model, data)

        R = get_body_orientation(data)
        assert is_close_to_identity(R @ R.T, tol=1e-6), "R is not orthonormal!"

        resid_cross, resid_skew = check_identities(R, rng)
        print(f"{log_i:5d} {data.time:8.3f} {resid_cross:28.3e} {resid_skew:24.3e}")
        results.append((log_i, data.time, resid_cross, resid_skew))

    csv_path = save_residuals(results)

    worst_cross = max(r[2] for r in results)
    worst_skew = max(r[3] for r in results)
    eps = np.finfo(float).eps
    print()
    print(f"Saved residuals to: {csv_path}")
    print(f"Worst-case residual, R(vxw)=(Rv)x(Rw): {worst_cross:.3e}  "
          f"(~{worst_cross / eps:.1f} x machine eps = {eps:.3e})")
    print(f"Worst-case residual, R w^ R^T=(Rw)^  : {worst_skew:.3e}  "
          f"(~{worst_skew / eps:.1f} x machine eps)")

if __name__ == "__main__":
    main()
