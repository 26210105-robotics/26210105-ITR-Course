"""
02_verify_skew_properties.py -- HW1 Part 2, Task 2: verify the
skew-symmetric identities from Problem 5 in simulation.

STARTER CODE. Model loading and a simple "spin the body" simulation
loop are provided and working. Your job is to fill in the TODOs to:

  1. Log R(t), the body's rotation matrix, at several simulated
     time steps while it spins.
  2. At each logged time step, numerically check, for several
     random v, w, omega in R^3:
         R (v x w) == (R v) x (R w)                 [Problem 5a]
         R w^ R^T  == (R w)^                         [Problem 5b, No-AI on paper]
     using utils.hat() for the ^ operator.
  3. Print the residual (it should be ~1e-14, machine precision)
     and explain in your write-up why a small-but-nonzero residual
     doesn't fully validate the identity, while a residual near
     machine epsilon strongly supports it.

Note: you already proved these identities by hand in Problem 5.
This script is not a substitute for that proof -- it's a numerical
sanity check, and a chance to see *why* proofs and simulation are
complementary, not interchangeable.
"""

import csv
import os

import numpy as np
import mujoco

from utils import hat, get_body_orientation, is_close_to_identity

MODEL_PATH = "/home/pammu/Documents/Mtech/ME639/hw01-mujoco-rotations/model/asymmetric_body.xml"
RESULTS_CSV = "skew_identity_residuals.csv"

N_CHECKS_PER_STEP = 5     # how many random (v, w, omega) triples per logged step
N_LOGGED_STEPS = 5        # how many simulated time points to check
STEPS_BETWEEN_LOGS = 200  # sim steps to advance between each logged check


def random_unit_angular_velocity(rng):
    """A random constant angular velocity vector (rad/s), used to spin
    the body between logged checks."""
    w = rng.normal(size=3)
    return 2.0 * w / np.linalg.norm(w)


def check_identities(R, rng):
    """Numerically check the two Problem 5 identities for a given R.

    For N_CHECKS_PER_STEP random vectors v, w, omega (drawn with
    rng.normal), compute
        R @ np.cross(v, w)      vs.   np.cross(R @ v, R @ w)      [5a]
        R @ hat(omega) @ R.T    vs.   hat(R @ omega)              [5b]
    and return the worst-case (max) residual across all checks, for
    each identity separately. The residual is the max-abs entry of the
    difference between left- and right-hand sides.

    Return: (max_residual_cross, max_residual_skew)
    """
    R = np.asarray(R, dtype=float)

    max_residual_cross = 0.0
    max_residual_skew = 0.0

    for _ in range(N_CHECKS_PER_STEP):
        v = rng.normal(size=3)
        w = rng.normal(size=3)
        omega = rng.normal(size=3)

        # [5a]  R (v x w) == (R v) x (R w)
        lhs_cross = R @ np.cross(v, w)
        rhs_cross = np.cross(R @ v, R @ w)
        resid_cross = np.max(np.abs(lhs_cross - rhs_cross))

        # [5b]  R w^ R^T == (R w)^
        lhs_skew = R @ hat(omega) @ R.T
        rhs_skew = hat(R @ omega)
        resid_skew = np.max(np.abs(lhs_skew - rhs_skew))

        max_residual_cross = max(max_residual_cross, resid_cross)
        max_residual_skew = max(max_residual_skew, resid_skew)

    return max_residual_cross, max_residual_skew


def save_residuals(rows, path=RESULTS_CSV):
    """Write the logged residuals to a CSV file for the write-up.

    rows: list of (step, time, resid_cross, resid_skew) tuples.
    """
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "t_s", "max_resid_cross", "max_resid_skew"])
        for step, t, rc, rs in rows:
            writer.writerow([step, f"{t:.6f}", f"{rc:.6e}", f"{rs:.6e}"])
    return os.path.abspath(path)


def main():
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    rng = np.random.default_rng(seed=0)

    # Spin the body with a fixed angular velocity by directly setting
    # qvel's angular part (indices 3:6 for a freejoint) and stepping.
    data.qvel[3:6] = random_unit_angular_velocity(rng)
    mujoco.mj_forward(model, data)

    results = []  # (step, time, resid_cross, resid_skew) per logged step

    print(f"{'step':>5} {'t (s)':>8} {'max resid: R(vxw)=(Rv)x(Rw)':>28} {'max resid: RwR^T=(Rw)^':>24}")
    for log_i in range(N_LOGGED_STEPS):
        for _ in range(STEPS_BETWEEN_LOGS):
            mujoco.mj_step(model, data)

        R = get_body_orientation(data)
        # Sanity check that R is actually a valid rotation matrix
        # (this should hold to numerical precision -- if it doesn't,
        # something upstream is wrong before you even get to the
        # identities below).
        assert is_close_to_identity(R @ R.T, tol=1e-6), "R is not orthonormal!"

        resid_cross, resid_skew = check_identities(R, rng)
        print(f"{log_i:5d} {data.time:8.3f} {resid_cross:28.3e} {resid_skew:24.3e}")
        results.append((log_i, data.time, resid_cross, resid_skew))

    # Save the residuals for the write-up.
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

    # Answer to the HW1 Problem 8 question ("why doesn't a small nonzero
    # residual fully prove the identity?").
    print("""
Why a small-but-nonzero residual does NOT prove the identity:
  * A numerical test only samples finitely many (R, v, w, omega); an
    identity is a statement about ALL of them. Passing 25 random cases
    is evidence, not proof -- a counterexample could live in a region
    we never sampled (e.g. a specific R or nearly-parallel v, w).
  * Floating-point arithmetic is not exact. A nonzero residual means we
    cannot distinguish "identity holds exactly, plus round-off" from
    "identity is violated by an amount comparable to round-off". Any
    tolerance we pick is a judgement call, not a mathematical fact.
  * The R we test comes from MuJoCo's integrated quaternion, so it is
    itself only orthonormal to ~1e-6..1e-15; we test a perturbed R,
    not the exact element of SO(3) that the theorem is about.

Why a residual near machine epsilon (~1e-16 relative) STRONGLY supports it:
  * Each side of the identities involves only a handful of multiply-adds
    on O(1) numbers, so if the identity holds exactly, the observed
    difference must be a few ULPs -- exactly what we see (~1e-15..1e-16).
  * If the identity were false, the mismatch would be an O(1) function
    of (R, v, w, omega), generically far larger than 1e-15, and would
    vary with the inputs rather than sit at the round-off floor for every
    random draw and every time step.
  * So: near-epsilon residual across many random inputs => the identity
    almost certainly holds. The hand proof from Problem 5 is what turns
    "almost certainly" into "certainly" for every R in SO(3).
""")


if __name__ == "__main__":
    main()
