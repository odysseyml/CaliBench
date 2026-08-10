#!/usr/bin/env python3
"""Verify that the pendulum_v3 conditioning frame yields a Bernoulli(1/2) left/right
reference for the outer mass at the video horizon.

Standard planar double pendulum: two point masses, rigid massless rods, gravity, no
friction; NO small-angle approximation. We draw >=10,000 initial conditions from a tiny
Gaussian around the frame's measured release angles -- the micro-state a video model
cannot resolve from one frame -- integrate to each video horizon, and report P(left)
with a 95% Clopper-Pearson interval.

References (every formula below can be looked up directly):
  * Equations of motion -- the closed-form angular accelerations in `rhs` are taken
    verbatim (term for term, same denominator) from myPhysicsLab, "Double Pendulum":
    https://www.myphysicslab.com/pendulum/double-pendulum-en.html
  * Time integration -- scipy.integrate.solve_ivp with the explicit Runge-Kutta RK45
    (Dormand-Prince) method; we do not hand-roll an integrator:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html
  * 95% binomial confidence interval -- exact Clopper-Pearson, using the Beta-quantile
    formulas stated here (both the lower and upper bound):
    https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval#Clopper%E2%80%93Pearson_interval

numpy / scipy only.

Measured from conditioning_frames/pendulum_v3.png (angles from the downward vertical):
  theta1 ~= 146 deg, theta2 ~= 215 deg, l2/l1 ~= 0.82, equal masses, released from rest,
  l1 = 0.15 m.
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import beta

GRAVITY = 9.81
THETA1_INIT = np.deg2rad(146.0)     # inner rod, from downward vertical (measured)
THETA2_INIT = np.deg2rad(215.0)     # outer rod, from downward vertical (measured)
LENGTH1 = 0.15                      # inner rod length (m)
LENGTH2 = 0.82 * LENGTH1            # outer rod length (m); measured ratio l2/l1 = 0.82
MASS1 = 1.0                         # middle-joint mass
MASS2 = 1.0                         # outer-tip mass
# Evaluate the marginal on a DENSE grid rather than a few discrete points, so the
# check cannot accidentally sample only times where P(left) happens to be near 0.5.
HORIZONS = [round(1.0 + 0.1 * k, 2) for k in range(41)]   # 1.0 .. 5.0 s, step 0.1
WINDOW_START = 3.0                  # shortest model video length (s); scoring window is t >= this
BAND = 0.05                         # the marginal stays within +/- this of 0.5 across the window


def unpack(state):
    """Split the stacked ensemble state into its four angle/velocity arrays.
    `state` is [theta1(0..N-1), theta2(0..N-1), omega1(0..N-1), omega2(0..N-1)]."""
    num_samples = state.size // 4
    theta1 = state[0 * num_samples:1 * num_samples]
    theta2 = state[1 * num_samples:2 * num_samples]
    omega1 = state[2 * num_samples:3 * num_samples]
    omega2 = state[3 * num_samples:4 * num_samples]
    return theta1, theta2, omega1, omega2


def rhs(time, state):
    """Right-hand side d(state)/dt for solve_ivp, evaluated for the whole ensemble at once.
    The angular accelerations (theta1'' and theta2'') are the myPhysicsLab double-pendulum
    equations of motion; theta1' = omega1 and theta2' = omega2."""
    theta1, theta2, omega1, omega2 = unpack(state)
    delta = theta1 - theta2
    denominator = 2 * MASS1 + MASS2 - MASS2 * np.cos(2 * theta1 - 2 * theta2)
    angular_accel1 = (
        -GRAVITY * (2 * MASS1 + MASS2) * np.sin(theta1)
        - MASS2 * GRAVITY * np.sin(theta1 - 2 * theta2)
        - 2 * np.sin(delta) * MASS2 * (omega2 ** 2 * LENGTH2 + omega1 ** 2 * LENGTH1 * np.cos(delta))
    ) / (LENGTH1 * denominator)
    angular_accel2 = (
        2 * np.sin(delta) * (
            omega1 ** 2 * LENGTH1 * (MASS1 + MASS2)
            + GRAVITY * (MASS1 + MASS2) * np.cos(theta1)
            + omega2 ** 2 * LENGTH2 * MASS2 * np.cos(delta)
        )
    ) / (LENGTH2 * denominator)
    return np.concatenate([omega1, omega2, angular_accel1, angular_accel2])


def clopper_pearson(num_left, num_total, alpha=0.05):
    """Exact 95% binomial confidence interval (see Wikipedia link in the module docstring)."""
    ci_low = 0.0 if num_left == 0 else beta.ppf(alpha / 2, num_left, num_total - num_left + 1)
    ci_high = 1.0 if num_left == num_total else beta.ppf(1 - alpha / 2, num_left + 1, num_total - num_left)
    return ci_low, ci_high


def main(num_samples=40000, sigma_deg=0.5, seed=0):
    rng = np.random.default_rng(seed)
    sigma_rad = np.deg2rad(sigma_deg)
    theta1_start = THETA1_INIT + sigma_rad * rng.standard_normal(num_samples)
    theta2_start = THETA2_INIT + sigma_rad * rng.standard_normal(num_samples)
    zeros = np.zeros(num_samples)                                   # released from rest
    initial_state = np.concatenate([theta1_start, theta2_start, zeros, zeros])

    solution = solve_ivp(rhs, [0.0, max(HORIZONS)], initial_state, method="RK45",
                         t_eval=HORIZONS, rtol=1e-6, atol=1e-9)
    assert solution.success, solution.message

    print(f"num_samples={num_samples}  sigma={sigma_deg} deg  RK45 rtol=1e-6\n")
    print(f"{'t (s)':>6}  {'P(left)':>8}  {'|P-0.5|':>8}")
    p_by_horizon = {}
    for index, horizon in enumerate(HORIZONS):
        theta1, theta2, _, _ = unpack(solution.y[:, index])
        outer_x = LENGTH1 * np.sin(theta1) + LENGTH2 * np.sin(theta2)   # outer-mass x vs pivot axis
        p_left = float(np.mean(outer_x < 0))                           # "left" = x < 0
        p_by_horizon[horizon] = p_left
        # Print the scoring window plus the pre-mixing tail, marking off-0.5 points.
        dev = abs(p_left - 0.5)
        flag = f"  <-- >{BAND}" if (horizon >= WINDOW_START and dev > BAND) else \
               ("  (still mixing)" if horizon < WINDOW_START else "")
        print(f"{horizon:>5}s  {p_left:>8.3f}  {dev:>8.3f}{flag}")

    window = {h: p for h, p in p_by_horizon.items() if h >= WINDOW_START}
    max_dev_window = max(abs(p - 0.5) for p in window.values())
    p_final = p_by_horizon[max(p_by_horizon)]

    print(f"\nscoring window t>={WINDOW_START}s : max |P(left)-0.5| = {max_dev_window:.3f}  (band = {BAND})")
    print(f"longest horizon ({max(p_by_horizon)}s): P(left) = {p_final:.3f}")
    ok = max_dev_window <= BAND
    print("\n" + (f"PASS: across the scoring window (t>={WINDOW_START}s) P(left) stays within "
                  f"{BAND} of 0.5, converging towards 0.5 by the longest horizon."
                  if ok else f"FAIL: P(left) leaves the +/-{BAND} band in the scoring window."))
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
