"""
S2 - Algebraic connectivity of a frequency-hopping network under partial-band jamming
====================================================================================

Purpose
-------
Put the S1 link model on a graph. N nodes are scattered at random; every pair
is a potential link whose bit error rate depends on distance (signal) and on
the jammer (J/S, band fraction rho). A link is "up" if its BER is below an
outage criterion. The resulting graph is summarised by the algebraic
connectivity lambda_2 (second-smallest eigenvalue of the Laplacian L = D - A):

    lambda_2 = 0   <=>  the graph is disconnected
    lambda_2 large  ->  robust connectivity, fast consensus

Two questions are answered numerically:
    (a) For a given jammer power, which band fraction rho hurts the network most?
    (b) Under a jammer that picks its worst rho, at what J/S does the network
        stop being connected, and how does that threshold move with N?

Model
-----
Geometry   N nodes uniform in an L x L square (fixed seed, M placements).
Signal     Eb/N0(d) = EBN0_REF - 10*n*log10(d/D_REF)     [path-loss exponent n]
Jammer     Far from all nodes, so every receiver sees the same J. Received
           J/S grows with link distance only because S falls with d:
           J/S(d) = JS_REF + 10*n*log10(d/D_REF),   Eb/N_J(d) = G_p - J/S(d)
Link BER   S1 closed-form partial-band mixture with fraction rho.
Link up    BER <= BER_OUTAGE.
Graph      Undirected, unweighted adjacency A; L = D - A; lambda_2 = eig(L)[1].

Consequence used by the script: the BER of a link increases with its length,
so "BER <= BER_OUTAGE" is the same as "d <= r", where r(J/S, rho) is the link
range. The jammer's best rho is therefore the one that minimises r. It does
not depend on N or on the placement, and it is found once per J/S value.
(See theory/s2_laplacian_lambda2.pdf, section 6. Checked below in
VERIFICATION 5 and 6.)

Treatment of rho
----------------
rho is optimised as a continuous variable, as in S1. A jammer facing N_CH
channels cannot jam less than one channel (rho >= 1/N_CH); the figures mark
that limit and the summary reports how much the threshold moves if it is
enforced.

Verification
------------
1. Complete graph K_N has lambda_2 = N.
2. lambda_2 > 0 if and only if breadth-first search finds one component
   (every graph evaluated; mismatches must be 0).
3. lambda_2 <= N/(N-1) * minimum degree (Fiedler's bound), every graph.
4. For every placement, lambda_2 never increases as J/S increases, and it is
   0 at the highest J/S.
5. Adjacency from the BER rule equals the distance rule d <= r, every graph
   of sweep (b).
6. The rho that minimises mean lambda_2 on the grid of sweep (a) is the grid
   neighbour of the rho that minimises the link range.
7. The closed-form range-power relation reproduces the exact J/S at the
   collapse thresholds within 0.5 dB.

Outputs
-------
figures/s2_lambda2_vs_rho.png       (a) lambda_2 vs rho, N = 10/20/30
figures/s2_lambda2_vs_js.png        (b) lambda_2 vs J/S under worst-case rho, threshold J/S*
figures/s2_example_topologies.png   one placement at three jamming levels
results/s2_lambda2_vs_rho.csv, results/s2_lambda2_vs_js.csv, results/s2_summary.txt

Parameters are assumed values chosen for the study (see README).
"""

import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq, minimize_scalar
from scipy.special import erfc

# ---------------------------------------------------------------------------
# Parameters (assumed values)
# ---------------------------------------------------------------------------
SEED = 2026
N_LIST = [10, 20, 30]           # network sizes to compare
M_PLACEMENTS = 500              # random placements per N (standard error of a fraction near 0.5: 0.022)
AREA_SIDE = 1000.0              # [m] square side L; results are also reported as r/L
D_REF = 100.0                   # [m] reference distance for EBN0_REF and JS_REF
PATH_LOSS_EXP = 2.0             # free-space exponent (line of sight)
EBN0_REF_DB = 27.0              # thermal Eb/N0 at D_REF: gives an unjammed link range of 1.02 km ~ L,
                                # so the unjammed network is dense but not complete
N_CH = 100                      # hop channels, G_p = 20 dB (same as S1)
BER_OUTAGE = 1e-3               # link outage criterion (same as S1)
RHO_MIN_PHYSICAL = 1.0 / N_CH   # one channel

RHO_GRID = np.logspace(-3, 0, 20)              # rho sweep for figure (a), log-spaced 0.001 .. 1
JS_REF_GRID = np.arange(-30, 7, 1.0)           # J/S at D_REF [dB], 1 dB steps
JS_REF_FOR_RHO_SWEEP_DB = -14.0                # J/S used in figure (a): inside the collapse region
N_BOOTSTRAP = 1000                             # resamples for the threshold confidence interval
WORST_CASE_CONSTANT = 0.08286                  # S1: BER_worst = 0.08286 / (Eb/N_J)

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
RES_DIR = ROOT / "results"
FIG_DIR.mkdir(exist_ok=True)
RES_DIR.mkdir(exist_ok=True)

GP_DB = 10.0 * np.log10(N_CH)


# ---------------------------------------------------------------------------
# Link model (from S1)
# ---------------------------------------------------------------------------
def db_to_linear(x_db):
    return 10.0 ** (np.asarray(x_db, dtype=float) / 10.0)


def q_func(x):
    return 0.5 * erfc(x / np.sqrt(2.0))


def ber_partial_band(ebn0_lin, ebnj_lin, rho):
    """Average BER of BPSK/FH under partial-band noise jamming (S1 closed form)."""
    n0 = 1.0 / ebn0_lin
    nj = 1.0 / ebnj_lin
    ber_clear = q_func(np.sqrt(2.0 / n0))
    ber_jammed = q_func(np.sqrt(2.0 / (n0 + nj / rho)))
    return (1.0 - rho) * ber_clear + rho * ber_jammed


def link_ber(d, js_ref_db, rho):
    """BER of a link of length d [m] for jammer power JS_REF and band fraction rho."""
    path_db = 10.0 * PATH_LOSS_EXP * np.log10(d / D_REF)
    ebn0_db = EBN0_REF_DB - path_db
    ebnj_db = GP_DB - (js_ref_db + path_db)                 # S falls with d, J does not
    return ber_partial_band(db_to_linear(ebn0_db), db_to_linear(ebnj_db), rho)


def link_range(js_ref_db, rho):
    """Largest link length [m] that still meets the outage criterion."""
    return brentq(lambda d: link_ber(d, js_ref_db, rho) - BER_OUTAGE, 1e-2, 1e5)


def worst_rho(js_ref_db, rho_min=1e-5):
    """Jammer-optimal rho for the network: the rho that minimises the link range.

    A coarse log-spaced scan locates the minimum, then a bounded search between
    its two neighbours refines it. Returns (rho_star, range_at_rho_star).
    """
    grid = np.logspace(np.log10(rho_min), 0.0, 41)
    ranges = np.array([link_range(js_ref_db, r) for r in grid])
    k = int(np.argmin(ranges))
    lo, hi = grid[max(k - 1, 0)], grid[min(k + 1, len(grid) - 1)]
    res = minimize_scalar(lambda lr: link_range(js_ref_db, 10.0 ** lr),
                          bounds=(np.log10(lo), np.log10(hi)), method="bounded",
                          options={"xatol": 1e-6})
    return 10.0 ** res.x, res.fun


def js_ref_closed_form(r):
    """Jamming-dominated approximation of the J/S_ref at which the link range equals r.

    From S1, the worst-case BER is 0.08286/(Eb/N_J); setting it equal to
    BER_OUTAGE gives the Eb/N_J of the longest surviving link, and
    Eb/N_J(d) = G_p - J/S_ref - 10 n log10(d/D_REF) converts that to J/S_ref.
    Thermal noise is neglected.
    """
    return (GP_DB - 10.0 * np.log10(WORST_CASE_CONSTANT / BER_OUTAGE)
            - 10.0 * PATH_LOSS_EXP * np.log10(r / D_REF))


# ---------------------------------------------------------------------------
# Graph model
# ---------------------------------------------------------------------------
def place_nodes(n, rng):
    """Uniform random positions in the square."""
    return rng.uniform(0.0, AREA_SIDE, size=(n, 2))


def pairwise_distance(pos):
    diff = pos[:, None, :] - pos[None, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=-1))


def adjacency(dist, js_ref_db, rho):
    """Adjacency matrix: link (i,j) is up if its BER is below the outage criterion.

    dist is the pairwise distance matrix; its diagonal is ignored. Passing
    js_ref_db = -inf removes the jammer.
    """
    d = dist.copy()
    np.fill_diagonal(d, D_REF)                                       # placeholder, overwritten below
    a = (link_ber(d, js_ref_db, rho) <= BER_OUTAGE).astype(int)
    np.fill_diagonal(a, 0)                                           # no self links
    return a


def lambda2(a):
    """Second-smallest eigenvalue of the graph Laplacian L = D - A."""
    lap = np.diag(a.sum(axis=1)) - a
    eig = np.linalg.eigvalsh(lap)                                    # ascending, symmetric matrix
    return max(eig[1], 0.0)                                          # clip round-off of order 1e-15


def is_connected_bfs(a):
    """Independent connectivity check (does not use eigenvalues)."""
    n = a.shape[0]
    seen = np.zeros(n, dtype=bool)
    stack = [0]
    seen[0] = True
    while stack:
        i = stack.pop()
        for j in np.flatnonzero(a[i]):
            if not seen[j]:
                seen[j] = True
                stack.append(j)
    return bool(seen.all())


def crossing(x, y, level):
    """First x at which the decreasing curve y(x) falls below `level` (linear interpolation)."""
    below = np.flatnonzero(y < level)
    if len(below) == 0 or below[0] == 0:
        return np.nan
    k = below[0]
    return x[k - 1] + (y[k - 1] - level) * (x[k] - x[k - 1]) / (y[k - 1] - y[k])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    rng = np.random.default_rng(SEED)
    placements = {n: [place_nodes(n, rng) for _ in range(M_PLACEMENTS)] for n in N_LIST}
    dists = {n: [pairwise_distance(p) for p in placements[n]] for n in N_LIST}

    # bookkeeping for the exact graph-theory checks
    counts = {"graphs": 0, "mismatch": 0, "bound_viol": 0}

    def evaluate(a):
        l2 = lambda2(a)
        conn = is_connected_bfs(a)
        counts["graphs"] += 1
        if (l2 > 1e-9) != conn:
            counts["mismatch"] += 1
        n = a.shape[0]
        if l2 > n / (n - 1) * a.sum(axis=1).min() + 1e-9:            # Fiedler: lambda_2 <= N/(N-1) * delta_min
            counts["bound_viol"] += 1
        return l2, conn

    # --- Verification 1: complete graph --------------------------------------
    ok1 = all(np.isclose(lambda2(np.ones((n, n), int) - np.eye(n, dtype=int)), n) for n in N_LIST)
    print(f"VERIFICATION 1 (K_N has lambda_2 = N): {'PASS' if ok1 else 'FAIL'}")

    # --- Unjammed baseline ---------------------------------------------------
    range_unjammed = link_range(-np.inf, 1.0)
    print(f"Unjammed link range: {range_unjammed:.0f} m  ({range_unjammed / AREA_SIDE:.2f} L)")
    baseline = {}
    for n in N_LIST:
        vals = [evaluate(adjacency(d, -np.inf, 1.0))[0] for d in dists[n]]
        baseline[n] = float(np.mean(vals))
        print(f"N = {n:2d}: unjammed lambda_2 (mean over {M_PLACEMENTS} placements) = {baseline[n]:.2f}")

    # --- Sweep (a): lambda_2 vs rho at fixed J/S ---------------------------
    rho_star_a, range_star_a = worst_rho(JS_REF_FOR_RHO_SWEEP_DB)
    rows_a = []
    l2_vs_rho, l2_std_vs_rho, rho_worst_grid = {}, {}, {}
    for n in N_LIST:
        means, stds = [], []
        for rho in RHO_GRID:
            vals = [evaluate(adjacency(d, JS_REF_FOR_RHO_SWEEP_DB, rho))[0] for d in dists[n]]
            means.append(np.mean(vals))
            stds.append(np.std(vals))
            rows_a.append((n, rho, means[-1], stds[-1]))
        l2_vs_rho[n], l2_std_vs_rho[n] = np.array(means), np.array(stds)
        rho_worst_grid[n] = RHO_GRID[int(np.argmin(means))]
        print(f"N = {n:2d}: at J/S_ref = {JS_REF_FOR_RHO_SWEEP_DB:.0f} dB the grid rho with the lowest mean lambda_2 "
              f"is {rho_worst_grid[n]:.4f} (range-minimising rho = {rho_star_a:.4f})")

    # --- Sweep (b): lambda_2 vs J/S under the network-worst rho -------------
    worst = [worst_rho(js) for js in JS_REF_GRID]                     # (rho*, range) per J/S, independent of N
    rho_star_b = np.array([w[0] for w in worst])
    range_b = np.array([w[1] for w in worst])

    l2_all, conn_all = {}, {}
    n_range_mismatch = 0
    for n in N_LIST:
        l2 = np.zeros((M_PLACEMENTS, len(JS_REF_GRID)))
        cn = np.zeros((M_PLACEMENTS, len(JS_REF_GRID)), dtype=bool)
        for m, d in enumerate(dists[n]):
            off_diag = ~np.eye(n, dtype=bool)
            for j, js in enumerate(JS_REF_GRID):
                a = adjacency(d, js, rho_star_b[j])
                if not np.array_equal(a[off_diag] == 1, d[off_diag] <= range_b[j]):
                    n_range_mismatch += 1
                l2[m, j], cn[m, j] = evaluate(a)
        l2_all[n], conn_all[n] = l2, cn

    # --- Thresholds, knees and confidence intervals --------------------------
    rng_boot = np.random.default_rng(SEED + 1)
    js_star, js_star_ci, js_knee, range_star, js_star_phys = {}, {}, {}, {}, {}
    for n in N_LIST:
        frac = conn_all[n].mean(axis=0)
        js_star[n] = crossing(JS_REF_GRID, frac, 0.5)
        boots = []
        for _ in range(N_BOOTSTRAP):                                  # resample the placements
            idx = rng_boot.integers(0, M_PLACEMENTS, size=M_PLACEMENTS)
            boots.append(crossing(JS_REF_GRID, conn_all[n][idx].mean(axis=0), 0.5))
        js_star_ci[n] = np.percentile(boots, [2.5, 97.5])
        js_knee[n] = crossing(JS_REF_GRID, l2_all[n].mean(axis=0) / baseline[n], 0.5)
        range_star[n] = worst_rho(js_star[n])[1]
        # same link range, but with the jammer limited to rho >= 1/N_CH
        js_star_phys[n] = brentq(lambda js: worst_rho(js, RHO_MIN_PHYSICAL)[1] - range_star[n], -40.0, 20.0)
        print(f"N = {n:2d}: J/S* = {js_star[n]:6.2f} dB  (95% CI {js_star_ci[n][0]:.2f} .. {js_star_ci[n][1]:.2f}), "
              f"link range at J/S* = {range_star[n]:.0f} m = {range_star[n] / AREA_SIDE:.3f} L, "
              f"50% knee of mean lambda_2 at {js_knee[n]:.2f} dB")

    # --- Verifications 2-7 -----------------------------------------------------
    ok2 = counts["mismatch"] == 0
    ok3 = counts["bound_viol"] == 0
    print(f"VERIFICATION 2 (lambda_2 > 0 iff BFS-connected): {'PASS' if ok2 else 'FAIL'} "
          f"({counts['mismatch']} mismatches in {counts['graphs']} graphs)")
    print(f"VERIFICATION 3 (Fiedler bound): {'PASS' if ok3 else 'FAIL'} ({counts['bound_viol']} violations)")

    ok4 = all(np.all(np.diff(l2_all[n], axis=1) <= 1e-9) and np.all(l2_all[n][:, -1] < 1e-9) for n in N_LIST)
    print(f"VERIFICATION 4 (lambda_2 non-increasing in J/S for every placement; 0 at the top): "
          f"{'PASS' if ok4 else 'FAIL'}")

    ok5 = n_range_mismatch == 0
    print(f"VERIFICATION 5 (BER rule equals distance rule d <= r): {'PASS' if ok5 else 'FAIL'} "
          f"({n_range_mismatch} mismatches)")

    k_star = int(np.argmin(np.abs(np.log10(RHO_GRID) - np.log10(rho_star_a))))
    ok6 = all(abs(int(np.argmin(l2_vs_rho[n])) - k_star) <= 1 for n in N_LIST)
    print(f"VERIFICATION 6 (lambda_2-minimising rho = range-minimising rho, within one grid step): "
          f"{'PASS' if ok6 else 'FAIL'}")

    cf = {n: js_ref_closed_form(range_star[n]) for n in N_LIST}
    ok7 = all(abs(cf[n] - js_star[n]) < 0.5 for n in N_LIST)
    print("VERIFICATION 7 (closed-form range-power relation): "
          + ", ".join(f"N={n}: {cf[n]:.2f} vs {js_star[n]:.2f} dB" for n in N_LIST)
          + f" -> {'PASS' if ok7 else 'FAIL'}")

    print("\nOVERALL:", "PASS" if all([ok1, ok2, ok3, ok4, ok5, ok6, ok7]) else "FAIL")

    # --- Save numeric results ------------------------------------------------
    with open(RES_DIR / "s2_lambda2_vs_rho.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["N", "rho", "lambda2_mean", "lambda2_std"])
        w.writerows(rows_a)
    with open(RES_DIR / "s2_lambda2_vs_js.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["N", "js_ref_db", "lambda2_mean", "lambda2_std", "connected_fraction", "worst_rho", "link_range_m"])
        for n in N_LIST:
            for j, js in enumerate(JS_REF_GRID):
                w.writerow([n, js, l2_all[n][:, j].mean(), l2_all[n][:, j].std(),
                            conn_all[n][:, j].mean(), rho_star_b[j], range_b[j]])
    summary = [f"Placements per N: {M_PLACEMENTS}; unjammed link range {range_unjammed:.0f} m; "
               f"network-worst rho at J/S_ref = {JS_REF_FOR_RHO_SWEEP_DB:.0f} dB: {rho_star_a:.5f} "
               f"(one channel = {RHO_MIN_PHYSICAL:.2f})"]
    for n in N_LIST:
        summary.append(
            f"N={n}: unjammed lambda_2={baseline[n]:.2f}, collapse threshold J/S*={js_star[n]:.2f} dB "
            f"(95% CI {js_star_ci[n][0]:.2f}..{js_star_ci[n][1]:.2f}), with rho>=1/N_CH: {js_star_phys[n]:.2f} dB, "
            f"link range at threshold={range_star[n]:.0f} m ({range_star[n] / AREA_SIDE:.3f} L), "
            f"closed form={cf[n]:.2f} dB, 50% knee of mean lambda_2={js_knee[n]:.2f} dB")
    (RES_DIR / "s2_summary.txt").write_text("\n".join(summary) + "\n")

    # --- Figure (a): lambda_2 vs rho ----------------------------------------
    plt.figure(figsize=(7, 4.5))
    for n in N_LIST:
        line, = plt.semilogx(RHO_GRID, l2_vs_rho[n], "o-", markersize=4, label=f"N = {n}")
        plt.fill_between(RHO_GRID, l2_vs_rho[n] - l2_std_vs_rho[n], l2_vs_rho[n] + l2_std_vs_rho[n],
                         color=line.get_color(), alpha=0.15)
    plt.axvline(rho_star_a, color="k", linestyle=":", label=f"range-minimising rho = {rho_star_a:.4f}")
    plt.axvspan(RHO_GRID[0], RHO_MIN_PHYSICAL, color="gray", alpha=0.15,
                label="rho < 1/N_CH (less than one channel)")
    plt.xlabel("Jammer band fraction rho (log scale)")
    plt.ylabel("lambda_2 (mean over placements, band = +/- 1 std)")
    plt.title(f"lambda_2 vs. rho at J/S_ref = {JS_REF_FOR_RHO_SWEEP_DB:.0f} dB")
    plt.grid(True)
    plt.legend(fontsize=8, loc="lower right")
    plt.savefig(FIG_DIR / "s2_lambda2_vs_rho.png", dpi=200, bbox_inches="tight")
    plt.close()

    # --- Figure (b): lambda_2 vs J/S under worst rho ------------------------
    fig, ax1 = plt.subplots(figsize=(7.5, 4.8))
    ax2 = ax1.twinx()
    for n in N_LIST:
        mean, std = l2_all[n].mean(axis=0), l2_all[n].std(axis=0)
        line, = ax1.plot(JS_REF_GRID, mean, "o-", markersize=3, label=f"lambda_2, N = {n}")
        ax1.fill_between(JS_REF_GRID, mean - std, mean + std, color=line.get_color(), alpha=0.15)
        ax2.plot(JS_REF_GRID, conn_all[n].mean(axis=0), "--", color=line.get_color(), alpha=0.7)
        ax1.axvline(js_star[n], color=line.get_color(), linestyle=":", alpha=0.9)
    ax2.axhline(0.5, color="gray", linewidth=0.8)
    ax1.set_xlabel("J/S at reference distance [dB]  (jammer uses its worst rho at each point)")
    ax1.set_ylabel("lambda_2 (mean, band = +/- 1 std)")
    ax2.set_ylabel("Fraction of placements still connected (dashed)")
    ax2.set_ylim(0, 1.05)
    ax1.set_title("Connectivity under a jammer-optimal partial-band jammer  (dotted: J/S*)")
    ax1.grid(True)
    ax1.legend(loc="upper right")
    fig.savefig(FIG_DIR / "s2_lambda2_vs_js.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # --- Figure (c): example topologies ------------------------------------
    pos = placements[20][0]
    d = pairwise_distance(pos)
    cases = [(-np.inf, "unjammed"),
             (js_star[20] - 4.0, "J/S* - 4 dB"),
             (js_star[20] + 2.0, "J/S* + 2 dB")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, (js, label) in zip(axes, cases):
        rho = 1.0 if np.isinf(js) else worst_rho(js)[0]
        a = adjacency(d, js, rho)
        for i in range(20):
            for j in range(i + 1, 20):
                if a[i, j]:
                    ax.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]], "-", color="tab:blue", alpha=0.5, linewidth=0.8)
        ax.plot(pos[:, 0], pos[:, 1], "o", color="k", markersize=5)
        head = "no jammer" if np.isinf(js) else f"J/S_ref = {js:.1f} dB, rho = {rho:.4f}"
        ax.set_title(f"{head}\nlambda_2 = {lambda2(a):.2f}  ({label})", fontsize=9)
        ax.set_xlim(0, AREA_SIDE); ax.set_ylim(0, AREA_SIDE); ax.set_aspect("equal")
        ax.set_xlabel("x [m]"); ax.set_yticks([])
    axes[0].set_ylabel("y [m]"); axes[0].set_yticks([0, 500, 1000])
    fig.suptitle("N = 20 example placement: surviving links as jamming increases", y=1.04)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "s2_example_topologies.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
