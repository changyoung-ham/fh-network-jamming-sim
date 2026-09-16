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
    (a) For a given jammer power, which band fraction rho hurts the NETWORK most?
        (S1 answered this for one link; the answer can differ for a graph.)
    (b) Under a jammer that picks its worst rho, at what J/S does lambda_2
        collapse to zero, and how does that threshold move with N?

Model
-----
Geometry   N nodes uniform in an L x L square (fixed seeds, M placements).
Signal     Eb/N0(d) = EBN0_REF - 10*n*log10(d/D_REF)     [path-loss exponent n]
Jammer     Received J/S grows with link distance because S falls with d:
           J/S(d) = JS_REF + 10*n*log10(d/D_REF)           [jammer far from all nodes]
           Eb/N_J(d) = G_p - J/S(d)
Link BER   S1 closed-form partial-band mixture with fraction rho.
Link up    BER <= BER_OUTAGE  (assumed criterion, same as S1).
Graph      Undirected, unweighted adjacency A; Laplacian L = D - A; lambda_2 = eig(L)[1].

Verification (graph theory, exact)
----------------------------------
1. Complete graph K_N has lambda_2 = N.
2. lambda_2 > 0 if and only if breadth-first search finds one component.
   Checked for every graph evaluated (thousands of cases); mismatches must be 0.
3. lambda_2 <= N/(N-1) * minimum degree (Fiedler's bound) for every graph.
4. Limits: J/S -> -inf reproduces the unjammed lambda_2; J/S -> +inf gives 0.

Outputs
-------
figures/s2_lambda2_vs_rho.png       (a) lambda_2 vs rho, N = 10/20/30
figures/s2_lambda2_vs_js.png        (b) lambda_2 vs J/S under worst-case rho, threshold J/S*
figures/s2_example_topologies.png   one placement at three jamming levels
results/s2_lambda2_vs_rho.csv, results/s2_lambda2_vs_js.csv, results/s2_summary.txt

All parameters are pedagogical values, NOT specifications of any real system.
"""

import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc

# ---------------------------------------------------------------------------
# Parameters (assumed / pedagogical values)
# ---------------------------------------------------------------------------
SEED = 2026
N_LIST = [10, 20, 30]           # network sizes to compare
M_PLACEMENTS = 30               # random placements averaged per point
AREA_SIDE = 1000.0              # [m] square side (assumed)
D_REF = 100.0                   # [m] reference distance (assumed)
PATH_LOSS_EXP = 2.0             # free-space exponent (assumed; 2 = line of sight)
EBN0_REF_DB = 27.0              # thermal Eb/N0 at D_REF (assumed); unjammed network is dense but not complete
N_CH = 100                      # hop channels, G_p = 20 dB (same as S1)
BER_OUTAGE = 1e-3               # link outage criterion (assumed, same as S1)

RHO_GRID = np.logspace(-3, 0, 20)              # 20-step rho sweep, log-spaced 0.001 .. 1 (project spec)
JS_REF_GRID = np.arange(-30, 8, 2.0)           # J/S at D_REF [dB]; -30 dB must reproduce the unjammed graph
JS_REF_FOR_RHO_SWEEP_DB = -14.0                # J/S used in figure (a): inside the collapse region

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
RES_DIR = ROOT / "results"
FIG_DIR.mkdir(exist_ok=True)
RES_DIR.mkdir(exist_ok=True)


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


# ---------------------------------------------------------------------------
# Graph model
# ---------------------------------------------------------------------------
def place_nodes(n, rng):
    """Uniform random positions in the square."""
    return rng.uniform(0.0, AREA_SIDE, size=(n, 2))


def pairwise_distance(pos):
    diff = pos[:, None, :] - pos[None, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=-1))


def adjacency(pos, js_ref_db, rho):
    """Adjacency matrix: link (i,j) is up if its BER is below the outage criterion."""
    d = pairwise_distance(pos)
    np.fill_diagonal(d, np.nan)                                      # no self links
    gp_db = 10.0 * np.log10(N_CH)
    ebn0_db = EBN0_REF_DB - 10.0 * PATH_LOSS_EXP * np.log10(d / D_REF)
    js_db = js_ref_db + 10.0 * PATH_LOSS_EXP * np.log10(d / D_REF)   # S falls with d, J does not
    ebnj_db = gp_db - js_db
    with np.errstate(invalid="ignore"):
        ber = ber_partial_band(db_to_linear(ebn0_db), db_to_linear(ebnj_db), rho)
    a = (ber <= BER_OUTAGE).astype(int)
    np.fill_diagonal(a, 0)
    return a


def lambda2(a):
    """Second-smallest eigenvalue of the graph Laplacian L = D - A."""
    lap = np.diag(a.sum(axis=1)) - a
    eig = np.linalg.eigvalsh(lap)                                    # ascending, symmetric matrix
    return max(eig[1], 0.0)                                          # clip tiny negative round-off


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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    rng = np.random.default_rng(SEED)
    placements = {n: [place_nodes(n, rng) for _ in range(M_PLACEMENTS)] for n in N_LIST}

    # bookkeeping for the exact graph-theory checks
    n_graphs = 0
    n_mismatch = 0          # lambda_2 > 0 vs BFS disagreement
    n_bound_viol = 0        # lambda_2 > min degree

    def evaluate(a):
        nonlocal n_graphs, n_mismatch, n_bound_viol
        l2 = lambda2(a)
        conn = is_connected_bfs(a)
        n_graphs += 1
        if (l2 > 1e-9) != conn:
            n_mismatch += 1
        n = a.shape[0]
        if l2 > n / (n - 1) * a.sum(axis=1).min() + 1e-9:   # Fiedler: lambda_2 <= N/(N-1) * delta_min
            n_bound_viol += 1
        return l2, conn

    # --- Verification 1: complete graph --------------------------------------
    ok1 = all(np.isclose(lambda2(np.ones((n, n), int) - np.eye(n, dtype=int)), n) for n in N_LIST)
    print(f"VERIFICATION 1 (K_N has lambda_2 = N): {'PASS' if ok1 else 'FAIL'}")

    # --- Unjammed baseline ---------------------------------------------------
    baseline = {}
    for n in N_LIST:
        vals = [evaluate(adjacency(p, -100.0, 1.0))[0] for p in placements[n]]
        baseline[n] = float(np.mean(vals))
        print(f"N = {n:2d}: unjammed lambda_2 (mean over {M_PLACEMENTS} placements) = {baseline[n]:.3f}")

    # --- Sweep (a): lambda_2 vs rho at fixed J/S ---------------------------
    rows_a = []
    l2_vs_rho = {}
    rho_worst_net = {}
    for n in N_LIST:
        means = []
        for rho in RHO_GRID:
            vals = [evaluate(adjacency(p, JS_REF_FOR_RHO_SWEEP_DB, rho))[0] for p in placements[n]]
            means.append(np.mean(vals))
            rows_a.append((n, rho, np.mean(vals), np.std(vals)))
        l2_vs_rho[n] = np.array(means)
        rho_worst_net[n] = RHO_GRID[np.argmin(means)]
        print(f"N = {n:2d}: at J/S_ref = {JS_REF_FOR_RHO_SWEEP_DB:.0f} dB the network-worst rho = {rho_worst_net[n]:.2f} "
              f"(lambda_2 {means[-1]:.3f} at rho=1 -> {min(means):.3f} at worst rho)")

    # --- Sweep (b): lambda_2 vs J/S under network-worst rho -----------------
    rows_b = []
    l2_vs_js = {}
    conn_vs_js = {}
    js_star = {}
    for n in N_LIST:
        means, conns = [], []
        for js in JS_REF_GRID:
            # jammer picks the rho that minimises mean lambda_2 at this power
            best = None
            for rho in RHO_GRID:
                res = [evaluate(adjacency(p, js, rho)) for p in placements[n]]
                m = np.mean([r[0] for r in res])
                c = np.mean([r[1] for r in res])
                if best is None or m < best[0]:
                    best = (m, c, rho)
            means.append(best[0])
            conns.append(best[1])
            rows_b.append((n, js, best[0], best[1], best[2]))
        l2_vs_js[n] = np.array(means)
        conn_vs_js[n] = np.array(conns)
        # collapse threshold: first J/S at which fewer than half the placements stay connected
        idx = np.argmax(conn_vs_js[n] < 0.5)
        js_star[n] = JS_REF_GRID[idx] if conn_vs_js[n].min() < 0.5 else np.nan
        print(f"N = {n:2d}: collapse threshold J/S* = {js_star[n]:.0f} dB  "
              f"(lambda_2 at J/S={JS_REF_GRID[0]:.0f}: {means[0]:.3f}, at J/S={JS_REF_GRID[-1]:.0f}: {means[-1]:.3f})")

    # --- Verification 2 & 3 (all graphs evaluated above) --------------------
    ok2 = n_mismatch == 0
    ok3 = n_bound_viol == 0
    print(f"VERIFICATION 2 (lambda_2 > 0 iff BFS-connected): {'PASS' if ok2 else 'FAIL'} "
          f"({n_mismatch} mismatches in {n_graphs} graphs)")
    print(f"VERIFICATION 3 (Fiedler bound lambda_2 <= N/(N-1)*min degree): {'PASS' if ok3 else 'FAIL'} ({n_bound_viol} violations)")
    # --- Verification 4: limits ----------------------------------------------
    ok4 = all(np.isclose(l2_vs_js[n][0], baseline[n], rtol=0.05) for n in N_LIST) and \
          all(l2_vs_js[n][-1] < 1e-9 for n in N_LIST)
    print(f"VERIFICATION 4 (limits: low J/S -> unjammed, high J/S -> 0): {'PASS' if ok4 else 'FAIL'}")
    print("\nOVERALL:", "PASS" if (ok1 and ok2 and ok3 and ok4) else "FAIL")

    # --- Save numeric results ------------------------------------------------
    with open(RES_DIR / "s2_lambda2_vs_rho.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["N", "rho", "lambda2_mean", "lambda2_std"]); w.writerows(rows_a)
    with open(RES_DIR / "s2_lambda2_vs_js.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["N", "js_ref_db", "lambda2_mean", "connected_fraction", "worst_rho"]); w.writerows(rows_b)
    summary = [f"N={n}: unjammed lambda_2={baseline[n]:.3f}, network-worst rho at J/S_ref={JS_REF_FOR_RHO_SWEEP_DB:.0f} dB: "
               f"{rho_worst_net[n]:.2f}, collapse threshold J/S*={js_star[n]:.0f} dB" for n in N_LIST]
    (RES_DIR / "s2_summary.txt").write_text("\n".join(summary) + "\n")

    # --- Figure (a): lambda_2 vs rho ----------------------------------------
    plt.figure(figsize=(7, 4.5))
    for n in N_LIST:
        line, = plt.semilogx(RHO_GRID, l2_vs_rho[n], "o-", markersize=4, label=f"N = {n}")
        plt.axvline(rho_worst_net[n], color=line.get_color(), linestyle=":", alpha=0.7)
    plt.xlabel("Jammer band fraction rho (log scale)")
    plt.ylabel("Algebraic connectivity lambda_2 (mean over placements)")
    plt.title(f"lambda_2 vs. rho at J/S_ref = {JS_REF_FOR_RHO_SWEEP_DB:.0f} dB  (dotted: network-worst rho)")
    plt.grid(True)
    plt.legend()
    plt.savefig(FIG_DIR / "s2_lambda2_vs_rho.png", dpi=200, bbox_inches="tight")
    plt.close()

    # --- Figure (b): lambda_2 vs J/S under worst rho ------------------------
    fig, ax1 = plt.subplots(figsize=(7.5, 4.8))
    ax2 = ax1.twinx()
    for n in N_LIST:
        line, = ax1.plot(JS_REF_GRID, l2_vs_js[n], "o-", markersize=4, label=f"lambda_2, N = {n}")
        ax2.plot(JS_REF_GRID, conn_vs_js[n], "--", color=line.get_color(), alpha=0.6)
        if not np.isnan(js_star[n]):
            ax1.axvline(js_star[n], color=line.get_color(), linestyle=":", alpha=0.8)
    ax1.set_xlabel("J/S at reference distance [dB]  (jammer chooses its worst rho at each point)")
    ax1.set_ylabel("Algebraic connectivity lambda_2 (mean)")
    ax2.set_ylabel("Fraction of placements still connected (dashed)")
    ax2.set_ylim(0, 1.05)
    ax1.set_title("Connectivity collapse under a jammer-optimal partial-band jammer  (dotted: J/S*)")
    ax1.grid(True)
    ax1.legend(loc="upper right")
    fig.savefig(FIG_DIR / "s2_lambda2_vs_js.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # --- Figure (c): example topologies ------------------------------------
    pos = placements[20][0]
    cases = [(JS_REF_GRID[0], "unjammed"), (js_star[20] - 4, "before collapse"), (js_star[20] + 2, "after collapse")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, (js, label) in zip(axes, cases):
        rho = RHO_GRID[np.argmin([np.mean([lambda2(adjacency(p, js, r)) for p in placements[20]]) for r in RHO_GRID])]
        a = adjacency(pos, js, rho)
        for i in range(20):
            for j in range(i + 1, 20):
                if a[i, j]:
                    ax.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]], "-", color="tab:blue", alpha=0.5, linewidth=0.8)
        ax.plot(pos[:, 0], pos[:, 1], "o", color="k", markersize=5)
        ax.set_title(f"J/S_ref = {js:.0f} dB, rho = {rho:.3f}\nlambda_2 = {lambda2(a):.2f}  ({label})", fontsize=9)
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
