"""
S3a - How many hops does a receiver need to notice a partial-band jammer?
==========================================================================

Purpose
-------
S0-S2 treat the jammer as a known input. S3a asks how many hops a receiver
must observe before it can say "a jammer is present". The detector is an
energy detector on known pilot symbols; its performance has a closed form,
which is used to verify the simulator.

Model (continues S1: same N_CH, Eb/N0 and link budget)
------------------------------------------------------
Each hop carries N_PILOT known BPSK pilot symbols (amplitude sqrt(Eb) = 1).
The receiver subtracts the known pilots; what remains is noise only:

    unjammed hop : residual samples ~ N(0, N0/2)
    jammed hop   : residual samples ~ N(0, (N0 + N_J/rho)/2)

Per-hop statistic and its distribution

    T = sum(residual^2) / (N0/2)
    unjammed : T     ~ chi2(N_PILOT)
    jammed   : T / a ~ chi2(N_PILOT),   a = 1 + (N_J/N0) / rho

The two cases differ only in the noise variance, so the likelihood ratio is
increasing in T and the energy test "T > gamma" is the most powerful test for
one hop (Neyman-Pearson; S3a note, Section 2):

    per-hop false alarm   P_fa = P(chi2_n > gamma)
    per-hop detection     P_d  = P(chi2_n > gamma / a)

Window of M hops (count detector): flag every hop with T > gamma and declare
"jammer present" if at least k of the M hops are flagged. A hop lands in the
jammed part of the band with probability rho (the jammer does not know the
hopping pattern, as in S1), so

    no jammer : count ~ Binomial(M, P_fa)
    jammer    : count ~ Binomial(M, p1),   p1 = rho * P_d + (1 - rho) * P_fa

k is the smallest count that keeps the window false-alarm probability at or
below PFA_WINDOW. M* is the smallest M whose window detection probability
reaches PD_WINDOW, minimised over a grid of per-hop thresholds.

Benchmark at rho = 1: every hop is jammed, all M * N_PILOT residual samples
are i.i.d., and the most powerful test is the pooled energy
sum_M T ~ chi2(M * N_PILOT) (scaled by a when jammed).

Verification
------------
1. Per-hop Monte Carlo P_fa and P_d (pilots generated, received, subtracted)
   vs. the chi-square closed form: |z| < 4 at every point.
2. Window Monte Carlo P(count >= k) vs. the binomial closed form, with and
   without the jammer: |z| < 4 at every point.
3. Strong-jammer limit: if every jammed hop is flagged and false alarms are
   negligible, the window only needs to contain one jammed hop, so
   M* -> ceil( ln(1 - PD_WINDOW) / ln(1 - rho) ). At J/S = 30 dB the computed
   M* must be within one hop of this for every rho.
4. Neyman-Pearson at rho = 1: for every M and every per-hop threshold, the
   count detector's window detection probability must not exceed that of the
   pooled energy detector, which is optimal there.

z-scores use the binomial form z = (K - N p) / sqrt(N p (1 - p)); S0 used the
small-p form sqrt(N p), which is not valid here because P_d can be close to 1.

Outputs
-------
figures/s3a_roc_per_hop.png          per-hop ROC, closed form + Monte Carlo
figures/s3a_hops_to_detect_vs_rho.png   M* vs rho at two J/S values
results/s3a_hops_to_detect.csv       M* for every (N_PILOT, J/S, rho)
results/s3a_monte_carlo_checks.csv   every Monte Carlo point of checks 1-2
results/s3a_summary.txt
"""

import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom, chi2

# S1 is the single source of the link budget: same N_CH, thermal Eb/N0 and
# outage criterion, and the same worst-case-rho and required-Eb/N_J routines.
from s1_fh_partial_band_jamming import (
    N_CH, EBN0_DB, BER_REQUIRED, db_to_linear, linear_to_db,
    worst_case_rho, required_ebnj_db,
)

# Parameters (assumed values)
SEED = 2026
N_PILOT = 16                      # pilot symbols per hop (main results); sensitivity below
N_PILOT_SENSITIVITY = [8, 16, 32]
PD_WINDOW = 0.9                   # required window detection probability (assumed)
PFA_WINDOW = 1e-3                 # allowed window false-alarm probability (assumed)
JS_HIGH_DB = 12.0                 # the J/S of the S1 BER-vs-rho figure
JS_STRONG_DB = 30.0               # used only for verification 3 (strong-jammer limit)

RHO_GRID = np.logspace(np.log10(0.005), 0.0, 36)            # jammer band fractions
PFA_HOP_GRID = np.logspace(-9, np.log10(0.5), 70)           # per-hop thresholds searched
M_CHUNK = 256                     # window lengths evaluated per step of the search
M_MAX = 20_000                    # search limit for M*

N_HOPS_MC = 200_000               # hops per case in verification 1
N_WINDOWS_MC = 20_000             # windows per case in verification 2
PFA_HOP_TEST = [1e-1, 1e-2, 1e-3]
Z_LIMIT = 4.0

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
RES_DIR = ROOT / "results"
FIG_DIR.mkdir(exist_ok=True)
RES_DIR.mkdir(exist_ok=True)


# Closed-form detection probabilities
def noise_ratio(js_db, rho):
    """a = (noise variance in a jammed hop) / (thermal noise variance) = 1 + (N_J/N0)/rho."""
    gp_db = linear_to_db(N_CH)
    nj_over_n0 = db_to_linear(EBN0_DB) / db_to_linear(gp_db - js_db)   # (Eb/N0)/(Eb/N_J)
    return 1.0 + nj_over_n0 / rho


def hop_detection(pfa_hop, a, n):
    """Per-hop energy test: threshold gamma for a given P_fa, and the resulting P_d."""
    gamma = chi2.isf(pfa_hop, n)
    return gamma, chi2.sf(gamma / a, n)


def count_threshold(m, p0):
    """Smallest k with P(Binomial(m, p0) >= k) <= PFA_WINDOW (vectorised over m)."""
    k = binom.ppf(1.0 - PFA_WINDOW, m, p0) + 1.0
    for _ in range(3):                                   # guard against floating-point edge cases
        k = np.where(binom.sf(k - 1, m, p0) > PFA_WINDOW, k + 1, k)                 # k too small
        k = np.where((k > 1) & (binom.sf(k - 2, m, p0) <= PFA_WINDOW), k - 1, k)    # k not minimal
    return k


def hops_to_detect(a, rho, n):
    """M* of the count detector: smallest window reaching PD_WINDOW at PFA_WINDOW.

    Searches every per-hop threshold in PFA_HOP_GRID and, for each, every M up to
    the best M found so far. Returns (M*, P_fa per hop, P_d per hop, k).
    """
    best = (M_MAX + 1, np.nan, np.nan, np.nan)
    for pfa_hop in PFA_HOP_GRID:
        _, pd_hop = hop_detection(pfa_hop, a, n)
        p1 = rho * pd_hop + (1.0 - rho) * pfa_hop
        m_start = 1
        while m_start < best[0]:
            m = np.arange(m_start, min(m_start + M_CHUNK, best[0]))
            k = count_threshold(m, pfa_hop)
            pd_window = binom.sf(k - 1, m, p1)
            hit = np.nonzero(pd_window >= PD_WINDOW)[0]
            if hit.size:
                i = hit[0]
                best = (int(m[i]), pfa_hop, pd_hop, int(k[i]))
                break
            m_start += M_CHUNK
    if best[0] > M_MAX:
        raise RuntimeError(f"M* above M_MAX for a = {a:.3g}, rho = {rho:.3g}")
    return best


def pooled_detection(m, a, n):
    """Window P_d of the pooled energy detector (optimal when rho = 1) at PFA_WINDOW."""
    dof = m * n
    gamma = chi2.isf(PFA_WINDOW, dof)
    return chi2.sf(gamma / a, dof)


def hops_to_detect_pooled(a, n):
    m = np.arange(1, M_MAX + 1)
    return int(m[np.nonzero(pooled_detection(m, a, n) >= PD_WINDOW)[0][0]])


def strong_jammer_limit(rho):
    """Smallest M with 1 - (1 - rho)^M >= PD_WINDOW (at least one jammed hop in the window)."""
    if rho >= 1.0:
        return 1
    return int(np.ceil(np.log(1.0 - PD_WINDOW) / np.log(1.0 - rho)))


# Monte Carlo
def simulate_hop_energy(num_hops, a, n, rng):
    """T for num_hops hops: send pilots, add noise, subtract the known pilots.

    Eb = 1, so N0 = 1/(Eb/N0). The jammed-hop noise variance is a * N0/2.
    """
    n0 = 1.0 / db_to_linear(EBN0_DB)
    sigma = np.sqrt(a * n0 / 2.0)
    pilots = 2.0 * rng.integers(0, 2, size=(num_hops, n)) - 1.0
    received = pilots + sigma * rng.normal(size=(num_hops, n))
    residual = received - pilots
    return np.sum(residual ** 2, axis=1) / (n0 / 2.0)


def simulate_window_counts(num_windows, m, rho, a, gamma, n, rng, chunk=4000):
    """Number of flagged hops in each of num_windows windows of m hops (rho = 0: no jammer)."""
    counts = []
    for start in range(0, num_windows, chunk):
        w = min(chunk, num_windows - start)
        jammed = rng.random((w, m)) < rho                       # which hops land in the jammed part
        t_clear = simulate_hop_energy(w * m, 1.0, n, rng).reshape(w, m)
        t_jam = simulate_hop_energy(w * m, a, n, rng).reshape(w, m)
        t = np.where(jammed, t_jam, t_clear)
        counts.append(np.sum(t > gamma, axis=1))
    return np.concatenate(counts)


def z_binomial(events, trials, p):
    return (events - trials * p) / np.sqrt(trials * p * (1.0 - p))


# Main
def main():
    rng = np.random.default_rng(SEED)
    gp_db = linear_to_db(N_CH)
    ebn0_lin = db_to_linear(EBN0_DB)

    # Operating points taken from S1
    req_ebnj_db = required_ebnj_db(ebn0_lin, BER_REQUIRED, rho_min=1e-5)
    js_outage_db = gp_db - req_ebnj_db                       # worst-case link just reaches outage
    rho_worst = {
        js_outage_db: worst_case_rho(ebn0_lin, db_to_linear(req_ebnj_db), 1e-5)[0],
        JS_HIGH_DB: worst_case_rho(ebn0_lin, db_to_linear(gp_db - JS_HIGH_DB))[0],
    }
    js_study = [js_outage_db, JS_HIGH_DB]
    print(f"S1 operating points: outage J/S = {js_outage_db:.2f} dB (worst rho {rho_worst[js_outage_db]:.4f}), "
          f"J/S = {JS_HIGH_DB:.0f} dB (worst rho {rho_worst[JS_HIGH_DB]:.3f})")

    mc_rows = []

    # Verification 1: per-hop P_fa and P_d
    roc_cases = [(js_outage_db, rho_worst[js_outage_db]), (js_outage_db, 0.1), (js_outage_db, 1.0)]
    t_h0 = simulate_hop_energy(N_HOPS_MC, 1.0, N_PILOT, rng)
    z1 = []
    roc_mc = []
    print(f"\n{'case':>28} {'P_fa hop':>9} {'theory':>10} {'Monte Carlo':>12} {'z':>6}")
    for pfa in PFA_HOP_TEST:
        gamma = chi2.isf(pfa, N_PILOT)
        k = int(np.sum(t_h0 > gamma))
        z = z_binomial(k, N_HOPS_MC, pfa)
        z1.append(z)
        mc_rows.append(("1", "no jammer", "", "", N_PILOT, "", "", "P_fa", pfa, N_HOPS_MC, k, z))
        print(f"{'no jammer (P_fa)':>28} {pfa:9.0e} {pfa:10.3e} {k / N_HOPS_MC:12.3e} {z:6.2f}")
    for js_db, rho in roc_cases:
        a = noise_ratio(js_db, rho)
        t_h1 = simulate_hop_energy(N_HOPS_MC, a, N_PILOT, rng)
        pd_mc = []
        for pfa in PFA_HOP_TEST:
            gamma, pd = hop_detection(pfa, a, N_PILOT)
            k = int(np.sum(t_h1 > gamma))
            z = z_binomial(k, N_HOPS_MC, pd)
            z1.append(z)
            pd_mc.append(k / N_HOPS_MC)
            mc_rows.append(("1", "jammed hop", f"{js_db:.2f}", rho, N_PILOT, "", "", "P_d", pd, N_HOPS_MC, k, z))
            print(f"{f'J/S {js_db:.2f} dB, rho {rho:.4f}':>28} {pfa:9.0e} {pd:10.3e} {k / N_HOPS_MC:12.3e} {z:6.2f}")
        roc_mc.append(pd_mc)
    z1 = np.array(z1)
    ok1 = bool(np.all(np.abs(z1) < Z_LIMIT))
    print(f"VERIFICATION 1 (per-hop Monte Carlo vs chi-square): {'PASS' if ok1 else 'FAIL'} "
          f"(max |z| = {np.abs(z1).max():.2f} over {len(z1)} points)")

    # Verification 2: window counts
    window_cases = [  # (J/S, rho, M, per-hop P_fa, k values tested)
        (js_outage_db, 0.10, 50, 0.05, [3, 5, 7]),
        (JS_HIGH_DB, 0.05, 20, 0.01, [1, 2]),
    ]
    z2 = []
    print(f"\n{'case':>34} {'hyp.':>8} {'k':>3} {'theory':>10} {'Monte Carlo':>12} {'z':>6}")
    for js_db, rho, m, pfa, k_list in window_cases:
        a = noise_ratio(js_db, rho)
        gamma, pd = hop_detection(pfa, a, N_PILOT)
        p1 = rho * pd + (1.0 - rho) * pfa
        for hyp, rho_sim, p in (("none", 0.0, pfa), ("jammer", rho, p1)):
            counts = simulate_window_counts(N_WINDOWS_MC, m, rho_sim, a, gamma, N_PILOT, rng)
            for k in k_list:
                p_theory = binom.sf(k - 1, m, p)
                events = int(np.sum(counts >= k))
                z = z_binomial(events, N_WINDOWS_MC, p_theory)
                z2.append(z)
                mc_rows.append(("2", hyp, f"{js_db:.2f}", rho, N_PILOT, m, k, "P(count>=k)",
                                p_theory, N_WINDOWS_MC, events, z))
                label = f"J/S {js_db:.2f} dB, rho {rho:.2f}, M {m}"
                print(f"{label:>34} {hyp:>8} {k:3d} {p_theory:10.3e} {events / N_WINDOWS_MC:12.3e} {z:6.2f}")
    z2 = np.array(z2)
    ok2 = bool(np.all(np.abs(z2) < Z_LIMIT))
    print(f"VERIFICATION 2 (window Monte Carlo vs binomial): {'PASS' if ok2 else 'FAIL'} "
          f"(max |z| = {np.abs(z2).max():.2f} over {len(z2)} points)")

    # Main sweep: M* vs rho
    sweep_rows = []
    m_star = {}
    for n in N_PILOT_SENSITIVITY:
        for js_db in js_study:
            values = []
            for rho in RHO_GRID:
                a = noise_ratio(js_db, rho)
                m, pfa_hop, pd_hop, k = hops_to_detect(a, rho, n)
                values.append(m)
                sweep_rows.append((n, round(js_db, 2), rho, a, m, pfa_hop, pd_hop, k))
            m_star[(n, js_db)] = np.array(values)

    # Verification 3: strong-jammer limit
    m_strong = np.array([hops_to_detect(noise_ratio(JS_STRONG_DB, r), r, N_PILOT)[0] for r in RHO_GRID])
    m_limit = np.array([strong_jammer_limit(r) for r in RHO_GRID])
    diff3 = m_strong - m_limit
    ok3 = bool(np.all(np.abs(diff3) <= 1))
    print(f"\nVERIFICATION 3 (strong-jammer limit, J/S = {JS_STRONG_DB:.0f} dB): {'PASS' if ok3 else 'FAIL'} "
          f"(M* - limit ranges {diff3.min()} .. {diff3.max()} hops over {len(RHO_GRID)} rho values; "
          f"{int(np.sum(diff3 == 0))} exact)")

    # Verification 4: Neyman-Pearson at rho = 1
    violations = 0
    checked = 0
    pooled_m = {}
    for js_db in js_study + [JS_STRONG_DB]:
        a = noise_ratio(js_db, 1.0)
        m_count = hops_to_detect(a, 1.0, N_PILOT)[0]
        pooled_m[js_db] = hops_to_detect_pooled(a, N_PILOT)
        m = np.arange(1, m_count + 1)
        pd_pool = pooled_detection(m, a, N_PILOT)
        for pfa_hop in PFA_HOP_GRID:
            _, pd_hop = hop_detection(pfa_hop, a, N_PILOT)
            pd_count = binom.sf(count_threshold(m, pfa_hop) - 1, m, pd_hop)
            violations += int(np.sum(pd_count > pd_pool + 1e-12))
            checked += m.size
    ok4 = violations == 0
    print(f"VERIFICATION 4 (count detector never beats pooled energy at rho = 1): "
          f"{'PASS' if ok4 else 'FAIL'} ({violations} violations in {checked} comparisons)")

    ok = ok1 and ok2 and ok3 and ok4
    print("\nOVERALL:", "PASS" if ok else "FAIL")

    # Summary
    lines = [
        f"Pilots per hop N_PILOT            : {N_PILOT} (sensitivity {N_PILOT_SENSITIVITY})",
        f"Window target                     : P_d >= {PD_WINDOW}, P_fa <= {PFA_WINDOW:g}",
        f"S1 outage J/S (worst-case rho)    : {js_outage_db:.2f} dB, rho = {rho_worst[js_outage_db]:.4f}",
        f"S1 worst-case rho at J/S = {JS_HIGH_DB:.0f} dB  : {rho_worst[JS_HIGH_DB]:.3f}",
        "",
        "M* = hops needed by the count detector (N_PILOT = 16 unless noted)",
    ]
    for js_db in js_study:
        ms = m_star[(N_PILOT, js_db)]
        i_min = int(np.argmin(ms))
        m_at_worst = hops_to_detect(noise_ratio(js_db, rho_worst[js_db]), rho_worst[js_db], N_PILOT)[0]
        rho_one_ch = max(rho_worst[js_db], 1.0 / N_CH)
        m_at_one_ch = hops_to_detect(noise_ratio(js_db, rho_one_ch), rho_one_ch, N_PILOT)[0]
        lines += [
            f"  J/S = {js_db:5.2f} dB: M* at S1 worst rho ({rho_worst[js_db]:.4f}) = {m_at_worst} "
            f"(limit {strong_jammer_limit(rho_worst[js_db])}; {m_at_one_ch} at rho >= 1/N_CH), "
            f"at rho = 1 = {ms[-1]} (pooled-energy optimum {pooled_m[js_db]}), "
            f"smallest M* on the grid = {ms[i_min]} at rho = {RHO_GRID[i_min]:.3f}",
        ]
    lines += ["", "Sensitivity to N_PILOT (M* at S1 worst rho / at rho = 0.1 / at rho = 1)"]
    for js_db in js_study:
        parts = []
        for n in N_PILOT_SENSITIVITY:
            m_w = hops_to_detect(noise_ratio(js_db, rho_worst[js_db]), rho_worst[js_db], n)[0]
            m_01 = hops_to_detect(noise_ratio(js_db, 0.1), 0.1, n)[0]
            m_1 = hops_to_detect(noise_ratio(js_db, 1.0), 1.0, n)[0]
            parts.append(f"n={n}: {m_w}/{m_01}/{m_1}")
        lines.append(f"  J/S = {js_db:5.2f} dB: " + ",  ".join(parts))
    lines += ["", f"Verification: 1 {'PASS' if ok1 else 'FAIL'}, 2 {'PASS' if ok2 else 'FAIL'}, "
                  f"3 {'PASS' if ok3 else 'FAIL'}, 4 {'PASS' if ok4 else 'FAIL'}"]
    print("\n" + "\n".join(lines))
    (RES_DIR / "s3a_summary.txt").write_text("\n".join(lines) + "\n")

    with open(RES_DIR / "s3a_hops_to_detect.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n_pilot", "js_db", "rho", "noise_ratio_a", "m_star", "pfa_hop", "pd_hop", "k"])
        w.writerows(sweep_rows)
    with open(RES_DIR / "s3a_monte_carlo_checks.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["check", "hypothesis", "js_db", "rho", "n_pilot", "window_m", "k", "quantity",
                    "p_theory", "trials", "events", "z_score"])
        w.writerows(mc_rows)

    # Figure 1: per-hop ROC
    pfa_curve = np.logspace(-6, 0, 300)
    plt.figure(figsize=(7, 4.8))
    for (js_db, rho), pd_mc in zip(roc_cases, roc_mc):
        a = noise_ratio(js_db, rho)
        line, = plt.semilogx(pfa_curve, hop_detection(pfa_curve, a, N_PILOT)[1],
                             label=f"rho = {rho:.3g}  (jammed-hop noise x{a:.2f})")
        plt.semilogx(PFA_HOP_TEST, pd_mc, "o", color=line.get_color(), markersize=5)
    plt.semilogx(pfa_curve, pfa_curve, "k:", label="no information (P_d = P_fa)")
    plt.xlabel("Per-hop false-alarm probability P_fa")
    plt.ylabel("Per-hop detection probability P_d")
    plt.title(f"Energy detector on {N_PILOT} pilots, J/S = {js_outage_db:.2f} dB (S1 outage point)\n"
              "lines: chi-square closed form, dots: Monte Carlo")
    plt.ylim(0, 1.02)
    plt.grid(True, which="both")
    plt.legend(fontsize=8, loc="center left")
    plt.savefig(FIG_DIR / "s3a_roc_per_hop.png", dpi=200, bbox_inches="tight")
    plt.close()

    # Figure 2: hops needed vs rho
    plt.figure(figsize=(7.5, 5))
    plt.axvspan(RHO_GRID[0], 1.0 / N_CH, color="0.9", label="rho < 1/N_CH (not realisable)")
    for js_db in js_study:
        line, = plt.loglog(RHO_GRID, m_star[(N_PILOT, js_db)], "o-", markersize=3,
                           label=f"count detector, J/S = {js_db:.2f} dB")
        plt.axvline(rho_worst[js_db], color=line.get_color(), linestyle=":",
                    label=f"S1 worst-case rho at {js_db:.2f} dB = {rho_worst[js_db]:.4f}")
        plt.loglog([1.0], [pooled_m[js_db]], "*", color=line.get_color(), markersize=11)
    plt.loglog(RHO_GRID, m_limit, "k--", label="strong-jammer limit ceil(ln 0.1 / ln(1-rho))")
    plt.loglog([], [], "k*", markersize=11, label="pooled energy at rho = 1 (optimal there)")
    plt.xlabel("Jammer band fraction rho")
    plt.ylabel(f"Hops needed M*  (P_d >= {PD_WINDOW}, P_fa <= {PFA_WINDOW:g})")
    plt.title(f"How long it takes to notice the jammer  ({N_PILOT} pilots per hop, N_CH = {N_CH})")
    plt.grid(True, which="both")
    plt.legend(fontsize=7.5, loc="lower left")
    plt.savefig(FIG_DIR / "s3a_hops_to_detect_vs_rho.png", dpi=200, bbox_inches="tight")
    plt.close()

    print(f"\nSaved figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
