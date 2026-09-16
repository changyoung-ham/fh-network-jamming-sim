"""
S1 - Frequency hopping under partial-band noise jamming
=======================================================

Purpose
-------
Extend the S0 link (BPSK/AWGN) with slow frequency hopping over N_CH channels
and a partial-band noise jammer that spreads its total power J over a fraction
rho of the hopping band. Verify the Monte Carlo simulator against the
closed-form mixture formula, then use the formula to map BER vs. J/S and to
find the jammer's worst-case rho.

Model
-----
Per hop, the transmitter picks one of N_CH channels uniformly at random
(pseudo-random hopping pattern; the jammer does not know it).
The jammer occupies a fraction rho of the channels with noise of total
power J. Inside the jammed channels the jammer's power spectral density is
N_J / rho, where N_J = J / W_ss is the density it would have if spread over
the whole band.

    hop lands in a jammed channel with probability rho
    unjammed hop : BER = Q( sqrt(2 Eb / N0) )
    jammed hop   : BER = Q( sqrt(2 Eb / (N0 + N_J / rho)) )
    average      : BER = (1 - rho) * BER_unjammed + rho * BER_jammed

Link budget relations (all power ratios, 10*log10 convention):

    processing gain   G_p [dB] = 10 log10( W_ss / W_data ) = 10 log10( N_CH )
    Eb/N_J [dB]       = G_p [dB] - (J/S) [dB]
    jamming margin    M_J [dB] = G_p - (Eb/N_J)_required - L_sys

Verification
------------
1. Monte Carlo BER (random hop, random jammed flag, AWGN) must match the
   closed-form mixture at every (J/S, rho) point within statistical spread.
2. rho = 1 must reduce to the S0 AWGN result with N0 replaced by N0 + N_J.
3. Worst-case rho found by the simulator must agree with the closed-form
   minimiser.

Outputs
-------
figures/s1_ber_vs_js.png        BER vs J/S for several rho, plus worst-case envelope
figures/s1_ber_vs_rho.png       BER vs rho at fixed J/S (shows the worst-case rho)
results/s1_ber_vs_js.csv
results/s1_link_budget.txt

All parameters are pedagogical values, NOT specifications of any real radio.
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
N_CH = 100                 # hop channels -> processing gain 20 dB (assumed)
EBN0_DB = 10.0             # thermal Eb/N0 [dB]; jamming, not thermal noise, is the limit here (assumed)
JS_DB = np.arange(0, 22, 2.0)            # J/S sweep [dB]
RHO_LIST = [0.1, 0.3, 0.6, 1.0]          # jammer band fractions to show
RHO_GRID = np.linspace(0.02, 1.0, 50)    # fine grid for worst-case search
JS_FOR_RHO_PLOT_DB = 12.0                # J/S used in the BER-vs-rho figure
BER_REQUIRED = 1e-3                      # link-outage criterion (assumed; justified in S2)
L_SYS_DB = 2.0                           # implementation loss (assumed)

BITS_PER_CHUNK = 500_000
MIN_ERRORS = 200
MAX_BITS = 20_000_000

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
RES_DIR = ROOT / "results"
FIG_DIR.mkdir(exist_ok=True)
RES_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def db_to_linear(x_db):
    return 10.0 ** (np.asarray(x_db, dtype=float) / 10.0)


def linear_to_db(x):
    return 10.0 * np.log10(x)


def q_func(x):
    """Gaussian tail probability Q(x)."""
    return 0.5 * erfc(x / np.sqrt(2.0))


def ber_bpsk(ebn_lin):
    """Coherent BPSK BER for an effective Eb/N (linear)."""
    return q_func(np.sqrt(2.0 * ebn_lin))


def ber_partial_band_theory(ebn0_lin, ebnj_lin, rho):
    """Closed-form average BER of BPSK/FH under partial-band noise jamming.

    ebn0_lin : thermal Eb/N0 (linear)
    ebnj_lin : Eb/N_J with the jammer spread over the FULL band (linear)
    rho      : fraction of the band the jammer actually occupies (0 < rho <= 1)
    """
    n0 = 1.0 / ebn0_lin                       # Eb = 1 -> N0 = 1/(Eb/N0)
    nj = 1.0 / ebnj_lin                       # full-band equivalent jammer density
    ber_clear = ber_bpsk(1.0 / n0)
    ber_jammed = ber_bpsk(1.0 / (n0 + nj / rho))
    return (1.0 - rho) * ber_clear + rho * ber_jammed


def simulate_partial_band(ebn0_lin, ebnj_lin, rho, rng):
    """Monte Carlo BER: one bit per hop, random channel, random jammed flag.

    Each hop is jammed with probability rho (the hopping pattern is unknown to
    the jammer, so the channel choice is independent of the jammed set).
    Noise variance of a real AWGN sample is (N0_eff)/2 with N0_eff the
    effective one-sided density for that hop.
    """
    n0 = 1.0 / ebn0_lin
    nj = 1.0 / ebnj_lin
    sigma_clear = np.sqrt(n0 / 2.0)
    sigma_jammed = np.sqrt((n0 + nj / rho) / 2.0)

    n_bits = n_err = 0
    while n_err < MIN_ERRORS and n_bits < MAX_BITS:
        bits = rng.integers(0, 2, size=BITS_PER_CHUNK)
        symbols = 2 * bits - 1
        # Which hops land in the jammed part of the band?
        jammed = rng.random(BITS_PER_CHUNK) < rho
        sigma = np.where(jammed, sigma_jammed, sigma_clear)
        received = symbols + rng.normal(0.0, 1.0, size=BITS_PER_CHUNK) * sigma
        n_err += int(np.sum((received > 0).astype(int) != bits))
        n_bits += BITS_PER_CHUNK
    return n_err / n_bits, n_bits, n_err


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    rng = np.random.default_rng(SEED)
    gp_db = linear_to_db(N_CH)
    ebn0_lin = db_to_linear(EBN0_DB)

    # --- Link budget numbers -------------------------------------------------
    # Required Eb/N_J for the outage criterion in the WORST case (jammer-optimal
    # rho) is found numerically from the closed form.
    ebnj_grid_db = np.linspace(-5, 25, 601)
    worst_ber = np.array([
        max(ber_partial_band_theory(ebn0_lin, db_to_linear(e), r) for r in RHO_GRID)
        for e in ebnj_grid_db
    ])
    idx = np.argmax(worst_ber <= BER_REQUIRED)          # first Eb/N_J that meets the target
    ebnj_req_worst_db = ebnj_grid_db[idx]
    # Same requirement if the jammer stupidly spreads over the full band (rho = 1)
    full_ber = ber_partial_band_theory(ebn0_lin, db_to_linear(ebnj_grid_db), 1.0)
    ebnj_req_full_db = ebnj_grid_db[np.argmax(full_ber <= BER_REQUIRED)]

    margin_worst = gp_db - ebnj_req_worst_db - L_SYS_DB
    margin_full = gp_db - ebnj_req_full_db - L_SYS_DB

    budget = [
        f"Hop channels N_CH             : {N_CH}",
        f"Processing gain G_p           : {gp_db:.2f} dB",
        f"Thermal Eb/N0                 : {EBN0_DB:.1f} dB",
        f"Implementation loss L_sys     : {L_SYS_DB:.1f} dB",
        f"Outage criterion BER          : {BER_REQUIRED:g}",
        "",
        f"Required Eb/N_J, rho = 1      : {ebnj_req_full_db:.2f} dB  (full-band jammer)",
        f"Required Eb/N_J, worst rho    : {ebnj_req_worst_db:.2f} dB  (jammer-optimal rho)",
        f"Jamming margin,  rho = 1      : {margin_full:.2f} dB  -> link survives J/S up to this value",
        f"Jamming margin,  worst rho    : {margin_worst:.2f} dB",
        f"Margin lost to smart jamming  : {margin_full - margin_worst:.2f} dB",
    ]
    print("\n".join(budget))
    (RES_DIR / "s1_link_budget.txt").write_text("\n".join(budget) + "\n")

    # --- Sweep 1: BER vs J/S for several rho (Monte Carlo vs theory) --------
    print(f"\n{'J/S[dB]':>7} {'rho':>5} {'BER_sim':>10} {'BER_theory':>11} {'ratio':>6} {'errors':>7}")
    rows = []
    ratios = []
    sim = {r: np.zeros(len(JS_DB)) for r in RHO_LIST}
    thy = {r: np.zeros(len(JS_DB)) for r in RHO_LIST}
    for j, js_db in enumerate(JS_DB):
        ebnj_lin = db_to_linear(gp_db - js_db)
        for r in RHO_LIST:
            t = ber_partial_band_theory(ebn0_lin, ebnj_lin, r)
            s, nb, ne = simulate_partial_band(ebn0_lin, ebnj_lin, r, rng)
            sim[r][j], thy[r][j] = s, t
            ratio = s / t if t > 0 else np.nan
            if ne >= MIN_ERRORS:                 # only well-sampled points enter the check
                ratios.append(ratio)
            rows.append((js_db, r, s, t, ratio, nb, ne))
            print(f"{js_db:7.1f} {r:5.2f} {s:10.3e} {t:11.3e} {ratio:6.3f} {ne:7d}")

    ratios = np.array(ratios)
    ok1 = np.all((ratios > 0.7) & (ratios < 1.4))
    print(f"\nVERIFICATION 1 (Monte Carlo vs closed form): {'PASS' if ok1 else 'FAIL'} "
          f"(ratio range {ratios.min():.3f} .. {ratios.max():.3f}, {len(ratios)} points)")

    # Verification 2: rho = 1 reduces to AWGN with N0 + N_J
    js_check = 10.0
    ebnj_lin = db_to_linear(gp_db - js_check)
    n_eff = 1.0 / ebn0_lin + 1.0 / ebnj_lin
    ber_rho1 = ber_partial_band_theory(ebn0_lin, ebnj_lin, 1.0)
    ber_awgn = ber_bpsk(1.0 / n_eff)
    ok2 = np.isclose(ber_rho1, ber_awgn, rtol=1e-12)
    print(f"VERIFICATION 2 (rho=1 equals AWGN with N0+N_J): {'PASS' if ok2 else 'FAIL'} "
          f"({ber_rho1:.4e} vs {ber_awgn:.4e})")

    with open(RES_DIR / "s1_ber_vs_js.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["js_db", "rho", "ber_sim", "ber_theory", "ratio", "n_bits", "n_errors"])
        w.writerows(rows)

    # --- Sweep 2: BER vs rho at fixed J/S; worst-case rho ------------------
    ebnj_lin = db_to_linear(gp_db - JS_FOR_RHO_PLOT_DB)
    ber_rho_theory = ber_partial_band_theory(ebn0_lin, ebnj_lin, RHO_GRID)
    rho_star_theory = RHO_GRID[np.argmax(ber_rho_theory)]

    rho_sim_pts = np.linspace(0.05, 1.0, 12)
    ber_rho_sim = np.array([simulate_partial_band(ebn0_lin, ebnj_lin, r, rng)[0] for r in rho_sim_pts])
    rho_star_sim = rho_sim_pts[np.argmax(ber_rho_sim)]
    ok3 = abs(rho_star_sim - rho_star_theory) <= 0.15
    print(f"VERIFICATION 3 (worst-case rho): theory {rho_star_theory:.2f}, simulation {rho_star_sim:.2f} "
          f"-> {'PASS' if ok3 else 'FAIL'}")
    print(f"  at J/S = {JS_FOR_RHO_PLOT_DB:.0f} dB: BER(rho=1) = {ber_rho_theory[-1]:.2e}, "
          f"BER(rho*) = {ber_rho_theory.max():.2e}  (x{ber_rho_theory.max()/ber_rho_theory[-1]:.0f} worse)")

    print("\nOVERALL:", "PASS" if (ok1 and ok2 and ok3) else "FAIL")

    # --- Figure 1: BER vs J/S --------------------------------------------------
    js_fine = np.linspace(JS_DB[0], JS_DB[-1], 200)
    envelope = np.array([
        max(ber_partial_band_theory(ebn0_lin, db_to_linear(gp_db - js), r) for r in RHO_GRID)
        for js in js_fine
    ])
    plt.figure(figsize=(7.5, 5))
    for r in RHO_LIST:
        line, = plt.semilogy(js_fine,
                             ber_partial_band_theory(ebn0_lin, db_to_linear(gp_db - js_fine), r),
                             label=f"theory, rho = {r:.1f}")
        plt.semilogy(JS_DB, sim[r], "o", color=line.get_color(), markersize=4)
    plt.semilogy(js_fine, envelope, "k--", linewidth=2, label="worst-case rho (jammer-optimal)")
    plt.axhline(BER_REQUIRED, color="gray", linestyle=":", label=f"outage criterion BER = {BER_REQUIRED:g}")
    plt.xlabel("J/S at the receiver [dB]")
    plt.ylabel("Bit error rate")
    plt.title(f"FH/BPSK under partial-band jamming  (N_CH = {N_CH}, G_p = {gp_db:.0f} dB, Eb/N0 = {EBN0_DB:.0f} dB)")
    plt.ylim(1e-6, 0.5)
    plt.grid(True, which="both")
    plt.legend(fontsize=8)
    plt.savefig(FIG_DIR / "s1_ber_vs_js.png", dpi=200, bbox_inches="tight")
    plt.close()

    # --- Figure 2: BER vs rho --------------------------------------------------
    plt.figure(figsize=(7, 4.5))
    plt.semilogy(RHO_GRID, ber_rho_theory, "-", label="closed form")
    plt.semilogy(rho_sim_pts, ber_rho_sim, "o", label="Monte Carlo")
    plt.axvline(rho_star_theory, color="k", linestyle="--", label=f"worst-case rho* = {rho_star_theory:.2f}")
    plt.xlabel("Jammer band fraction rho")
    plt.ylabel("Bit error rate")
    plt.title(f"BER vs. rho at J/S = {JS_FOR_RHO_PLOT_DB:.0f} dB (Eb/N_J = {gp_db - JS_FOR_RHO_PLOT_DB:.0f} dB)")
    plt.grid(True, which="both")
    plt.legend()
    plt.savefig(FIG_DIR / "s1_ber_vs_rho.png", dpi=200, bbox_inches="tight")
    plt.close()

    print(f"\nSaved figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
