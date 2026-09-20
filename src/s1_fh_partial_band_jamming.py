"""
S1 - Frequency hopping under partial-band noise jamming
=======================================================

Purpose
-------
Extend the S0 link (BPSK/AWGN) with frequency hopping over N_CH channels and a
partial-band noise jammer that spreads its total power J over a fraction rho
of the hopping band. Verify the Monte Carlo simulator against the closed-form
mixture formula, then use the formula to map BER vs. J/S, to find the jammer's
worst-case rho, and to compute the jamming margin.

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

The simulation draws one bit per hop so that successive bits are jammed
independently. The AVERAGE BER does not depend on how many bits share a hop;
only the burstiness of the errors does, and that is not modelled here.

Link budget relations (all power ratios, 10*log10 convention):

    processing gain   G_p [dB] = 10 log10( W_ss / W_data ) = 10 log10( N_CH )
    Eb/N_J [dB]       = G_p [dB] - (J/S) [dB]
    jamming margin    M_J [dB] = G_p - (Eb/N_J)_required - L_sys

Treatment of rho
----------------
rho is optimised as a continuous variable, which matches the textbook
constants (rho* ~ 0.709/(Eb/N_J), worst BER ~ 0.083/(Eb/N_J)). A real jammer
facing N_CH channels cannot jam less than one channel, so rho >= 1/N_CH.
Results are reported for both cases; with N_CH = 100 they differ by ~0.01 dB.

Verification
------------
1. Monte Carlo BER must match the closed-form mixture at every (J/S, rho)
   point: z-score |z| < 4 at each point (same test as S0).
2. rho = 1 must reduce to the S0 AWGN result with N0 replaced by N0 + N_J.
3. With thermal noise removed, the numerical optimiser must reproduce the
   analytical constants rho* x (Eb/N_J) = 0.709 and BER_worst x (Eb/N_J) = 0.0829.

Outputs
-------
figures/s1_ber_vs_js.png        BER vs J/S for several rho, plus worst-case envelope
figures/s1_ber_vs_rho.png       BER vs rho at fixed J/S (shows the worst-case rho)
results/s1_ber_vs_js.csv
results/s1_link_budget.txt

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
N_CH = 100                 # hop channels -> processing gain 20 dB; also sets the smallest rho = 1/N_CH
EBN0_DB = 10.0             # thermal Eb/N0 [dB]: thermal-only BER = 3.9e-6 << outage BER,
                           # so every outage in this study is caused by the jammer
JS_DB = np.arange(0, 22, 2.0)            # J/S sweep [dB]
RHO_LIST = [0.1, 0.3, 0.6, 1.0]          # jammer band fractions shown in figure 1
JS_FOR_RHO_PLOT_DB = 12.0                # J/S used in the BER-vs-rho figure
BER_REQUIRED = 1e-3                      # link-outage criterion (assumed; sensitivity discussed in the S2 note)
L_SYS_DB = 2.0                           # implementation loss (assumed; shifts both margins equally)
RHO_MIN_PHYSICAL = 1.0 / N_CH            # one channel

BITS_PER_CHUNK = 500_000
MIN_ERRORS = 200
MAX_BITS = 20_000_000
Z_LIMIT = 4.0

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

    ebn0_lin : thermal Eb/N0 (linear); np.inf removes thermal noise
    ebnj_lin : Eb/N_J with the jammer spread over the FULL band (linear)
    rho      : fraction of the band the jammer actually occupies (0 < rho <= 1)
    """
    n0 = 1.0 / ebn0_lin                       # Eb = 1 -> N0 = 1/(Eb/N0)
    nj = 1.0 / ebnj_lin                       # full-band equivalent jammer density
    ber_clear = ber_bpsk(ebn0_lin)
    ber_jammed = ber_bpsk(1.0 / (n0 + nj / rho))
    return (1.0 - rho) * ber_clear + rho * ber_jammed


def worst_case_rho(ebn0_lin, ebnj_lin, rho_min=1e-5):
    """Jammer-optimal band fraction: the rho in [rho_min, 1] that maximises the BER.

    The BER has a single maximum in rho, so a bounded one-dimensional search
    (on log10 rho, because the optimum can be very small) is sufficient.
    Returns (rho_star, ber_at_rho_star).
    """
    res = minimize_scalar(lambda lr: -ber_partial_band_theory(ebn0_lin, ebnj_lin, 10.0 ** lr),
                          bounds=(np.log10(rho_min), 0.0), method="bounded",
                          options={"xatol": 1e-8})
    candidates = [10.0 ** res.x, rho_min, 1.0]            # interior optimum or either end point
    bers = [ber_partial_band_theory(ebn0_lin, ebnj_lin, r) for r in candidates]
    i = int(np.argmax(bers))
    return candidates[i], bers[i]


def required_ebnj_db(ebn0_lin, ber_target, rho_min=None, full_band=False):
    """Smallest Eb/N_J [dB] at which the link meets ber_target.

    full_band=True : jammer spreads over the whole band (rho = 1)
    otherwise      : jammer uses its worst-case rho >= rho_min
    """
    def excess(ebnj_db):
        ebnj_lin = db_to_linear(ebnj_db)
        if full_band:
            ber = ber_partial_band_theory(ebn0_lin, ebnj_lin, 1.0)
        else:
            ber = worst_case_rho(ebn0_lin, ebnj_lin, rho_min)[1]
        return ber - ber_target
    return brentq(excess, -5.0, 40.0)


def simulate_partial_band(ebn0_lin, ebnj_lin, rho, rng):
    """Monte Carlo BER: one bit per hop, random jammed flag per hop.

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
        jammed = rng.random(BITS_PER_CHUNK) < rho               # which hops land in the jammed part
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
    req_full = required_ebnj_db(ebn0_lin, BER_REQUIRED, full_band=True)
    req_cont = required_ebnj_db(ebn0_lin, BER_REQUIRED, rho_min=1e-5)
    req_phys = required_ebnj_db(ebn0_lin, BER_REQUIRED, rho_min=RHO_MIN_PHYSICAL)
    rho_at_req_cont = worst_case_rho(ebn0_lin, db_to_linear(req_cont), 1e-5)[0]
    rho_at_req_phys = worst_case_rho(ebn0_lin, db_to_linear(req_phys), RHO_MIN_PHYSICAL)[0]

    margin_full = gp_db - req_full - L_SYS_DB
    margin_cont = gp_db - req_cont - L_SYS_DB
    margin_phys = gp_db - req_phys - L_SYS_DB

    budget = [
        f"Hop channels N_CH                    : {N_CH}",
        f"Processing gain G_p                  : {gp_db:.2f} dB",
        f"Thermal Eb/N0                        : {EBN0_DB:.1f} dB",
        f"Implementation loss L_sys            : {L_SYS_DB:.1f} dB",
        f"Outage criterion BER                 : {BER_REQUIRED:g}",
        "",
        f"Required Eb/N_J, rho = 1             : {req_full:.2f} dB  (full-band jammer)",
        f"Required Eb/N_J, worst rho           : {req_cont:.2f} dB  (continuous rho, optimum rho = {rho_at_req_cont:.4f})",
        f"Required Eb/N_J, worst rho >= 1/N_CH : {req_phys:.2f} dB  (optimum rho = {rho_at_req_phys:.4f})",
        f"Jamming margin,  rho = 1             : {margin_full:.2f} dB  -> link survives J/S up to this value",
        f"Jamming margin,  worst rho           : {margin_cont:.2f} dB",
        f"Jamming margin,  worst rho >= 1/N_CH : {margin_phys:.2f} dB",
        f"Margin lost to partial-band jamming  : {margin_full - margin_cont:.2f} dB",
    ]
    print("\n".join(budget))
    (RES_DIR / "s1_link_budget.txt").write_text("\n".join(budget) + "\n")

    # --- Sweep 1: BER vs J/S for several rho (Monte Carlo vs theory) --------
    print(f"\n{'J/S[dB]':>7} {'rho':>5} {'BER_sim':>10} {'BER_theory':>11} {'ratio':>6} {'errors':>7} {'z':>6}")
    rows = []
    z_all = []
    sim = {r: np.zeros(len(JS_DB)) for r in RHO_LIST}
    for j, js_db in enumerate(JS_DB):
        ebnj_lin = db_to_linear(gp_db - js_db)
        for r in RHO_LIST:
            t = ber_partial_band_theory(ebn0_lin, ebnj_lin, r)
            s, nb, ne = simulate_partial_band(ebn0_lin, ebnj_lin, r, rng)
            sim[r][j] = s
            z = (ne - nb * t) / np.sqrt(nb * t)
            z_all.append(z)
            rows.append((js_db, r, s, t, s / t, nb, ne, z))
            print(f"{js_db:7.1f} {r:5.2f} {s:10.3e} {t:11.3e} {s / t:6.3f} {ne:7d} {z:6.2f}")

    z_all = np.array(z_all)
    ok1 = bool(np.all(np.abs(z_all) < Z_LIMIT))
    print(f"\nVERIFICATION 1 (Monte Carlo vs closed form): {'PASS' if ok1 else 'FAIL'} "
          f"(max |z| = {np.abs(z_all).max():.2f} over {len(z_all)} points, limit {Z_LIMIT:.0f})")

    # Verification 2: rho = 1 reduces to AWGN with N0 + N_J
    ebnj_lin = db_to_linear(gp_db - 10.0)
    n_eff = 1.0 / ebn0_lin + 1.0 / ebnj_lin
    ber_rho1 = ber_partial_band_theory(ebn0_lin, ebnj_lin, 1.0)
    ber_awgn = ber_bpsk(1.0 / n_eff)
    ok2 = bool(np.isclose(ber_rho1, ber_awgn, rtol=1e-12))
    print(f"VERIFICATION 2 (rho=1 equals AWGN with N0+N_J): {'PASS' if ok2 else 'FAIL'} "
          f"({ber_rho1:.4e} vs {ber_awgn:.4e})")

    # Verification 3: optimiser vs analytical constants, thermal noise removed
    ebnj_test = db_to_linear(15.0)
    rho_opt, ber_opt = worst_case_rho(np.inf, ebnj_test)
    c_rho, c_ber = rho_opt * ebnj_test, ber_opt * ebnj_test
    ok3 = bool(abs(c_rho - 0.7088) < 0.002 and abs(c_ber - 0.08286) < 0.0001)
    print(f"VERIFICATION 3 (no thermal noise): rho* x Eb/N_J = {c_rho:.4f} (theory 0.7088), "
          f"BER_worst x Eb/N_J = {c_ber:.5f} (theory 0.08286) -> {'PASS' if ok3 else 'FAIL'}")

    with open(RES_DIR / "s1_ber_vs_js.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["js_db", "rho", "ber_sim", "ber_theory", "ratio", "n_bits", "n_errors", "z_score"])
        w.writerows(rows)

    # --- Sweep 2: BER vs rho at fixed J/S; worst-case rho ------------------
    ebnj_lin = db_to_linear(gp_db - JS_FOR_RHO_PLOT_DB)
    rho_star, ber_star = worst_case_rho(ebn0_lin, ebnj_lin)
    ber_full = ber_partial_band_theory(ebn0_lin, ebnj_lin, 1.0)
    rho_curve = np.linspace(RHO_MIN_PHYSICAL, 1.0, 400)
    ber_curve = ber_partial_band_theory(ebn0_lin, ebnj_lin, rho_curve)

    rho_sim_pts = np.linspace(0.05, 1.0, 12)
    ber_rho_sim = np.array([simulate_partial_band(ebn0_lin, ebnj_lin, r, rng)[0] for r in rho_sim_pts])
    print(f"\nWorst-case rho at J/S = {JS_FOR_RHO_PLOT_DB:.0f} dB: rho* = {rho_star:.3f} "
          f"(no-thermal formula 0.709/(Eb/N_J) = {0.7088 / ebnj_lin:.3f})")
    print(f"  BER(rho=1) = {ber_full:.2e}, BER(rho*) = {ber_star:.2e}  (x{ber_star / ber_full:.1f})")

    print("\nOVERALL:", "PASS" if (ok1 and ok2 and ok3) else "FAIL")

    # --- Figure 1: BER vs J/S --------------------------------------------------
    js_fine = np.linspace(JS_DB[0], JS_DB[-1], 200)
    envelope = np.array([worst_case_rho(ebn0_lin, db_to_linear(gp_db - js))[1] for js in js_fine])
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
    plt.semilogy(rho_curve, ber_curve, "-", label="closed form (rho >= 1/N_CH)")
    plt.semilogy(rho_sim_pts, ber_rho_sim, "o", label="Monte Carlo")
    plt.axvline(rho_star, color="k", linestyle="--", label=f"worst-case rho* = {rho_star:.3f}")
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
