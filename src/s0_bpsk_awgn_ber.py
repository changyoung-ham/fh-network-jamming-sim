"""
S0 - BPSK over AWGN: Monte Carlo bit error rate vs. closed-form theory
=====================================================================

Purpose
-------
This is the calibration step of the whole project. Before simulating jamming
(S1) or multi-node networks (S2), the simulator must reproduce a result whose
exact answer is known: the bit error rate (BER) of BPSK in additive white
Gaussian noise (AWGN),

    BER_theory = Q( sqrt(2 * Eb/N0) ) = 0.5 * erfc( sqrt(Eb/N0) ).

If the Monte Carlo curve does not fall on top of this formula, the simulator is
wrong, not "approximately right". Every later stage is built on top of this one.

Outputs
-------
figures/s0_ber_curve.png       Monte Carlo BER vs. theory (log y-axis)
figures/s0_received_hist.png   histogram of received samples at two Eb/N0 values
results/s0_ber_results.csv     numeric results used for the plot

Reproducibility
---------------
All randomness comes from a single numpy Generator with a fixed seed, so
re-running this script reproduces the same numbers and figures exactly.

Parameters
----------
All parameters are pedagogical values, NOT specifications of any real radio.
"""

import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc

# ---------------------------------------------------------------------------
# Parameters (assumed / pedagogical values; see docstring)
# ---------------------------------------------------------------------------
SEED = 2026                       # fixed seed for reproducibility
EBN0_DB = np.arange(0, 10, 1.0)   # Eb/N0 sweep [dB]; 0..9 dB in 1 dB steps

BITS_PER_CHUNK = 1_000_000        # bits simulated per chunk
MIN_ERRORS = 200                  # stop a point once this many errors are seen
MAX_BITS = 40_000_000             # hard cap per point (limits run time)

# Where to write outputs (paths relative to the repository root)
ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
RES_DIR = ROOT / "results"
FIG_DIR.mkdir(exist_ok=True)
RES_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def db_to_linear(value_db):
    """Convert a power ratio in dB to a linear ratio (10*log10 convention)."""
    return 10.0 ** (value_db / 10.0)


def ber_bpsk_theory(ebn0_linear):
    """Closed-form BER of coherent BPSK in AWGN.

    Q(x) = 0.5 * erfc(x / sqrt(2)), and BER = Q(sqrt(2*Eb/N0)),
    which simplifies to 0.5 * erfc(sqrt(Eb/N0)).
    """
    return 0.5 * erfc(np.sqrt(ebn0_linear))


def simulate_ber_point(ebn0_db, rng):
    """Monte Carlo BER for one Eb/N0 value.

    Model (per bit):
        bit b in {0, 1}
        symbol s = +1 if b == 1 else -1        (energy per bit Eb = 1)
        received r = s + n,  n ~ N(0, sigma^2)
        decision  b_hat = 1 if r > 0 else 0

    Noise variance:
        With Eb = 1, the one-sided noise PSD is N0 = Eb / (Eb/N0).
        A real-valued AWGN sample has variance N0/2, so
        sigma = sqrt(N0 / 2) = sqrt(1 / (2 * Eb/N0)).
        Getting this factor of 2 wrong is the most common reason a simulated
        BER curve does not match theory.

    Returns (ber, n_bits, n_errors).
    """
    ebn0_lin = db_to_linear(ebn0_db)
    sigma = np.sqrt(1.0 / (2.0 * ebn0_lin))

    n_bits = 0
    n_errors = 0
    while n_errors < MIN_ERRORS and n_bits < MAX_BITS:
        bits = rng.integers(0, 2, size=BITS_PER_CHUNK)          # 0/1 with equal probability
        symbols = 2 * bits - 1                                    # map 0 -> -1, 1 -> +1
        noise = rng.normal(loc=0.0, scale=sigma, size=BITS_PER_CHUNK)
        received = symbols + noise                                # r = s + n
        decisions = (received > 0).astype(int)                    # threshold at 0
        n_errors += int(np.sum(decisions != bits))                # count mismatches
        n_bits += BITS_PER_CHUNK

    return n_errors / n_bits, n_bits, n_errors


# ---------------------------------------------------------------------------
# Main: sweep Eb/N0, compare with theory, save results and figures
# ---------------------------------------------------------------------------
def main():
    rng = np.random.default_rng(SEED)

    ber_sim = np.zeros_like(EBN0_DB)
    ber_theory = ber_bpsk_theory(db_to_linear(EBN0_DB))
    rows = []

    print(f"{'Eb/N0[dB]':>9} {'BER_sim':>12} {'BER_theory':>12} {'ratio':>7} {'bits':>10} {'errors':>7}")
    for i, ebn0_db in enumerate(EBN0_DB):
        ber, n_bits, n_err = simulate_ber_point(ebn0_db, rng)
        ber_sim[i] = ber
        ratio = ber / ber_theory[i]
        rows.append((ebn0_db, ber, ber_theory[i], ratio, n_bits, n_err))
        print(f"{ebn0_db:9.1f} {ber:12.3e} {ber_theory[i]:12.3e} {ratio:7.3f} {n_bits:10d} {n_err:7d}")

    # -----------------------------------------------------------------------
    # Verification: the ratio sim/theory must stay close to 1.
    # With >= 200 errors per point the statistical spread of the ratio is a
    # few percent; a systematic deviation (all ratios > 1.3 or < 0.7, or a
    # different slope) indicates a bug, not noise.
    # -----------------------------------------------------------------------
    ratios = ber_sim / ber_theory
    ok = np.all((ratios > 0.7) & (ratios < 1.4))
    print("\nVERIFICATION:", "PASS" if ok else "FAIL",
          f"(ratio range {ratios.min():.3f} .. {ratios.max():.3f})")

    # Save numeric results
    with open(RES_DIR / "s0_ber_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ebn0_db", "ber_sim", "ber_theory", "ratio", "n_bits", "n_errors"])
        w.writerows(rows)

    # -----------------------------------------------------------------------
    # Figure 1: BER curve, simulation vs theory
    # -----------------------------------------------------------------------
    plt.figure(figsize=(7, 4.5))
    plt.semilogy(EBN0_DB, ber_theory, "k-", label="Theory: Q(sqrt(2 Eb/N0))")
    plt.semilogy(EBN0_DB, ber_sim, "o", label="Monte Carlo simulation")
    plt.xlabel("Eb/N0 [dB]")
    plt.ylabel("Bit error rate")
    plt.title("BPSK over AWGN: simulation vs. theory")
    plt.grid(True, which="both")
    plt.legend()
    plt.savefig(FIG_DIR / "s0_ber_curve.png", dpi=200, bbox_inches="tight")
    plt.close()

    # -----------------------------------------------------------------------
    # Figure 2: what the receiver actually sees.
    # Histogram of received samples r = s + n for two Eb/N0 values.
    # Errors are the parts of each bell that cross the decision threshold r=0.
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, ebn0_db in zip(axes, (0.0, 6.0)):
        sigma = np.sqrt(1.0 / (2.0 * db_to_linear(ebn0_db)))
        n = 200_000
        bits = rng.integers(0, 2, size=n)
        received = (2 * bits - 1) + rng.normal(0.0, sigma, size=n)
        ax.hist(received[bits == 0], bins=120, density=True, alpha=0.6, label="sent -1 (bit 0)")
        ax.hist(received[bits == 1], bins=120, density=True, alpha=0.6, label="sent +1 (bit 1)")
        ax.axvline(0.0, color="k", linestyle="--", label="decision threshold")
        ax.set_title(f"Eb/N0 = {ebn0_db:.0f} dB  (sigma = {sigma:.2f})")
        ax.set_xlabel("received sample r")
        ax.grid(True)
    axes[0].set_ylabel("probability density")
    axes[1].legend(loc="upper right", fontsize=8)
    fig.suptitle("Received samples: overlap across the threshold is the error probability")
    fig.savefig(FIG_DIR / "s0_received_hist.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved: {FIG_DIR / 's0_ber_curve.png'}")
    print(f"Saved: {FIG_DIR / 's0_received_hist.png'}")
    print(f"Saved: {RES_DIR / 's0_ber_results.csv'}")


if __name__ == "__main__":
    main()
