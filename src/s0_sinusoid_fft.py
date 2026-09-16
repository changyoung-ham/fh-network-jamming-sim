"""
S0-0 - Sinusoid in white Gaussian noise: time domain and spectrum
================================================================

Purpose
-------
Smallest end-to-end check of the signal chain used everywhere else in this
repository: build a discrete-time signal, add white Gaussian noise with a
fixed seed, take an FFT, and verify three numbers against theory:

    peak frequency     == F0
    peak magnitude     ~= A            (after the 2/N normalisation)
    noise-floor level  ~= 1.77 * SIGMA / sqrt(N)   (mean bin magnitude of white noise)

Outputs
-------
figures/s0_sinusoid_time.png
figures/s0_sinusoid_spectrum.png

Parameters
----------
Pedagogical values only; not specifications of any real system.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Parameters (assumed values)
# ---------------------------------------------------------------------------
FS = 1000.0     # sampling rate [Hz]; chosen so that FS > 2*F0 (Nyquist)
N = 1000        # number of samples; duration N/FS = 1 s, resolution FS/N = 1 Hz
F0 = 50.0       # sinusoid frequency [Hz]; integer multiple of FS/N to avoid leakage
A = 1.0         # amplitude
SIGMA = 0.5     # noise standard deviation
SEED = 42

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)


def main():
    # Time axis: sample index divided by sampling rate. np.arange avoids the
    # end-point pitfall of np.linspace(0, T, N), which includes t = T.
    t = np.arange(N) / FS
    clean = A * np.sin(2 * np.pi * F0 * t)

    rng = np.random.default_rng(SEED)
    noise = rng.normal(loc=0.0, scale=SIGMA, size=N)   # scale = standard deviation
    noisy = clean + noise                              # r = s + n (additive noise)

    # Real-valued FFT: only the non-negative frequencies are returned.
    spectrum = np.fft.rfft(noisy)
    freqs = np.fft.rfftfreq(N, d=1 / FS)
    # Normalise so that a sinusoid of amplitude A shows a peak of height A:
    # divide by N (FFT sums N samples) and multiply by 2 (energy is split
    # between +F0 and -F0, and rfft keeps only the +F0 half).
    magnitude = np.abs(spectrum) * 2 / N

    # --- numerical verification -------------------------------------------
    k = np.argmax(magnitude)
    snr_db = 10 * np.log10((A ** 2 / 2) / SIGMA ** 2)
    print(f"SNR                 : {snr_db:.2f} dB")
    print(f"peak frequency [Hz] : {freqs[k]:.3f}   expected {F0}")
    print(f"peak magnitude      : {magnitude[k]:.4f}  expected ~{A}")
    print(f"noise floor (median): {np.median(magnitude):.4f}  expected ~{1.77 * SIGMA / np.sqrt(N):.4f}")

    # --- figure 1: time domain ------------------------------------------------
    plt.figure(figsize=(9, 3))
    plt.plot(t[:200], clean[:200], label="clean")
    plt.plot(t[:200], noisy[:200], label="clean + noise", alpha=0.7)
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.title(f"Sinusoid F0 = {F0:.0f} Hz with AWGN (sigma = {SIGMA}, SNR = {snr_db:.1f} dB)")
    plt.legend()
    plt.grid(True)
    plt.savefig(FIG_DIR / "s0_sinusoid_time.png", dpi=200, bbox_inches="tight")
    plt.close()

    # --- figure 2: spectrum --------------------------------------------------
    plt.figure(figsize=(9, 4))
    plt.plot(freqs, magnitude)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Magnitude")
    plt.title("Spectrum of the noisy sinusoid")
    plt.grid(True)
    plt.savefig(FIG_DIR / "s0_sinusoid_spectrum.png", dpi=200, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
