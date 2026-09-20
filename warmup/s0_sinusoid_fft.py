"""
Warm-up - Sinusoid in white Gaussian noise: time domain and spectrum
====================================================================

A first NumPy exercise written before S0. It is not part of the S0-S2 model
chain (none of the later stages uses an FFT). It is kept because it fixes
three conventions the later scripts rely on: a seeded random generator,
`scale` = standard deviation in rng.normal, and additive noise r = s + n.

Three numbers are checked against theory:

    peak frequency   == F0
    peak magnitude   ~= A                        (after the 2/N normalisation)
    noise floor      ~= sqrt(pi) * SIGMA / sqrt(N)

Noise floor: for white Gaussian noise each FFT bin is complex Gaussian with
total variance N*SIGMA^2, so its magnitude is Rayleigh distributed with mean
(sqrt(pi)/2) * sqrt(N) * SIGMA. After the 2/N normalisation the MEAN bin
magnitude is sqrt(pi) * SIGMA / sqrt(N) = 1.772 * SIGMA / sqrt(N).
The check uses the mean of the noise-only bins (DC, the F0 bin and the
Nyquist bin are excluded). The median would be about 6 % lower
(Rayleigh median / mean = 0.939), so it must not be compared with this formula.

Outputs (written next to this script)
-------
warmup/s0_sinusoid_time.png
warmup/s0_sinusoid_spectrum.png
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Parameters (assumed values)
# ---------------------------------------------------------------------------
FS = 1000.0     # sampling rate [Hz]; FS > 2*F0 (Nyquist)
N = 1000        # number of samples; duration N/FS = 1 s, resolution FS/N = 1 Hz
F0 = 50.0       # sinusoid frequency [Hz]; integer multiple of FS/N to avoid leakage
A = 1.0         # amplitude
SIGMA = 0.5     # noise standard deviation
SEED = 42

OUT_DIR = Path(__file__).resolve().parent


def main():
    # Time axis: sample index divided by sampling rate (N samples, t = 0 .. (N-1)/FS).
    t = np.arange(N) / FS
    clean = A * np.sin(2 * np.pi * F0 * t)

    rng = np.random.default_rng(SEED)
    noise = rng.normal(loc=0.0, scale=SIGMA, size=N)   # scale = standard deviation
    noisy = clean + noise                              # r = s + n (additive noise)

    # Real-valued FFT: only the non-negative frequencies are returned.
    spectrum = np.fft.rfft(noisy)
    freqs = np.fft.rfftfreq(N, d=1 / FS)
    # Normalise so that a sinusoid of amplitude A shows a peak of height A:
    # divide by N (FFT sums N samples) and multiply by 2 (rfft keeps only +F0).
    magnitude = np.abs(spectrum) * 2 / N

    # --- numerical verification -------------------------------------------
    k = int(np.argmax(magnitude))
    noise_bins = np.ones(len(magnitude), dtype=bool)
    noise_bins[[0, k, len(magnitude) - 1]] = False      # drop DC, signal bin, Nyquist
    floor_meas = magnitude[noise_bins].mean()
    floor_theory = np.sqrt(np.pi) * SIGMA / np.sqrt(N)
    snr_db = 10 * np.log10((A ** 2 / 2) / SIGMA ** 2)

    ok_freq = freqs[k] == F0
    ok_peak = abs(magnitude[k] - A) < 4 * floor_theory            # peak = A +/- the noise in that bin
    # Mean of ~N/2 Rayleigh bins: relative std = 0.523 / sqrt(N/2) ~ 2.3 %, so 10 % is about 4 sigma.
    ok_floor = abs(floor_meas / floor_theory - 1.0) < 0.10
    print(f"SNR                 : {snr_db:.2f} dB")
    print(f"peak frequency [Hz] : {freqs[k]:.3f}   expected {F0}     -> {'PASS' if ok_freq else 'FAIL'}")
    print(f"peak magnitude      : {magnitude[k]:.4f}  expected ~{A}     -> {'PASS' if ok_peak else 'FAIL'}")
    print(f"noise floor (mean)  : {floor_meas:.4f}  expected ~{floor_theory:.4f} -> {'PASS' if ok_floor else 'FAIL'}")
    print("OVERALL:", "PASS" if (ok_freq and ok_peak and ok_floor) else "FAIL")

    # --- figure 1: time domain ------------------------------------------------
    plt.figure(figsize=(9, 3))
    plt.plot(t[:200], clean[:200], label="clean")
    plt.plot(t[:200], noisy[:200], label="clean + noise", alpha=0.7)
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.title(f"Sinusoid F0 = {F0:.0f} Hz with AWGN (sigma = {SIGMA}, SNR = {snr_db:.1f} dB)")
    plt.legend()
    plt.grid(True)
    plt.savefig(OUT_DIR / "s0_sinusoid_time.png", dpi=200, bbox_inches="tight")
    plt.close()

    # --- figure 2: spectrum --------------------------------------------------
    plt.figure(figsize=(9, 4))
    plt.plot(freqs, magnitude)
    plt.axhline(floor_theory, color="k", linestyle=":", label="theoretical mean noise floor")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Magnitude")
    plt.title("Spectrum of the noisy sinusoid")
    plt.legend()
    plt.grid(True)
    plt.savefig(OUT_DIR / "s0_sinusoid_spectrum.png", dpi=200, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
