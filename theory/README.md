# Theory notes

Derivations behind each simulation stage, written so that every symbol in the
scripts can be traced to an equation and every verification check can be read
as a test of a stated result.

| Note | Mathematics | What it derives | Accompanies |
|------|-------------|-----------------|-------------|
| [S0 — BPSK/AWGN BER](s0_bpsk_awgn_ber.pdf) | Gaussian tail, conditional probability, binomial estimator | BER = Q(√(2Eb/N0)); why σ² = N0/2; why 200 errors per point; the z-score pass criterion | `src/s0_bpsk_awgn_ber.py` |
| [S1 — Partial-band jamming](s1_partial_band_jamming.pdf) | Total probability, one-variable optimisation | Processing gain; Eb/N_J = G_p − J/S; mixture BER; jammer-optimal ρ* ≈ 0.709/(Eb/N_J); worst-case BER ≈ 0.083/(Eb/N_J); outage band fraction ρ_out ≈ 8.55·BER_out; jamming margin 8.39 → −1.54 dB | `src/s1_fh_partial_band_jamming.py` |
| [S2 — Laplacian and λ₂](s2_laplacian_lambda2.pdf) | Symmetric matrices, quadratic forms, Courant–Fischer | λ₂ > 0 ⇔ connected (proof); K_N ⇒ λ₂ = N; Fiedler bound λ₂ ≤ N/(N−1)·δ_min (proof); link range and why the jammer's best ρ does not depend on N; closed-form collapse threshold | `src/s2_network_lambda2.py` |

Source (`.tex`) is included; PDFs are committed so they render on GitHub.
References are listed at the end of the S1 and S2 notes.
