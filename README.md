# Connectivity of a frequency-hopping network under partial-band jamming

As a signal officer I operated multi-node frequency-hopping VHF radio nets.
Hopping was our main protection against jamming, and when links began to fail
we judged link by link, largely from experience, whether the net as a whole
would hold. This project is my attempt to put numbers on that question: how
much jamming a hopping network can absorb before it splits apart.

Simulation study of how the connectivity of a multi-node frequency-hopping
network degrades under partial-band jamming, measured by the algebraic
connectivity [1] (second-smallest Laplacian eigenvalue, λ₂) of the link graph.

This is a self-directed learning project built in stages. Each stage is
verified against a closed-form result or a known limiting case before the next
stage is started. All physical parameters are assumed values, listed with their
reasons in [Parameters](#parameters); nothing here models a specific radio.

## Main result

![lambda2 vs J/S](figures/s2_lambda2_vs_js.png)

N nodes are placed at random in a square of side L and a distant partial-band
jammer uses the band fraction ρ that damages the network most. J/S\* is the
jammer-to-signal ratio, referred to a 100 m link, at which half of 500 random
placements are no longer connected. The values are negative only because of
that reference: a longer link receives a weaker signal and therefore a higher
J/S. At J/S\*, the longest surviving link (0.42 L for N = 10) receives
J/S ≈ +0.7 dB, and about +0.8 dB for N = 20 and 30. This is the single-link
outage condition of S1: the worst-case BER reaches 10⁻³ at Eb/N_J ≈ 19.2–19.5 dB,
i.e. at a received J/S of +0.5 to +0.8 dB depending on the thermal noise.

| N | J/S\* (95 % interval) | link range at J/S\* | closed form (S2 note, Eq. 2) | 50 % knee of mean λ₂ |
|---|---|---|---|---|
| 10 | −11.8 dB (−12.0 .. −11.6) | 0.42 L | −11.7 dB | −17.4 dB |
| 20 | −9.4 dB (−9.6 .. −9.3) | 0.32 L | −9.4 dB | −17.2 dB |
| 30 | −7.9 dB (−8.1 .. −7.8) | 0.27 L | −7.9 dB | −17.2 dB |

- Ten more nodes buy 2.4 dB of jamming tolerance from N = 10 to 20 and 1.5 dB
  from 20 to 30: the gain per added node shrinks.
- λ₂ has lost half of its unjammed value about 6–9 dB *before* the network
  disconnects, and that knee is nearly the same for all three sizes.
- The thresholds follow from two known ingredients: the inverse-linear
  worst-case BER law of partial-band jamming (S1) and the link range needed to
  keep N random points connected. The closed form built from them agrees with
  the simulation within 0.11 dB.

## Motivation

In a multi-node wireless network, individual links fail as jamming intensifies.
Counting failed links does not tell you whether the network is still one piece.
The Laplacian eigenvalue λ₂ does: it is zero exactly when the graph is
disconnected, and it sets the convergence speed of consensus-type coordination
[2], so it already shrinks while the network is still in one piece. The
question this project asks is

> As a partial-band jammer raises its power and picks its most damaging band
> fraction ρ, at what J/S does the network stop being connected, and how does
> that threshold depend on the number of nodes and of hopping channels?

The chain of models is

```
bit → AWGN → Eb/N0 → BER → hopping → partial-band jamming
    → per-link outage → adjacency matrix → Laplacian → λ₂ → collapse threshold J/S*
```

The dependence on the number of hopping channels is answered by the closed
form rather than by a sweep: every threshold moves by +10 dB per tenfold
increase of N_CH (S2 note, Section 7).

## Stages

| Stage | Content | Verification | Status |
|------:|---------|--------------|--------|
| S0 | BPSK over AWGN, Monte Carlo BER | Must match Q(√(2Eb/N0)); z-score of every point | **done** (max \|z\| = 2.2 over 10 points) |
| S1 | FH + partial-band jammer, BER vs J/S, jamming margin | Monte Carlo vs closed-form mixture at 44 points; ρ=1 limit; optimiser vs analytical constants 0.709 / 0.0829 | **done** (max \|z\| = 1.9) |
| S2 | N-node graph, per-link outage → λ₂ vs ρ and vs J/S | K_N ⇒ λ₂=N; λ₂>0 ⇔ BFS-connected; Fiedler bound [1]; monotonicity; BER rule = distance rule; closed-form threshold | **done** (7 checks pass) |
| S3a | Detecting the jammer: energy test on known pilots per hop, count of flagged hops per window; hops needed vs ρ | Monte Carlo vs chi-square (12 points) and binomial (10 points); strong-jammer limit; count detector never beats the optimal pooled-energy test at ρ = 1 | **done** (4 checks pass) |

## Repository layout

```
src/        simulation scripts, one per stage
theory/     derivation notes (LaTeX + PDF), one per stage — see theory/README.md
figures/    generated figures (committed so results are viewable without running)
results/    numeric results (CSV, TXT)
warmup/     first NumPy/FFT exercise, not part of the S0–S3a chain
environment.yml
```

## Theory notes

Each stage has a short derivation note in [`theory/`](theory/README.md) that
maps the code to the equations and states what each verification check tests:
the BPSK/AWGN error probability and the z-score test (S0), the jammer's optimal
band fraction, the inverse-linear worst-case BER law and the band fraction that
causes an outage (S1), the Laplacian eigenvalue theorems, the link-range
argument and the closed-form threshold (S2), and the energy test on pilots, why
it is the best test for one hop, and the number of hops a window detector needs
(S3a).

## Reproducing the results

```bash
conda env create -f environment.yml
conda activate fh-jamming-sim
python src/s0_bpsk_awgn_ber.py
python src/s1_fh_partial_band_jamming.py
python src/s2_network_lambda2.py
python src/s3a_jammer_detection.py
```

Run times on a laptop: S0 ~10 s, S1 ~1 min, S2 under 1 min, S3a ~1–2 min.
All scripts use a fixed random seed, so the figures in `figures/` and the
numbers in `results/` are reproduced exactly. Each script prints `PASS` or
`FAIL` for its verification checks.

## S0 result

![BER curve](figures/s0_ber_curve.png)

Simulated BER matches the theoretical curve at every point. Each point is
tested by its z-score (observed minus expected errors, in standard deviations);
the ten values lie between −2.1 and +2.2. The second figure shows why errors
occur: the two received-sample distributions overlap across the decision
threshold, and the overlap shrinks as Eb/N0 grows.

![Received samples](figures/s0_received_hist.png)

## S1 result — the right band fraction costs the link 9.9 dB

![BER vs J/S](figures/s1_ber_vs_js.png)

With N_CH = 100 hop channels (processing gain 20 dB) and thermal Eb/N0 = 10 dB:

| | required Eb/N_J for BER ≤ 10⁻³ | jamming margin (2 dB implementation loss) |
|---|---|---|
| full-band jammer (ρ = 1) | 9.61 dB | **8.39 dB** |
| jammer-optimal ρ | 19.54 dB | **−1.54 dB** |
| jammer-optimal ρ, at least one channel (ρ ≥ 0.01) | 19.53 dB | **−1.53 dB** |

Concentrating its power on the right fraction of the band costs the link 9.9 dB
of margin. Against the optimal ρ the BER falls only as 0.083/(Eb/N_J) instead
of exponentially (derived in the S1 note; partial-band and pulse noise jamming
are textbook material, e.g. [3], Sec. 12.6). At J/S = 12 dB the worst-case ρ is 0.135 and the BER is
5.2× that of a full-band jammer of the same power (figure
`s1_ber_vs_rho.png`). Monte Carlo points agree with the closed-form mixture at
all 44 (J/S, ρ) points.

## S2 result — which ρ, and how the network degrades

The main figure and table are at the top of this page.

![lambda2 vs rho](figures/s2_lambda2_vs_rho.png)

Because the jammer is far away, every receiver sees the same J and a link is up
exactly when it is shorter than a link range r(J/S, ρ); the link graph is a
random geometric graph [4]. The jammer's best ρ is the one that minimises that
range. It is ρ ≈ 8.55 × BER_out ≈ 0.009, the same for every N and placement.
This is the S1 formula ρ\* = 0.709/(Eb/N_J) evaluated at the outage point
(Eb/N_J = 19.2 dB) instead of at the 8 dB of the S1 figure, not a separate
network effect. It is just below 1/N_CH = 0.01, the smallest fraction a jammer
can occupy with 100 channels (shaded); enforcing that limit moves the
thresholds by 0.03 dB. Below ρ = 2 × BER_out no jammer power can bring a link
down, which is the rise on the left.

![example topologies](figures/s2_example_topologies.png)

## S3a result — how many hops it takes to notice the jammer

![hops needed vs rho](figures/s3a_hops_to_detect_vs_rho.png)

S0–S2 treat the jammer as known. In S3a each hop carries 16 known pilot
symbols; the receiver removes them and tests the energy of what is left, which
is the most powerful test for a single hop (S3a note, Section 2; energy
detection is classical [5]). It declares a jammer when at least k of M hops are
flagged. M\* is the smallest window that reaches detection probability 0.9 at
false-alarm probability 10⁻³.

| J/S | M\* at the S1 worst-case ρ | M\* at ρ = 1 (optimal pooled-energy test) | smallest M\* over ρ |
|---|---|---|---|
| 0.46 dB (S1 outage point) | 242 (ρ = 0.0095) | 337 (212) | 92 (ρ ≈ 0.04) |
| 12 dB | 16 (ρ = 0.135) | 4 (3) | 4 |

- At the jammer's worst-case ρ, every jammed hop is detected almost surely, so
  the delay is set only by how rarely a hop lands in the jammed band:
  M\* ≈ ln 10 / ρ. More pilots do not shorten it (242 hops with 16 or 32
  pilots, 261 with 8).
- Against a full-band jammer of the same power each hop changes little, and
  M\* falls roughly in proportion to the number of pilots (671 / 337 / 170 hops
  for 8 / 16 / 32).
- The count detector is not optimal when ρ < 1 (S3a note, Section 3), so these
  M\* describe this detector, not the best possible one. Counts are in hops;
  converting them to seconds needs a hop rate, which is not modelled.

The per-hop ROC, with Monte Carlo points, is in `figures/s3a_roc_per_hop.png`.

## Parameters

| Parameter | Value | Reason |
|---|---|---|
| Thermal Eb/N0 (S1) | 10 dB | thermal-only BER 3.9×10⁻⁶ ≪ 10⁻³, so every outage is caused by the jammer |
| N_CH | 100 | processing gain of 20 dB; also sets the smallest realisable ρ = 0.01 |
| Outage BER | 10⁻³ | assumed. Thresholds shift by 10 dB per decade of this value; the spacing between N = 10/20/30 does not depend on it |
| Implementation loss | 2 dB | assumed; shifts all jamming margins equally |
| Eb/N0 at 100 m (S2) | 27 dB | gives an unjammed link range of 1.02 km ≈ L: a dense but not complete unjammed network |
| Path-loss exponent | 2 | line of sight. The spacing between the N curves is proportional to this exponent |
| Area side L | 1 km | sets the scale only; results are also given as r/L |
| Placements per N | 500 | 95 % interval of J/S\* within ±0.2 dB (bootstrap) |
| Pilots per hop (S3a) | 16 | at the jammer's worst-case ρ a jammed hop is already detected almost surely: M\* is the same with 32 pilots and rises from 242 to 261 with 8 |
| Window target (S3a) | P_d ≥ 0.9, P_fa ≤ 10⁻³ | assumed |

## Assumptions and limitations

- Coherent BPSK is used because it continues the S0 model and has a closed-form
  BER. Frequency-hopping systems more commonly use noncoherent FSK; the same
  argument then gives a 1/(Eb/N_J) law with a different constant (S1 note,
  Section 3).
- AWGN and free-space path loss; no fading, shadowing, or timing/phase error.
- S0–S2 treat jamming as a known input with a fixed band fraction. S3a adds
  detection, but assumes the receiver knows the thermal noise level and the
  pilots (perfect channel knowledge), and uses a count detector that is not
  optimal for ρ < 1. Telling jamming apart from fading or hop collisions is not
  modelled.
- The jammer is far from all nodes (equal received J everywhere) and does not
  know the hopping pattern. A jammer at a finite distance would make J differ
  between nodes; the graph would then no longer be a plain random geometric graph.
- Nodes are static; no mobility, no re-routing, no channel re-allocation.
- Each node is assumed to observe the full link graph.
- Links do not interfere with one another: hop collisions between node pairs,
  adjacent-channel leakage and medium access are not modelled.
- The link ranges in the main table belong to a finite square with 10–30 nodes,
  where nodes near the edge have fewer neighbours; they should not be
  extrapolated to much larger networks.
- Link outage is a hard BER threshold; no coding or interleaving. The
  network-worst ρ and the absolute thresholds depend on this choice.

## Possible extensions (not started)

Each item removes one of the assumptions above.

- Channel re-allocation after the jammed band is detected, and how much of λ₂
  it recovers, with the S3a detection delay as a parameter.
- Telling jamming apart from fading and from hop collisions, with the S3a
  energy test as the baseline.
- An unknown or drifting thermal noise level at the receiver.
- A jammer at a finite distance, so that the received J differs between nodes.
- Noncoherent FSK and a coded packet-error criterion instead of a hard BER
  threshold.
- Fading and shadowing on the links.
- Moving nodes, and λ₂ estimated by each node from local information only.

## References

1. M. Fiedler, "Algebraic connectivity of graphs," *Czechoslovak Mathematical Journal*, vol. 23, no. 2, pp. 298–305, 1973.
2. R. Olfati-Saber, J. A. Fax, R. M. Murray, "Consensus and cooperation in networked multi-agent systems," *Proceedings of the IEEE*, vol. 95, no. 1, pp. 215–233, Jan. 2007.
3. B. Sklar, *Digital Communications: Fundamentals and Applications*, 2nd ed., Prentice Hall, 2001, ch. 12 (Spread-Spectrum Techniques).
4. M. Penrose, *Random Geometric Graphs*, Oxford University Press, 2003.
5. H. Urkowitz, "Energy detection of unknown deterministic signals," *Proceedings of the IEEE*, vol. 55, no. 4, pp. 523–531, Apr. 1967.

## Author

Chang Young Ham — preparing for graduate study in ECE (Fall 2027).
Model choices, parameters and verification criteria are mine; the code was
written with AI assistance and each stage is checked against the closed-form
results in theory/.
