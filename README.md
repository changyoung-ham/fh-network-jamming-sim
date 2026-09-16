# swarm-network-jamming-sim

Simulation study of how the connectivity of a multi-node frequency-hopping
network degrades under partial-band jamming, measured by the algebraic
connectivity (second-smallest Laplacian eigenvalue, λ₂) of the link graph.

This is a self-directed learning project built in stages. Each stage is
verified against a closed-form result or a known limiting case before the next
stage is started. Nothing here models a specific real-world radio; all physical
parameters are stated as assumptions in the source files.

## Motivation

In a swarm of cooperating nodes, individual links fail as jamming intensifies.
Counting failed links does not tell you whether the network is still one piece.
The Laplacian eigenvalue λ₂ does: it is zero when the graph is disconnected and
grows with the graph's robustness. The question this project asks is

> As a jammer concentrates its power on a fraction ρ of the hopping band,
> at what ρ does λ₂ collapse, and how does that threshold depend on the
> number of nodes and hopping channels?

The chain of models is

```
bit → AWGN → Eb/N0 → BER → hopping → partial-band jamming
    → per-link outage → adjacency matrix → Laplacian → λ₂ → threshold ρ*
```

## Stages

| Stage | Content | Verification | Status |
|------:|---------|--------------|--------|
| S0 | BPSK over AWGN, Monte Carlo BER | Must match Q(√(2Eb/N0)) | **done** (ratio 0.96–1.03 across 0–9 dB) |
| S1 | FH pattern + partial-band jammer, BER vs J/S | Processing gain / jamming margin vs. hand calculation | planned |
| S2 | N-node graph, per-link outage → λ₂ vs ρ | λ₂ = 0 iff graph disconnected; N = 10/20/30 comparison | planned |
| S3 | Channel re-allocation after detection, λ₂ recovery | Recovery ratio vs. no-jamming baseline | optional |

## Repository layout

```
src/        simulation scripts, one per stage
figures/    generated figures (committed so results are viewable without running)
results/    numeric results (CSV)
environment.yml
```

## Reproducing the results

```bash
conda env create -f environment.yml
conda activate swarm-sim
python src/s0_sinusoid_fft.py
python src/s0_bpsk_awgn_ber.py
```

All scripts use a fixed random seed, so the figures in `figures/` and the
numbers in `results/` are reproduced exactly.

## S0 result

![BER curve](figures/s0_ber_curve.png)

Simulated BER falls on the theoretical curve at every point (200+ errors per
point; ratio simulation/theory between 0.96 and 1.03). The second figure shows
why errors occur: the two received-sample distributions overlap across the
decision threshold, and the overlap shrinks as Eb/N0 grows.

![Received samples](figures/s0_received_hist.png)

## Assumptions and limitations (current)

- Coherent BPSK is used for its closed-form BER, not because it models any
  particular radio.
- AWGN only; no fading, no timing/phase error.
- Jamming (S1) will be treated as a known input; detection is out of scope.
- Nodes (S2) will be static.

These simplifications are deliberate. Each is a candidate extension rather than
a claim about real systems.

## Author

Chang Young Ham — preparing for graduate study in ECE (Fall 2027).
Modelling decisions, parameter justification and verification are the author's;
code is written with AI assistance and reviewed line by line.
