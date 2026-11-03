"""
S3b - Telling jamming apart from a fade and a hop collision
===========================================================

Purpose
-------
The S3a energy test on pilot residuals notices that something is wrong with a
hop, but it cannot say what. Three causes all raise the residual energy:

    jammed hop      noise inside the hop is larger
    faded hop       the signal amplitude drops and the receiver, which removes
                    the pilots at their nominal amplitude, leaves part of them
                    in the residual
    hop collision   another node transmits on the same channel at the same time

S3b asks whether a receiver can tell these apart from the same 16 pilot
residuals, and what a learned classifier adds over the energy test.

Model (continues S3a: N_PILOT pilots of amplitude 1, thermal Eb/N0 from S1)
--------------------------------------------------------------------------
x_i = r_i - s_i is the residual of pilot i, sigma^2 = N0/2 the thermal variance.

    clean      : x_i = w_i
    jammed     : x_i = sqrt(1 + D) * w_i
    faded      : x_i = -g * s_i + w_i,        g = sqrt(D) * sigma   (gain 1 - g)
    collision  : x_i =  c * b_i + w_i,        c = sqrt(D) * sigma,  b_i = +-1 at random

w_i ~ N(0, sigma^2). D is the disturbance-to-noise ratio. All three disturbed
classes have the same mean residual energy, (1 + D) sigma^2 per sample. A faded
and a collided hop have exactly the same energy distribution (noncentral
chi-square, n degrees of freedom, noncentrality n D), so no rule based on the
energy alone can tell them apart.

Features (per hop)
    T = sum x_i^2 / sigma^2                 energy (the S3a statistic)
    C = sum x_i s_i / (sigma sqrt(n))       correlation with the known pilots
    K = n sum x_i^4 / (sum x_i^2)^2         sample kurtosis

Classifiers (4 classes, equal priors)
    energy only     multinomial logistic regression on (log T, (log T)^2); the
                    square lets one class (jammed) occupy both tails of T
    three features  multinomial logistic regression on (log T, C, K)
    small MLP       two hidden layers of 32 units on the sorted pilot-aligned
                    residuals y_i = x_i s_i / sigma (16 inputs; their sum is
                    sqrt(n) C, the other features are not given)

Verification
------------
1. Generator vs closed form: mean T = n (1 + D) for every disturbed class and
   n for a clean hop; mean C = -sqrt(n D) for a faded hop and 0 otherwise;
   mean K = 3n/(n + 2) for clean and jammed hops (Gaussian residuals).
   |z| < 4 at every point. In addition, T of faded and of collided hops must
   follow the noncentral chi-square distribution (Kolmogorov-Smirnov p > 1e-4).
2. Energy-only bound: the best possible accuracy from T alone follows from the
   four densities of T (maximum of the four, integrated numerically). The
   energy-only classifier must not exceed it at any D.
3. Neyman-Pearson bound: for clean vs jammed hops the energy test is optimal
   (S3a note, Section 2). On independent test hops the energy score must lie on
   the chi-square ROC, and neither learned classifier may lie above it. The
   tolerance includes the uncertainty of the measured false-alarm rate.

Outputs
-------
figures/s3b_accuracy_vs_dnr.png        accuracy of the three classifiers vs D
figures/s3b_roc_np_bound.png           clean vs jammed ROC against the chi-square bound
figures/s3b_feature_scatter.png        C and K for the four classes at one D
results/s3b_accuracy.csv
results/s3b_checks.csv
results/s3b_summary.txt

Parameters are assumed values chosen for the study (see README). Data
generation uses fixed seeds; classifier accuracies can differ in the third
decimal between platforms because of floating-point differences in training.
"""

import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2, kstest, ncx2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from s1_fh_partial_band_jamming import (
    EBN0_DB, BER_REQUIRED, N_CH, db_to_linear, linear_to_db, required_ebnj_db, worst_case_rho,
)
from s3a_jammer_detection import N_PILOT, noise_ratio

# ---------------------------------------------------------------------------
# Parameters (assumed values)
# ---------------------------------------------------------------------------
SEED = 2026
CLASSES = ["clean", "jammed", "faded", "collision"]
DNR_DB_GRID = np.arange(-6.0, 13.0, 2.0)     # disturbance-to-noise ratio D [dB]
N_TRAIN = 20_000                              # training hops per class and D
N_TEST = 20_000                               # test hops per class and D
DNR_SCATTER_DB = 6.0                          # D used for the feature-scatter figure
DNR_CONFUSION_DB = [0.0, 6.0]                 # D values with a confusion matrix in the summary

N_CHECK = 50_000                              # hops per class in verification 1
DNR_CHECK_DB = [0.0, 6.0]
N_ROC = 200_000                               # hops per set in verification 2
DNR_ROC_DB = 0.0                              # jammed-hop noise doubled (a = 2)
PFA_ROC_TEST = [1e-1, 1e-2]
Z_LIMIT = 4.0
KS_P_MIN = 1e-4

MLP_HIDDEN = (32, 32)

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
RES_DIR = ROOT / "results"
FIG_DIR.mkdir(exist_ok=True)
RES_DIR.mkdir(exist_ok=True)

N = N_PILOT
SIGMA = np.sqrt(0.5 / db_to_linear(EBN0_DB))   # Eb = 1, sigma^2 = N0/2


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def generate(label, num, dnr, rng):
    """Pilots s and residuals x for num hops of one class at D = dnr (linear)."""
    s = 2.0 * rng.integers(0, 2, size=(num, N)) - 1.0
    w = SIGMA * rng.normal(size=(num, N))
    if label == 0:                                   # clean
        x = w
    elif label == 1:                                 # jammed: larger Gaussian noise
        x = np.sqrt(1.0 + dnr) * w
    elif label == 2:                                 # faded: gain 1 - g, pilots removed at gain 1
        x = -np.sqrt(dnr) * SIGMA * s + w
    else:                                            # collision: another node's BPSK symbols
        b = 2.0 * rng.integers(0, 2, size=(num, N)) - 1.0
        x = np.sqrt(dnr) * SIGMA * b + w
    return s, x


def features(s, x):
    """Energy T, pilot correlation C and kurtosis K of each hop."""
    t = np.sum(x ** 2, axis=1) / SIGMA ** 2
    c = np.sum(x * s, axis=1) / (SIGMA * np.sqrt(N))
    k = N * np.sum(x ** 4, axis=1) / np.sum(x ** 2, axis=1) ** 2
    return t, c, k


def dataset(num, dnr, rng):
    """Inputs for the three classifiers and the labels, num hops per class."""
    rows_f, rows_r, labels = [], [], []
    for label in range(len(CLASSES)):
        s, x = generate(label, num, dnr, rng)
        t, c, k = features(s, x)
        rows_f.append(np.column_stack([np.log(t), c, k]))
        rows_r.append(np.sort(x * s / SIGMA, axis=1))  # pilot-aligned residuals, order removed
        labels.append(np.full(num, label))
    return np.vstack(rows_f), np.vstack(rows_r), np.concatenate(labels)


def train(features_train, raw_train, y_train):
    energy = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    energy.fit(energy_inputs(features_train), y_train)
    three = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    three.fit(features_train, y_train)
    mlp = make_pipeline(StandardScaler(),
                        MLPClassifier(MLP_HIDDEN, max_iter=300, early_stopping=True, random_state=SEED))
    mlp.fit(raw_train, y_train)
    return energy, three, mlp


def energy_inputs(f):
    """(log T, (log T)^2) for the energy-only classifier."""
    return np.column_stack([f[:, 0], f[:, 0] ** 2])


def bayes_accuracy_energy(dnr):
    """Best 4-class accuracy from T alone (equal priors): integral of the largest density / 4."""
    t = np.linspace(1e-6, chi2.isf(1e-12, N) * (1.0 + dnr), 200_001)
    pdfs = [chi2.pdf(t, N),                                   # clean
            chi2.pdf(t / (1.0 + dnr), N) / (1.0 + dnr),        # jammed
            ncx2.pdf(t, N, N * dnr),                           # faded
            ncx2.pdf(t, N, N * dnr)]                           # collision (same as faded)
    return float(np.trapezoid(np.max(pdfs, axis=0), t) / len(pdfs))


def z_mean(values, expected):
    return (values.mean() - expected) / (values.std(ddof=1) / np.sqrt(values.size))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    rng = np.random.default_rng(SEED)
    check_rows = []

    # --- Verification 1: generator vs closed form ------------------------------
    k_gauss = 3.0 * N / (N + 2.0)
    z1 = []
    ks_p = []
    print(f"{'D [dB]':>6} {'class':>10} {'quantity':>9} {'expected':>9} {'sample':>9} {'z':>6}")
    for dnr_db in DNR_CHECK_DB:
        dnr = db_to_linear(dnr_db)
        for label, name in enumerate(CLASSES):
            s, x = generate(label, N_CHECK, dnr, rng)
            t, c, k = features(s, x)
            tests = [("mean T", t, N * (1.0 + dnr) if label else float(N)),
                     ("mean C", c, -np.sqrt(N * dnr) if label == 2 else 0.0)]
            if label in (0, 1):
                tests.append(("mean K", k, k_gauss))
            if label in (2, 3):
                p_ks = kstest(t, ncx2(N, N * dnr).cdf).pvalue
                ks_p.append(p_ks)
                check_rows.append(("1", dnr_db, name, "KS p of T vs noncentral chi2", "", p_ks, ""))
                print(f"{dnr_db:6.1f} {name:>10} {'KS p(T)':>9} {'> 1e-4':>9} {p_ks:9.3f}")
            for quantity, values, expected in tests:
                z = z_mean(values, expected)
                z1.append(z)
                check_rows.append(("1", dnr_db, name, quantity, expected, values.mean(), z))
                print(f"{dnr_db:6.1f} {name:>10} {quantity:>9} {expected:9.3f} {values.mean():9.3f} {z:6.2f}")
    z1 = np.array(z1)
    ok1 = bool(np.all(np.abs(z1) < Z_LIMIT) and np.all(np.array(ks_p) > KS_P_MIN))
    print(f"VERIFICATION 1 (generator vs closed form): {'PASS' if ok1 else 'FAIL'} "
          f"(max |z| = {np.abs(z1).max():.2f} over {len(z1)} means; smallest KS p = {min(ks_p):.3f} over {len(ks_p)} tests)")

    # --- Accuracy sweep ------------------------------------------------------------
    acc = {"energy only": [], "three features": [], "small MLP": []}
    bayes_energy = []
    confusion = {}
    acc_rows = []
    models_at = {}
    for dnr_db in DNR_DB_GRID:
        dnr = db_to_linear(dnr_db)
        f_tr, r_tr, y_tr = dataset(N_TRAIN, dnr, rng)
        f_te, r_te, y_te = dataset(N_TEST, dnr, rng)
        energy, three, mlp = train(f_tr, r_tr, y_tr)
        predictions = {"energy only": energy.predict(energy_inputs(f_te)),
                       "three features": three.predict(f_te),
                       "small MLP": mlp.predict(r_te)}
        row = [dnr_db]
        for name, pred in predictions.items():
            a = float(np.mean(pred == y_te))
            acc[name].append(a)
            row.append(a)
        bayes_energy.append(bayes_accuracy_energy(dnr))
        row.append(bayes_energy[-1])
        acc_rows.append(row)
        if dnr_db in DNR_CONFUSION_DB:
            confusion[dnr_db] = {name: confusion_matrix(y_te, pred, normalize="true")
                                 for name, pred in predictions.items() if name != "energy only"}
        if dnr_db == DNR_ROC_DB:
            models_at[dnr_db] = (energy, three, mlp)
        print(f"D = {dnr_db:5.1f} dB: accuracy  energy only {row[1]:.3f} (bound {row[4]:.3f})  "
              f"three features {row[2]:.3f}  small MLP {row[3]:.3f}")
    acc = {k: np.array(v) for k, v in acc.items()}
    bayes_energy = np.array(bayes_energy)

    # --- Verification 2: energy-only classifier vs the best accuracy from T --------
    se = np.sqrt(bayes_energy * (1.0 - bayes_energy) / (len(CLASSES) * N_TEST))
    gap3 = acc["energy only"] - bayes_energy
    ok2 = bool(np.all(gap3 <= Z_LIMIT * se))
    for d_db, g, b in zip(DNR_DB_GRID, gap3, bayes_energy):
        check_rows.append(("2", d_db, "energy only", "accuracy - bound", b, b + g, g / se[0]))
    print(f"VERIFICATION 2 (energy-only classifier never above the best accuracy from T): "
          f"{'PASS' if ok2 else 'FAIL'} (accuracy - bound from {gap3.min():+.3f} to {gap3.max():+.3f})")

    # --- S3a operating points expressed as D ------------------------------------
    gp_db = linear_to_db(N_CH)
    ebn0_lin = db_to_linear(EBN0_DB)
    req_db = required_ebnj_db(ebn0_lin, BER_REQUIRED, rho_min=1e-5)
    js_outage_db = gp_db - req_db
    rho_worst = worst_case_rho(ebn0_lin, db_to_linear(req_db), 1e-5)[0]
    s3a_points = []
    for js_db, rho, label in ((js_outage_db, rho_worst, "S1 outage J/S, worst-case rho"),
                              (js_outage_db, 0.1, "S1 outage J/S, rho = 0.1")):
        dnr = noise_ratio(js_db, rho) - 1.0
        s3a_points.append((label, 10.0 * np.log10(dnr)))

    # --- Verification 3: Neyman-Pearson bound at a = 1 + D ------------------------
    dnr = db_to_linear(DNR_ROC_DB)
    a = 1.0 + dnr
    energy, three, mlp = models_at[DNR_ROC_DB]

    def scores(label):
        s, x = generate(label, N_ROC, dnr, rng)
        t, c, k = features(s, x)
        f = np.column_stack([np.log(t), c, k])
        r = np.sort(x * s / SIGMA, axis=1)
        p3 = three.predict_proba(f)
        pm = mlp.predict_proba(r)
        tiny = 1e-300
        return {"energy test": t,
                "three features": np.log(p3[:, 1] + tiny) - np.log(p3[:, 0] + tiny),
                "small MLP": np.log(pm[:, 1] + tiny) - np.log(pm[:, 0] + tiny)}

    calib, clean, jammed = scores(0), scores(0), scores(1)   # three independent sets
    ok3 = True
    roc_points = {}
    print(f"\n{'score':>15} {'P_fa':>8} {'P_d':>8} {'NP P_d':>8} {'diff':>8} {'tol':>7}")
    for name in calib:
        roc_points[name] = []
        for pfa_target in PFA_ROC_TEST:
            thr = np.quantile(calib[name], 1.0 - pfa_target)
            pfa = float(np.mean(clean[name] > thr))
            pd = float(np.mean(jammed[name] > thr))
            gamma = chi2.isf(pfa, N)
            pd_np = float(chi2.sf(gamma / a, N))
            slope = (chi2.pdf(gamma / a, N) / a) / chi2.pdf(gamma, N)        # dP_d / dP_fa on the NP curve
            tol = Z_LIMIT * np.sqrt(pd * (1 - pd) / N_ROC + slope ** 2 * pfa * (1 - pfa) / N_ROC)
            diff = pd - pd_np
            ok = abs(diff) <= tol if name == "energy test" else diff <= tol
            ok3 = ok3 and ok
            roc_points[name].append((pfa, pd))
            check_rows.append(("3", DNR_ROC_DB, name, f"P_d at P_fa {pfa_target:g}", pd_np, pd, diff / (tol / Z_LIMIT)))
            print(f"{name:>15} {pfa:8.4f} {pd:8.4f} {pd_np:8.4f} {diff:+8.4f} {tol:7.4f}  {'ok' if ok else 'VIOLATION'}")
    print(f"VERIFICATION 3 (no classifier above the chi-square ROC; energy test on it): {'PASS' if ok3 else 'FAIL'}")

    print("\nOVERALL:", "PASS" if (ok1 and ok2 and ok3) else "FAIL")

    # --- Summary -----------------------------------------------------------------
    i0 = int(np.flatnonzero(DNR_DB_GRID == 0.0)[0])
    lines = [
        f"Pilots per hop: {N}; thermal Eb/N0 {EBN0_DB:.0f} dB; {N_TRAIN} training and {N_TEST} test hops per class and D",
        f"Classes: {', '.join(CLASSES)}; the three disturbed classes have equal mean residual energy",
        "",
        "Accuracy (4 classes, chance = 0.25; best possible from T alone is below 0.75 at every D)",
        "  D [dB]  energy only  bound from T  three features  small MLP",
    ]
    for d, e, t3, m, b in acc_rows:
        lines.append(f"  {d:6.1f}  {e:11.3f}  {b:12.3f}  {t3:14.3f}  {m:9.3f}")
    lines += ["", "S3a operating points expressed as D (jammed-hop noise above thermal):"]
    for label, d_db in s3a_points:
        lines.append(f"  {label}: D = {d_db:.1f} dB")
    for d_db, mats in confusion.items():
        lines += ["", f"Confusion at D = {d_db:.0f} dB (rows: true class, columns: predicted; {', '.join(CLASSES)})"]
        for name, mat in mats.items():
            lines.append(f"  {name}:")
            for cls, row in zip(CLASSES, mat):
                lines.append(f"    {cls:>9}: " + "  ".join(f"{v:.3f}" for v in row))
    lines += ["", f"Clean vs jammed at D = {DNR_ROC_DB:.0f} dB (a = {a:.0f}), P_d at the measured P_fa:"]
    for name, pts in roc_points.items():
        lines.append(f"  {name}: " + ",  ".join(f"P_fa {pf:.4f} -> P_d {pd:.4f}" for pf, pd in pts))
    lines += ["", f"Verification: 1 {'PASS' if ok1 else 'FAIL'}, 2 {'PASS' if ok2 else 'FAIL'}, "
                  f"3 {'PASS' if ok3 else 'FAIL'}"]
    print("\n" + "\n".join(lines))
    (RES_DIR / "s3b_summary.txt").write_text("\n".join(lines) + "\n")

    with open(RES_DIR / "s3b_accuracy.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dnr_db", "acc_energy_only", "acc_three_features", "acc_small_mlp", "bound_from_energy"])
        w.writerows(acc_rows)
    with open(RES_DIR / "s3b_checks.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["check", "dnr_db", "case", "quantity", "expected_or_bound", "observed", "z_or_scaled_diff"])
        w.writerows(check_rows)

    # --- Figure 1: accuracy vs D --------------------------------------------------
    plt.figure(figsize=(7, 4.6))
    for name, marker in (("energy only", "s"), ("three features", "o"), ("small MLP", "^")):
        plt.plot(DNR_DB_GRID, acc[name], marker + "-", markersize=5, label=name)
    plt.plot(DNR_DB_GRID, bayes_energy, "k:", linewidth=1.2, label="best possible from energy alone")
    plt.axhline(0.25, color="gray", linestyle=":", label="chance (4 classes)")
    for (label, d_db), style in zip(s3a_points, ("--", "-.")):
        plt.axvline(d_db, color="k", linestyle=style, linewidth=0.8, alpha=0.7,
                    label=f"S3a jammed hop,\n{label}\n(D = {d_db:.1f} dB)")
    plt.xlabel("Disturbance-to-noise ratio D [dB]  (same for jamming, fade and collision)")
    plt.ylabel("Accuracy on test hops")
    plt.title(f"Clean / jammed / faded / collision from {N} pilot residuals")
    plt.ylim(0.2, 1.02)
    plt.grid(True)
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    plt.savefig(FIG_DIR / "s3b_accuracy_vs_dnr.png", dpi=200, bbox_inches="tight")
    plt.close()

    # --- Figure 2: ROC against the NP bound ------------------------------------------
    pfa_curve = np.logspace(-4, 0, 300)
    plt.figure(figsize=(7, 4.6))
    plt.semilogx(pfa_curve, chi2.sf(chi2.isf(pfa_curve, N) / a, N), "k-",
                 label="energy test, chi-square closed form (optimal)")
    for name in ("three features", "small MLP"):
        thr = np.quantile(calib[name], 1.0 - np.logspace(-3, -0.05, 25))
        pf = [np.mean(clean[name] > t) for t in thr]
        pdv = [np.mean(jammed[name] > t) for t in thr]
        plt.semilogx(pf, pdv, "o", markersize=4, label=f"{name} (test hops)")
    plt.xlabel("P_fa (clean hop flagged as jammed)")
    plt.ylabel("P_d (jammed hop flagged)")
    plt.title(f"Clean vs jammed, D = {DNR_ROC_DB:.0f} dB: learned classifiers against the bound")
    plt.ylim(0, 1.02)
    plt.grid(True, which="both")
    plt.legend(fontsize=8, loc="upper left")
    plt.savefig(FIG_DIR / "s3b_roc_np_bound.png", dpi=200, bbox_inches="tight")
    plt.close()

    # --- Figure 3: feature scatter -----------------------------------------------------
    dnr = db_to_linear(DNR_SCATTER_DB)
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for label, name in enumerate(CLASSES):
        s, x = generate(label, 1500, dnr, rng)
        _, c, k = features(s, x)
        ax.plot(c, k, ".", markersize=3, alpha=0.5, label=name)
    ax.axhline(k_gauss, color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel("C: correlation of the residual with the known pilots")
    ax.set_ylabel("K: sample kurtosis of the residual")
    ax.set_title(f"Two shape features per hop, D = {DNR_SCATTER_DB:.0f} dB (dotted: Gaussian mean 3n/(n+2))")
    ax.grid(True)
    ax.legend(markerscale=4, fontsize=8)
    fig.savefig(FIG_DIR / "s3b_feature_scatter.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
