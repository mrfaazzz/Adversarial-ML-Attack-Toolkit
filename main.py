

import os
import sys
import json
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

from data.data_loader import load_data
from models.train_model import train_and_save, load_model
from attacks.adversarial_attacks import (
    build_art_classifier,
    fgsm_attack,
    pgd_attack,
    feature_perturbation_attack,
    evaluate_attack,
)
from defenses.adversarial_defense import (
    adversarial_training,
    feature_squeezing,
    gaussian_smoothing,
    compare_defenses,
    save_hardened_model,
)
from utils.visualizer import (
    plot_accuracy_comparison,
    plot_perturbation_heatmap,
    plot_confusion_matrices,
    plot_eps_sweep,
    plot_feature_importance,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ─── Epsilon sweep ────────────────────────────────────────────────────────────
def run_eps_sweep(art_clf, model, X_test, y_test, eps_range=None):
    if eps_range is None:
        eps_range = [0.01, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40]

    from sklearn.metrics import accuracy_score
    baseline_acc = accuracy_score(y_test, model.predict(X_test))

    clean_accs, adv_accs = [], []
    for eps in eps_range:
        X_adv   = fgsm_attack(art_clf, X_test, eps=eps)
        adv_acc = accuracy_score(y_test, model.predict(X_adv))
        clean_accs.append(baseline_acc)
        adv_accs.append(adv_acc)
        print(f"  ε={eps:.2f}  clean={baseline_acc:.3f}  adv={adv_acc:.3f}")

    return eps_range, clean_accs, adv_accs


# ─── Report saver ─────────────────────────────────────────────────────────────
def save_report(all_metrics: dict):
    timestamp   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_path = os.path.join(RESULTS_DIR, "report.txt")
    json_path   = os.path.join(RESULTS_DIR, "metrics.json")

    with open(report_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("  ADVERSARIAL ML ATTACK TOOLKIT — RESULTS REPORT\n")
        f.write(f"  Generated: {timestamp}\n")
        f.write("=" * 60 + "\n\n")
        for section, metrics in all_metrics.items():
            f.write(f"[{section}]\n")
            if isinstance(metrics, dict):
                for k, v in metrics.items():
                    if k not in ("y_pred", "y_proba"):
                        f.write(f"  {k:30s}: {v}\n")
            else:
                f.write(f"  {metrics}\n")
            f.write("\n")

    clean = {}
    for section, metrics in all_metrics.items():
        if isinstance(metrics, dict):
            clean[section] = {
                k: float(v) if isinstance(v, (float, np.floating)) else v
                for k, v in metrics.items()
                if k not in ("y_pred", "y_proba")
            }
        else:
            clean[section] = str(metrics)

    with open(json_path, "w") as f:
        json.dump(clean, f, indent=2)

    print(f"\n[Report] Saved → {report_path}")
    print(f"[Report] JSON  → {json_path}")


# ─── Main pipeline ────────────────────────────────────────────────────────────
def main():
    print("\n" + "=" * 60)
    print("  ADVERSARIAL ML ATTACK TOOLKIT")
    print("=" * 60)

    # Step 1 — Load data
    print("\n[Step 1] Loading data ...")
    X_train, X_test, y_train, y_test, features, scaler = load_data()
    N          = min(1000, len(X_test))
    X_test_sub = X_test[:N]
    y_test_sub = y_test[:N]

    # Step 2 — Train models
    print("\n[Step 2] Training models ...")
    best_model, train_metrics = train_and_save(X_train, X_test, y_train, y_test)

    # Step 3 — Build ART wrapper (PyTorchClassifier — supports gradients)
    print("\n[Step 3] Building ART PyTorchClassifier wrapper ...")
    art_clf = build_art_classifier(best_model)

    # Step 4 — Attacks
    print("\n[Step 4] Running adversarial attacks ...")
    all_metrics = {}

    print("\n  → FGSM (eps=0.15) — fast single-step")
    X_fgsm       = fgsm_attack(art_clf, X_test_sub, eps=0.15)
    fgsm_metrics = evaluate_attack(best_model, X_test_sub, X_fgsm, y_test_sub, "FGSM")
    all_metrics["FGSM Attack"] = fgsm_metrics

    print("\n  → PGD (eps=0.15, 20 iterations) — stronger iterative attack")
    X_pgd       = pgd_attack(art_clf, X_test_sub, eps=0.15, max_iter=20)
    pgd_metrics = evaluate_attack(best_model, X_test_sub, X_pgd, y_test_sub, "PGD")
    all_metrics["PGD Attack"] = pgd_metrics

    print("\n  → Feature Perturbation (black-box)")
    X_fp       = feature_perturbation_attack(X_test_sub, noise_scale=0.4)
    fp_metrics = evaluate_attack(best_model, X_test_sub, X_fp, y_test_sub, "Feature Perturbation")
    all_metrics["Feature Perturbation"] = fp_metrics

    # Step 5 — Defenses
    print("\n[Step 5] Applying defenses ...")
    print("  → Adversarial training ...")
    # LIMIT DATA FOR SPEED
    subset = 30000

    X_train_sub = X_train[:subset]
    y_train_sub = y_train[:subset]

    # Generate adversarial samples on subset
    X_adv_train = fgsm_attack(art_clf, X_train_sub, eps=0.15)

    # Train hardened model on subset
    hardened_model = adversarial_training(
        best_model,
        X_train_sub,
        y_train_sub,
        X_adv_train
    )
    save_hardened_model(hardened_model)

    X_squeezed     = feature_squeezing(X_fgsm, bit_depth=4)
    X_smoothed     = gaussian_smoothing(X_fgsm, sigma=0.05)

    defense_results = compare_defenses(
        best_model, hardened_model,
        X_test_sub, X_fgsm, y_test_sub,
        squeezed_X_adv=X_squeezed,
    )
    all_metrics["Defense Comparison"] = defense_results

    # Step 6 — Plots
    print("\n[Step 6] Generating plots ...")
    acc_data = {
        "Baseline (clean)":        fgsm_metrics["clean_accuracy"],
        "Under FGSM attack":       fgsm_metrics["adversarial_accuracy"],
        "Under PGD attack":        pgd_metrics["adversarial_accuracy"],
        "After adversarial train": defense_results.get("Hardened — adv input", 0),
        "Feature squeezing":       defense_results.get("Original + squeezing — adv", 0),
    }
    plot_accuracy_comparison(acc_data)
    plot_perturbation_heatmap(X_test_sub, X_fgsm, features, n_samples=50, attack_name="FGSM")
    plot_perturbation_heatmap(X_test_sub, X_pgd,  features, n_samples=50, attack_name="PGD")
    plot_confusion_matrices(
        best_model.predict(X_test_sub),
        best_model.predict(X_fgsm),
        y_test_sub,
    )
    plot_feature_importance(best_model, features)

    print("\n  → Running ε sweep ...")
    eps_vals, clean_accs, adv_accs = run_eps_sweep(art_clf, best_model, X_test_sub, y_test_sub)
    plot_eps_sweep(eps_vals, clean_accs, adv_accs, "FGSM")

    # Step 7 — Report
    print("\n[Step 7] Saving report ...")
    save_report(all_metrics)

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print(f"  Results saved to: {RESULTS_DIR}/")
    print("  Dashboard: streamlit run dashboard/dashboard_app.py")
    print("=" * 60)
    return all_metrics


if __name__ == "__main__":
    main()