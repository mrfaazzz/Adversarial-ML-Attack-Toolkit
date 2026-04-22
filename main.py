import os
import sys
import json
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.data_loader import load_data
from models.train_model import train_and_save
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

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def _banner(title: str):
    print("\n" + "=" * 62)
    print(f"  {title}")
    print("=" * 62)


def _section(title: str):
    print(f"\n{'─'*62}")
    print(f"  {title}")
    print(f"{'─'*62}")


def _ask_continue(next_stage: str) -> bool:
    """Ask the user whether to proceed to the next stage."""
    print(f"\n{'─'*62}")
    print(f"  Ready to run: {next_stage}")
    while True:
        answer = input("  Continue? [Y/n]: ").strip().lower()
        if answer in ("", "y", "yes"):
            return True
        if answer in ("n", "no"):
            print("  Stopping here. Re-run main.py to continue from the start.")
            return False


def _eps_sweep(art_clf, model, X_test, y_test):
    """Run FGSM at several epsilon values and return the results."""
    from sklearn.metrics import accuracy_score
    eps_range   = [0.01, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40]
    baseline    = accuracy_score(y_test, model.predict(X_test))
    clean_accs  = []
    adv_accs    = []
    for eps in eps_range:
        X_adv   = fgsm_attack(art_clf, X_test, eps=eps)
        adv_acc = accuracy_score(y_test, model.predict(X_adv))
        clean_accs.append(baseline)
        adv_accs.append(adv_acc)
        print(f"    ε={eps:.2f}  clean={baseline:.3f}  adv={adv_acc:.3f}")
    return eps_range, clean_accs, adv_accs


def _save_report(all_metrics: dict):
    """Write results/report.txt and results/metrics.json."""
    timestamp   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_path = os.path.join(RESULTS_DIR, "report.txt")
    json_path   = os.path.join(RESULTS_DIR, "metrics.json")

    with open(report_path, "w") as f:
        f.write("=" * 62 + "\n")
        f.write("  ADVERSARIAL ML ATTACK TOOLKIT — RESULTS REPORT\n")
        f.write(f"  Generated: {timestamp}\n")
        f.write("=" * 62 + "\n\n")
        for section, metrics in all_metrics.items():
            f.write(f"[{section}]\n")
            if isinstance(metrics, dict):
                for k, v in metrics.items():
                    if k not in ("y_pred", "y_proba"):
                        f.write(f"  {k:35s}: {v}\n")
            else:
                f.write(f"  {metrics}\n")
            f.write("\n")

    # JSON version (floats only, no numpy types)
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


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — IDS: Load Data + Train Models
# ══════════════════════════════════════════════════════════════════════════════
def stage_ids():
    _banner("STAGE 1 — IDS: Load Data & Train Models")

    _section("Loading data")
    X_train, X_test, y_train, y_test, features, scaler = load_data()
    N          = min(1000, len(X_test))
    X_test_sub = X_test[:N]
    y_test_sub = y_test[:N]

    _section("Training models")
    models = train_and_save(X_train, X_test, y_train, y_test)

    _banner("STAGE 1 COMPLETE")
    print("  Three models trained and saved:")
    print("   • PyTorch MLP   → models/saved/baseline_model.pkl")
    print("   • Random Forest → models/saved/random_forest.pkl")
    print("   • XGBoost       → models/saved/xgboost.pkl")

    return {
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "X_test_sub": X_test_sub, "y_test_sub": y_test_sub,
        "features": features, "scaler": scaler,
        "models": models,
    }


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — Attacks
# ══════════════════════════════════════════════════════════════════════════════
def stage_attacks(ctx: dict) -> dict:
    _banner("STAGE 2 — Adversarial Attacks")

    best_model = ctx["models"]["mlp"]
    X_test_sub = ctx["X_test_sub"]
    y_test_sub = ctx["y_test_sub"]
    X_train    = ctx["X_train"]
    features   = ctx["features"]

    _section("Building ART classifier wrapper")
    art_clf = build_art_classifier(
        best_model,
        clip_values=(float(X_train.min()), float(X_train.max()))
    )
    print("  ART PyTorchClassifier ready.")

    all_metrics = {}

    _section("FGSM Attack  (eps=0.15, single step)")
    X_fgsm       = fgsm_attack(art_clf, X_test_sub, eps=0.15)
    fgsm_metrics = evaluate_attack(best_model, X_test_sub, X_fgsm, y_test_sub, "FGSM")
    all_metrics["FGSM Attack"] = fgsm_metrics

    _section("PGD Attack  (eps=0.15, 20 iterations)")
    X_pgd       = pgd_attack(art_clf, X_test_sub, eps=0.15, max_iter=20)
    pgd_metrics = evaluate_attack(best_model, X_test_sub, X_pgd, y_test_sub, "PGD")
    all_metrics["PGD Attack"] = pgd_metrics

    _section("Feature Perturbation  (black-box, no gradients)")
    X_fp       = feature_perturbation_attack(X_test_sub, noise_scale=0.4)
    fp_metrics = evaluate_attack(best_model, X_test_sub, X_fp, y_test_sub, "Feature Perturbation")
    all_metrics["Feature Perturbation"] = fp_metrics

    _section("Epsilon sweep (FGSM at multiple strengths)")
    eps_vals, clean_accs, adv_accs = _eps_sweep(art_clf, best_model, X_test_sub, y_test_sub)

    _section("Generating attack plots")
    rf = ctx["models"]["rf"]
    plot_perturbation_heatmap(X_test_sub, X_fgsm, features, n_samples=50, attack_name="FGSM")
    plot_perturbation_heatmap(X_test_sub, X_pgd,  features, n_samples=50, attack_name="PGD")
    plot_confusion_matrices(
        best_model.predict(X_test_sub),
        best_model.predict(X_fgsm),
        y_test_sub,
    )
    if hasattr(rf, "feature_importances_"):
        plot_feature_importance(rf, features)
    plot_eps_sweep(eps_vals, clean_accs, adv_accs, "FGSM")

    _banner("STAGE 2 COMPLETE")
    print("  Attack results summary:")
    for name, m in all_metrics.items():
        print(f"   • {name}: clean={m['clean_accuracy']:.3f} → adv={m['adversarial_accuracy']:.3f} "
              f"(drop={m['accuracy_drop']:.3f})")
    print("\n  Plots saved to: results/")

    return {
        **ctx,
        "art_clf":      art_clf,
        "X_fgsm":       X_fgsm,
        "X_pgd":        X_pgd,
        "X_fp":         X_fp,
        "fgsm_metrics": fgsm_metrics,
        "pgd_metrics":  pgd_metrics,
        "fp_metrics":   fp_metrics,
        "all_metrics":  all_metrics,
    }


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — Defences
# ══════════════════════════════════════════════════════════════════════════════
def stage_defences(ctx: dict) -> dict:
    _banner("STAGE 3 — Defences")

    best_model = ctx["models"]["mlp"]
    art_clf    = ctx["art_clf"]
    X_train    = ctx["X_train"]
    y_train    = ctx["y_train"]
    X_test_sub = ctx["X_test_sub"]
    y_test_sub = ctx["y_test_sub"]
    X_fgsm     = ctx["X_fgsm"]
    all_metrics = ctx["all_metrics"]
    fgsm_metrics = ctx["fgsm_metrics"]
    pgd_metrics  = ctx["pgd_metrics"]

    _section("Adversarial Training")
    # Use a subset for speed
    subset      = min(30000, len(X_train))
    X_train_sub = X_train[:subset]
    y_train_sub = y_train[:subset]
    X_adv_train = fgsm_attack(art_clf, X_train_sub, eps=0.15)

    hardened_model = adversarial_training(
        best_model, X_train_sub, y_train_sub, X_adv_train
    )
    save_hardened_model(hardened_model)

    _section("Feature Squeezing & Gaussian Smoothing")
    X_squeezed = feature_squeezing(X_fgsm, bit_depth=4)
    X_smoothed = gaussian_smoothing(X_fgsm, sigma=0.05)
    print("  Squeezed and smoothed adversarial samples ready.")

    _section("Comparing all defences")
    defense_results = compare_defenses(
        best_model, hardened_model,
        X_test_sub, X_fgsm, y_test_sub,
        squeezed_X_adv=X_squeezed,
        smoothed_X_adv=X_smoothed,
    )
    all_metrics["Defense Comparison"] = defense_results

    _section("Generating accuracy comparison plot")
    acc_data = {
        "Baseline (clean)":        fgsm_metrics["clean_accuracy"],
        "Under FGSM attack":       fgsm_metrics["adversarial_accuracy"],
        "Under PGD attack":        pgd_metrics["adversarial_accuracy"],
        "After adversarial train": defense_results.get("Hardened — adv input", 0),
        "Feature squeezing":       defense_results.get("Original + squeezing — adv", 0),
    }
    plot_accuracy_comparison(acc_data)

    _section("Saving report")
    _save_report(all_metrics)

    _banner("STAGE 3 COMPLETE  —  PIPELINE DONE")
    print("  Defence results summary:")
    for label, acc in defense_results.items():
        print(f"   • {label}: {acc:.4f}")
    print(f"\n  All results → {RESULTS_DIR}/")
    print("  Open results/*.png to view the charts.")
    print("\n  Dashboard: streamlit run dashboard/dashboard_app.py")

    return {**ctx, "defense_results": defense_results, "all_metrics": all_metrics}


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════
def main():
    run_all = "--all" in sys.argv

    _banner("ADVERSARIAL ML ATTACK TOOLKIT")
    print("  This toolkit demonstrates how adversarial attacks break an")
    print("  ML-based Intrusion Detection System (IDS), then shows how")
    print("  defences recover the lost accuracy.")
    print("\n  Stages:")
    print("    1. IDS     — Load data, train three classifiers")
    print("    2. Attack  — FGSM, PGD, Feature Perturbation")
    print("    3. Defence — Adversarial training, squeezing, smoothing")
    if not run_all:
        print("\n  You will be asked before each stage starts.")
        print("  (Run with --all to skip prompts and run everything.)")

    # ── Stage 1 (always runs) ─────────────────────────────────────────────────
    ctx = stage_ids()

    # ── Stage 2 ───────────────────────────────────────────────────────────────
    if run_all or _ask_continue("Stage 2 — Adversarial Attacks"):
        ctx = stage_attacks(ctx)
    else:
        return

    # ── Stage 3 ───────────────────────────────────────────────────────────────
    if run_all or _ask_continue("Stage 3 — Defences"):
        ctx = stage_defences(ctx)


if __name__ == "__main__":
    main()
