"""
utils/visualizer.py
--------------------
Plotting and reporting utilities for the Adversarial ML Toolkit.
Saves all figures to results/ directory as PNG files.

POSSIBLE ERRORS:
    - results/ folder missing         →  created automatically
    - model has no feature_importances_  →  only RF/XGBoost have this;
                                            if another model is passed,
                                            the feature importance plot is skipped
    - Plots not showing on screen     →  they are saved to results/ as PNG files,
                                         not displayed. Open them from the folder.
    - seaborn import error            →  pip install seaborn
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

# ── Results folder (created automatically) ────────────────────────────────────
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Colour palette ────────────────────────────────────────────────────────────
PALETTE = {
    "clean":    "#1D9E75",   # green  — baseline / clean
    "attacked": "#D85A30",   # red    — under attack
    "hardened": "#378ADD",   # blue   — after defense
    "squeezed": "#7F77DD",   # purple — feature squeezing
}


def _save(fig, name: str) -> str:
    """Save figure to results/ and close it."""
    path = os.path.join(RESULTS_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[Viz] Saved → {path}")
    plt.close(fig)
    return path


# ─── 1. Accuracy comparison bar chart ────────────────────────────────────────
def plot_accuracy_comparison(
    results: dict,
    title: str = "Accuracy: clean vs adversarial vs hardened",
) -> str:
    """
    Horizontal bar chart comparing accuracy across conditions.

    Parameters
    ----------
    results : dict  {label: accuracy_float}
              e.g. {"Baseline (clean)": 0.95, "Under FGSM": 0.61}
    """
    fig, ax = plt.subplots(figsize=(9, 4))
    labels = list(results.keys())
    values = [results[k] for k in labels]

    # Auto-colour bars based on label content
    colors = []
    for lbl in labels:
        l = lbl.lower()
        if "harden" in l:                  colors.append(PALETTE["hardened"])
        elif "squeez" in l:                colors.append(PALETTE["squeezed"])
        elif "adv" in l or "attack" in l:  colors.append(PALETTE["attacked"])
        else:                              colors.append(PALETTE["clean"])

    bars = ax.barh(labels, values, color=colors, height=0.5)
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("Accuracy", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.axvline(x=0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    for bar, val in zip(bars, values):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=10)

    fig.tight_layout()
    return _save(fig, "accuracy_comparison.png")


# ─── 2. Perturbation heatmap ─────────────────────────────────────────────────
def plot_perturbation_heatmap(
    X_clean: np.ndarray,
    X_adv: np.ndarray,
    feature_names: list,
    n_samples: int = 50,
    attack_name: str = "FGSM",
) -> str:
    """
    Heatmap showing how much each feature was perturbed per sample.
    Shows only the top 20 most-perturbed features.
    """
    delta     = np.abs(X_adv[:n_samples] - X_clean[:n_samples])
    top_idx   = np.argsort(delta.mean(axis=0))[::-1][:20]
    delta_top = delta[:, top_idx]
    feat_lbls = [feature_names[i] for i in top_idx]

    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(delta_top.T, aspect="auto", cmap="Oranges", vmin=0)
    ax.set_yticks(range(len(feat_lbls)))
    ax.set_yticklabels(feat_lbls, fontsize=8)
    ax.set_xlabel("Sample index")
    ax.set_title(f"Perturbation magnitude per feature — {attack_name}", fontsize=12)
    fig.colorbar(im, ax=ax, label="|Δ feature|")
    fig.tight_layout()
    return _save(fig, f"perturbation_heatmap_{attack_name.lower()}.png")


# ─── 3. Confusion matrices (before vs after attack) ──────────────────────────
def plot_confusion_matrices(
    model_clean_pred: np.ndarray,
    model_adv_pred: np.ndarray,
    y_true: np.ndarray,
) -> str:
    """
    Side-by-side confusion matrices: clean input vs adversarial input.
    Shows how many Normal/Attack samples get mis-classified under attack.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, preds, title in zip(
        axes,
        [model_clean_pred, model_adv_pred],
        ["Clean input", "Adversarial input"],
    ):
        cm = confusion_matrix(y_true, preds)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Normal", "Attack"],
                    yticklabels=["Normal", "Attack"])
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    fig.tight_layout()
    return _save(fig, "confusion_matrices.png")


# ─── 4. ROC curves ───────────────────────────────────────────────────────────
def plot_roc_curves(
    model,
    X_clean: np.ndarray,
    X_adv: np.ndarray,
    X_hardened_adv: np.ndarray,
    y_true: np.ndarray,
) -> str:
    """
    Overlaid ROC curves for clean, adversarial, and hardened conditions.
    AUC shown in legend — higher = better.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    configs = [
        ("Clean input",    X_clean,        PALETTE["clean"]),
        ("Adv input",      X_adv,          PALETTE["attacked"]),
        ("Hardened + adv", X_hardened_adv, PALETTE["hardened"]),
    ]
    for label, X, color in configs:
        proba     = model.predict_proba(X)[:, 1]
        fpr, tpr, _ = roc_curve(y_true, proba)
        roc_auc   = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{label} (AUC={roc_auc:.3f})",
                color=color, linewidth=2)

    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — clean vs adversarial vs hardened", fontsize=12)
    ax.legend(fontsize=9)
    fig.tight_layout()
    return _save(fig, "roc_curves.png")


# ─── 5. Epsilon sweep (attack strength vs accuracy) ──────────────────────────
def plot_eps_sweep(
    eps_values: list,
    clean_accs: list,
    adv_accs: list,
    attack_name: str = "FGSM",
) -> str:
    """
    Line chart showing how accuracy drops as ε (perturbation budget) increases.
    The shaded area between the two lines shows the 'damage zone'.
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(eps_values, clean_accs, "o-",
            color=PALETTE["clean"],    label="Clean accuracy",      linewidth=2)
    ax.plot(eps_values, adv_accs, "s--",
            color=PALETTE["attacked"], label="Adversarial accuracy", linewidth=2)
    ax.fill_between(eps_values, adv_accs, clean_accs,
                    alpha=0.12, color=PALETTE["attacked"])
    ax.set_xlabel("Perturbation ε")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{attack_name}: perturbation strength vs accuracy drop", fontsize=12)
    ax.legend()
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    return _save(fig, f"eps_sweep_{attack_name.lower()}.png")


# ─── 6. Feature importance ───────────────────────────────────────────────────
def plot_feature_importance(
    model,
    feature_names: list,
    top_n: int = 15,
) -> str | None:
    """
    Bar chart of the top N most important features.

    Only works for tree-based models (Random Forest, XGBoost).
    Silently skipped for other model types.
    """
    if not hasattr(model, "feature_importances_"):
        print("[Viz] Skipping feature importance — model has no feature_importances_")
        return None

    importances = model.feature_importances_
    idx   = np.argsort(importances)[::-1][:top_n]
    names = [feature_names[i] for i in idx]
    vals  = importances[idx]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(range(top_n), vals, color=PALETTE["clean"])
    ax.set_xticks(range(top_n))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Importance")
    ax.set_title(f"Top {top_n} feature importances", fontsize=12)
    fig.tight_layout()
    return _save(fig, "feature_importance.png")