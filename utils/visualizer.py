"""
utils/visualizer.py
--------------------
Generates and saves all charts to the results/ folder.
Charts are saved as PNG files — open them from the results/ folder after running.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — works in any environment
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Colour scheme
PALETTE = {
    "clean":    "#1D9E75",   # green
    "attacked": "#D85A30",   # red
    "hardened": "#378ADD",   # blue
    "squeezed": "#7F77DD",   # purple
}


def _save(fig, name: str) -> str:
    path = os.path.join(RESULTS_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[Plot] Saved → {path}")
    plt.close(fig)
    return path


# ── 1. Accuracy comparison bar chart ─────────────────────────────────────────
def plot_accuracy_comparison(results: dict, title: str = "Accuracy: clean vs adversarial vs hardened") -> str:
    fig, ax = plt.subplots(figsize=(9, 4))
    labels = list(results.keys())
    values = [results[k] for k in labels]

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


# ── 2. Perturbation heatmap ───────────────────────────────────────────────────
def plot_perturbation_heatmap(X_clean: np.ndarray, X_adv: np.ndarray,
                               feature_names: list, n_samples: int = 50,
                               attack_name: str = "FGSM") -> str:
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


# ── 3. Confusion matrices ─────────────────────────────────────────────────────
def plot_confusion_matrices(pred_clean: np.ndarray, pred_adv: np.ndarray,
                             y_true: np.ndarray) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, preds, title in zip(axes,
                                 [pred_clean, pred_adv],
                                 ["Clean input", "Adversarial input"]):
        cm = confusion_matrix(y_true, preds)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Normal", "Attack"],
                    yticklabels=["Normal", "Attack"])
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    fig.tight_layout()
    return _save(fig, "confusion_matrices.png")


# ── 4. Epsilon sweep ──────────────────────────────────────────────────────────
def plot_eps_sweep(eps_values: list, clean_accs: list, adv_accs: list,
                   attack_name: str = "FGSM") -> str:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(eps_values, clean_accs, "o-", color=PALETTE["clean"],
            label="Clean accuracy", linewidth=2)
    ax.plot(eps_values, adv_accs, "s--", color=PALETTE["attacked"],
            label="Adversarial accuracy", linewidth=2)
    ax.fill_between(eps_values, adv_accs, clean_accs,
                    alpha=0.12, color=PALETTE["attacked"])
    ax.set_xlabel("Perturbation ε")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{attack_name}: perturbation strength vs accuracy drop", fontsize=12)
    ax.legend()
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    return _save(fig, f"eps_sweep_{attack_name.lower()}.png")


# ── 5. Feature importance ─────────────────────────────────────────────────────
def plot_feature_importance(model, feature_names: list, top_n: int = 15) -> str:
    if not hasattr(model, "feature_importances_"):
        print("[Plot] Skipping feature importance — not a tree-based model.")
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
