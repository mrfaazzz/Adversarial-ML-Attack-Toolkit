import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

PALETTE = {
    "clean":    "#1D9E75",
    "attacked": "#D85A30",
    "hardened": "#378ADD",
    "squeezed": "#7F77DD",
    "smoothed": "#F5A623",
}

plt.rcParams.update({
    "font.family":   "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":     True,
    "grid.alpha":    0.3,
    "grid.linestyle": "--",
})


def _save(fig, name: str) -> str:
    path = os.path.join(RESULTS_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"[Plot] Saved → {path}")
    plt.close(fig)
    return path


# ── 1. Accuracy comparison ────────────────────────────────────────────────────
def plot_accuracy_comparison(results: dict,
                              title: str = "Accuracy: clean vs attacked vs defended") -> str:
    labels = list(results.keys())
    values = [results[k] for k in labels]

    color_map = {
        "harden": PALETTE["hardened"],
        "squeez": PALETTE["squeezed"],
        "smooth": PALETTE["smoothed"],
        "adv":    PALETTE["attacked"],
        "attack": PALETTE["attacked"],
    }
    colors = []
    for lbl in labels:
        l = lbl.lower()
        colors.append(next((v for k, v in color_map.items() if k in l), PALETTE["clean"]))

    fig, ax = plt.subplots(figsize=(9, max(3, len(labels) * 0.65)))
    bars = ax.barh(labels, values, color=colors, height=0.55, edgecolor="white", linewidth=0.8)
    ax.set_xlim(0, 1.18)
    ax.set_xlabel("Accuracy", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.axvline(x=0.5, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax.axvline(x=0.9, color="#1D9E75", linestyle=":", linewidth=0.8, alpha=0.4)

    for bar, val in zip(bars, values):
        ax.text(val + 0.012, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=10, fontweight="bold")

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

    fig, axes = plt.subplots(1, 2, figsize=(14, 5),
                              gridspec_kw={"width_ratios": [3, 1]})

    # Left: heatmap
    ax = axes[0]
    im = ax.imshow(delta_top.T, aspect="auto", cmap="Oranges", vmin=0,
                   interpolation="nearest")
    ax.set_yticks(range(len(feat_lbls)))
    ax.set_yticklabels(feat_lbls, fontsize=8)
    ax.set_xlabel("Sample index", fontsize=10)
    ax.set_title(f"{attack_name} — perturbation per feature (top 20)", fontsize=12, fontweight="bold")
    fig.colorbar(im, ax=ax, label="|Δ feature|", shrink=0.85)

    # Right: mean perturbation bar
    ax2 = axes[1]
    mean_per_feat = delta_top.mean(axis=0)
    ax2.barh(range(len(feat_lbls)), mean_per_feat[::-1],
             color=PALETTE["attacked"], alpha=0.8, height=0.7)
    ax2.set_yticks(range(len(feat_lbls)))
    ax2.set_yticklabels(feat_lbls[::-1], fontsize=8)
    ax2.set_xlabel("Mean |Δ|", fontsize=10)
    ax2.set_title("Mean perturbation", fontsize=11, fontweight="bold")

    fig.tight_layout()
    return _save(fig, f"perturbation_heatmap_{attack_name.lower()}.png")


# ── 3. Confusion matrices ─────────────────────────────────────────────────────
def plot_confusion_matrices(pred_clean: np.ndarray, pred_adv: np.ndarray,
                             y_true: np.ndarray) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    clean_acc = np.mean(pred_clean == y_true)
    adv_acc   = np.mean(pred_adv   == y_true)

    for ax, preds, title, acc_val in zip(
        axes,
        [pred_clean, pred_adv],
        ["Clean input", "Adversarial input"],
        [clean_acc, adv_acc],
    ):
        cm = confusion_matrix(y_true, preds)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Normal", "Attack"],
                    yticklabels=["Normal", "Attack"],
                    linewidths=0.5, linecolor="white",
                    annot_kws={"fontsize": 12})
        ax.set_title(f"{title}\nAccuracy: {acc_val:.4f}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("Actual", fontsize=10)

    fig.suptitle("Confusion matrices — clean vs adversarial input", fontsize=13,
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    return _save(fig, "confusion_matrices.png")


# ── 4. Epsilon sweep ──────────────────────────────────────────────────────────
def plot_eps_sweep(eps_values: list, clean_accs: list, adv_accs: list,
                   attack_name: str = "FGSM") -> str:
    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.plot(eps_values, clean_accs, "o-", color=PALETTE["clean"],
            label="Clean accuracy", linewidth=2.5, markersize=7)
    ax.plot(eps_values, adv_accs, "s--", color=PALETTE["attacked"],
            label="Adversarial accuracy", linewidth=2.5, markersize=7)
    ax.fill_between(eps_values, adv_accs, clean_accs,
                    alpha=0.12, color=PALETTE["attacked"], label="Accuracy gap")

    # Annotate each point
    for e, a in zip(eps_values, adv_accs):
        ax.annotate(f"{a:.3f}", (e, a), textcoords="offset points",
                    xytext=(0, -14), ha="center", fontsize=8, color=PALETTE["attacked"])

    ax.set_xlabel("Perturbation ε", fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_title(f"{attack_name}: perturbation strength vs accuracy", fontsize=13,
                 fontweight="bold")
    ax.set_ylim(0, 1.08)
    ax.legend(fontsize=10)
    fig.tight_layout()
    return _save(fig, f"eps_sweep_{attack_name.lower()}.png")


# ── 5. Feature importance ─────────────────────────────────────────────────────
def plot_feature_importance(model, feature_names: list, top_n: int = 20) -> str:
    if not hasattr(model, "feature_importances_"):
        print("[Plot] Skipping feature importance — not a tree-based model.")
        return None

    importances = model.feature_importances_
    idx   = np.argsort(importances)[::-1][:top_n]
    names = [feature_names[i] for i in idx]
    vals  = importances[idx]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(top_n), vals, color=PALETTE["clean"], edgecolor="white",
                  linewidth=0.6, alpha=0.9)
    ax.set_xticks(range(top_n))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Importance", fontsize=11)
    ax.set_title(f"Top {top_n} feature importances (Random Forest)", fontsize=13,
                 fontweight="bold")

    # Label top 5 bars
    for i, (bar, val) in enumerate(zip(bars, vals)):
        if i < 5:
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.001,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8,
                    fontweight="bold", color=PALETTE["clean"])

    fig.tight_layout()
    return _save(fig, "feature_importance.png")
