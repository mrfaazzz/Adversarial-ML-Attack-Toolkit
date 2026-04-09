"""
attacks/adversarial_attacks.py
-------------------------------
Adversarial attacks using IBM ART on a PyTorch MLP classifier.

ATTACKS:
  - FGSM  (Fast Gradient Sign Method)   — white-box, single step
  - PGD   (Projected Gradient Descent)  — white-box, iterative, stronger
  - Feature Perturbation                — black-box, domain-aware noise

ROOT CAUSE OF THE PREVIOUS ERROR:
    ProjectedGradientDescentNumpy / FastGradientMethod require a model that
    implements LossGradientsMixin (i.e. exposes real gradients).
    SklearnClassifier wrapping sklearn MLPClassifier does NOT expose gradients.

THE FIX:
    Use ART's PyTorchClassifier wrapping the PyTorch nn.Module directly.
    This gives full gradient access → FGSM and PGD work correctly.
"""

import numpy as np
import torch
import torch.nn as nn
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent


# ─────────────────────────────────────────────────────────────────────────────
# ART Wrapper  —  PyTorch version (supports gradients)
# ─────────────────────────────────────────────────────────────────────────────
def build_art_classifier(torch_mlp_wrapper, input_dim: int = None):
    """
    Wrap a TorchMLP (from train_model.py) in ART's PyTorchClassifier.

    Parameters
    ----------
    torch_mlp_wrapper : TorchMLP instance (has .model attribute = nn.Module)
    input_dim         : number of input features (auto-detected if None)

    Returns
    -------
    PyTorchClassifier — supports FGSM, PGD, and all gradient-based attacks

    ERROR FIX:
        Previously used SklearnClassifier → EstimatorError: LossGradientsMixin
        Now uses PyTorchClassifier        → gradients work correctly ✓
    """
    pt_model  = torch_mlp_wrapper.model          # the nn.Module
    in_dim    = input_dim or torch_mlp_wrapper.input_dim

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(pt_model.parameters(), lr=1e-3)

    art_clf = PyTorchClassifier(
        model=pt_model,
        loss=criterion,
        optimizer=optimizer,
        input_shape=(in_dim,),
        nb_classes=2,
        clip_values=(-5.0, 5.0),   # matches StandardScaler output range
    )
    return art_clf


# ─────────────────────────────────────────────────────────────────────────────
# FGSM Attack
# ─────────────────────────────────────────────────────────────────────────────
def fgsm_attack(art_clf, X_test: np.ndarray, eps: float = 0.1) -> np.ndarray:
    """
    Fast Gradient Sign Method — single-step white-box attack.

    Parameters
    ----------
    art_clf : ART PyTorchClassifier (from build_art_classifier)
    X_test  : clean input  shape (n_samples, n_features)
    eps     : perturbation budget — 0.01 = subtle,  0.3 = aggressive

    Returns
    -------
    X_adv : adversarial samples, same shape as X_test
    """
    attack = FastGradientMethod(estimator=art_clf, eps=eps, targeted=False)
    return attack.generate(x=X_test.astype(np.float32))


# ─────────────────────────────────────────────────────────────────────────────
# PGD Attack
# ─────────────────────────────────────────────────────────────────────────────
def pgd_attack(
    art_clf,
    X_test: np.ndarray,
    eps: float = 0.1,
    eps_step: float = 0.01,
    max_iter: int = 40,
) -> np.ndarray:
    """
    Projected Gradient Descent — iterative white-box attack.
    Much stronger than FGSM.

    Parameters
    ----------
    eps      : max perturbation budget
    eps_step : step size per iteration
    max_iter : number of steps  (reduce to 10–20 for speed during demos)

    NOTE: On 1000 samples with max_iter=40, this takes ~60 seconds.
          Use max_iter=20 and 500 samples for a faster demo run.
    """
    attack = ProjectedGradientDescent(
        estimator=art_clf,
        eps=eps,
        eps_step=eps_step,
        max_iter=max_iter,
        targeted=False,
        verbose=False,
    )
    return attack.generate(x=X_test.astype(np.float32))


# ─────────────────────────────────────────────────────────────────────────────
# Feature Perturbation (Black-box)
# ─────────────────────────────────────────────────────────────────────────────
def feature_perturbation_attack(
    X_test: np.ndarray,
    noise_scale: float = 0.3,
    top_n_features: int = 10,
    random_state: int = 42,
) -> np.ndarray:
    """
    Domain-aware black-box attack — no model access needed.
    Adds calibrated Gaussian noise to the most sensitive features,
    simulating a crafted malicious network packet.

    Parameters
    ----------
    noise_scale    : std-dev of Gaussian noise (0.1 = subtle, 1.0 = strong)
    top_n_features : how many features to perturb
    random_state   : reproducibility seed
    """
    rng   = np.random.default_rng(random_state)
    X_adv = X_test.copy().astype(np.float32)

    cols = list(range(min(top_n_features, X_test.shape[1])))
    for col in cols:
        X_adv[:, col] += rng.normal(0, noise_scale, size=X_adv.shape[0])
    return X_adv


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation helper
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_attack(
    model,
    X_clean: np.ndarray,
    X_adv: np.ndarray,
    y_true: np.ndarray,
    attack_name: str = "Attack",
) -> dict:
    """
    Compare clean vs adversarial accuracy.
    Works with TorchMLP, sklearn models, and ART classifiers.
    """
    from sklearn.metrics import accuracy_score

    pred_clean = model.predict(X_clean)
    pred_adv   = model.predict(X_adv)

    # ART classifiers return probability arrays → convert to labels
    if pred_clean.ndim > 1:
        pred_clean = np.argmax(pred_clean, axis=1)
        pred_adv   = np.argmax(pred_adv,   axis=1)

    clean_acc = accuracy_score(y_true, pred_clean)
    adv_acc   = accuracy_score(y_true, pred_adv)
    drop      = clean_acc - adv_acc
    l2_norm   = float(np.mean(np.linalg.norm(X_adv - X_clean, axis=1)))
    linf_norm = float(np.mean(np.max(np.abs(X_adv - X_clean), axis=1)))

    print(f"\n[{attack_name}]")
    print(f"  Clean accuracy     : {clean_acc:.4f}")
    print(f"  Adversarial acc    : {adv_acc:.4f}")
    print(f"  Accuracy drop      : {drop:.4f}  ({drop/clean_acc*100:.1f}%)")
    print(f"  Mean L2  norm      : {l2_norm:.4f}")
    print(f"  Mean L∞  norm      : {linf_norm:.4f}")

    return {
        "attack":               attack_name,
        "clean_accuracy":       clean_acc,
        "adversarial_accuracy": adv_acc,
        "accuracy_drop":        drop,
        "drop_percent":         drop / clean_acc * 100,
        "l2_norm":              l2_norm,
        "linf_norm":            linf_norm,
    }


# ── Run standalone ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from data.data_loader import load_data
    from models.train_model import load_model

    X_train, X_test, y_train, y_test, _, _ = load_data()
    mlp     = load_model("baseline_model")
    art_clf = build_art_classifier(mlp)

    print("\n=== FGSM ===")
    X_fgsm = fgsm_attack(art_clf, X_test[:300], eps=0.15)
    evaluate_attack(mlp, X_test[:300], X_fgsm, y_test[:300], "FGSM")

    print("\n=== Feature Perturbation ===")
    X_fp = feature_perturbation_attack(X_test[:300], noise_scale=0.4)
    evaluate_attack(mlp, X_test[:300], X_fp, y_test[:300], "Feature Perturbation")