import numpy as np
import torch
import torch.nn as nn
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent

# ── Build the ART wrapper around the PyTorch MLP ─────────────────────────────
def build_art_classifier(torch_mlp, clip_values=None):

    if clip_values is None:
        clip_values = (-10.0, 10.0)

    pt_model  = torch_mlp.model
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(pt_model.parameters(), lr=1e-3)

    return PyTorchClassifier(
        model=pt_model,
        loss=criterion,
        optimizer=optimizer,
        input_shape=(torch_mlp.input_dim,),
        nb_classes=2,
        clip_values=clip_values,
    )


# ── Attack 1: FGSM ────────────────────────────────────────────────────────────
def fgsm_attack(art_clf, X_test: np.ndarray, eps: float = 0.15) -> np.ndarray:

    attack = FastGradientMethod(estimator=art_clf, eps=eps, targeted=False)
    return attack.generate(x=X_test.astype(np.float32))


# ── Attack 2: PGD ─────────────────────────────────────────────────────────────
def pgd_attack(art_clf, X_test: np.ndarray, eps: float = 0.15,
               eps_step: float = None, max_iter: int = 20) -> np.ndarray:

    if eps_step is None:
        eps_step = eps / 3

    attack = ProjectedGradientDescent(
        estimator=art_clf,
        eps=eps,
        eps_step=eps_step,
        max_iter=max_iter,
        targeted=False,
        verbose=False,
    )
    return attack.generate(x=X_test.astype(np.float32))


# ── Attack 3: Feature Perturbation ───────────────────────────────────────────
def feature_perturbation_attack(X_test: np.ndarray, noise_scale: float = 0.4,
                                 top_n_features: int = 10, random_state: int = 42) -> np.ndarray:

    rng   = np.random.default_rng(random_state)
    X_adv = X_test.copy().astype(np.float32)

    variances = X_test.var(axis=0)
    cols = np.argsort(variances)[::-1][:top_n_features]
    for col in cols:
        X_adv[:, col] += rng.normal(0, noise_scale, size=X_adv.shape[0])
    return X_adv


# ── Evaluation helper ─────────────────────────────────────────────────────────
def evaluate_attack(model, X_clean: np.ndarray, X_adv: np.ndarray,
                    y_true: np.ndarray, attack_name: str = "Attack") -> dict:

    from sklearn.metrics import accuracy_score

    pred_clean = model.predict(X_clean)
    pred_adv   = model.predict(X_adv)

    # Handle probability arrays (from ART classifiers)
    if pred_clean.ndim > 1:
        pred_clean = np.argmax(pred_clean, axis=1)
        pred_adv   = np.argmax(pred_adv,   axis=1)

    clean_acc = accuracy_score(y_true, pred_clean)
    adv_acc   = accuracy_score(y_true, pred_adv)
    drop      = clean_acc - adv_acc
    l2_norm   = float(np.mean(np.linalg.norm(X_adv - X_clean, axis=1)))
    linf_norm = float(np.mean(np.max(np.abs(X_adv - X_clean), axis=1)))

    print(f"\n  [{attack_name}]")
    print(f"  Clean accuracy  : {clean_acc:.4f}")
    print(f"  Adv accuracy    : {adv_acc:.4f}")
    print(f"  Accuracy drop   : {drop:.4f}  ({drop/clean_acc*100:.1f}%)")
    print(f"  Mean L2 norm    : {l2_norm:.4f}")
    print(f"  Mean L∞ norm    : {linf_norm:.4f}")

    return {
        "attack":               attack_name,
        "clean_accuracy":       clean_acc,
        "adversarial_accuracy": adv_acc,
        "accuracy_drop":        drop,
        "drop_percent":         drop / clean_acc * 100,
        "l2_norm":              l2_norm,
        "linf_norm":            linf_norm,
    }
