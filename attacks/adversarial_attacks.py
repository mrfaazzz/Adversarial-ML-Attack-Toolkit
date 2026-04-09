import numpy as np
import torch
import torch.nn as nn
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent



# ART Wrapper  —  PyTorch version (supports gradients)
# ─────────────────────────────────────────────────────────────────────────────
def build_art_classifier(torch_mlp_wrapper, input_dim: int = None, clip_values=None ):

    pt_model  = torch_mlp_wrapper.model
    in_dim = input_dim if input_dim is not None else torch_mlp_wrapper.input_dim

    if in_dim is None:
        raise ValueError("input_dim could not be determined")

    if clip_values is None:
        clip_values = (-10.0, 10.0)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(pt_model.parameters(), lr=1e-3)

    art_clf = PyTorchClassifier(
        model=pt_model,
        loss=criterion,
        optimizer=optimizer,
        input_shape=(in_dim,),
        nb_classes=2,
        clip_values=clip_values,
    )
    return art_clf



# FGSM Attack
# ─────────────────────────────────────────────────────────────────────────────
def fgsm_attack(art_clf, x_test: np.ndarray, eps: float = 0.1) -> np.ndarray:

    attack = FastGradientMethod(estimator=art_clf, eps=eps, targeted=False)
    return attack.generate(x=x_test.astype(np.float32))



# PGD Attack
# ─────────────────────────────────────────────────────────────────────────────
def pgd_attack(
    art_clf,
    x_test: np.ndarray,
    eps: float = 0.1,
    eps_step: float = None,
    max_iter: int = 40,
) -> np.ndarray:

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
    return attack.generate(x=x_test.astype(np.float32))



# Feature Perturbation (Black-box)
# ─────────────────────────────────────────────────────────────────────────────
def feature_perturbation_attack(
    x_test: np.ndarray,
    noise_scale: float = 0.3,
    top_n_features: int = 10,
    random_state: int = 42,
) -> np.ndarray:

    rng   = np.random.default_rng(random_state)
    x_adv = x_test.copy().astype(np.float32)

    # cols = list(range(min(top_n_features, x_test.shape[1])))
    variances = x_test.var(axis=0)
    cols = np.argsort(variances)[::-1][:top_n_features]
    for col in cols:
        x_adv[:, col] += rng.normal(0, noise_scale, size=x_adv.shape[0])
    return x_adv



# Evaluation helper
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_attack(
    model,
    x_clean: np.ndarray,
    x_adv: np.ndarray,
    y_true: np.ndarray,
    attack_name: str = "Attack",
) -> dict:

    from sklearn.metrics import accuracy_score

    pred_clean = model.predict(x_clean)
    pred_adv   = model.predict(x_adv)

    # ART classifiers return probability arrays → convert to labels
    if pred_clean.ndim > 1:
        pred_clean = np.argmax(pred_clean, axis=1)
        pred_adv   = np.argmax(pred_adv,   axis=1)

    clean_acc = accuracy_score(y_true, pred_clean)
    adv_acc   = accuracy_score(y_true, pred_adv)
    drop      = clean_acc - adv_acc
    l2_norm   = float(np.mean(np.linalg.norm(x_adv - x_clean, axis=1)))
    linf_norm = float(np.mean(np.max(np.abs(x_adv - x_clean), axis=1)))

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

    x_train, x_test, y_train, y_test, _, _ = load_data()
    mlp     = load_model("baseline_model")
    art_clf = build_art_classifier(
    clip_values=(float(x_train.min()), float(x_train.max())))

    print("\n=== FGSM ===")
    x_fgsm = fgsm_attack(art_clf, x_test[:300], eps=0.15)
    evaluate_attack(mlp, x_test[:300], x_fgsm, y_test[:300], "FGSM")

    print("\n=== Feature Perturbation ===")
    x_fp = feature_perturbation_attack(x_test[:300], noise_scale=0.4)
    evaluate_attack(mlp, x_test[:300], x_fp, y_test[:300], "Feature Perturbation")