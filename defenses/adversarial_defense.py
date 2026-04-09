import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score
import xgboost as xgb

SAVE_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "saved")
os.makedirs(SAVE_DIR, exist_ok=True)


# ─── 1. Adversarial Training ─────────────────────────────────────────────────
def adversarial_training(
    base_model,
    x_train_clean: np.ndarray,
    y_train: np.ndarray,
    x_train_adv: np.ndarray,
    augment_ratio: float = 0.5,
):

    from models.train_model import TorchMLP

    n_adv  = int(len(x_train_adv) * augment_ratio)
    idx    = np.random.choice(len(x_train_adv), n_adv, replace=False)
    x_aug  = np.vstack([x_train_clean, x_train_adv[idx]])
    y_aug  = np.concatenate([y_train, y_train[idx]])

    perm   = np.random.permutation(len(x_aug))
    x_aug, y_aug = x_aug[perm], y_aug[perm]

    print(f"[Defense] Adversarial training: {len(x_aug)} samples "
          f"({len(x_train_clean)} clean + {n_adv} adversarial)")

    if isinstance(base_model, TorchMLP):
        # Retrain a fresh PyTorch MLP from scratch
        hardened = TorchMLP(input_dim=base_model.input_dim, epochs=30)
        hardened.fit(x_aug, y_aug)

    elif isinstance(base_model, RandomForestClassifier):
        hardened = RandomForestClassifier(
            n_estimators=base_model.n_estimators,
            max_depth=base_model.max_depth,
            random_state=42, n_jobs=-1, class_weight="balanced",
        )
        hardened.fit(x_aug, y_aug)

    else:
        # xGBoost or other
        hardened = xgb.XGBClassifier(
            n_estimators=300, max_depth=8, learning_rate=0.05,
            eval_metric="logloss", random_state=42, n_jobs=-1,
        )
        hardened.fit(x_aug, y_aug)

    print("[Defense] Hardened model trained.")
    return hardened


# ─── 2. Feature Squeezing ────────────────────────────────────────────────────
def feature_squeezing(x: np.ndarray, bit_depth: int = 4) -> np.ndarray:

    x_min = x.min(axis=0)
    x_max = x.max(axis=0)
    span  = x_max - x_min
    span[span == 0] = 1.0

    levels = (2 ** bit_depth) - 1
    return (np.round((x - x_min) / span * levels) / levels * span + x_min).astype(np.float32)


# ─── 3. Gaussian Smoothing ───────────────────────────────────────────────────
def gaussian_smoothing(x: np.ndarray, sigma: float = 0.05) -> np.ndarray:

    copies = [x + np.random.normal(0, sigma, x.shape) for _ in range(5)]
    return np.mean(copies, axis=0).astype(np.float32)


# ─── 4. Ensemble Defense ─────────────────────────────────────────────────────
def build_ensemble_defense(x_train: np.ndarray, y_train: np.ndarray):
    rf1  = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=1, n_jobs=-1)
    rf2  = RandomForestClassifier(n_estimators=150, max_depth=14, random_state=2, n_jobs=-1)
    xgb1 = xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        eval_metric="logloss", random_state=3, n_jobs=-1,
    )
    ensemble = VotingClassifier(
        estimators=[("rf1", rf1), ("rf2", rf2), ("xgb", xgb1)],
        voting="soft",
    )
    print("[Defense] Training ensemble ...")
    ensemble.fit(x_train, y_train)
    print("[Defense] Ensemble ready.")
    return ensemble


# ─── Comparison table ────────────────────────────────────────────────────────
def compare_defenses(
    original_model,
    hardened_model,
    x_clean: np.ndarray,
    x_adv: np.ndarray,
    y_true: np.ndarray,
    squeezed_x_adv: np.ndarray = None,
    smoothed_x_adv: np.ndarray = None,

) -> dict:
    results = {}
    configs = [
        ("Original — clean input",  original_model, x_clean),
        ("Original — adv input",    original_model, x_adv),
        ("Hardened — adv input",    hardened_model, x_adv),
    ]
    if squeezed_x_adv is not None:
        configs.append(("Original + squeezing — adv", original_model, squeezed_x_adv))

    if smoothed_x_adv is not None:
        configs.append(("Original + smoothing — adv", original_model, smoothed_x_adv))
    print(f"\n{'─'*62}")
    print(f"  {'Configuration':<40} {'Accuracy':>8}")
    print(f"{'─'*62}")
    for label, model, x in configs:
        acc = accuracy_score(y_true, model.predict(x))
        results[label] = acc
        print(f"  {label:<40} {acc:>8.4f}")
    print(f"{'─'*62}")
    return results


# ─── Save ────────────────────────────────────────────────────────────────────
def save_hardened_model(model, name="hardened_model"):
    path = os.path.join(SAVE_DIR, f"{name}.pkl")
    joblib.dump(model, path)
    print(f"[Defense] Saved hardened model → {path}")
    return path