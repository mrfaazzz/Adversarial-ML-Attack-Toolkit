import os
import joblib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import xgboost as xgb

SAVE_DIR = os.path.join(os.path.dirname(__file__), "saved")
os.makedirs(SAVE_DIR, exist_ok=True)


# ── PyTorch MLP architecture ──────────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        return self.net(x)


# ── sklearn-compatible wrapper around PyTorch MLP ─────────────────────────────
class TorchMLP:
    def __init__(self, input_dim: int, epochs: int = 50, lr: float = 1e-3, batch_size: int = 128):
        self.input_dim  = input_dim
        self.epochs     = epochs
        self.lr         = lr
        self.batch_size = batch_size
        self.device     = torch.device("cpu")
        self.model      = MLP(input_dim).to(self.device)
        self.classes_   = np.array([0, 1])

    def fit(self, X, y):
        x_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.long)
        loader    = DataLoader(TensorDataset(x_t, y_t), batch_size=self.batch_size, shuffle=True)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.model(xb), yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"  Epoch {epoch+1}/{self.epochs}  loss={total_loss:.4f}")
        return self

    def predict_proba(self, X):
        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.tensor(X, dtype=torch.float32))
            return torch.softmax(logits, dim=1).numpy()

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


# ── Model constructors ────────────────────────────────────────────────────────
def _train_random_forest(X_train, y_train):
    return RandomForestClassifier(
        n_estimators=200, max_depth=12, random_state=42,
        n_jobs=-1, class_weight="balanced"
    ).fit(X_train, y_train)


def _train_xgboost(X_train, y_train):
    scale_pos = float(np.sum(y_train == 0)) / float(np.sum(y_train == 1))
    return xgb.XGBClassifier(
        n_estimators=300, max_depth=8, learning_rate=0.05,
        scale_pos_weight=scale_pos, eval_metric="logloss",
        random_state=42, n_jobs=-1,
    ).fit(X_train, y_train, verbose=False)


# ── Evaluation helper ─────────────────────────────────────────────────────────
def _evaluate(model, X_test, y_test, name="Model"):
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    print(f"\n  {name}")
    print(f"  Accuracy: {acc:.4f}  |  ROC-AUC: {auc:.4f}")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Attack"], zero_division=0))
    return {"accuracy": acc, "roc_auc": auc}


# ── Save / Load ───────────────────────────────────────────────────────────────
def save_model(model, name: str) -> str:
    path = os.path.join(SAVE_DIR, f"{name}.pkl")
    joblib.dump(model, path)
    print(f"[Model] Saved → {path}")
    return path


def load_model(name: str = "baseline_model"):
    path = os.path.join(SAVE_DIR, f"{name}.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model not found at {path}\n"
            "Run the IDS step first (python main.py) to train and save models."
        )
    return joblib.load(path)


# ── Main training function ────────────────────────────────────────────────────
def train_and_save(X_train, X_test, y_train, y_test) -> dict:
    """Train all three models and save them. Returns a dict with model objects."""
    input_dim = X_train.shape[1]

    print("\n[Model] Training PyTorch MLP (10 epochs)...")
    mlp = TorchMLP(input_dim=input_dim, epochs=10)
    mlp.fit(X_train, y_train)
    mlp_metrics = _evaluate(mlp, X_test, y_test, "PyTorch MLP (Baseline)")

    print("\n[Model] Training Random Forest...")
    rf         = _train_random_forest(X_train, y_train)
    rf_metrics = _evaluate(rf, X_test, y_test, "Random Forest")

    print("\n[Model] Training XGBoost...")
    xgb_model   = _train_xgboost(X_train, y_train)
    xgb_metrics = _evaluate(xgb_model, X_test, y_test, "XGBoost")

    save_model(mlp,       "baseline_model")
    save_model(rf,        "random_forest")
    save_model(xgb_model, "xgboost")

    return {
        "mlp": mlp,
        "rf":  rf,
        "xgb": xgb_model,
        "metrics": {
            "pytorch_mlp":   mlp_metrics,
            "random_forest": rf_metrics,
            "xgboost":       xgb_metrics,
        }
    }
