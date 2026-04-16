# Adversarial ML Attack Toolkit for Security Models


> An offensive ML toolkit that crafts adversarial inputs to fool intrusion detection classifiers — then hardens them against such attacks.

---

## Project Structure

```
adversarial_ml_toolkit/
│
├── data/
│   └── data_loader.py          # NSL-KDD loader + synthetic data generator
│
├── models/
│   ├── train_model.py          # Train Random Forest + XGBoost classifiers
│   └── saved/                  # Saved .pkl model files (auto-created)
│
├── attacks/
│   └── adversarial_attacks.py  # FGSM, PGD, ZOO, Feature Perturbation
│
├── defenses/
│   └── adversarial_defense.py  # Adversarial training, feature squeezing, ensemble
│
├── utils/
│   └── visualizer.py           # All plots — heatmaps, ROC, confusion matrix, bar charts
│
├── dashboard/
│   └── app.py                  # Interactive Streamlit demo dashboard
│
├── results/                    # All output plots + report (auto-created)
│
├── main.py                     # Full pipeline runner
└── requirements.txt
```

---

## Setup

### 1. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. (Optional) Download the real NSL-KDD dataset
Download `KDDTrain+.txt` from https://www.unb.ca/cic/datasets/nsl.html  
Place it at: `data/KDDTrain+.txt`

> If you skip this step, the toolkit will automatically generate synthetic data and run perfectly for demo purposes.

---

## Running the Project

### Option A — Full pipeline (recommended)
```bash
python main.py
```
This runs all 7 steps:
1. Load / generate data
2. Train baseline model
3. Run FGSM, PGD, Feature Perturbation attacks
4. Apply adversarial training + feature squeezing defenses
5. Generate all comparison plots
6. Save results to `results/`
7. Write `results/report.txt` and `results/metrics.json`

### Option B — Interactive dashboard (for presentation day)
```bash
streamlit run dashboard/app.py
```
Opens a browser dashboard where you can:
- Select attack type (FGSM / PGD / Feature Perturbation)
- Adjust ε (perturbation strength) in real time
- Switch between defenses
- See accuracy drop live

### Option C — Run individual modules
```bash
python data/data_loader.py       # Test data loading
python models/train_model.py     # Train and save models
python attacks/adversarial_attacks.py   # Run attacks
python defenses/adversarial_defense.py  # Run defenses
```

---

## What Each Attack Does

| Attack | Type | How it works |
|--------|------|--------------|
| **FGSM** | White-box | Adds gradient-direction noise in one step |
| **PGD** | White-box | Iterative FGSM — stronger, multi-step |
| **Feature Perturbation** | Black-box | Adds calibrated noise to key network features |
| **ZOO** | Black-box | Estimates gradients without model access |

## What Each Defense Does

| Defense | How it works |
|---------|--------------|
| **Adversarial Training** | Retrains model on clean + adversarial examples |
| **Feature Squeezing** | Quantizes features to remove small perturbations |
| **Gaussian Smoothing** | Averages noisy copies to blur adversarial noise |
| **Ensemble Voting** | Uses 3 diverse models — harder to fool all at once |

---

## Results Outputs

After running `main.py`, find these in `results/`:

- `accuracy_comparison.png`  — bar chart of all configurations
- `perturbation_heatmap_fgsm.png` — which features were perturbed
- `perturbation_heatmap_pgd.png`
- `confusion_matrices.png`  — clean vs adversarial predictions
- `eps_sweep_fgsm.png`      — accuracy drop as ε increases
- `feature_importance.png`  — which features matter most
- `report.txt`              — human-readable summary
- `metrics.json`            — machine-readable metrics

---

## Tech Stack

- **scikit-learn** — Random Forest classifier
- **XGBoost** — gradient boosting classifier
- **adversarial-robustness-toolbox (ART)** — IBM's adversarial attack library
- **PyTorch** — neural network for gradient-based attacks
- **Streamlit** — interactive dashboard
- **Plotly / Matplotlib** — visualizations
- **Dataset** — NSL-KDD (or synthetic fallback)

---
