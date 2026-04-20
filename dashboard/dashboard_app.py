
import os
import sys
import warnings
import logging

warnings.filterwarnings("ignore")
logging.getLogger("streamlit").setLevel(logging.ERROR)
os.environ["PYTHONWARNINGS"] = "ignore"

import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.data_loader import load_data
from models.train_model import load_model, train_and_save
from attacks.adversarial_attacks import (
    build_art_classifier,
    fgsm_attack,
    pgd_attack,
    feature_perturbation_attack,
)
from defenses.adversarial_defense import (
    adversarial_training,
    feature_squeezing,
    gaussian_smoothing,
    save_hardened_model,
)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Adversarial ML Toolkit",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Colour palette ─────────────────────────────────────────────────────────────
PALETTE = {
    "clean":    "#1D9E75",
    "attacked": "#D85A30",
    "hardened": "#378ADD",
    "squeezed": "#7F77DD",
    "smoothed": "#F5A623",
}

EPS_SWEEP   = [0.01, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40]
PGD_ITERS   = [10, 20, 40]
FP_NOISES   = [0.1, 0.2, 0.4, 0.6, 1.0]
MAX_SAMPLES = 1000

# ── Plain-English explanation bank ────────────────────────────────────────────
ATTACK_INFO = {
    "FGSM": (
        "**⚡ FGSM — Fast Gradient Sign Method** (white-box, 1 step)  \n"
        "The model's gradient tells us which direction each input feature should shift to "
        "maximise the model's error. FGSM takes *one big step* in that direction, scaled by ε. "
        "Higher ε = bigger, more obvious change = bigger accuracy drop. "
        "Fast to run, but relatively easy to defend against because it's just one step."
    ),
    "PGD": (
        "**💥 PGD — Projected Gradient Descent** (white-box, iterative)  \n"
        "PGD is FGSM repeated many times. Each iteration takes a *small* gradient step, "
        "then projects the result back inside the ε-ball to keep the perturbation bounded. "
        "After many iterations the adversarial example is much harder to defend — "
        "it has had many chances to find the model's weakest spot. "
        "More iterations = stronger attack, but slower."
    ),
    "Feature Perturbation": (
        "**🎲 Feature Perturbation** (black-box, no model access needed)  \n"
        "No gradient access required. The attacker finds the features with highest variance "
        "and adds random Gaussian noise to them. "
        "Simulates a realistic attacker who can observe traffic data but cannot inspect the model. "
        "σ (noise scale) controls how much noise is added. "
        "Generally weaker than FGSM/PGD but works against *any* model."
    ),
}

DEFENCE_INFO = {
    "None": (
        "**🚫 No defence active.**  \n"
        "The model is evaluated as-is on adversarial inputs. "
        "What you see is the raw damage from the attack with nothing to soften it."
    ),
    "Feature Squeezing": (
        "**🗜️ Feature Squeezing** (no retraining needed, instant)  \n"
        "Quantizes all feature values to a fixed bit-depth (fewer levels). "
        "Tiny adversarial perturbations that fall below the quantization step get erased. "
        "Works best against gradient attacks with small ε. "
        "As ε grows, perturbations survive the squeezing and the defence weakens. "
        "The bar/chart changes below show what survives after squeezing is applied."
    ),
    "Gaussian Smoothing": (
        "**🌊 Gaussian Smoothing** (no retraining needed, instant)  \n"
        "Creates 5 noisy copies of each adversarial sample and averages them. "
        "Adversarial perturbations are directional and structured — "
        "random averaging cancels them out. "
        "The real underlying signal (consistent across all copies) survives. "
        "Less effective against black-box noise attacks which are already random."
    ),
    "Adversarial Training": (
        "**🏋️ Adversarial Training** (most effective, requires retraining)  \n"
        "Retrains the model on a 50/50 mix of clean and adversarial examples. "
        "The model sees adversarial inputs during training and learns to classify them correctly. "
        "This is the strongest defence but takes time. "
        "The hardened model loaded here was pre-trained using FGSM adversarial examples."
    ),
}

CHART_INFO = {
    "kpi": (
        "**📈 How to read these numbers:**  \n"
        "- **Baseline accuracy** = model performance on normal, unmodified inputs (higher is better).  \n"
        "- **Under attack** = performance after adversarial examples are fed in — this should drop.  \n"
        "- **Accuracy drop** = the difference between the two. Bigger drop = more vulnerable model.  \n"
        "- **Mean L2 perturbation** = average magnitude of changes made to inputs. "
        "Small L2 = attack is subtle and hard to detect by eye."
    ),
    "bar": (
        "**📊 What the bars mean:**  \n"
        "- 🟢 **Green** = baseline accuracy on clean inputs — the model's 'normal' performance.  \n"
        "- 🔴 **Red/orange** = accuracy under attack — how badly the model is hurt.  \n"
        "- 🔵/🟣/🟡 **Coloured bar** = accuracy after applying the selected defence.  \n"
        "A good defence pushes the coloured bar close to the green bar, "
        "recovering the accuracy that was lost. If it barely moves, the defence is weak against this attack."
    ),
    "features": (
        "**🔥 What the feature perturbation chart shows:**  \n"
        "Each bar is one input feature (network traffic attribute like `src_bytes`, `count`, etc.). "
        "The bar length = how much that feature was changed on average across all samples.  \n"
        "- For FGSM/PGD: long bars = features the model's gradient is most sensitive to.  \n"
        "- For Feature Perturbation: long bars = features with highest data variance.  \n"
        "- When a defence is active, this chart shows perturbation *after* the defence is applied — "
        "you can see how much it 'cleaned up' the adversarial signal."
    ),
    "confusion": (
        "**🧮 How to read the confusion matrices:**  \n"
        "Rows = actual labels (Normal / Attack). Columns = predicted labels.  \n"
        "- **Top-left** = correctly classified Normal traffic (true negatives).  \n"
        "- **Bottom-right** = correctly detected Attacks (true positives).  \n"
        "- **Top-right** = Normal traffic flagged as Attack (false alarms).  \n"
        "- **Bottom-left** = Attacks that slipped through undetected (missed detections — dangerous!).  \n"
        "A successful attack causes the bottom-left number to grow (attacks get missed). "
        "A good defence shrinks it back down."
    ),
    "sweep": (
        "**📉 How to read the ε sweep chart:**  \n"
        "The x-axis is ε — how large the attack perturbation budget is.  \n"
        "- 🟢 **Green line** = clean accuracy. Flat — ε doesn't affect clean inputs.  \n"
        "- 🔴 **Red dashed** = adversarial accuracy. Drops as ε increases.  \n"
        "- **Coloured dotted line** (when defence is active) = accuracy after defence at each ε.  \n"
        "- **Shaded area** = the 'damage zone' between clean and adversarial.  \n"
        "A strong defence keeps the dotted line close to the green line across all ε values. "
        "A weak defence converges toward the red line as ε grows."
    ),
    "table": (
        "**🔍 What the sample table shows:**  \n"
        "Each row is one network connection from the test set.  \n"
        "- **True label** = ground truth (what the traffic actually is).  \n"
        "- **Clean pred** = what the model predicted on the original input.  \n"
        "- **Adv pred** = what the model predicted after the adversarial attack.  \n"
        "- **Flipped?** ⚠️ = the attack changed the model's answer for this sample — "
        "a tiny invisible change caused a wrong classification.  \n"
        "- **Recovered?** (when defence is active) = did the defence fix the flipped prediction?  \n"
        "Highlighted rows (orange) are the dangerous ones — the model was fooled."
    ),
    "summary": (
        "**🏆 Defence summary — how to interpret:**  \n"
        "- **Accuracy lost** = what the attack took away from the model.  \n"
        "- **Accuracy recovered** = how much the defence gave back. 100% recovery = perfect defence.  \n"
        "- The percentage is *of the lost accuracy*, not absolute accuracy.  \n"
        "Example: if the model lost 0.20 accuracy and the defence recovered 0.15, "
        "that's 75% recovery — a strong result."
    ),
}


# ── Load data, models, precomputed arrays ──────────────────────────────────────
@st.cache_resource(show_spinner="Loading data and models…")
def load_everything():
    x_train, x_test, y_train, y_test, features, scaler = load_data()
    try:
        model = load_model("baseline_model")
    except FileNotFoundError:
        result = train_and_save(x_train, x_test, y_train, y_test)
        model  = result["mlp"]
    try:
        hardened = load_model("hardened_model")
    except FileNotFoundError:
        hardened = None
    return x_train, x_test, y_train, y_test, features, scaler, model, hardened


@st.cache_data(show_spinner="Pre-computing attacks (one-time, ~30 s)…")
def precompute_attacks(_model, _x_test):
    x_sub = _x_test[:MAX_SAMPLES].astype(np.float32)
    art   = build_art_classifier(_model, clip_values=(float(x_sub.min()), float(x_sub.max())))
    cache = {"fgsm": {}, "pgd": {}, "fp": {}}
    for e in EPS_SWEEP:
        cache["fgsm"][e] = fgsm_attack(art, x_sub, eps=e)
    for e in EPS_SWEEP:
        cache["pgd"][e] = {}
        for iters in PGD_ITERS:
            cache["pgd"][e][iters] = pgd_attack(art, x_sub, eps=e, max_iter=iters)
    for noise in FP_NOISES:
        cache["fp"][noise] = feature_perturbation_attack(x_sub, noise_scale=noise)
    return cache, art


@st.cache_data(show_spinner="Pre-computing defences…")
def precompute_defences(_cache):
    sq, sm = {}, {}
    for e, arr in _cache["fgsm"].items():
        sq[("fgsm", e)] = feature_squeezing(arr)
        sm[("fgsm", e)] = gaussian_smoothing(arr)
    for e, iters_d in _cache["pgd"].items():
        for it, arr in iters_d.items():
            sq[("pgd", e, it)] = feature_squeezing(arr)
            sm[("pgd", e, it)] = gaussian_smoothing(arr)
    for noise, arr in _cache["fp"].items():
        sq[("fp", noise)] = feature_squeezing(arr)
        sm[("fp", noise)] = gaussian_smoothing(arr)
    return sq, sm


def acc(model, x, y):
    return float(np.mean(model.predict(x) == y))


def explain(key: str):
    with st.expander("💡 What does this mean?", expanded=False):
        st.markdown(CHART_INFO[key])


def bar_chart(labels, values, colors, height=260):
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker_color=colors,
        text=[f"{v:.3f}" for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        xaxis=dict(range=[0, 1.22], title="Accuracy"),
        height=height,
        margin=dict(l=10, r=70, t=10, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    fig.add_vline(x=0.5, line_dash="dash", line_color="gray", opacity=0.4)
    return fig


def cm_fig(y_true, y_pred, title):
    from sklearn.metrics import confusion_matrix as _cm
    cm = _cm(y_true, y_pred)
    fig = go.Figure(go.Heatmap(
        z=cm,
        x=["Pred: Normal", "Pred: Attack"],
        y=["True: Normal", "True: Attack"],
        colorscale="Blues",
        text=cm.astype(str),
        texttemplate="%{text}",
        showscale=False,
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=11)),
        height=240,
        margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Load + precompute
# ══════════════════════════════════════════════════════════════════════════════
x_train, x_test, y_train, y_test, features, scaler, model, hardened_model = load_everything()
x_full = x_test[:MAX_SAMPLES].astype(np.float32)
y_full = y_test[:MAX_SAMPLES]
adv_cache, art_clf = precompute_attacks(model, x_test)
sq_cache, sm_cache = precompute_defences(adv_cache)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("⚙️ Settings")
    st.markdown("---")

    attack_type = st.selectbox(
        "Attack type",
        ["FGSM", "PGD", "Feature Perturbation"],
        help="FGSM = 1-step white-box | PGD = iterative, stronger | FP = black-box",
    )

    eps = st.select_slider("Perturbation ε  (FGSM / PGD)", options=EPS_SWEEP, value=0.15)

    pgd_iter_choice = 20
    fp_noise        = 0.4
    if attack_type == "PGD":
        pgd_iter_choice = st.select_slider("PGD iterations", options=PGD_ITERS, value=20)
    if attack_type == "Feature Perturbation":
        fp_noise = st.select_slider("Feature noise σ", options=FP_NOISES, value=0.4)

    st.markdown("---")

    defense_type = st.selectbox(
        "Defence",
        ["None", "Feature Squeezing", "Gaussian Smoothing", "Adversarial Training"],
    )

    n_samples = st.slider("Samples to evaluate", 100, MAX_SAMPLES, 500, step=100)

    st.markdown("---")
    st.caption("Built with IBM ART · PyTorch · scikit-learn · Streamlit · Plotly")


# ── Resolve arrays ─────────────────────────────────────────────────────────────
x_sub = x_full[:n_samples]
y_sub = y_full[:n_samples]

if attack_type == "FGSM":
    x_adv  = adv_cache["fgsm"][eps][:n_samples]
    sq_key = ("fgsm", eps)
    sm_key = ("fgsm", eps)
elif attack_type == "PGD":
    x_adv  = adv_cache["pgd"][eps][pgd_iter_choice][:n_samples]
    sq_key = ("pgd", eps, pgd_iter_choice)
    sm_key = ("pgd", eps, pgd_iter_choice)
else:
    x_adv  = adv_cache["fp"][fp_noise][:n_samples]
    sq_key = ("fp", fp_noise)
    sm_key = ("fp", fp_noise)

clean_acc = acc(model, x_sub, y_sub)
adv_acc   = acc(model, x_adv, y_sub)
drop      = clean_acc - adv_acc
l2        = float(np.mean(np.linalg.norm(x_adv - x_sub, axis=1)))

# ── Resolve defence ────────────────────────────────────────────────────────────
defended_acc   = None
defended_label = None
x_def          = None
def_color      = PALETTE["hardened"]
def_model      = model   # which model predicts defended samples

if defense_type == "Feature Squeezing":
    x_def          = sq_cache[sq_key][:n_samples]
    defended_acc   = acc(model, x_def, y_sub)
    defended_label = "After squeezing"
    def_color      = PALETTE["squeezed"]

elif defense_type == "Gaussian Smoothing":
    x_def          = sm_cache[sm_key][:n_samples]
    defended_acc   = acc(model, x_def, y_sub)
    defended_label = "After smoothing"
    def_color      = PALETTE["smoothed"]

elif defense_type == "Adversarial Training" and hardened_model is not None:
    x_def          = x_adv
    defended_acc   = acc(hardened_model, x_adv, y_sub)
    defended_label = "Hardened model"
    def_color      = PALETTE["hardened"]
    def_model      = hardened_model


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PAGE
# ══════════════════════════════════════════════════════════════════════════════
st.title("🛡️ Adversarial ML Attack Toolkit")
st.caption("Adjust attack and defence settings in the sidebar — results update instantly.")

with st.expander("📖 How to use this dashboard", expanded=False):
    st.markdown("""
This dashboard demonstrates adversarial machine learning on a neural network
trained to detect network intrusions (IDS — Intrusion Detection System).

**Workflow:**
1. **Pick an attack** in the sidebar → watch accuracy drop in the metrics below
2. **Adjust ε** (perturbation strength) → stronger ε = bigger drop
3. **Pick a defence** → see how much of the lost accuracy is recovered
4. **Click any "💡 What does this mean?"** expander for a plain-English explanation

**Core idea:** small, invisible changes to input data (adversarial examples)
can fool a trained ML model into making wrong predictions.
The defence techniques try to detect or cancel out those changes.
    """)

st.markdown("---")

# ── Attack + Defence explanation side-by-side ──────────────────────────────────
col_a, col_d = st.columns(2)
with col_a:
    st.info(ATTACK_INFO[attack_type])
with col_d:
    st.info(DEFENCE_INFO[defense_type])
st.markdown("---")

# ── Adversarial Training expander ─────────────────────────────────────────────
if defense_type == "Adversarial Training":
    if hardened_model is not None:
        recovery = defended_acc - adv_acc
        st.success(
            f"**Hardened model accuracy on adversarial input: {defended_acc:.3f}** "
            f"— recovered +{recovery:.3f} accuracy vs. undefended model.",
            icon="✅",
        )
    else:
        st.warning("No saved hardened model found. Use the button below to train one.", icon="⚠️")

    with st.expander("🔄 Train a new hardened model (~30 s)"):
        st.markdown(
            "Clicking below will generate FGSM adversarial training examples "
            "and retrain the MLP on a 50/50 mix of clean + adversarial data. "
            "The model is saved and used immediately."
        )
        if st.button("Start adversarial training"):
            with st.spinner("Generating adversarial training examples…"):
                x_adv_tr = fgsm_attack(art_clf, x_train[:30000], eps=eps)
            with st.spinner("Retraining model (30 epochs)…"):
                hardened_model = adversarial_training(
                    model, x_train[:30000], y_train[:30000], x_adv_tr
                )
                save_hardened_model(hardened_model)
            st.success(f"Done! Hardened model accuracy under attack: "
                       f"{acc(hardened_model, x_adv, y_sub):.3f}")
            st.rerun()
    st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Key Metrics
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("📈 Key Metrics")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Baseline accuracy",    f"{clean_acc:.3f}",
          help="Accuracy on clean, unmodified inputs.")
c2.metric("Under attack",         f"{adv_acc:.3f}",
          delta=f"{-drop:.3f}", delta_color="inverse",
          help="Accuracy when adversarial examples are used as input.")
c3.metric("Accuracy drop",        f"{drop:.3f}",
          delta=f"▼ {drop/max(clean_acc,1e-9)*100:.1f}%", delta_color="inverse",
          help="How much accuracy was lost due to the attack.")
c4.metric("Mean L2 perturbation", f"{l2:.3f}",
          help="Average size of changes made to inputs. Smaller = more subtle attack.")

if defended_acc is not None:
    st.metric(
        label=f"✅ After defence ({defended_label})",
        value=f"{defended_acc:.3f}",
        delta=f"+{defended_acc - adv_acc:.3f} recovered",
        delta_color="normal",
    )

explain("kpi")
st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Accuracy bar + feature perturbation
# ══════════════════════════════════════════════════════════════════════════════
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("📊 Accuracy Comparison")

    bar_labels = ["Baseline (clean)", f"Under {attack_type}"]
    bar_values = [clean_acc, adv_acc]
    bar_colors = [PALETTE["clean"], PALETTE["attacked"]]

    if defended_acc is not None:
        bar_labels.append(defended_label)
        bar_values.append(defended_acc)
        bar_colors.append(def_color)

    st.plotly_chart(bar_chart(bar_labels, bar_values, bar_colors), use_container_width=True)
    explain("bar")

with col_right:
    st.subheader("🔥 Top 15 Most Perturbed Features")

    # Show perturbation in the defended array if active, otherwise adversarial
    x_compare = x_def[:50] if x_def is not None else x_adv[:50]
    delta      = np.abs(x_compare - x_sub[:50]).mean(axis=0)
    top_idx    = np.argsort(delta)[::-1][:15]
    feat_df    = pd.DataFrame({
        "Feature":  [features[i] for i in top_idx],
        "Mean |Δ|": delta[top_idx],
    }).sort_values("Mean |Δ|", ascending=True)

    cscale = (
        "Purples" if defense_type == "Feature Squeezing" else
        "YlOrBr"  if defense_type == "Gaussian Smoothing" else
        "Blues"   if defense_type == "Adversarial Training" else
        "Oranges"
    )
    subtitle = "defended input" if x_def is not None else "adversarial input"
    fig2 = px.bar(
        feat_df, x="Mean |Δ|", y="Feature", orientation="h",
        color="Mean |Δ|", color_continuous_scale=cscale,
        title=f"Features perturbed in: {subtitle}",
    )
    fig2.update_layout(
        height=300, margin=dict(l=10, r=10, t=40, b=10),
        coloraxis_showscale=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig2, use_container_width=True)
    explain("features")

st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Confusion Matrices
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("🧮 Confusion Matrices")

clean_preds = model.predict(x_sub)
adv_preds   = model.predict(x_adv)

if x_def is not None:
    cm1, cm2, cm3 = st.columns(3)
    with cm1:
        st.plotly_chart(cm_fig(y_sub, clean_preds, "✅ Clean input"), use_container_width=True)
    with cm2:
        st.plotly_chart(cm_fig(y_sub, adv_preds, f"🔴 Under {attack_type}"), use_container_width=True)
    with cm3:
        def_preds = def_model.predict(x_def)
        st.plotly_chart(cm_fig(y_sub, def_preds, f"🔵 After {defense_type}"), use_container_width=True)
else:
    cm1, cm2 = st.columns(2)
    with cm1:
        st.plotly_chart(cm_fig(y_sub, clean_preds, "✅ Clean input"), use_container_width=True)
    with cm2:
        st.plotly_chart(cm_fig(y_sub, adv_preds, f"🔴 Under {attack_type}"), use_container_width=True)

explain("confusion")
st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — ε sweep (defence-aware)
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("📉 Attack Strength vs Accuracy (ε sweep)")

# For Feature Perturbation, use FGSM as reference since FP doesn't use ε
sweep_base = (
    [acc(model, adv_cache["fgsm"][e][:n_samples], y_sub) for e in EPS_SWEEP]
    if attack_type == "FGSM"
    else [acc(model, adv_cache["pgd"][e][pgd_iter_choice][:n_samples], y_sub) for e in EPS_SWEEP]
    if attack_type == "PGD"
    else [acc(model, adv_cache["fgsm"][e][:n_samples], y_sub) for e in EPS_SWEEP]
)
atk_label = attack_type if attack_type != "Feature Perturbation" else "FGSM (reference)"

# Defence sweep — shows how the defence performs across all ε values
if defense_type == "Feature Squeezing":
    def atk_key(e):
        return ("fgsm", e) if attack_type in ("FGSM", "Feature Perturbation") else ("pgd", e, pgd_iter_choice)
    sweep_def = [acc(model, sq_cache[atk_key(e)][:n_samples], y_sub) for e in EPS_SWEEP]
    sweep_def_label = "Feature squeezing"
    sweep_def_color = PALETTE["squeezed"]

elif defense_type == "Gaussian Smoothing":
    def atk_key(e):
        return ("fgsm", e) if attack_type in ("FGSM", "Feature Perturbation") else ("pgd", e, pgd_iter_choice)
    sweep_def = [acc(model, sm_cache[atk_key(e)][:n_samples], y_sub) for e in EPS_SWEEP]
    sweep_def_label = "Gaussian smoothing"
    sweep_def_color = PALETTE["smoothed"]

elif defense_type == "Adversarial Training" and hardened_model is not None:
    def adv_at_e(e):
        if attack_type in ("FGSM", "Feature Perturbation"):
            return adv_cache["fgsm"][e][:n_samples]
        return adv_cache["pgd"][e][pgd_iter_choice][:n_samples]
    sweep_def = [acc(hardened_model, adv_at_e(e), y_sub) for e in EPS_SWEEP]
    sweep_def_label = "Hardened model"
    sweep_def_color = PALETTE["hardened"]
else:
    sweep_def       = None
    sweep_def_label = None
    sweep_def_color = None

fig_sweep = go.Figure()
fig_sweep.add_trace(go.Scatter(
    x=EPS_SWEEP, y=[clean_acc] * len(EPS_SWEEP),
    mode="lines+markers", name="Clean (baseline)",
    line=dict(color=PALETTE["clean"], width=2),
))
fig_sweep.add_trace(go.Scatter(
    x=EPS_SWEEP, y=sweep_base,
    mode="lines+markers", name=f"{atk_label} adversarial",
    line=dict(color=PALETTE["attacked"], width=2, dash="dash"),
))
if sweep_def is not None:
    fig_sweep.add_trace(go.Scatter(
        x=EPS_SWEEP, y=sweep_def,
        mode="lines+markers", name=sweep_def_label,
        line=dict(color=sweep_def_color, width=2, dash="dot"),
    ))
# Shaded damage zone
fig_sweep.add_trace(go.Scatter(
    x=EPS_SWEEP + EPS_SWEEP[::-1],
    y=[clean_acc] * len(EPS_SWEEP) + sweep_base[::-1],
    fill="toself", fillcolor="rgba(216,90,48,0.07)",
    line=dict(color="rgba(0,0,0,0)"), showlegend=False, name="Damage zone",
))
fig_sweep.update_layout(
    xaxis_title="Perturbation ε",
    yaxis=dict(title="Accuracy", range=[0, 1.05]),
    height=340,
    margin=dict(l=10, r=10, t=10, b=10),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    legend=dict(orientation="h", y=1.16),
)
st.plotly_chart(fig_sweep, use_container_width=True)
explain("sweep")
st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Sample predictions table
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("🔍 Sample Predictions — First 10 Samples")

clean_p10 = model.predict(x_sub[:10])
adv_p10   = model.predict(x_adv[:10])

tbl = {
    "True label": ["Attack" if v else "Normal" for v in y_sub[:10]],
    "Clean pred": ["Attack" if v else "Normal" for v in clean_p10],
    "Adv pred":   ["Attack" if v else "Normal" for v in adv_p10],
    "Flipped?":   ["⚠️ Yes" if a != b else "✅ No" for a, b in zip(clean_p10, adv_p10)],
}

if x_def is not None:
    def_p10 = def_model.predict(x_def[:10])
    tbl[f"{defended_label}"]  = ["Attack" if v else "Normal" for v in def_p10]
    tbl["Recovered?"] = [
        "✅ Yes" if (ap != cp and dp == cp) else
        "❌ No"  if (ap != cp and dp != cp) else "—"
        for cp, ap, dp in zip(clean_p10, adv_p10, def_p10)
    ]

df_tbl = pd.DataFrame(tbl)

def _hl(row):
    base = "background-color: rgba(216,90,48,0.15)" if row.get("Flipped?", "") == "⚠️ Yes" else ""
    return [base] * len(row)

st.dataframe(df_tbl.style.apply(_hl, axis=1), use_container_width=True)
explain("table")
st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — Defence summary (only when defence is active)
# ══════════════════════════════════════════════════════════════════════════════
if defended_acc is not None:
    st.subheader("🏆 Defence Summary")

    recovery     = defended_acc - adv_acc
    recovery_pct = recovery / max(drop, 1e-9) * 100

    s1, s2, s3 = st.columns(3)
    s1.metric("Accuracy lost to attack",  f"{drop:.4f}",
              delta=f"▼ {drop/clean_acc*100:.1f}%", delta_color="inverse")
    s2.metric("Accuracy recovered",       f"{recovery:.4f}",
              delta=f"+{recovery_pct:.1f}% of lost", delta_color="normal")
    s3.metric("Final defended accuracy",  f"{defended_acc:.4f}")

    if recovery_pct >= 90:
        st.success(
            f"**Excellent defence!** {defense_type} recovered **{recovery_pct:.0f}%** "
            f"of the accuracy lost to {attack_type} at ε={eps}."
        )
    elif recovery_pct >= 50:
        st.warning(
            f"**Partial defence.** {defense_type} recovered **{recovery_pct:.0f}%** "
            f"of the lost accuracy. Some vulnerability remains at ε={eps}."
        )
    else:
        st.error(
            f"**Weak defence against this attack.** {defense_type} only recovered "
            f"**{recovery_pct:.0f}%** of the lost accuracy at ε={eps}. "
            f"Try a different defence or reduce ε."
        )

    explain("summary")
