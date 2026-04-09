import os
import sys
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from attacks.adversarial_attacks import build_art_classifier
from data.data_loader import load_data
from models.train_model import train_and_save
from attacks.adversarial_attacks import (
    fgsm_attack,
    pgd_attack,
    feature_perturbation_attack,
)
from defenses.adversarial_defense import (
    adversarial_training,
    feature_squeezing,
    gaussian_smoothing,
)

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Adversarial ML Attack Toolkit",
    page_icon="🛡️",
    layout="wide",
)

PALETTE = {
    "clean":    "#1D9E75",
    "attacked": "#D85A30",
    "hardened": "#378ADD",
    "squeezed": "#7F77DD",
}


# ─── Cache: data + model loaded ONCE, reused every interaction ────────────────
@st.cache_resource(show_spinner="🔄 Loading data & training model (first run only) ...")
def get_data_and_model():

    x_train, x_test, y_train, y_test, features, scaler = load_data()
    model, _ = train_and_save(x_train, x_test, y_train, y_test)
    return x_train, x_test, y_train, y_test, features, scaler, model


# ─── Cache: epsilon sweep is slow — compute once per (n_samples) value ───────
@st.cache_data(show_spinner="📈 Computing ε sweep ...")
def compute_eps_sweep(_art_clf, _model, x_sub, y_sub):

    eps_values    = [0.01, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40]
    sweep_adv_acc = []

    for e in eps_values:
        xa = fgsm_attack(_art_clf, x_sub, eps=e)
        sweep_adv_acc.append(float(np.mean(_model.predict(xa) == y_sub)))

    return eps_values, sweep_adv_acc


# ─── Helper ───────────────────────────────────────────────────────────────────
def accuracy(model, x, y):
    return float(np.mean(model.predict(x) == y))


def make_bar_chart(labels, values, colors, title=""):
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker_color=colors,
        text=[f"{v:.3f}" for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        title=title, xaxis_range=[0, 1.15],
        xaxis_title="Accuracy", height=280,
        margin=dict(l=10, r=50, t=40, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    fig.add_vline(x=0.5, line_dash="dash", line_color="gray", opacity=0.5)
    return fig


# ─── Sidebar controls ─────────────────────────────────────────────────────────
st.sidebar.title("⚙️ Settings")

attack_type = st.sidebar.selectbox(
    "Attack type",
    ["FGSM", "PGD", "Feature Perturbation"],
    help="FGSM = fast; PGD = stronger but slower; Feature Perturbation = black-box"
)

eps       = st.sidebar.slider("Perturbation ε (FGSM / PGD)", 0.01, 0.50, 0.15, step=0.01)
pgd_iters = st.sidebar.slider("PGD iterations",              5, 100, 40, step=5)
fp_noise  = st.sidebar.slider("Feature noise σ",             0.05, 1.0, 0.4, step=0.05)

st.sidebar.markdown("---")
defense_type = st.sidebar.selectbox(
    "Defense type",
    ["None", "Feature Squeezing", "Gaussian Smoothing", "Adversarial Training"],
    help="Adversarial Training takes ~20 seconds"
)

n_samples = st.sidebar.slider("Samples to evaluate", 100, 1000, 500, step=100)

# ─── Main header ──────────────────────────────────────────────────────────────
st.title(" Adversarial ML Attack Toolkit")
st.caption("Attack a security classifier in real time — then defend it.")

# ─── Load data + model ────────────────────────────────────────────────────────
x_train, x_test, y_train, y_test, features, scaler, model = get_data_and_model()

x_sub = x_test[:n_samples].astype(np.float32)
y_sub = y_test[:n_samples]

art_clf = build_art_classifier(model, x_sub.shape[1])
clean_acc = accuracy(model, x_sub, y_sub)

# ─── Metric tiles ─────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("Baseline accuracy", f"{clean_acc:.3f}", help="Before any attack")

# ─── Run attack ───────────────────────────────────────────────────────────────
with st.spinner(f"⚔️ Running {attack_type} attack ..."):
    if attack_type == "FGSM":
        x_adv = fgsm_attack(art_clf, x_sub, eps=eps)
    elif attack_type == "PGD":
        x_adv = pgd_attack(art_clf, x_sub, eps=eps, max_iter=pgd_iters)
    else:
        x_adv = feature_perturbation_attack(x_sub, noise_scale=fp_noise)

adv_acc = accuracy(model, x_adv, y_sub)
drop    = clean_acc - adv_acc
l2      = float(np.mean(np.linalg.norm(x_adv - x_sub, axis=1)))

col2.metric("Under attack",       f"{adv_acc:.3f}", delta=f"{-drop:.3f}",    delta_color="inverse")
col3.metric("Accuracy drop",      f"{drop:.3f}",    delta=f"{drop/clean_acc*100:.1f}%", delta_color="inverse")
col4.metric("Mean L2 perturbation", f"{l2:.3f}",    help="Average feature change magnitude")

st.markdown("---")

# ─── Defense ──────────────────────────────────────────────────────────────────
defended_acc   = None
defended_model = model
x_defended     = x_adv.copy()

if defense_type == "Feature Squeezing":
    x_defended   = feature_squeezing(x_adv)
    defended_acc = accuracy(model, x_defended, y_sub)

elif defense_type == "Gaussian Smoothing":
    x_defended   = gaussian_smoothing(x_adv)
    defended_acc = accuracy(model, x_defended, y_sub)

elif defense_type == "Adversarial Training":
    with st.spinner("🔄 Adversarial training in progress (~20 seconds) ..."):
        x_adv_tr       = fgsm_attack(art_clf, x_train, eps=eps)
        defended_model = adversarial_training(model, x_train, y_train, x_adv_tr)
    defended_acc = accuracy(defended_model, x_adv, y_sub)

# ─── Accuracy bar chart + Perturbation heatmap ───────────────────────────────
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("📊 Accuracy comparison")
    labels = ["Baseline (clean)", f"Under {attack_type}"]
    values = [clean_acc, adv_acc]
    colors = [PALETTE["clean"], PALETTE["attacked"]]

    if defended_acc is not None:
        labels.append(f"After {defense_type}")
        values.append(defended_acc)
        colors.append(PALETTE["hardened"])

    st.plotly_chart(make_bar_chart(labels, values, colors), use_container_width=True)

with col_right:
    st.subheader("🔥 Perturbation per feature (top 15)")
    delta      = np.abs(x_adv[:50] - x_sub[:50])
    mean_delta = delta.mean(axis=0)
    top_idx    = np.argsort(mean_delta)[::-1][:15]

    feat_df = pd.DataFrame({
        "feature":  [features[i] for i in top_idx],
        "mean |Δ|": mean_delta[top_idx],
    }).sort_values("mean |Δ|", ascending=True)

    fig2 = px.bar(feat_df, x="mean |Δ|", y="feature", orientation="h",
                  color="mean |Δ|", color_continuous_scale="Oranges")
    fig2.update_layout(
        height=280, margin=dict(l=10, r=10, t=10, b=10),
        coloraxis_showscale=False,
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# ─── Epsilon sweep ────────────────────────────────────────────────────────────
st.subheader("📉 Attack strength vs accuracy (ε sweep)")

# FIx: compute sweep inline but with a spinner so it doesn't look frozen
eps_values = [0.01, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40]
with st.spinner("Running ε sweep (FGSM only) ..."):
    sweep_adv_accs = []
    for e in eps_values:
        xa = fgsm_attack(art_clf, x_sub, eps=e)
        sweep_adv_accs.append(accuracy(model, xa, y_sub))

fig3 = go.Figure()
fig3.add_trace(go.Scatter(
    x=eps_values, y=[clean_acc] * len(eps_values),
    mode="lines+markers", name="Clean",
    line=dict(color=PALETTE["clean"], width=2),
))
fig3.add_trace(go.Scatter(
    x=eps_values, y=sweep_adv_accs,
    mode="lines+markers", name="Adversarial",
    line=dict(color=PALETTE["attacked"], width=2, dash="dash"),
))
fig3.add_traces([go.Scatter(
    x=eps_values + eps_values[::-1],
    y=[clean_acc] * len(eps_values) + sweep_adv_accs[::-1],
    fill="toself", fillcolor="rgba(216,90,48,0.08)",
    line=dict(color="rgba(0,0,0,0)"), showlegend=False,
)])
fig3.update_layout(
    xaxis_title="Perturbation ε", yaxis_title="Accuracy",
    yaxis_range=[0, 1.05], height=300,
    margin=dict(l=10, r=10, t=10, b=10),
    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    legend=dict(orientation="h", y=1.1),
)
st.plotly_chart(fig3, use_container_width=True)

# ─── Sample predictions table ─────────────────────────────────────────────────
st.markdown("---")
st.subheader("🔍 Sample predictions — clean vs adversarial")
n_show = 10

sample_df = pd.DataFrame({
    "True label":    ["Attack" if v else "Normal" for v in y_sub[:n_show]],
    "Clean pred":    ["Attack" if v else "Normal" for v in model.predict(x_sub[:n_show])],
    "Adv pred":      ["Attack" if v else "Normal" for v in model.predict(x_adv[:n_show])],
    "Changed?":      ["⚠️ yes" if a != b else "✅ No"
                      for a, b in zip(model.predict(x_sub[:n_show]),
                                      model.predict(x_adv[:n_show]))],
})


def highlight_changed(row):
    if row["Changed?"].startswith("⚠️"):
        return ["background-color: rgba(216,90,48,0.12)"] * len(row)
    return [""] * len(row)


st.dataframe(sample_df.style.apply(highlight_changed, axis=1), use_container_width=True)

st.markdown("---")
st.caption("Built with IBM ART · scikit-learn · XGBoost · Streamlit · Plotly")