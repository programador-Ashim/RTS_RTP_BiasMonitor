from __future__ import annotations
import time
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from src.data import SyntheticReadinessGenerator, load_any
from src.preprocess import split_xy
from src.train import train_random_forest, train_gradient_boosting, evaluate
from src.monitor import batch_fairness_report
from src.realtime import stream_batches
from src.health import dataset_health
import numpy as np


def make_shap_ready(pre, X_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Works for numeric + categorical datasets:
    - transforms using your fitted preprocessor
    - converts sparse -> dense
    - forces numeric dtype (float32)
    - replaces NaN/inf with 0
    - returns a DataFrame with feature names (if available)
    """
    X_trans = pre.transform(X_raw)

    # sparse -> dense
    if hasattr(X_trans, "toarray"):
        X_trans = X_trans.toarray()

    # force numeric (prevents SHAP np.isnan crash)
    X_trans = np.asarray(X_trans).astype(np.float32)
    X_trans = np.nan_to_num(X_trans, nan=0.0, posinf=0.0, neginf=0.0)

    # feature names
    try:
        names = pre.get_feature_names_out()
    except Exception:
        names = [f"f{i}" for i in range(X_trans.shape[1])]

    return pd.DataFrame(X_trans, columns=names, index=X_raw.index)



try:
    from streamlit_option_menu import option_menu
except Exception:
    option_menu = None

st.set_page_config(page_title="Real-Time Bias Monitoring", layout="wide")

# ---------- SESSION STATE ----------
if "df" not in st.session_state:
    st.session_state.df = None
if "trained" not in st.session_state:
    st.session_state.trained = False
if "pipes" not in st.session_state:
    st.session_state.pipes = {"RTS": None, "RTP": None}
if "perfs" not in st.session_state:
    st.session_state.perfs = {"RTS": None, "RTP": None}
if "X_train" not in st.session_state:
    st.session_state.X_train = None
if "history" not in st.session_state:
    st.session_state.history = []
if "run_stream" not in st.session_state:
    st.session_state.run_stream = False

# ---------- SIDEBAR NAVIGATION ----------
with st.sidebar:
    st.title("AI Bias Monitor")

    if option_menu:
        page = option_menu(
            menu_title="TABS",
            options=["Dashboard", "Bias Monitoring", "Explainability", "Data Health", "Settings"],
            icons=["speedometer2", "shield-exclamation", "graph-up", "activity", "gear"],
            default_index=0,
            styles={
                "container": {"padding": "8px", "background-color": "#ffffff"},
                "icon": {"font-size": "18px"},
                "nav-link": {
                    "font-size": "16px",
                    "font-weight": "700",
                    "padding": "12px 14px",
                    "border-radius": "10px",
                    "margin": "6px 0px",
                    "color": "#111827",
                },
                "nav-link-selected": {
                    "background-color": "#e9edf5",
                    "color": "#111827",
                    "border-radius": "10px",
                },
            },
        )
    else:
        page = st.radio(
            "Navigation",
            ["Dashboard", "Bias Monitoring", "Explainability", "Data Health", "Settings"],
            index=0
        )

    st.divider()

# ===== Styling (kept close to your dashboard look) =====
st.markdown("""
<style>
.block-container { padding-top: 1.2rem; }
[data-testid="stSidebar"] { background: #f7f8fb; }

.card {
  border: 1px solid #eceef5;
  border-radius: 16px;
  padding: 14px 16px;
  background: white;
  box-shadow: 0 1px 10px rgba(0,0,0,0.04);
  margin-bottom: 10px;
}

.kpi {
  border: 1px solid #eceef5;
  border-radius: 16px;
  padding: 14px 16px;
  background: white;
  box-shadow: 0 1px 10px rgba(0,0,0,0.04);
}
.big-number { font-size: 34px; font-weight: 800; line-height: 1; margin: 0; }
.big-label { font-size: 13px; color: #6b7280; margin-top: 6px; }
.small-note { font-size: 12px; color: #6b7280; }

.good { border-left: 6px solid #22c55e; }
.warn { border-left: 6px solid #f59e0b; }
.bad  { border-left: 6px solid #ef4444; }
</style>
""", unsafe_allow_html=True)

def kpi_card(label, value, note="", mood="good"):
    st.markdown(f"""
    <div class="kpi {mood}">
      <p class="big-number">{value}</p>
      <div class="big-label">{label}</div>
      <div class="small-note">{note}</div>
    </div>
    """, unsafe_allow_html=True)

def ring(value_pct: float, good: bool, title: str = ""):
    color = "#02762d" if good else "#ef4444"
    fig = go.Figure(go.Pie(
        values=[value_pct, 100 - value_pct],
        hole=0.78,
        marker=dict(colors=[color, "#eef2f7"]),
        textinfo="none",
        sort=False,
        direction="clockwise"
    ))
    fig.update_layout(
        margin=dict(l=0, r=0, t=10, b=0),
        showlegend=False,
        annotations=[
            dict(text=f"<b>{value_pct:.0f}%</b>", x=0.5, y=0.52, showarrow=False, font=dict(size=28)),
            dict(text=title, x=0.5, y=0.18, showarrow=False, font=dict(size=12, color="#6b7280")),
        ],
        height=220
    )
    return fig
def root_cause_text(details: list[dict], threshold: float) -> str:
    if not details:
        return "No protected attributes selected."

    worst = max(
        details,
        key=lambda d: max(float(d.get("dp_diff", 0)), float(d.get("eo_diff", 0)))
    )

    attr = str(worst.get("attr", "Unknown")).title()
    dp = float(worst.get("dp_diff", 0))
    eo = float(worst.get("eo_diff", 0))
    sev = max(dp, eo)

    driver = "Demographic Parity" if dp >= eo else "Equalized Odds"

    if sev <= threshold:
        return (
            f"The model shows stable behavior across **{attr}** groups. "
            f"Both fairness metrics are within the acceptable threshold "
            f"(DP={dp:.3f}, EO={eo:.3f})."
        )

    return (
        f"Bias is primarily associated with **{attr}**\n\n"
        f"Demographic Parity gap: **{dp:.3f}**\n\n"
        f"Equalized Odds gap: **{eo:.3f}**\n\n"
        f"The larger disparity is observed in **{driver}**, "
        f"indicating unequal model performance across groups."
    )

def ema(series: list[float], alpha: float = 0.25) -> list[float]:
    out = []
    s = None
    for v in series:
        s = v if s is None else alpha * v + (1 - alpha) * s
        out.append(s)
    return out

# ===== Header =====
st.title("Real-Time Bias-Aware AI for Readiness Monitoring")
st.caption("Primary model: **Gradient Boosting** (plus RandomForest baseline) + real-time fairness monitoring (DP/EO) over streaming batches.")

# ===== Sidebar Controls (single block) =====
with st.sidebar:
    st.header("Controls")

    st.subheader("Data source")
    uploaded = st.file_uploader("Upload CSV/Excel (optional)", type=["csv", "xlsx", "xls"])
    use_synth = st.toggle("Use synthetic demo data", value=(uploaded is None))
    synth_n = st.slider("Synthetic rows", 500, 5000, 2000, 500)

    st.subheader("Model")
    model_choice = st.selectbox("Prediction model", ["Gradient Boosting", "Random Forest"])
    model_key = "gb" if model_choice.startswith("Gradient") else "rf"

    st.subheader("Monitoring")
    batch_size = st.slider("Batch size", 25, 300, 100, 25)
    threshold = st.slider("Alert threshold (dp/eo diff)", 0.01, 0.50, 0.10, 0.01)
    interval = st.slider("Update interval (seconds)", 0.0, 2.0, 0.5, 0.1)

    st.subheader("Train / Run")
    train_btn = st.button("Train model", type="primary", use_container_width=True)
    
    st.subheader("Real-time Monitoring")

    cS1, cS2 = st.columns(2)
    with cS1:
        start_btn = st.button("🟢 Start", use_container_width=True)
    with cS2:
        stop_btn = st.button("🔴 Stop", use_container_width=True)

    reset_btn = st.button("Reset History", use_container_width=True)

if reset_btn:
    st.session_state.history = []
    st.success("History cleared.")

if start_btn:
    st.session_state.run_stream = True
if stop_btn:
    st.session_state.run_stream = False

# ===== Load data =====
def get_dataframe() -> pd.DataFrame | None:
    if use_synth:
        gen = SyntheticReadinessGenerator(seed=42)
        return gen.make(n=int(synth_n))
    if uploaded is None:
        return None
    return load_any(uploaded)


def _norm_name(s: str) -> str:
    # lowercase, remove non-alphanum to make matching robust
    return "".join(ch for ch in s.lower() if ch.isalnum())


def _find_col(df: pd.DataFrame, keywords: list[str]) -> str | None:
    """
    Smart column finder:
    - matches by normalized names
    - supports partial keyword match
    """
    cols = list(df.columns)
    norm_cols = {c: _norm_name(c) for c in cols}

    norm_keys = [_norm_name(k) for k in keywords]

    # 1) exact/contains match
    for c, nc in norm_cols.items():
        if any(k in nc for k in norm_keys):
            return c

    return None


def add_rts_rtp_smart(df: pd.DataFrame) -> pd.DataFrame:
    """
    SMART MODE: auto-detect columns and create rts/rtp labels if missing.
    Works with:
      - your collegiate dataset
      - similar athlete datasets
      - slightly different column names
    """
    df = df.copy()

    # If RTS/RTP already exist (any case), normalize to lowercase columns too.
    for cand in ["rts", "RTS", "return_to_sport", "returntosport"]:
        if cand in df.columns and "rts" not in df.columns:
            df["rts"] = pd.to_numeric(df[cand], errors="coerce").fillna(0).astype(int)
            break

    for cand in ["rtp", "RTP", "return_to_performance", "returntoperformance"]:
        if cand in df.columns and "rtp" not in df.columns:
            df["rtp"] = pd.to_numeric(df[cand], errors="coerce").fillna(0).astype(int)
            break

    # --- Smart detect feature columns ---
    col_perf = _find_col(df, ["performance_score", "performance", "perfscore", "perf_score", "rating", "score"])
    col_fat  = _find_col(df, ["fatigue_score", "fatigue", "tired", "exertion", "rpe"])
    col_rec  = _find_col(df, ["recovery_days_per_week", "recovery", "rest_days", "rehab_days", "days_recovery"])
    col_acl  = _find_col(df, ["acl_risk_score", "acl_risk", "reinjury_risk", "injury_risk", "risk_score", "risk"])
    col_lb   = _find_col(df, ["load_balance_score", "load_balance", "balance_score", "workload_balance"])
    col_team = _find_col(df, ["team_contribution_score", "team_contribution", "contribution", "impact_score"])

    # injury indicator (0/1) detection
    col_inj = _find_col(df, ["injury_indicator", "injury", "injured", "injury_flag", "injury_status"])

    # --- Create RTS if missing ---
    if "rts" not in df.columns:
        # Prefer medically relevant rule if we have the right signals
        if col_fat and col_acl and col_rec and col_inj:
            inj = pd.to_numeric(df[col_inj], errors="coerce")
            df["rts"] = (
                (inj.fillna(1) == 0) &
                (pd.to_numeric(df[col_acl], errors="coerce") <= 55) &
                (pd.to_numeric(df[col_fat], errors="coerce") <= 5) &
                (pd.to_numeric(df[col_rec], errors="coerce") >= 1)
            ).astype(int)

        # Fallback: if no injury flag but have fatigue+risk+recovery
        elif col_fat and col_acl and col_rec:
            df["rts"] = (
                (pd.to_numeric(df[col_acl], errors="coerce") <= 55) &
                (pd.to_numeric(df[col_fat], errors="coerce") <= 5) &
                (pd.to_numeric(df[col_rec], errors="coerce") >= 1)
            ).astype(int)

        # Last fallback: if only performance exists, create a proxy (less ideal)
        elif col_perf:
            perf = pd.to_numeric(df[col_perf], errors="coerce")
            thr = perf.quantile(0.55)
            df["rts"] = (perf >= thr).astype(int)

    # --- Create RTP if missing ---
    if "rtp" not in df.columns:
        # Strong RTP rule if we have performance + balance + team + fatigue
                # ✅ ELITE RTP: quantile-based (works on any dataset scale)
        if col_perf:
            perf = pd.to_numeric(df[col_perf], errors="coerce")
            thr_perf = perf.quantile(0.35)   # top 30%
            rtp_mask = (perf >= thr_perf)

            if col_lb:
                lb = pd.to_numeric(df[col_lb], errors="coerce")
                rtp_mask = rtp_mask & (lb >= lb.quantile(0.35))  # top 40%

            if col_team:
                team = pd.to_numeric(df[col_team], errors="coerce")
                rtp_mask = rtp_mask & (team >= team.quantile(0.35))  # top 40%

            if col_fat:
                fat = pd.to_numeric(df[col_fat], errors="coerce")
                rtp_mask = rtp_mask & (fat <= fat.quantile(0.80))  # lower fatigue is better

           

            df["rtp"] = rtp_mask.fillna(False).astype(int)

        elif "rts" in df.columns:
            # fallback: RTP is stricter RTS (still gives both classes usually)
            df["rtp"] = (df["rts"].astype(int) & (np.random.default_rng(42).random(len(df)) > 0.30)).astype(int)


        # Fallback: performance + low risk
        elif col_perf and col_acl:
            perf = pd.to_numeric(df[col_perf], errors="coerce")
            risk = pd.to_numeric(df[col_acl], errors="coerce")
            df["rtp"] = ((perf >= perf.quantile(0.75)) & (risk <= risk.quantile(0.50))).astype(int)

        # Last fallback: just stricter than RTS
        elif "rts" in df.columns:
            df["rtp"] = (df["rts"].astype(int) & (np.random.default_rng(42).random(len(df)) > 0.25)).astype(int)

    return df


df = get_dataframe()
st.session_state.df = df

if df is None:
    st.info("Upload a dataset or enable synthetic data.")
    st.stop()

# ✅ SMART auto-create RTS/RTP if missing
df = add_rts_rtp_smart(df)
# enforce real-world rule
if "rts" in df.columns and "rtp" in df.columns:
    df.loc[df["rts"] == 0, "rtp"] = 0
st.session_state.df = df

st.caption("Target distributions (debug):")
if "rts" in df.columns: st.write("RTS:", df["rts"].value_counts())
if "rtp" in df.columns: st.write("RTP:", df["rtp"].value_counts())



# ===== Dataset mapping (used by all pages) =====
cols = df.columns.tolist()
default_target = "ready" if "ready" in cols else cols[-1]

st.subheader("Dataset mapping")

# Two targets: RTS + RTP (both must be binary 0/1)
# Defaults: use rts/rtp if present, otherwise fall back to ready
_default_rts = "rts" if "rts" in cols else ("ready" if "ready" in cols else cols[-1])
_default_rtp = "rtp" if "rtp" in cols else ("ready" if "ready" in cols else cols[-1])

cfg_target_rts = st.selectbox("Select RTS target column (binary 0/1)", cols, index=cols.index(_default_rts))
cfg_target_rtp = st.selectbox("Select RTP target column (binary 0/1)", cols, index=cols.index(_default_rtp))
cfg_protected = st.multiselect(
    "Select protected attribute columns (e.g., gender, race, age_group)",
    cols,
    default=[c for c in ["race", "gender", "age_group"] if c in cols]
)
drop_cols = st.multiselect(
    "Columns to drop (IDs, timestamps, leakage)",
    cols,
   default=[c for c in cols if c.lower() in {
"id","patient_id","athlete_id","timestamp",
"performance_score",
"load_balance_score",
"team_contribution_score",
"fatigue_score"
}]

)
# ===== Train =====
if train_btn:
    # --- validate both targets ---
    for _name, _t in [("RTS", cfg_target_rts), ("RTP", cfg_target_rtp)]:
        if _t not in df.columns:
            st.error(f"Choose a valid target column for {_name}.")
            st.stop()

        uniq = set(pd.Series(df[_t].dropna().unique()).tolist())
        if not uniq.issubset({0, 1, "0", "1", True, False, "True", "False"}):
            st.error(
                f"{_name} target is not binary. Choose a binary target (0/1), or preprocess the dataset to create one."
            )
            st.stop()

    # VERY IMPORTANT FIX — remove BOTH targets from features
    drop_cols_train = list(set(drop_cols + [cfg_target_rtp, cfg_target_rts]))

    X_train, X_test, y_train_rts, y_test_rts = split_xy(
        df,
        target=cfg_target_rts,
        drop_cols=drop_cols_train
    )

    st.session_state.feature_cols = list(X_train.columns)

    # Align RTP labels to same train/test rows
    y_rtp = df[cfg_target_rtp].astype(int)
    y_train_rtp = y_rtp.loc[X_train.index]
    y_test_rtp = y_rtp.loc[X_test.index]

    #  Guard: both classes must exist for training  (MUST be indented!)
    if y_train_rts.nunique() < 2:
        st.error("RTS has only one class in the training split. Adjust RTS rules/thresholds.")
        st.write(df[cfg_target_rts].value_counts())
        st.stop()

    if y_train_rtp.nunique() < 2:
        st.error("RTP has only one class in the training split. Adjust RTP rules/thresholds (quantiles fix this).")
        st.write(df[cfg_target_rtp].value_counts())
        st.stop()



    # --- train two pipelines (RTS + RTP) with the same model family choice ---
    if model_key == "gb":
        pipe_rts = train_gradient_boosting(X_train, y_train_rts, n_estimators=250, learning_rate=0.05, max_depth=3)
        pipe_rtp = train_gradient_boosting(X_train, y_train_rtp, n_estimators=250, learning_rate=0.05, max_depth=3)
        model_name = "GradientBoosting"
    else:
        pipe_rts = train_random_forest(X_train, y_train_rts, n_estimators=300, max_depth=None, min_samples_leaf=2)
        pipe_rtp = train_random_forest(X_train, y_train_rtp, n_estimators=300, max_depth=None, min_samples_leaf=2)
        model_name = "RandomForest"

    st.session_state.pipes = {"RTS": pipe_rts, "RTP": pipe_rtp}
    st.session_state.trained = True
    st.session_state.perfs = {
        "RTS": evaluate(pipe_rts, X_test, y_test_rts),
        "RTP": evaluate(pipe_rtp, X_test, y_test_rtp),
    }
    st.session_state.X_train = X_train
    st.success(f"Trained {model_name} successfully for BOTH RTS and RTP!")

# ======================================================================
# PAGES
# ======================================================================


if page == "Dashboard":

    cA, cB = st.columns([1, 1])
    with cA:
        st.subheader("Model performance (test split)")

        perfs = st.session_state.perfs
        if perfs and perfs.get("RTS") and perfs.get("RTP"):
          

           
                st.markdown("#### RTS (Return to Sport)")
                perf = perfs["RTS"]
                m1, m2, m3, m4, m5 = st.columns(5)
                with m1: st.markdown(f"**Accuracy**  \n{perf['Accuracy']:.3f}")
                with m2: st.markdown(f"**Precision** \n{perf['Precision']:.3f}")
                with m3: st.markdown(f"**Recall**    \n{perf['Recall']:.3f}")
                with m4: st.markdown(f"**F1-score**  \n{perf['f1']:.3f}")
                with m5: st.markdown(f"**ROC-AUC**   \n{perf['Roc_Auc']:.3f}")

           
                st.markdown("#### RTP (Return to Performance)")
                perf = perfs["RTP"]
                m1, m2, m3, m4, m5 = st.columns(5)
                with m1: st.markdown(f"**Accuracy**  \n{perf['Accuracy']:.3f}")
                with m2: st.markdown(f"**Precision** \n{perf['Precision']:.3f}")
                with m3: st.markdown(f"**Recall**    \n{perf['Recall']:.3f}")
                with m4: st.markdown(f"**F1-score**  \n{perf['f1']:.3f}")
                with m5: st.markdown(f"**ROC-AUC**   \n{perf['Roc_Auc']:.3f}")
        else:
            st.info("Train a model to see RTS/RTP metrics.")

    with cB:
        st.subheader("How prediction works (simple)")
        st.write(
            "- We clean the data and fill missing values (imputation).\n"
            "- We convert categories to numbers (OneHot).\n"
            "- The model outputs a class: **ready (1)** or **not ready (0)**.\n"
            "- Fairness monitoring checks if the model behaves differently across **race / gender / age_group** per batch."
        )

elif page == "Bias Monitoring":
    st.subheader("Streaming fairness monitoring")
    placeholder = st.empty()

    if (not st.session_state.trained) or (st.session_state.pipes.get("RTS") is None) or (st.session_state.pipes.get("RTP") is None):
        st.info("Train a model first, then click Start.")
        st.stop()
    if not cfg_protected:
        st.warning("Select at least one protected attribute for fairness monitoring.")
        st.stop()

    if st.session_state.run_stream:
        pipe_rts = st.session_state.pipes["RTS"]
        pipe_rtp = st.session_state.pipes["RTP"]
        stream_df = df.sample(frac=0.35, random_state=7).copy()

        missing = [c for c in cfg_protected if c not in stream_df.columns]
        if missing:
            st.error(f"Protected columns missing: {missing}")
            st.stop()

        for i, batch in enumerate(stream_batches(stream_df, batch_size=batch_size), start=1):
            if not st.session_state.run_stream:
                st.warning("Stopped.")
                break

            Xb = batch.drop(columns=[cfg_target_rts, cfg_target_rtp] + drop_cols, errors="ignore") 

            feature_cols = st.session_state.get("feature_cols", None)
            if feature_cols is not None:
             Xb = Xb.reindex(columns=feature_cols, fill_value=np.nan)
            
            yb_rts = batch[cfg_target_rts].astype(int)
            yb_rtp = batch[cfg_target_rtp].astype(int)
            pb = batch[cfg_protected].copy()

            y_pred_rts = pipe_rts.predict(Xb)
            y_pred_rtp = pipe_rtp.predict(Xb)

            # ✅ REAL-WORLD GATING (deployment logic):
# if athlete is not cleared (RTS=0), then RTP cannot be 1
            y_pred_rtp = ((y_pred_rts == 1) & (y_pred_rtp == 1)).astype(int)

            reports_rts = batch_fairness_report(yb_rts, y_pred_rts, pb, threshold=threshold)
            reports_rtp = batch_fairness_report(yb_rtp, y_pred_rtp, pb, threshold=threshold)

            # Combine both targets (keeps the same dashboard layout):
            # We prefix attributes so the root-cause text can still pick the worst driver.
            reports = []
            for r in reports_rts:
                r.protected_attr = f"RTS:{r.protected_attr}"
                reports.append(r)
            for r in reports_rtp:
                r.protected_attr = f"RTP:{r.protected_attr}"
                reports.append(r)

            max_dp = max(r.dp_diff for r in reports) if reports else 0.0
            max_eo = max(r.eo_diff for r in reports) if reports else 0.0
            has_alert = any(r.alert for r in reports)

            st.session_state.history.append({
                "batch": i,
                "max_dp_diff": float(max_dp),
                "max_eo_diff": float(max_eo),
                "alert": bool(has_alert),
                "details": [{
                    "attr": r.protected_attr,
                    "dp_diff": float(r.dp_diff),
                    "eo_diff": float(r.eo_diff),
                    "alert": bool(r.alert),
                    "by_group": r.by_group.reset_index()
                } for r in reports]
            })

            last = st.session_state.history[-1]

            with placeholder.container():
                k1, k2, k3 = st.columns(3)
                with k1: kpi_card("Batch", str(last["batch"]), "Incoming data chunk", "good")
                with k2:
                    mood = "bad" if last["max_dp_diff"] > threshold else "warn" if last["max_dp_diff"] > threshold*0.7 else "good"
                    kpi_card("Max demographic parity diff", f"{last['max_dp_diff']:.3f}", "Selection-rate gap", mood)
                with k3:
                    mood = "bad" if last["max_eo_diff"] > threshold else "warn" if last["max_eo_diff"] > threshold*0.7 else "good"
                    kpi_card("Max equalized odds diff", f"{last['max_eo_diff']:.3f}", "TPR / FPR gap", mood)

                st.markdown(
                    "<div class='card'><b>Root-cause attribution (automatic)</b><br/>"
                    + root_cause_text(last["details"], threshold)
                    + "</div>",
                    unsafe_allow_html=True
                )

                hist = st.session_state.history
                dp_series = [h["max_dp_diff"] for h in hist]
                eo_series = [h["max_eo_diff"] for h in hist]
                dp_s = ema(dp_series, alpha=0.25)
                eo_s = ema(eo_series, alpha=0.25)

                fig1 = go.Figure()
                fig1.add_trace(go.Scatter(x=list(range(1, len(dp_series)+1)), y=dp_series, mode="lines+markers", name="DP (raw)"))
                fig1.add_trace(go.Scatter(x=list(range(1, len(dp_s)+1)), y=dp_s, mode="lines", name="DP (smoothed)"))
                fig1.update_layout(title="Demographic Parity Difference (max across protected attrs)", xaxis_title="Batch", yaxis_title="Difference")
                st.plotly_chart(fig1, use_container_width=True, key=f"trend_dp_{last['batch']}")


                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=list(range(1, len(eo_series)+1)), y=eo_series, mode="lines+markers", name="EO (raw)"))
                fig2.add_trace(go.Scatter(x=list(range(1, len(eo_s)+1)), y=eo_s, mode="lines", name="EO (smoothed)"))
                fig2.update_layout(title="Equalized Odds Difference (max across protected attrs)", xaxis_title="Batch", yaxis_title="Difference")
                st.plotly_chart(fig2, use_container_width=True, key=f"trend_eo_{last['batch']}")

                with st.expander("Batch details (per protected attribute)"):
                    for info in last["details"]:
                        attr = str(info.get("attr", "Unknown")).title()
                        safe_attr = attr.replace(" ", "_").lower()

                        dp = float(info.get("dp_diff", 0))
                        eo = float(info.get("eo_diff", 0))
                        sev = max(dp, eo)
                        good = sev <= threshold
                        sev_pct = min(sev / threshold, 1.0) * 100

                        st.markdown(f"### {attr}")
                        cR1, cR2 = st.columns([1, 1])
                        with cR1:
                            st.plotly_chart(
                                ring(sev_pct, good, title="Bias severity"),
                                use_container_width=True,
                                key=f"ring_{last['batch']}_{safe_attr}"
                            )
                        with cR2:
                            summary = pd.DataFrame([{
                                "Attribute": attr,
                                "DP diff": round(dp, 3),
                                "EO diff": round(eo, 3),
                                "Severity (max)": round(sev, 3),
                                "Status": "🟢 OK" if good else "🔴 ALERT"
                            }])
                            st.dataframe(summary, use_container_width=True, hide_index=True)

                        st.markdown("**Group breakdown**")
                        st.dataframe(info["by_group"], use_container_width=True, hide_index=True)

            if interval > 0:
                time.sleep(interval)

        st.session_state.run_stream = False

    else:
        st.info("Click **Start** in the sidebar to stream in real time.")

    st.caption("Tip: Best results when your target is binary (0/1) and protected columns are categorical (or binned like age_group).")

elif page == "Explainability":
    st.header("AI Explainability (SHAP)")

    if not st.session_state.get("trained", False):
        st.warning("Train model first from Dashboard.")
        st.stop()

    try:
        import shap
        import matplotlib.pyplot as plt
    except Exception:
        st.error("Missing SHAP or matplotlib.")
        st.code("pip install shap matplotlib", language="bash")
        st.stop()

    which = st.selectbox("Explain which target?", ["RTS", "RTP"], index=1)

    pipe = st.session_state.get("pipes", {}).get(which, None)
    X_train = st.session_state.get("X_train", None)

    if pipe is None or X_train is None or len(X_train) == 0:
        st.error("No training pipeline/data found. Train first.")
        st.stop()

    # --- sample rows to explain ---
    X_sample = X_train.sample(min(200, len(X_train)), random_state=42)

    # --- pull steps from pipeline ---
    pre = pipe.named_steps["pre"]
    model = pipe.named_steps["clf"]

    # --- make SHAP-safe transformed dataframe (numeric + named) ---
    X_shap = make_shap_ready(pre, X_sample)

    # --- build explainer & compute shap values ---
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_shap)

    # Handle classifier outputs (list for classes)
    if isinstance(shap_vals, list):
        shap_vals_plot = shap_vals[1]                 # positive class
        expected_value = explainer.expected_value[1]
    else:
        shap_vals_plot = shap_vals
        expected_value = explainer.expected_value

    # =========================================================
    # 1) GLOBAL EXPLANATION (SUMMARY PLOT)
    # =========================================================
    st.subheader("Global Feature Importance")
    fig, ax = plt.subplots()
    shap.summary_plot(shap_vals_plot, X_shap, show=False)
    st.pyplot(fig, clear_figure=True)

    st.divider()

    # =========================================================
    # 2) EXPLAIN ONE PREDICTION (FORCE PLOT)
    # =========================================================
    st.subheader("Explain one prediction")
    row_index = st.slider("Select row to explain", 0, len(X_sample) - 1, 0)

    single_raw = X_sample.iloc[[row_index]]
    single_shap = make_shap_ready(pre, single_raw)

    shap_single = explainer.shap_values(single_shap)
    if isinstance(shap_single, list):
        shap_single_plot = shap_single[1]            # positive class
    else:
        shap_single_plot = shap_single

    fig2 = plt.figure()
    shap.plots.force(
        expected_value,
        shap_single_plot[0],
        single_shap.iloc[0],
        matplotlib=True,
        show=False
    )
    st.pyplot(fig2, clear_figure=True)

    st.divider()

    # =========================================================
    # 3) TABULAR EXPLANATION (FEATURES + VALUES + SHAP IMPACT)
    # =========================================================
    st.subheader("Feature Contribution Table")

    # shap values for that single row (1D)
    shap_row = shap_single_plot[0]
    table = pd.DataFrame({
        "Feature": single_shap.columns,
        "Feature Value": single_shap.iloc[0].values,
        "Impact (SHAP)": shap_row,
    })

    # sort by strongest absolute impact
    table["AbsImpact"] = table["Impact (SHAP)"].abs()
    table = table.sort_values("AbsImpact", ascending=False).drop(columns="AbsImpact")

    st.dataframe(
        table.style.background_gradient(cmap="RdYlGn", subset=["Impact (SHAP)"]),
        use_container_width=True
    )
elif page == "Data Health":

    st.header("Dataset Health Check")

    h = dataset_health(df, target=cfg_target_rtp)
    h1, h2, h3, h4 = st.columns(4)
    with h1: kpi_card("Rows", str(h["rows"]), f"Columns: {h['cols']}", "good")
    with h2:
        mood = "good" if h["missing_pct"] < 5 else "warn" if h["missing_pct"] < 15 else "bad"
        kpi_card("Missingness", f"{h['missing_pct']:.1f}%", "Average across cells", mood)
    with h3:
        mood = "good" if h["dup_pct"] < 1 else "warn" if h["dup_pct"] < 5 else "bad"
        kpi_card("Duplicates", f"{h['dup_pct']:.1f}%", "Row-duplicate rate", mood)
    with h4:
        if h["target_pos_rate"] is None:
            kpi_card("Target + rate", "—", "Not numeric/binary yet", "warn")
        else:
            mood = "good" if 25 <= h["target_pos_rate"] <= 75 else "warn"
            kpi_card("Target + rate", f"{h['target_pos_rate']:.1f}%", "Class balance", mood)

    st.divider()

    st.subheader("Missing values per column")
    st.dataframe(df.isna().sum().sort_values(ascending=False), use_container_width=True)

    st.subheader("Column types")
    st.dataframe(pd.DataFrame({"dtype": df.dtypes.astype(str)}), use_container_width=True)

    st.subheader("Preview")
    st.dataframe(df.head(50), use_container_width=True)

elif page == "Settings":
    st.header("Settings")

    st.subheader("Project info")
    st.write("Real-Time Bias-Aware AI for Readiness Monitoring")

    st.subheader("Current configuration")
    st.write("Model:", model_choice)
    st.write("Batch size:", batch_size)
    st.write("Threshold:", threshold)
    st.write("Interval:", interval)

    st.subheader("Export monitoring history")
    if st.session_state.history:
        hist_df = pd.DataFrame([{
            "batch": h["batch"],
            "max_dp_diff": h["max_dp_diff"],
            "max_eo_diff": h["max_eo_diff"],
            "alert": h["alert"],
        } for h in st.session_state.history])
        csv = hist_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download history CSV", csv, file_name="bias_monitor_history.csv", mime="text/csv")
    else:
        st.info("No monitoring history yet.")
