from __future__ import annotations
import time
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.figure_factory as ff

from src.data import SyntheticReadinessGenerator
from src.ingestion.loader_service import DataLoaderService
from src.preprocess import split_xy, clean_dataset 
from src.train import train_random_forest, train_gradient_boosting, evaluate, cross_validate_model
from src.monitor import batch_fairness_report
from src.realtime import stream_batches
from src.health import dataset_health
from src.processing.engine_selector import choose_processing_engine
from src.processing.profiler import dataset_profile
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
if "cv_results" not in st.session_state:
    st.session_state.cv_results = {"RTS": None, "RTP": None}    
if "X_train" not in st.session_state:
    st.session_state.X_train = None
if "history" not in st.session_state:
    st.session_state.history = []
if "run_stream" not in st.session_state:
    st.session_state.run_stream = False
if "raw_df" not in st.session_state:
    st.session_state.raw_df = None

if "clean_df" not in st.session_state:
    st.session_state.clean_df = None

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

with st.sidebar:
    st.header("Controls")

    st.subheader("Data source")

    source_mode = st.selectbox(
        "Choose source",
        ["Upload File", "MySQL", "PostgreSQL", "SQL Server", "Oracle"]
    )

    uploaded = None
    db_config = {}
    use_synth = False
    synth_n = 2000

    if source_mode == "Upload File":
        uploaded = st.file_uploader("Upload CSV/Excel (optional)", type=["csv", "xlsx", "xls"])
        use_synth = st.toggle("Use synthetic demo data", value=(uploaded is None))
        synth_n = st.slider("Synthetic rows", 500, 5000, 2000, 500)

    elif source_mode == "MySQL":
        db_config = {
            "source_type": "mysql",
            "host": st.text_input("MySQL Host", "localhost"),
            "port": st.number_input("MySQL Port", value=3306),
            "user": st.text_input("MySQL User"),
            "password": st.text_input("MySQL Password", type="password"),
            "database": st.text_input("MySQL Database"),
            "query": st.text_area("MySQL Query", "SELECT * FROM athlete_data LIMIT 1000")
        }

    elif source_mode == "PostgreSQL":
        db_config = {
            "source_type": "postgresql",
            "host": st.text_input("PostgreSQL Host", "localhost"),
            "port": st.number_input("PostgreSQL Port", value=5432),
            "user": st.text_input("PostgreSQL User"),
            "password": st.text_input("PostgreSQL Password", type="password"),
            "database": st.text_input("PostgreSQL Database"),
            "query": st.text_area("PostgreSQL Query", "SELECT * FROM athlete_data LIMIT 1000")
        }

    elif source_mode == "SQL Server":
        db_config = {
            "source_type": "sqlserver",
            "host": st.text_input("SQL Server Host", "localhost"),
            "database": st.text_input("SQL Server Database"),
            "user": st.text_input("SQL Server User"),
            "password": st.text_input("SQL Server Password", type="password"),
            "query": st.text_area("SQL Server Query", "SELECT TOP 1000 * FROM athlete_data")
        }

    elif source_mode == "Oracle":
        db_config = {
            "source_type": "oracle",
            "host": st.text_input("Oracle Host", "localhost"),
            "port": st.number_input("Oracle Port", value=1521),
            "service_name": st.text_input("Oracle Service Name"),
            "user": st.text_input("Oracle User"),
            "password": st.text_input("Oracle Password", type="password"),
            "query": st.text_area("Oracle Query", "SELECT * FROM athlete_data FETCH FIRST 1000 ROWS ONLY")
        }

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

/* Buttons */
div.stButton > button {
  background: #003b8e !important;
  color: white !important;
  border: 1px solid #003b8e !important;
  border-radius: 10px !important;
  font-weight: 600 !important;
}

div.stButton > button:hover {
  background: #002a66 !important;
  color: white !important;
  border: 1px solid #002a66 !important;
}

/* Toggle */
div[data-baseweb="switch"] > div {
  background-color: #b9c7db !important;
}

div[data-baseweb="switch"][aria-checked="true"] > div {
  background-color: #003b8e !important;
}
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
            dict(
                text=f"<b>{value_pct:.0f}%</b>",
                x=0.5,
                y=0.52,
                showarrow=False,
                font=dict(size=28, color=color)
            ),
            dict(
                text=title,
                x=0.5,
                y=0.18,
                showarrow=False,
                font=dict(size=12, color="#6b7280")
            ),
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

def show_perf_row(perf):
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Accuracy", f"{perf['Accuracy']:.3f}")
    with c2:
        st.metric("Precision", f"{perf['Precision']:.3f}")
    with c3:
        st.metric("Recall", f"{perf['Recall']:.3f}")
    with c4:
        st.metric("F1-score", f"{perf['f1']:.3f}")
    with c5:
        st.metric("ROC-AUC", f"{perf['Roc_Auc']:.3f}")


def plot_confusion_matrix_card(perf: dict, title: str):
    cm = [
        [perf["TN"], perf["FP"]],
        [perf["FN"], perf["TP"]],
    ]

    total = sum(sum(row) for row in cm)
    pct = [[(v / total * 100) if total > 0 else 0 for v in row] for row in cm]

    annotation_text = [
        [f"{cm[i][j]}<br>{pct[i][j]:.1f}%" for j in range(2)]
        for i in range(2)
    ]

    fig = ff.create_annotated_heatmap(
        z=cm,
        x=["Pred 0", "Pred 1"],
        y=["Actual 0", "Actual 1"],
        annotation_text=annotation_text,
        colorscale="Blues",
        showscale=True,
    )

    fig.update_layout(
        title=title,
        xaxis_title="Predicted Label",
        yaxis_title="True Label",
        margin=dict(l=30, r=20, t=60, b=30),
        height=340,
    )
    fig.update_yaxes(autorange="reversed")

    st.plotly_chart(fig, use_container_width=True)

def run_end_to_end_checks(df, pipes, perfs):
    """
    Simple end-to-end smoke checks for the core workflow.
    """
    results = []

    if df is not None and len(df) > 0:
        results.append(("PASS", "Dataset loaded successfully"))
    else:
        results.append(("FAIL", "Dataset not loaded"))

    if pipes.get("RTS") is not None:
        results.append(("PASS", "RTS model trained"))
    else:
        results.append(("FAIL", "RTS model missing"))

    if pipes.get("RTP") is not None:
        results.append(("PASS", "RTP model trained"))
    else:
        results.append(("FAIL", "RTP model missing"))

    if perfs.get("RTS") is not None:
        results.append(("PASS", "RTS evaluation completed"))
    else:
        results.append(("FAIL", "RTS evaluation missing"))

    if perfs.get("RTP") is not None:
        results.append(("PASS", "RTP evaluation completed"))
    else:
        results.append(("FAIL", "RTP evaluation missing"))

    return results
    
# ===== Header =====
st.title("Real-Time Bias-Aware AI for Readiness Monitoring")
st.caption("Dual-target RTS/RTP prediction with Gradient Boosting as the primary model, "
    "Random Forest as baseline, and real-time fairness monitoring using DP/EO over streaming batches.")


# ===== Load data =====
def get_dataframe() -> pd.DataFrame | None:
    # Upload + synthetic mode
    if source_mode == "Upload File" and use_synth:
        gen = SyntheticReadinessGenerator(seed=42)
        df = gen.make(n=int(synth_n))
        st.session_state.raw_df = df.copy()
        st.session_state.clean_df = df.copy()
        return df

    # Upload file mode
    if source_mode == "Upload File":
        if uploaded is None:
            st.session_state.raw_df = None
            st.session_state.clean_df = None
            return None

        try:
            raw_df = DataLoaderService.load_from_uploaded_file(uploaded)
        except Exception as e:
            st.error(f"Failed to load uploaded file: {e}")
            return None

    # Database mode
    else:
        # basic check so empty form does not immediately crash
        required_keys = {
            "MySQL": ["host", "port", "user", "password", "database", "query"],
            "PostgreSQL": ["host", "port", "user", "password", "database", "query"],
            "SQL Server": ["host", "database", "user", "password", "query"],
            "Oracle": ["host", "port", "service_name", "user", "password", "query"],
        }

        missing = []
        for key in required_keys.get(source_mode, []):
            val = db_config.get(key)
            if val is None or (isinstance(val, str) and not val.strip()):
                missing.append(key)

        if missing:
            st.info(f"Enter database details to load data from {source_mode}: {', '.join(missing)}")
            return None

        try:
            raw_df = DataLoaderService.load_from_database(**db_config)
        except Exception as e:
            st.error(f"Failed to load from {source_mode}: {e}")
            return None

    # reset clean_df when a new source/file is loaded
    if (
        st.session_state.raw_df is None
        or len(raw_df) != len(st.session_state.raw_df)
        or list(raw_df.columns) != list(st.session_state.raw_df.columns)
    ):
        st.session_state.raw_df = raw_df.copy()
        st.session_state.clean_df = None
        return raw_df

    st.session_state.raw_df = raw_df.copy()

    if st.session_state.clean_df is None:
        return raw_df

    return st.session_state.clean_df.copy()

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
    Auto-detect columns and create RTS/RTP labels if missing.

    RTS is created from rehabilitation-style readiness signals such as
    injury status, fatigue, recovery, and risk-related variables.

    RTP is created using stricter performance-oriented signals such as
    performance, load balance, team contribution, and fatigue filtering.

    Important:
    - RTP cannot be positive if RTS is negative.
    - If the dataset does not contain enough information to create RTP safely,
      the function raises an error instead of generating artificial labels. 
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
    col_perf = _find_col(df, [ "performance_score", "performance", "perfscore", "perf_score", "rating", "score",
    "jump_height", "gait_speed", "speed", "recovery_score"])
    col_fat  = _find_col(df, [ "fatigue_score", "fatigue_index", "exertion", "rpe",
    "fatigue_index", "stress_level"])
    col_rec  = _find_col(df, ["recovery_days_per_week", "recovery", "rest_days", "rehab_days", "days_recovery",
    "recovery_score", "sleep_quality", "hydration_level"])
    col_acl  = _find_col(df, [ "acl_risk_score", "acl_risk", "reinjury_risk", "injury_risk", "risk_score", "risk",
    "injury_occurred", "injury_flag", "fatigue_index", "stress_level"])
    col_lb   = _find_col(df, [ "load_balance_score", "load_balance", "balance_score", "workload_balance",
    "training_load", "hydration_level"])
    col_team = _find_col(df, [ "team_contribution_score", "team_contribution", "contribution", "impact_score",
    "range_of_motion", "jump_height", "gait_speed"])

    # injury indicator (0/1) detection
    col_inj = _find_col(df, [ "injury_indicator", "injury", "injured", "injury_flag", "injury_status",
    "injury_occurred"])

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
            thr = perf.quantile(0.35)
            df["rts"] = (perf >= thr).astype(int)

    # --- Create RTP if missing ---
    if "rtp" not in df.columns:
        # Strong RTP rule if we have performance + balance + team + fatigue
                #  ELITE RTP: quantile-based (works on any dataset scale)
        if col_perf:
            perf = pd.to_numeric(df[col_perf], errors="coerce")
            thr_perf = perf.quantile(0.25)   # top 30%
            rtp_mask = (perf >= thr_perf)

            if col_lb:
                lb = pd.to_numeric(df[col_lb], errors="coerce")
                rtp_mask = rtp_mask & (lb >= lb.quantile(0.15))  # top 40%

            if col_team:
                team = pd.to_numeric(df[col_team], errors="coerce")
                rtp_mask = rtp_mask & (team >= team.quantile(0.15))  # top 40%

            if col_fat:
                fat = pd.to_numeric(df[col_fat], errors="coerce")
                rtp_mask = rtp_mask & (fat <= fat.quantile(0.60))  # lower fatigue is better

           

            df["rtp"] = rtp_mask.fillna(False).astype(int)

        elif "rts" in df.columns:
            raise ValueError(
                "RTP could not be created safely. The dataset has RTS but lacks enough performance-related signals for a defensible RTP label."
            )


        # Fallback: performance + low risk
        elif col_perf and col_acl:
            perf = pd.to_numeric(df[col_perf], errors="coerce")
            risk = pd.to_numeric(df[col_acl], errors="coerce")
            df["rtp"] = ((perf >= perf.quantile(0.75)) & (risk <= risk.quantile(0.50))).astype(int)

        # Last fallback: just stricter than RTS
        elif "rts" in df.columns:
            raise ValueError(
                "RTP could not be created safely. The dataset does not contain enough information to derive RTP without artificial randomness."
            )
    return df


df = get_dataframe()

if st.session_state.clean_df is not None:
    df = st.session_state.clean_df.copy()

st.session_state.df = df

if df is not None:
    file_size_mb = None
    if source_mode == "Upload File" and uploaded is not None and hasattr(uploaded, "size"):
        file_size_mb = uploaded.size / (1024 ** 2)

    engine_used = choose_processing_engine(
        n_rows=len(df) if df is not None else None,
        file_size_mb=file_size_mb
    )

    profile = dataset_profile(df, target="rtp" if "rtp" in df.columns else None)

    st.caption(
        f"Processing engine: **{engine_used}** | "
        f"Rows: **{profile['rows']:,}** | "
        f"Cols: **{profile['cols']}** | "
        f"Memory: **{profile['memory_mb']:.2f} MB**"
    )

if df is None:
    st.info("Upload a dataset or enable synthetic data.")
    st.stop()

if df is not None:
    if "age" in df.columns and "age_group" not in df.columns:
        df["age_group"] = pd.cut(
            pd.to_numeric(df["age"], errors="coerce"),
            bins=[0, 18, 22, 26, 30, 35, 100],
            labels=["<=18", "19-22", "23-26", "27-30", "31-35", "36+"]
        ).astype(str)

    if "injury_occured" in df.columns:
        df["injury_occured"] = pd.to_numeric(df["injury_occured"], errors="coerce")
        df["injury_occured"] = (df["injury_occured"].fillna(0) > 0).astype(int)


#  SMART auto-create RTS/RTP if missing
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
    default=[c for c in ["race", "gender", "age_group", "Position", "Gender", "Age"] if c in cols]
)
drop_cols = st.multiselect(
    "Columns to drop (IDs, timestamps, leakage)",
    cols,
   default=[c for c in cols if c.lower() in {
"id","patient_id","athlete_id","timestamp",
"performance_score",
"load_balance_score",
"team_contribution_score",
"fatigue_score", "injury_occurred", "injury_flag",
    "fatigue_index", "recovery_score",
    "jump_height", "range_of_motion",
    "training_load", "session_id"
}]

)
# ===== Train =====
# 1. validates that both selected targets are binary
# 2. removes target and leakage-prone columns from features
# 3. splits the dataset using the RTS split and aligns RTP labels by index
# 4. trains two separate models (RTS + RTP)
# 5. evaluates both models on held-out data
# 6. computes 5-fold cross-validation summaries 
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

    # remove BOTH targets from features
    drop_cols_train = list(set(drop_cols + [cfg_target_rtp, cfg_target_rts]))

    # split for RTS
    X_train, X_test, y_train_rts, y_test_rts = split_xy(
        df,
        target=cfg_target_rts,
        drop_cols=drop_cols_train
    )

    st.session_state.feature_cols = list(X_train.columns)

    # align RTP labels to same train/test rows
    y_rtp = df[cfg_target_rtp].astype(int)
    y_train_rtp = y_rtp.loc[X_train.index]
    y_test_rtp = y_rtp.loc[X_test.index]

    # both classes must exist
    if y_train_rts.nunique() < 2:
        st.error("RTS has only one class in the training split. Adjust RTS rules/thresholds.")
        st.write(df[cfg_target_rts].value_counts())
        st.stop()

    if y_train_rtp.nunique() < 2:
        st.error("RTP has only one class in the training split. Adjust RTP rules/thresholds.")
        st.write(df[cfg_target_rtp].value_counts())
        st.stop()

    # --- train two pipelines (RTS + RTP) with optional SMOTE ---
    # RTS: no SMOTE
    # RTP: use SMOTE
    
    if model_key == "gb":
        pipe_rts = train_gradient_boosting(
            X_train, y_train_rts,
            n_estimators=250,
            learning_rate=0.05,
            max_depth=3,
            use_smote=False
        )
        pipe_rtp = train_gradient_boosting(
            X_train, y_train_rtp,
            n_estimators=250,
            learning_rate=0.05,
            max_depth=3,
            use_smote=True
        )
        model_name = "GradientBoosting"
    else:
        pipe_rts = train_random_forest(
            X_train, y_train_rts,
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            use_smote=False
        )
        pipe_rtp = train_random_forest(
            X_train, y_train_rtp,
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            use_smote=True
        )
        model_name = "RandomForest"

    st.session_state.pipes = {"RTS": pipe_rts, "RTP": pipe_rtp}
    st.session_state.trained = True

    perf_rts = evaluate(pipe_rts, X_test, y_test_rts)
    perf_rtp = evaluate(pipe_rtp, X_test, y_test_rtp)

    cv_rts = cross_validate_model(
        X_train,
        y_train_rts,
        model_name=model_key,
        cv=5,
        seed=42,
    )
    cv_rtp = cross_validate_model(
        X_train,
        y_train_rtp,
        model_name=model_key,
        cv=5,
        seed=42,
    )

    st.session_state.perfs = {
        "RTS": perf_rts,
        "RTP": perf_rtp,
    }
    st.session_state.cv_results = {
        "RTS": cv_rts,
        "RTP": cv_rtp,
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
            cv = st.session_state.cv_results.get("RTS")

            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Accuracy", f"{perf['Accuracy']:.3f}")
            with m2:
                st.metric("Precision", f"{perf['Precision']:.3f}")
            with m3:
                st.metric("Recall", f"{perf['Recall']:.3f}")

            m4, m5, m6 = st.columns(3)
            with m4:
                st.metric("F1-score", f"{perf['f1']:.3f}")
            with m5:
                st.metric("ROC-AUC", f"{perf['Roc_Auc']:.3f}")
            with m6:
                st.metric("Specificity", f"{perf['Specificity']:.3f}")

            st.caption(
                f"Test size: {perf['Test_Size']} | "
                f"Positive rate: {perf['Positive_Rate']:.3f} | "
                f"TP={perf['TP']} TN={perf['TN']} FP={perf['FP']} FN={perf['FN']}"
            )

            if cv:
                st.caption(
                    f"Cross-validation F1: mean={cv['cv_f1_mean']:.3f}, std={cv['cv_f1_std']:.3f}"
                )

            st.markdown("#### RTP (Return to Performance)")
            perf = perfs["RTP"]
            cv = st.session_state.cv_results.get("RTP")

            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Accuracy", f"{perf['Accuracy']:.3f}")
            with m2:
                st.metric("Precision", f"{perf['Precision']:.3f}")
            with m3:
                st.metric("Recall", f"{perf['Recall']:.3f}")

            m4, m5, m6 = st.columns(3)
            with m4:
                st.metric("F1-score", f"{perf['f1']:.3f}")
            with m5:
                st.metric("ROC-AUC", f"{perf['Roc_Auc']:.3f}")
            with m6:
                st.metric("Specificity", f"{perf['Specificity']:.3f}")

            st.caption(
                f"Test size: {perf['Test_Size']} | "
                f"Positive rate: {perf['Positive_Rate']:.3f} | "
                f"TP={perf['TP']} TN={perf['TN']} FP={perf['FP']} FN={perf['FN']}"
            )

            if cv:
                st.caption(
                    f"Cross-validation F1: mean={cv['cv_f1_mean']:.3f}, std={cv['cv_f1_std']:.3f}"
                )

        else:
            st.info("Train a model to see RTS/RTP metrics.")

    with cB:
        st.subheader("How prediction works (simple)")
        st.write(
            "- **RTS** predicts whether the athlete is clinically ready to return to sport.\n"
            "- **RTP** predicts whether the athlete is likely to return to prior performance level.\n"
            "- We clean the data, fill missing values, and encode categories numerically.\n"
            "- The models output a binary class: **1 = positive outcome**, **0 = negative outcome**.\n"
            "- Fairness monitoring checks whether model behavior differs across **race / gender / age_group** in streaming batches."
        )

        # =========================================================
    # K-FOLD CROSS-VALIDATION RESULTS (FULL WIDTH)
    # =========================================================
    cv_results = st.session_state.get("cv_results", {})
    cv_rts = cv_results.get("RTS")
    cv_rtp = cv_results.get("RTP")

    if (
        cv_rts and cv_rts.get("cv_scores") and
        cv_rtp and cv_rtp.get("cv_scores")
    ):
        st.markdown("### K-Fold Cross-Validation Results")

        st.markdown("#### RTS Fold Scores")
        rts_scores = cv_rts["cv_scores"]
        fig_rts = go.Figure()
        fig_rts.add_trace(go.Bar(
            x=[f"Fold {i+1}" for i in range(len(rts_scores))],
            y=rts_scores,
            text=[f"{s:.3f}" for s in rts_scores],
            textposition="outside",
            textfont=dict(size=16, color="black"),
            name="RTS F1"
        ))
        fig_rts.add_hline(
            y=cv_rts["cv_f1_mean"],
            line_dash="dash",
            annotation_text=f"Mean = {cv_rts['cv_f1_mean']:.3f}",
            annotation_position="top left"
        )
        fig_rts.update_layout(
            title=dict(
                text="RTS 5-Fold Cross-Validation",
                font=dict(size=22, color="black")
            ),
            xaxis=dict(
                title=dict(text="Fold", font=dict(size=18, color="black")),
                tickfont=dict(size=16, color="black")
            ),
            yaxis=dict(
                title=dict(text="F1 Score", font=dict(size=18, color="black")),
                tickfont=dict(size=16, color="black"),
                range=[0, 1]
            ),
            height=430,
            showlegend=False
        )
        fig_rts.update_annotations(font=dict(size=16, color="black"))
        st.plotly_chart(fig_rts, use_container_width=True)
        st.caption(
            f"RTS mean = {cv_rts['cv_f1_mean']:.3f} | "
            f"std = {cv_rts['cv_f1_std']:.3f} | "
            f"best = {max(rts_scores):.3f} | "
            f"worst = {min(rts_scores):.3f}"
        )

        st.markdown("#### RTP Fold Scores")
        rtp_scores = cv_rtp["cv_scores"]
        fig_rtp = go.Figure()
        fig_rtp.add_trace(go.Bar(
            x=[f"Fold {i+1}" for i in range(len(rtp_scores))],
            y=rtp_scores,
            text=[f"{s:.3f}" for s in rtp_scores],
            textposition="outside",
            textfont=dict(size=16, color="black"),
            name="RTP F1"
        ))
        fig_rtp.add_hline(
            y=cv_rtp["cv_f1_mean"],
            line_dash="dash",
            annotation_text=f"Mean = {cv_rtp['cv_f1_mean']:.3f}",
            annotation_position="top left"
        )
        fig_rtp.update_layout(
            title=dict(
                text="RTP 5-Fold Cross-Validation",
                font=dict(size=22, color="black")
            ),
            xaxis=dict(
                title=dict(text="Fold", font=dict(size=18, color="black")),
                tickfont=dict(size=16, color="black")
            ),
            yaxis=dict(
                title=dict(text="F1 Score", font=dict(size=18, color="black")),
                tickfont=dict(size=16, color="black"),
                range=[0, 1]
            ),
            height=430,
            showlegend=False
        )
        fig_rtp.update_annotations(font=dict(size=16, color="black"))
        st.plotly_chart(fig_rtp, use_container_width=True)
        st.caption(
            f"RTP mean = {cv_rtp['cv_f1_mean']:.3f} | "
            f"std = {cv_rtp['cv_f1_std']:.3f} | "
            f"best = {max(rtp_scores):.3f} | "
            f"worst = {min(rtp_scores):.3f}"
        )

    else:
        st.info("No K-fold results available yet. Train the model first.")
    # =========================================================
    # CONFUSION MATRICES BELOW THE DASHBOARD, SIDE BY SIDE
    # =========================================================
    perfs = st.session_state.perfs
    if perfs and perfs.get("RTS") and perfs.get("RTP"):
        st.markdown("### Confusion Matrix Analysis")

        cm1, cm2 = st.columns(2)

        with cm1:
            st.markdown("#### RTS")
            plot_confusion_matrix_card(perfs["RTS"], "RTS Confusion Matrix")

        with cm2:
            st.markdown("#### RTP")
            plot_confusion_matrix_card(perfs["RTP"], "RTP Confusion Matrix")

 # ===== Bias Monitoring Page =====
# Streams batches of data through the trained RTS/RTP models,
# computes fairness metrics for selected protected attributes,
# stores batch-level history, and displays trends + root-cause summaries.

elif page == "Bias Monitoring":
    st.subheader("Streaming fairness monitoring")
    placeholder = st.empty()

    if (not st.session_state.trained) or (st.session_state.pipes.get("RTS") is None) or (st.session_state.pipes.get("RTP") is None):
        st.info("Train a model first, then click Start.")
        st.stop()
    
    if not cfg_protected:
        st.warning("Select at least one protected attribute for fairness monitoring.")
        st.stop()

    for col in cfg_protected:
        if col in df.columns:
            group_counts = df[col].value_counts(dropna=False)
            small_groups = group_counts[group_counts < 5]
            if not small_groups.empty:
                st.warning(
                    f"Fairness caution: column '{col}' has groups with fewer than 5 samples. "
                    "Bias estimates for those groups may be unstable."
                )

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

# ===== Explainability Page =====
# Uses SHAP to show:
# - global feature importance for RTS/RTP
# - local explanation for one selected row
# - tabular SHAP contribution breakdown

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

    # ===== Data Health Page =====
# Summarizes missingness, duplicates, class balance, and dtypes.
# Also allows the user to explicitly clean the uploaded dataset
# before training or fairness monitoring.

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
           
    if st.session_state.raw_df is None:
        st.info("Upload a dataset first.")
        st.stop()

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns([0.8, 1.2, 2.0])

    with c1:
        clean_btn = st.button("Clean Data", type="primary", use_container_width=True)

    with c2:
        show_clean = st.toggle(
            "Show cleaned data",
            value=(st.session_state.clean_df is not None)
        )

    with c3:
     
     st.empty()

    if clean_btn:
        cleaned = clean_dataset(
            st.session_state.raw_df.copy(),
            target_cols=["rts", "rtp", "ready"],
            protected_cols=["gender", "race", "age", "age_group"],
            drop_suspicious=False,
        )

        cleaned = add_rts_rtp_smart(cleaned) 

        if "rts" in cleaned.columns and "rtp" in cleaned.columns:
            cleaned.loc[cleaned["rts"] == 0, "rtp"] = 0

        st.session_state.clean_df = cleaned.copy()
        st.success("Data cleaned. RTS/RTP created after cleaning.")

    if show_clean and st.session_state.clean_df is not None:
        view_df = st.session_state.clean_df.copy()
    else:
        view_df = st.session_state.raw_df.copy() 
        
    st.divider()

    st.subheader("Missing values per column")
    st.dataframe(df.isna().sum().sort_values(ascending=False), use_container_width=True)

    st.subheader("Column types")
    st.dataframe(pd.DataFrame({"dtype": df.dtypes.astype(str)}), use_container_width=True)

    st.subheader("Preview")
    st.dataframe(df.head(50), use_container_width=True)



elif page == "Settings":

    st.subheader("Fairness assumptions")
    st.markdown("""
- Protected attributes are selected by the user from available columns.
- Bias is monitored using Demographic Parity Difference and Equalized Odds Difference.
- The default alert threshold is 0.10.
- This threshold is a prototype monitoring boundary, not a legal or clinical standard.
- Fairness estimates may be unstable when subgroup sizes are very small.
- RTP is gated by RTS, so an athlete not cleared for sport cannot be positive for RTP.
""")
    
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

        st.subheader("End-to-end functionality checks")
    checks = run_end_to_end_checks(
        st.session_state.get("df"),
        st.session_state.get("pipes", {}),
        st.session_state.get("perfs", {}),
    )

    for status, message in checks:
        if status == "PASS":
            st.success(message)
        else:
            st.error(message)
