from __future__ import annotations

import re
import numpy as np
import pandas as pd

from collections.abc import Iterable 
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


# -------------------------------
# 1) Robust type cleaning
# -------------------------------
_TRUTHY = {"1", "true", "t", "yes", "y", "ready", "cleared", "pass", "passed"}
_FALSY  = {"0", "false", "f", "no", "n", "not ready", "uncleared", "fail", "failed"}


def _to_boolish_int(s: pd.Series) -> pd.Series:
    """Convert mixed boolean-like strings/numbers to 0/1 when possible. Otherwise return original."""
    if not pd.api.types.is_object_dtype(s) and not pd.api.types.is_string_dtype(s):
        return s
    low = s.astype(str).str.strip().str.lower()
    if low.isin(_TRUTHY | _FALSY).mean() > 0.6:
        return low.map(lambda v: 1 if v in _TRUTHY else (0 if v in _FALSY else np.nan))
    return s


def coerce_datetime_cols(X: pd.DataFrame) -> pd.DataFrame:
    """Convert datetime columns to numeric timestamps (seconds) so sklearn can handle them."""
    out = X.copy()
    for c in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[c]):
            out[c] = out[c].view("int64") / 1e9
    return out


def coerce_numeric_cols(X: pd.DataFrame) -> pd.DataFrame:
    out = X.copy()

    for c in out.columns:
        if pd.api.types.is_object_dtype(out[c]) or pd.api.types.is_string_dtype(out[c]):
            tmp = _to_boolish_int(out[c])
            if not tmp.equals(out[c]):
                out[c] = tmp
                continue

            cleaned = (
                out[c]
                .astype(str)
                .str.replace(r"[\$,]", "", regex=True)
                .str.replace("%", "", regex=False)
                .str.strip()
            )
            numeric = pd.to_numeric(cleaned, errors="coerce")

            if numeric.notna().mean() >= 0.70:
                out[c] = numeric

    return out


def normalize_categorical_values(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    value_maps = {
        "gender": {
            "m": "Male",
            "male": "Male",
            "man": "Male",
            "f": "Female",
            "female": "Female",
            "woman": "Female",
            "unknown": "Unknown",
            "unk": "Unknown",
            "na": "Unknown",
            "n/a": "Unknown",
            "none": "Unknown",
            "nan": "Unknown",
        },
        "race": {
            "africanamerican": "AfricanAmerican",
            "african_american": "AfricanAmerican",
            "black": "AfricanAmerican",
            "caucasian": "Caucasian",
            "white": "Caucasian",
            "hispanic": "Hispanic",
            "latino": "Hispanic",
            "asian": "Asian",
            "other": "Other",
            "unknown": "Unknown",
            "unk": "Unknown",
            "na": "Unknown",
            "n/a": "Unknown",
            "none": "Unknown",
            "nan": "Unknown",
        },
    }

    for col in out.columns:
        if not (pd.api.types.is_object_dtype(out[col]) or pd.api.types.is_string_dtype(out[col])):
            continue

        s = out[col].astype(str).str.strip()
        s = s.replace(
            {
                "": np.nan,
                "nan": np.nan,
                "None": np.nan,
                "none": np.nan,
                "NULL": np.nan,
                "null": np.nan,
                "N/A": np.nan,
                "n/a": np.nan,
            }
        )

        low = s.astype(str).str.lower().str.replace(r"[^a-z0-9]+", "", regex=True)

        if col.lower() in value_maps:
            mapped = low.map(value_maps[col.lower()])
            out[col] = mapped.where(mapped.notna(), s)
        else:
            nunique = s.nunique(dropna=True)
            if nunique <= 30:
                out[col] = s.astype(str).str.strip().str.title()

    return out


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates().reset_index(drop=True)


def cap_outliers_iqr(
    df: pd.DataFrame,
    exclude_cols: Iterable[str] | None = None,
    whisker_width: float = 1.5,
) -> pd.DataFrame:
    out = df.copy()
    exclude_cols = set(exclude_cols or [])

    num_cols = [
        c for c in out.columns
        if c not in exclude_cols and pd.api.types.is_numeric_dtype(out[c])
    ]

    for c in num_cols:
        s = pd.to_numeric(out[c], errors="coerce")
        if s.notna().sum() < 5:
            continue

        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1

        if not np.isfinite(iqr) or iqr == 0:
            continue

        lower = q1 - whisker_width * iqr
        upper = q3 + whisker_width * iqr
        out[c] = s.clip(lower, upper)

    return out


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [
        str(c).strip().lower().replace(" ", "_").replace("-", "_")
        for c in out.columns
    ]
    return out

def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    num_cols = [c for c in out.columns if pd.api.types.is_numeric_dtype(out[c])]
    cat_cols = [c for c in out.columns if c not in num_cols]

    for c in num_cols:
        med = pd.to_numeric(out[c], errors="coerce").median()
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(med)

    for c in cat_cols:
        mode = out[c].mode(dropna=True)
        fill_value = mode.iloc[0] if not mode.empty else "Unknown"
        out[c] = out[c].fillna(fill_value)

    return out

def clean_dataset(
        
    df: pd.DataFrame,
    target_cols: list[str] | None = None,
    protected_cols: list[str] | None = None,
    drop_suspicious: bool = False,
) -> pd.DataFrame:
    out = df.copy()
    target_cols = target_cols or []
    protected_cols = protected_cols or []
    """
    Run the main dataset-cleaning pipeline before training or monitoring.

    Steps:
    - normalize column names
    - convert datetimes to numeric timestamps
    - coerce numeric-like strings into numeric values
    - normalize low-cardinality categorical values
    - remove duplicate rows
    - add athlete-specific helper features such as age_group
    - fill missing values
    - cap numeric outliers using IQR

    Why this matters:
    Keeps raw uploaded data and cleaned data separate, and ensures
    downstream training/fairness analysis runs on consistent input.
    """
    out = normalize_column_names(out)
    out = coerce_datetime_cols(out)
    out = coerce_numeric_cols(out)
    out = normalize_categorical_values(out)
    out = remove_duplicates(out)
    out = adapt_athlete_dataset(out)
    out = fill_missing_values(out)

    exclude = set(target_cols) | set(protected_cols)
    out = cap_outliers_iqr(out, exclude_cols=exclude)
    

    if drop_suspicious and "suspicious_row" in out.columns:
        out = out[out["suspicious_row"] == 0].copy()

    return out.reset_index(drop=True)


# -------------------------------
# 2) Small dataset adapter
# -------------------------------
def adapt_athlete_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add lightweight athlete-specific helper features without changing
    the meaning of the original dataset.

    Current behavior:
    - creates age_group when age exists but age_group is missing

    Why this matters:
    age_group is useful for protected-attribute fairness monitoring,
    while preserving the original uploaded columns.
    """
    out = df.copy()

   
    # Create age_group for fairness monitoring if only age exists
    if "age" in out.columns and "age_group" not in out.columns:
        age_num = pd.to_numeric(out["age"], errors="coerce")
        out["age_group"] = pd.cut(
            age_num,
            bins=[0, 20, 25, 30, 100],
            labels=["<=20", "21-25", "26-30", "30+"]
        ).astype(str)

    return out


# -------------------------------
# 3) Auto-create RTS/RTP targets
# -------------------------------
def _find_col(cols: list[str], patterns: list[str]) -> str | None:
    """Find first column whose lowercase name matches any regex pattern."""
    for p in patterns:
        rx = re.compile(p)
        for c in cols:
            if rx.search(c.lower()):
                return c
    return None


def ensure_rts_rtp(
    df: pd.DataFrame,
    rts_col: str = "rts",
    rtp_col: str = "rtp",
) -> pd.DataFrame:
    """
    Create RTS and RTP targets when they are missing.

    RTS is a rehabilitation-readiness proxy.
    RTP is a stricter sustained-performance proxy.

    Rules:
    - use semantically matched rehabilitation/performance columns when possible
    - build a composite readiness score from available numeric signals
    - derive RTS and RTP using adaptive quantile thresholds
    - enforce the domain rule that RTP cannot be positive when RTS is negative

    Important:
    These targets are operational proxies for prototyping and validation.
    They do not replace clinician-verified ground truth labels.
    """
    out = adapt_athlete_dataset(df)
    cols = out.columns.tolist()

    if rts_col in cols and rtp_col in cols:
        return out

    # Make a cleaned view for building labels
    cleaned = coerce_numeric_cols(coerce_datetime_cols(out))

    # Candidate columns by common names
    # Added support for your uploaded athlete dataset
    strength = _find_col(cols, [
        r"strength", r"quad", r"hamstring", r"power", r"jump"
    ])
    balance = _find_col(cols, [
        r"balance", r"stability", r"rom", r"mobility", r"hop", r"gait"
    ])
    perf = _find_col(cols, [
        r"performance", r"score", r"rating", r"speed", r"jump_height", r"gait_speed"
    ])
    recovery = _find_col(cols, [
        r"recovery", r"rehab", r"days", r"duration", r"time",
        r"recovery_score", r"sleep", r"hydration"
    ])
    risk = _find_col(cols, [
        r"risk", r"reinjury", r"injury_risk", r"prob",
        r"injury_flag", r"injury_occurred", r"fatigue", r"stress"
    ])

    # Helper to normalize numeric column into [0,1]
    def norm(colname: str) -> pd.Series:
        s = pd.to_numeric(cleaned[colname], errors="coerce")
        lo, hi = np.nanpercentile(s, 5), np.nanpercentile(s, 95)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return pd.Series(np.nan, index=cleaned.index)
        return ((s - lo) / (hi - lo)).clip(0, 1)

    # Build a composite readiness score
    parts = []
    if strength:
        parts.append(norm(strength))
    if balance:
        parts.append(norm(balance))
    if perf:
        parts.append(norm(perf))

    # recovery: lower is better in the original generic logic, so invert
    if recovery:
        r = norm(recovery)
        parts.append(1 - r)

    # risk: lower is better, so invert
    if risk:
        r = norm(risk)
        parts.append(1 - r)

    if parts:
        score = pd.concat(parts, axis=1).mean(axis=1)
    else:
        # fallback: use mean of all numeric columns
        num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(cleaned[c])]

        if not num_cols:
            raise ValueError(
                "Could not construct RTS/RTP labels because the dataset lacks enough rehabilitation or performance signals."
            )
        score = cleaned[num_cols].mean(axis=1)

    # Thresholds: RTS easier, RTP stricter
    # Use quantiles so it adapts to dataset scale.
    if rts_col not in cols:
        thr_rts = np.nanquantile(score, 0.55)
        out[rts_col] = (score >= thr_rts).astype(int)

    if rtp_col not in cols:
        thr_rtp = np.nanquantile(score, 0.75)
        out[rtp_col] = (score >= thr_rtp).astype(int)

    # Real-world constraint: if not RTS, cannot be RTP
    if rts_col in out.columns and rtp_col in out.columns:
        out.loc[out[rts_col] == 0, rtp_col] = 0

    return out


# -------------------------------
# 4) Preprocessor builder
# -------------------------------
def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    # Convert datetime and numeric-like strings first
    """
    Build the sklearn preprocessing pipeline for mixed-type athlete data.

    Numeric columns:
    - median imputation
    - standard scaling

    Categorical columns:
    - most-frequent imputation
    - one-hot encoding

    Why this matters:
    Converts raw uploaded datasets into a consistent numeric format that
    can be used by both training and cross-validation pipelines.
    - Uses a 70/30 split by default, matching the report and dashboard
    """
    X = coerce_numeric_cols(coerce_datetime_cols(X))

    cat_cols = [
        c for c in X.columns
        if (X[c].dtype == "object" or pd.api.types.is_categorical_dtype(X[c]))
    ]
    num_cols = [c for c in X.columns if c not in cat_cols]

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
    )
    return pre


# -------------------------------
# 5) Split helper
# -------------------------------
def split_xy(
    df: pd.DataFrame,
    target: str,
    drop_cols: list[str] | None = None,
    test_size: float = 0.3,
    seed: int = 42,
):
    """
    Split a dataframe into train/test features and labels.

    Steps:
    - Coerce target into binary 0/1
    - Drop leakage or non-feature columns
    - Apply a stratified train/test split when possible

    Why this matters:
    - Keeps evaluation on unseen data
    - Makes reported performance more credible
    - Supports company-required validation
    """
    drop_cols = drop_cols or []

    y_raw = _to_boolish_int(df[target]) if target in df.columns else df[target]
    y_num = pd.to_numeric(y_raw, errors="coerce").fillna(0)
    y = (y_num > 0).astype(int)

    X = df.drop(columns=[target] + drop_cols, errors="ignore")
    X = coerce_numeric_cols(coerce_datetime_cols(X))

    strat = y if y.nunique() > 1 else None

    # Use a stratified 70/30 split when both classes are present so
    # the held-out evaluation is more reliable and class proportions are preserved.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=strat
    )
    return X_train, X_test, y_train, y_test