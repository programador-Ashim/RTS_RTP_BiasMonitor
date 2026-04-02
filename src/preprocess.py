from __future__ import annotations

import re
import numpy as np
import pandas as pd

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
    """
    Try to convert numeric-looking strings to numbers.
    Handles things like '12', '12.5', '$1,200', '45%' where possible.
    """
    out = X.copy()
    for c in out.columns:
        if pd.api.types.is_object_dtype(out[c]) or pd.api.types.is_string_dtype(out[c]):
            # first try boolish conversion
            tmp = _to_boolish_int(out[c])
            if not tmp.equals(out[c]):
                out[c] = tmp
                continue

            # strip common symbols for numeric parsing
            cleaned = (
                out[c]
                .astype(str)
                .str.replace(r"[\$,]", "", regex=True)
                .str.replace("%", "", regex=False)
                .str.strip()
            )
            numeric = pd.to_numeric(cleaned, errors="coerce")
            # if enough values convert successfully, keep numeric
            if numeric.notna().mean() >= 0.70:
                out[c] = numeric
    return out


# -------------------------------
# 2) Auto-create RTS/RTP targets
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
    If rts/rtp do not exist, create them using best-effort heuristics.
    This is intentionally generic so it can work on many datasets.

    Rules (best effort):
      RTS: "cleared / ready to return" proxy from rehab readiness signals
      RTP: "back to performance" proxy stricter than RTS

    If no good signals exist, falls back to a numeric composite score.
    """
    out = df.copy()
    cols = out.columns.tolist()

    if rts_col in cols and rtp_col in cols:
        return out

    # Make a cleaned view for building labels
    cleaned = coerce_numeric_cols(coerce_datetime_cols(out))

    # Candidate columns by common names
    strength = _find_col(cols, [r"strength", r"quad", r"hamstring", r"power"])
    balance  = _find_col(cols, [r"balance", r"stability", r"rom", r"mobility", r"hop"])
    perf     = _find_col(cols, [r"performance", r"score", r"rating", r"speed"])
    recovery = _find_col(cols, [r"recovery", r"rehab", r"days", r"duration", r"time"])
    risk     = _find_col(cols, [r"risk", r"reinjury", r"injury_risk", r"prob"])

    # Helper to normalize numeric column into [0,1]
    def norm(colname: str) -> pd.Series:
        s = pd.to_numeric(cleaned[colname], errors="coerce")
        lo, hi = np.nanpercentile(s, 5), np.nanpercentile(s, 95)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return pd.Series(np.nan, index=cleaned.index)
        return ((s - lo) / (hi - lo)).clip(0, 1)

    # Build a composite readiness score
    parts = []
    if strength: parts.append(norm(strength))
    if balance:  parts.append(norm(balance))
    if perf:     parts.append(norm(perf))

    # recovery: lower is better, so invert
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
            # absolute fallback: random labels (won't be great but prevents crash)
            rng = np.random.default_rng(42)
            if rts_col not in cols:
                out[rts_col] = rng.integers(0, 2, size=len(out))
            if rtp_col not in cols:
                out[rtp_col] = out[rts_col].copy()
            return out
        score = cleaned[num_cols].mean(axis=1)

    # Thresholds: RTS easier, RTP stricter
    # Use quantiles so it adapts to dataset scale.
    if rts_col not in cols:
        thr_rts = np.nanquantile(score, 0.55)
        out[rts_col] = (score >= thr_rts).astype(int)

    if rtp_col not in cols:
        thr_rtp = np.nanquantile(score, 0.75)
        out[rtp_col] = (score >= thr_rtp).astype(int)

    return out


# -------------------------------
# 3) Preprocessor builder
# -------------------------------
def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    # Convert datetime and numeric-like strings first
    X = coerce_numeric_cols(coerce_datetime_cols(X))

    cat_cols = [c for c in X.columns if (X[c].dtype == "object" or pd.api.types.is_categorical_dtype(X[c]))]
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
# 4) Split helper
# -------------------------------
def split_xy(
    df: pd.DataFrame,
    target: str,
    drop_cols: list[str] | None = None,
    test_size: float = 0.2,
    seed: int = 42,
):
    drop_cols = drop_cols or []

    # Ensure target is numeric 0/1 if possible
    y_raw = _to_boolish_int(df[target]) if target in df.columns else df[target]
    y = pd.to_numeric(y_raw, errors="coerce").fillna(0).astype(int)

    X = df.drop(columns=[target] + drop_cols, errors="ignore")
    X = coerce_numeric_cols(coerce_datetime_cols(X))

    # Avoid stratify crash if only 1 class
    strat = y if y.nunique() > 1 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=strat
    )
    return X_train, X_test, y_train, y_test
