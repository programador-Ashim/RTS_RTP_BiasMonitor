from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass
class SynthConfig:
    n: int = 2000
    seed: int = 42

class SyntheticReadinessGenerator:
    """Synthetic generator tailored to the RTS/RTP (ACL rehab) project.

    Outputs:
      - Features (injury + recovery + performance + athlete info)
      - Protected attributes for fairness monitoring (age, gender, race, age_group)
      - Two binary targets:
          RTS = Return to Sport (medical clearance / safety)
          RTP = Return to Performance (sustained performance after return)
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def make(self, n: int = 2000) -> pd.DataFrame:
        rng = self.rng

        # Protected attributes (fairness monitoring)
        gender = rng.choice(["Female", "Male"], size=n, p=[0.45, 0.55])
        race = rng.choice(
            ["AfricanAmerican", "Caucasian", "Hispanic", "Asian", "Other", "Unknown"],
            size=n,
            p=[0.18, 0.42, 0.18, 0.08, 0.10, 0.04],
        )
        age = np.clip(rng.normal(26, 6.5, size=n), 16, 45)  # athlete-like distribution

        # --- Core athlete features (matches your RTS/RTP description) ---
        training_intensity = np.clip(rng.normal(6.5, 1.8, size=n), 1, 10)      # 1..10
        training_hours = np.clip(rng.normal(10, 4.5, size=n), 0, 25)           # hrs/week
        match_count = np.clip(rng.normal(18, 8, size=n), 0, 45).round().astype(int)

        fatigue_score = np.clip(rng.normal(5.2, 2.2, size=n), 0, 10)           # 0..10
        recovery_days = np.clip(rng.normal(9, 6, size=n), 0, 45).round().astype(int)

        performance_score = np.clip(rng.normal(70, 12, size=n), 25, 98)        # 0..100-ish
        load_balance_score = np.clip(rng.normal(6.0, 2.0, size=n), 0, 10)      # 0..10
        acl_risk_score = np.clip(rng.normal(4.6, 2.2, size=n), 0, 10)          # 0..10

        injury_indicator = (rng.random(n) < np.clip((acl_risk_score / 12), 0.05, 0.65)).astype(int)
        team_contribution = np.clip(rng.normal(6.2, 2.0, size=n), 0, 10)       # 0..10

        # --- Intentionally inject a small bias (demo) ---
        # (this helps your fairness dashboard show meaningful DP/EO signals)
        age_bias = (age - 26) * -0.02  # older athletes slightly penalized
        gender_bias = np.where(gender == "Female", -0.08, 0.0)
        race_bias = np.where(np.isin(race, ["AfricanAmerican", "Hispanic"]), -0.15, 0.0)

        # -------------------------
        # Target 1: RTS (clearance/safety)
        # More influenced by injury risk, fatigue, and recovery days.
        # -------------------------
        rts_score = (
            (10 - acl_risk_score) * 0.22
            + (10 - fatigue_score) * 0.18
            + np.clip(recovery_days / 30, 0, 1) * 0.14
            + load_balance_score * 0.12
            + training_intensity * 0.10
            + np.clip(training_hours / 25, 0, 1) * 0.06
            + (10 - injury_indicator * 10) * 0.10
            + age_bias + gender_bias + race_bias
        )

        # Threshold chosen to create both classes
        y_rts = (rts_score >= 5.7).astype(int)

        # -------------------------
        # Target 2: RTP (sustained performance)
        # More influenced by performance_score, training load balance, and team contribution.
        # RTP is typically harder than RTS (so we use a stricter threshold).
        # -------------------------
        rtp_score = (
            (performance_score / 100) * 0.28
            + load_balance_score * 0.16
            + team_contribution * 0.12
            + (10 - fatigue_score) * 0.10
            + np.clip(training_hours / 25, 0, 1) * 0.10
            + training_intensity * 0.08
            + (10 - acl_risk_score) * 0.10
            + np.clip(match_count / 45, 0, 1) * 0.06
            + np.clip(recovery_days / 30, 0, 1) * 0.06
            + age_bias * 0.6 + gender_bias * 0.6 + race_bias * 0.6
        )

        y_rtp = (rtp_score >= 5.9).astype(int)

        # Small label noise (keeps it realistic)
        flip_rts = rng.random(n) < 0.05
        flip_rtp = rng.random(n) < 0.06
        y_rts = np.where(flip_rts, 1 - y_rts, y_rts)
        y_rtp = np.where(flip_rtp, 1 - y_rtp, y_rtp)

        df = pd.DataFrame({
            "age": age,
            "gender": gender,
            "race": race,
            "training_intensity": training_intensity,
            "training_hours_per_week": training_hours,
            "match_count": match_count,
            "fatigue_score": fatigue_score,
            "recovery_days": recovery_days,
            "performance_score": performance_score,
            "load_balance_score": load_balance_score,
            "acl_risk_score": acl_risk_score,
            "injury_indicator": injury_indicator,
            "team_contribution": team_contribution,
            "rts": y_rts,
            "rtp": y_rtp,
        })

        # Helpful bins for fairness
        df["age_group"] = pd.cut(
            df["age"],
            bins=[0, 18, 22, 26, 30, 35, 120],
            labels=["<=18", "19-22", "23-26", "27-30", "31-35", "36+"],
            include_lowest=True
        ).astype(str)

        return df

def load_any(uploaded) -> pd.DataFrame:
    """Read CSV or Excel from Streamlit uploader."""
    name = (uploaded.name or "").lower()
    if name.endswith(".csv"):
        try:
            return pd.read_csv(uploaded)
        except UnicodeDecodeError:
            return pd.read_csv(uploaded, encoding="latin-1")
    if name.endswith(".xlsx") or name.endswith(".xls"):
        try:
            return pd.read_excel(uploaded)
        except ImportError as e:
            raise ImportError("Reading Excel requires 'openpyxl'. Install with: pip install openpyxl") from e
    raise ValueError("Unsupported file type. Upload a .csv or .xlsx/.xls")
