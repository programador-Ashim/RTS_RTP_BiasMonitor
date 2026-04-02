from __future__ import annotations
from dataclasses import dataclass
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from .preprocess import build_preprocessor

@dataclass
class TrainedBundle:
    model_name: str
    pipeline: Pipeline
    feature_cols: list[str]
    target: str

def train_gradient_boosting(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 250,
    learning_rate: float = 0.05,
    max_depth: int = 3,
    seed: int = 42,
) -> Pipeline:
    """Gradient Boosting (primary model for the ACL rehab proposal style)."""
    pre = build_preprocessor(X_train)
    gb = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=seed,
    )
    pipe = Pipeline([("pre", pre), ("clf", gb)])
    pipe.fit(X_train, y_train)
    return pipe

def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 300,
    max_depth: int | None = None,
    min_samples_leaf: int = 2,
    seed: int = 42,
) -> Pipeline:
    pre = build_preprocessor(X_train)
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )
    pipe = Pipeline([("pre", pre), ("clf", rf)])
    pipe.fit(X_train, y_train)
    return pipe

def evaluate(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = pipeline.predict(X_test)
    if hasattr(pipeline, "predict_proba"):
        y_proba = pipeline.predict_proba(X_test)[:, 1]
    else:
        y_proba = y_pred.astype(float)

    return {
        "Accuracy": float(accuracy_score(y_test, y_pred)),
        "Precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "Roc_Auc": float(roc_auc_score(y_test, y_proba)),
    }

def save_bundle(bundle: TrainedBundle, path: str):
    joblib.dump(bundle, path)

def load_bundle(path: str) -> TrainedBundle:
    return joblib.load(path)
