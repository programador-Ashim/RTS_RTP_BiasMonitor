from __future__ import annotations
from dataclasses import dataclass
import joblib
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from imblearn.over_sampling import SMOTE

from .preprocess import build_preprocessor

@dataclass
class TrainedBundle:
    model_name: str
    pipeline: Pipeline
    feature_cols: list[str]
    target: str


def _prepare_training_data(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    use_smote: bool = False,
    seed: int = 42,
):
    """
    1. Fit preprocessor on raw training data
    2. Transform to numeric matrix
    3. Optionally apply SMOTE on the transformed training data only
    """
    pre = build_preprocessor(X_train)
    X_train_trans = pre.fit_transform(X_train)

    # If sparse, convert to dense because SMOTE works best on dense arrays
    if hasattr(X_train_trans, "toarray"):
        X_train_trans = X_train_trans.toarray()

    y_train = pd.Series(y_train).astype(int)

    if use_smote:
        class_counts = y_train.value_counts()
        minority_count = int(class_counts.min())

        # SMOTE needs at least 2 minority samples
        if minority_count >= 2:
            k_neighbors = min(5, minority_count - 1)
            smote = SMOTE(random_state=seed, k_neighbors=k_neighbors)
            X_train_trans, y_train = smote.fit_resample(X_train_trans, y_train)

    return pre, X_train_trans, y_train


def train_gradient_boosting(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 250,
    learning_rate: float = 0.05,
    max_depth: int = 3,
    seed: int = 42,
    use_smote: bool = False,
) -> Pipeline:
    """
    Train a Gradient Boosting model wrapped in a preprocessing pipeline.

    Steps:
    1. Fit the preprocessing pipeline on raw training data
    2. Transform the data into numeric model-ready form
    3. Optionally apply SMOTE on the transformed training matrix
    4. Fit the Gradient Boosting classifier
    5. Return a sklearn Pipeline containing both preprocessing and model

    Why this matters:
    Keeps preprocessing and modeling consistent between training,
    evaluation, and deployment.
    """
    pre, X_train_trans, y_train_bal = _prepare_training_data(
        X_train, y_train, use_smote=use_smote, seed=seed
    )

    gb = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=seed,
    )

    gb.fit(X_train_trans, y_train_bal)

    pipe = Pipeline([
        ("pre", pre),
        ("clf", gb),
    ])
    return pipe


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 300,
    max_depth: int | None = None,
    min_samples_leaf: int = 2,
    seed: int = 42,
    use_smote: bool = False,
) -> Pipeline:
    """
    Train a Random Forest model wrapped in a preprocessing pipeline.

    Steps:
    1. Fit the preprocessing pipeline on raw training data
    2. Transform the data into numeric model-ready form
    3. Optionally apply SMOTE on the transformed training matrix
    4. Fit the Random Forest classifier
    5. Return a sklearn Pipeline containing both preprocessing and model

    Why this matters:
    Provides a benchmark model that can be compared against
    Gradient Boosting under the same preprocessing workflow.
    """
    pre, X_train_trans, y_train_bal = _prepare_training_data(
        X_train, y_train, use_smote=use_smote, seed=seed
    )

    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )

    rf.fit(X_train_trans, y_train_bal)

    pipe = Pipeline([
        ("pre", pre),
        ("clf", rf),
    ])
    return pipe


def evaluate(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:

    """
    Evaluate a trained RTS or RTP model on unseen test data.

    Returns:
    - Accuracy
    - Precision
    - Recall
    - F1-score
    - ROC-AUC
    - Specificity
    - TP, TN, FP, FN
    - Test set size
    - Positive class rate

    Why this matters:
    These values support the company requirement for stronger validation,
    confusion-matrix analysis, and clearer reporting of model behavior.
    """
    y_test = pd.Series(y_test).astype(int)
    y_pred = pipeline.predict(X_test)

    if hasattr(pipeline, "predict_proba"):
        y_proba = pipeline.predict_proba(X_test)[:, 1]
    else:
        y_proba = y_pred.astype(float)

    try:
        roc = float(roc_auc_score(y_test, y_proba))
    except Exception:
        roc = 0.0

    # Compute confusion-matrix counts explicitly so the dashboard/report
    # can show TP, TN, FP, FN and specificity in addition to aggregate metrics.
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "Accuracy": float(accuracy_score(y_test, y_pred)),
        "Precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "Roc_Auc": roc,
        "Specificity": float(specificity),
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "Test_Size": int(len(y_test)),
        "Positive_Rate": float(y_test.mean()),
    }



def cross_validate_model(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str = "rf",
    cv: int = 5,
    seed: int = 42,
) -> dict:
    """
    Run stratified K-fold cross-validation using a standard sklearn pipeline.

    Parameters:
    - X: feature matrix
    - y: binary target
    - model_name: "rf" or "gb"
    - cv: number of folds
    - seed: random seed for reproducibility

    Returns:
    - mean F1-score across folds
    - standard deviation of F1-score
    - raw fold scores

    Why this matters:
    This checks whether model performance is stable across multiple
    train/validation splits instead of depending on one split only.
    """
    y = pd.Series(y).astype(int)

    if y.nunique() < 2:
        return {
            "cv_f1_mean": 0.0,
            "cv_f1_std": 0.0,
            "cv_scores": [],
        }

    pre = build_preprocessor(X)

    if model_name == "gb":
        clf = GradientBoostingClassifier(
            n_estimators=250,
            learning_rate=0.05,
            max_depth=3,
            random_state=seed,
        )
    else:
        clf = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )

    model = Pipeline([
        ("pre", pre),
        ("clf", clf),
    ])
    
    # Stratification helps preserve class balance in each fold,
    # which is important for reliable validation on imbalanced targets.
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)

    try:
        scores = cross_val_score(model, X, y, cv=skf, scoring="f1")
    except Exception:
        return {
            "cv_f1_mean": 0.0,
            "cv_f1_std": 0.0,
            "cv_scores": [],
        }

    return {
        "cv_f1_mean": float(np.mean(scores)),
        "cv_f1_std": float(np.std(scores)),
        "cv_scores": [float(s) for s in scores],
    }