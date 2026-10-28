"""Training routines for the Random Forest model used in the app.

This module contains the training logic, evaluation metrics, feature importance,
per-tree spread calculations, and cache-aware training workflow.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from config import (
    MATERIAL_TYPE,
    RANDOM_SEED,
    REQUIRED_COLUMNS,
)


@st.cache_resource
def train_and_cache_model(
    data: pd.DataFrame,
    n_estimators: int = 300,
    max_depth: int | None = None,
    min_samples_leaf: int = 1,
):
    """Train and cache a Random Forest model on the provided dataframe.

    The cache key is derived from the data itself plus hyperparameters, which is
    enough for this academic project and avoids stale models after data updates.
    """
    return fit_random_forest_model(
        data=data,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
    )


def fit_random_forest_model(
    data: pd.DataFrame,
    n_estimators: int = 300,
    max_depth: int | None = None,
    min_samples_leaf: int = 1,
) -> dict[str, Any]:
    """Fit the full preprocessing + Random Forest pipeline and return metrics and model."""
    if data.empty:
        raise ValueError("No data available to train the Random Forest model.")

    X = data.drop(columns=["tensile_strength"]).copy()
    y = data["tensile_strength"].astype(float).copy()

    numeric_features = [
        "layer_height",
        "nozzle_temperature",
        "print_speed",
        "infill_density",
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                "passthrough",
                numeric_features,
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                [MATERIAL_TYPE],
            ),
        ],
        remainder="drop",
    )

    rf_model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=RANDOM_SEED,
        oob_score=True,
        n_jobs=-1,
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("rf", rf_model),
    ])

    pipeline.fit(X, y)

    train_pred = pipeline.predict(X)
    train_r2 = r2_score(y, train_pred)
    train_mae = mean_absolute_error(y, train_pred)
    train_rmse = float(np.sqrt(mean_squared_error(y, train_pred)))

    X_train = X.sample(frac=0.8, random_state=RANDOM_SEED)
    X_test = X.drop(index=X_train.index)
    y_train = y.loc[X_train.index]
    y_test = y.loc[X_test.index]

    pipeline_split = Pipeline([
        ("preprocessor", preprocessor),
        ("rf", RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=RANDOM_SEED,
            oob_score=True,
            n_jobs=-1,
        )),
    ])
    pipeline_split.fit(X_train, y_train)
    test_pred = pipeline_split.predict(X_test)

    test_r2 = r2_score(y_test, test_pred)
    test_mae = mean_absolute_error(y_test, test_pred)
    test_rmse = float(np.sqrt(mean_squared_error(y_test, test_pred)))

    cv = KFold(n_splits=min(5, len(data)), shuffle=True, random_state=RANDOM_SEED)
    cv_pred = cross_val_predict(
        estimator=Pipeline([
            ("preprocessor", preprocessor),
            ("rf", RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                random_state=RANDOM_SEED,
                oob_score=True,
                n_jobs=-1,
            )),
        ]),
        X=X,
        y=y,
        cv=cv,
        n_jobs=-1,
    )

    cv_r2 = r2_score(y, cv_pred)
    cv_mae = mean_absolute_error(y, cv_pred)
    cv_rmse = float(np.sqrt(mean_squared_error(y, cv_pred)))

    # OOB score is not available for all RF settings, but we keep it as a metric
    # when the model reports it.
    oob_r2 = None
    if hasattr(pipeline.named_steps["rf"], "oob_score_"):
        oob_r2 = getattr(pipeline.named_steps["rf"], "oob_score_", None)

    baseline_pred = np.full(len(y), y.mean())
    baseline_mae = mean_absolute_error(y, baseline_pred)

    feature_importance = pipeline.named_steps["rf"].feature_importances_
    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    feature_df = pd.DataFrame({
        "feature": feature_names,
        "importance": feature_importance,
    })

    feature_df["group"] = feature_df["feature"].str.replace("num__", "").str.replace("cat__", "")
    feature_df["feature_name"] = feature_df["feature"].apply(lambda x: x.split("__")[-1])

    importance_summary = (
        feature_df.groupby("feature_name", as_index=False)["importance"].sum().sort_values("importance", ascending=False)
    )

    result = {
        "model": pipeline,
        "train_r2": train_r2,
        "train_mae": train_mae,
        "train_rmse": train_rmse,
        "test_r2": test_r2,
        "test_mae": test_mae,
        "test_rmse": test_rmse,
        "cv_r2": cv_r2,
        "cv_mae": cv_mae,
        "cv_rmse": cv_rmse,
        "oob_r2": oob_r2,
        "baseline_mae": baseline_mae,
        "importance": importance_summary,
        "feature_names": feature_names,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }
    return result


def predict_with_model(model: Any, params_df: pd.DataFrame) -> pd.DataFrame:
    """Predict tensile strength and return the tree spread as a DataFrame."""
    predictions = model.predict(params_df)
    tree_predictions = []
    for row in params_df.to_dict(orient="records"):
        transformed = model.named_steps["preprocessor"].transform(pd.DataFrame([row]))
        per_tree = np.array([
            tree.predict(transformed) for tree in model.named_steps["rf"].estimators_
        ]).ravel()
        tree_predictions.append(float(per_tree.std()))

    result = pd.DataFrame({
        "predicted_tensile_strength": predictions,
        "spread_across_trees": tree_predictions,
    })
    return result


def model_overfitting_warning(train_r2: float | None, test_r2: float | None, rows: int) -> str | None:
    """Return a warning message when the training/test gap is large or data is sparse."""
    if train_r2 is None or test_r2 is None:
        return None
    gap = train_r2 - test_r2
    if gap > 0.2 or rows < 30:
        return (
            "The model shows a large train-vs-test gap or a small dataset. "
            "This suggests the apparent performance may be optimistic; treat the metric as a rough estimate only."
        )
    return None


def get_importance_chart_data(importance_df: pd.DataFrame) -> pd.DataFrame:
    """Prepare feature-importance data for a Plotly bar chart."""
    return importance_df.sort_values("importance", ascending=False).reset_index(drop=True)
