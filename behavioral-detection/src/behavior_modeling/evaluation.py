"""Evaluate candidate classifiers and select the final behavior model."""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_mapping: dict[int, str],
) -> dict[str, Any]:
    """Calculate aggregate and named per-class classification metrics.

    Args:
        y_true: Ground-truth class identifiers.
        y_pred: Predicted class identifiers.
        class_mapping: Ordered mapping from class ID to class name.

    Returns:
        Aggregate metrics and a nested per-class metric mapping. Undefined
        precision or recall values are reported as zero.
    """
    class_ids = list(class_mapping)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=class_ids,
        zero_division=0,
    )
    per_class = {
        class_mapping[class_id]: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, class_id in enumerate(class_ids)
    }
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "per_class": per_class,
    }


def predict_labels_and_probabilities(
    model: ClassifierMixin,
    features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray | None, float]:
    """Run one timed prediction pass and return optional probabilities.

    Args:
        model: Fitted scikit-learn-compatible classifier.
        features: Feature matrix to predict.

    Returns:
        Predicted labels, optional class-probability matrix, and elapsed seconds.
        Probability-to-class mapping must be read from ``model.classes_`` by the
        caller rather than assumed from column position.
    """
    start = time.perf_counter()
    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(features))
        model_classes = np.asarray(model.classes_)
        predictions = model_classes[np.argmax(probabilities, axis=1)]
    else:
        probabilities = None
        predictions = np.asarray(model.predict(features))
    return predictions, probabilities, time.perf_counter() - start


def model_complexity_score(model_name: str, params: Mapping[str, Any]) -> float:
    """Calculate the existing within-experiment relative complexity score.

    Args:
        model_name: Supported candidate model name.
        params: Resolved estimator hyperparameters.

    Returns:
        Relative score used only as a late near-tie selection criterion. Scores
        are not meaningful outside this candidate set.
    """
    depth = params.get("max_depth")
    effective_depth = 32 if depth in (None, -1) else max(1, int(depth))
    estimators = int(params.get("n_estimators", params.get("max_iter", 1)))
    leaves = int(
        params.get(
            "num_leaves",
            params.get("max_leaf_nodes", params.get("max_leaves", 2) or 64),
        )
    )
    min_leaf = max(
        1,
        int(params.get("min_samples_leaf", params.get("min_child_samples", 1))),
    )

    if model_name == "Decision Tree":
        return float(effective_depth / min_leaf)
    if model_name in {"Random Forest", "Extra Trees", "Gradient Boosting"}:
        return float(estimators * effective_depth / min_leaf)
    if model_name in {"HistGradientBoosting", "LightGBM"}:
        return float(estimators * leaves / min_leaf)
    if model_name == "XGBoost":
        return float(2000 * effective_depth * max(1, leaves))
    return math.inf


def build_model_comparison(
    tuning_summary: pd.DataFrame,
    stability_summary: pd.DataFrame,
    best_params: Mapping[str, Mapping[str, Any]],
) -> pd.DataFrame:
    """Combine tuning and seed-stability results for final model selection.

    Args:
        tuning_summary: One row per model from Optuna tuning.
        stability_summary: Aggregated multi-seed validation results.
        best_params: Resolved best parameters for successful models.

    Returns:
        Model comparison sorted by stable validation performance, with relative
        complexity and concise reliability notes.
    """
    comparison = tuning_summary.merge(stability_summary, on="model_name", how="left")
    comparison["complexity_score"] = comparison["model_name"].map(
        lambda name: (
            model_complexity_score(name, best_params[name])
            if name in best_params
            else math.nan
        )
    )

    def reliability_note(row: pd.Series) -> str:
        """Summarize the three reliability diagnostics used in reporting."""
        if pd.isna(row.get("mean_validation_macro_f1")):
            return str(row.get("notes", "No successful stability result."))
        gap = float(row["mean_generalization_gap"])
        gap_note = "large gap" if gap > 0.08 else "moderate gap" if gap > 0.04 else "controlled gap"
        seed_note = (
            "seed-sensitive"
            if float(row["validation_macro_f1_std"]) > 0.01
            else "seed-stable"
        )
        class_note = (
            "uneven class F1"
            if float(row["validation_per_class_f1_range"]) > 0.15
            else "consistent class F1"
        )
        return f"{gap_note}; {seed_note}; {class_note}"

    comparison["reliability_note"] = comparison.apply(reliability_note, axis=1)
    return comparison.sort_values(
        [
            "mean_validation_macro_f1",
            "validation_macro_f1_std",
            "mean_generalization_gap",
            "complexity_score",
            "mean_validation_inference_time_seconds",
        ],
        ascending=[False, True, True, True, True],
        na_position="last",
    ).reset_index(drop=True)


def select_final_model(
    comparison: pd.DataFrame,
    best_params: Mapping[str, Mapping[str, Any]],
    seed_results: pd.DataFrame,
    near_tie_threshold: float,
) -> dict[str, Any]:
    """Select the final candidate and a representative random seed.

    Args:
        comparison: Combined tuning and stability comparison.
        best_params: Resolved best parameters by model name.
        seed_results: Individual multi-seed validation runs.
        near_tie_threshold: Maximum macro-F1 distance from the best mean score
            for a model to enter the near-tie candidate set.

    Returns:
        Selected model details and the ranked near-tie candidate table.

    Raises:
        RuntimeError: If no model completed stability evaluation.

    Notes:
        The representative seed is closest to the model's mean validation score,
        preventing selection of an unusually lucky seed.
    """
    eligible = comparison.dropna(subset=["mean_validation_macro_f1"]).copy()
    if eligible.empty:
        raise RuntimeError("No model completed seed-stability verification.")

    best_mean = float(eligible["mean_validation_macro_f1"].max())
    near_ties = eligible[
        eligible["mean_validation_macro_f1"] >= best_mean - near_tie_threshold
    ].copy()
    near_ties["absolute_generalization_gap"] = near_ties[
        "mean_generalization_gap"
    ].abs()
    near_ties = near_ties.sort_values(
        [
            "validation_macro_f1_std",
            "absolute_generalization_gap",
            "complexity_score",
            "mean_validation_inference_time_seconds",
            "mean_validation_macro_f1",
        ],
        ascending=[True, True, True, True, False],
    ).reset_index(drop=True)

    selected_row = near_ties.iloc[0]
    model_name = str(selected_row["model_name"])
    model_seed_runs = seed_results.query(
        "model_name == @model_name and status == 'COMPLETE'"
    ).copy()
    mean_score = float(model_seed_runs["validation_macro_f1"].mean())
    model_seed_runs["distance_from_mean"] = (
        model_seed_runs["validation_macro_f1"] - mean_score
    ).abs()
    representative = model_seed_runs.sort_values(
        ["distance_from_mean", "seed"]
    ).iloc[0]
    iteration_value = representative.get("best_iteration")
    iteration_count = (
        None if iteration_value is None or pd.isna(iteration_value) else int(iteration_value)
    )

    return {
        "model_name": model_name,
        "hyperparameters": dict(best_params[model_name]),
        "seed": int(representative["seed"]),
        "iteration_count": iteration_count,
        "selected_row": selected_row,
        "near_tie_candidates": near_ties,
    }


def compute_permutation_importance(
    model: ClassifierMixin,
    features: np.ndarray,
    targets: np.ndarray,
    feature_names: list[str],
    random_seed: int,
    repeats: int = 10,
) -> pd.DataFrame:
    """Measure selected-model feature importance on validation data.

    Args:
        model: Selected development model fitted only on training data.
        features: Validation feature matrix.
        targets: Validation targets.
        feature_names: Ordered feature names matching the matrix columns.
        random_seed: Seed controlling permutation order.
        repeats: Number of permutations per feature.

    Returns:
        Feature importance values sorted from most to least influential.

    Notes:
        This validation-only analysis occurs before test prediction and does not
        change model ranking. Predictive importance is not causal evidence.
    """
    result = permutation_importance(
        model,
        features,
        targets,
        scoring="f1_macro",
        n_repeats=repeats,
        n_jobs=-1,
        random_state=random_seed,
    )
    return pd.DataFrame(
        {
            "feature_name": feature_names,
            "permutation_importance_mean": result.importances_mean,
            "permutation_importance_std": result.importances_std,
        }
    ).sort_values("permutation_importance_mean", ascending=False, ignore_index=True)


def evaluate_final_test_set(
    model: ClassifierMixin,
    features: np.ndarray,
    targets: np.ndarray,
    class_mapping: dict[int, str],
    probability_order: list[str],
) -> dict[str, Any]:
    """Evaluate the locked final model in one test prediction pass.

    Args:
        model: Final model already refitted on train plus validation.
        features: Untouched test feature matrix.
        targets: Untouched test targets.
        class_mapping: Ordered mapping from class ID to class name.
        probability_order: Stable output order for probability columns.

    Returns:
        Aggregate summary, per-class table, CSV-ready metric table, confusion
        matrix, and row-level predictions.

    Notes:
        The caller must invoke this function only after model selection and final
        refit. Results must not be used to reselect or retune the model.
    """
    predictions, probabilities, inference_time = predict_labels_and_probabilities(
        model, features
    )
    metrics = classification_metrics(targets, predictions, class_mapping)
    class_ids = list(class_mapping)
    confusion = pd.DataFrame(
        confusion_matrix(targets, predictions, labels=class_ids),
        index=[f"actual_{class_mapping[class_id]}" for class_id in class_ids],
        columns=[f"predicted_{class_mapping[class_id]}" for class_id in class_ids],
    )
    per_class = (
        pd.DataFrame(metrics["per_class"])
        .transpose()
        .reset_index(names="class_name")
    )
    per_class["support"] = per_class["support"].astype(int)

    summary = {
        "accuracy": metrics["accuracy"],
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "inference_time_seconds": inference_time,
        "per_class": metrics["per_class"],
    }
    metric_rows = [{"scope": "overall", **{key: value for key, value in summary.items() if key != "per_class"}}]
    metric_rows.extend(
        {"scope": row["class_name"], **row.drop(labels="class_name").to_dict()}
        for _, row in per_class.iterrows()
    )
    metrics_table = pd.DataFrame(metric_rows)

    predictions_table = pd.DataFrame(
        {
            "true_class": [class_mapping[int(value)] for value in targets],
            "predicted_class": [class_mapping[int(value)] for value in predictions],
        }
    )
    for class_name in probability_order:
        predictions_table[f"predicted_probability_{class_name}"] = np.nan

    if probabilities is not None:
        # Estimator probability columns follow model.classes_, not display order.
        for probability_index, class_id in enumerate(model.classes_):
            class_name = class_mapping[int(class_id)]
            predictions_table[f"predicted_probability_{class_name}"] = probabilities[
                :, probability_index
            ]

    return {
        "summary": summary,
        "per_class": per_class,
        "metrics_table": metrics_table,
        "confusion_matrix": confusion,
        "predictions": predictions_table,
    }
