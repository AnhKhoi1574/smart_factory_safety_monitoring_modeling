"""Build, tune, stabilize, and refit tree-based behavior classifiers."""

from __future__ import annotations

import gc
import json
import time
import warnings
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import optuna
import pandas as pd
from optuna.trial import TrialState
from sklearn.base import ClassifierMixin
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.tree import DecisionTreeClassifier
from tqdm.auto import tqdm

from .evaluation import classification_metrics, predict_labels_and_probabilities

try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
    XGBOOST_IMPORT_ERROR = ""
except Exception as exc:  # Optional dependency failures should skip one candidate.
    XGBClassifier = None
    XGBOOST_AVAILABLE = False
    XGBOOST_IMPORT_ERROR = str(exc)

try:
    import lightgbm as lgb
    from lightgbm import LGBMClassifier

    LIGHTGBM_AVAILABLE = True
    LIGHTGBM_IMPORT_ERROR = ""
except Exception as exc:  # Optional dependency failures should skip one candidate.
    lgb = None
    LGBMClassifier = None
    LIGHTGBM_AVAILABLE = False
    LIGHTGBM_IMPORT_ERROR = str(exc)

MODEL_NAMES = (
    "Decision Tree",
    "Random Forest",
    "Extra Trees",
    "Gradient Boosting",
    "HistGradientBoosting",
    "XGBoost",
    "LightGBM",
)


def get_model_availability() -> pd.DataFrame:
    """Report which candidate model libraries are available.

    Returns:
        Two-column candidate status table plus a concise note for unavailable
        optional dependencies.
    """
    availability = {
        "Decision Tree": (True, ""),
        "Random Forest": (True, ""),
        "Extra Trees": (True, ""),
        "Gradient Boosting": (True, ""),
        "HistGradientBoosting": (True, ""),
        "XGBoost": (XGBOOST_AVAILABLE, XGBOOST_IMPORT_ERROR),
        "LightGBM": (LIGHTGBM_AVAILABLE, LIGHTGBM_IMPORT_ERROR),
    }
    return pd.DataFrame(
        [
            {
                "model_name": model_name,
                "available": available,
                "note": "" if available else error,
            }
            for model_name, (available, error) in availability.items()
        ]
    )


def build_model(
    model_name: str,
    params: Mapping[str, Any],
    seed: int,
    *,
    early_stopping_rounds: int = 50,
    final_training: bool = False,
    fixed_iterations: int | None = None,
) -> ClassifierMixin:
    """Construct one supported classifier from resolved parameters.

    Args:
        model_name: Candidate name from ``MODEL_NAMES``.
        params: Resolved estimator parameters from the Optuna search space.
        seed: Estimator random seed.
        early_stopping_rounds: Validation rounds without improvement for
            supported boosting models.
        final_training: Disable held-out early stopping for train-plus-validation
            refitting when ``True``.
        fixed_iterations: Validation-derived boosting iteration count for final
            refitting.

    Returns:
        Unfitted scikit-learn-compatible classifier.

    Raises:
        ImportError: If the requested optional model library is unavailable.
        KeyError: If ``model_name`` is unsupported.
    """
    model_params = dict(params)
    if model_name == "Decision Tree":
        return DecisionTreeClassifier(random_state=seed, **model_params)
    if model_name == "Random Forest":
        return RandomForestClassifier(n_jobs=-1, random_state=seed, **model_params)
    if model_name == "Extra Trees":
        return ExtraTreesClassifier(n_jobs=-1, random_state=seed, **model_params)
    if model_name == "Gradient Boosting":
        return GradientBoostingClassifier(random_state=seed, **model_params)
    if model_name == "HistGradientBoosting":
        if final_training:
            model_params["max_iter"] = int(fixed_iterations or model_params["max_iter"])
            return HistGradientBoostingClassifier(
                early_stopping=False,
                random_state=seed,
                **model_params,
            )
        return HistGradientBoostingClassifier(
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=30,
            random_state=seed,
            **model_params,
        )
    if model_name == "XGBoost":
        if not XGBOOST_AVAILABLE or XGBClassifier is None:
            raise ImportError("Install XGBoost to enable this candidate.")
        constructor_params = {
            "objective": "multi:softprob",
            "eval_metric": "mlogloss",
            "n_estimators": int(fixed_iterations or 2000),
            "tree_method": "hist",
            "n_jobs": -1,
            "random_state": seed,
            **model_params,
        }
        if not final_training:
            constructor_params["early_stopping_rounds"] = early_stopping_rounds
        return XGBClassifier(**constructor_params)
    if model_name == "LightGBM":
        if not LIGHTGBM_AVAILABLE or LGBMClassifier is None:
            raise ImportError("Install LightGBM to enable this candidate.")
        return LGBMClassifier(
            objective="multiclass",
            n_estimators=int(fixed_iterations or 2000),
            n_jobs=-1,
            random_state=seed,
            verbosity=-1,
            subsample_freq=1,
            **model_params,
        )
    raise KeyError(f"Unknown model: {model_name}")


def fit_model(
    model_name: str,
    model: ClassifierMixin,
    features: np.ndarray,
    targets: np.ndarray,
    *,
    validation_features: np.ndarray | None = None,
    validation_targets: np.ndarray | None = None,
    early_stopping_rounds: int = 50,
) -> float:
    """Fit a model and return elapsed seconds.

    Args:
        model_name: Supported model name.
        model: Unfitted estimator.
        features: Training features.
        targets: Training targets.
        validation_features: Optional validation features used only for early
            stopping by XGBoost and LightGBM.
        validation_targets: Validation targets paired with validation features.
        early_stopping_rounds: LightGBM early-stopping patience.

    Returns:
        Training elapsed time in seconds.

    Notes:
        Test data must never be passed as either training or validation input.
    """
    start = time.perf_counter()
    if model_name == "XGBoost" and validation_features is not None:
        model.fit(
            features,
            targets,
            eval_set=[(validation_features, validation_targets)],
            verbose=False,
        )
    elif model_name == "LightGBM" and validation_features is not None:
        if lgb is None:
            raise ImportError("Install LightGBM to enable this candidate.")
        callbacks = [
            lgb.early_stopping(early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=0),
        ]
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="The argument 'eval_set' is deprecated.*",
            )
            model.fit(
                features,
                targets,
                eval_set=[(validation_features, validation_targets)],
                eval_metric="multi_logloss",
                callbacks=callbacks,
            )
    else:
        model.fit(features, targets)
    return time.perf_counter() - start


def effective_iteration_count(
    model_name: str,
    model: ClassifierMixin,
) -> int | None:
    """Read the effective boosting iteration count for an explicit final refit.

    Args:
        model_name: Supported model name.
        model: Fitted estimator.

    Returns:
        One-based/effective iteration count, or ``None`` for non-boosting models.
    """
    if model_name == "XGBoost":
        best_iteration = getattr(model, "best_iteration", None)
        return None if best_iteration is None else int(best_iteration) + 1
    if model_name == "LightGBM":
        best_iteration = getattr(model, "best_iteration_", None)
        return None if not best_iteration else int(best_iteration)
    if model_name == "HistGradientBoosting":
        iterations = getattr(model, "n_iter_", None)
        return None if iterations is None else int(iterations)
    return None


def _suggest_max_features(trial: optuna.Trial) -> str | float:
    """Resolve the conditional max-features search used by forest models."""
    mode = trial.suggest_categorical("max_features_mode", ["sqrt", "log2", "float"])
    if mode == "float":
        return trial.suggest_float("max_features_fraction", 0.3, 1.0)
    return mode


def suggest_model_params(
    model_name: str,
    trial: optuna.Trial,
) -> dict[str, Any]:
    """Sample the existing resolved estimator parameters for one Optuna trial.

    Args:
        model_name: Supported candidate name.
        trial: Active Optuna trial.

    Returns:
        Parameters ready to pass to ``build_model``.

    Raises:
        KeyError: If the model name is unsupported.
    """
    if model_name == "Decision Tree":
        return {
            "criterion": trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"]),
            "splitter": trial.suggest_categorical("splitter", ["best", "random"]),
            "max_depth": trial.suggest_int("max_depth", 2, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 30),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features": trial.suggest_categorical("max_features", [None, "sqrt", "log2"]),
            "ccp_alpha": trial.suggest_float("ccp_alpha", 1e-7, 1e-2, log=True),
        }
    if model_name == "Random Forest":
        bootstrap = trial.suggest_categorical("bootstrap", [True, False])
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
            "max_depth": trial.suggest_categorical("max_depth", [None, *range(4, 31)]),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 30),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features": _suggest_max_features(trial),
            "bootstrap": bootstrap,
            "max_samples": trial.suggest_float("max_samples", 0.6, 1.0) if bootstrap else None,
            "criterion": trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"]),
        }
    if model_name == "Extra Trees":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
            "max_depth": trial.suggest_categorical("max_depth", [None, *range(4, 31)]),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 30),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features": _suggest_max_features(trial),
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
            "criterion": trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"]),
        }
    if model_name == "Gradient Boosting":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500, step=25),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 1, 6),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 30),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "max_features": trial.suggest_categorical("max_features", [None, "sqrt", "log2"]),
        }
    if model_name == "HistGradientBoosting":
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "max_iter": trial.suggest_int("max_iter", 100, 600, step=50),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 7, 63),
            "max_depth": trial.suggest_categorical("max_depth", [None, *range(2, 13)]),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 60),
            "l2_regularization": trial.suggest_float("l2_regularization", 1e-6, 10.0, log=True),
            "max_bins": trial.suggest_int("max_bins", 64, 255),
        }
    if model_name == "XGBoost":
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-6, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 20.0, log=True),
            "max_leaves": trial.suggest_int("max_leaves", 0, 64),
        }
    if model_name == "LightGBM":
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 7, 63),
            "max_depth": trial.suggest_categorical("max_depth", [-1, *range(3, 13)]),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 80),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-6, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-6, 20.0, log=True),
            "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 5.0),
            "max_bin": trial.suggest_int("max_bin", 63, 255),
        }
    raise KeyError(f"Unknown model: {model_name}")


def _make_objective(
    model_name: str,
    train_features: np.ndarray,
    train_targets: np.ndarray,
    validation_features: np.ndarray,
    validation_targets: np.ndarray,
    class_mapping: dict[int, str],
    random_seed: int,
    early_stopping_rounds: int,
) -> Callable[[optuna.Trial], float]:
    """Create the validation-macro-F1 objective for one candidate model."""

    def objective(trial: optuna.Trial) -> float:
        """Fit one sampled configuration and return validation macro F1."""
        params = suggest_model_params(model_name, trial)
        trial.set_user_attr("resolved_hyperparameters", json.dumps(params, sort_keys=True))
        model = build_model(
            model_name,
            params,
            random_seed,
            early_stopping_rounds=early_stopping_rounds,
        )
        training_time = fit_model(
            model_name,
            model,
            train_features,
            train_targets,
            validation_features=validation_features,
            validation_targets=validation_targets,
            early_stopping_rounds=early_stopping_rounds,
        )
        train_predictions = np.asarray(model.predict(train_features))
        validation_predictions, _, inference_time = predict_labels_and_probabilities(
            model, validation_features
        )
        train_metrics = classification_metrics(train_targets, train_predictions, class_mapping)
        validation_metrics = classification_metrics(
            validation_targets, validation_predictions, class_mapping
        )
        attributes = {
            "training_macro_f1": train_metrics["macro_f1"],
            "validation_accuracy": validation_metrics["accuracy"],
            "validation_macro_precision": validation_metrics["macro_precision"],
            "validation_macro_recall": validation_metrics["macro_recall"],
            "validation_weighted_f1": validation_metrics["weighted_f1"],
            "generalization_gap": train_metrics["macro_f1"] - validation_metrics["macro_f1"],
            "training_time_seconds": training_time,
            "validation_inference_time_seconds": inference_time,
        }
        for key, value in attributes.items():
            trial.set_user_attr(key, float(value))
        iteration_count = effective_iteration_count(model_name, model)
        if iteration_count is not None:
            trial.set_user_attr("best_iteration", iteration_count)
        del model
        return float(validation_metrics["macro_f1"])

    return objective


def tune_models(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    validation_features: np.ndarray,
    validation_targets: np.ndarray,
    class_mapping: dict[int, str],
    *,
    n_trials: int,
    random_seed: int,
    early_stopping_rounds: int,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    """Tune all available candidate models using validation macro F1.

    Args:
        train_features: Fixed training feature matrix.
        train_targets: Fixed training targets.
        validation_features: Fixed validation feature matrix.
        validation_targets: Fixed validation targets.
        class_mapping: Ordered class ID-to-name mapping.
        n_trials: Equal Optuna budget for each available candidate.
        random_seed: Reproducible sampler and estimator seed.
        early_stopping_rounds: Patience for supported boosting models.

    Returns:
        Concise one-row-per-model tuning summary and resolved best parameters.

    Raises:
        ValueError: If ``n_trials`` is not positive.

    Notes:
        Studies are intentionally in memory. No Optuna database or trial-history
        artifact is created. A model-level failure does not stop later candidates.
    """
    if n_trials <= 0:
        raise ValueError("n_trials must be positive")

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    availability = get_model_availability().set_index("model_name")
    rows: list[dict[str, Any]] = []
    best_params: dict[str, dict[str, Any]] = {}

    for model_name in MODEL_NAMES:
        if not bool(availability.loc[model_name, "available"]):
            rows.append(
                {
                    "model_name": model_name,
                    "model_status": "SKIPPED",
                    "notes": str(availability.loc[model_name, "note"]),
                }
            )
            continue

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=optuna.exceptions.ExperimentalWarning,
            )
            sampler = optuna.samplers.TPESampler(
                seed=random_seed,
                n_startup_trials=15,
                multivariate=True,
            )
        study = optuna.create_study(direction="maximize", sampler=sampler)
        progress_bar = tqdm(
            total=n_trials,
            desc=f"Tuning {model_name}",
            unit="trial",
            dynamic_ncols=True,
            leave=True,
        )

        def update_progress_bar(
            current_study: optuna.Study,
            completed_trial: optuna.trial.FrozenTrial,
        ) -> None:
            """Show the current model and best completed validation score."""
            progress_bar.update(1)
            try:
                best_trial_so_far = current_study.best_trial
            except ValueError:
                progress_bar.set_postfix(last_status=completed_trial.state.name)
                return
            progress_bar.set_postfix(
                best_trial=best_trial_so_far.number,
                best_f1=f"{float(best_trial_so_far.value):.4f}",
                last_status=completed_trial.state.name,
            )

        try:
            try:
                study.optimize(
                    _make_objective(
                        model_name,
                        train_features,
                        train_targets,
                        validation_features,
                        validation_targets,
                        class_mapping,
                        random_seed,
                        early_stopping_rounds,
                    ),
                    n_trials=n_trials,
                    n_jobs=1,
                    catch=(Exception,),
                    # Release estimator memory once after each trial without
                    # duplicating garbage collection inside the objective.
                    gc_after_trial=True,
                    callbacks=[update_progress_bar],
                    show_progress_bar=False,
                )
            finally:
                progress_bar.close()
            complete_trials = [
                trial
                for trial in study.trials
                if trial.state == TrialState.COMPLETE and trial.value is not None
            ]
            failed_trials = sum(trial.state == TrialState.FAIL for trial in study.trials)
            if not complete_trials:
                rows.append(
                    {
                        "model_name": model_name,
                        "model_status": "FAILED",
                        "successful_trials": 0,
                        "failed_trials": failed_trials,
                        "notes": "No Optuna trial completed successfully.",
                    }
                )
                continue

            best_trial = study.best_trial
            resolved_params = json.loads(best_trial.user_attrs["resolved_hyperparameters"])
            best_params[model_name] = resolved_params
            rows.append(
                {
                    "model_name": model_name,
                    "best_optuna_trial_number": best_trial.number,
                    "best_single_validation_macro_f1": float(best_trial.value),
                    "best_trial_training_macro_f1": best_trial.user_attrs["training_macro_f1"],
                    "best_trial_validation_accuracy": best_trial.user_attrs["validation_accuracy"],
                    "best_trial_validation_macro_precision": best_trial.user_attrs["validation_macro_precision"],
                    "best_trial_validation_macro_recall": best_trial.user_attrs["validation_macro_recall"],
                    "best_trial_validation_weighted_f1": best_trial.user_attrs["validation_weighted_f1"],
                    "best_trial_generalization_gap": best_trial.user_attrs["generalization_gap"],
                    "best_trial_training_time_seconds": best_trial.user_attrs["training_time_seconds"],
                    "best_trial_validation_inference_time_seconds": best_trial.user_attrs["validation_inference_time_seconds"],
                    "best_iteration": best_trial.user_attrs.get("best_iteration"),
                    "best_hyperparameters": json.dumps(resolved_params, sort_keys=True),
                    "successful_trials": len(complete_trials),
                    "failed_trials": failed_trials,
                    "model_status": "COMPLETE",
                    "notes": f"{len(complete_trials)} complete, {failed_trials} failed",
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "model_name": model_name,
                    "model_status": "FAILED",
                    "notes": f"{type(exc).__name__}: {exc}",
                }
            )

    summary = pd.DataFrame(rows).sort_values(
        "best_single_validation_macro_f1",
        ascending=False,
        na_position="last",
        ignore_index=True,
    )
    return summary, best_params


def evaluate_seed_stability(
    best_params: Mapping[str, Mapping[str, Any]],
    train_features: np.ndarray,
    train_targets: np.ndarray,
    validation_features: np.ndarray,
    validation_targets: np.ndarray,
    class_mapping: dict[int, str],
    seeds: list[int],
    early_stopping_rounds: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retrain each best configuration across fixed random seeds.

    Args:
        best_params: Resolved best parameters by successful model.
        train_features: Fixed training features.
        train_targets: Fixed training targets.
        validation_features: Fixed validation features.
        validation_targets: Fixed validation targets.
        class_mapping: Ordered class ID-to-name mapping.
        seeds: Seeds used to measure estimator randomness.
        early_stopping_rounds: Patience for supported boosting models.

    Returns:
        Individual seed runs and one aggregated stability row per model.

    Raises:
        RuntimeError: If every seed run fails.

    Notes:
        This is a randomness check on one fixed split, not cross-validation. Test
        data is intentionally absent from the API.
    """
    rows: list[dict[str, Any]] = []
    class_names = list(class_mapping.values())
    for model_name in MODEL_NAMES:
        if model_name not in best_params:
            continue
        for seed in seeds:
            try:
                model = build_model(
                    model_name,
                    best_params[model_name],
                    seed,
                    early_stopping_rounds=early_stopping_rounds,
                )
                training_time = fit_model(
                    model_name,
                    model,
                    train_features,
                    train_targets,
                    validation_features=validation_features,
                    validation_targets=validation_targets,
                    early_stopping_rounds=early_stopping_rounds,
                )
                train_predictions = np.asarray(model.predict(train_features))
                validation_predictions, _, inference_time = predict_labels_and_probabilities(
                    model, validation_features
                )
                train_metrics = classification_metrics(train_targets, train_predictions, class_mapping)
                validation_metrics = classification_metrics(
                    validation_targets, validation_predictions, class_mapping
                )
                row = {
                    "model_name": model_name,
                    "seed": seed,
                    "status": "COMPLETE",
                    "training_macro_f1": train_metrics["macro_f1"],
                    "validation_accuracy": validation_metrics["accuracy"],
                    "validation_macro_precision": validation_metrics["macro_precision"],
                    "validation_macro_recall": validation_metrics["macro_recall"],
                    "validation_macro_f1": validation_metrics["macro_f1"],
                    "validation_weighted_f1": validation_metrics["weighted_f1"],
                    "generalization_gap": train_metrics["macro_f1"] - validation_metrics["macro_f1"],
                    "training_time_seconds": training_time,
                    "validation_inference_time_seconds": inference_time,
                    "best_iteration": effective_iteration_count(model_name, model),
                    "error_message": "",
                }
                for class_name in class_names:
                    row[f"validation_{class_name}_f1"] = validation_metrics["per_class"][class_name]["f1"]
                rows.append(row)
                del model
                gc.collect()
            except Exception as exc:
                rows.append(
                    {
                        "model_name": model_name,
                        "seed": seed,
                        "status": "FAIL",
                        "error_message": f"{type(exc).__name__}: {exc}",
                    }
                )

    seed_results = pd.DataFrame(rows)
    complete = seed_results.query("status == 'COMPLETE'").copy()
    if complete.empty:
        raise RuntimeError("No model completed seed-stability verification.")

    summary = complete.groupby("model_name", as_index=False).agg(
        mean_validation_macro_f1=("validation_macro_f1", "mean"),
        validation_macro_f1_std=("validation_macro_f1", "std"),
        min_validation_macro_f1=("validation_macro_f1", "min"),
        max_validation_macro_f1=("validation_macro_f1", "max"),
        validation_accuracy=("validation_accuracy", "mean"),
        validation_macro_precision=("validation_macro_precision", "mean"),
        validation_macro_recall=("validation_macro_recall", "mean"),
        validation_weighted_f1=("validation_weighted_f1", "mean"),
        mean_training_macro_f1=("training_macro_f1", "mean"),
        mean_generalization_gap=("generalization_gap", "mean"),
        mean_training_time_seconds=("training_time_seconds", "mean"),
        mean_validation_inference_time_seconds=("validation_inference_time_seconds", "mean"),
        successful_seed_runs=("seed", "count"),
    )
    summary["validation_macro_f1_std"] = summary["validation_macro_f1_std"].fillna(0.0)
    per_class = (
        complete.groupby("model_name")[
            [f"validation_{name}_f1" for name in class_names]
        ]
        .mean()
        .reset_index()
        .rename(
            columns={
                f"validation_{name}_f1": f"mean_validation_{name}_f1"
                for name in class_names
            }
        )
    )
    summary = summary.merge(per_class, on="model_name", how="left")
    class_columns = [f"mean_validation_{name}_f1" for name in class_names]
    summary["minimum_mean_validation_class_f1"] = summary[class_columns].min(axis=1)
    summary["validation_per_class_f1_range"] = (
        summary[class_columns].max(axis=1) - summary[class_columns].min(axis=1)
    )
    summary = summary.sort_values(
        "mean_validation_macro_f1", ascending=False, ignore_index=True
    )
    return seed_results, summary


def refit_final_model(
    model_name: str,
    params: Mapping[str, Any],
    seed: int,
    train_features: np.ndarray,
    train_targets: np.ndarray,
    validation_features: np.ndarray,
    validation_targets: np.ndarray,
    *,
    fixed_iterations: int | None,
    early_stopping_rounds: int,
) -> tuple[ClassifierMixin, float, int]:
    """Refit the selected model on train plus validation without test data.

    Args:
        model_name: Selected model name.
        params: Selected resolved hyperparameters.
        seed: Representative seed closest to mean validation performance.
        train_features: Original training features.
        train_targets: Original training targets.
        validation_features: Original validation features to add to final fit.
        validation_targets: Original validation targets to add to final fit.
        fixed_iterations: Validation-derived boosting iteration count.
        early_stopping_rounds: Constructor compatibility setting for boosting.

    Returns:
        Fitted final model, elapsed training seconds, and combined sample count.

    Notes:
        Boosting iteration counts are fixed before this call so final fitting can
        use all train-plus-validation rows without test-based early stopping.
    """
    combined_features = np.concatenate([train_features, validation_features], axis=0)
    combined_targets = np.concatenate([train_targets, validation_targets], axis=0)
    model = build_model(
        model_name,
        params,
        seed,
        early_stopping_rounds=early_stopping_rounds,
        final_training=True,
        fixed_iterations=fixed_iterations,
    )
    elapsed = fit_model(model_name, model, combined_features, combined_targets)
    return model, elapsed, len(combined_targets)
