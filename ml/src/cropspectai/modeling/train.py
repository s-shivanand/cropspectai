"""Build regression pipelines, score folds, and tune XGBoost hyperparameters."""

from typing import Literal

import numpy as np
import optuna
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


def train_model(
    model: BaseEstimator,
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.DataFrame | pd.Series | np.ndarray,
    X_test: pd.DataFrame | np.ndarray,
    y_test: pd.DataFrame | pd.Series | np.ndarray,
    scale: bool = True,
) -> tuple[Pipeline, dict[str, float]]:
    """Fit a regression pipeline and evaluate training and test predictions.

    Args:
        model: Scikit-learn-compatible regressor implementing fit and predict.
        X_train: Prepared training feature matrix with samples as rows.
        y_train: Training targets in the same sample order as ``X_train``.
        X_test: Test feature matrix with the same columns as ``X_train``.
        y_test: Test targets in the same sample order as ``X_test``.
        scale: Whether to standardize features after mean imputation.

    Returns:
        Pair of the fitted pipeline and its metrics dictionary. Metrics
        include training and test MAE, MAPE in percent, and RMSE, plus test
        R-squared. MAE and RMSE use the target's units.

    Raises:
        ValueError: Inputs are incompatible with fitting, prediction, or
            metric calculations.

    Note:
        Preprocessing is fitted only on training data. The supplied model
        instance is used directly and fitted in place. No model or results
        are saved to disk.
    """
    # Pipeline
    pipeline = build_pipeline(model, scale)

    # Train
    pipeline.fit(X_train, y_train)

    # Predict
    predictions_train = pipeline.predict(X_train)
    predictions_test = pipeline.predict(X_test)

    # Train metrics
    mae_train = mean_absolute_error(y_train, predictions_train)
    mape_train = mean_absolute_percentage_error(y_train, predictions_train) * 100
    rmse_train = root_mean_squared_error(y_train, predictions_train)

    # Test metrics
    mae_test = mean_absolute_error(y_test, predictions_test)
    mape_test = mean_absolute_percentage_error(y_test, predictions_test) * 100
    r2_test = r2_score(y_test, predictions_test)
    rmse_test = root_mean_squared_error(y_test, predictions_test)

    metrics = {
        "MAE (train)": mae_train,
        "MAPE (train) [%]": mape_train,
        "RMSE (train)": rmse_train,
        "MAE (test)": mae_test,
        "MAPE (test) [%]": mape_test,
        "RMSE (test)": rmse_test,
        "R² (test)": r2_test,
    }

    return pipeline, metrics


def build_pipeline(model: BaseEstimator, scale: bool = True) -> Pipeline:
    """Build an unfitted pipeline with mean imputation and optional scaling.

    Args:
        model: Scikit-learn-compatible regressor to use as the final step.
        scale: Whether to include StandardScaler between imputation and
            the regressor.

    Returns:
        Unfitted pipeline with steps named "imputer", optionally "scaler",
        and "model". The regressor is the supplied instance, not a clone.
    """
    steps = [
        ("imputer", SimpleImputer(strategy="mean")),
    ]

    if scale:
        steps.append(("scaler", StandardScaler()))

    steps.append(("model", model))

    pipeline = Pipeline(steps)

    return pipeline


def cross_validate_model(
    model: BaseEstimator,
    X: pd.DataFrame | np.ndarray,
    y: pd.DataFrame | pd.Series | np.ndarray,
    n_splits: int = 5,
    scale: bool = True,
    scoring: str = "neg_mean_absolute_percentage_error",
    random_state: int = 42,
) -> np.ndarray:
    """Score a regression pipeline over shuffled, reproducible K-fold splits.

    Args:
        model: Scikit-learn-compatible regressor to evaluate.
        X: Prepared feature matrix with samples as rows.
        y: Target values in the same sample order as ``X``.
        n_splits: Number of folds, at least two and no greater than the
            sample count.
        scale: Whether each fold's pipeline includes standard scaling.
        scoring: Scikit-learn scoring name. The default negative MAPE score
            is converted to positive percentage error before returning.
        random_state: Seed controlling shuffled fold assignments. It does
            not set the regressor's own random state.

    Returns:
        One-dimensional array with one score per fold. MAPE is returned as
        positive percentage error; all other scorers retain scikit-learn's
        sign and units.

    Raises:
        ValueError: Fold settings or the scorer are invalid, or every fit
            fails.

    Note:
        Scikit-learn clones the pipeline for each fold, fitting imputation
        and scaling only on that fold's training data. Folds run serially.
        Individual failed fits may produce NaN scores and warnings.
    """
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    scores = cross_val_score(
        build_pipeline(model, scale=scale),
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=1,
    )

    if scoring == "neg_mean_absolute_percentage_error":
        scores = -scores * 100  # positive MAPE
    else:
        scores = scores

    return scores


def tune_xgboost(
    X: pd.DataFrame | np.ndarray,
    y: pd.DataFrame | pd.Series | np.ndarray,
    n_splits: int = 5,
    n_trials: int = 50,
    scale: bool = True,
    scoring: Literal[
        "r2", "neg_mean_absolute_percentage_error"
    ] = "neg_mean_absolute_percentage_error",
    random_state: int = 42,
) -> optuna.Study:
    """Tune XGBoost parameters with Optuna and seeded K-fold validation.

    Args:
        X: Prepared feature matrix with samples as rows.
        y: Regression targets in the same sample order as ``X``.
        n_splits: Number of folds used to evaluate every trial.
        n_trials: Number of trials to run. At least one successful trial is
            required to report the best result.
        scale: Whether each validation pipeline includes standard scaling.
        scoring: "r2" to maximize mean R-squared, or
            "neg_mean_absolute_percentage_error" to minimize positive MAPE
            in percent.
        random_state: Seed for the Optuna sampler, fold assignments, and
            XGBoost estimators.

    Returns:
        Completed Optuna study. Each successful trial stores model_params,
        fold_scores, and score_std in its user attributes. The best model
        is not refitted on all samples or saved to disk.

    Raises:
        ValueError: The scorer is unsupported, validation fails, or no
            successful trial is available when retrieving the best result.

    Note:
        Evaluates the existing reference configuration first, then samples
        additional configurations. Trials run serially, while each XGBoost
        estimator may use all CPU cores. Prints the best parameters and
        mean cross-validation score.
    """
    directions = {
        "r2": "maximize",
        "neg_mean_absolute_percentage_error": "minimize",
    }

    if scoring not in directions:
        raise ValueError("scoring must be 'r2' or 'neg_mean_absolute_percentage_error'")

    fixed_params = {
        "tree_method": "hist",
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "n_jobs": -1,
        "random_state": random_state,
    }

    def objective(trial: optuna.Trial) -> float:
        """Sample one configuration and return its mean validation score.

        Args:
            trial: Optuna trial used to suggest and record hyperparameters.

        Returns:
            Mean score across folds, using the outer function's scorer.
            Model parameters and fold statistics are stored on the trial.
        """
        # Include exactly zero as an option for L1 regularization.
        use_l1 = trial.suggest_categorical("use_l1", [False, True])

        reg_alpha = trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True) if use_l1 else 0.0

        params = {
            "n_estimators": trial.suggest_int("n_estimators", 300, 2000, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 30.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": reg_alpha,
            "reg_lambda": trial.suggest_float("reg_lambda", 0.001, 100.0, log=True),
        }

        model = XGBRegressor(**fixed_params, **params)

        # Your helper uses identical seeded folds for every trial
        # and converts negative MAPE scores to positive percentages.
        cv_scores = cross_validate_model(
            model,
            X,
            y,
            n_splits=n_splits,
            scale=scale,
            scoring=scoring,
            random_state=random_state,
        )

        # Store actual model parameters, excluding Optuna's use_l1 flag.
        trial.set_user_attr("model_params", {**fixed_params, **params})
        trial.set_user_attr("fold_scores", cv_scores.tolist())
        trial.set_user_attr("score_std", float(cv_scores.std()))

        return float(cv_scores.mean())

    study = optuna.create_study(
        direction=directions[scoring],
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )

    # Evaluate your current configuration as the first trial.
    study.enqueue_trial(
        {
            "n_estimators": 1000,
            "learning_rate": 0.01,
            "max_depth": 6,
            "min_child_weight": 1.0,
            "subsample": 0.8,
            "colsample_bytree": 1.0,
            "use_l1": False,
            "reg_lambda": 1.0,
        }
    )

    study.optimize(
        objective,
        n_trials=n_trials,
        n_jobs=1,
    )

    print("\n🏆 Best trial:")

    best_params = study.best_trial.user_attrs["model_params"]

    for name, value in best_params.items():
        if isinstance(value, float):
            print(f"   {name} = {value:.6f}")
        else:
            print(f"   {name} = {value}")

    if scoring == "r2":
        print(f"   Mean CV R² = {study.best_value:.4f}")
    else:
        print(f"   Mean CV MAPE = {study.best_value:.4f}%")

    return study
