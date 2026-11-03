from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
    root_mean_squared_error
)
from sklearn.model_selection import KFold, cross_val_score
import optuna
from xgboost import XGBRegressor


def train_model(model, 
                X_train: pd.DataFrame, 
                y_train: pd.DataFrame, 
                X_test: pd.DataFrame, 
                y_test: pd.DataFrame,
                scale=True) -> None :

    # Pipeline
    pipeline = build_pipeline(model, scale)

    # Train
    pipeline.fit(X_train, y_train)

    # Predict
    predictions_train = pipeline.predict(X_train)
    predictions_test = pipeline.predict(X_test)

    # Train metrics
    mae_train = mean_absolute_error(y_train, predictions_train)
    mape_train = mean_absolute_percentage_error(
        y_train, predictions_train
    ) * 100
    rmse_train = root_mean_squared_error(y_train, predictions_train)

    # Test metrics
    mae_test = mean_absolute_error(y_test, predictions_test)
    mape_test = mean_absolute_percentage_error(
        y_test, predictions_test
    ) * 100
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

def build_pipeline(model, scale=True):

    steps = [
        ("imputer", SimpleImputer(strategy="mean")),
    ]

    if scale:
        steps.append(("scaler", StandardScaler()))

    steps.append(("model", model))

    pipeline = Pipeline(steps)

    return pipeline

def cross_validate_model(model, X, y, n_splits=5, scale=True, scoring="neg_mean_absolute_percentage_error", random_state=42):

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
        scores = -scores * 100 # positive MAPE
    else:
        scores = scores

    return scores

def tune_xgboost(X, y, n_splits=5, n_trials=50, scale=True, scoring="neg_mean_absolute_percentage_error", random_state=42):

    directions = {
        "r2": "maximize",
        "neg_mean_absolute_percentage_error": "minimize",
    }

    if scoring not in directions:
        raise ValueError(
            "scoring must be 'r2' or "
            "'neg_mean_absolute_percentage_error'"
        )

    fixed_params = {
        "tree_method": "hist",
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "n_jobs": -1,
        "random_state": random_state,
    }

    def objective(trial):
        # Include exactly zero as an option for L1 regularization.
        use_l1 = trial.suggest_categorical("use_l1", [False, True])

        reg_alpha = (
            trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True)
            if use_l1
            else 0.0
        )

        params = {
            "n_estimators": trial.suggest_int(
                "n_estimators", 300, 2000, step=100
            ),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.005, 0.1, log=True
            ),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "min_child_weight": trial.suggest_float(
                "min_child_weight", 1.0, 30.0, log=True
            ),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.6, 1.0
            ),
            "reg_alpha": reg_alpha,
            "reg_lambda": trial.suggest_float(
                "reg_lambda", 0.001, 100.0, log=True
            ),
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
    study.enqueue_trial({
        "n_estimators": 1000,
        "learning_rate": 0.01,
        "max_depth": 6,
        "min_child_weight": 1.0,
        "subsample": 0.8,
        "colsample_bytree": 1.0,
        "use_l1": False,
        "reg_lambda": 1.0,
    })

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

