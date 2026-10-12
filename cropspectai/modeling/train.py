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

def cross_validate_model(model, X, y, n_splits=5, scale=True, scoring="neg_mean_absolute_percentage_error"):

    cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)

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