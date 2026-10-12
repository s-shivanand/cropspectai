from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
    root_mean_squared_error
)


def train_model(model, 
                X_train: pd.DataFrame, 
                y_train: pd.DataFrame, 
                X_test: pd.DataFrame, 
                y_test: pd.DataFrame,
                scale=True) -> None :

    # Pipeline
    steps = [
        ("imputer", SimpleImputer(strategy="mean")),
    ]

    if scale:
        steps.append(("scaler", StandardScaler()))

    steps.append(("model", model))

    pipeline = Pipeline(steps)

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


