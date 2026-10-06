from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
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

    # Metrics
    mae_train = mean_absolute_error(y_train, predictions_train)
    mape_train = mean_absolute_percentage_error(
        y_train, predictions_train
    ) * 100

    # Test metrics
    mae_test = mean_absolute_error(y_test, predictions_test)
    mape_test = mean_absolute_percentage_error(
        y_test, predictions_test
    ) * 100

    r2_test = r2_score(y_test, predictions_test)

    print(f"MAE (train):  {mae_train:.4f}")
    print(f"MAPE (train): {mape_train:.2f}%")
    print(f"MAE (test):   {mae_test:.4f}")
    print(f"MAPE (test):  {mape_test:.2f}%")
    print(f"R² (test):    {r2_test:.4f}")


