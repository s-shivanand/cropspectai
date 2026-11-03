from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
    root_mean_squared_error
)
import joblib


def evaluate_model( 
                    X: pd.DataFrame, 
                    y: pd.DataFrame,
                    model_path):

    #load best model
    pipeline = joblib.load(model_path)

    # Predict
    predictions_test = pipeline.predict(X)

    # Test metrics
    mae_test = mean_absolute_error(y, predictions_test)
    mape_test = mean_absolute_percentage_error(
        y, predictions_test
    ) * 100
    r2_test = r2_score(y, predictions_test)
    rmse_test = root_mean_squared_error(y, predictions_test)

    metrics = {
        "MAE (test)": mae_test,
        "MAPE (test) [%]": mape_test,
        "RMSE (test)": rmse_test,
        "R² (test)": r2_test,
    }

    return metrics