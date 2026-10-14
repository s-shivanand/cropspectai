import joblib
import pandas as pd


def predict_model(X: pd.DataFrame, model_path):

    #load best model
    pipeline = joblib.load(model_path)

    # Predict
    predictions = pipeline.predict(X)

    return predictions
