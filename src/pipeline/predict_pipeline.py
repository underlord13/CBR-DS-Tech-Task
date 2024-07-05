import os
import sys
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.exception import CustomException
from src.utils import load_object, evaluate_model, forecast_future

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, X_train, y_train, X_test, y_test, model):
        try:
            model_path = os.path.join("artifacts", f"model_{model.split('_')[0]}.pkl")
            model = load_object(file_path = model_path)
            
            predictions = evaluate_model(X_train, y_train, X_test, y_test, model)
            return predictions
        
        except Exception as e:
            raise CustomException(e,sys)
    def forecast(self, X_train, y_train, model, target_variable, steps):
        try:
            model_path = os.path.join("artifacts", f"model_{model.split('_')[0]}.pkl")
            model = load_object(file_path = model_path)
            
            predictions = forecast_future(X_train, y_train, model, target_variable, steps)
            return predictions
        
        except Exception as e:
            raise CustomException(e,sys)
        