import os
import sys
import numpy as np 
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from dataclasses import dataclass

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_model, forecast_future, plot_models


class Models:
    def __init__(self):
        self.best_models = {}

    def mean_forecast(self, train, test):
        mean_value = train.mean()
        predictions = [mean_value] * len(test)
        return predictions

    def naive_forecast(self, train, test):
        predictions = np.repeat(train.iloc[-1], len(test))
        return predictions

    def lr_forecast(self, X_train, y_train, X_test):
        model = LinearRegression()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        return predictions

    def hyperparameter_tuning(self, model, param_grid, X_train, y_train):
        tscv = TimeSeriesSplit(n_splits=5)
        grid_search = GridSearchCV(model, param_grid, cv=tscv)
        grid_search.fit(X_train, y_train)
        print(f"Best parameters for {model.__class__.__name__}: {grid_search.best_params_}")
        return grid_search.best_estimator_

    def tune_models(self, X_train, y_train):
        param_grids = {
            'RandomForest': {
                'random_state': [42],
                'bootstrap': [True],
                'criterion': ['mse', 'mae'],
                'n_estimators': [8, 16, 32],
                'max_depth': [None, 4, 8],
                'min_samples_split': [2, 6, 10]
            },
            'CatBoost': {
                'loss_function': ['MAPE'],
                'iterations': [100, 200],
                'depth': [4, 6, 8],
                'learning_rate': [0.1]
            },
            'XGBoost': {
                'n_estimators': [100, 150],
                'max_depth': [2, 3, 4],
                'learning_rate': [0.01, 0.1],
                'reg_alpha': [0, 0.5, 1]
            }
        }

        self.best_models['RandomForest'] = self.hyperparameter_tuning(
            RandomForestRegressor(), param_grids['RandomForest'], X_train, y_train)
        self.best_models['CatBoost'] = self.hyperparameter_tuning(
            CatBoostRegressor(verbose=0), param_grids['CatBoost'], X_train, y_train)
        self.best_models['XGBoost'] = self.hyperparameter_tuning(
            XGBRegressor(objective = 'reg:squarederror'), param_grids['XGBoost'], X_train, y_train)

    def rf_forecast(self, X_train, y_train, X_test):
        model = self.best_models['RandomForest']
        predictions = model.predict(X_test)
        return predictions

    def catboost_forecast(self, X_train, y_train, X_test):
        model = self.best_models['CatBoost']
        predictions = model.predict(X_test)
        return predictions

    def xgboost_forecast(self, X_train, y_train, X_test):
        model = self.best_models['XGBoost']
        predictions = model.predict(X_test)
        return predictions

class ModelWrapper:
    def __init__(self, model, name):
        self.model = model
        self.name = name

    def predict(self, *args):
        if callable(self.model):
            return self.model(*args)
        else:
            X_test = args[2] if len(args) == 3 else args[0]
            return self.model.predict(X_test)

    def __call__(self, *args):
        return self.predict(*args)

@dataclass
class ModelTrainingConfig:
    trained_model_file_path_mean = os.path.join("artifacts", "model_mean.pkl")
    trained_model_file_path_naive = os.path.join("artifacts", "model_naive.pkl")
    trained_model_file_path_lr = os.path.join("artifacts", "model_lr.pkl")
    trained_model_file_path_rf = os.path.join("artifacts", "model_rf.pkl")
    trained_model_file_path_catboost = os.path.join("artifacts", "model_catboost.pkl")
    trained_model_file_path_xgboost = os.path.join("artifacts", "model_xgboost.pkl")

class ModelTraining:
    def __init__(self):
        self.model_trainer_config = ModelTrainingConfig()
        self.models = Models()


    def initiate_model_training(self, X_train, y_train, X_test, y_test):
        try:

            logging.info("Begin hyperparameter tuning for ML models")

            self.models.tune_models(X_train, y_train)

            models = [
                ('Mean', Models().mean_forecast),
                ('Naive', Models().naive_forecast),
                ('Lin. Reg.', Models().lr_forecast),
                ('RF', Models().rf_forecast),
                ('CatBoost', Models().catboost_forecast),
                ('XGBoost', Models().xgboost_forecast)
            ]

            logging.info("Save models as .pkl files")
            save_object(
                file_path = self.model_trainer_config.trained_model_file_path_mean,
                obj = ModelWrapper(self.models.mean_forecast, 'mean_forecast')
            )

            save_object(
                file_path = self.model_trainer_config.trained_model_file_path_naive,
                obj = ModelWrapper(self.models.naive_forecast, 'naive_forecast')
            )

            save_object(
                file_path = self.model_trainer_config.trained_model_file_path_lr,
                obj = ModelWrapper(self.models.lr_forecast, 'lr_forecast')
            )

            save_object(
                file_path = self.model_trainer_config.trained_model_file_path_rf,
                obj = ModelWrapper(self.models.best_models['RandomForest'], 'rf_forecast')
            )

            save_object(
                file_path = self.model_trainer_config.trained_model_file_path_catboost,
                obj = ModelWrapper(self.models.best_models['CatBoost'], 'catboost_forecast')
            )

            save_object(
                file_path = self.model_trainer_config.trained_model_file_path_xgboost,
                obj = ModelWrapper(self.models.best_models['XGBoost'], 'xgboost_forecast')
            )

        except Exception as e:
            raise CustomException(e,sys)


