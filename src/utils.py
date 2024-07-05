import os
import sys
import numpy as np 
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
import dill
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_squared_error

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from src.exception import CustomException

def create_lag_features(df, variable, num_lags = 9):
    '''
    Function to generate lag features for a variable in a dataframe.
    '''
    for i in range(1, num_lags + 1):
        df[f'{variable}_lag{i}'] = df[variable].shift(i)

def save_object(file_path, obj):
    '''
    Function to save object as .pkl file.
    '''
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok = True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    '''
    Function to load object from .pkl file.
    '''
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_model(X_train, y_train, X_test, y_test, model):
    '''
    Function to evaluate models using an expanding window approach (train on observations from 1 to T, predict T+1; then train on obs. from 1 to T+1, predict T+2; ...).
    '''
    predictions = []
    for i in range(len(X_test)):
        X_train_expanding = pd.concat([X_train, X_test[:i]])
        y_train_expanding = pd.concat([y_train, y_test[:i]])
        if model in ['mean_forecast', 'naive_forecast']:
            pred = model(y_train_expanding, y_test[i:i+1])[0]
        else:
            pred = model(X_train_expanding, y_train_expanding, X_test[i:i+1])[0]
        predictions.append(pred)
    predictions = np.array(predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    #print(f"{name} RMSE: {rmse:.3f}")
    return predictions

def forecast_future(X_train, y_train, model, target_variable = 'CPI', steps = 6):
    '''
    Function to make a N month forecast into the future (base case is N = 6).
    '''
    X_train_expanding = X_train.copy()
    y_train_expanding = y_train.copy()
    predictions = []
    for _ in range(steps):
        if model.__name__ in ['mean_forecast', 'naive_forecast']:
            pred = model(y_train_expanding, [0])[0]
        else:
            pred = model(X_train_expanding, y_train_expanding, [X_train_expanding.iloc[-1]])[0]
        predictions.append(pred)

        num_lags = len(X_train.columns)
        new_row = {f'{target_variable}_lag1': y_train_expanding.iloc[-1]}
        for i in range(2, num_lags + 1):
            new_row[f'{target_variable}_lag{i}'] = X_train_expanding.iloc[-1][f'{target_variable}_lag{i-1}']
        
        X_train_expanding = X_train_expanding.append(new_row, ignore_index=True)
        y_train_expanding = y_train_expanding.append(pd.Series([pred]), ignore_index=True)
    return predictions


def plot_models(models, X_train, y_train, X_test, y_test, X_full, y_full, file_path = None):
    '''
    Function to visualize models' performance on out-of-sample test data. Also plots future (6M ahead predictions) and saves matplotlib plot as a picture.
    '''
    plt.figure(figsize=(14, 8))
    plt.plot(y_train.index, y_train, label = 'Train', linewidth = 3, color = 'blue', marker = 'o')
    plt.plot(y_test.index, y_test, label = 'Test', linewidth = 3, color = 'red', marker = 'o')

    plot_handles = {}

    for name, model in models:
        predictions, rmse = evaluate_model(name, X_train, y_train, X_test, y_test, model)
        line, = plt.plot(y_test.index, predictions, label=f'{name} (RMSE: {rmse:.2f})')
        plot_handles[name] = line.get_color()

        future_dates = pd.date_range(start = y_test.index[-1], periods = 7, freq = 'M')[1:]
        future_predictions = forecast_future(model, X_full, y_full)
        plt.plot(future_dates, future_predictions, linestyle = '--', color = plot_handles[name])

    plt.legend()
    plt.title('Forecasting Models Comparison')
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.xlabel('Year')
    plt.tight_layout()

    if file_path:
        try:
            dir_path = os.path.dirname(file_path)
            os.makedirs(dir_path, exist_ok = True)
            plt.savefig(file_path)
        except Exception as e:
            raise Exception
        
    #plt.show()