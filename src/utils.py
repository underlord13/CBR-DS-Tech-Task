import os
import sys
from datetime import datetime
from glob import glob
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
    
def evaluate_model(model, X_train, y_train, X_test, y_test):
    '''
    Function to evaluate models using an expanding window approach (train on observations from 1 to T, predict T+1; then train on obs. from 1 to T+1, predict T+2; ...).
    '''
    predictions = []
    for i in range(len(X_test)):
        X_train_expanding = pd.concat([X_train, X_test[:i]])
        y_train_expanding = pd.concat([y_train, y_test[:i]])
        if model.name in ["mean_forecast", "naive_forecast"]:
            pred = model(y_train_expanding, y_test[i:i+1])[0]
        else:
            pred = model(X_train_expanding, y_train_expanding, X_test[i:i+1])[0]
        predictions.append(pred)
    predictions = np.array(predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    print(f"{model.name} RMSE: {rmse:.3f}")
    return predictions, rmse

def forecast_future(model, X_train, y_train, target_variable = 'CPI', steps = 6):
    '''
    Function to make a N month forecast into the future (base case is N = 6).
    '''
    X_train_expanding = X_train.copy()
    y_train_expanding = y_train.copy()
    predictions = []
    for _ in range(steps):
        if model.name in ['mean_forecast', 'naive_forecast']:
            pred = model(y_train_expanding, [0])[0]
        else:
            pred = model(X_train_expanding, y_train_expanding, [X_train_expanding.iloc[-1]])[0]
        predictions.append(pred)

        num_lags = len(X_train.columns)
        new_row = {f'{target_variable}_lag1': y_train_expanding.iloc[-1]}
        for i in range(2, num_lags + 1):
            new_row[f'{target_variable}_lag{i}'] = X_train_expanding.iloc[-1][f'{target_variable}_lag{i-1}']
        
        X_train_expanding = X_train_expanding.append(new_row, ignore_index = True)
        y_train_expanding = y_train_expanding.append(pd.Series([pred]), ignore_index = True)
    return predictions


def plot_models(models, X_train, y_train, X_test, y_test, folder_path = None):
    '''
    Function to visualize models' performance on out-of-sample test data. Also plots future (6M ahead predictions) and saves matplotlib plot as a picture.
    '''
    X_train.index  = pd.to_datetime(X_train.index)
    X_test.index  = pd.to_datetime(X_test.index)
    y_train.index = pd.to_datetime(y_train.index)
    y_test.index = pd.to_datetime(y_test.index)
    X_full = pd.concat([X_train, X_test])
    y_full = pd.concat([y_train, y_test])

    plt.figure(figsize=(14, 8))
    plt.plot(y_train.index, y_train, label = 'Train', linewidth = 3, color = 'blue', marker = 'o')
    plt.plot(y_test.index, y_test, label = 'Test', linewidth = 3, color = 'red', marker = 'o')

    plot_handles = {}

    for name, model in models:
        predictions, rmse = evaluate_model(model, X_train, y_train, X_test, y_test)
        line, = plt.plot(y_test.index, predictions, label=f'{name} (RMSE: {rmse:.2f})')
        plot_handles[name] = line.get_color()

        future_dates = pd.date_range(start = y_test.index[-1], periods = 7, freq = 'MS')[1:]
        future_predictions = forecast_future(model, X_full, y_full)
        plt.plot(future_dates, future_predictions, linestyle = '--', color = plot_handles[name])

    plt.legend()
    plt.title('Forecasting Models Comparison')
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.xlabel('Year')
    plt.tight_layout()

    if folder_path:
        try:
            manage_static_folder(folder_path)

            file_name = f"plot_{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
            file_path = os.path.join(folder_path, file_name)

            os.makedirs(folder_path, exist_ok=True)

            plt.savefig(file_path)
            return file_path
        except Exception as e:
            raise CustomException(e, sys)
        
    #plt.show()

def manage_static_folder(folder_path, max_files = 10):
    '''
    Function to limit the amount of images contained in static folder (no more than 10, otherwise delete the oldest file).
    '''
    files = sorted(glob(os.path.join(folder_path, '*.png')), key = os.path.getmtime)
    
    while len(files) > max_files:
        os.remove(files.pop(0))

def convert_to_json(val):
        if isinstance(val, (np.float32, np.float64)):
            return float(val)
        if isinstance(val, (np.int32, np.int64)):
            return int(val)
        if isinstance(val, np.ndarray):
            return val.tolist()
        if isinstance(val, (np.bool_)):
            return bool(val)
        if val is np.nan or val is None:
            return None
        return val