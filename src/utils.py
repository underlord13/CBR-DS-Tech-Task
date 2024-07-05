import os
import sys
import numpy as np 
import pandas as pd
import dill

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from src.exception import CustomException

def create_lag_features(df, variable, num_lags):
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