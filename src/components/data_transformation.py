import os
import sys
import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from dataclasses import dataclass

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        Function to process the data, creates .pkl file with preprocessor.
        As for now, only creates lag features based on given data and imputes NA's, but can be upgraded to perform some more advanced preprocessing.
        '''
        try:
            
            num_pipeline = Pipeline(
                steps=[
                ("imputer", SimpleImputer(strategy = "mean"))
                ]
            )
            
            preprocessor = ColumnTransformer(
                transformers = [
                ("num_pipeline", num_pipeline, make_column_selector(dtype_include = np.number))
                ],
                
                remainder = 'passthrough'
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self, train_path, test_path):

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read train and test files")

            train_df.set_index('date', inplace=True)
            test_df.set_index('date', inplace=True)

            preprocessing_obj = self.get_data_transformer_object()
            logging.info("Get preprocessing object")

            target_variable = "CPI"

            numerical_columns = train_df.columns[1:]

            X_train = train_df.drop(columns = [target_variable], axis = 1)
            y_train = train_df[target_variable]

            X_test = test_df.drop(columns = [target_variable], axis = 1)
            y_test = test_df[target_variable]

            logging.info("Applying preprocessing object on training and testing dataframes")

            X_train_arr = preprocessing_obj.fit_transform(X_train)
            X_test_arr = preprocessing_obj.transform(X_test)

            X_train = pd.DataFrame(X_train_arr, columns = numerical_columns, index = y_train.index)
            X_test = pd.DataFrame(X_test_arr, columns = numerical_columns, index = y_test.index)

            #train = pd.concat([y_train, X_train])
            #train = train.rename(columns = {0: target_variable})

            #test = pd.concat([y_test, X_test])
            #test = test.rename(columns = {0: target_variable})
            
            logging.info(f"Saving preprocessing object")
            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj 
 
            )

            return (
                X_train,
                y_train,
                X_test,
                y_test,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        
        except Exception as e:
            raise CustomException(e,sys)