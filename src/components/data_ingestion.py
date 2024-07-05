import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.exception import CustomException
from src.logger import logging
from src.utils import create_lag_features

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        '''
        Function to read data from source and split it to train/test. Also creates directories to store the files.
        '''
        logging.info("Start data ingestion")
        try:
            df = pd.read_csv("manual_data_download\data.csv", encoding='cp1251')
            logging.info('Process the dataset as a pandas dataframe')
            
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok = True)

            df.set_index('date', inplace = True)

            create_lag_features(df = df, variable = 'CPI', num_lags = 9)
            logging.info('Create lag features')
            df.to_csv(self.ingestion_config.raw_data_path, header = True)
            
            logging.info("Start train/test split")
            train, test = train_test_split(df, test_size = 0.2, shuffle = False)

            train.to_csv(self.ingestion_config.train_data_path, header = True)

            test.to_csv(self.ingestion_config.test_data_path, header = True)

            logging.info("Data ingestion is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data,test_data)