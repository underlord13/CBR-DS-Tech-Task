import os
import sys
import datetime
import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.exception import CustomException
from src.logger import logging
from src.utils import create_lag_features

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_training import Models
from src.components.model_training import ModelTrainingConfig
from src.components.model_training import ModelTraining

@dataclass
class DataRosstatConfig:
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "data_raw.xlsx")
    processed_data_path: str = os.path.join("artifacts", "data.xlsx")

    filtered_train_data_path: str = os.path.join("artifacts", "train_filtered.csv")
    filtered_test_data_path: str = os.path.join("artifacts", "test_filtered.csv")
    filtered_processed_data_path: str = os.path.join("artifacts", "data_filtered.xlsx")

class DataRosstat:
    def __init__(self):
        self.ingestion_config = DataRosstatConfig()
        self.latest_url = None

    def get_url(self):
        '''
        Function to search through Rosstat uploadings.
        Assumes that the file name structure is consistent: if for July 2024 the latest available file is "Ipc_mes_05-2024.xlsx", than in the upcoming update it will be "Ipc_mes_06-2024.xlsx" and so on.
        Checks the latest available url for 3 months: for July 2024 checks May, June and July ("Ipc_mes_05-2024.xlsx", "Ipc_mes_06-2024.xlsx" and "Ipc_mes_07-2024.xlsx").
        '''
        current_date = datetime.datetime.now()
        current_month = current_date.month
        current_year = current_date.year
        months_to_check = [current_month - 2, current_month - 1, current_month]

        for month in months_to_check:
            if month <= 0:
                month = 12
                year = current_year - 1
            elif month > 12:
                month = 1
                year = current_year + 1
            else:
                year = current_year

            url = f"https://rosstat.gov.ru/storage/mediabank/Ipc_mes_{month:02d}-{year}.xlsx"
            response = requests.get(url)

            if response.status_code == 200:
                self.latest_url = url
                logging.info("File collected from Rosstat")
                break

        if not self.latest_url:
            raise CustomException("Failed to find a valid file URL.", sys)
        
    def download_file(self):
        '''
        Function to download the file from Rosstat url and save it as "data.xlsx". Creates the folder to store the file.
        '''
        if not self.latest_url:
            raise CustomException("No URL to download the file from.", sys)

        os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok = True)
        response = requests.get(self.latest_url)
        with open(self.ingestion_config.raw_data_path, 'wb') as file:
            file.write(response.content)

        logging.info(f"File successfully downloaded and saved as {self.ingestion_config.raw_data_path}")

    def load_dataframe(self, sheet_name = '01', skiprows = 3, usecols = 'A:AI', nrows = 13):
        '''
        Function to load the data from Rosstat Excel file.
        Assumes that the inner structure of files remain consistent. In 2025 should be updated to usecols='A:AJ' (!).
        A specific sheet can be specified ("01" -- All goods and utils, "02" -- Only food goods, "03" -- Only non-food goods, "04" -- Only utils).
        '''
        xls = pd.ExcelFile(self.ingestion_config.raw_data_path)
        df = pd.read_excel(xls, sheet_name = sheet_name, skiprows = skiprows, usecols = usecols, nrows = nrows)
        return df
    
    def transform_dataframe(self, df):
        '''
        Function to transform the raw data from Rosstat from MoM NSA to YoY (December 2001 as base).
        Remove data before December 2001 entirely as too old. Returns YoY data starting from January 2003.
        '''
        df = df.drop([0], axis = 0)
        df.rename(columns = {df.columns[0]: 'month'}, inplace = True)
        df = pd.melt(df, id_vars = ['month'], var_name = 'year', value_name = 'CPI_mom')
        
        month_map = {
            'январь': '01', 'февраль': '02', 'март': '03', 'апрель': '04', 'май': '05', 
            'июнь': '06', 'июль': '07', 'август': '08', 'сентябрь': '09', 'октябрь': '10', 
            'ноябрь': '11', 'декабрь': '12'
        }

        df['month'] = df['month'].map(month_map)
        df['date'] = '01.' + df['month'] + '.' + df['year'].astype(str)
        df['date'] = pd.to_datetime(df['date'], format='%d.%m.%Y')
        df = df.drop(['month', 'year'], axis = 1)
        df.set_index('date', inplace = True)

        df = df[df.index >= '2001-12-01']
        df = df.dropna(axis=0)

        df['CPI_base'] = 100
        for i in range(1, len(df.index)):
            df.loc[df.index[i], 'CPI_base'] = (df['CPI_mom'][i] * df['CPI_base'][i - 1]) / 100

        df["CPI"] = 100
        for i in range(13, len(df.index)):
            df.loc[df.index[i], 'CPI'] = ((df['CPI_base'][i] / df['CPI_base'][i - 12]) * 100).round(2)

        df = df.drop(['CPI_mom', 'CPI_base'], axis=1)
        df = df[df.index >= '2003-01-01']

        return df

    def initiate_data_ingestion(self, sheet_name = '01', start_date=None, end_date=None):
        '''
        Function to read data from source, create lag features and split dataframe to train/test.
        '''
        logging.info("Start data ingestion")
        try:
            self.get_url()
            self.download_file()

            df = self.load_dataframe(sheet_name = sheet_name, skiprows = 3, usecols = 'A:AI', nrows = 13)
            logging.info('Load raw Rosstat file')

            df = self.transform_dataframe(df)
            logging.info('Transform raw Rosstat file to standard time-series')

            create_lag_features(df = df, variable = 'CPI', num_lags = 4)
            logging.info('Create lag features')

            df.to_excel(self.ingestion_config.processed_data_path, header = True)
            
            logging.info("Start train/test split")
            train, test = train_test_split(df, test_size = 0.2, shuffle = False)

            train.to_csv(self.ingestion_config.train_data_path, header = True)
            test.to_csv(self.ingestion_config.test_data_path, header = True)

            df_filtered = df.copy()
            if start_date:
                df_filtered = df_filtered[df_filtered.index >= pd.to_datetime(start_date)]
                logging.info('Specify start date')
            if end_date:
                df_filtered = df_filtered[df_filtered.index <= pd.to_datetime(end_date)]
                logging.info('Specify end date')
                df_filtered.to_excel(self.ingestion_config.filtered_processed_data_path, header = True)
                train_filtered, test_filtered = train_test_split(df_filtered, test_size = 0.2, shuffle = False)
                train_filtered.to_csv(self.ingestion_config.filtered_train_data_path, header = True)
                test_filtered.to_csv(self.ingestion_config.filtered_test_data_path, header = True)

            logging.info("Data ingestion is completed")

            if start_date:
                return(
                self.ingestion_config.filtered_train_data_path,
                self.ingestion_config.filtered_test_data_path
            )
            else:
                return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
                )
        except Exception as e:
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    obj = DataRosstat()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    X_train, y_train, X_test, y_test, _ = data_transformation.initiate_data_transformation(train_data, test_data)

    model_trainer = ModelTraining()
    model_trainer.initiate_model_training(X_train, y_train, X_test, y_test)