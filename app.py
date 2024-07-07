from flask import Flask, request, render_template, url_for, redirect, jsonify
import os
import pandas as pd

from src.utils import load_object, save_object, evaluate_model, forecast_future, plot_models, manage_static_folder, convert_to_json
from src.components.data_ingestion import DataIngestion
from src.components.data_ingestion import DataIngestionConfig
from src.components.data_rosstat import DataRosstat
from src.components.data_rosstat import DataRosstatConfig
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_training import ModelTraining
from src.components.model_training import ModelTrainingConfig


app = application=Flask(__name__)

@app.route('/', methods = ['GET'])
def index():
    # Choose start date and end date, currently works only for Rosstat
    data_file_path = os.path.join('artifacts', 'data.xlsx')
    df = pd.read_excel(data_file_path)
    date_list = df['date'].astype(str).tolist()
    return render_template('index.html', date_list=date_list)

@app.route('/show', methods = ['POST'])
def show_plot():
    plot_folder = os.path.join('static')

    mean_forecast = load_object(os.path.join('artifacts', 'model_mean.pkl'))
    naive_forecast = load_object(os.path.join('artifacts', 'model_naive.pkl'))
    lr_forecast = load_object(os.path.join('artifacts', 'model_lr.pkl'))
    rf_forecast = load_object(os.path.join('artifacts', 'model_rf.pkl'))
    catboost_forecast = load_object(os.path.join('artifacts', 'model_catboost.pkl'))
    xgboost_forecast = load_object(os.path.join('artifacts', 'model_xgboost.pkl'))

    models = [
        ('Mean', mean_forecast),
        ('Naive', naive_forecast),
        ('Lin. Reg.', lr_forecast),
        ('RF', rf_forecast),
        ('CatBoost', catboost_forecast),
        ('XGBoost', xgboost_forecast)
    ]

    # Determine data source: GosKomStat or Rosstat
    data_source = request.form.get('data_source')
    start_date = request.form.get('start_date')
    end_date = request.form.get('end_date')
    cpi_category = request.form.get('cpi_category', '01')  # Default to '01' if not provided

    if data_source == 'GKS':
        obj = DataIngestion()
        train_data, test_data = obj.initiate_data_ingestion()
    else:
        obj = DataRosstat()
        train_data, test_data = obj.initiate_data_ingestion(sheet_name=cpi_category, start_date=start_date, end_date=end_date)

    data_transformation = DataTransformation()
    X_train, y_train, X_test, y_test, _ = data_transformation.initiate_data_transformation(train_data, test_data)

    plot_file_path = plot_models(models, X_train, y_train, X_test, y_test, folder_path=plot_folder)

    return render_template('show_plot.html', plot_url = url_for('static', filename=os.path.basename(plot_file_path)))

@app.route('/tune', methods = ['POST'])
def tune_hyperparameters():
    # Determine data source: GosKomStat or Rosstat
    data_source = request.form.get('data_source')
    if data_source == 'GKS':
        obj = DataIngestion()
    else:
        obj = DataRosstat()
    
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    X_train, y_train, X_test, y_test, _ = data_transformation.initiate_data_transformation(train_data, test_data)

    model_trainer = ModelTraining()
    model_trainer.initiate_model_training(X_train, y_train, X_test, y_test)

    tuning_results = {}
    for model_name, model in model_trainer.models.best_models.items():
        tuning_results[model_name] = model.get_params()
    
    json_compatible_results = {k: {param: convert_to_json(v) for param, v in params.items()} for k, params in tuning_results.items()}
    return jsonify(json_compatible_results)

if __name__ == "__main__":        
    app.run(host = '0.0.0.0', port = 5000)