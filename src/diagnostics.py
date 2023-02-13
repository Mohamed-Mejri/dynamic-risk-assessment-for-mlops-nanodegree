
import pandas as pd
import numpy as np
import timeit
import os
import json
import pickle
import subprocess
from ingestion import merge_multiple_dataframe
from training import train_model
from utils import create_dir, remove_all_files
from constants import TIMING_OUTPUT

##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path']) 
prod_path = os.path.join(config['prod_deployment_path']) 




##################Function to get model predictions
def model_predictions(data=None, prod_path=prod_path):
    #read the deployed model and a test dataset, calculate predictions
    if type(data) == type(None):
        data = pd.read_csv(os.path.join(test_data_path, 'testdata.csv'))
    else:
        data = pd.read_csv(data)
    model = pickle.load(open(os.path.join(prod_path, 'trainedmodel.pkl'), 'rb'))
    X = data[['lastmonth_activity','lastyear_activity','number_of_employees']]
    preds = model.predict(X)
    assert len(X) == len(preds), "ERROR: length mismatch"
    return list(preds)

##################Function to get summary statistics
def dataframe_summary(data_path=dataset_csv_path):
    
    data = pd.read_csv(os.path.join(data_path, 'finaldata.csv'))
    #calculate summary statistics here
    summary = []
    for col_name in data.columns:
        col = data[col_name]
        if col.dtype == int or col.dtype == float:
            summary.append(col.mean())
            summary.append(col.median())
            summary.append(col.std())
    return summary

def dataframe_missing_data(data_path=dataset_csv_path):
    df = pd.read_csv(os.path.join(data_path, 'finaldata.csv'))
    return list(df.isnull().sum()/len(df))

##################Function to get timings
def execution_time(tmp_dir=TIMING_OUTPUT):
    #calculate timing of training.py and ingestion.py
    starttime_ingestion = timeit.default_timer()
    merge_multiple_dataframe(config['input_folder_path'], tmp_dir)
    ingestion_timing = timeit.default_timer() - starttime_ingestion

    starttime_training = timeit.default_timer()
    train_model(dataset_csv_path, tmp_dir)
    training_timing = timeit.default_timer() - starttime_training

    return ingestion_timing, training_timing

##################Function to check dependencies
def outdated_packages_list():
    #get a list of 
    outdated = subprocess.check_output(['pip', 'list', '--outdated'])
    outdated = str(outdated).replace('wheel\\n', ',')
    outdated = outdated.replace('\\n', ',')
    outdated = outdated.split(',')[2:]
    for i in range(len(outdated)):
        outdated[i] = outdated[i].split()
    return outdated[:-1]


if __name__ == '__main__':
    create_dir(TIMING_OUTPUT)
    preds = model_predictions()
    summary = dataframe_summary()
    ingestion_timing, training_timing = execution_time()
    outdated_pkg = outdated_packages_list()
    remove_all_files(TIMING_OUTPUT)
    print(f"preds = {preds} \n summary = {summary} \n ingestion_timing = {ingestion_timing} \n training_timing = {training_timing} \n outdated_pkg = {outdated_pkg}")
