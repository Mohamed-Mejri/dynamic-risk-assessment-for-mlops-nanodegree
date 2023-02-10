import os
import re
import json
import pickle
import pandas as pd
import numpy as np
from flask import Flask, session, jsonify, request
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

###################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
model_path = os.path.join(config['output_model_path']) 


#################Function for training the model
def train_model(dataset_csv_path=dataset_csv_path, model_path=model_path):
    
    #use this logistic regression for training
    lr = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                    intercept_scaling=1, l1_ratio=None, max_iter=100,
                    n_jobs=None, penalty='l2',random_state=0, 
                    solver='liblinear', tol=0.0001, verbose=0,
                    warm_start=False)
    
    #fit the logistic regression to your data
    for filename in os.listdir(dataset_csv_path):
        if re.search('.csv$', filename):
            X = pd.read_csv(os.path.join(dataset_csv_path, filename))
    y = list(X.pop('exited'))
    _ = X.pop('corporation')
    lr.fit(X,y)

    #write the trained model to your workspace in a file called trainedmodel.pkl
    path = os.path.join(model_path, 'trainedmodel.pkl')
    pickle.dump(lr, open(path, 'wb'))

if __name__ == '__main__':
    train_model()