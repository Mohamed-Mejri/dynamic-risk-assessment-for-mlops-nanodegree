from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os, re
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json



#################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

model_path = os.path.join(config['output_model_path']) 
test_data_path = os.path.join(config['test_data_path']) 

test_data = os.path.join(test_data_path, 'testdata.csv')

#################Function for model scoring
def score_model(model_path=model_path, test_data=test_data):
    #this function should take a trained model, load test data, and calculate an F1 score for the model relative to the test data
    #it should write the result to the latestscore.txt file

    # import model
    path = os.path.join(model_path, 'trainedmodel.pkl') 
    model = pickle.load(open(path, 'rb'))

    # import test dataset
    X_test = pd.read_csv(test_data)
    y_test = X_test.pop('exited')
    _ = X_test.pop('corporation')

    preds = model.predict(X_test)

    # Scoring
    score = metrics.f1_score(y_test, preds)
    with open(os.path.join(model_path, 'latestscore.txt'), 'w') as f:
        f.write(str(score))
    return score


if __name__ == '__main__':
    score = score_model()
    print(score)
    