from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import diagnostics 
import json
import os
from scoring import score_model


######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 

prediction_model = None


#######################Prediction Endpoint
@app.route("/prediction", methods=['POST','OPTIONS'])
def predict():        
    #call the prediction function you created in Step 3
    data = request.args.get('data')
    preds = diagnostics.model_predictions(data)
    return str(preds)

#######################Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def scoring():        
    score = score_model()
    return str(score)

#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def stats():        
    #check means, medians, and modes for each column
    summary = diagnostics.dataframe_summary()
    return str(summary)

#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def diags():        
    #check timing and percent NA values
    missing_data = diagnostics.dataframe_missing_data()
    ingestion_timing, training_timing = diagnostics.execution_time()
    outdated_pkg = diagnostics.outdated_packages_list()
    msg = "missing_data = %s \n ingestion_timing = %s \n training_timing = %s \n outdated_pkg = %s" %(missing_data, ingestion_timing, training_timing, outdated_pkg)
    return msg

if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
