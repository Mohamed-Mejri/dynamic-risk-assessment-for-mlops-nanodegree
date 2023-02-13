import os
import json
import ast
import subprocess
import pickle
import training
import scoring
import deployment
import diagnostics
import reporting


with open('config.json','r') as f:
    config = json.load(f) 

prod_path = os.path.join(config["prod_deployment_path"])
input_path = os.path.join(config["input_folder_path"])
ingested_data_path = os.path.join(config["output_folder_path"])

##################Check and read new data
#first, read ingestedfiles.txt
with open(os.path.join(prod_path, 'ingestedfiles.txt'), 'r') as f:
    ingested_files = ast.literal_eval(f.read())

#second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt
new_data = False
for filename in os.listdir(input_path):
    if filename not in ingested_files:
        new_data = True
        break

if new_data:
    subprocess.run(["python", "ingestion.py"])


##################Deciding whether to proceed, part 1
#if you found new data, you should proceed. otherwise, do end the process here
else:
    exit()


##################Checking for model drift
#check whether the score from the deployed model is different from the score from the model that uses the newest ingested data
with open(os.path.join(prod_path, 'latestscore.txt'), 'r') as f:
    latest_score = float(f.read())

data_path = os.path.join(ingested_data_path, 'finaldata.csv')
preds = diagnostics.model_predictions(data=data_path)

score = scoring.score_model(test_data_path=data_path)

##################Deciding whether to proceed, part 2
#if you found model drift, you should proceed. otherwise, do end the process here

if latest_score < score:
    exit()

subprocess.run(["python", "training.py"])

##################Re-deployment
#if you found evidence for model drift, re-run the deployment.py script
subprocess.run(["python", "deployment.py"])

##################Diagnostics and reporting
#run diagnostics.py and reporting.py for the re-deployed model

subprocess.run(["python", "apicalls.py"])
