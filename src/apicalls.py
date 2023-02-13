import requests
import json
import os

#Specify a URL that resolves to your workspace
URL = "http://127.0.0.1/"

with open('config.json','r') as f:
    config = json.load(f)

test_data = os.path.join(config['test_data_path'], 'testdata.csv')
output_path = os.path.join(config["output_model_path"])

#Call each API endpoint and store the responses
response1 = requests.post(f'http://127.0.0.1:8000/prediction?data={str(test_data)}').content
response2 = requests.get("http://127.0.0.1:8000/scoring").content
response3 = requests.get("http://127.0.0.1:8000/summarystats").content
response4 = requests.get("http://127.0.0.1:8000/diagnostics").content

#combine all API responses
responses = f" ******* RESPONSE 1 ******* \n {str(response1)} \n ******* RESPONSE 2 ******* \n {str(response2)} \n ******* RESPONSE 3 ******* \n  {str(response3)} \n ******* RESPONSE 4 ******* \n  {str(response4)}"

#write the responses to your workspace

with open(os.path.join(output_path, 'apireturns.txt'), 'w') as f:
    f.write(str(responses))

