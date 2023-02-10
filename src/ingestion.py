import os
import json
import re
import pandas as pd
import numpy as np
from datetime import datetime







#############Function for data ingestion
def merge_multiple_dataframe(input_path, output_path):
    """Takes input and output folders' paths. 

    """
    data_list = []
    ingested_files = []
    for data_file in os.listdir(input_path):
        if re.search('.csv$', data_file):
            data_list.append(pd.read_csv(os.path.join(input_path, data_file)))
            ingested_files.append(data_file)
    if len(data_list) > 0:
        final_data = pd.DataFrame(data_list[0])
        if len(data_list) > 1:
            for df in data_list[1:]:
                final_data.append(df)
    else:
        final_data = pd.DataFrame([])

    final_data.drop_duplicates(inplace=True)
    #check for datasets, compile them together, and write to an output file
    final_data.to_csv(os.path.join(output_path, 'finaldata.csv'), index=False)

    return ingested_files



if __name__ == '__main__':
    with open('config.json','r') as f:
        config = json.load(f) 

    input_folder_path = config['input_folder_path']
    output_folder_path = config['output_folder_path']
    
    ingested_files = merge_multiple_dataframe(input_folder_path, output_folder_path)

    if len(ingested_files):
        with open(os.path.join(output_folder_path, 'ingestedfiles.txt'), 'w') as f:
            f.write(str(ingested_files))
