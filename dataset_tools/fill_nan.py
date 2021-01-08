import os
import numpy as np
import pandas as pd
from pathlib import Path
from common.paths import WORKING_DIR, PARENT_WORKING_DIR, DATASET_DIR, PROCESSED_DATASET_DIR


"""
Replace nan values in the dataset with interpolation. The nan values in
first and last rows are filled with 'forward fill' and 'backward fill' methods.
  
To load the arrays please use:
np.load('path to file', allow_pickle=True)
"""

def fill_nan_values(input_folder, output_folder, files, norm=True):
    """
    Replace nan values in the dataset.
    norm: (bool) Normalize dataset between [0,1]
    """
    for file in files:
        
        file_path = os.path.join(input_folder, file)
        df_table = pd.read_csv(file_path)
        df_table.interpolate(inplace=True)
        df_table.fillna(method='ffill', inplace=True)
        df_table.fillna(method='bfill', inplace=True)
        np_table = df_table.to_numpy()
        if norm:
            max_val = np_table[:,1:].max()
            min_val = np_table[:,1:].min()                      
            np_table[:,1:] = (np_table[:,1:] - min_val)/(max_val - min_val)

        new_filename = Path(file).stem
        new_filepath = os.path.join(output_folder, new_filename)
        np.save(new_filepath, np_table)
    

def main():

    if not os.path.exists(PROCESSED_DATASET_DIR):
        os.makedirs(PROCESSED_DATASET_DIR)
    
    files = ['humidity.csv', 'pressure.csv', 'temperature.csv',
             'wind_direction.csv', 'wind_speed.csv']
    
    fill_nan_values(DATASET_DIR, PROCESSED_DATASET_DIR, files)


if __name__ == "__main__":
    main()    
