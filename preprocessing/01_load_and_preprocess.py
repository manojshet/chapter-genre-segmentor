import pandas as pd
import argparse
import configparser
import os.path
import json
# import spacy

def preProcessPlot(dataset):

    return

def correctGenresInDataset(dataset):
    for i, row in dataset.iterrows():
        row['Genres'] = list(json.loads(row['Genres']).values())
    return dataset

def readTxtFile(fileName):
    dataframe = pd.read_csv(fileName,error_bad_lines=False)
    dataframe = dataframe[dataframe['Genres'].notna()]
    dataframe = dataframe[['Genres', 'Plot_summary']]
    return dataframe

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', help='Configuration file', required=True)
    args = parser.parse_args()
    config_file = args.config_file

    config = configparser.ConfigParser()
    config.read_file(open(config_file))

    train_data_loc = config.get('01_load_and_preprocess', 'dataset_loc')
    if not os.path.isfile(train_data_loc):
        print('Please provide a valid location of the dataset.')
        exit()

    df = readTxtFile(train_data_loc)
    df = correctGenresInDataset(df)

    log_file =config.get('01_load_and_preprocess', 'pre_processed_dataset_loc')
    df.to_csv(log_file, index=False)
    print('Saved results to log file!')

