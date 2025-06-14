import pandas as pd
from sklearn.model_selection import StratifiedKFold
import numpy as np
import os
import argparse
import sys

def main(argv):
    #ARGUMENTS:------------------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description='')
    #Script Parameters
    parser.add_argument('--d', dest='datafile', help='name of data file (REQUIRED)', type=str, default = 'myData') #output folder name
    parser.add_argument('--o', dest='outputpath', help='directory path to write output (default=CWD)', type=str, default = 'myOutput') #full path/filename
    parser.add_argument('--n', dest='data_name', help='name of dataset', type=str, default = 'mydataset') #full path/filename

    parser.add_argument('--cv', dest='partitions', help='number of cv partitions', type=int, default= 10)
    parser.add_argument('--l', dest='outcome_label',help='outcome label', type=str, default='Class')

    options=parser.parse_args(argv[1:])

    datafile= options.datafile
    outputpath = options.outputpath
    data_name = options.data_name
    partitions = options.partitions
    outcome_label = options.outcome_label

    #Make output subfolder (to contain all random seed runs)
    if not os.path.exists(outputpath+'/'+data_name):
        os.mkdir(outputpath+'/'+data_name) 
    data_output_path = outputpath+'/'+data_name #random seed run output saved in separate output folders named based on simulated dataset

    # Load the dataset
    data = pd.read_csv(datafile, sep="\t")  # Assuming tab-delimited format
    print(data.shape)
    print(data.columns)
    data = data.dropna(axis=1, how='all')
    print(data.shape)
    if not 'InstanceID' in data.columns: #adds an instance id column if it is not already present
        data['InstanceID'] = range(len(data))
    print(data.shape)

    # Set up StratifiedKFold
    skf = StratifiedKFold(n_splits=int(partitions), shuffle=True)

    X = data.drop(columns=[outcome_label])
    print(X.shape)
    y = data[outcome_label]

    # Perform stratified splits and save partitions
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        train_data = data.iloc[train_idx]
        test_data = data.iloc[test_idx]

        # Save training and testing partitions
        train_file = os.path.join(data_output_path, f"{data_name}_CV_Train_{fold}.txt")
        test_file = os.path.join(data_output_path, f"{data_name}_CV_Test_{fold}.txt")

        train_data.to_csv(train_file, sep="\t", index=False)
        test_data.to_csv(test_file, sep="\t", index=False)

if __name__=="__main__":
    sys.exit(main(sys.argv))

