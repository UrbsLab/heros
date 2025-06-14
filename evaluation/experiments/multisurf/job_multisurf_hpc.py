import os 
import sys 
import argparse 
from skrebate import MultiSURF,TURF
import pandas as pd
sys.path.append('/project/kamoun_shared/code_shared/scikit-HEROS/')


def main(argv):
    #ARGUMENTS:------------------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description='')
    #Script Parameters
    parser.add_argument('--d', dest='datapath', help='name of data file (REQUIRED)', type=str, default = 'myData') #output folder name
    parser.add_argument('--o', dest='outputPath', help='directory path to write output (default=CWD)', type=str, default = 'myOutput') #full path/filename
    #Algorithm Paramters
    parser.add_argument('--t', dest='use_turf', help='indicate whether to use TuRF wrapper algorithm or not', type=str, default ='False')
    parser.add_argument('--tp', dest='turf_pct', help='indicate whether to use TuRF wrapper algorithm or not', type=float, default = 0.5) 
    #Dataset Parameters
    parser.add_argument('--m', dest='max_instances', help='', type=int, default = 2000) #full path/filename
    parser.add_argument('--ol', dest='outcome_label', help='', type=str, default = 'Class') #full path/filename
    parser.add_argument('--il', dest='instanceID_label', help='', type=str, default = 'InstanceID') #full path/filename
    parser.add_argument('--el', dest='excluded_column', help='', type=str, default = 'Group') #full path/filename

    options = parser.parse_args(argv[1:])

    datapath = options.datapath #full path to target dataset
    outputPath = options.outputPath
    use_turf = options.use_turf
    turf_pct = options.turf_pct
    if use_turf:
        print("Using TuRF")
    else:
        print("Using MuliSURF Only")

    max_instances = options.max_instances
    outcome_label = options.outcome_label
    instanceID_label = options.instanceID_label
    excluded_column = options.excluded_column

    #Load target dataset
    train_df = pd.read_csv(datapath, sep="\t")

    #Subsample training dataset if it's larger than the maximum instances (to limit computational time)
    train_df = balanced_sampling(train_df, outcome_label, max_instances)

    # Create unique output folder for specific target dataset
    data_name = os.path.splitext(os.path.basename(datapath))[0]
    print(data_name)

    #Prepare Training Data
    try:
        X = train_df.drop(excluded_column, axis=1) #Remove excluded column from consideration in this notebook
    except:
        X = train_df
        print('Excluded column not available')
    try:
        X = X.drop(instanceID_label,axis=1)
    except:
        print('Instance ID coulmn not available')

    X = X.drop(outcome_label, axis=1)
    feature_names = X.columns 

    #Finalize separate array-like objects for X and y
    X = X.values
    y = train_df[outcome_label].values #outcome values

    if use_turf:
        score_path_name = outputPath+'/'+data_name+'_MultiSURF_TuRF_Scores.csv' 
    else:
        score_path_name = outputPath+'/'+data_name+'_MultiSURF_Scores.csv' 

    num_scores_to_return = int(X.shape[1]/2.0) # With Turf, filter down to half the original number of features. 

    if use_turf: # Run MultiSURF with TuRF wrapper
        print("Generating MultiSURF/TuRF Scores:")
        clf = TURF(MultiSURF(n_jobs=None), pct=turf_pct, num_scores_to_return=num_scores_to_return).fit(X, y)
        ek = clf.feature_importances_
        score_data = pd.DataFrame({'Feature':feature_names,'Score':ek})
        score_data.to_csv(score_path_name,index=False)

    else: #Just run MultiSURF
        print("Generating MultiSURF Scores:")
        clf = MultiSURF(n_jobs=None).fit(X, y)
        ek = clf.feature_importances_
        score_data = pd.DataFrame({'Feature':feature_names,'Score':ek})
        score_data.to_csv(score_path_name,index=False)


def balanced_sampling(df,outcome_label, max_instances):
    """ Returns a dataframe with a sampled number of rows from the original (retaining class balance if possible).
        Assumes that outcome values are either 0 or 1. """

    # Split the DataFrame by class
    df_class_0 = df[df[outcome_label] == 0]
    df_class_1 = df[df[outcome_label] == 1]

    # Determine the number of rows to sample per class
    n_class_0 = max_instances // 2
    n_class_1 = max_instances - n_class_0  # Assign remaining rows to Class 1

    # Ensure we don't sample more rows than available
    n_class_0 = min(n_class_0, len(df_class_0))
    n_class_1 = min(n_class_1, len(df_class_1))

    # Sample rows from each class
    sampled_class_0 = df_class_0.sample(n=n_class_0, replace=False)
    sampled_class_1 = df_class_1.sample(n=n_class_1, replace=False)

    # Combine the sampled rows
    sampled_df = pd.concat([sampled_class_0, sampled_class_1]).reset_index(drop=True)
    return sampled_df
    
if __name__=="__main__":
    sys.exit(main(sys.argv))
    

