import os 
import sys 
import argparse 
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
sys.path.append('/project/kamoun_shared/code_shared/scikit-heros/')
from src.skheros.heros import HEROS
#from skheros.heros import HEROS #PIP INSTALL RUN

def main(argv):
    #ARGUMENTS:------------------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description='')
    #Script Parameters
    parser.add_argument('--o', dest='outputPath', help='path to target output folder', type=str, default = 'mypath/myOutputFolder') 
    #Dataset Parameters
    parser.add_argument('--ol', dest='outcome_label', help='outcome label (i.e. class label)', type=str, default = 'Class') 
    parser.add_argument('--il', dest='instanceID_label', help='label of instance ID column (if present)', type=str, default = 'InstanceID')
    parser.add_argument('--el', dest='excluded_column', help='label of another column to drop (if present)', type=str, default = 'Group') 
    #Experiment Parameters
    parser.add_argument('--cv', dest='cv_partitions', help='number of cv partitions', type=int, default = 10) 
    parser.add_argument('--r', dest='random_seeds', help='number of random seeds to run', type=int, default= 30)

    #----------------------------------------------------------------------------------------------
    options=parser.parse_args(argv[1:])
    #Script Parameters
    outputPath = options.outputPath
    #Dataset Parameters
    outcome_label = options.outcome_label
    instanceID_label = options.instanceID_label
    excluded_column = options.excluded_column
    #Experiment Parameters
    cv_partitions = options.cv_partitions
    random_seeds = options.random_seeds

    # Get Data Output Folder Name
    folder_name = outputPath.rstrip('/').split('/')[-1]

    # Create CV level summary files
    for entry in os.listdir(outputPath): #for each subfolder within target dataset folder
        if os.path.isdir(os.path.join(outputPath,entry)):
            data_level_path = os.path.join(outputPath,entry)
            for i in range(0,random_seeds):
                seed_level_path = os.path.join(data_level_path,'seed_'+str(i))
                dataframes = []
                for j in range(1,cv_partitions+1):
                    cv_level_path = os.path.join(seed_level_path,'cv_'+str(j))
                    file_path = cv_level_path+'/evaluation_summary.csv'
                    #Load Eval File
                    df = pd.read_csv(file_path)
                    df_X = df.drop('Row Indexes', axis=1)
                    row_names = df['Row Indexes']
                    dataframes.append(df_X)
                #Make averages file
                averaged_df_X = pd.concat(dataframes).groupby(level=0).mean()
                combined_df = pd.concat([row_names, averaged_df_X], axis=1)
                combined_df.to_csv(seed_level_path+'/mean_CV_evaluation_summary.csv', index=False)
                #Make standard deviation file
                sd_df_X = pd.concat(dataframes).groupby(level=0).std()
                combined_df = pd.concat([row_names, sd_df_X], axis=1)
                combined_df.to_csv(seed_level_path+'/sd_CV_evaluation_summary.csv', index=False)

    # Create seed level summary files
    for entry in os.listdir(outputPath): #for each subfolder within target dataset folder
        if os.path.isdir(os.path.join(outputPath,entry)):
            data_level_path = os.path.join(outputPath,entry)
            dataframes = []
            for i in range(0,random_seeds):
                seed_level_path = os.path.join(data_level_path,'seed_'+str(i))
                file_path = seed_level_path+'/mean_CV_evaluation_summary.csv'
                #Load Eval File
                df = pd.read_csv(file_path)
                df_X = df.drop('Row Indexes', axis=1)
                row_names = df['Row Indexes']
                dataframes.append(df_X)
            #Make averages file
            averaged_df_X = pd.concat(dataframes).groupby(level=0).mean()
            combined_df = pd.concat([row_names, averaged_df_X], axis=1)
            combined_df.to_csv(data_level_path+'/mean_seed_evaluation_summary.csv', index=False)
            #Make standard deviation file
            sd_df_X = pd.concat(dataframes).groupby(level=0).std()
            combined_df = pd.concat([row_names, sd_df_X], axis=1)
            combined_df.to_csv(data_level_path+'/sd_seed_evaluation_summary.csv', index=False)

    # Create Global results list for each evaluation point
    header = None
    for entry in os.listdir(outputPath): #for each subfolder within target dataset folder
        if os.path.isdir(os.path.join(outputPath,entry)):
            data_level_path = os.path.join(outputPath,entry)
            eval_point_list = {} #row index dictionary each with list of lists storing results list for every seed and cv combo
            for i in range(0,random_seeds):
                seed_level_path = os.path.join(data_level_path,'seed_'+str(i))
                for j in range(1,cv_partitions+1):
                    cv_level_path = os.path.join(seed_level_path,'cv_'+str(j))
                    file_path = cv_level_path+'/evaluation_summary.csv'
                    #Load Eval File
                    df = pd.read_csv(file_path)
                    header = list(df.columns)
                    header.remove('Row Indexes')
                    for index, row in df.iterrows():
                        #initialize dictionary entry
                        row_index = row['Row Indexes']
                        if row_index not in eval_point_list:
                            eval_point_list[row_index] = []
                        temp_list = list(row)
                        temp_list.remove(row_index)
                        eval_point_list[row_index].append(temp_list)
            #print(eval_point_list)
            #Create global evaluation summary for given dataset
            for each in eval_point_list:
                results = eval_point_list[each]
                df_results = pd.DataFrame(results, columns=header)
                df_results.to_csv(data_level_path+'/all_'+str(each)+'_evaluations.csv', index=False)


    # Create Global (CV average) results list for each evaluation point
    header = None
    for entry in os.listdir(outputPath): #for each subfolder within target dataset folder
        if os.path.isdir(os.path.join(outputPath,entry)):
            data_level_path = os.path.join(outputPath,entry)
            eval_point_list = {} #row index dictionary each with list of lists storing results list for every seed and cv combo
            for i in range(0,random_seeds):
                seed_level_path = os.path.join(data_level_path,'seed_'+str(i))
                file_path = seed_level_path+'/mean_CV_evaluation_summary.csv'
                #Load Eval File
                df = pd.read_csv(file_path)
                header = list(df.columns)
                header.remove('Row Indexes')
                for index, row in df.iterrows():
                    #initialize dictionary entry
                    row_index = row['Row Indexes']
                    if row_index not in eval_point_list:
                        eval_point_list[row_index] = []
                    temp_list = list(row)
                    temp_list.remove(row_index)
                    eval_point_list[row_index].append(temp_list)
            #Create global evaluation summary for given dataset
            for each in eval_point_list:
                results = eval_point_list[each]
                df_results = pd.DataFrame(results, columns=header)
                df_results.to_csv(data_level_path+'/cv_ave_'+str(each)+'_evaluations.csv', index=False)

    #Make testing accuracy boxplots (all runs)
    row_index = ['rule_500','rule_1000','rule_10000','rule_100000','rule_200000','rule_post_compact','test_selected_model_5','test_selected_model_10','test_selected_model_50','test_selected_model_100','test_selected_model_200']
    #for each in eval_point_list:
    #    row_index.append(each)
    for entry in os.listdir(outputPath): #for each subfolder within target dataset folder
        if os.path.isdir(os.path.join(outputPath,entry)):
            data_level_path = os.path.join(outputPath,entry)
            accuracy_values = []
            for each in row_index: #each eval iteration
                #Load Eval File
                file_path = data_level_path+'/all_'+str(each)+'_evaluations.csv'
                df = pd.read_csv(file_path)
                accuracy_values.append(df['test_balanced_accuracy'])
            # Create a boxplot for all 10 accuracy columns
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=accuracy_values)
            # Set the labels for the x-axis to represent each file
            plt.xticks(ticks=range(len(row_index)), labels=row_index,rotation=90)
            # Title and labels
            plt.xlabel("Evaluation Points")
            plt.ylabel("Balanced Testing Accuracy")
            plt.savefig(data_level_path+'/boxplot_testing_accuracy_all.png', bbox_inches="tight")


    #Make  rule count boxplots (all runs)
    for entry in os.listdir(outputPath): #for each subfolder within target dataset folder
        if os.path.isdir(os.path.join(outputPath,entry)):
            data_level_path = os.path.join(outputPath,entry)
            count_values = []
            for each in row_index: #each eval iteration
                #Load Eval File
                file_path = data_level_path+'/all_'+str(each)+'_evaluations.csv'
                df = pd.read_csv(file_path)
                count_values.append(df['rule_count'])
            # Create a boxplot for all 10 accuracy columns
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=count_values)
            # Set the labels for the x-axis to represent each file
            plt.xticks(ticks=range(len(row_index)), labels=row_index,rotation=90)
            # Title and labels
            plt.xlabel("Evaluation Points")
            plt.ylabel("Rule Count")
            plt.savefig(data_level_path+'/boxplot_rule_count_all.png', bbox_inches="tight")


if __name__=="__main__":
    sys.exit(main(sys.argv))