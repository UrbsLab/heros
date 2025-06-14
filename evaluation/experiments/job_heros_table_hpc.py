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
from scipy.stats import wilcoxon
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
    outputPath = options.outputPath #goes to the main output folder (not individual experiments)
    #Dataset Parameters
    outcome_label = options.outcome_label
    instanceID_label = options.instanceID_label
    excluded_column = options.excluded_column
    #Experiment Parameters
    cv_partitions = options.cv_partitions
    random_seeds = options.random_seeds
    p_val = 0.05

    #Make folder to save output tables. 
    table_out_path = outputPath+'/heros_paper_tables'
    if not os.path.exists(table_out_path):
        os.mkdir(table_out_path)  

    #Lay out experiment identifiers
    file_precuror = 'all' #'cv_ave' #use all individual runs, or averages across cvs to run stats. 
    phase1_checkpoints = ['500', '1000','10000','100000','200000','post_compact']
    phase2_checkpoints = ['5', '10', '50', '100','200']
    gametes_datasets = ['A_uni_4add', 'B_univariate','C_2way_epistasis','D_2way_epi_2het','E_uni_4het','F_3way_epistasis']
    mux_datasets = ['A_multiplexer_6_bit_500_inst','B_multiplexer_11_bit_5000_inst','C_multiplexer_20_bit_10000_inst','D_multiplexer_37_bit_10000_inst','E_multiplexer_70_bit_20000_inst']
    #mux_datasets = ['A_multiplexer_6_bit_500_inst','B_multiplexer_11_bit_5000_inst','C_multiplexer_20_bit_10000_inst','D_multiplexer_37_bit_10000_inst','E_multiplexer_70_bit_20000_inst','F_multiplexer_135_bit_20000_inst']
    algorithms = ['ExSTraCS','HEROS']
    #'HEROS_multiplexer_cv', 'HEROS_multiplexer_cv_acc_init', 'HEROS_multiplexer_cv_acc_init_nu10', 'HEROS_gametes_cv', 'HEROS_gametes_cv_acc_init'
    #'ExSTraCS_multiplexer_cv', 'ExSTraCS_multiplexer_cv_nu10', 'ExSTraCS_gametes_cv'

    significance_metrics = ['test_balanced_accuracy','test_coverage','rule_count','run_time']
    sig_digits = [3,3,None,None]
    sig_digits_dict = {}
    i = 0
    for metric in significance_metrics:
        sig_digits_dict[metric] = sig_digits[i]
        i+=1

    #Make Global Tables ------------------------------------------------------------------------------------------

    #All Gametes nu=1 Table ---------------------
    dataname = 'gametes_cv'
    nu_use = 1
    table_name = 'Global_'+str(dataname)+'_nu_'+str(nu_use)
    global_lists = []
    expname = 'acc_init'
    exp_names = ['HEROS(I)-Pre','HEROS(I)-Post','HEROS(II)-RI-D','HEROS(II)-RI-F','HEROS(II)-TI-D','HEROS(II)-TI-F']
    header = ['Dataset','Scenario','Test Acc.', 'Test Cover', 'Rule Count', 'Run Time']
    basename = 'ExSTraCS'
    #Automated below
    for specdataname in gametes_datasets:
        base_path = outputPath+'/ExSTraCS_'+str(dataname)+'/'+str(specdataname)+'/'+str(file_precuror)+'_rule_post_compact_evaluations'
        paths = [outputPath+'/HEROS_'+str(dataname)+'/'+str(specdataname)+'/'+str(file_precuror)+'_rule_200000_evaluations',
                outputPath+'/HEROS_'+str(dataname)+'/'+str(specdataname)+'/'+str(file_precuror)+'_rule_post_compact_evaluations',
                outputPath+'/HEROS_'+str(dataname)+'/'+str(specdataname)+'/'+str(file_precuror)+'_default_model_200_evaluations',
                outputPath+'/HEROS_'+str(dataname)+'/'+str(specdataname)+'/'+str(file_precuror)+'_test_selected_model_200_evaluations',
                outputPath+'/HEROS_'+str(dataname)+'_'+str(expname)+'/'+str(specdataname)+'/'+str(file_precuror)+'_default_model_200_evaluations',
                outputPath+'/HEROS_'+str(dataname)+'_'+str(expname)+'/'+str(specdataname)+'/'+str(file_precuror)+'_test_selected_model_200_evaluations']
        table_list = run_analysis(basename, exp_names,base_path,significance_metrics,paths,p_val,sig_digits_dict,specdataname)
        global_lists = global_lists + table_list

    table_df = pd.DataFrame(global_lists, columns=header)
    table_df.to_csv(table_out_path+'/'+str(table_name)+'_Table.csv', index=False)
    #table_df_T = table_df.T
    #table_df_T.to_csv(table_out_path+'/'+str(table_name)+'_Table.csv', index=True, header=False)

    #All Multiplexer nu=1 Table ---------------------
    dataname = 'multiplexer_cv'
    nu_use = 1
    table_name = 'Global_'+str(dataname)+'_nu_'+str(nu_use)
    global_lists = []
    expname = 'acc_init'
    exp_names = ['HEROS(I)-Pre','HEROS(I)-Post','HEROS(II)-RI-D','HEROS(II)-RI-F','HEROS(II)-TI-D','HEROS(II)-TI-F']
    header = ['Dataset','Scenario','Test Acc.', 'Test Cover', 'Rule Count', 'Run Time','Ideal Solution']
    basename = 'ExSTraCS'
    #Automated below
    for specdataname in mux_datasets:
        base_path = outputPath+'/ExSTraCS_'+str(dataname)+'/'+str(specdataname)+'/'+str(file_precuror)+'_rule_post_compact_evaluations'
        paths = [outputPath+'/HEROS_'+str(dataname)+'/'+str(specdataname)+'/'+str(file_precuror)+'_rule_200000_evaluations',
                outputPath+'/HEROS_'+str(dataname)+'/'+str(specdataname)+'/'+str(file_precuror)+'_rule_post_compact_evaluations',
                outputPath+'/HEROS_'+str(dataname)+'/'+str(specdataname)+'/'+str(file_precuror)+'_default_model_200_evaluations',
                outputPath+'/HEROS_'+str(dataname)+'/'+str(specdataname)+'/'+str(file_precuror)+'_test_selected_model_200_evaluations',
                outputPath+'/HEROS_'+str(dataname)+'_'+str(expname)+'/'+str(specdataname)+'/'+str(file_precuror)+'_default_model_200_evaluations',
                outputPath+'/HEROS_'+str(dataname)+'_'+str(expname)+'/'+str(specdataname)+'/'+str(file_precuror)+'_test_selected_model_200_evaluations']
        table_list = run_analysis_mux(basename, exp_names,base_path,significance_metrics,paths,p_val,sig_digits_dict,specdataname)
        global_lists = global_lists + table_list

    table_df = pd.DataFrame(global_lists, columns=header)
    table_df.to_csv(table_out_path+'/'+str(table_name)+'_Table.csv', index=False)

    #All Multiplexer nu=10 Table ---------------------
    dataname = 'multiplexer_cv' #+'_nu'+str(nu_use)
    nu_use = 10
    table_name = 'Global_'+str(dataname)+'_nu_'+str(nu_use)
    global_lists = []
    expname = 'acc_init'
    exp_names = ['HEROS(I)-Pre','HEROS(I)-Post','HEROS(II)-RI-D','HEROS(II)-RI-F','HEROS(II)-TI-D','HEROS(II)-TI-F']
    header = ['Dataset','Scenario','Test Acc.', 'Test Cover', 'Rule Count', 'Run Time','Ideal Solution']
    basename = 'ExSTraCS'
    #Automated below
    for specdataname in mux_datasets:
        base_path = outputPath+'/ExSTraCS_'+str(dataname)+'_nu'+str(nu_use)+'/'+str(specdataname)+'/'+str(file_precuror)+'_rule_post_compact_evaluations'
        paths = [outputPath+'/HEROS_'+str(dataname)+'_nu'+str(nu_use)+'/'+str(specdataname)+'/'+str(file_precuror)+'_rule_200000_evaluations',
                outputPath+'/HEROS_'+str(dataname)+'_nu'+str(nu_use)+'/'+str(specdataname)+'/'+str(file_precuror)+'_rule_post_compact_evaluations',
                outputPath+'/HEROS_'+str(dataname)+'_nu'+str(nu_use)+'/'+str(specdataname)+'/'+str(file_precuror)+'_default_model_200_evaluations',
                outputPath+'/HEROS_'+str(dataname)+'_nu'+str(nu_use)+'/'+str(specdataname)+'/'+str(file_precuror)+'_test_selected_model_200_evaluations',
                outputPath+'/HEROS_'+str(dataname)+'_'+expname+'_nu'+str(nu_use)+'/'+str(specdataname)+'/'+str(file_precuror)+'_default_model_200_evaluations',
                outputPath+'/HEROS_'+str(dataname)+'_'+expname+'_nu'+str(nu_use)+'/'+str(specdataname)+'/'+str(file_precuror)+'_test_selected_model_200_evaluations']
        table_list = run_analysis_mux(basename, exp_names,base_path,significance_metrics,paths,p_val,sig_digits_dict,specdataname)
        global_lists = global_lists + table_list

    table_df = pd.DataFrame(global_lists, columns=header)
    table_df.to_csv(table_out_path+'/'+str(table_name)+'_Table.csv', index=False)

    #RULE POP Performance Tracking boxplots ------------------------------------------------------------------------------------------

    custom_palette = [
        "#f0f0f0",  # very light grey
        "#FFCC99",  # light orange

        "#d3d3d3",  # light grey
        "#FFFF99",  # light yellow

        "#b0b0b0",  # light medium grey
        "#99FF99",  # light green

        "#808080",  # medium grey
        "#99CCFF",  # light blue

        "#707070",  # slightly dark grey
        "#9999FF",  # light indego
        
        "#585858",   # medium dark grey
        "#CC99FF"   # light violet
    ]

    #All Gametes nu=1 Table ---------------------
    print('generating Gametes Plot')
    dataname = 'gametes_cv'
    nu_use = 1
    target_metric = "Balanced Testing Accuracy"
    imageName = 'Phase1_Testing_Accuracy_Tracking_'+str(dataname)+'_nu_'+str(nu_use)
    subplot_count = len(phase1_checkpoints)*len(algorithms)
    # Gathering data
    data = []
    for specdataname in gametes_datasets:
        for timepoint in phase1_checkpoints:
            for algorithm in algorithms:
                #Load specific data file
                file_path = outputPath+'/'+algorithm+'_'+dataname+'/'+specdataname+'/'+str(file_precuror)+'_rule_'+timepoint+'_evaluations.csv'
                tmp_df = pd.read_csv(file_path) #load results dataset
                column_list = tmp_df['test_balanced_accuracy'].tolist()
                for val in column_list:
                    data.append({
                        "Dataset": specdataname,
                        "Time Point": timepoint,
                        "Algorithm": algorithm,
                        "Metric": val
                    })
    # Convert to DataFrame
    df = pd.DataFrame(data)
    # Create a combined column for grouping within each dataset
    df['Group'] = df['Time Point'] + " | " + df['Algorithm']
    make_boxplot(df,table_out_path,imageName,mux_datasets,phase1_checkpoints,p_val,subplot_count,algorithms,target_metric,custom_palette)

    #All Multiplexer nu=1 Table ---------------------
    print('generating MUX Plot')
    dataname = 'multiplexer_cv'
    nu_use = 1
    target_metric = "Balanced Testing Accuracy"
    imageName = 'Phase1_Testing_Accuracy_Tracking_'+str(dataname)+'_nu_'+str(nu_use)
    subplot_count = len(phase1_checkpoints)*len(algorithms)
    # Gathering data
    data = []
    for specdataname in mux_datasets:
        for timepoint in phase1_checkpoints:
            for algorithm in algorithms:
                #Load specific data file
                file_path = outputPath+'/'+algorithm+'_'+dataname+'/'+specdataname+'/'+str(file_precuror)+'_rule_'+timepoint+'_evaluations.csv'
                tmp_df = pd.read_csv(file_path) #load results dataset
                column_list = tmp_df['test_balanced_accuracy'].tolist()
                for val in column_list:
                    data.append({
                        "Dataset": specdataname,
                        "Time Point": timepoint,
                        "Algorithm": algorithm,
                        "Metric": val
                    })
    # Convert to DataFrame
    df = pd.DataFrame(data)
    # Create a combined column for grouping within each dataset
    df['Group'] = df['Time Point'] + " | " + df['Algorithm']
    make_boxplot(df,table_out_path,imageName,mux_datasets,phase1_checkpoints,p_val,subplot_count,algorithms,target_metric,custom_palette)


    #All Multiplexer nu=10 Table ---------------------
    dataname = 'multiplexer_cv'
    nu_use = 10
    target_metric = "Balanced Testing Accuracy"
    imageName = 'Phase1_Testing_Accuracy_Tracking_'+str(dataname)+'_nu_'+str(nu_use)
    subplot_count = len(phase1_checkpoints)*len(algorithms)
    # Gathering data
    data = []
    for specdataname in mux_datasets:
        for timepoint in phase1_checkpoints:
            for algorithm in algorithms:
                #Load specific data file
                file_path = outputPath+'/'+algorithm+'_'+dataname+'_nu'+str(nu_use)+'/'+specdataname+'/'+str(file_precuror)+'_rule_'+timepoint+'_evaluations.csv'
                tmp_df = pd.read_csv(file_path) #load results dataset
                column_list = tmp_df['test_balanced_accuracy'].tolist()
                for val in column_list:
                    data.append({
                        "Dataset": specdataname,
                        "Time Point": timepoint,
                        "Algorithm": algorithm,
                        "Metric": val
                    })
    # Convert to DataFrame
    df = pd.DataFrame(data)
    # Create a combined column for grouping within each dataset
    df['Group'] = df['Time Point'] + " | " + df['Algorithm']
    make_boxplot(df,table_out_path,imageName,mux_datasets,phase1_checkpoints,p_val,subplot_count,algorithms,target_metric,custom_palette)


    #Phase 2 Accuracy Performance Tracking boxplots ------------------------------------------------------------------------------------------

    custom_palette = [
        "#00ffff",  # 
        "#faebe6",  # 

        "#00bfff",  # 
        "#fcd4cd",  # 

        "#009fff",  # 
        "#febdb3",  # 

        "#0080ff",  # 
        "#fcb0a4",  # 

        "#0060ff",  # 
        "#ffa598",  # 

        "#0040ff",  # 
        "#ff9486"   # 
    ]
    #All Gametes nu=1 Table ---------------------
    dataname = 'gametes_cv'
    nu_use = 1
    target_metric = "Balanced Testing Accuracy"
    algorithm = 'HEROS'
    scenarios = ['HEROS(II)-RI-D','HEROS(II)-TI-D']
    my_dict = {
        "HEROS(II)-RI-D": '',
        "HEROS(II)-TI-D": '_acc_init'
    }
    imageName = 'Phase2_Testing_Accuracy_Tracking_'+str(dataname)+'_nu_'+str(nu_use)
    subplot_count = len(phase2_checkpoints)*len(algorithms)
    # Gathering data
    data = []
    for specdataname in gametes_datasets:
        for timepoint in phase2_checkpoints:
            for scenario in scenarios:
                #Load specific data file
                file_path = outputPath+'/'+algorithm+'_'+dataname+my_dict[scenario]+'/'+specdataname+'/'+str(file_precuror)+'_default_model_'+timepoint+'_evaluations.csv'
                tmp_df = pd.read_csv(file_path) #load results dataset
                column_list = tmp_df['test_balanced_accuracy'].tolist()
                for val in column_list:
                    data.append({
                        "Dataset": specdataname,
                        "Time Point": timepoint,
                        "Algorithm": scenario,
                        "Metric": val
                    })
    # Convert to DataFrame
    df = pd.DataFrame(data)
    # Create a combined column for grouping within each dataset
    df['Group'] = df['Time Point'] + " | " + df['Algorithm']
    make_boxplot(df,table_out_path,imageName,mux_datasets,phase1_checkpoints,p_val,subplot_count,scenarios,target_metric,custom_palette)

    #All MUX nu=1 Table ---------------------
    dataname = 'multiplexer_cv'
    nu_use = 1
    target_metric = "Balanced Testing Accuracy"
    algorithm = 'HEROS'
    scenarios = ['HEROS(II)-RI-D','HEROS(II)-TI-D']
    my_dict = {
        "HEROS(II)-RI-D": '',
        "HEROS(II)-TI-D": '_acc_init'
    }
    imageName = 'Phase2_Testing_Accuracy_Tracking_'+str(dataname)+'_nu_'+str(nu_use)
    subplot_count = len(phase2_checkpoints)*len(algorithms)
    # Gathering data
    data = []
    for specdataname in mux_datasets:
        for timepoint in phase2_checkpoints:
            for scenario in scenarios:
                #Load specific data file
                file_path = outputPath+'/'+algorithm+'_'+dataname+my_dict[scenario]+'/'+specdataname+'/'+str(file_precuror)+'_default_model_'+timepoint+'_evaluations.csv'
                tmp_df = pd.read_csv(file_path) #load results dataset
                column_list = tmp_df['test_balanced_accuracy'].tolist()
                for val in column_list:
                    data.append({
                        "Dataset": specdataname,
                        "Time Point": timepoint,
                        "Algorithm": scenario,
                        "Metric": val
                    })
    # Convert to DataFrame
    df = pd.DataFrame(data)
    # Create a combined column for grouping within each dataset
    df['Group'] = df['Time Point'] + " | " + df['Algorithm']
    make_boxplot(df,table_out_path,imageName,mux_datasets,phase1_checkpoints,p_val,subplot_count,scenarios,target_metric,custom_palette)

    #All MUX nu=10 Table ---------------------
    dataname = 'multiplexer_cv'
    nu_use = 10
    target_metric = "Balanced Testing Accuracy"
    algorithm = 'HEROS'
    scenarios = ['HEROS(II)-RI-D','HEROS(II)-TI-D']
    my_dict = {
        "HEROS(II)-RI-D": '',
        "HEROS(II)-TI-D": '_acc_init'
    }
    imageName = 'Phase2_Testing_Accuracy_Tracking_'+str(dataname)+'_nu_'+str(nu_use)
    subplot_count = len(phase2_checkpoints)*len(algorithms)
    # Gathering data
    data = []
    for specdataname in mux_datasets:
        for timepoint in phase2_checkpoints:
            for scenario in scenarios:
                #Load specific data file
                file_path = outputPath+'/'+algorithm+'_'+dataname+my_dict[scenario]+'_nu'+str(nu_use)+'/'+specdataname+'/'+str(file_precuror)+'_default_model_'+timepoint+'_evaluations.csv'
                tmp_df = pd.read_csv(file_path) #load results dataset
                column_list = tmp_df['test_balanced_accuracy'].tolist()
                for val in column_list:
                    data.append({
                        "Dataset": specdataname,
                        "Time Point": timepoint,
                        "Algorithm": scenario,
                        "Metric": val
                    })
    # Convert to DataFrame
    df = pd.DataFrame(data)
    # Create a combined column for grouping within each dataset
    df['Group'] = df['Time Point'] + " | " + df['Algorithm']
    make_boxplot(df,table_out_path,imageName,mux_datasets,phase1_checkpoints,p_val,subplot_count,scenarios,target_metric,custom_palette)

    #Phase 2 Rule Count Performance Tracking boxplots ------------------------------------------------------------------------------------------

    #All Gametes nu=1 Table ---------------------
    dataname = 'gametes_cv'
    nu_use = 1
    target_metric = "Rule Count"
    algorithm = 'HEROS'
    scenarios = ['HEROS(II)-RI-D','HEROS(II)-TI-D']
    my_dict = {
        "HEROS(II)-RI-D": '',
        "HEROS(II)-TI-D": '_acc_init'
    }
    imageName = 'Phase2_Rule_Count_Tracking_'+str(dataname)+'_nu_'+str(nu_use)
    subplot_count = len(phase2_checkpoints)*len(algorithms)
    # Gathering data
    data = []
    for specdataname in gametes_datasets:
        for timepoint in phase2_checkpoints:
            for scenario in scenarios:
                #Load specific data file
                file_path = outputPath+'/'+algorithm+'_'+dataname+my_dict[scenario]+'/'+specdataname+'/'+str(file_precuror)+'_default_model_'+timepoint+'_evaluations.csv'
                tmp_df = pd.read_csv(file_path) #load results dataset
                column_list = tmp_df['rule_count'].tolist()
                for val in column_list:
                    data.append({
                        "Dataset": specdataname,
                        "Time Point": timepoint,
                        "Algorithm": scenario,
                        "Metric": val
                    })
    # Convert to DataFrame
    df = pd.DataFrame(data)
    # Create a combined column for grouping within each dataset
    df['Group'] = df['Time Point'] + " | " + df['Algorithm']
    make_boxplot(df,table_out_path,imageName,mux_datasets,phase1_checkpoints,p_val,subplot_count,scenarios,target_metric,custom_palette)

    #All MUX nu=1 Table ---------------------
    dataname = 'multiplexer_cv'
    nu_use = 1
    target_metric = "Rule Count"
    algorithm = 'HEROS'
    scenarios = ['HEROS(II)-RI-D','HEROS(II)-TI-D']
    my_dict = {
        "HEROS(II)-RI-D": '',
        "HEROS(II)-TI-D": '_acc_init'
    }
    imageName = 'Phase2_Rule_Count_Tracking_'+str(dataname)+'_nu_'+str(nu_use)
    subplot_count = len(phase2_checkpoints)*len(algorithms)
    # Gathering data
    data = []
    for specdataname in mux_datasets:
        for timepoint in phase2_checkpoints:
            for scenario in scenarios:
                #Load specific data file
                file_path = outputPath+'/'+algorithm+'_'+dataname+my_dict[scenario]+'/'+specdataname+'/'+str(file_precuror)+'_default_model_'+timepoint+'_evaluations.csv'
                tmp_df = pd.read_csv(file_path) #load results dataset
                column_list = tmp_df['rule_count'].tolist()
                for val in column_list:
                    data.append({
                        "Dataset": specdataname,
                        "Time Point": timepoint,
                        "Algorithm": scenario,
                        "Metric": val
                    })
    # Convert to DataFrame
    df = pd.DataFrame(data)
    # Create a combined column for grouping within each dataset
    df['Group'] = df['Time Point'] + " | " + df['Algorithm']
    make_boxplot(df,table_out_path,imageName,mux_datasets,phase1_checkpoints,p_val,subplot_count,scenarios,target_metric,custom_palette)

    #All MUX nu=10 Table ---------------------
    dataname = 'multiplexer_cv'
    nu_use = 10
    target_metric = "Rule Count"
    algorithm = 'HEROS'
    scenarios = ['HEROS(II)-RI-D','HEROS(II)-TI-D']
    my_dict = {
        "HEROS(II)-RI-D": '',
        "HEROS(II)-TI-D": '_acc_init'
    }
    imageName = 'Phase2_Rule_Count_Tracking_'+str(dataname)+'_nu_'+str(nu_use)
    subplot_count = len(phase2_checkpoints)*len(algorithms)
    # Gathering data
    data = []
    for specdataname in mux_datasets:
        for timepoint in phase2_checkpoints:
            for scenario in scenarios:
                #Load specific data file
                file_path = outputPath+'/'+algorithm+'_'+dataname+my_dict[scenario]+'_nu'+str(nu_use)+'/'+specdataname+'/'+str(file_precuror)+'_default_model_'+timepoint+'_evaluations.csv'
                tmp_df = pd.read_csv(file_path) #load results dataset
                column_list = tmp_df['rule_count'].tolist()
                for val in column_list:
                    data.append({
                        "Dataset": specdataname,
                        "Time Point": timepoint,
                        "Algorithm": scenario,
                        "Metric": val
                    })
    # Convert to DataFrame
    df = pd.DataFrame(data)
    # Create a combined column for grouping within each dataset
    df['Group'] = df['Time Point'] + " | " + df['Algorithm']
    make_boxplot(df,table_out_path,imageName,mux_datasets,phase1_checkpoints,p_val,subplot_count,scenarios,target_metric,custom_palette)


def make_boxplot(df,table_out_path,imageName,datasets,timepoints,p_val,subplot_count,algorithms,target_metric,custom_palette):

    # Create the boxplot
    plt.figure(figsize=(15, 8))
    ax = sns.boxplot(
        x="Dataset",
        y="Metric",
        hue="Group",
        data=df,
        dodge=True,
        palette=sns.color_palette(custom_palette),
        showmeans=True,
        meanprops={'marker':'o',
                        'markerfacecolor':'black', 
                        'markeredgecolor':'white',
                        'markersize':'4'}
    )

    # Add a horizontal dashed line at y = 0
    ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1)

    # Perform Wilcoxon rank-sum test and annotate significant differences
    i = 0
    for dataset in datasets:
        j= 0
        for timepoint in timepoints:
            # Get the data for Algorithm 1 and Algorithm 2
            alg1_data = df[(df['Dataset'] == dataset) & (df['Time Point'] == timepoint) & (df['Algorithm'] == algorithms[0])]['Metric']
            alg2_data = df[(df['Dataset'] == dataset) & (df['Time Point'] == timepoint) & (df['Algorithm'] == algorithms[1])]['Metric']

            # Perform the Wilcoxon rank-sum test
            success = False
            try:
                statistic, p_value = wilcoxon(alg1_data, alg2_data)
                #print('Comparison: '+str(statistic)+ ' '+ str(p_value))
                success = True
            except:
                success = False
            if success and p_value < p_val:
                print('Sig Found: '+str(p_value))
                mean_alg2 = alg2_data.mean()
                ymin, ymax = ax.get_ylim()
                y_coordinate = ymin + (mean_alg2 - ymin) * (ymax - ymin) / (ymax - ymin)

                # Define the minimum and maximum values
                min_value = i - 0.4
                max_value = i + 0.4
                width = max_value - min_value
                boxwidth = width / subplot_count
                
                # Generate n equally spaced values between min_value and max_value
                values = np.linspace(min_value+(boxwidth*1.5), max_value-(boxwidth/2), 6)
                x_coordinate = values[j]
                plt.text(x_coordinate, y_coordinate, "*", ha='center', va='bottom', fontsize=10, color='red')
            j+= 1
        i += 1

    # Customize the plot
    plt.xlabel("Dataset", fontsize=12)
    plt.ylabel(target_metric, fontsize=12)
    plt.xticks(rotation=45, fontsize=10)
    plt.legend(title="Time Point | Algorithm", loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    # Save plot
    plt.savefig(table_out_path+'/'+str(imageName)+'.png', bbox_inches="tight")


def run_analysis(basename, exp_names,base_path,significance_metrics,paths,p_val,sig_digits_dict,specdataname):
    #Initialize results list
    table_list = []
    #initialize base list
    base_list = [specdataname,basename]
    #calculate Base stats (exstracs)
    base_df = pd.read_csv(base_path+'.csv') #load results dataset
    base_mean_dict = {}
    for metric in significance_metrics:
        if metric == 'run_time':
            b_mean = base_df[metric].mean()/60
            b_std = base_df[metric].std()/60
        else:
            b_mean = base_df[metric].mean()
            b_std = base_df[metric].std()
        base_list.append(str(round(b_mean,sig_digits_dict[metric]))+ ' ('+str(round(b_std,sig_digits_dict[metric]))+')')
        base_mean_dict[metric] = b_mean
    table_list.append(base_list)
    
    #Calculate all other experiment stats
    for i in range(len(paths)):
        df = pd.read_csv(paths[i]+'.csv') #load results dataset
        exp_list = [specdataname,exp_names[i]]
        for metric in significance_metrics:
            if metric == 'run_time':
                t_mean = df[metric].mean()/60
                t_std = df[metric].std()/60
            else:
                t_mean = df[metric].mean()
                t_std = df[metric].std()
            #check for significant difference from baseline
            is_sig = wilcoxon_sig(base_df[metric],df[metric],p_val)
            if is_sig: # indicate significance within dataframe stat_list
                if t_mean > base_mean_dict[metric]:
                    exp_list.append(str(round(t_mean,sig_digits_dict[metric]))+ ' ('+str(round(t_std,sig_digits_dict[metric]))+')*+')
                else:
                    exp_list.append(str(round(t_mean,sig_digits_dict[metric]))+ ' ('+str(round(t_std,sig_digits_dict[metric]))+')*-')
            else: #not significant
                exp_list.append(str(round(t_mean,sig_digits_dict[metric]))+ ' ('+str(round(t_std,sig_digits_dict[metric]))+')')
        table_list.append(exp_list)
    return table_list

def run_analysis_mux(basename, exp_names,base_path,significance_metrics,paths,p_val,sig_digits_dict,specdataname):
    #header = ['Dataset','Scenario','Test Acc.', 'Test Cover', 'Rule Count', 'Run Time','Ideal Solution']
    ideal_counts = {'A_multiplexer_6_bit_500_inst': 8,'B_multiplexer_11_bit_5000_inst': 16,'C_multiplexer_20_bit_10000_inst':32,'D_multiplexer_37_bit_10000_inst':64,'E_multiplexer_70_bit_20000_inst':128} #ideal number of rules for each MUX solution
    #ideal_counts = {'A_multiplexer_6_bit_500_inst': 8,'B_multiplexer_11_bit_5000_inst': 16,'C_multiplexer_20_bit_10000_inst':32,'D_multiplexer_37_bit_10000_inst':64,'E_multiplexer_70_bit_20000_inst':128,'F_multiplexer_135_bit_20000_inst':256} #ideal number of rules for each MUX solution
    #Initialize results list
    table_list = []
    #initialize base list
    base_list = [specdataname,basename]
    #calculate Base stats (exstracs)
    base_df = pd.read_csv(base_path+'.csv') #load results dataset
    base_mean_dict = {}
    for metric in significance_metrics:
        if metric == 'run_time':
            b_mean = base_df[metric].mean()/60
            b_std = base_df[metric].std()/60
        else:
            b_mean = base_df[metric].mean()
            b_std = base_df[metric].std()
        base_list.append(str(round(b_mean,sig_digits_dict[metric]))+ ' ('+str(round(b_std,sig_digits_dict[metric]))+')')
        base_mean_dict[metric] = b_mean
    #Identify frequency of ideal solution discovery
    ideal_count = base_df[(base_df['test_balanced_accuracy'] == 1.0) & (base_df['rule_count'] == ideal_counts[specdataname])].shape[0]
    base_list.append(str(ideal_count)+'/'+str(base_df.shape[0]))
    table_list.append(base_list)

    #Calculate all other experiment stats
    for i in range(len(paths)):
        df = pd.read_csv(paths[i]+'.csv') #load results dataset
        exp_list = [specdataname,exp_names[i]]
        for metric in significance_metrics:
            if metric == 'run_time':
                t_mean = df[metric].mean()/60
                t_std = df[metric].std()/60
            else:
                t_mean = df[metric].mean()
                t_std = df[metric].std()
            #check for significant difference from baseline
            is_sig = wilcoxon_sig(base_df[metric],df[metric],p_val)
            if is_sig: # indicate significance within dataframe stat_list
                if t_mean > base_mean_dict[metric]:
                    exp_list.append(str(round(t_mean,sig_digits_dict[metric]))+ ' ('+str(round(t_std,sig_digits_dict[metric]))+')*+')
                else:
                    exp_list.append(str(round(t_mean,sig_digits_dict[metric]))+ ' ('+str(round(t_std,sig_digits_dict[metric]))+')*-')
            else: #not significant
                exp_list.append(str(round(t_mean,sig_digits_dict[metric]))+ ' ('+str(round(t_std,sig_digits_dict[metric]))+')')
        #Identify frequency of ideal solution discovery
        ideal_count = df[(df['test_balanced_accuracy'] == 1.0) & (df['rule_count'] == ideal_counts[specdataname])].shape[0]
        exp_list.append(str(ideal_count)+'/'+str(df.shape[0]))
        table_list.append(exp_list)
    return table_list

def wilcoxon_sig(col1,col2,p_val):
    success = False
    try:
        statistic, p_value = wilcoxon(col1, col2)
        print('Comparison: '+str(statistic)+ ' '+ str(p_value))
        success = True
    except:
        success = False

    if success and p_value <= p_val:
        return True
    else:
        return False

if __name__=="__main__":
    sys.exit(main(sys.argv))