import os 
import sys 
import argparse 
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
sys.path.append('/project/kamoun_shared/code_shared/scikit-heros/')
#sys.path.append('/project/kamoun_shared/code_shared/new_heros/scikit-heros/')
from src.skheros.heros import HEROS
#from skheros.heros import HEROS #PIP INSTALL RUN

def main(argv):
    #ARGUMENTS:------------------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description='')
    #Script Parameters
    parser.add_argument('--d', dest='full_data_path', help='path to target dataset ', type=str, default = 'mypath/myDataFolder/myDataset.txt') 
    parser.add_argument('--o', dest='outputPath', help='path to target output folder', type=str, default = 'mypath/myOutputFolder') 
    parser.add_argument('--ekf', dest='ekfolder', help='path to the folder containing ek scores for datasets', type=str, default = 'mypath/ekScores') 
    #Dataset Parameters
    parser.add_argument('--ol', dest='outcome_label', help='outcome label (i.e. class label)', type=str, default = 'Class') 
    parser.add_argument('--il', dest='instanceID_label', help='label of instance ID column (if present)', type=str, default = 'InstanceID')
    parser.add_argument('--el', dest='excluded_column', help='label of another column to drop (if present)', type=str, default = 'Group') 
    #Experiment Parameters
    parser.add_argument('--in', dest='model_pop_init', help='model population initialization method (Must be random, probabilistic, bootstrap, or target_acc)', type=str, default='random')
    #Critical HEROS Parameters
    parser.add_argument('--ot', dest='outcome_type', help='outcome type', type=str, default='class')
    parser.add_argument('--it', dest='iterations', help='number of rule training cycles', type=int, default=100000)
    parser.add_argument('--ps', dest='pop_size', help='maximum micro rule population size', type=int, default=1000)
    parser.add_argument('--nu', dest='nu', help='power parameter', type=int, default=1)
    parser.add_argument('--mi', dest='model_iterations', help='number of model training cycles', type=int, default=500)
    parser.add_argument('--ms', dest='model_pop_size', help='maximum model population size ', type=int, default=100)
    #Other HEROS Parameters
    parser.add_argument('--cp', dest='cross_prob', help='probability of applying crossover in rule discovery', type=float, default=0.8)
    parser.add_argument('--mp', dest='mut_prob', help='probability of applying mutation in rule discovery', type=float, default=0.04) 
    parser.add_argument('--b', dest='beta', help='learning parameter', type=float, default=0.2) 
    parser.add_argument('--ts', dest='theta_sel', help='fraction of correct set', type=float, default=0.5)   
    parser.add_argument('--ff', dest='fitness_function', help='fitness function', type=str, default='pareto')
    parser.add_argument('--s', dest='subsumption', help='subsumption strategy', type=str, default='both')
    parser.add_argument('--rsl', dest='rsl', help='rule specificity limit', type=int, default=0)   
    parser.add_argument('--ft', dest='feat_track', help='feature tracking mechanism', type=str, default='None')
    parser.add_argument('--ng', dest='new_gen', help='proportion of max model population size', type=float, default=1.0)
    parser.add_argument('--mg', dest='merge_prob', help='probability of applying merge in model discovery', type=float, default=0.1)
    parser.add_argument('--pt', dest='rule_pop_init', help='type of population initialization (load, dt, or None)', type=str, default=None)
    parser.add_argument('--c', dest='compaction', help='rule-compaction strategy', type=str, default='sub')
    parser.add_argument('--tp', dest='track_performance', help='performance tracking', type=int, default=1000)
    parser.add_argument('--sr', dest='stored_rule_iterations', help='comma-separated string indicating rule pop iterations to run full evaluation', type=str, default=None)
    parser.add_argument('--sm', dest='stored_model_iterations', help='comma-separated string indicating model pop iterations to run full evaluation', type=str, default=None)
    parser.add_argument('--rs', dest='random_state', help='random state seed', type=int, default=42)
    parser.add_argument('--v', dest='verbose', help='boolean flag to run in verbose mode', type=str, default='False')

    #----------------------------------------------------------------------------------------------
    options=parser.parse_args(argv[1:])
    #Script Parameters
    full_data_path = options.full_data_path #full path to target dataset
    outputPath = options.outputPath
    ekfolder = options.ekfolder
    #Dataset Parameters
    outcome_label = options.outcome_label
    instanceID_label = options.instanceID_label
    excluded_column = options.excluded_column
    #Experiment Parameters
    if options.model_pop_init is None or options.model_pop_init == 'None':
        model_pop_init = None
    else: 
        model_pop_init = str(options.model_pop_init)
    #Critical HEROS Parameters
    outcome_type = options.outcome_type
    iterations = options.iterations
    pop_size = options.pop_size
    nu = options.nu
    model_iterations = int(options.model_iterations)
    model_pop_size = options.model_pop_size
    #Other HEROS Parameters
    cross_prob = options.cross_prob
    mut_prob = options.mut_prob
    beta = options.beta
    theta_sel = options.theta_sel
    fitness_function = options.fitness_function
    subsumption = options.subsumption
    rsl = int(options.rsl)
    if options.feat_track is None or options.feat_track == 'None':
        feat_track = None
    else: 
        feat_track = str(options.feat_track)
    new_gen = options.new_gen
    merge_prob = options.merge_prob
    if options.rule_pop_init is None or options.rule_pop_init == 'None':
        rule_pop_init = None
    else: 
        rule_pop_init = str(options.rule_pop_init)
    if options.compaction is None or options.compaction == 'None':
        compaction = None
    else: 
        compaction = str(options.compaction)
    track_performance = options.track_performance
    stored_rule_iterations = options.stored_rule_iterations
    stored_model_iterations = options.stored_model_iterations
    if options.random_state is None or options.random_state == 'None':
        random_state = None
    else: 
        random_state = int(options.random_state)
    if options.verbose == 'True':
        verbose = True
    else:
        verbose = False

    # Get Dataset Name
    data_name = os.path.splitext(os.path.basename(full_data_path))[0]

    #Load/Process Training Dataset
    train_df = pd.read_csv(full_data_path, sep="\t")
    try:
        train_X = train_df.drop(excluded_column, axis=1) #Remove excluded column from consideration in this notebook
    except:
        train_X = train_df
        print('Excluded column not available')
    try:
        train_X = train_X.drop(instanceID_label,axis=1)
    except:
        print('Instance ID coulmn not available')
    print(train_X.columns)
    train_X = train_X.drop(outcome_label, axis=1)
    feature_names = train_X.columns # 6-bit multiplexer feature names are ['A_0','A_1','R_0', 'R_1', 'R_2','R_3']
    cat_feat_indexes = list(range(train_X.shape[1])) #assumes all feature are categorical so provide indexes 0-5 in this list for 6-bit multiplexer dataset e.g. [0,1,2,3,4,5]
    train_X = train_X.values
    train_y = train_df[outcome_label].values #outcome values
    try:
        row_id = train_df[instanceID_label].values #instance id values
    except:
        row_id = None

    #Load expert knowledge scores
    score_path_name = ekfolder+'/'+str(data_name)+'_MultiSURF_Scores.csv' #No need to change
    loaded_data = pd.read_csv(score_path_name)
    ek = loaded_data['Score'].tolist()

    #Train HEROS --------------------------------------------------------------------------------------------------------------------
    heros = HEROS(outcome_type=outcome_type,iterations=iterations, pop_size=pop_size, cross_prob=cross_prob, mut_prob=mut_prob, nu=nu, beta=beta, theta_sel=theta_sel,
                fitness_function=fitness_function,subsumption=subsumption, rsl=rsl, feat_track=feat_track, model_iterations=model_iterations,
                model_pop_size=model_pop_size, model_pop_init = model_pop_init, new_gen=new_gen, merge_prob=merge_prob, rule_pop_init=rule_pop_init, compaction=compaction,
                track_performance=track_performance,stored_rule_iterations=stored_rule_iterations,stored_model_iterations=stored_model_iterations,random_state=random_state, verbose=verbose)

    heros = heros.fit(train_X, train_y, row_id, cat_feat_indexes=cat_feat_indexes, ek=ek)

    # Save Rule Population
    pop_df = heros.get_pop()
    pop_df.to_csv(outputPath+'/rule_pop.csv', index=False)

    #Save Rule Population Performance Tracking Estimates
    tracking_df = heros.get_performance_tracking()
    tracking_df.to_csv(outputPath+'/rule_pop_tracking.csv', index=False)

    #Save Plot Rule Pop Pareto Front
    if nu == 1: #updated 3/29/24
        resolution = 500
        plot_rules = True
        color_rules = True
        heros.get_rule_pareto_landscape(resolution, heros.rule_population, plot_rules, color_rules,show=True,save=True,output_path=outputPath)

    #Save Feature Tracking Scores
    if feat_track != None:
        ft_df = heros.get_ft(feature_names)
        ft_df.to_csv(outputPath+'/feature_tracking_scores.csv', index=False)

    #Load/Process Testing Data
    test_data_path = full_data_path.replace("Train","Test")
    test_df = pd.read_csv(test_data_path, sep="\t")
    try:
        test_X = test_df.drop(excluded_column, axis=1)
    except:
        test_X = test_df
    try:
        test_X = test_X.drop(instanceID_label, axis=1)
    except:
        pass
    test_X = test_X.drop(outcome_label, axis=1)
    test_X = test_X.values
    test_y = test_df[outcome_label].values #outcome values

    #Create Performance Summary File -----------------------------------------------------------
    #Convert population evaluation iterations from comma separated strings to lists
    if stored_rule_iterations == 'None' or stored_rule_iterations is None:
        stored_rule_iterations = None
    else:
        stored_rule_iterations = [int(value) for value in stored_rule_iterations.split(',')]
    if stored_model_iterations == 'None' or stored_model_iterations is None:
        stored_model_iterations = None
    else:
        stored_model_iterations = [int(value) for value in stored_model_iterations.split(',')]

    #Initialize results dataframe information
    headers = ['train_balanced_accuracy',
                'train_tp',
                'train_fp',
                'train_tn',
                'train_fn',
                'train_coverage',
                'test_balanced_accuracy',
                'test_tp',
                'test_fp',
                'test_tn',
                'test_fn',
                'test_coverage',
                'rule_count',
                'run_time']
    
    row_indexes = []
    if stored_rule_iterations is not None:
        for iter in stored_rule_iterations:
            row_indexes.append('rule_'+str(iter))
    row_indexes.append('rule_post_compact')
    if stored_model_iterations is not None:
        for iter in stored_model_iterations:
            row_indexes.append('default_model_'+str(iter))
    row_indexes.append('default_model_'+str(model_iterations))
    if stored_model_iterations is not None:
        for iter in stored_model_iterations:
            row_indexes.append('test_selected_model_'+str(iter))
    row_indexes.append('test_selected_model_'+str(model_iterations))

    #Gather Results ----------------------------------------------------------------------
    results_list = []
    print('stored rule iterations')
    print(stored_rule_iterations)
    print(heros.rule_population.pop_set_archive.keys())
    print(heros.stored_rule_iterations)
    print('stored model iterations')
    print(stored_model_iterations)
    print(heros.model_population.pop_set_archive.keys())
    print(heros.stored_model_iterations)
    #Archived Rule Population Evaluations--------------------------
    if stored_rule_iterations is not None:
        for iter in stored_rule_iterations:
            #Training Evaluations-----
            pred_y = heros.predict(train_X,whole_rule_pop=True,rule_pop_iter=iter)
            tn, fp, fn, tp, balanced_accuracy = evaluate_stats(train_y, pred_y)
            cov_y = heros.predict_covered(train_X,whole_rule_pop=True,rule_pop_iter=iter)
            coverage = sum(cov_y)/len(cov_y)
            train_list = [balanced_accuracy,tp,fp,tn,fn,coverage]
            #Testing Evaluations-----
            pred_y = heros.predict(test_X,whole_rule_pop=True,rule_pop_iter=iter)
            tn, fp, fn, tp, balanced_accuracy = evaluate_stats(test_y, pred_y)
            cov_y = heros.predict_covered(test_X,whole_rule_pop=True,rule_pop_iter=iter)
            coverage = sum(cov_y)/len(cov_y)
            test_list = [balanced_accuracy,tp,fp,tn,fn,coverage]
            full_list = train_list + test_list
            #Other Data
            rule_count = len(heros.rule_population.pop_set_archive[iter])
            run_time = heros.timer.rule_time_archive[iter]
            #Combine into results list
            full_list = full_list + [rule_count,run_time]
            results_list.append(full_list)

    # Final (Compacted) Rule Population Evaluation
    pred_y = heros.predict(train_X,whole_rule_pop=True,rule_pop_iter=None)
    tn, fp, fn, tp, balanced_accuracy = evaluate_stats(train_y, pred_y)
    cov_y = heros.predict_covered(train_X,whole_rule_pop=True,rule_pop_iter=None)
    coverage = sum(cov_y)/len(cov_y)
    train_list = [balanced_accuracy,tp,fp,tn,fn,coverage]
    #Testing Evaluations-----
    pred_y = heros.predict(test_X,whole_rule_pop=True,rule_pop_iter=None)
    tn, fp, fn, tp, balanced_accuracy = evaluate_stats(test_y, pred_y)
    cov_y = heros.predict_covered(test_X,whole_rule_pop=True,rule_pop_iter=None)
    coverage = sum(cov_y)/len(cov_y)
    test_list = [balanced_accuracy,tp,fp,tn,fn,coverage]
    full_list = train_list + test_list
    #Other Data
    rule_count = len(heros.rule_population.pop_set)
    run_time = heros.timer.time_phase1
    #Combine into results list
    full_list = full_list + [rule_count,run_time]
    results_list.append(full_list)

    #Archived Model Population Evaluations (Default Selection)--------------------------
    if stored_model_iterations is not None:
        for iter in stored_model_iterations:
            #Training Evaluations-----
            pred_y = heros.predict(train_X,whole_rule_pop=False, target_model=0,model_pop_iter=iter)
            tn, fp, fn, tp, balanced_accuracy = evaluate_stats(train_y, pred_y)
            cov_y = heros.predict_covered(train_X,whole_rule_pop=False, target_model=0,model_pop_iter=iter)
            coverage = sum(cov_y)/len(cov_y)
            train_list = [balanced_accuracy,tp,fp,tn,fn,coverage]
            #Testing Evaluations-----
            pred_y = heros.predict(test_X,whole_rule_pop=False, target_model=0,model_pop_iter=iter)
            tn, fp, fn, tp, balanced_accuracy = evaluate_stats(test_y, pred_y)
            cov_y = heros.predict_covered(test_X,whole_rule_pop=False, target_model=0,model_pop_iter=iter)
            coverage = sum(cov_y)/len(cov_y)
            test_list = [balanced_accuracy,tp,fp,tn,fn,coverage]
            full_list = train_list + test_list
            #Other Data
            rule_count = len(heros.model_population.pop_set_archive[iter][0].rule_IDs)
            run_time = heros.timer.model_time_archive[iter]
            #Combine into results list
            full_list = full_list + [rule_count,run_time]
            results_list.append(full_list)

    # Final Model Population Evaluation (Default Selection)
    pred_y = heros.predict(train_X,whole_rule_pop=False, target_model=0,model_pop_iter=None)
    tn, fp, fn, tp, balanced_accuracy = evaluate_stats(train_y, pred_y)
    cov_y = heros.predict_covered(train_X,whole_rule_pop=False, target_model=0,model_pop_iter=None)
    coverage = sum(cov_y)/len(cov_y)
    train_list = [balanced_accuracy,tp,fp,tn,fn,coverage]
    #Testing Evaluations-----
    pred_y = heros.predict(test_X,whole_rule_pop=False, target_model=0,model_pop_iter=None)
    tn, fp, fn, tp, balanced_accuracy = evaluate_stats(test_y, pred_y)
    cov_y = heros.predict_covered(test_X,whole_rule_pop=False, target_model=0,model_pop_iter=None)
    coverage = sum(cov_y)/len(cov_y)
    test_list = [balanced_accuracy,tp,fp,tn,fn,coverage]
    full_list = train_list + test_list
    #Other Data
    rule_count = len(heros.model_population.pop_set[0].rule_IDs)
    run_time = heros.timer.time_phase1 + heros.timer.time_phase2 #total time from start to end of phase 2
    #Combine into results list
    full_list = full_list + [rule_count,run_time]
    results_list.append(full_list)

    #Archived Model Population Evaluations (Testing Data Selection)--------------------------
    if stored_model_iterations is not None:
        for iter in stored_model_iterations:
            #Identify best model on front (based on testing accuracy and coverage)
            best_model_index = get_best_testing_model_on_front(heros,test_X,test_y,iter)
            #Training Evaluations-----
            pred_y = heros.predict(train_X,whole_rule_pop=False, target_model=best_model_index,model_pop_iter=iter)
            tn, fp, fn, tp, balanced_accuracy = evaluate_stats(train_y, pred_y)
            cov_y = heros.predict_covered(train_X,whole_rule_pop=False, target_model=best_model_index,model_pop_iter=iter)
            coverage = sum(cov_y)/len(cov_y)
            train_list = [balanced_accuracy,tp,fp,tn,fn,coverage]
            #Testing Evaluations-----
            pred_y = heros.predict(test_X,whole_rule_pop=False, target_model=best_model_index,model_pop_iter=iter)
            tn, fp, fn, tp, balanced_accuracy = evaluate_stats(test_y, pred_y)
            cov_y = heros.predict_covered(test_X,whole_rule_pop=False, target_model=best_model_index,model_pop_iter=iter)
            coverage = sum(cov_y)/len(cov_y)
            test_list = [balanced_accuracy,tp,fp,tn,fn,coverage]
            full_list = train_list + test_list
            #Other Data
            rule_count = len(heros.model_population.pop_set_archive[iter][best_model_index].rule_IDs)
            run_time = heros.timer.model_time_archive[iter]
            #Combine into results list
            full_list = full_list + [rule_count,run_time]
            results_list.append(full_list)

    # Final Model Population Evaluation (Testing Data Selection)-------------------------------

    #Identify best model on front (based on testing accuracy and coverage)-----------
    model_pop_df = heros.get_model_pop()
    #Identify model indexes of all models on front
    model_on_front_indexes = []
    model_on_front_rule_count = []
    # Loop through each row
    for index, row in model_pop_df.iterrows():
        if row['Model on Front'] == 1:
            model_on_front_indexes.append(index)
            model_on_front_rule_count.append(row['Number of Rules'])
    #print(model_on_front_indexes)
    #For each one run prediction to get prediction accuracy and instance coverage on testing data
    model_accuracies = []
    model_coverages = []
    for index in model_on_front_indexes:
        #Handle class prediction and accuracy
        pred_y = heros.predict(test_X,whole_rule_pop=False, target_model=index,model_pop_iter=None)
        tn, fp, fn, tp, balanced_accuracy = evaluate_stats(test_y, pred_y)
        model_accuracies.append(balanced_accuracy)
        #Handle model coverage
        coverages = heros.predict_covered(test_X,whole_rule_pop=False, target_model=index,model_pop_iter=None)
        coverage = sum(coverages)/len(coverages) #proportion of instances covered
        model_coverages.append(coverage)
    #Identify the model index with the highest prediction accuracy
    best_accuracy = 0
    best_coverage = 0
    best_rule_count = np.inf
    best_model_index = 0
    for i in range(0,len(model_on_front_indexes)):
        if (model_accuracies[i] > best_accuracy and model_coverages[i] >= best_coverage) or (model_accuracies[i] >= best_accuracy and model_coverages[i] >= best_coverage and model_on_front_rule_count[i] < best_rule_count):
        #if model_accuracies[i] > best_accuracy and model_coverages[i] >= best_coverage:
            best_accuracy = model_accuracies[i]
            best_coverage = model_coverages[i]
            best_rule_count = model_on_front_rule_count[i]
            best_model_index = model_on_front_indexes[i]
            
    # Run evaluation for target model
    pred_y = heros.predict(train_X,whole_rule_pop=False, target_model=best_model_index,model_pop_iter=None)
    tn, fp, fn, tp, balanced_accuracy = evaluate_stats(train_y, pred_y)
    cov_y = heros.predict_covered(train_X,whole_rule_pop=False, target_model=best_model_index,model_pop_iter=None)
    coverage = sum(cov_y)/len(cov_y)
    train_list = [balanced_accuracy,tp,fp,tn,fn,coverage]
    #Testing Evaluations-----
    pred_y = heros.predict(test_X,whole_rule_pop=False, target_model=best_model_index,model_pop_iter=None)
    tn, fp, fn, tp, balanced_accuracy = evaluate_stats(test_y, pred_y)
    cov_y = heros.predict_covered(test_X,whole_rule_pop=False, target_model=best_model_index,model_pop_iter=None)
    coverage = sum(cov_y)/len(cov_y)
    test_list = [balanced_accuracy,tp,fp,tn,fn,coverage]
    full_list = train_list + test_list
    #Other Data
    rule_count = len(heros.model_population.pop_set[best_model_index].rule_IDs)
    run_time = heros.timer.time_phase1 + heros.timer.time_phase2 #total time from start to end of phase 2
    #Combine into results list
    full_list = full_list + [rule_count,run_time]
    results_list.append(full_list)

    #REPORT EVALUATION RESULTS
    results_df = pd.DataFrame(results_list, columns=headers)
    results_df['Row Indexes'] = row_indexes
    results_df.to_csv(outputPath+'/evaluation_summary.csv', index=False)

    #Save Final Model Population
    model_pop_df = heros.get_model_pop()
    model_pop_df.to_csv(outputPath+'/model_pop.csv', index=False)

    #Save Top Model (Default Selection)
    set_df = heros.get_model_rules() #returns top training model by default based on balanced accuracy, then covering, then rule-set size.
    set_df.to_csv(outputPath+'/top_default_model_rules.csv', index=False)

    #Save Top model (Testing Data Selection)
    set_df = heros.get_model_rules(best_model_index)
    set_df.to_csv(outputPath+'/top_testing_model_rules.csv', index=False)

    #Save Plot Model Pop Pareto Front
    resolution = 500
    plot_models = True
    #heros.get_model_pareto_landscape(resolution, heros.model_population, plot_models, show=True,save=True,output_path=outputPath) #original first submission
    heros.get_model_pareto_fronts(show=True,save=True,output_path=outputPath)

    #Save Plot Model Tracking
    top_models = heros.export_model_growth()
    top_models.to_csv(outputPath+'/model_tracking.csv', index=False)
    

    # Create the plot
    fig, ax1 = plt.subplots()
    # Plot the first line on the left y-axis
    ax1.plot(top_models.index, top_models["Accuracy"], 'b-', label='Model Balanced Accuracy')  # 'b-' specifies a blue solid line
    ax1.plot(top_models.index, top_models["Coverage"], 'g-', label='Model Coverage')  # 'b-' specifies a blue solid line

    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Balanced Accuracy (Blue) and Coverage (Green)')
    ax1.tick_params(axis='y')
    # Create a second y-axis sharing the same x-axis
    ax2 = ax1.twinx()
    ax2.plot(top_models.index, top_models["Number of Rules"], 'r--', label='Rules in Model')  # 'r--' specifies a red dashed line
    ax2.set_ylabel('Rules in Model', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    plt.title(f'{"Top Model Accuracy and # of Rules"} vs Iteration')
    plt.savefig(outputPath+'/model_tracking_line_graph.png', bbox_inches="tight")

    # Save Runtime Summary
    time_df = heros.get_runtimes()
    time_df.to_csv(outputPath+'/runtimes.csv', index=False)

    #Pickle Heros object
    #with open(outputPath+'/heros.pickle', 'wb') as f:
    #    pickle.dump(heros, f)


def get_best_testing_model_on_front(heros,test_X,test_y,iter):
    model_pop_df = export_model_population(heros.model_population.pop_set_archive[iter])
    #Identify model indexes of all models on front
    model_on_front_indexes = []
    model_on_front_rule_count = []
    # Loop through each row
    for index, row in model_pop_df.iterrows():
        if row['Model on Front'] == 1:
            model_on_front_indexes.append(index)
            model_on_front_rule_count.append(row['Number of Rules'])

    #For each one run prediction to get prediction accuracy and instance coverage on testing data
    model_accuracies = []
    model_coverages = []
    for index in model_on_front_indexes:
        #Handle class prediction and accuracy
        pred_y = heros.predict(test_X,whole_rule_pop=False, target_model=index,model_pop_iter=iter)
        tn, fp, fn, tp, balanced_accuracy = evaluate_stats(test_y, pred_y)
        model_accuracies.append(balanced_accuracy)
        #Handle model coverage
        coverages = heros.predict_covered(test_X,whole_rule_pop=False, target_model=index,model_pop_iter=iter)
        coverage = sum(coverages)/len(coverages) #proportion of instances covered
        model_coverages.append(coverage)

    #Identify the model index with the highest prediction accuracy
    best_accuracy = 0
    best_coverage = 0
    best_rule_count = np.inf
    best_model_index = 0
    for i in range(0,len(model_on_front_indexes)):
        if (model_accuracies[i] > best_accuracy and model_coverages[i] >= best_coverage) or (model_accuracies[i] == best_accuracy and model_coverages[i] >= best_coverage and model_on_front_rule_count[i] < best_rule_count):
        #if model_accuracies[i] > best_accuracy and model_coverages[i] >= best_coverage:
            best_accuracy = model_accuracies[i]
            best_coverage = model_coverages[i]
            best_rule_count = model_on_front_rule_count[i]
            best_model_index = model_on_front_indexes[i]
    return best_model_index

def evaluate_stats(y_true, y_pred):
    # Compute confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Calculate sensitivity and specificity
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    # Calculate balanced accuracy
    balanced_accuracy = (sensitivity + specificity) / 2
    return tn, fp, fn, tp, balanced_accuracy

def export_model_population(pop_set):
    """ Prepares and exports a dataframe capturing the rule population. """
    pop_list = []
    column_names = ['Rule IDs', 
                    'Number of Rules',
                    'Fitness', 
                    'Accuracy',
                    'Coverage', 
                    'Birth Iteration', 
                    'Deletion Probability', 
                    'Model on Front']
    for model in pop_set: 
        model_list = [str(model.rule_IDs), 
                        len(model.rule_set), 
                        model.fitness, 
                        model.accuracy,
                    model.coverage, 
                    model.birth_iteration, 
                    model.deletion_prob,
                    model.model_on_front]
        pop_list.append(model_list)
    pop_df = pd.DataFrame(pop_list, columns = column_names)
    return pop_df

if __name__=="__main__":
    sys.exit(main(sys.argv))