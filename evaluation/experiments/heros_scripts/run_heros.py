import sys
import os
import pandas as pd 
#import matplotlib.pyplot as plt
#import numpy as np
#import seaborn as sns
#import scipy.stats as scs
#import random
#from sklearn.preprocessing import StandardScaler
#from sklearn.experimental import enable_iterative_imputer
#from sklearn.impute import IterativeImputer
#import csv
import time
import argparse

from sklearn.model_selection import train_test_split

" HEROS Run "

def main(argv):
    #ARGUMENTS:------------------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description='')
    #Script Parameters
    parser.add_argument('--d', dest='datafolder', help='path to data folder includind data subfolders ', type=str, default = 'mypath/myDataFolder') 
    parser.add_argument('--w', dest='writepath', help='path to folder where all outputs from jobs/scripts will be saved', type=str, default = 'mypath/myWritePath') 
    parser.add_argument('--o', dest='outputfolder', help='unique folder name for this analysis', type=str, default = 'myAnalysis') 
    parser.add_argument('--ekf', dest='ekfolder', help='path to the folder containing ek scores for datasets', type=str, default = 'mypath/ekScores') 
    #Dataset Parameters
    parser.add_argument('--ol', dest='outcome_label', help='outcome label (i.e. class label)', type=str, default = 'Class') 
    parser.add_argument('--il', dest='instanceID_label', help='label of instance ID column (if present)', type=str, default = 'InstanceID')
    parser.add_argument('--el', dest='excluded_column', help='label of another column to drop (if present)', type=str, default = 'Group') 
    #Experiment Parameters
    parser.add_argument('--cv', dest='cv_partitions', help='number of cv partitions', type=int, default = 10) 
    parser.add_argument('--r', dest='random_seeds', help='number of random seeds to run', type=int, default= 30)
    parser.add_argument('--pt', dest='rule_pop_init', help='type of population initialization (load, dt, dt_bstrap, or None)', type=str, default='None')
    parser.add_argument('--a', dest='alternate', help='phase alternations', type = int, default = 0)
    parser.add_argument('--m', dest='alternate_mode', help='alternation mode (Must be "limit", "converge", "equal", or None)', type = str, default='None')
    #Unused Experimental
    parser.add_argument('--fb', dest='feedback', action='store_true', help='use alternation weights')
    #Critical HEROS Parameters
    parser.add_argument('--ot', dest='outcome_type', help='outcome type', type=str, default='class')
    parser.add_argument('--it', dest='iterations', help='number of rule training cycles', type=int, default=100000)
    parser.add_argument('--ps', dest='pop_size', help='maximum micro rule population size', type=int, default=1000)
    parser.add_argument('--nu', dest='nu', help='power parameter', type=int, default=1)
    parser.add_argument('--mi', dest='model_iterations', help='number of model training cycles', type=int, default=500)
    parser.add_argument('--ms', dest='model_pop_size', help='maximum model population size ', type=int, default=100)
    parser.add_argument('--in', dest='model_pop_init', help='model population initialization method (Must be random, probabilistic, bootstrap, or target_acc)', type=str, default='random')
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
    parser.add_argument('--c', dest='compaction', help='rule-compaction strategy', type=str, default='sub')
    parser.add_argument('--tp', dest='track_performance', help='performance tracking', type=int, default=1000)
    parser.add_argument('--sr', dest='stored_rule_iterations', help='comma-separated string indicating rule pop iterations to run full evaluation', type=str, default='None')
    parser.add_argument('--sm', dest='stored_model_iterations', help='comma-separated string indicating model pop iterations to run full evaluation', type=str, default='None')
    parser.add_argument('--rs', dest='random_state', help='random state seed', type=int, default=42)
    #parser.add_argument('--v', dest='verbose', help='boolean flag to run in verbose mode', type=bool, default=False)
    parser.add_argument('--v', dest='verbose', help='boolean flag to run in verbose mode', action='store_true')
    parser.add_argument('--min', dest='min_output', help='boolean flag to minimize saved HEROS output', action='store_true')
    parser.add_argument('--opt', dest='optimization_method', help='quantitative feature optimization method', type=str, default='None')
                        
    #HPC parameters
    parser.add_argument('--rc', dest='run_cluster', help='cluster type', type=str, default='SLURM')
    parser.add_argument('--rm', dest='reserved_memory', help='reserved memory for job', type=int, default= 4)
    #parser.add_argument('--q', dest='queue', help='cluster queue name', type=str, default= 'i2c2_normal') #ONLY FOR UPENN
    parser.add_argument('--p', dest='partition', help='slurm partition name', type=str, default= 'defq')
    #parser.add_argument('--check', dest='check', help='boolean flag to check and report on what jobs have not yet completed', type=bool, action= False)
    parser.add_argument('--check', dest='check', help='boolean flag to check and report on what jobs have not yet completed', action='store_true')
    #parser.add_argument('--resub', dest='resubmit', help='boolean flag to resubmit incomplete jobs', type=bool, default= False)
    parser.add_argument('--resub', dest='resubmit', help='boolean flag to resubmit incomplete jobs', action='store_true')
    #----------------------------------------------------------------------------------------------
    options=parser.parse_args(argv[1:])
    print(options)
    #Script Parameters
    datafolder = options.datafolder #full path to target dataset
    writepath = options.writepath
    print(writepath)
    outputfolder = options.outputfolder
    ekfolder = options.ekfolder
    #Dataset Parameters
    outcome_label = options.outcome_label
    instanceID_label = options.instanceID_label
    excluded_column = options.excluded_column
    #Experiment Parameters
    cv_partitions = options.cv_partitions
    random_seeds = options.random_seeds
    alternate = options.alternate
    alternate_mode = options.alternate_mode
    feedback = options.feedback

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
    optimization_method = options.optimization_method

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
    verbose = options.verbose
    if options.model_pop_init is None or options.model_pop_init == 'None':
        model_pop_init = None
    else: 
        model_pop_init = str(options.model_pop_init)
    min_output = options.min_output
    #HPC parameters
    run_cluster = options.run_cluster
    reserved_memory = options.reserved_memory
    queue = options.queue
    check = options.check
    resubmit = options.resubmit
    algorithm = 'HEROS'
    #Folder Management------------------------------
    
    #Main Write Path-----------------
    if not os.path.exists(writepath):
        os.mkdir(writepath)  
    #Output Path--------------------
    base_output_path_0 = writepath+'/output/'
    if not os.path.exists(base_output_path_0):
        os.mkdir(base_output_path_0) 
    #Scratch Path-------------------- 
    scratchPath = writepath+'/scratch'
    if not os.path.exists(scratchPath):
        os.mkdir(scratchPath) 
    #LogFile Path--------------------
    logPath = writepath+'/logs'
    if not os.path.exists(logPath):
        os.mkdir(logPath) 
    base_output_path_0 = base_output_path_0+algorithm+'_'+outputfolder
    if not os.path.exists(base_output_path_0):
        os.mkdir(base_output_path_0) 

    # Experiment loop to submit jobs (datasets, random seeds, cv partitions)
    jobCount = 0
    missing_count = 0
    jobCount = 0

    # datafolder = FULL PATH to a single dataset file
    full_dataset_path = datafolder
    dataset_name = os.path.splitext(os.path.basename(full_dataset_path))[0]

    base_output_path_1 = os.path.join(base_output_path_0, dataset_name)
    if not os.path.exists(base_output_path_1):
        os.mkdir(base_output_path_1)

    df = pd.read_csv(full_dataset_path, sep=None, engine='python')

    for i in range(random_seeds):
        target_random_seed = i if random_seeds > 1 else random_state

        try:
            X = df.drop(excluded_column, axis=1)
        except:
            X = df.copy()
            print('Excluded column not available')

        try:
            X = X.drop(instanceID_label, axis=1)
        except:
            print('Instance ID column not available')

        X = X.drop(outcome_label, axis=1)
        y = df[outcome_label]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            random_state=target_random_seed,
            test_size=0.25,
            shuffle=True
        )

        train_df = pd.concat([X_train, y_train], axis=1)
        test_df  = pd.concat([X_test, y_test], axis=1)

        base_output_path_2 = os.path.join(base_output_path_1, f'seed_{i}')
        if not os.path.exists(base_output_path_2):
            os.mkdir(base_output_path_2)

        train_file = os.path.join(base_output_path_2, f'{dataset_name}_Train.txt')
        test_file  = os.path.join(base_output_path_2, f'{dataset_name}_Test.txt')

        train_df.to_csv(train_file, sep='	', index=False)
        test_df.to_csv(test_file, sep='	', index=False)

        outputPath = os.path.join(base_output_path_2, 'run_1')
        if not os.path.exists(outputPath):
            os.mkdir(outputPath)

        full_data_name = f'{dataset_name}_Train'

        if run_cluster == 'LSF':
            submit_lsf_cluster_job(
                scratchPath, logPath, reserved_memory, queue,
                full_data_name, outputPath, train_file, ekfolder,
                outcome_label, instanceID_label, excluded_column,
                model_pop_init, outcome_type, iterations, pop_size, nu,
                model_iterations, model_pop_size, cross_prob, mut_prob,
                beta, theta_sel, fitness_function, subsumption, rsl,
                feat_track, new_gen, merge_prob, rule_pop_init,
                compaction, track_performance, stored_rule_iterations,
                stored_model_iterations, target_random_seed, verbose,
                min_output, alternate, alternate_mode, feedback
            )
            jobCount += 1

        elif run_cluster == 'SLURM':
            submit_slurm_cluster_job(
                scratchPath, logPath, reserved_memory, queue,
                full_data_name, outputPath, train_file, ekfolder,
                outcome_label, instanceID_label, excluded_column,
                model_pop_init, outcome_type, iterations, pop_size, nu,
                model_iterations, model_pop_size, cross_prob, mut_prob,
                beta, theta_sel, fitness_function, subsumption, rsl,
                feat_track, new_gen, merge_prob, rule_pop_init,
                compaction, track_performance, stored_rule_iterations,
                stored_model_iterations, target_random_seed, verbose,
                min_output, alternate, alternate_mode, feedback, optimization_method
            )
            jobCount += 1

        else:
            print('ERROR: Cluster type not found')


    print(str(jobCount)+' jobs submitted successfully')
    if check:
        print(str(missing_count)+' jobs incomplete')

#UPenn Cluster
def submit_lsf_cluster_job(scratchPath,logPath,reserved_memory,queue,full_data_name,outputPath,full_data_path,ekfolder,outcome_label,instanceID_label,excluded_column,model_pop_init,outcome_type,iterations,pop_size,nu,model_iterations,model_pop_size,cross_prob,mut_prob,beta,theta_sel,fitness_function,subsumption,rsl,feat_track,new_gen,merge_prob,rule_pop_init,compaction,track_performance,stored_rule_iterations,stored_model_iterations,target_random_seed,verbose,min_output,alternate,alternate_mode,feedback): 
    job_ref = str(time.time())
    job_name = 'HEROS_'+full_data_name+'_seed_'+str(target_random_seed)+'_'+job_ref
    job_path = scratchPath+'/'+job_name+ '_run.sh'
    sh_file = open(job_path, 'w')
    sh_file.write('#!/bin/bash\n')
    sh_file.write('#BSUB -q ' + queue + '\n')
    sh_file.write('#BSUB -J ' + job_name + '\n')
    sh_file.write('#BSUB -R "rusage[mem=' + str(reserved_memory) + 'G]"' + '\n')
    sh_file.write('#BSUB -M ' + str(reserved_memory) + 'GB' + '\n')
    sh_file.write('#BSUB -o ' + logPath+'/'+job_name + '.o\n')
    sh_file.write('#BSUB -e ' + logPath+'/'+job_name + '.e\n')
    fb_arg = ' --fb' if feedback else ''
    verbose_arg = ' --v' if verbose else ''
    min_arg = ' --min' if min_output else ''
    sh_file.write('python job_heros.py'+' --d '+str(full_data_path)+' --o '+str(outputPath)+' --ekf '+str(ekfolder)+' --ol '+str(outcome_label) +' --il '+str(instanceID_label) +' --el '+str(excluded_column) + ' --in '+str(model_pop_init)+' --ot '+str(outcome_type)+' --it '+str(iterations)+' --ps '+str(pop_size)+' --nu '+str(nu)+' --mi '+str(model_iterations)+' --ms '+str(model_pop_size)+' --cp '+str(cross_prob)+' --mp '+str(mut_prob)+' --b '+str(beta)+' --ts '+str(theta_sel)+' --ff '+str(fitness_function)+' --s '+str(subsumption)+' --rsl '+str(rsl)+' --ft '+str(feat_track)+' --ng '+str(new_gen)+' --mg '+str(merge_prob)+' --pt '+str(rule_pop_init)+' --c '+str(compaction)+' --tp '+str(track_performance)+' --sr '+str(stored_rule_iterations)+' --sm '+str(stored_model_iterations)+' --rs '+str(target_random_seed)+ verbose_arg + min_arg+ ' --a ' + str(alternate)+ ' --m ' + str(alternate_mode) + fb_arg +'\n')
    sh_file.close()
    os.system('bsub < ' + job_path)
    
#Cedars Cluster
def submit_slurm_cluster_job(scratchPath,logPath,reserved_memory,queue,full_data_name,outputPath,full_data_path,ekfolder,outcome_label,instanceID_label,excluded_column,model_pop_init,outcome_type,iterations,pop_size,nu,model_iterations,model_pop_size,cross_prob,mut_prob,beta,theta_sel,fitness_function,subsumption,rsl,feat_track,new_gen,merge_prob,rule_pop_init,compaction,track_performance,stored_rule_iterations,stored_model_iterations,target_random_seed,verbose,min_output,alternate,alternate_mode,feedback, optimization_method): 
    job_ref = str(time.time())
    job_name = 'HEROS_'+full_data_name+'_seed_'+str(target_random_seed)+'_'+job_ref
    job_path = scratchPath+'/'+job_name+ '_run.sh'
    sh_file = open(job_path, 'w')
    sh_file.write('#!/bin/bash\n')
    sh_file.write('#SBATCH -p ' + queue + '\n')
    sh_file.write('#SBATCH --job-name=' + job_name + '\n')
    sh_file.write('#SBATCH --mem=' + str(reserved_memory) + 'G' + '\n')
    # sh_file.write('#BSUB -M '+str(maximum_memory)+'GB'+'\n')
    sh_file.write('#SBATCH -o ' + logPath+'/'+job_name + '.o\n')
    sh_file.write('#SBATCH -e ' + logPath+'/'+job_name + '.e\n')
    fb_arg = ' --fb' if feedback else ''
    verbose_arg = ' --v' if verbose else ''
    min_arg = ' --min' if min_output else ''
    sh_file.write('srun python job_heros.py'+' --d '+str(full_data_path)+' --o '+str(outputPath)+' --ekf '+str(ekfolder)+' --ol '+str(outcome_label) +' --il '+str(instanceID_label) +' --el '+str(excluded_column)+' --in '+str(model_pop_init)+' --ot '+str(outcome_type)+' --it '+str(iterations)+' --ps '+str(pop_size)+' --nu '+str(nu)+' --mi '+str(model_iterations)+' --ms '+str(model_pop_size)+' --cp '+str(cross_prob)+' --mp '+str(mut_prob)+' --b '+str(beta)+' --ts '+str(theta_sel)+' --ff '+str(fitness_function)+' --s '+str(subsumption)+' --rsl '+str(rsl)+' --ft '+str(feat_track)+' --ng '+str(new_gen)+' --mg '+str(merge_prob)+' --pt '+str(rule_pop_init)+' --c '+str(compaction)+' --tp '+str(track_performance)+' --sr '+str(stored_rule_iterations)+' --sm '+str(stored_model_iterations)+' --rs '+str(target_random_seed)+ verbose_arg + min_arg + ' --a ' + str(alternate)+ ' --m ' + str(alternate_mode) + ' --OPT ' + str(optimization_method) + fb_arg +'\n')
    sh_file.close()
    os.system('sbatch ' + job_path)

if __name__=="__main__":
    sys.exit(main(sys.argv))