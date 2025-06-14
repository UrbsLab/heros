import sys
import os
import pandas as pd 
import time
import argparse

" MultiSURF HPC Runner "

def main(argv):
    #ARGUMENTS:------------------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description='')
    #Script Parameters
    parser.add_argument('--d', dest='datafolder', help='path to data folder includind data subfolders ', type=str, default = 'mypath/myDataFolder') 
    parser.add_argument('--w', dest='writepath', help='path to folder where all outputs from jobs/scripts will be saved', type=str, default = 'mypath/myWritePath') 
    parser.add_argument('--o', dest='outputfolder', help='unique folder name for this analysis', type=str, default = 'myAnalysis') 
    #Algorithm Paramters
    #parser.add_argument('--t', dest='use_turf', help='indicate whether to use TuRF wrapper algorithm or not', type=bool, default = False) 
    parser.add_argument('--t', dest='use_turf', help='indicate whether to use TuRF wrapper algorithm or not', action='store_true')
    parser.add_argument('--tp', dest='turf_pct', help='indicate whether to use TuRF wrapper algorithm or not', type=float, default = 0.5) 
    #Dataset Parameters
    parser.add_argument('--cv', dest='cv_partitions', help='number of cv partitions', type=int, default = 10)
    parser.add_argument('--m', dest='max_instances', help='maximum number of instances to use in calculating Multisurf scores', type=int, default = 2000) 
    parser.add_argument('--ol', dest='outcome_label', help='outcome label (i.e. class label)', type=str, default = 'Class') 
    parser.add_argument('--il', dest='instanceID_label', help='label of instance ID column (if present)', type=str, default = 'InstanceID') 
    parser.add_argument('--el', dest='excluded_column', help='label of another column to drop (if present)', type=str, default = 'Group')
    #HPC parameters
    parser.add_argument('--rc', dest='run_cluster', help='cluster type', type=str, default='LSF')
    parser.add_argument('--rm', dest='reserved_memory', help='reserved memory for job', type=int, default= 4)
    parser.add_argument('--q', dest='queue', help='cluster queue name', type=str, default= 'i2c2_normal')
    #----------------------------------------------------------------------------------------------
    options=parser.parse_args(argv[1:])

    datafolder = options.datafolder #full path to target dataset
    writepath = options.writepath
    outputfolder = options.outputfolder
    use_turf = options.use_turf
    turf_pct = options.turf_pct
    cv_partitions = options.cv_partitions
    max_instances = options.max_instances
    outcome_label = options.outcome_label
    instanceID_label = options.instanceID_label
    excluded_column = options.excluded_column

    run_cluster = options.run_cluster
    reserved_memory = options.reserved_memory
    queue = options.queue

    algorithm = 'MultiSURF'

    #Folder Management------------------------------
    #Main Write Path-----------------
    if not os.path.exists(writepath):
        os.mkdir(writepath)  
    #Output Path--------------------
    outputPath = writepath+'/output/'
    if not os.path.exists(outputPath):
        os.mkdir(outputPath) 
    #Scratch Path-------------------- 
    scratchPath = writepath+'/scratch'
    if not os.path.exists(scratchPath):
        os.mkdir(scratchPath) 
    #LogFile Path--------------------
    logPath = writepath+'/logs'
    if not os.path.exists(logPath):
        os.mkdir(logPath) 
    outputPath = outputPath+algorithm+'_'+outputfolder
    if not os.path.exists(outputPath):
        os.mkdir(outputPath) 

    jobCount = 0
    for entry in os.listdir(datafolder): #for each subfolder within target dataset folder
        if os.path.isdir(os.path.join(datafolder,entry)):
            datapath = os.path.join(datafolder,entry)
        for i in range(1,cv_partitions+1):
            full_data_path = datapath+'/'+str(entry)+'_CV_Train_'+str(i)+'.txt'
            full_data_name = entry+'_CV_Train_'+str(i)

            if run_cluster == 'LSF':
                submit_lsf_cluster_job(scratchPath,logPath,reserved_memory,queue,full_data_name,outputPath,full_data_path,outcome_label,instanceID_label,excluded_column,max_instances,use_turf,turf_pct)
                jobCount +=1
            elif run_cluster == 'SLURM':
                submit_slurm_cluster_job(scratchPath,logPath,reserved_memory,queue,full_data_name,outputPath,full_data_path,outcome_label,instanceID_label,excluded_column,max_instances,use_turf,turf_pct)
                jobCount +=1
            else:
                print('ERROR: Cluster type not found')
    print(str(jobCount)+' jobs submitted successfully')

#UPENN - Legacy mode (using shell file) - memory on head node
def submit_lsf_cluster_job(scratchPath,logPath,reserved_memory,queue,full_data_name,outputPath,full_data_path,outcome_label,instanceID_label,excluded_column,max_instances,use_turf,turf_pct): 
    job_ref = str(time.time())
    if use_turf:
        job_name = 'MultiSURF_TuRF_'+full_data_name+'_'+job_ref
    else:
        job_name = 'MultiSURF_'+full_data_name+'_'+job_ref
    job_path = scratchPath+'/'+job_name+ '_run.sh'
    sh_file = open(job_path, 'w')
    sh_file.write('#!/bin/bash\n')
    sh_file.write('#BSUB -q ' + queue + '\n')
    sh_file.write('#BSUB -J ' + job_name + '\n')
    sh_file.write('#BSUB -R "rusage[mem=' + str(reserved_memory) + 'G]"' + '\n')
    sh_file.write('#BSUB -M ' + str(reserved_memory) + 'GB' + '\n')
    sh_file.write('#BSUB -o ' + logPath+'/'+job_name + '.o\n')
    sh_file.write('#BSUB -e ' + logPath+'/'+job_name + '.e\n')
    sh_file.write('python job_multisurf_hpc.py'+' --d '+str(full_data_path)+' --o '+str(outputPath)+' --ol '+str(outcome_label) +' --il '+str(instanceID_label) +' --el '+str(excluded_column)+' --m '+str(max_instances)+' --t '+str(use_turf)+' --tp '+str(turf_pct)+'\n')
    sh_file.close()
    os.system('bsub < ' + job_path)

def submit_slurm_cluster_job(scratchPath,logPath,reserved_memory,queue,full_data_name,outputPath,full_data_path,outcome_label,instanceID_label,excluded_column,max_instances,use_turf,turf_pct): #legacy mode just for cedars (no head node) note cedars has a different hpc - we'd need to write a method for (this is the more recent one)
    job_ref = str(time.time())
    if use_turf:
        job_name = 'MultiSURF_TuRF_'+full_data_name+'_'+job_ref
    else:
        job_name = 'MultiSURF_'+full_data_name+'_'+job_ref
    job_path = scratchPath+'/'+job_name+ '_run.sh'
    sh_file = open(job_path, 'w')
    sh_file.write('#!/bin/bash\n')
    sh_file.write('#SBATCH -p ' + queue + '\n')
    sh_file.write('#SBATCH --job-name=' + job_name + '\n')
    sh_file.write('#SBATCH --mem=' + str(reserved_memory) + 'G' + '\n')
    # sh_file.write('#BSUB -M '+str(maximum_memory)+'GB'+'\n')
    sh_file.write('#SBATCH -o ' + logPath+'/'+job_name + '.o\n')
    sh_file.write('#SBATCH -e ' + logPath+'/'+job_name + '.e\n')
    sh_file.write('srun python job_multisurf_hpc.py'+' --d '+str(full_data_path)+' --o '+str(outputPath)+' --ol '+str(outcome_label) +' --il '+str(instanceID_label) +' --el '+str(excluded_column)+' --m '+str(max_instances)+' --t '+str(use_turf)+' --tp '+str(turf_pct)+'\n')
    sh_file.close()
    os.system('sbatch ' + job_path)

if __name__=="__main__":
    sys.exit(main(sys.argv))