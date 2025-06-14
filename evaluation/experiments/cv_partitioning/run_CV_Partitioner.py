import os
import sys
import time
import argparse

""" Creates and saves stratified CV datasets of all datasets within a target folder (assumes .txt datasets) """

def main(argv):
    #ARGUMENTS:------------------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description='')
    #Script Parameters
    parser.add_argument('--d', dest='datafolder', help='name of data file (REQUIRED)', type=str, default = 'myData') #output folder name
    parser.add_argument('--w', dest='writepath', help='', type=str, default = 'myWritePath') #full path/filename
    parser.add_argument('--o', dest='outputname', help='directory path to write output (default=CWD)', type=str, default = 'myOutput') #full path/filename

    parser.add_argument('--cv', dest='partitions', help='number of cv partitions', type=int, default= 10)
    parser.add_argument('--l',dest='outcome_label',help='outcome label', type=str, default='Class')

    parser.add_argument('--rc', dest='run_cluster', help='cluster type', type=str, default='LSF')
    parser.add_argument('--rm', dest='reserved_memory', help='reserved memory for job', type=int, default= 4)
    parser.add_argument('--q', dest='queue', help='cluster queue name', type=str, default= 'i2c2_normal')
    options=parser.parse_args(argv[1:])

    datafolder= options.datafolder
    writepath = options.writepath
    outputname = options.outputname
    partitions = options.partitions
    outcome_label = options.outcome_label
    run_cluster = options.run_cluster
    reserved_memory = options.reserved_memory
    queue = options.queue    

    #Main Write Path-----------------
    if not os.path.exists(writepath):
        os.mkdir(writepath)  
    #Output Path--------------------
    outputpath = writepath+'/'+outputname
    if not os.path.exists(outputpath):
        os.mkdir(outputpath) 
    #Scratch Path-------------------- 
    scratchPath = writepath+'/scratch'
    if not os.path.exists(scratchPath):
        os.mkdir(scratchPath) 
    #LogFile Path--------------------
    logPath = writepath+'/logs'
    if not os.path.exists(logPath):
        os.mkdir(logPath) 

    jobCount = 0
    #For each simulated dataset
    for dataname in os.listdir(datafolder):
        if os.path.isfile(os.path.join(datafolder, dataname)):
            datafile = os.path.join(datafolder, dataname)
        data_name = os.path.splitext(dataname)[0]
        if run_cluster == 'LSF':
            submit_lsf_cluster_job(datafile,data_name,outputpath,logPath,scratchPath,reserved_memory,queue,partitions,outcome_label)
        elif run_cluster == 'SLURM':
            submit_slurm_cluster_job(datafile,data_name,outputpath,logPath,scratchPath,reserved_memory,queue,partitions,outcome_label)
        else:
            print('ERROR: Cluster type not found')
        jobCount += 1
    print(str(jobCount)+' jobs submitted successfully')


#legacy mode just for cedars (no head node) note cedars has a different hpc - we'd need to write a method for (this is the more recent one)
def submit_slurm_cluster_job(datafile,data_name,outputpath,logPath,scratchPath,reserved_memory,queue,partitions,outcome_label): 
    job_ref = str(time.time())
    job_name = 'DataCV_'+job_ref
    job_path = scratchPath+'/'+job_name+ '_run.sh'
    sh_file = open(job_path, 'w')
    sh_file.write('#!/bin/bash\n')
    sh_file.write('#SBATCH -p ' + queue + '\n')
    sh_file.write('#SBATCH --job-name=' + job_name + '\n')
    sh_file.write('#SBATCH --mem=' + str(reserved_memory) + 'G' + '\n')
    sh_file.write('#SBATCH -o ' + logPath+'/'+job_name + '.o\n')
    sh_file.write('#SBATCH -e ' + logPath+'/'+job_name + '.e\n')
    sh_file.write('srun python job_CV_Partitioner.py'+' --d '+str(datafile)+' --o '+str(outputpath)+' --n '+str(data_name)+' --cv '+str(partitions)+' --l '+str(outcome_label)+ '\n')
    sh_file.close()
    os.system('sbatch ' + job_path)


#UPENN - Legacy mode (using shell file) - memory on head node
def submit_lsf_cluster_job(datafile,data_name,outputpath,logPath,scratchPath,reserved_memory,queue,partitions,outcome_label): 
    job_ref = str(time.time())
    job_name = 'DataCV_'+job_ref
    job_path = scratchPath+'/'+job_name+ '_run.sh'
    sh_file = open(job_path, 'w')
    sh_file.write('#!/bin/bash\n')
    sh_file.write('#BSUB -q ' + queue + '\n')
    sh_file.write('#BSUB -J ' + job_name + '\n')
    sh_file.write('#BSUB -R "rusage[mem=' + str(reserved_memory) + 'G]"' + '\n')
    sh_file.write('#BSUB -M ' + str(reserved_memory) + 'GB' + '\n')
    sh_file.write('#BSUB -o ' + logPath+'/'+job_name + '.o\n')
    sh_file.write('#BSUB -e ' + logPath+'/'+job_name + '.e\n')
    sh_file.write('python job_CV_Partitioner.py'+' --d '+str(datafile)+' --o '+str(outputpath)+' --n '+str(data_name)+' --cv '+str(partitions)+' --l '+str(outcome_label)+ '\n')
    sh_file.close()
    os.system('bsub < ' + job_path)

if __name__=="__main__":
    sys.exit(main(sys.argv))