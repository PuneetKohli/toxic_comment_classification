#!/bin/bash
#ENVIRONMENT SETTINGS; CHANGE WITH CAUTION
#SBATCH --export=NONE                #Do not propagate environment
#SBATCH --get-user-env=L             #Replicate login environment

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=weightnorm          #Set the job name to "JobExample4"
#SBATCH --time=05:00:00              #Set the wall clock limit to 14hr and 30min
#SBATCH --ntasks=20                   #Request 1 task
#SBATCH --mem=24000M                  #Request 2560MB (2.5GB) per node
#SBATCH --output=batch22.out      #Send stdout/err to "Example4Out.[jobID]"
#SBATCH --gres=gpu:2                 #Request 1 GPU per node can be 1 or 2
#SBATCH --partition=gpu              #Request the GPU partition/queue

##OPTIONAL JOB SPECIFICATIONS
#SBATCH --mail-type=ALL              #Send email on all job events
#SBATCH --mail-user=achadha7@tamu.edu

#First Executable Line
#cd CNN/My_Directory/layered_compression
ml Anaconda
source activate py3
module load CUDA/9.0.176
module load cuDNN/7.0.5-CUDA-9.0.176
runipy  PooledGRUandFastText.ipynb

