#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=ngstorm
#SBATCH --time=12:00:00
#SBATCH --gpus 1
#SBATCH --mem 4G
#SBATCH -c 4
#SBATCH --profile task

source /mnt/nci/scratch/nsconda39/etc/profile.d/conda.sh
conda activate /mnt/nci/scratch/nsconda39

echo "Slurm setup:"
echo "gpus    : ${SLURM_GPUS}"
echo "memory  : ${SLURM_MEM_PER_NODE} MB"
echo "cpus    : ${SLURM_CPUS_ON_NODE}"
echo "storm   : ${SLURM_JOB_NODELIST}"

cd /mnt/nci/scratch/gounleyj.nci/model_suite_v2/tests/end_to_end/ie_baseline

srun python ../../../train_model.py -m ie -args /mnt/nci/scratch/gounleyj.nci/model_suite_v2/tests/end_to_end/ie_baseline

sstat -j $SLURM_JOB_ID.batch --format=JobID,MaxVMSize,MaxRSS,AveCPU,MinCPU,NodeList

