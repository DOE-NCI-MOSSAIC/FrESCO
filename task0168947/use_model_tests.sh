#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=ngstorm
#SBATCH --time=12:00:00
#SBATCH --gpus 1
#SBATCH --mem 4G
#SBATCH -c 4
#SBATCH --profile task

source /mnt/nci/scratch/nsconda39b/etc/profile.d/conda.sh
conda activate /mnt/nci/scratch/nsconda39b

echo "Slurm setup:"
echo "gpus    : ${SLURM_GPUS}"
echo "memory  : ${SLURM_MEM_PER_NODE} MB"
echo "cpus    : ${SLURM_CPUS_ON_NODE}"
echo "storm   : ${SLURM_JOB_NODELIST}"


srun python use_model.py  -mp ./savedmodels/base_hisan.h5 -dp /mnt/nci/scratch/spannausa/Shared/testdata/LA 
srun python use_model.py  -mp ./savedmodels/dac_hisan.h5 -dp /mnt/nci/scratch/spannausa/Shared/testdata/LA 
srun python use_model.py  -mp ./savedmodels/ntask_hisan.h5 -dp /mnt/nci/scratch/spannausa/Shared/testdata/LA 
srun python use_model.py  -mp ./savedmodels/base_cnn.h5 -dp /mnt/nci/scratch/spannausa/Shared/testdata/LA 
srun python use_model.py  -mp ./savedmodels/dac_cnn.h5 -dp /mnt/nci/scratch/spannausa/Shared/testdata/LA 
srun python use_model.py  -mp ./savedmodels/ntask_cnn.h5 -dp /mnt/nci/scratch/spannausa/Shared/testdata/LA 

sstat -j $SLURM_JOB_ID.batch --format=JobID,MaxVMSize,MaxRSS,AveCPU,MinCPU,NodeList

