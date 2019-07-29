#!/bin/bash

mkdir -p logs

#SBATCH --job-name test_watcher
#SBATCH -e logs/test_watcher.%j.err
#SBATCH -o logs/test_watcher.%j.out


#module load psi-python36/4.4.0

#source /sf/alvra/anaconda/4.4.0/bin/activate 4.4.0

source /sf/photo/miniconda/etc/profile.d/conda.sh
conda activate dev

#conda list
#which python3


mkdir -p $(dirname $target)

time $script $options $source $target &
wait


