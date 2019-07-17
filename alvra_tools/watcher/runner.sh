#!/bin/bash
#SBATCH --job-name test_watcher
#SBATCH -e logs/test_watcher.%j.err
#SBATCH -o logs/test_watcher.%j.out
#SBATCH --exclude sf-cn-[3-10]

module load psi-python36/4.4.0

mkdir -p $(dirname $target)
source /sf/alvra/anaconda/4.4.0/bin/activate 4.4.0
#time ../alvra_beamline_new_scripts/XES_process_file.py $source $target &
time ../alvra_beamline_new_scripts/XES_assemble_image.py $source $target &
#time ../alvra_beamline_new_scripts/XES_process_file2.py $source $target &
wait


