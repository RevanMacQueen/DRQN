#!/bin/bash
#SBATCH --array=1-600
#SBATCH --time=01:00:00
#SBATCH --mem=2G
#SBATCH --account=def-jrwright
#SBATCH --mail-user=revan@ualberta.ca 
#SBATCH --gres=gpu:1 
#SBATCH --mail-type=ALL
#SBATCH --output=experiments/%x-%j.out

eval $(head -n $SLURM_ARRAY_TASK_ID experiments.txt | tail -n 1)
