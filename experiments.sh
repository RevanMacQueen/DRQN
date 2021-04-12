#!/bin/bash
#SBATCH --array=1-200
#SBATCH --time=02:00:00
#SBATCH --mem=6G
#SBATCH --account=def-jrwright
#SBATCH --mail-user=revan@ualberta.ca 
#SBATCH --mail-type=ALL
#SBATCH --output=experiments/%x-%j.out

eval $(head -n $SLURM_ARRAY_TASK_ID experiments.txt | tail -n 1)
