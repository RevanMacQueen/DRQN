#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --mem=20G
#SBATCH --account=def-jrwright
#SBATCH --mail-user=revan@ualberta.ca 
#SBATCH --mail-type=ALL
#SBATCH --output=plot.out

python3 generate_plots.py