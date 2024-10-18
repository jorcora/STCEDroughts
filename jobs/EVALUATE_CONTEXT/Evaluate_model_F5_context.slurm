#!/bin/bash
#SBATCH --exclude=nodo[01,08,17-24]
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --job-name=Evaluate_model_context
#SBATCH --output=/home/jorcora/Location_Aware_AM/outs2/Evaluate_model_context.out.%j 
 
module unload Anaconda3
module load Anaconda3/4.2.0
source activate drought_am

python -u /home/jorcora/Location_Aware_AM/code/Evaluate_context.py --model_name_1='alpha_Baseline' --model_name_2='omega_SPX' --fold_id='F5' --ref_timescale='model' --commonMask=1
