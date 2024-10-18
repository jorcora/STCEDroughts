#!/bin/bash
#SBATCH --exclude=nodo17
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --job-name=Compare_results
#SBATCH --output=/home/jorcora/Location_Aware_AM/outs/Compare_results.out.%j  
 
module unload Anaconda3
module load Anaconda3/4.2.0
source activate drought_am

python -u /home/jorcora/Location_Aware_AM/code/Compare.py --fold_id='F1' --model_name_1='alpha_Baseline' --model_name_2='omega_SPX' --ref_timescale='model' 
python -u /home/jorcora/Location_Aware_AM/code/Compare.py --fold_id='F3' --model_name_1='alpha_Baseline' --model_name_2='omega_SPX' --ref_timescale='model' 
python -u /home/jorcora/Location_Aware_AM/code/Compare.py --fold_id='F5' --model_name_1='alpha_Baseline' --model_name_2='omega_SPX' --ref_timescale='model' 
#python -u /home/jorcora/Location_Aware_AM/code/Compare.py --fold_id='FA' --ref_timescale='model' 
