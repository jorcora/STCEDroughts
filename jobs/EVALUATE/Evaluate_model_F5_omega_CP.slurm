#!/bin/bash
#SBATCH --exclude=nodo17
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --job-name=Evaluate_model
#SBATCH --output=/home/jorcora/Location_Aware_AM/outs/Evaluate_model.out.%j 
 
module unload Anaconda3
module load Anaconda3/4.2.0
source activate drought_am

python -u /home/jorcora/Location_Aware_AM/code/Evaluate.py --model_name='omega_CP' --fold_id='F5' --ref_timescale='model' 
python -u /home/jorcora/Location_Aware_AM/code/Evaluate.py --model_name='omega_CP' --fold_id='F5' --ref_timescale='model' --commonMask=1
