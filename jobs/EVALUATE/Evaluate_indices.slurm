#!/bin/bash
#SBATCH --exclude=nodo17
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --job-name=Evaluate_indices
#SBATCH --output=/home/jorcora/Location_Aware_AM/outs/Evaluate_indices.out 
 
module load Anaconda3
source activate drought_am

python -u /home/jorcora/Location_Aware_AM/code/aMain_Evaluate.py --fold_id='F1' --ref_timescale='indices' 
python -u /home/jorcora/Location_Aware_AM/code/aMain_Evaluate.py --fold_id='F3' --ref_timescale='indices' 
python -u /home/jorcora/Location_Aware_AM/code/aMain_Evaluate.py --fold_id='F5' --ref_timescale='indices' 

python -u /home/jorcora/Location_Aware_AM/code/aMain_Evaluate.py --fold_id='F1' --ref_timescale='indices' --commonMask=1
python -u /home/jorcora/Location_Aware_AM/code/aMain_Evaluate.py --fold_id='F3' --ref_timescale='indices' --commonMask=1
python -u /home/jorcora/Location_Aware_AM/code/aMain_Evaluate.py --fold_id='F5' --ref_timescale='indices' --commonMask=1

python -u /home/jorcora/Location_Aware_AM/code/aMain_Evaluate.py --fold_id='F1' --ref_timescale='indices' --balancedAUROC=1
python -u /home/jorcora/Location_Aware_AM/code/aMain_Evaluate.py --fold_id='F3' --ref_timescale='indices' --balancedAUROC=1
python -u /home/jorcora/Location_Aware_AM/code/aMain_Evaluate.py --fold_id='F5' --ref_timescale='indices' --balancedAUROC=1
