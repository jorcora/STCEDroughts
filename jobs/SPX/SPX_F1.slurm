#!/bin/bash
#SBATCH --exclude=nodo[01,17-24]
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --job-name=SPX_F1
#SBATCH --output=/home/jorcora/Location_Aware_AM/outs/SPX_F1.out.%j 

module unload Anaconda3
module load Anaconda3/4.2.0
source activate drought_am

python -u /home/jorcora/Location_Aware_AM/code/_MAIN_.py --fold_id='F1' --eps=0.05 --SPX=1 --metric_eps_score='conf_score_entr' --doPretrained=0 --doFit=0 --doTest=0 --doInference=1 --name='Umwelt_ThePublishRuns' 
