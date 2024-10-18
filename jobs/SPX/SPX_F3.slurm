#!/bin/bash
#SBATCH --exclude=nodo[01,17-24]
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --job-name=SPX_F3
#SBATCH --output=/home/jorcora/Location_Aware_AM/outs/SPX_F3.out.%j 
 
module unload Anaconda3
module load Anaconda3/4.2.0
source activate drought_am

python -u /home/jorcora/Location_Aware_AM/code/_MAIN_.py --fold_id='F3' --eps=0.05 --SPX=1 --metric_eps_score='conf_score_entr' --doPretrained=0 --doFit=1 --doTest=1 --doInference=0 --name='Umwelt_ThePublishRuns' 
