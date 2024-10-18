#!/bin/bash
#SBATCH --exclude=nodo[01,17-24]
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --job-name=ALTERNATIVES_F5
#SBATCH --output=/home/jorcora/Location_Aware_AM/outs/ALTERNATIVES_F5.out.%j 
 
module unload Anaconda3
module load Anaconda3/4.2.0
source activate drought_am

python -u /home/jorcora/Location_Aware_AM/code/_MAIN_.py --fold_id='F5' --eps=0.05 --SPX=1 --metric='negative_preds' --metric_eps_score='conf_score_entr_pointwise' --doPretrained=0 --doFit=1 --doTest=1 --doInference=1 --name='Umwelt_ThePublishRuns' 
