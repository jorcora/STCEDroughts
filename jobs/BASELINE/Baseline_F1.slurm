#!/bin/bash
#SBATCH --exclude=nodo[01,08,17-24]
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --job-name=Baseline_F1
#SBATCH --output=/home/jorcora/Location_Aware_AM/outs/Original_UMWELT_F1.out.%j

module unload Anaconda3
module load Anaconda3/4.2.0
source activate drought_am

python -u /home/jorcora/Location_Aware_AM/code/_MAIN_.py --fold_id='F1' --doPretrained=0 --doFit=1 --doTest=1 --doInference=1 --name='Umwelt_ThePublishRuns' 
