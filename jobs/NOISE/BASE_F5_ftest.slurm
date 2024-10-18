#!/bin/bash
#SBATCH --exclude=nodo[01,17-24]
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --job-name=BASE_F5_noise
#SBATCH --output=/home/jorcora/Location_Aware_AM/outs/BASE_F5_noise.out.%j 
 
module unload Anaconda3
module load Anaconda3/4.2.0
source activate drought_am

python -u /home/jorcora/Location_Aware_AM/code/_MAIN_.py --gen_id=7 --fold_id='F5' --eps=0.05 --SPX=0 --noise_form='slic' --noise_level=0.6 --id_process=5 --metric_eps_score='conf_score_entr' --doPretrained=0 --doFit=1 --doTest=1 --doInference=0 --name='Noise_Umwelt_ThePublishRuns' 
python -u /home/jorcora/Location_Aware_AM/code/_MAIN_.py --gen_id=8 --fold_id='F5' --eps=0.05 --SPX=0 --noise_form='slic' --noise_level=0.6 --id_process=5 --metric_eps_score='conf_score_entr' --doPretrained=0 --doFit=1 --doTest=1 --doInference=0 --name='Noise_Umwelt_ThePublishRuns' 
python -u /home/jorcora/Location_Aware_AM/code/_MAIN_.py --gen_id=9 --fold_id='F5' --eps=0.05 --SPX=0 --noise_form='slic' --noise_level=0.6 --id_process=5 --metric_eps_score='conf_score_entr' --doPretrained=0 --doFit=1 --doTest=1 --doInference=0 --name='Noise_Umwelt_ThePublishRuns' 
