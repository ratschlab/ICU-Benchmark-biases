#!/bin/bash

#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --time=45:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --job-name=hp_search_only_neuro

source activate icu-benchmark

for (( i=1; i <= 90; ++i ))
do
    python -m icu_benchmarks.run train \
                             -c configs/hirid/Classification/LGBM_only_Neuro.gin \
                             -l logs/random_search/dynamic_circ/LGBM_only_Neuro_multiple/run \
                             -t  Dynamic_CircFailure_12Hours\
                             -rs True \
                             -sd 1111 \
                             --depth 3 4 5 6 7 \
                             --loss-weight balanced None \
                             --subsample-feat 0.33 0.66 1.00 \
                             --subsample-data 0.33 0.66 1.00
    sleep 60
done
