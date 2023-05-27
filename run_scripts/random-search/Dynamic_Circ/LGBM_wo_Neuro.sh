#!/bin/bash

#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --time=04:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --job-name=hp_search

source activate icu-benchmark

python -m icu_benchmarks.run train \
                             -c configs/hirid/Classification/LGBM_wo_Neuro.gin \
                             -l logs/random_search/dynamic_circ/LGBM_wo_Neuro/run \
                             -t  Dynamic_CircFailure_12Hours\
                             -rs True \
                             -sd 1111 2222 3333  \
                             --depth 4  \
                             --loss-weight None \
                             --subsample-feat 0.33 \
                             --subsample-data 0.66 