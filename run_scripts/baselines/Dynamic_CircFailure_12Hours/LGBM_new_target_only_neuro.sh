#!/bin/bash

#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --time=20:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --job-name=predict_new_target_only_neuro

source activate icu-benchmark

for (( i=90; i <= 105; i=i+5 ))
do
    python -m icu_benchmarks.run train \
                             -c configs/hirid/Classification/new_target_only_neuro/LGBM_map_thres${i}.gin \
                             -l logs/benchmark_exp/new_target_only_neuro/LGBM_map_thres${i}/ \
                             -t Dynamic_CircFailure_12Hours_new_target\
                             -o True \
                             --depth 3 \
                             --subsample-feat 0.33 \
                             --subsample-data 0.33 \
                             -sd  1111 2222 3333 4444 5555 6666 7777 8888 9999 0000 
done