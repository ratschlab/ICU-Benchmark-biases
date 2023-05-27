#!/bin/bash

#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --time=2:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --job-name=predict

source activate icu-benchmark
python -m icu_benchmarks.run train \
                             -c configs/hirid/Classification/LGBM_new_target_w_APACHE.gin \
                             -l logs/benchmark_exp/LGBM_new_target_w_APACHE/ \
                             -t Dynamic_CircFailure_12Hours_new_target\
                             -o True \
                             --depth 4 \
                             --subsample-feat 0.33 \
                             --subsample-data 0.33 \
                             -sd  1111 2222 3333 4444 5555 6666 7777 8888 9999 0000 
