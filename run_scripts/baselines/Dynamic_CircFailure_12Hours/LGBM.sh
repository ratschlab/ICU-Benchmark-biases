#!/bin/bash

#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --time=5:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --job-name=predict_val

source activate icu-benchmark
python -m icu_benchmarks.run train \
                             -c configs/hirid/Classification/LGBM.gin \
                             -l logs/benchmark_exp/LGBM/with_val/ \
                             -t Dynamic_CircFailure_12Hours\
                             -o True \
                             --depth 4 \
                             --subsample-feat 0.33 \
                             --subsample-data 0.33 \
                             -sd  1111 2222 3333 4444 5555 6666 7777 8888 9999 0000
