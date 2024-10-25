#!/bin/bash

root_dir=<absolute_path_to_the_root_of_this_repository>

env_python=<point_to_your_conda_python_interpreter>
gpu_idx=0
dataset=FedISIC2019
seeds=(91)

for seed in "${seeds[@]}"; do

    save_dir=flamby/local_training/${dataset}_${seed}
    log_dir=$root_dir/results/$save_dir
    mkdir -p $log_dir

    # count time for experiment in hh mm ss
    start=`date +%s`

    $env_python $root_dir/flamby/main.py \
        --dataset $dataset \
        --seed $seed \
        --gpu_idx $gpu_idx \
        --result_dir $log_dir \
        --proxy_frac 0.1 \
        --test_every 1 \
        --lm_lr 1e-4 \
        --lm_epochs 100 \
        --nn_lr 5e-5 \
        --nn_epochs 200 \
        --epochs 1

    end=`date +%s`
    runtime=$((end-start))
    # Print time in hh:mm:ss
    echo "==> Time taken: $(($runtime / 3600 )):$((($runtime / 60) % 60)):$(( $runtime % 60 ))"

done