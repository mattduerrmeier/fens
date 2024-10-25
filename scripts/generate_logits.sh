#!/bin/bash

root_dir=<absolute_path_to_the_root_of_this_repository>
save_dir=${root_dir}/results/logits
mkdir -p $root_dir

env_python=<point_to_your_conda_python_interpreter>

dataset=CIFAR10
datapath=<point_to_your_dataset_path>
seeds=(90)
alphas=(0.1)
proxy_ratio=0.1


for alpha in "${alphas[@]}"; do

    for seed in "${seeds[@]}"; do

        modelpath=${root_dir}/results/local_training/${dataset}_localSGD_${alpha}_${seed}_${proxy_ratio}
        log_dir=${save_dir}/${dataset}_${alpha}_${seed}_${proxy_ratio}
        mkdir -p $log_dir

        $env_python $root_dir/generate_logits_v2.py \
            --model QResNet8 \
            --dataset $dataset \
            --datapath $datapath \
            --modelpath $modelpath \
            --save_dir $log_dir \
            --alpha $alpha \
            --seed $seed \
            --proxy_ratio $proxy_ratio \
            --batch_size 64 \
            --generate_local_testset
    
    done

done
