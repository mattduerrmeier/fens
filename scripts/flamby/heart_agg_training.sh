#!/bin/bash

# Value from Table 8 p.16 for FedHeart
declare -A fed_heart_config=(
    ["client_lr"]=0.1
    ["server_lr"]=0.1
    ["batch_size"]=2
    ["local_step"]=5
    ["global_rounds"]=50
    ["num_clients"]=4
)

# TODO: what's the difference between these two values?
size="${fed_heart_config["num_clients"]}"
totalclients="${fed_heart_config["num_clients"]}"
dataset=FedHeartDiseaseAgg

seeds=(89 90 91)
# TODO: what's the difference between these two values?
proxy_ratio=0.1
trainset_fraction=0.0

optimizer=fedadam
clr="${fed_heart_config["client_lr"]}"
slr="${fed_heart_config["server_lr"]}"

local_step="${fed_heart_config["local_step"]}"
bs="${fed_heart_config["batch_size"]}"
rounds="${fed_heart_config["global_rounds"]}"

model=SmallNN_FHD # TODO: is this the correct model to use?

# root_dir: <absolute_path_to_the_root_of_this_repository>
# run from the root of this project and it will work
root_dir=$PWD
log_dir=$root_dir/results/flamby/aggregator

# env_python: <point_to_your_conda_python_interpreter>
env_python="$(which python)"

# iterate over seeds
for seed in "${seeds[@]}"; do
    logit_path="${root_dir}/results/flamby/local_training/FedHeartDisease_${seed}"

    name="${dataset}_${optimizer}_${seed}_${proxy_ratio}"
    mkdir -p "$log_dir/$name"

    # count time for experiment
    start=$(date +%s)

    # for loop over ranks until size
    for rank in $(seq 0 $((size - 1))); do
        "$env_python" "$root_dir/train_LocalSGD.py" \
            --lr "$clr" \
            --slr "$slr" \
            --bs "$bs" \
            --mu 0.0 \
            --lowE "$local_step" --highE "$local_step" \
            --momentum 0.0 \
            --gmf 0.9 \
            --numclients "$size" \
            --rank "$rank" \
            --size "$size" \
            --totalclients "$totalclients" \
            --backend gloo --initmethod tcp://localhost:23000 \
            --weights data_based \
            --disable_wandb \
            --rounds "$rounds" \
            --seed "$seed" \
            --NIID \
            --print_freq 50 \
            --save -p \
            --name "$name" \
            --dataset "$dataset" \
            --optimizer "$optimizer" \
            --model "$model" \
            --savepath "$log_dir" \
            --evalafter 100 \
            --datapath /scratch/shared/datasets \
            --proxy_set --proxy_ratio "$proxy_ratio" \
            --gpu \
            --logitpath "$logit_path" \
            --include_trainset --include_trainset_frac "$trainset_fraction" &
    done

    wait
    end=$(date +%s)
    runtime=$((end - start))
    echo "==> Time taken: $((runtime / 3600)):$(((runtime / 60) % 60)):$((runtime % 60))"
done
