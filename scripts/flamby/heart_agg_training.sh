#!/bin/bash

# Value from Table 8 p.16 for FedHeart
# All hyperparams are used in the the agg_training.sh script.
# TODO: use the tables instead
declare -A fed_heart_config=(
    ["client_lr"]=0.1
    ["server_lr"]=0.1
    ["batch_size"]=2
    ["local_step"]=5
    ["global_rounds"]=50
    ["num_clients"]=4
)

# Value from Table 8 p.16 for FedISIC2019
declare -A fed_isic_config=(
    ["client_lr"]=1.0      # clrs
    ["server_lr"]=0.001    # slrs
    ["batch_size"]=16      # bs
    ["local_step"]=1       # local_steps
    ["global_rounds"]=2500 # value not found; aggr uses 500?
    ["num_clients"]=6      # num_clients
)

# TODO: what's the difference between these two values?
size=6
totalclients=6

# root_dir: <absolute_path_to_the_root_of_this_repository>
# run from the root of this project and it will work
root_dir=$PWD
log_dir=$root_dir/results/flamby/aggregator

# env_python: <point_to_your_conda_python_interpreter>
env_python="$(which python)"

seeds=(91)
# TODO: what's the difference between these two values?
proxy_ratio=0.1
trainset_fraction=0.0
dataset=FedISIC2019Agg

# iterate over seeds
for seed in "${seeds[@]}"; do
    optimizer=fedadam
    logit_path=${root_dir}/results/flamby/local_training/FedISIC2019_${seed}

    clrs=(1)
    slrs=(1e-3)
    local_steps=(1)
    bs=16

    for local_step in "${local_steps[@]}"; do
        for slr in "${slrs[@]}"; do
            for clr in "${clrs[@]}"; do
                name=${dataset}_${optimizer}_${seed}_${proxy_ratio}
                mkdir -p $log_dir/$name

                # count time for experiment
                start=$(date +%s)

                # for loop over ranks until size
                for rank in $(seq 0 $((size - 1))); do
                    # TODO: find these parameters in the paper
                    $env_python $root_dir/train_LocalSGD.py \
                        --lr "$clr" \
                        --slr "$slr" \
                        --bs "$bs" \
                        --mu 0.0 \
                        --lowE "$local_step" --highE "$local_step" \
                        --momentum 0.0 \
                        --gmf 0.9 \
                        --numclients $size \
                        --rank $rank \
                        --size $size \
                        --totalclients $totalclients \
                        --backend gloo --initmethod tcp://localhost:23000 \
                        --weights data_based \
                        --disable_wandb \
                        --rounds 500 \
                        --seed $seed \
                        --NIID \
                        --print_freq 50 \
                        --save -p \
                        --name $name \
                        --dataset $dataset \
                        --optimizer $optimizer \
                        --model SmallNN \
                        --savepath $log_dir \
                        --evalafter 100 \
                        --datapath /scratch/shared/datasets \
                        --proxy_set --proxy_ratio $proxy_ratio \
                        --gpu \
                        --logitpath $logit_path \
                        --include_trainset --include_trainset_frac $trainset_fraction &
                done

                wait

                end=$(date +%s)
                runtime=$((end - start))
                echo "==> Time taken: $(($runtime / 3600)):$((($runtime / 60) % 60)):$(($runtime % 60))"
            done
        done
    done
done
