#!/bin/bash

# root_dir: <absolute_path_to_the_root_of_this_repository>
# run from the root of this project and it will work
root_dir=$PWD

# env_python: <point_to_your_conda_python_interpreter>
env_python="$(which python)"

gpu_idx=0
seeds=(89 90 91)
dataset=FedHeartDisease

# 90/10 split (D1 for training, D2 for aggr)
proxy_frac=0.1
test_every=1
epochs=100

# these 6 params are used in `evaluate_all_aggregations()` (aggs.py)
# learning rate and number of epochs for the linear mapping and neural net mapping
lm_lr=1e-4
lm_epochs=200
nn_lr=5e-5
nn_epochs=200
# student VAE settings (for distillation)
distillation_lr=1e-3
distillation_epochs=10

for seed in "${seeds[@]}"; do
    save_dir="local_training/${dataset}_${seed}_epochs${epochs}"
    log_dir="$root_dir/results/$save_dir"
    mkdir -p "$log_dir"

    # count time for experiment
    start=$(date +%s)

    "$env_python" "$root_dir/flamby/main.py" \
        --dataset "$dataset" \
        --seed "$seed" \
        --gpu_idx "$gpu_idx" \
        --result_dir "$log_dir" \
        --proxy_frac "$proxy_frac" \
        --test_every "$test_every" \
        --lm_lr "$lm_lr" \
        --lm_epochs "$lm_epochs" \
        --nn_lr "$nn_lr" \
        --nn_epochs "$nn_epochs" \
        --distillation_lr "$distillation_lr" \
        --distillation_epochs "$distillation_epochs" \
        --epochs "$epochs" \
        --save_model \
        --disable_wandb

    end=$(date +%s)
    runtime=$((end - start))
    echo "==> Time taken: $((runtime / 3600)):$(((runtime / 60) % 60)):$((runtime % 60))"
done
