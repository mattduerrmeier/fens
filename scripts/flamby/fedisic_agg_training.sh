#!/bin/bash

size=6
totalclients=6
root_dir=<absolute_path_to_the_root_of_this_repository>
log_dir=$root_dir/results/flamby/aggregator

env_python=<point_to_your_conda_python_interpreter>

seeds=(91)
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

    for local_step in "${local_steps[@]}"; do

        for slr in "${slrs[@]}"; do

            for clr in "${clrs[@]}"; do

                name=${dataset}_${optimizer}_${seed}_${proxy_ratio}
                mkdir -p $log_dir/$name

                # count time for experiment in hh mm ss
                start=`date +%s`

                # for loop over ranks until size
                for rank in $(seq 0 $((size-1))); do

                    $env_python $root_dir/train_LocalSGD.py --lr $clr --slr $slr --bs 16 --mu 0.0 \
                                        --lowE $local_step --highE $local_step \
                                        --momentum 0.0 --gmf 0.9 \
                                        --numclients $size --rank $rank --size $size --totalclients $totalclients \
                                        --backend gloo --initmethod tcp://localhost:23000 \
                                        --weights data_based --disable_wandb \
                                        --rounds 500 --seed $seed --NIID --print_freq 50 \
                                        --save -p --name $name \
                                        --dataset $dataset --optimizer $optimizer --model SmallNN \
                                        --savepath $log_dir --evalafter 100 \
                                        --datapath /scratch/shared/datasets \
                                        --proxy_set --proxy_ratio $proxy_ratio --gpu \
                                        --logitpath $logit_path \
                                        --include_trainset --include_trainset_frac $trainset_fraction &
                done

                wait

                end=`date +%s`
                runtime=$((end-start))
                # Print time in hh:mm:ss
                echo "==> Time taken: $(($runtime / 3600 )):$((($runtime / 60) % 60)):$(( $runtime % 60 ))"

            done

        done

    done

done

