#!/bin/bash

size=20
totalclients=20
root_dir=<absolute_path_to_the_root_of_this_repository>

env_python=<point_to_your_conda_python_interpreter>

seeds=(90)
alphas=(0.1)
proxy_ratio=0.1
train_dataset=CIFAR10
dataset=CIFAR10QAgg
datapath=<point_to_your_dataset_path>

for alpha in "${alphas[@]}"; do

    for seed in "${seeds[@]}"; do

        save_logit_dir=${root_dir}/results/logits/${train_dataset}_${alpha}_${seed}_${proxy_ratio}

        mu=0.0
        optimizer=fedadam
        log_dir=${root_dir}/results/aggregator

        clrs=(1)
        slrs=(0.001)
        local_steps=(1)

        for local_step in "${local_steps[@]}"; do

            for slr in "${slrs[@]}"; do

                for clr in "${clrs[@]}"; do

                    name=${dataset}_${optimizer}_${alpha}_${seed}
                    mkdir -p ${log_dir}/${name}

                    # count time for experiment in hh mm ss
                    start=`date +%s`

                    # for loop over ranks until size
                    for rank in $(seq 0 $((size-1))); do

                        $env_python $root_dir/train_LocalSGD.py --lr $clr --slr $slr --bs 128 --alpha $alpha --mu $mu \
                                            --lowE $local_step --highE $local_step \
                                            --momentum 0.0 --gmf 0.9 --d 2 \
                                            --numclients $size --rank $rank --size $size --totalclients $totalclients \
                                            --backend gloo --initmethod tcp://localhost:23000 \
                                            --weights data_based --disable_wandb \
                                            --rounds 500 --seed $seed --NIID --print_freq 50 \
                                            --save -p --name $name \
                                            --dataset $dataset --optimizer $optimizer --model SmallNN \
                                            --savepath $log_dir --evalafter 5 \
                                            --datapath $datapath \
                                            --proxy_set --proxy_ratio $proxy_ratio --gpu \
                                            --logitpath $save_logit_dir -sm \
                                            --procs_per_machine $size &
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

done
