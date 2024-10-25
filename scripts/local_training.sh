#!/bin/bash

size=20
clr=0.0025
slr=1.0
mu=0.0
momentum=0.0
gmf=0.0
proxy_ratio=0.1
optimizer=localSGD
dataset=CIFAR10
datapath=<point_to_your_dataset_path>

root_dir=<absolute_path_to_the_root_of_this_repository>
save_dir=local_training
log_dir=$root_dir/results/$save_dir
mkdir -p $log_dir/$name 

env_python=<point_to_your_conda_python_interpreter>

alphas=(0.1)
seeds=(90)
le=2
global_rounds=1

for alpha in ${alphas[@]}; do
    
    for seed in ${seeds[@]}; do

        name=${dataset}_${optimizer}_${alpha}_${seed}_${proxy_ratio}

        mkdir -p $log_dir/$name 
        echo "==> Running $name"

        # count time for experiment in hh mm ss
        start=`date +%s`

        # for loop over ranks until size
        for rank in $(seq 0 $((size-1))); do

            $env_python $root_dir/train_LocalSGD.py --lr $clr --slr $slr --bs 16 --alpha $alpha --mu $mu \
                                --lowE $le --highE $le -iE --use_scheduler \
                                --momentum $momentum --gmf $gmf \
                                --numclients $size --rank $rank --size $size --totalclients $size \
                                --backend gloo --initmethod tcp://localhost:23000 \
                                --weights data_based --diff_init \
                                --rounds $global_rounds --seed $seed --NIID --print_freq 100 \
                                --save -p --name $name --disable_wandb \
                                --dataset $dataset --optimizer $optimizer --model ResNet8 \
                                --savepath $log_dir --evalafter 5 --test_bs 16 \
                                --datapath $datapath \
                                --proxy_set --proxy_ratio $proxy_ratio \
                                --val_set --val_ratio 0.5 -sm -g --procs_per_machine -1 &  
        done

        wait

        end=`date +%s`
        runtime=$((end-start))
        # Print time in hh:mm:ss
        echo "==> Time taken: $(($runtime / 3600 )):$((($runtime / 60) % 60)):$(( $runtime % 60 ))"

    done
done