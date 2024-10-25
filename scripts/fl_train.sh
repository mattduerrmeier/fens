#!/bin/bash

############# Fixed settings #############
size=20
slr=0.01
clr=0.01
mu=0.0
gmf=0.9
optimizer=fedadam
dataset=CIFAR10
datapath=<point_to_your_dataset_path>
root_dir=<absolute_path_to_the_root_of_this_repository>
save_dir=fl_training
log_dir=$root_dir/results/$save_dir

env_python=<point_to_your_conda_python_interpreter>

############# Tunable settings #############

seeds=(90)
alphas=(0.1)
le=2

############# Run experiments #############

for seed in ${seeds[@]}; do
    
    for alpha in ${alphas[@]}; do

        name=${dataset}_${optimizer}_${alpha}_${seed}
        
        mkdir -p $log_dir/$name 
        echo "==> Running $name"

        # count time for experiment in hh mm ss
        start=`date +%s`

        # for loop over ranks until size
        for rank in $(seq 0 $((size-1))); do

            $env_python $root_dir/train_LocalSGD.py --lr $clr --slr $slr --bs 16 --alpha $alpha --mu $mu \
                                --lowE $le --highE $le -iE \
                                --momentum 0.0 --gmf $gmf \
                                --numclients $size --rank $rank --size $size --totalclients $size \
                                --backend gloo --initmethod tcp://localhost:23000 \
                                --weights data_based \
                                --rounds 10 --print_freq 100 \
                                --save -p --name $name --seed $seed --NIID \
                                --dataset $dataset --optimizer $optimizer --model ResNet8 \
                                --savepath $log_dir --evalafter 5 \
                                --datapath $datapath \
                                -sm --gpu --disable_wandb \
                                --val_set --val_ratio 0.5 --procs_per_machine -1 &
        done

        wait

        end=`date +%s`
        runtime=$((end-start))
        # Print time in hh:mm:ss
        echo "==> Time taken: $(($runtime / 3600 )):$((($runtime / 60) % 60)):$(( $runtime % 60 ))"

    done
done