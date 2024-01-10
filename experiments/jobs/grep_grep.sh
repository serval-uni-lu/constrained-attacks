#!/bin/sh

# for dataset in lcld_v2_iid url wids ctu_13_neris
for dataset in ctu_13_neris
do
    for model in tabtransformer tabnet stg torchrln vime
    do
        # ls ./cache | grep ${dataset}_${model}* | wc -l
        for training in default madry
        do
            ls ./cache | grep ${dataset}_${model}_${training}* | wc -l
        done
    done
done