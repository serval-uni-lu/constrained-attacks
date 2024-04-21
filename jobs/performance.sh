#!/bin/sh
SUBSET=100000
FILTER_CLASS=1
DEVICE="cpu"

for DATASET in lcld_v2_iid url ctu_13_neris
do
    # for MODEL in deepfm
    for MODEL in tabtransformer torchrln vime
    do
        
        # for MODEL_PATH in "../models/mlc/best_models/${MODEL}_${DATASET}_default.model" "../models/mlc/best_models/${MODEL}_${DATASET}_madry.model"
        for MODEL_PATH in "../models/mlc/best_models/${MODEL}_${DATASET}_default.model"
        do
            
            eval "CUDA_VISIBLE_DEVICES=$DEVICES python run/performance.py --dataset_name $DATASET --model_name $MODEL --custom_path $MODEL_PATH  --subset $SUBSET --filter_class=$FILTER_CLASS"
        done
    done
done

