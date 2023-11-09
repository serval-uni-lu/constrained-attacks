#!/bin/sh
SUBSET=100000
FILTER_CLASS=1
DEVICE="cpu"

for DATASET in lcld_v2_iid
do
    # for MODEL in deepfm
    for MODEL in deepfm tabtransformer
    do

        # for MODEL_PATH in "../models/mlc/best_models/${MODEL}_${DATASET}_default.model" "../models/mlc/best_models/${MODEL}_${DATASET}_madry.model"
        for MODEL_PATH in "../models/mlc/best_models/${MODEL}_${DATASET}_default.model"
        do

            eval "CUDA_VISIBLE_DEVICES=$DEVICES python experiments/run/performance.py --dataset_name $DATASET --model_name $MODEL --custom_path $MODEL_PATH  --subset $SUBSET --filter_class=$FILTER_CLASS"
        done
    done
done
