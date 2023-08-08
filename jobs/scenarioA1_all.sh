#!/bin/sh

DEVICES=0
SUBSET=1000
BATCH_SIZE=1024
MAX_EPS=0.5
ATTACK="pgdl2+apgd+fab+moeva+caa"
FILTER_CLASS=1
DEVICE="cpu"

for DATASET in url ctu_13_neris lcld_v2_iid malware wids
do
  for MODEL in tabtransformer deepfm torchrln vime
  do
    MODEL_PATH="../models/mlc/best_models/${MODEL}_${DATASET}_default.model"
    CUDA_VISIBLE_DEVICES=$DEVICES python run/scenarioA1.py --dataset_name $DATASET --model_name $MODEL --custom_path $MODEL_PATH --attacks_name $ATTACK --max_eps $MAX_EPS --subset $SUBSET --batch_size $BATCH_SIZE --device $DEVICE
    CUDA_VISIBLE_DEVICES=$DEVICES python run/scenarioA1.py --dataset_name $DATASET --model_name $MODEL --custom_path $MODEL_PATH --attacks_name $ATTACK --max_eps $MAX_EPS --subset $SUBSET --batch_size $BATCH_SIZE --device $DEVICE --filter_class=$FILTER_CLASS

    MODEL_PATH="../models/mlc/best_models/${MODEL}_${DATASET}_madry.model"
    CUDA_VISIBLE_DEVICES=$DEVICES python run/scenarioA1.py --dataset_name $DATASET --model_name $MODEL --custom_path $MODEL_PATH --attacks_name $ATTACK --max_eps $MAX_EPS --subset $SUBSET --batch_size $BATCH_SIZE --device $DEVICE
    CUDA_VISIBLE_DEVICES=$DEVICES python run/scenarioA1.py --dataset_name $DATASET --model_name $MODEL --custom_path $MODEL_PATH --attacks_name $ATTACK --max_eps $MAX_EPS --subset $SUBSET --batch_size $BATCH_SIZE --device $DEVICE --filter_class=$FILTER_CLASS

  done
done

