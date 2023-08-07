#!/bin/sh

DEVICES=0
SUBSET=10000
BATCH_SIZE=1024
MAX_EPS=0.5
DATASET="${1:-url}"
ATTACK="pgdl2+apgd+fab+moeva+caa"
ATTACK="pgdl2+apgd+fab"
FILTER_CLASS=1
DEVICE="cpu"

for MODEL in tabtransformer deepfm torchrln vime
do
  MODEL_PATH="../models/mlc/best_models/${MODEL}_${DATASET}_madry.model"
  CUDA_VISIBLE_DEVICES=$DEVICES python run/scenarioA1.py --dataset_name $DATASET --model_name $MODEL --custom_path $MODEL_PATH --attacks_name $ATTACK --max_eps $MAX_EPS --subset $SUBSET --batch_size $BATCH_SIZE --device $DEVICE
  CUDA_VISIBLE_DEVICES=$DEVICES python run/scenarioA1.py --dataset_name $DATASET --model_name $MODEL --custom_path $MODEL_PATH --attacks_name $ATTACK --max_eps $MAX_EPS --subset $SUBSET --batch_size $BATCH_SIZE --device $DEVICE --filter_class=$FILTER_CLASS
done

