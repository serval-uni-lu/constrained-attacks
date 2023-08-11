#!/bin/sh

DEVICES=0
SUBSET=1000
BATCH_SIZE=1024
MAX_EPS=0.5
ATTACK="caa"
FILTER_CLASS=1
DEVICE="cpu"

for DATASET in url ctu_13_neris lcld_v2_iid malware wids
do
  for MODEL_TRG in tabtransformer deepfm torchrln vime
  do
    for MODEL_SRC in tabtransformer deepfm torchrln vime
    do
      MODEL_PATH_SRC="../models/mlc/best_models/${MODEL_SRC}_${DATASET}_default.model"
      MODEL_PATH_TRG="../models/mlc/best_models/${MODEL_TRG}_${DATASET}_default.model"
      CUDA_VISIBLE_DEVICES=$DEVICES python run/scenarioC1.py --dataset_name $DATASET --model_name_source $MODEL_SRC --custom_path_source $MODEL_PATH_SRC --model_name_target $MODEL_TRG --custom_path_target $MODEL_PATH_TRG --attacks_name $ATTACK --max_eps $MAX_EPS --subset $SUBSET --batch_size $BATCH_SIZE --device $DEVICE 

      MODEL_PATH_TRG="../models/mlc/best_models/${MODEL_TRG}_${DATASET}_madry.model"
      CUDA_VISIBLE_DEVICES=$DEVICES python run/scenarioC1.py --dataset_name $DATASET --model_name_source $MODEL_SRC --custom_path_source $MODEL_PATH_SRC --model_name_target $MODEL_TRG --custom_path_target $MODEL_PATH_TRG --attacks_name $ATTACK --max_eps $MAX_EPS --subset $SUBSET --batch_size $BATCH_SIZE --device $DEVICE
    done
  done
done

