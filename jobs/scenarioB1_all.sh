#!/bin/sh

DEVICES=0
SUBSET=1000
BATCH_SIZE=1024
MAX_EPS=0.5
ATTACK="pgdl2+apgd+caa"
FILTER_CLASS=1
DEVICE="cpu"

for DATASET in url ctu_13_neris lcld_v2_iid malware wids
do
  for MODEL in tabtransformer deepfm torchrln vime
  do
    MODEL_PATH="../models/mlc/best_models/${MODEL}_${DATASET}_default.model"
    CUDA_VISIBLE_DEVICES=$DEVICES python run/scenarioB1.py --n_offsprings 100 --n_gen 100 --dataset_name $DATASET --model_name $MODEL --custom_path $MODEL_PATH --attacks_name $ATTACK --max_eps $MAX_EPS --subset $SUBSET --batch_size $BATCH_SIZE --device $DEVICE
    CUDA_VISIBLE_DEVICES=$DEVICES python run/scenarioB1.py --n_offsprings 50 --n_gen 50 --dataset_name $DATASET --model_name $MODEL --custom_path $MODEL_PATH --attacks_name $ATTACK --max_eps $MAX_EPS --subset $SUBSET --batch_size $BATCH_SIZE --device $DEVICE
    CUDA_VISIBLE_DEVICES=$DEVICES python run/scenarioB1.py --n_offsprings 200 --n_gen 200 --dataset_name $DATASET --model_name $MODEL --custom_path $MODEL_PATH --attacks_name $ATTACK --max_eps $MAX_EPS --subset $SUBSET --batch_size $BATCH_SIZE --device $DEVICE
    CUDA_VISIBLE_DEVICES=$DEVICES python run/scenarioB1.py --n_offsprings 200 --n_gen 100  --dataset_name $DATASET --model_name $MODEL --custom_path $MODEL_PATH --attacks_name $ATTACK --max_eps $MAX_EPS --subset $SUBSET --batch_size $BATCH_SIZE --device $DEVICE
    CUDA_VISIBLE_DEVICES=$DEVICES python run/scenarioB1.py --n_offsprings 100 --n_gen 200 --dataset_name $DATASET --model_name $MODEL --custom_path $MODEL_PATH --attacks_name $ATTACK --max_eps $MAX_EPS --subset $SUBSET --batch_size $BATCH_SIZE --device $DEVICE


    MODEL_PATH="../models/mlc/best_models/${MODEL}_${DATASET}_madry.model"
    CUDA_VISIBLE_DEVICES=$DEVICES python run/scenarioB1.py --n_offsprings 100 --n_gen 100 --dataset_name $DATASET --model_name $MODEL --custom_path $MODEL_PATH --attacks_name $ATTACK --max_eps $MAX_EPS --subset $SUBSET --batch_size $BATCH_SIZE --device $DEVICE
    CUDA_VISIBLE_DEVICES=$DEVICES python run/scenarioB1.py --n_offsprings 50 --n_gen 50 --dataset_name $DATASET --model_name $MODEL --custom_path $MODEL_PATH --attacks_name $ATTACK --max_eps $MAX_EPS --subset $SUBSET --batch_size $BATCH_SIZE --device $DEVICE
    CUDA_VISIBLE_DEVICES=$DEVICES python run/scenarioB1.py --n_offsprings 200 --n_gen 200 --dataset_name $DATASET --model_name $MODEL --custom_path $MODEL_PATH --attacks_name $ATTACK --max_eps $MAX_EPS --subset $SUBSET --batch_size $BATCH_SIZE --device $DEVICE
    CUDA_VISIBLE_DEVICES=$DEVICES python run/scenarioB1.py --n_offsprings 200 --n_gen 100  --dataset_name $DATASET --model_name $MODEL --custom_path $MODEL_PATH --attacks_name $ATTACK --max_eps $MAX_EPS --subset $SUBSET --batch_size $BATCH_SIZE --device $DEVICE
    CUDA_VISIBLE_DEVICES=$DEVICES python run/scenarioB1.py --n_offsprings 100 --n_gen 200 --dataset_name $DATASET --model_name $MODEL --custom_path $MODEL_PATH --attacks_name $ATTACK --max_eps $MAX_EPS --subset $SUBSET --batch_size $BATCH_SIZE --device $DEVICE
    
  done
done

