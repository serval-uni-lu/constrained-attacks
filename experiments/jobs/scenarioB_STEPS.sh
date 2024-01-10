#!/bin/bash

DEVICES=0
SUBSET=1000
BATCH_SIZE=1024
MAX_EPS=0.5
ATTACK="moeva"
FILTER_CLASS=1
DEVICE="cpu"
SEED="${SEED:-0}"

for SEED in 0 1 2 3 4
do
  for DATASET in wids
  do
      for MODEL in tabnet stg tabtransformer torchrln vime
      do
          for MODEL_PATH in "../models/mlc/best_models/${MODEL}_${DATASET}_default.model" "../models/mlc/best_models/${MODEL}_${DATASET}_madry.model"
          do
            for SCENARIO in "--constraints_access" "--no-constraints_access"
            do
              sbatch ./experiments/jobs/launch-cpu.sh "./experiments/jobs/scenarioB_inner.sh CUDA_VISIBLE_DEVICES=$DEVICES python experiments/run/scenarioA.py --dataset_name $DATASET --model_name $MODEL --custom_path $MODEL_PATH --attacks_name $ATTACK --max_eps $MAX_EPS --subset $SUBSET --batch_size $BATCH_SIZE --device $DEVICE ${SCENARIO} --filter_class=$FILTER_CLASS --seed=$SEED --project_name scenario_B_${DATASET}_STEPS"
            done
          done
      done
  done
done
