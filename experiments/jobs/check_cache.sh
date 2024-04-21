#!/bin/sh

DEVICES=0
SUBSET=1000
BATCH_SIZE=1024
MAX_EPS=0.5
ATTACK="pgdl2+apgd+moeva+caa3"
FILTER_CLASS=1
DEVICE="cpu"
SEED="${SEED:-0}"

n_gen=100
n_offsprings=100


for DATASET in wids
do
    # for MODEL in stg torchrln vime tabnet tabtransformer
    for MODEL in torchrln 
    do
        # for MODEL_PATH in "../models/mlc/best_models/${MODEL}_${DATASET}_default.model" "../models/mlc/best_models/${MODEL}_${DATASET}_madry.model"
        for MODEL_PATH in "../models/mlc/best_models/${MODEL}_${DATASET}_default.model"
        do
            # for SCENARIO in "--constraints_access" "--no-constraints_access"
            for SCENARIO in "--constraints_access"
            do
                sbatch ./experiments/jobs/launch-cpu.sh "CUDA_VISIBLE_DEVICES=$DEVICES python experiments/run/scenarioA.py --dataset_name $DATASET --model_name $MODEL --custom_path $MODEL_PATH --attacks_name $ATTACK --max_eps $MAX_EPS --subset $SUBSET --batch_size $BATCH_SIZE --device $DEVICE --filter_class=$FILTER_CLASS ${SCENARIO} --seed=$SEED --n_gen $n_gen --n_offsprings $n_offsprings" --project_name test --save_adv 1 # scenario_AB_${DATASET}
            done
        done
    done
done

# expected 72 * 5 per xp
