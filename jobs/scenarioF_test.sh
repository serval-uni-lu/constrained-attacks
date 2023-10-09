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


MODELS=("tabtransformer" "torchrln" "vime")


for DATASET in lcld_v2_iid url ctu_13_neris
# for DATASET in url 
do
    for MODEL in tabtransformer torchrln vime
    # for MODEL in tabtransformer
    do
        for SCENARIO in "--constraints_access --project_name scenario_F1v4"
        do
            model_taget_name="${MODEL}:${MODEL}"
            list_target_path="../models/mlc/best_models/${MODEL}_${DATASET}_default.model:../models/mlc/best_models/${MODEL}_${DATASET}_madry.model"
            source=""
            source_path=""
            for ((j=0; j<${#MODELS[@]}; j++)); do
                if [ $MODEL != ${MODELS[j]} ]; then
                    if [ -n "$source" ]; then
                        separe=":"
                    else
                        separe=""
                    fi
                    source="${source}${separe}${MODELS[j]}:${MODELS[j]}"
                    source_path="${source_path}${separe}../models/mlc/best_models/${MODELS[j]}_${DATASET}_default.model:../models/mlc/best_models/${MODELS[j]}_${DATASET}_dist.model"
                fi
            done
            # sbatch ./jobs/launch-cpu.sh "CUDA_VISIBLE_DEVICES=$DEVICES python run/scenarioA.py --dataset_name $DATASET --model_name $MODEL --custom_path $MODEL_PATH --attacks_name $ATTACK --max_eps $MAX_EPS --subset $SUBSET --batch_size $BATCH_SIZE --device $DEVICE --filter_class=$FILTER_CLASS ${SCENARIO} --seed=$SEED --n_gen $n_gen --n_offsprings $n_offsprings"
            sbatch ./jobs/launch-cpu.sh "CUDA_VISIBLE_DEVICES=$DEVICES python run/scenarioA.py --dataset_name $DATASET --model_name $source --custom_path $source_path --attacks_name $ATTACK --max_eps $MAX_EPS --subset $SUBSET --batch_size $BATCH_SIZE --device $DEVICE --filter_class=$FILTER_CLASS ${SCENARIO} --model_name_target $model_taget_name --custom_path_target $list_target_path --seed=$SEED --n_gen $n_gen --n_offsprings $n_offsprings"
        done
    done
done

