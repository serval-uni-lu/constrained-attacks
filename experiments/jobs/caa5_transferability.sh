DEVICES=-1
SUBSET=1000
BATCH_SIZE=1024
MAX_EPS=0.5
ATTACK="caa5"
FILTER_CLASS=1
DEVICE="cpu"


for SEED in 0 1 2 3 4
do
    for DATASET in lcld_v2_iid ctu_13_neris url wids
    do
        for MODEL in "tabtransformer" "torchrln" "vime" "stg" "tabnet"
        do
            MODEL_TARGET=("tabtransformer" "torchrln" "vime" "stg" "tabnet")
            model_taget_name=""
            for model in "${MODEL_TARGET[@]}"; do
                if [ -n "$model_taget_name" ]; then
                    model_taget_name="${model_taget_name}:${model}"  # Add semicolon separator
                else
                    model_taget_name="${model}"
                fi
            done

            for MODEL_PATH in "../models/mlc/best_models/${MODEL}_${DATASET}_default.model"
            do
                for SCENARIO in "--constraints_access"
                do
                    list_target_path=""
                    for model in "${MODEL_TARGET[@]}"; do
                        if [ -n "$list_target_path" ]; then
                            list_target_path="${list_target_path}:../models/mlc/best_models/${model}_${DATASET}_default.model"
                        else
                            list_target_path="../models/mlc/best_models/${model}_${DATASET}_default.model"
                        fi
                    done
                    eval "CUDA_VISIBLE_DEVICES=$DEVICES python experiments/run/scenarioA.py --dataset_name $DATASET --model_name $MODEL --custom_path $MODEL_PATH --attacks_name $ATTACK --max_eps $MAX_EPS --subset $SUBSET --batch_size $BATCH_SIZE --device $DEVICE ${SCENARIO} --filter_class=$FILTER_CLASS --model_name_target $model_taget_name --custom_path_target $list_target_path --seed=$SEED --project_name caa5_transferability_v2_${DATASET} --load_adv 1 --save_adv 0 --save_examples 0"
                done
            done
        done
    done
done