#!/bin/sh

DEVICES=0
# SUBSET=100
BATCH_SIZE=1024
# MAX_EPS=0.5
# ATTACK="pgdl2org+pgdl2rsae+pgdl2nrsnae+pgdl2+moeva+caa4"
ATTACK="apgd2+moeva+caa4"
FILTER_CLASS=1
DEVICE="cpu"
SEED="${SEED:-0}"

n_gen=100
n_offsprings=100
STEPS=10

for SEED in 0 1 2 3 4
# for SEED in 0
do
    # for DATASET in lcld_v2_iid url ctu_13_neris malware wids
    for DATASET in url
    do
        if [[ "$DATASET" == "malware" ]]; then
            MAX_EPS=5
            SUBSET=100
        else
            MAX_EPS=0.5
            SUBSET=1000
        fi

        # echo "MAX_EPS: $MAX_EPS"
        # for MODEL in tabtransformer
        # for MODEL in vime stg torchrln tabtransformer tabnet # tabtransformer
        for MODEL in vime stg torchrln tabtransformer tabnet
        do
            # for MODEL_PATH in "../models/mlc/best_models/${MODEL}_${DATASET}_default.model" "../models/mlc/best_models/${MODEL}_${DATASET}_madry.model" "../models/mlc/best_models/${MODEL}_${DATASET}_dist.model" "../models/mlc/best_models/${MODEL}_${DATASET}_subset.model"
            # for MODEL_PATH in "../models/mlc/best_models/${MODEL}_${DATASET}_default.model" # "../models/mlc/best_models/${MODEL}_${DATASET}_madry.model"
            for MODEL_PATH in "../models/mlc/best_models/${MODEL}_${DATASET}_dist.model" "../models/mlc/best_models/${MODEL}_${DATASET}_subset.model"
            do
                # for SCENARIO in "--constraints_access" "--no-constraints_access"
                for SCENARIO in "--constraints_access"
                do
                    # echo sbatch ./experiments/jobs/launch-cpu.sh
                    eval "CUDA_VISIBLE_DEVICES=$DEVICES python experiments/run/scenarioA.py --dataset_name $DATASET --model_name $MODEL --custom_path $MODEL_PATH --attacks_name $ATTACK --max_eps $MAX_EPS --subset $SUBSET --batch_size $BATCH_SIZE --device $DEVICE --filter_class=$FILTER_CLASS ${SCENARIO} --seed=$SEED --n_gen $n_gen --n_offsprings $n_offsprings" --project_name prepare_DE_${DATASET} --save_adv 1 --steps=$STEPS # scenario_AB_${DATASET}
                    # sleep 5
                done
            done
        done
    done
done

# expected 72 * 5 per xp
