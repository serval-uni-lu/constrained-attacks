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

n_gen=200
n_offsprings=100
STEPS=20

for SEED in 0 1 2 3 4
# for SEED in 0
do
    # for DATASET in lcld_v2_iid url ctu_13_neris malware wids
    for DATASET in url
    do
        if [[ "$DATASET" == "malware" ]]; then
            MAX_EPS=50
            SUBSET=100
        else
            MAX_EPS=5
            SUBSET=1000
        fi

        for MODEL in vime stg torchrln tabtransformer tabnet
        do
        
            for MODEL_PATH in "../models/mlc/best_models/${MODEL}_${DATASET}_default.model" 
            do
                for SCENARIO in "--constraints_access"
                do
                    # echo sbatch ./experiments/jobs/launch-cpu.sh
                    eval "CUDA_VISIBLE_DEVICES=$DEVICES python experiments/run/scenarioA.py --dataset_name $DATASET --model_name $MODEL --custom_path $MODEL_PATH --attacks_name $ATTACK --max_eps $MAX_EPS --subset $SUBSET --batch_size $BATCH_SIZE --device $DEVICE --filter_class=$FILTER_CLASS ${SCENARIO} --seed=$SEED --n_gen $n_gen --n_offsprings $n_offsprings" --project_name eps_caa_${DATASET} --save_adv 1 --steps=$STEPS # scenario_AB_${DATASET}
                done
            done
        done
    done
done

# expected 72 * 5 per xp
