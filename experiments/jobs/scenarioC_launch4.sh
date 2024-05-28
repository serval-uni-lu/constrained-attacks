for SEED in 0 1 2 3 4
do
    for DATASET in lcld_v2_iid
    do
        for MODEL in "tabtransformer" "torchrln" "vime" "stg" "tabnet"
        do
            eval "SEED=$SEED DATASET=$DATASET MODEL=$MODEL ./experiments/jobs/scenarioC_inner.sh"
        done
    done
done