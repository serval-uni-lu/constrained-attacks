for SEED in 1 2 3 4
do
    for DATASET in lcld_v2_iid url ctu_13_neris wids
    do
        for MODEL in "tabtransformer" "torchrln" "vime" "stg" "tabnet"
        do
            sbatch ./experiments/jobs/launch-cpu.sh "SEED=$SEED DATASET=$DATASET MODEL=$MODEL ./experiments/jobs/scenarioE_inner.sh"
        done
    done
done