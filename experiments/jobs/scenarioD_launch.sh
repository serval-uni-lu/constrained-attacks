for SEED in 1 2 3 4
do
    for DATASET in malware
    do
        for MODEL in "tabtransformer" "torchrln" "vime" "stg" "tabnet"
        do
            sbatch ./experiments/jobs/launch-cpu.sh "SEED=$SEED DATASET=$DATASET MODEL=$MODEL ./experiments/jobs/scenarioD_inner.sh"
        done
    done
done