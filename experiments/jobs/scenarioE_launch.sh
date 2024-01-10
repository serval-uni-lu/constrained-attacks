for SEED in 1
do
    for DATASET in lcld_v2_iid
    do
        for MODEL in "torchrln"
        do
            sbatch ./experiments/jobs/launch-cpu.sh "SEED=$SEED DATASET=$DATASET MODEL=$MODEL ./experiments/jobs/scenarioE_inner.sh"
        done
    done
done