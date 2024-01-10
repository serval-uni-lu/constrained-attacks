for SEED in 0
do
    for DATASET in lcld_v2_iid
    do
        for MODEL in "vime"
        do
            sbatch ./experiments/jobs/launch-cpu.sh "SEED=$SEED DATASET=$DATASET MODEL=$MODEL ./experiments/jobs/scenarioD_inner.sh"
        done
    done
done