for SEED in 0 1 2 3 4
do
    for DATASET in ctu_13_neris lcld_v2_iid wids
    do
        for MODEL in "tabtransformer"
        do
            sbatch ./experiments/jobs/launch-cpu.sh "SEED=$SEED DATASET=$DATASET MODEL=$MODEL ./experiments/jobs/scenarioC_inner.sh"
        done
    done
done