N_SEED=5


for ((i = 1; i < N_SEED; i++)); do
    SEED="$i" $@
done
