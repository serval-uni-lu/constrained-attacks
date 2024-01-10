N_SEED=5


for ((i = 0; i < N_SEED; i++)); do
    SEED="$i" $@
done
