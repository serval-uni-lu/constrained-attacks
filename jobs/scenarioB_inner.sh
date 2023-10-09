list_n_gen=("100" "50" "200" "200" "100")
list_n_offsprings=("100" "50" "200" "100" "200")

# Ensure both lists have the same length
if [ ${#list_n_gen[@]} -ne ${#list_n_offsprings[@]} ]; then
    echo "Error: The lists have different lengths."
    exit 1
fi

for index in "${!list_n_gen[@]}"
do
    n_gen="${list_n_gen[$index]}"
    n_offsprings="${list_n_offsprings[$index]}"
    eval "$@" --n_gen $n_gen --n_offsprings $n_offsprings
done
