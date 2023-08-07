DEVICE=0
SUBSET=1000
BATCH_SIZE=1024
MAX_EPS=0.5
DATASET="${1:-url}"
MODEL="${2:-deepfm}"
ATTACK="pgdl2+apgd+fab+moeva+caa"
FILTER_CLASS=1

CUDA_VISIBLE_DEVICES=$DEVICE python run/scenarioA1.py --dataset_name $DATASET --model_name $MODEL --custom_path "../models/mlc/$MODEL_$DATASET_default.model" --attacks_name "pgdl2" --max_eps $MAX_EPS --subset $SUBSET --batch_size $BATCH_SIZE
CUDA_VISIBLE_DEVICES=$DEVICE python run/scenarioA1.py --dataset_name $DATASET --model_name $MODEL --custom_path "../models/mlc/$MODEL_$DATASET_default.model" --attacks_name "pgdl2" --max_eps $MAX_EPS --subset $SUBSET --batch_size $BATCH_SIZE --filter_class=$FILTER_CLASS