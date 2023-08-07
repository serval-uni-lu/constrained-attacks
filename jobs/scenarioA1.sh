DEVICE=0
BATCH_SIZE=1024
DATASET="url"
MODEL="deepfm"
ATTACK="pgdl2+apgd+fab"
THREADS=2

CUDA_VISIBLE_DEVICES=$DEVICE python run/scenarioA1.py --dataset_name $DATASET --model_name $MODEL --custom_path "../models/mlc/$MODEL_$DATASET_default.model" --attacks_name "pgdl2" --max_eps 0.5