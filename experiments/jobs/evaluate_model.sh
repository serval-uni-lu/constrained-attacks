
DEVICE=-1

CUDA_VISIBLE_DEVICES="${1:-$DEVICE}" python -m experiments.run.evaluate_model
