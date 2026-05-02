#!/bin/bash
# ============================================================
# SBATCH training script for deepfake detection models.
#
# Usage:
#   sbatch scripts/train_sbatch.sh \
#     --config configs/training/dire.yaml \
#     --train-dataset configs/datasets/train.yaml \
#     [--val-dataset configs/datasets/val.yaml]
#
# Resource defaults below are conservative — tune for your cluster.
# ============================================================

#SBATCH --job-name=deepfake_train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4              # GPUs per node (match GPUS_PER_NODE below)
#SBATCH --cpus-per-task=16        # CPU workers (≥ dataloader_num_workers * GPUS_PER_NODE)
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# --- Parse arguments -----------------------------------------------------------
CONFIG=""
TRAIN_DATASET=""
VAL_DATASET=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)           CONFIG="$2";        shift 2 ;;
        --train-dataset)    TRAIN_DATASET="$2"; shift 2 ;;
        --val-dataset)      VAL_DATASET="$2";   shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "$CONFIG" || -z "$TRAIN_DATASET" ]]; then
    echo "Error: --config and --train-dataset are required."
    exit 1
fi

# --- Environment ---------------------------------------------------------------
# Activate your virtual environment or conda env here, e.g.:
#   source .venv/bin/activate
#   conda activate deepfake

# --- Distributed config --------------------------------------------------------
GPUS_PER_NODE=4   # Keep in sync with --gres=gpu: above

# First node in the allocation becomes the rendezvous master
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=29500

export MASTER_ADDR MASTER_PORT

# --- Build train.py arguments --------------------------------------------------
TRAIN_ARGS=(
    --config "$CONFIG"
    --train-dataset "$TRAIN_DATASET"
)
if [[ -n "$VAL_DATASET" ]]; then
    TRAIN_ARGS+=(--val-dataset "$VAL_DATASET")
fi

# --- Launch --------------------------------------------------------------------
mkdir -p logs

echo "Job:          $SLURM_JOB_NAME ($SLURM_JOB_ID)"
echo "Nodes:        $SLURM_NNODES  (master: $MASTER_ADDR:$MASTER_PORT)"
echo "GPUs/node:    $GPUS_PER_NODE"
echo "Config:       $CONFIG"
echo "Train data:   $TRAIN_DATASET"
echo "Val data:     ${VAL_DATASET:-none}"
echo "---"

srun torchrun \
    --nproc_per_node="$GPUS_PER_NODE" \
    --nnodes="$SLURM_NNODES" \
    --node_rank="$SLURM_NODEID" \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    train.py "${TRAIN_ARGS[@]}"
