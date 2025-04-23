#!/bin/bash
# export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
# export NCCL_BLOCKING_WAIT=0
# export TOKENIZERS_PARALLELISM=false
# export OMP_NUM_THREADS=8
# export NCCL_IB_DISABLE=0
# export NCCL_IB_GID_INDEX=3
# export NCCL_SOCKET_IFNAME=eth0
# export NCCL_DEBUG=INFO
# ----------- Config ----------- /data/chunhui/hf_cache/transformers/AudioOnlyThinker
OUT_DIR=exp/model
MODEL_NP=Qwen/Qwen2-Audio-7B-Instruct # /data/chunhui/hf_cache/transformers/Qwen2-Audio_mispeech/snapshots/e1068f8ea646f4ffeaf3636aa5501ab7f9e8b382/ #/data/chunhui/hf_cache/transformers/AudioOnlyThinker # Qwen/Qwen2-Audio-7B-Instruct
DATA_FILE=/data/chunhui/hf_cache/hub/datasets--chunhuizng--audiokk/snapshots/c8733a32c185f50a3ec1ace24d528b3cc7129658/3ppl/train.parquet
CONFIG_PATH=conf/ds_zero3.json
MASTER_ADDR="127.0.0.1"
MASTER_PORT=29501
NNODES=1
NODE_RANK=0

# ----------- GPU Setup ----------- 
# Use e.g., "0,1,2,3,4,5,6,7" to pick specific GPUs
export CUDA_VISIBLE_DEVICES="1,2,3,4,5,6,7"

# Auto-count how many GPUs you selected
IFS=',' read -ra DEVICES <<< "$CUDA_VISIBLE_DEVICES"
GPU_NUM=$(( ${#DEVICES[@]} - 1))  # 少一个进程，留给 vLLM

# ----------- Huggingface Caches -----------
# export HF_HOME=/data/chunhui/hf_cache
# export TRANSFORMERS_CACHE=$HF_HOME/transformers
# export HF_DATASETS_CACHE=$HF_HOME/datasets
# export HF_METRICS_CACHE=$HF_HOME/metrics
export HF_HOME=/data/chunhui/hf_cache
# ----------- Optional VLLM Caches -----------

# ----------- Optional WandB Disable -----------
export WANDB_DISABLED=true

# ----------- Launch Training -----------
VLLM_USE_V1=0 torchrun --nproc_per_node=$GPU_NUM \
    --nnodes=$NNODES \
    --node-rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    src/kk_train.py \
    --config_path $CONFIG_PATH \
    --model_name_or_path $MODEL_NP \
    --out_dir $OUT_DIR \
    --data_file $DATA_FILE \
    --use_wandb false || exit 1
