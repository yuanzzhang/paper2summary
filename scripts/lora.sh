#!/bin/bash
export CUDA_VISIBLE_DEVICES=4
export FSDP_CPU_RAM_EFFICIENT_LOADING=1

MODEL_FLAGS="\
    --model_name meta-llama/Llama-3.2-1B-Instruct \
    --context_length 1024 \
    --use_peft \
    --peft_method lora \
    --quantization False \
    --use_fast_kernels"

DATASET_FLAGS="\
    --dataset custom_dataset \
    --custom_dataset.file src/dataset/paper_dataset.py"

TRAINING_FLAGS="\
    --batch_size_training 1 \
    --val_batch_size 1 \
    --batching_strategy padding \
    --num_epochs 1 \
    --num_workers_dataloader 1 \
    --gradient_accumulation_steps 4 \
    --lr 3e-4"
    

OUTPUT_FLAGS="\
    --output_dir output/lora/model \
    --use_wandb \
    --wandb_config.project paper2summary \
    --save_metrics \
    --use_profiler \
    --profiler_dir output/lora/profiler"

ALL_FLAGS="${MODEL_FLAGS} ${DATASET_FLAGS} ${TRAINING_FLAGS} ${OUTPUT_FLAGS}"

if command -v uv &> /dev/null; then
    echo "Using uv for execution..."
    uv run -m llama_recipes.finetuning ${ALL_FLAGS}
else
    python -m llama_recipes.finetuning ${ALL_FLAGS}
fi