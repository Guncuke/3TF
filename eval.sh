#!/bin/bash

export HF_TOKEN=
export HUGGINGFACE_TOKEN=

cd path/to/lighteval
pip install -e .
pip install emoji
pip install vllm==0.10.0
pip install more-itertools
pip install langdetect
python -c "import nltk; nltk.download('punkt_tab')"

# 定义多个模型
MODELS=(
    "path1"
    "path2"
)

GPUS=(0 1 2 3)
TASKS=("lighteval|aime24|0" "lighteval|gsm8k|0" "lighteval|math_500|0" "extended|olympiad_bench:OE_TO_maths_en_COMP|0")

export VLLM_WORKER_MULTIPROC_METHOD=spawn

# 循环测试每个模型
for MODEL in "${MODELS[@]}"; do
    echo "=========================================="
    echo "Starting evaluation for model: $MODEL"
    echo "=========================================="
    
    MODEL_ARGS="model_name=$MODEL,dtype=bfloat16,max_model_length=32768,gpu_memory_utilization=0.9,tensor_parallel_size=1,generation_parameters={max_new_tokens:32768,temperature:0.8,top_p:0.95,repetition_penalty:1.0}"
    
    # 为每个任务启动并行评测
    for i in ${!GPUS[@]}; do
        export CUDA_VISIBLE_DEVICES=${GPUS[$i]}
        LOG_PATH="${MODEL}/eval_${TASKS[$i]}.log"
        nohup lighteval vllm $MODEL_ARGS ${TASKS[$i]} > "$LOG_PATH" 2>&1 &
        echo "Started task ${TASKS[$i]} on GPU ${GPUS[$i]}, log: $LOG_PATH"
    done
    
    # 等待所有后台任务完成
    echo "Waiting for all tasks to complete for $MODEL..."
    wait
    echo "All tasks completed for $MODEL"
    echo ""
done

echo "=========================================="
echo "All models evaluation completed!"
echo "=========================================="
