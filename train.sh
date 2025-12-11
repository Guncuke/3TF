NPROC_PER_NODE=8 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
swift sft \
    --model Qwen3-4B \
    --train_type full \
    --dataset 'path' \
    --torch_dtype bfloat16 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --learning_rate 2e-5 \
    --gradient_accumulation_steps 2 \
    --attn_impl flash_attn \
    --packing true \
    --save_strategy epoch \
    --save_steps 1 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 16384 \
    --output_dir outputs \
    --system '' \
    --deepspeed zero3_offload \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
