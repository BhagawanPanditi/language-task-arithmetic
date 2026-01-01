export CUDA_VISIBLE_DEVICES=0,1
export TMPDIR=/home/compiling-ganesh/24m0829/language-task-arithmetic/tmp

accelerate launch \
    --mixed_precision bf16 \
    --num_processes 2 \
    --use_deepspeed \
    --deepspeed_config_file /home/compiling-ganesh/24m0829/language-task-arithmetic/open-instruct/configs/ds_configs/stage3_no_offloading_accelerate.conf \
    --deepspeed_multinode_launcher standard \
    /home/compiling-ganesh/24m0829/language-task-arithmetic/open-instruct/open_instruct/finetune_with_val.py \
    --model_name_or_path /home/compiling-ganesh/24m0829/language-task-arithmetic/open-instruct/models/llama-3.2-3b-base \
    --tokenizer_name_or_path /home/compiling-ganesh/24m0829/language-task-arithmetic/open-instruct/models/llama-3.2-3b-instruct \
    --use_slow_tokenizer \
    --max_seq_length 4096 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --learning_rate 5e-06 \
    --lr_scheduler_type linear \
    --checkpointing_steps 100 \
    --eval_steps 100 \
    --warmup_ratio 0.03 \
    --weight_decay 0.0 \
    --num_train_epochs 3 \
    --output_dir /home/compiling-ganesh/24m0829/language-task-arithmetic/open-instruct/output/llama-sft-math-language-compliment/ \
    --with_tracking \
    --report_to tensorboard \
    --logging_steps 1 \
    --reduce_loss sum \
    --model_revision main \
    --exp_name llama-sft-math-language-compliment \
    --seed 123
