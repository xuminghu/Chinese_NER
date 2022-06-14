
## Directly Fine-tune
CUDA_VISIBLE_DEVICES=0 nohup python main.py --num_source_tag 23 --num_target_tag 17 --batch_size 16 --src_dm source --tgt_dm target >> directly_finetune.log 2>&1 &

## Pre-train then Fine-tune
CUDA_VISIBLE_DEVICES=0 nohup python main.py --num_source_tag 23 --num_target_tag 17 --batch_size 16 --src_dm source --tgt_dm target --source >> source_finetune.log 2>&1 &


