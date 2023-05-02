#!/bin/bash
src=src
tgt=tgt

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

data_dir=/opt/data/private/data/nar_chat/alpaca/data-bin
save_dir=/opt/data/private/ckpt/nar_chat/megatron8_ft/
xlmr_dir=/opt/data/private/data/xlmr/xlmr.xxl/parallel_8/model.pt
max_token=2048
update_freq=1
world_size=8


# --reset-optimizer --reset-dataloader --reset-meters \
# --restore-file $xlmr_dir \
torchrun --master_port 29000 --nproc_per_node $world_size xlmr/src/train_megatron.py $data_dir \
    --model-parallel-size $world_size \
    --distributed-world-size $world_size \
    --user-dir xlmr/src \
    --max-source-positions 512 \
    --max-target-positions 512 \
    --memory-efficient-fp16 \
    --fp16 --fp16-init-scale 4 \
    --checkpoint-activations \
    --task seq2seq_ft_task \
    --megatron-model \
    --arch nar_xlmr_xxl \
    --criterion cmlm_loss \
    -s $src -t $tgt \
    --max-tokens $max_token \
    --update-freq $update_freq \
    --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-06 --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr 2e-5 \
    --weight-decay 0.0 \
    --total-num-update 20000 --warmup-updates 500 \
    --max-epoch 10 \
    --no-progress-bar \
    --log-interval 100 \
    --save-dir $save_dir | tee -a $save_dir/train.log \
