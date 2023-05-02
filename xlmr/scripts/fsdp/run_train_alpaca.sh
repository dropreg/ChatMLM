#!/bin/bash
src=src
tgt=tgt

export OMP_NUM_THREADS=20
export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_LAUNCH_BLOCKING=1

data_dir=/opt/data/private/data/nar_chat/alpaca/data-bin
save_dir=/opt/data/private/ckpt/nar_chat/fsdp_xl_alpaca/
xlmr_dir=/opt/data/private/data/xlmr/xlmr.xl/model.pt
max_token=2048

# --reset-optimizer --reset-dataloader --reset-meters \
# --restore-file $save_dir \
python xlmr/src/train_fsdp.py $data_dir \
    --user-dir xlmr/src \
    --ddp-backend fully_sharded \
    --fp16 \
    --max-source-positions 512 \
    --max-target-positions 512 \
    --truncate-source \
    --checkpoint-activations \
    --no-reshard-after-forward \
    --no-save-optimizer-state \
    --task seq2seq_ft_task \
    --arch nar_xlmr_xl \
    --criterion cmlm_loss \
    -s $src -t $tgt \
    --max-tokens $max_token \
    --optimizer adam --adam-betas "(0.9, 0.98)" \
    --lr-scheduler polynomial_decay --lr 2e-5 \
    --weight-decay 0.01 \
    --total-num-update 10000 --warmup-updates 200 \
    --max-epoch 20 \
    --no-progress-bar \
    --log-interval 50 \
    --save-dir $save_dir | tee -a $save_dir/train.log \
