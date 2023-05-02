#!/bin/bash
src=src
tgt=tgt


export CUDA_VISIBLE_DEVICES=4,5,6,7
export CUDA_LAUNCH_BLOCKING=1

data_dir=/opt/data/private/data/nar_chat/iwslt_xl/data-bin
save_dir=/opt/data/private/ckpt/nar_chat/fsdp_xl/
xlmr_dir=/opt/data/private/data/xlmr/xlmr.xl/model.pt
max_token=2048


python xlmr/src/train_fsdp.py $data_dir \
    --reset-optimizer --reset-dataloader --reset-meters \
    --restore-file $xlmr_dir \
    --user-dir xlmr/src \
    --ddp-backend fully_sharded \
    --fp16 \
    --checkpoint-activations \
    --truncate-source \
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
    --log-interval 10 \
    --save-dir $save_dir | tee -a $save_dir/train.log \
