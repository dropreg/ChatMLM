#!/bin/bash
src=src
tgt=tgt

export CUDA_VISIBLE_DEVICES=7


data_dir=/opt/data/private/data/nar_chat/alpaca/data-bin
save_dir=/opt/data/private/ckpt/nar_chat/fsdp_xl_alpaca/
bpe_dir=/opt/data/private/data/xlmr/xlmr.xxl/sentencepiece.bpe.model


python xlmr/src/generate2.py $data_dir \
    --user-dir xlmr/src \
    --task seq2seq_ft_task \
    --arch nar_xlmr_xl \
    -s $src -t $tgt \
    --gen-subset test \
    --bpe 'sentencepiece' --sentencepiece-model $bpe_dir \
    --path $save_dir/checkpoint6.pt \
    --sacrebleu \
    --iter-decode-max-iter 9 \
    --iter-decode-with-beam 0 --remove-bpe \
    --iter-decode-force-max-iter \
    --batch-size 1 > test.txt
