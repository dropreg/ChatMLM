
export CUDA_VISIBLE_DEVICES=0

python xlmr/src/inference.py \
    --model-dir /opt/data/private/ckpt/nar_chat/fsdp_xl_belle/ \
    --model-file checkpoint10.pt \
    --bpe sentencepiece \
    --sentencepiece-model /opt/data/private/data/xlmr/xlmr.xxl/sentencepiece.bpe.model \
