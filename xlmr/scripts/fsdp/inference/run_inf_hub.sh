
export CUDA_VISIBLE_DEVICES=0

# --model-dir /opt/data/private/ckpt/nar_chat/fsdp_xl_alpaca/ \
python xlmr/src/inference.py \
    --model-dir /opt/data/private/ckpt/nar_chat/megatron8_ft/ \
    --model-file model.pt \
    --bpe sentencepiece \
    --sentencepiece-model /opt/data/private/data/xlmr/xlmr.xl/sentencepiece.bpe.model \

