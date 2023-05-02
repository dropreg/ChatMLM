SRC=src
TGT=tgt


DATA=/opt/data/private/data/nar_chat/belle/
SPM=/opt/data/private/code/sentencepiece/build/src/spm_encode
XLMR_MODEL=/opt/data/private/data/xlmr/xlmr.xl/sentencepiece.bpe.model
XLMR_DICT=/opt/data/private/data/xlmr/xlmr.xl/dict.txt

${SPM} --model=${XLMR_MODEL} < ${DATA}/train.${SRC} > ${DATA}/train.spm.${SRC}
${SPM} --model=${XLMR_MODEL} < ${DATA}/valid.${SRC} > ${DATA}/valid.spm.${SRC}
${SPM} --model=${XLMR_MODEL} < ${DATA}/test.${SRC} > ${DATA}/test.spm.${SRC}

${SPM} --model=${XLMR_MODEL} < ${DATA}/train.${TGT} > ${DATA}/train.spm.${TGT}
${SPM} --model=${XLMR_MODEL} < ${DATA}/valid.${TGT} > ${DATA}/valid.spm.${TGT}
${SPM} --model=${XLMR_MODEL} < ${DATA}/test.${TGT} > ${DATA}/test.spm.${TGT}


fairseq-preprocess \
  --source-lang ${SRC} \
  --target-lang ${TGT} \
  --trainpref ${DATA}/train.spm \
  --validpref ${DATA}/valid.spm \
  --testpref ${DATA}/test.spm \
  --destdir ${DATA}/data-bin \
  --srcdict ${XLMR_DICT} \
  --tgtdict ${XLMR_DICT} \
  --workers 40 \
