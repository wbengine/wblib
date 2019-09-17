#!/usr/bin/env bash


# cmp the best file
python ../../run_wer.py \
    --best tmp.best \
    --refer tmp.refer \
    --filter ../../filters/wer_hyp_filter \
    --special_word "<?>"


# cmp the N-best file with score
#python ../../run_wer.py \
#    --nbest tmp.nbest \
#    --score tmp.score \
#    --refer tmp.refer \
#    --filter ../../filters/wer_hyp_filter \
#    --special_word "<?>" \
#    > nbest.log
#
## cmp the oracle WER
#python ../../run_wer.py \
#    --nbest tmp.nbest \
#    --refer tmp.refer \
#    --filter ../../filters/wer_hyp_filter \
#    --special_word "<?>" \
#    --oracle \
#    > oracle-wer.log
