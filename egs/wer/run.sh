#!/usr/bin/env bash


# cmp the best file
python ../../run_wer.py \
    --best tmp.best \
    --refer tmp.refer \
    --filter ../../filters/wer_hyp_filter \
    --special_word "<?>" \
    > best.log


# cmp the N-best file with score
python ../../run_wer.py \
    --nbest tmp.nbest \
    --score tmp.score \
    --refer tmp.refer \
    --filter ../../filters/wer_hyp_filter \
    --special_word "<?>" \
    > nbest.log
