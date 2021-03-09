#!/bin/bash
model_folder = "exp_runs/tree2tree_branch_pred_change_seq_73b550a_max_act100_token_embedder_word_change512_tok_embed128_tok_enc128_act_embed128_wordpredict0._init_enc_dropout0..seed0.20180820-142107"

for model_file in ${model_folder}/*.bin
do
    echo ${model_file}
done