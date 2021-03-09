#!/bin/bash

gpu_id=0  # remember to change this!
seed=0    # remember to change this!
mode="tree2tree_subtree_copy"
max_action=100
token_embedder="word"
change_vector_size=512
token_embed_size=128
token_encoding_size=128
action_embed_size=128
wordpredict=0.
gnn_dropout=0.2
decoder_dropout=0.3
init_decode_vec_encoder_state_dropout=0.
branch=$(git branch | sed -n -e 's/^\* \(.*\)/\1/p')
commit=$(git rev-parse HEAD | cut -c 1-7)
timestamp=`date "+%Y%m%d-%H%M%S"`

work_dir=exp_runs/${mode}_branch_${branch}_${commit}_max_act${max_action}_token_embedder_${token_embedder}_change${change_vector_size}_tok_embed${token_embed_size}_tok_enc${token_encoding_size}_act_embed${action_embed_size}_wordpredict${wordpredict}_init_dev_drop${init_decode_vec_encoder_state_dropout}.g_drop${gnn_dropout}.drop${decoder_dropout}.seed${seed}.${timestamp}

echo work dir=${work_dir}

mkdir -p ${work_dir}
CUDA_VISIBLE_DEVICES=${gpu_id} python -u -m diff_representation.exp train \
	--cuda \
    --mode=${mode} \
	--seed=${seed} \
    --train_file=/home/pengchey/Research/datasets/best-practices/toy_exp.member_access_replace.jsonl.train \
    --dev_file=/home/pengchey/Research/datasets/best-practices/toy_exp.member_access_replace.jsonl.dev \
    --vocab=/home/pengchey/Research/datasets/best-practices/vocab.toy_exp.member_access_replace.bin \
    --prune_grammar \
    --batch_size=32 \
	--change_vector_size=${change_vector_size} \
	--token_embed_size=${token_embed_size} \
	--token_encoding_size=${token_encoding_size} \
	--action_embed_size=${action_embed_size} \
	--field_embed_size=32 \
	--decoder_hidden_size=256 \
    --log_every=50 \
	--max_epoch=30 \
	--num_data_load_worker=20 \
	--max_action_sequence_length=${max_action} \
	--gnn_layer_timesteps=[5,5] \
	--token_embedder=${token_embedder} \
	--save_each_epoch \
	--gnn_dropout=${gnn_dropout} \
	--decoder_dropout=${decoder_dropout} \
	--no_change_vector \
	--node_embed_method type \
	--init_decode_vec_encoder_state_dropout=${init_decode_vec_encoder_state_dropout} \
	--word_predict_loss_weight=${wordpredict} \
	--decoder_init_method=avg_pooling \
	--work_dir=${work_dir} 2>${work_dir}/err.log

 OMP_NUM_THREADS=2 python -m diff_representation.exp decode_updated_code \
    --beam_size=5 \
    --mode=${mode} \
    --debug \
    ${work_dir}/model.bin \
    /home/pengchey/Research/datasets/best-practices/toy_exp.member_access_replace.jsonl.test 2>${work_dir}/model.bin.decode.log

	# --no_copy_identifier \
	# --gnn_no_bottom_up_connection \
	# --gnn_no_bottom_up_connection
	# --gnn_no_top_down_connection \
	# --gnn_no_token_connection \
	# --use_syntax_token_rnn \
	# --fuse_rule_and_token_rnns \
	# --feed_in_token_rnn_state_to_rule_rnn \
	# --no_change_vector \
