#!/bin/bash

source activate code_mining_pytorch0.4

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

# datasets
# new data
# --train_file=../../data/commit_files.from_repo.sampled54.processed.080910.jsonl.filtered_max_act100_max_code70_max_node_300.down_sampled.train \
# --dev_file=../../data/commit_files.from_repo.sampled54.processed.080910.jsonl.filtered_max_act100_max_code70_max_node_300.down_sampled.dev \
# --vocab=../../data/vocab.from_repo.080910.freq10.bin \

train_file="/projects/tir1/users/pengchey/datasets/best-practices/commit_files.from_repo.top10.processed.080614.jsonl.train.filtered_max_act100_max_code70_max_node_300"
dev_file="/projects/tir1/users/pengchey/datasets/best-practices/commit_files.from_repo.top10.processed.080614.jsonl.dev.filtered_max_act100_max_code70_max_node_300"
test_file="/projects/tir1/users/pengchey/datasets/best-practices/commit_files.from_repo.top10.processed.080614.jsonl.dev.filtered_max_act100_max_code70_max_node_300.deduplicated"
vocab_file="/projects/tir1/users/pengchey/datasets/best-practices/vocab.from_repo.top10.080614.bin"

# train_file="/home/pengchey/Research/datasets/best-practices/commit_files.from_repo.sampled54.processed.080910.jsonl.filtered_max_act100_max_code70_max_node_300.down_sampled.train"
# dev_file="/home/pengchey/Research/datasets/best-practices/commit_files.from_repo.sampled54.processed.080910.jsonl.filtered_max_act100_max_code70_max_node_300.down_sampled.dev"
# test_file="/home/pengchey/Research/datasets/best-practices/commit_files.from_repo.sampled54.processed.080910.jsonl.filtered_max_act100_max_code70_max_node_300.down_sampled.test"
# vocab_file="/home/pengchey/Research/datasets/best-practices/vocab.from_repo.080910.freq10.bin"

# commit_files.from_repo.top10.processed.080614.jsonl.train.filtered_max_act100_max_code70_max_node_300
# --train_file=../../data/commit_files.from_repo.top10.processed.080614.jsonl.train.filtered_max_act100_max_code70_max_node_300 \
# --dev_file=../../data/commit_files.from_repo.top10.processed.080614.jsonl.dev.filtered_max_act100_max_code70_max_node_300 \

work_dir=exp_runs/${mode}_branch_${branch}_${commit}_max_act${max_action}_token_embedder_${token_embedder}_change${change_vector_size}_tok_embed${token_embed_size}_tok_enc${token_encoding_size}_act_embed${action_embed_size}_wordpredict${wordpredict}_init_dev_drop${init_decode_vec_encoder_state_dropout}.g_drop${gnn_dropout}.drop${decoder_dropout}.seed${seed}.${timestamp}

echo work dir=${work_dir}

mkdir -p ${work_dir}
python -u -m diff_representation.exp train \
	--cuda \
    --mode=${mode} \
	--seed=${seed} \
	--train_file=${train_file} \
	--dev_file=${dev_file} \
    --vocab=${vocab_file} \
    --prune_grammar \
    --batch_size=32 \
	--change_vector_size=${change_vector_size} \
	--token_embed_size=${token_embed_size} \
	--token_encoding_size=${token_encoding_size} \
	--action_embed_size=${action_embed_size} \
	--field_embed_size=32 \
	--decoder_hidden_size=256 \
    --log_every=100 \
	--max_epoch=30 \
	--num_data_load_worker=1 \
	--max_action_sequence_length=${max_action} \
	--gnn_layer_timesteps=[5,5] \
	--token_embedder=${token_embedder} \
	--gnn_dropout=${gnn_dropout} \
	--decoder_dropout=${decoder_dropout} \
    --gnn_no_token_connection \
    --no_change_vector \
	--init_decode_vec_encoder_state_dropout=${init_decode_vec_encoder_state_dropout} \
	--word_predict_loss_weight=${wordpredict} \
	--decoder_init_method=avg_pooling \
	--work_dir=${work_dir} 2>${work_dir}/err.log

	# --node_embed_method type_and_field \
	# --gnn_next_sibling_connection \
	# --gnn_prev_sibling_connection \
	# --no_copy_identifier \
	# --gnn_no_bottom_up_connection \
	# --gnn_no_bottom_up_connection
	# --gnn_no_top_down_connection \
	# --gnn_no_token_connection \
	# --use_syntax_token_rnn \
	# --fuse_rule_and_token_rnns \
	# --feed_in_token_rnn_state_to_rule_rnn \
	# --no_change_vector \

OMP_NUM_THREADS=2 python -m diff_representation.exp decode_updated_code \
    --beam_size=5 \
	--debug \
    --mode=${mode} \
    ${work_dir}/model.bin \
    ${test_file} 2>${work_dir}/model.bin.decode.log
