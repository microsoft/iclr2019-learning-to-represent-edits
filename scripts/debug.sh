#!/bin/bash

gpu_id=5  # remember to change this!
seed=0  # remember to change this!
mode="tree2tree_subtree_copy"
max_action=100
token_embedder="word"
change_vector_size=512
token_embed_size=128
token_encoding_size=128
action_embed_size=128
wordpredict=2.
init_decode_vec_encoder_state_dropout=0.
branch=$(git branch | sed -n -e 's/^\* \(.*\)/\1/p')
commit=$(git rev-parse HEAD | cut -c 1-7)
timestamp=`date "+%Y%m%d-%H%M%S"`

# datasets
# commit_files.from_repo.top10.processed.080614.jsonl.train
# commit_files.from_repo.top10.processed.080614.jsonl.dev

# commit_files.from_repo.top10.processed.080614.jsonl.train.filtered_max_act100_max_code70_max_node_300

work_dir=exp_runs/debug

echo work dir=${work_dir}

mkdir -p ${work_dir}
CUDA_VISIBLE_DEVICES=${gpu_id} python -u -m diff_representation.exp train \
	--cuda \
    --mode=${mode} \
	--seed=${seed} \
    --prune_grammar \
	--train_file=data/commit_files.from_repo.top10.processed.080614.jsonl.dev.filtered_max_act100_max_code70_max_node_300.top1000 \
	--dev_file=data/commit_files.from_repo.top10.processed.080614.jsonl.dev.filtered_max_act100_max_code70_max_node_300.top1000 \
	--vocab=data/vocab.from_repo.top10.080614.bin \
	--batch_size=32 \
	--change_vector_size=${change_vector_size} \
	--token_embed_size=${token_embed_size} \
	--token_encoding_size=${token_encoding_size} \
	--action_embed_size=${action_embed_size} \
	--field_embed_size=32 \
	--decoder_hidden_size=256 \
    --log_every=10 \
	--max_epoch=30 \
	--num_data_load_worker=20 \
	--max_action_sequence_length=${max_action} \
	--gnn_layer_timesteps=[5,1,5,1] \
	--token_embedder=${token_embedder} \
	--save_each_epoch \
	--init_decode_vec_encoder_state_dropout=${init_decode_vec_encoder_state_dropout} \
	--word_predict_loss_weight=${wordpredict} \
	--work_dir=${work_dir}


# 	--no_change_vector \
