#!/bin/bash

work_dir=exp_runs/align_rep_repo_change512_graph_prof
mkdir -p ${work_dir}
CUDA_VISIBLE_DEVICES=2 python -u -m diff_representation.exp train \
	--cuda \
    --mode tree \
    --prune_grammar \
	--train_file=data/commit_files.from_repo.processed.070523.jsonl.dev \
	--dev_file=data/commit_files.from_repo.processed.070523.jsonl.dev \
	--vocab=data/vocab.repo.070523.freq10.bin \
	--batch_size=32 \
	--change_vector_size=512 \
	--token_embed_size=64 \
	--token_encoding_size=64 \
	--action_embed_size=64 \
	--field_embed_size=32 \
	--decoder_hidden_size=256 \
    --log_every 25 \
	--num_data_load_worker 10 \
	--work_dir=${work_dir} 2>${work_dir}/err.log
