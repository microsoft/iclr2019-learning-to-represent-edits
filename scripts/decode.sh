#!/bin/bash

 OMP_NUM_THREADS=2 python -m diff_representation.exp decode_updated_code \
    --beam_size=5 \
    --mode=$1 \
    --debug \
    $2 \
    /home/pengchey/Research/datasets/best-practices/commit_files.from_repo.top10.processed.080614.jsonl.dev.filtered_max_act100_max_code70_max_node_300.deduplicated 2>$2.decode.log
