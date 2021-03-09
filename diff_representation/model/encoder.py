# coding=utf-8
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import torch.nn as nn
import torch.nn.utils
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from diff_representation.model.data_model import BatchedCodeChunk
from . import nn_utils


class SequentialCodeEncoder(nn.Module):
    """encode a code chunk"""
    def __init__(self, token_embed_size, token_encoding_size, code_token_embedder, vocab):
        super(SequentialCodeEncoder, self).__init__()

        self.vocab = vocab

        self.code_token_embedder = code_token_embedder
        self.encoder_lstm = nn.LSTM(token_embed_size, token_encoding_size // 2, bidirectional=True)

    @property
    def device(self):
        return self.code_token_embedder.device

    def forward(self, code_list, is_sorted=False, embedding_cache=None):
        batched_code_lens = [len(code) for code in code_list]

        if is_sorted is False:
            sorted_example_ids, example_old2new_pos = nn_utils.get_sort_map(batched_code_lens)
            code_list = [code_list[i] for i in sorted_example_ids]

        if embedding_cache:
            # (code_seq_len, batch_size, token_embed_size)
            token_embed = embedding_cache.get_embed_for_token_sequences(code_list)
        else:
            # (code_seq_len, batch_size, token_embed_size)
            token_embed = self.code_token_embedder(code_list)

        packed_token_embed = pack_padded_sequence(token_embed, [len(code) for code in code_list])

        # token_encodings: (tgt_query_len, batch_size, hidden_size)
        token_encodings, (last_state, last_cell) = self.encoder_lstm(packed_token_embed)
        token_encodings, _ = pad_packed_sequence(token_encodings)

        # (batch_size, hidden_size * 2)
        last_state = torch.cat([last_state[0], last_state[1]], 1)
        last_cell = torch.cat([last_cell[0], last_cell[1]], 1)

        # (batch_size, tgt_query_len, hidden_size)
        token_encodings = token_encodings.permute(1, 0, 2)
        if is_sorted is False:
            token_encodings = token_encodings[example_old2new_pos]
            last_state = last_state[example_old2new_pos]
            last_cell = last_cell[example_old2new_pos]

        return token_encodings, (last_state, last_cell)

    def encode(self, code_list, embedding_cache=None):
        batched_code = BatchedCodeChunk(code_list=code_list,
                                        vocab=self.vocab,
                                        device=self.device)

        token_encodings, (last_state, last_cell) = self.forward(batched_code.code_list, embedding_cache=embedding_cache)

        batched_code.encoding = token_encodings
        batched_code.last_state = last_state
        batched_code.last_cell = last_cell

        return batched_code


class ContextEncoder(SequentialCodeEncoder):
    def __init__(self, **kwargs):
        super(ContextEncoder, self).__init__(**kwargs)

    def forward(self, context_list, is_sorted=False):
        return super(ContextEncoder, self).forward(context_list, is_sorted=is_sorted)
