# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import string
from itertools import chain
from tqdm import tqdm
import sys
import numpy as np

import torch
import torch.nn as nn

from diff_representation.asdl.grammar import ASDLGrammar
from diff_representation.asdl.transition_system.transition import TransitionSystem
from diff_representation.model import utils, nn_utils
from diff_representation.model.edit_encoder.edit_encoder import EditEncoder
from diff_representation.model.encdec.sequential_encoder import SequentialEncoder
from diff_representation.vocab import VocabEntry
from diff_representation.model.embedder import CodeTokenEmbedder, SyntaxTreeEmbedder, EmbeddingTable, ConvolutionalCharacterEmbedder
from diff_representation.model.encdec.graph_encoder import SyntaxTreeEncoder
from diff_representation.model.encdec.decoder import Decoder
from diff_representation.model.encdec.sequential_decoder import SequentialDecoder
from diff_representation.model.encdec.transition_decoder import TransitionDecoder


def _prepare_edit_encoding(model, edit_encoding):
    if isinstance(edit_encoding, np.ndarray):
        edit_encoding = torch.from_numpy(edit_encoding).to(model.device)
    if len(edit_encoding.size()) == 1:
        edit_encoding = edit_encoding.unsqueeze(0)

    return edit_encoding


class NeuralEditor(nn.Module):
    def __init__(self,
                 encoder,
                 decoder,
                 context_encoder,
                 edit_encoder: EditEncoder,
                 args):
        super(NeuralEditor, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.context_encoder = context_encoder
        self.edit_encoder = edit_encoder
        self.args = args

    @property
    def vocab(self):
        return self.encoder.vocab

    @property
    def device(self):
        return self.encoder.device

    def forward(self, examples, return_change_vectors=False):
        """
        Given a batch of examples, compute the edit encodings, and use the edit encodings to predict the target data,
        return the log-likelihood of generating the target data
        """

        prev_data = [e.prev_data for e in examples]
        updated_data = [e.updated_data for e in examples]
        context_data = [e.context for e in examples]

        # prepare shared encodings
        prev_data_encoding = self.encoder(prev_data)

        context_encoding = self.context_encoder(context_data)

        updated_data_encoding = self.encoder(updated_data)

        # compute edit encoding
        edit_encoding = self.edit_encoder(examples, prev_data_encoding, updated_data_encoding)

        results = self.decoder(examples, prev_data_encoding, context_encoding, edit_encoding)

        if return_change_vectors:
            results['edit_encoding'] = edit_encoding

        return results

    def get_edit_encoding(self, examples):
        prev_data = [e.prev_data for e in examples]
        updated_data = [e.updated_data for e in examples]

        prev_data_encoding = self.encoder(prev_data)
        updated_data_encoding = self.encoder(updated_data)

        edit_encoding = self.edit_encoder(examples, prev_data_encoding, updated_data_encoding)

        return edit_encoding

    def get_edit_encoding_by_batch(self, examples, batch_size=32):
        sorted_example_ids, example_old2new_pos = nn_utils.get_sort_map([len(c.change_seq) for c in examples])
        sorted_examples = [examples[i] for i in sorted_example_ids]

        change_vecs = []

        for batch_examples in tqdm(nn_utils.batch_iter(sorted_examples, batch_size), file=sys.stdout,
                                   total=len(examples) // batch_size):

            batch_change_vecs = self.get_edit_encoding(batch_examples)
            change_vecs.append(batch_change_vecs)

        change_vecs = torch.cat(change_vecs, dim=0)
        # unsort all batch_examples
        change_vecs = change_vecs[example_old2new_pos]

        return change_vecs

    def decode_updated_data(self, example, edit_encoding=None, with_change_vec=False, beam_size=5, debug=False):
        prev_data = [example.prev_data]
        updated_data = [example.updated_data]
        context_data = [example.context]

        # prepare shared encodings
        prev_data_encoding = self.encoder(prev_data)

        context_encoding = self.context_encoder(context_data)

        updated_data_encoding = self.encoder(updated_data)

        if edit_encoding is not None:
            edit_encoding = _prepare_edit_encoding(self, edit_encoding)
        elif with_change_vec:
            edit_encoding = self.edit_encoder([example], prev_data_encoding, updated_data_encoding)
        else:
            edit_encoding = torch.zeros(1, self.args['edit_encoder']['edit_encoding_size'], device=self.device)

        hypotheses = self.decoder.beam_search_with_source_encodings(example.prev_data, prev_data_encoding,
                                                                    example.context, context_encoding,
                                                                    edit_encoding,
                                                                    beam_size=beam_size,
                                                                    max_decoding_time_step=70,
                                                                    debug=debug)

        return hypotheses

    @staticmethod
    def build(args, vocab=None, **kwargs):
        if args['mode'] == 'seq2seq':
            return Seq2SeqEditor.build(args, vocab=vocab, **kwargs)
        elif args['mode'] == 'graph2tree':
            return Graph2TreeEditor.build(args, vocab=vocab, **kwargs)

    def save(self, model_path, kwargs=None):
        params = {
            'args': self.args,
            'vocab': self.vocab,
            'state_dict': self.state_dict(),
            'kwargs': kwargs
        }

        torch.save(params, model_path)

    @staticmethod
    def load(model_path, use_cuda=True):
        device = torch.device("cuda:0" if use_cuda else "cpu")
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        args = params['args']
        kwargs = params['kwargs'] if params['kwargs'] is not None else dict()

        model = NeuralEditor.build(args, vocab=params['vocab'], **kwargs)
        model.load_state_dict(params['state_dict'])
        model = model.to(device)
        model.eval()

        return model


class Seq2SeqEditor(NeuralEditor):
    def __init__(self, *args, **kwargs):
        super(Seq2SeqEditor, self).__init__(*args, **kwargs)

    @staticmethod
    def build(args, vocab=None):
        if not vocab:
            vocab = VocabEntry.load(args['dataset']['vocab_path'])

        embedder = CodeTokenEmbedder(args['embedder']['token_embed_size'], vocab)

        edit_encoder = EditEncoder.build(args, vocab=vocab, embedder=embedder)

        encoder = SequentialEncoder(args['embedder']['token_embed_size'],
                                    args['encoder']['token_encoding_size'],
                                    token_embedder=embedder,
                                    vocab=vocab)

        decoder = SequentialDecoder(args['embedder']['token_embed_size'],
                                    args['encoder']['token_encoding_size'],
                                    args['edit_encoder']['edit_encoding_size'],
                                    args['decoder']['hidden_size'],
                                    dropout=args['decoder']['dropout'],
                                    init_decode_vec_encoder_state_dropout=args['decoder']['init_decode_vec_encoder_state_dropout'],
                                    code_token_embedder=embedder,
                                    vocab=vocab,
                                    no_copy=not args['decoder']['copy_token'])

        editor = Seq2SeqEditor(encoder, decoder, context_encoder=encoder, edit_encoder=edit_encoder, args=args)

        return editor


class Graph2TreeEditor(NeuralEditor):
    def __init__(self,
                 sequential_token_encoder,
                 graph_encoder,
                 decoder,
                 context_encoder,
                 edit_encoder: EditEncoder,
                 transition_system: TransitionSystem,
                 args):
        super().__init__(graph_encoder, decoder, context_encoder, edit_encoder, args)

        self.sequential_token_encoder = sequential_token_encoder
        self.transition_system = transition_system

    def forward(self, examples, return_change_vectors=False):
        """
        Given a batch of examples, compute the edit encodings, and use the edit encodings to predict the target data,
        return the log-likelihood of generating the target data
        """

        prev_data = [e.prev_data for e in examples]
        updated_data = [e.updated_data for e in examples]
        context_data = [e.context for e in examples]

        # prepare shared encodings
        prev_data_encoding = self.sequential_token_encoder(prev_data)

        context_encoding = self.context_encoder(context_data)

        updated_data_encoding = self.sequential_token_encoder(updated_data)

        # compute edit encoding
        edit_encoding = self.edit_encoder(examples, prev_data_encoding, updated_data_encoding)

        prev_data_graph_encoding = self.encoder([e.prev_code_ast for e in examples], prev_data_encoding.encoding)

        results = self.decoder(examples, prev_data_graph_encoding, context_encoding, edit_encoding)

        if return_change_vectors:
            results['edit_encoding'] = edit_encoding

        return results

    def get_edit_encoding(self, examples):
        prev_data = [e.prev_data for e in examples]
        updated_data = [e.updated_data for e in examples]

        prev_data_encoding = self.sequential_token_encoder(prev_data)
        updated_data_encoding = self.sequential_token_encoder(updated_data)

        edit_encoding = self.edit_encoder(examples, prev_data_encoding, updated_data_encoding)

        return edit_encoding

    def decode_updated_data(self, example, edit_encoding=None, with_change_vec=False, beam_size=5, debug=False):
        prev_data = [example.prev_data]
        updated_data = [example.updated_data]
        context_data = [example.context]

        # prepare shared encodings
        prev_data_encoding = self.sequential_token_encoder(prev_data)

        context_encoding = self.context_encoder(context_data)

        updated_data_encoding = self.sequential_token_encoder(updated_data)

        if edit_encoding is not None:
            edit_encoding = _prepare_edit_encoding(self, edit_encoding)
        elif with_change_vec:
            edit_encoding = self.edit_encoder([example], prev_data_encoding, updated_data_encoding)
        else:
            edit_encoding = torch.zeros(1, self.args['edit_encoder']['edit_encoding_size'], device=self.device)

        prev_data_graph_encoding = self.encoder([example.prev_code_ast], prev_data_encoding.encoding)

        hypotheses = self.decoder.beam_search_with_source_encodings(example.prev_code_ast, prev_data_graph_encoding,
                                                                    example.context, context_encoding,
                                                                    edit_encoding,
                                                                    beam_size=beam_size,
                                                                    max_decoding_time_step=70,
                                                                    transition_system=self.transition_system,
                                                                    debug=debug)

        return hypotheses

    @staticmethod
    def build(args, vocab=None, grammar=None):
        if not vocab:
            vocab = VocabEntry.load(args['dataset']['vocab_path'])
        if not grammar:
            grammar = ASDLGrammar.from_roslyn_xml(open(args['dataset']['grammar_path']).read(),
                                                  pruning=args['dataset']['prune_grammar'])

        embedder = SyntaxTreeEmbedder(args['embedder']['token_embed_size'],
                                      vocab,
                                      grammar,
                                      node_embed_method=args['embedder']['node_embed_method'])
        seq_token_encoder = SequentialEncoder(args['embedder']['token_embed_size'],
                                              args['encoder']['token_encoding_size'],
                                              token_embedder=embedder,
                                              vocab=vocab)

        edit_encoder = EditEncoder.build(args, vocab=vocab, embedder=embedder)

        graph_encoder = SyntaxTreeEncoder(hidden_size=args['encoder']['token_encoding_size'],
                                          syntax_tree_embedder=embedder,
                                          layer_timesteps=args['encoder']['layer_timesteps'],
                                          residual_connections=args['encoder']['residual_connections'],
                                          connections=args['encoder']['connections'],
                                          gnn_use_bias_for_message_linear=args['encoder']['use_bias_for_message_linear'],
                                          dropout=args['encoder']['dropout'],
                                          vocab=vocab,
                                          grammar=grammar)

        transition_decoder = TransitionDecoder(args['encoder']['token_encoding_size'],
                                               args['edit_encoder']['edit_encoding_size'],
                                               args['decoder']['hidden_size'],
                                               args['decoder']['action_embed_size'],
                                               args['decoder']['field_embed_size'],
                                               dropout=args['decoder']['dropout'],
                                               init_decode_vec_encoder_state_dropout=args['decoder']['init_decode_vec_encoder_state_dropout'],
                                               vocab=vocab, grammar=grammar,
                                               syntax_tree_encoder=graph_encoder,
                                               use_syntax_token_rnn=args['decoder']['use_syntax_token_rnn'],
                                               no_penalize_apply_tree_when_copy_subtree=args['decoder']['no_penalize_apply_tree_when_copy_subtree'],
                                               encode_change_vec_in_syntax_token_rnn=args['decoder']['encode_change_vec_in_syntax_token_rnn'],
                                               feed_in_token_rnn_state_to_rule_rnn=args['decoder']['feed_in_token_rnn_state_to_rule_rnn'],
                                               fuse_rule_and_token_rnns=args['decoder']['fuse_rule_and_token_rnns'],
                                               decoder_init_method=args['decoder']['init_method'],
                                               copy_identifier_node=args['decoder']['copy_identifier_node'],
                                               copy_syntax_token=args['decoder']['copy_token'],
                                               copy_sub_tree=args['decoder']['copy_subtree'])

        editor = Graph2TreeEditor(seq_token_encoder, graph_encoder, transition_decoder,
                                  context_encoder=seq_token_encoder,
                                  edit_encoder=edit_encoder,
                                  transition_system=TransitionSystem(grammar),
                                  args=args)

        return editor

    def save(self, model_path):
        NeuralEditor.save(self, model_path, kwargs=dict(grammar=self.decoder.grammar))


class SequentialAutoEncoder(nn.Module):
    def __init__(self,
                 token_embed_size, token_encoding_size, change_vector_size, change_tag_embed_size,
                 decoder_hidden_size, decoder_dropout, init_decode_vec_encoder_state_dropout,
                 vocab,
                 no_change_vector=False,
                 no_unchanged_token_encoding_in_diff_seq=False,
                 no_copy=False,
                 change_encoder_type='word',
                 token_embedder='word'):

        self.args = utils.get_method_args_dict(self.__init__, locals())
        super(SequentialAutoEncoder, self).__init__()

        if token_embedder == 'word':
            self.syntax_token_embedder = CodeTokenEmbedder(token_embed_size, vocab)
        elif token_embedder == 'char':
            self.syntax_token_embedder = ConvolutionalCharacterEmbedder(token_embed_size, max_character_size=20)

        self.sequential_code_encoder = SequentialCodeEncoder(token_embed_size, token_encoding_size,
                                                             code_token_embedder=self.syntax_token_embedder,
                                                             vocab=vocab)

        if change_encoder_type == 'word':
            self.code_change_encoder = SequentialChangeEncoder(token_encoding_size, change_vector_size,
                                                               change_tag_embed_size,
                                                               vocab,
                                                               no_unchanged_token_encoding_in_diff_seq=no_unchanged_token_encoding_in_diff_seq)
        elif change_encoder_type == 'bag':
            self.code_change_encoder = BagOfEditsChangeEncoder(self.syntax_token_embedder.weight,
                                                               vocab)

        self.decoder = SequentialDecoder(token_embed_size, token_encoding_size, change_vector_size, decoder_hidden_size,
                                         dropout=decoder_dropout,
                                         init_decode_vec_encoder_state_dropout=init_decode_vec_encoder_state_dropout,
                                         code_token_embedder=self.syntax_token_embedder,
                                         vocab=vocab,
                                         no_copy=no_copy)

        self.vocab = vocab

    @property
    def device(self):
        return self.code_change_encoder.device

    def forward(self, examples, return_change_vectors=False):
        previous_code_chunk_list = [e.previous_code_chunk for e in examples]
        updated_code_chunk_list = [e.updated_code_chunk for e in examples]
        context_list = [e.context for e in examples]

        embedding_cache = EmbeddingTable(
            chain.from_iterable(previous_code_chunk_list + updated_code_chunk_list + context_list))
        self.syntax_token_embedder.populate_embedding_table(embedding_cache)

        batched_prev_code = self.sequential_code_encoder.encode(previous_code_chunk_list,
                                                                embedding_cache=embedding_cache)
        batched_updated_code = self.sequential_code_encoder.encode(updated_code_chunk_list,
                                                                   embedding_cache=embedding_cache)
        batched_context = self.sequential_code_encoder.encode(context_list, embedding_cache=embedding_cache)

        if self.args['no_change_vector'] is False:
            change_vectors = self.code_change_encoder(examples, batched_prev_code, batched_updated_code)
        else:
            change_vectors = torch.zeros(batched_updated_code.batch_size, self.args['change_vector_size'], device=self.device)

        scores = self.decoder(examples, batched_prev_code, batched_context, change_vectors, embedding_cache=embedding_cache)

        if return_change_vectors:
            return scores, change_vectors
        else:
            return scores

    def decode_updated_code(self, example, with_change_vec=False, change_vec=None, beam_size=5, debug=False):
        previous_code_chunk_list = [example.previous_code_chunk]
        updated_code_chunk_list = [example.updated_code_chunk]
        context_list = [example.context]

        embedding_cache = EmbeddingTable(
            chain.from_iterable(previous_code_chunk_list + updated_code_chunk_list + context_list))
        self.syntax_token_embedder.populate_embedding_table(embedding_cache)

        batched_prev_code = self.sequential_code_encoder.encode(previous_code_chunk_list,
                                                                embedding_cache=embedding_cache)
        batched_updated_code = self.sequential_code_encoder.encode(updated_code_chunk_list,
                                                                   embedding_cache=embedding_cache)
        batched_context = self.sequential_code_encoder.encode(context_list, embedding_cache=embedding_cache)

        if change_vec is not None:
            change_vectors = torch.from_numpy(change_vec).to(self.device)
            if len(change_vectors.size()) == 1:
                change_vectors = change_vectors.unsqueeze(0)
        elif with_change_vec:
            change_vectors = self.code_change_encoder([example], batched_prev_code, batched_updated_code)
        else:
            change_vectors = torch.zeros(batched_updated_code.batch_size, self.args['change_vector_size'],
                                         device=self.device)

        hypotheses = self.decoder.beam_search_with_source_encodings(example.previous_code_chunk, batched_prev_code,
                                                                    example.context, batched_context,
                                                                    change_vectors,
                                                                    beam_size=beam_size, max_decoding_time_step=70,
                                                                    debug=debug)

        return hypotheses

    def save(self, model_path):
        params = {
            'args': self.args,
            'vocab': self.vocab,
            'state_dict': self.state_dict()
        }

        torch.save(params, model_path)

    @staticmethod
    def load(model_path, use_cuda=True):
        device = torch.device("cuda:0" if use_cuda else "cpu")
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        args = params['args']
        model = SequentialAutoEncoder(vocab=params['vocab'], **args)
        model.load_state_dict(params['state_dict'])
        model = model.to(device)

        return model


class TreeBasedAutoEncoderWithGraphEncoder(nn.Module):
    def __init__(self,
                 token_embed_size, token_encoding_size, change_vector_size, change_tag_embed_size,
                 action_embed_size, field_embed_size,
                 decoder_hidden_size, decoder_dropout, init_decode_vec_encoder_state_dropout,
                 gnn_layer_timesteps, gnn_residual_connections, gnn_dropout,
                 vocab,
                 grammar,
                 mode,
                 no_change_vector=False,
                 no_unchanged_token_encoding_in_diff_seq=False,
                 use_syntax_token_rnn=False,
                 change_encoder_type='word',
                 token_embedder='word',
                 node_embed_method='type',
                 no_penalize_apply_tree_when_copy_subtree=False,
                 encode_change_vec_in_syntax_token_rnn=False,
                 feed_in_token_rnn_state_to_rule_rnn=False,
                 fuse_rule_and_token_rnns=False,
                 gnn_no_token_connection=False,
                 gnn_no_top_down_connection=False,
                 gnn_no_bottom_up_connection=False,
                 gnn_prev_sibling_connection=False,
                 gnn_next_sibling_connection=False,
                 copy_identifier=True,
                 decoder_init_method='avg_pooling',
                 gnn_use_bias_for_message_linear=True,
                 change_encoder_master_node_option=None,
                 no_copy=False):

        self.args = utils.get_method_args_dict(self.__init__, locals())
        super(TreeBasedAutoEncoderWithGraphEncoder, self).__init__()

        self.syntax_tree_node_embedder = SyntaxTreeEmbedder(token_embed_size, vocab, grammar, node_embed_method=node_embed_method)

        if token_embedder == 'word':
            self.syntax_token_embedder = self.syntax_tree_node_embedder
        elif token_embedder == 'char':
            self.syntax_token_embedder = ConvolutionalCharacterEmbedder(token_embed_size, max_character_size=20)

        self.sequential_code_encoder = SequentialCodeEncoder(token_embed_size, token_encoding_size,
                                                             code_token_embedder=self.syntax_token_embedder,
                                                             vocab=vocab)

        if change_encoder_type == 'word':
            self.code_change_encoder = SequentialChangeEncoder(token_encoding_size, change_vector_size, change_tag_embed_size,
                                                               vocab,
                                                               no_unchanged_token_encoding_in_diff_seq=no_unchanged_token_encoding_in_diff_seq)
        elif change_encoder_type == 'graph':
            self.code_change_encoder = GraphChangeEncoder(change_vector_size, syntax_tree_embedder=self.syntax_tree_node_embedder,
                                                          layer_time_steps=gnn_layer_timesteps,
                                                          dropout=gnn_dropout,
                                                          gnn_use_bias_for_message_linear=gnn_use_bias_for_message_linear,
                                                          master_node_option=change_encoder_master_node_option)
        elif change_encoder_type == 'hybrid':
            self.code_change_encoder = HybridChangeEncoder(token_encoding_size=token_encoding_size,
                                                           change_vector_dim=change_vector_size,
                                                           syntax_tree_embedder=self.syntax_tree_node_embedder,
                                                           layer_timesteps=gnn_layer_timesteps,
                                                           dropout=gnn_dropout,
                                                           vocab=vocab,
                                                           gnn_use_bias_for_message_linear=gnn_use_bias_for_message_linear)
        elif change_encoder_type == 'bag':
            self.code_change_encoder = BagOfEditsChangeEncoder(self.syntax_token_embedder.weight,
                                                               vocab)

        else:
            raise ValueError('unknown code change encoder type %s' % change_encoder_type)

        self.prev_ast_encoder = GraphCodeEncoder(hidden_size=token_encoding_size,
                                                 syntax_tree_embedder=self.syntax_tree_node_embedder,
                                                 layer_timesteps=gnn_layer_timesteps, residual_connections=gnn_residual_connections, dropout=gnn_dropout,
                                                 vocab=vocab, grammar=grammar,
                                                 token_bidirectional_connection=not gnn_no_token_connection,
                                                 top_down_connection=not gnn_no_top_down_connection,
                                                 bottom_up_connection=not gnn_no_bottom_up_connection,
                                                 prev_sibling_connection=gnn_prev_sibling_connection,
                                                 next_sibling_connection=gnn_next_sibling_connection,
                                                 gnn_use_bias_for_message_linear=gnn_use_bias_for_message_linear)

        if '2tree' in mode:
            self.decoder = TransitionDecoderWithGraphEncoder(node_encoding_size=token_encoding_size,
                                                             change_vector_size=change_vector_size,
                                                             hidden_size=decoder_hidden_size,
                                                             action_embed_size=action_embed_size,
                                                             field_embed_size=field_embed_size,
                                                             dropout=decoder_dropout,
                                                             init_decode_vec_encoder_state_dropout=init_decode_vec_encoder_state_dropout,
                                                             vocab=vocab, grammar=grammar, mode=mode,
                                                             syntax_tree_embedder=self.syntax_tree_node_embedder,
                                                             use_syntax_token_rnn=use_syntax_token_rnn,
                                                             no_penalize_apply_tree_when_copy_subtree=no_penalize_apply_tree_when_copy_subtree,
                                                             encode_change_vec_in_syntax_token_rnn=encode_change_vec_in_syntax_token_rnn,
                                                             feed_in_token_rnn_state_to_rule_rnn=feed_in_token_rnn_state_to_rule_rnn,
                                                             fuse_rule_and_token_rnns=fuse_rule_and_token_rnns,
                                                             decoder_init_method=decoder_init_method,
                                                             copy_identifier=copy_identifier,
                                                             no_copy=no_copy)
        else:
            self.decoder = SequentialDecoderWithTreeEncoder(token_embed_size, token_encoding_size, change_vector_size,
                                                            decoder_hidden_size,
                                                            dropout=decoder_dropout,
                                                            init_decode_vec_encoder_state_dropout=init_decode_vec_encoder_state_dropout,
                                                            code_token_embedder=self.syntax_token_embedder,
                                                            vocab=vocab,
                                                            decoder_init_method=decoder_init_method)

        self.vocab = vocab
        self.grammar = grammar

    @property
    def device(self):
        return self.code_change_encoder.device

    def forward(self, examples, return_change_vectors=False, **kwargs):
        previous_code_chunk_list = [e.previous_code_chunk for e in examples]
        updated_code_chunk_list = [e.updated_code_chunk for e in examples]
        context_list = [e.context for e in examples]

        embedding_cache = EmbeddingTable(chain.from_iterable(previous_code_chunk_list + updated_code_chunk_list + context_list))
        self.syntax_token_embedder.populate_embedding_table(embedding_cache)

        batched_prev_code = self.sequential_code_encoder.encode(previous_code_chunk_list, embedding_cache=embedding_cache)
        batched_updated_code = self.sequential_code_encoder.encode(updated_code_chunk_list, embedding_cache=embedding_cache)
        batched_context = self.sequential_code_encoder.encode(context_list, embedding_cache=embedding_cache)

        if self.args['no_change_vector'] is False:
            change_vectors = self.code_change_encoder(examples, batched_prev_code, batched_updated_code)
        else:
            change_vectors = torch.zeros(batched_updated_code.batch_size, self.args['change_vector_size'],
                                         device=self.device)

        batched_prev_ast_node_encoding, \
        batched_prev_ast_node_mask, \
        batched_prev_ast_syntax_token_mask = self.prev_ast_encoder([e.prev_code_ast for e in examples], batched_prev_code.encoding)

        batched_prev_asts = type('BatchedDatum', (object,), {'encoding': batched_prev_ast_node_encoding,
                                                             'mask': batched_prev_ast_node_mask,
                                                             'syntax_token_mask': batched_prev_ast_syntax_token_mask})

        results = self.decoder(examples, batched_prev_asts, batched_context, change_vectors, embedding_cache=embedding_cache, **kwargs)

        if return_change_vectors:
            return results, change_vectors
        else:
            return results

    def decode_updated_code(self, example, transition_system, with_change_vec=False, change_vec=None, beam_size=5, debug=False):
        previous_code_chunk_list = [example.previous_code_chunk]
        updated_code_chunk_list = [example.updated_code_chunk]
        context_list = [example.context]

        embedding_cache = EmbeddingTable(
            chain.from_iterable(previous_code_chunk_list + updated_code_chunk_list + context_list))
        self.syntax_token_embedder.populate_embedding_table(embedding_cache)

        batched_prev_code = self.sequential_code_encoder.encode(previous_code_chunk_list,
                                                                embedding_cache=embedding_cache)
        batched_updated_code = self.sequential_code_encoder.encode(updated_code_chunk_list,
                                                                   embedding_cache=embedding_cache)
        batched_context = self.sequential_code_encoder.encode(context_list, embedding_cache=embedding_cache)

        if change_vec is not None:
            change_vectors = torch.from_numpy(change_vec).to(self.device)
            if len(change_vectors.size()) == 1:
                change_vectors = change_vectors.unsqueeze(0)
        elif with_change_vec:
            change_vectors = self.code_change_encoder([example], batched_prev_code, batched_updated_code)
        else:
            change_vectors = torch.zeros(batched_updated_code.batch_size, self.args['change_vector_size'],
                                         device=self.device)

        batched_prev_ast_node_encoding, \
        batched_prev_ast_node_mask, \
        batched_prev_ast_syntax_token_mask = self.prev_ast_encoder([example.prev_code_ast],
                                                                   batched_prev_code.encoding)

        batched_prev_asts = type('BatchedDatum', (object,), {'encoding': batched_prev_ast_node_encoding,
                                                             'mask': batched_prev_ast_node_mask,
                                                             'syntax_token_mask': batched_prev_ast_syntax_token_mask})

        hypotheses = self.decoder.beam_search_with_source_encodings(example.prev_code_ast, batched_prev_asts,
                                                                    example.context, batched_context,
                                                                    change_vectors,
                                                                    beam_size=beam_size, max_decoding_time_step=70,
                                                                    transition_system=transition_system, debug=debug)

        return hypotheses

    def save(self, model_path):
        params = {
            'args': self.args,
            'vocab': self.vocab,
            'grammar': self.grammar,
            'state_dict': self.state_dict()
        }

        torch.save(params, model_path)

    @staticmethod
    def load(model_path, use_cuda=True):
        device = torch.device("cuda:0" if use_cuda else "cpu")
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        args = params['args']
        model = TreeBasedAutoEncoderWithGraphEncoder(vocab=params['vocab'], grammar=params['grammar'], **args)
        model.load_state_dict(params['state_dict'])
        model = model.to(device)

        return model


class WordPredictionMultiTask(nn.Module):
    def __init__(self, change_vector_size, vocab, device):
        super(WordPredictionMultiTask, self).__init__()

        self.vocab = vocab
        self.device = device
        self.change_vec_to_vocab = nn.Linear(change_vector_size, len(vocab))
        self.words_to_discard = {'VAR0', 'int', 'long', 'string', 'float', 'LITERAL', 'var'}

    def forward(self, examples, change_vecs):
        # change_vecs: (batch_size, change_vec_size)

        # (batch_size, max_word_num)
        tgt_word_ids, tgt_word_mask = self.get_word_ids_to_predict(examples)

        # (batch_size, vocab_size)
        log_probs = F.log_softmax(self.change_vec_to_vocab(change_vecs), dim=-1)

        tgt_log_probs = torch.gather(log_probs, 1, tgt_word_ids)
        tgt_log_probs = (tgt_log_probs * tgt_word_mask).sum(dim=-1)
        tgt_log_probs = tgt_log_probs / (tgt_word_mask.sum(dim=-1) + 1e-7)  # to avoid underflow

        return tgt_log_probs

    def get_word_ids_to_predict(self, examples):
        tgt_words = []
        for example in examples:
            example_tgt_words = []

            example_tgt_words.extend(filter(lambda x: x not in self.words_to_discard and not all(c in string.punctuation for c in x), example.previous_code_chunk))
            example_tgt_words.extend(filter(lambda x: x not in self.words_to_discard and not all(c in string.punctuation for c in x), example.updated_code_chunk))

            tgt_words.append(example_tgt_words)
            # if len(example_tgt_words) == 0:
            #     print(example.prev_data)
            #     print(example.updated_data)

        max_word_num = max(len(x) for x in tgt_words)
        tgt_word_ids = torch.zeros(len(examples), max_word_num, dtype=torch.long, device=self.device)
        tgt_word_mask = torch.zeros(len(examples), max_word_num, dtype=torch.float, device=self.device)

        for batch_id, example_words in enumerate(tgt_words):
            tgt_word_ids[batch_id, :len(example_words)] = torch.LongTensor([self.vocab[word] for word in example_words], device=self.device)
            tgt_word_mask[batch_id, :len(example_words)] = 1

        return tgt_word_ids, tgt_word_mask


class ChangedWordPredictionMultiTask(nn.Module):
    def __init__(self, change_vector_size, vocab, device):
        super(ChangedWordPredictionMultiTask, self).__init__()

        self.vocab = vocab
        self.device = device
        self.change_vec_to_vocab = nn.Linear(change_vector_size, len(vocab) * 2)
        self.offset = len(vocab)
        self.words_to_discard = {'VAR', 'LITERAL', 'var'}  # 'int', 'long', 'string', 'float', 

    def forward(self, examples, change_vecs):
        # change_vecs: (batch_size, change_vec_size)

        # (batch_size, max_word_num)
        tgt_word_ids, tgt_word_mask = self.get_word_ids_to_predict(examples)

        if len(tgt_word_ids.size()) == 1:
            return None

        # (batch_size, vocab_size)
        log_probs = F.log_softmax(self.change_vec_to_vocab(change_vecs), dim=-1)

        tgt_log_probs = torch.gather(log_probs, 1, tgt_word_ids)
        tgt_log_probs = (tgt_log_probs * tgt_word_mask).sum(dim=-1)
        tgt_log_probs = tgt_log_probs / (tgt_word_mask.sum(dim=-1) + 1e-7)  # to avoid underflow

        return tgt_log_probs

    def get_changed_words_from_change_seq(self, change_seq):
        add_del_words = []
        for entry in change_seq:
            tag, token = entry

            if tag == 'ADD':
                add_del_words.append(('ADD', token))
            elif tag == 'DEL':
                add_del_words.append(('DEL', token))
            elif tag == 'REPLACE':
                add_del_words.append(('DEL', token[0]))
                add_del_words.append(('ADD', token[1]))

        add_del_words = list(filter(lambda t: t[1] not in self.words_to_discard and \
                                                   not t[1].startswith('VAR') and \
                                                   not all(c in string.punctuation for c in t[1]), add_del_words))

        return add_del_words

    def get_word_ids_to_predict(self, examples):
        tgt_words = []
        for example in examples:
            example_tgt_words = self.get_changed_words_from_change_seq(example.change_seq)
            tgt_words.append(example_tgt_words)

        max_word_num = max(len(x) for x in tgt_words)
        tgt_word_ids = torch.zeros(len(examples), max_word_num, dtype=torch.long, device=self.device)
        tgt_word_mask = torch.zeros(len(examples), max_word_num, dtype=torch.float, device=self.device)

        for batch_id, example_words in enumerate(tgt_words):
            if len(example_words) > 0:
                tgt_word_ids[batch_id, :len(example_words)] = torch.LongTensor([self.vocab[word] if tag == 'ADD' else (self.offset + self.vocab[word]) 
                                                                                for tag, word in example_words], 
                                                                            device=self.device)
                tgt_word_mask[batch_id, :len(example_words)] = 1

        return tgt_word_ids, tgt_word_mask
