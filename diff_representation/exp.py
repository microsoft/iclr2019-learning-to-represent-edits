#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Neural Representations of code revisions

Usage:
    exp.py (train | train_simple) --mode=<str>  --train_file=<file> --dev_file=<file> --vocab=<file> [options]
    exp.py test_ppl --mode=<str> MODEL_PATH TEST_SET_PATH [options]
    exp.py decode_updated_code --mode=<str> MODEL_PATH TEST_SET_PATH [options]
    exp.py decode_updated_code_for_neighbors --mode=<str> MODEL_PATH TEST_SET_PATH SEED_QUERIES [options]

Options:
    -h --help                               Show this screen.
    --mode=<str>                            Mode: seq2seq|seq2tree|tree2tree|tree2seq
    --grammar_path=<file>                   Path to the grammar file
    --prune_grammar                         Prune the grammar
    --train_file=<file>                     Train file
    --dev_file=<file>                       Dev file
    --vocab=<file>                          Vocab file
    --cuda                                  Use GPU
    --seed=<int>                            Seed [default: 0]
    --batch_size=<int>                      batch size [default: 32]
    --token_embed_size=<int>                token_embed_size [default: 128]
    --token_encoding_size=<int>             token_encoding_size [default: 256]
    --change_vector_size=<int>              change_vector_size [default: 256]
    --change_tag_embed_size=<int>           change_tag_embed_size [default: 32]
    --action_embed_size=<int>               action embed size [default: 128]
    --field_embed_size=<int>                field embed size [default: 64]
    --decoder_hidden_size=<int>             decoder_hidden_size [default: 256]
    --decoder_dropout=<float>               decoder dropout [default: 0.3]
    --gnn_layer_timesteps=<string>          layer_timesteps [default: [5, 1, 5, 1]]
    --gnn_residual_connections=<string>     residual_connections [default: {1: [0], 3: [0, 1]}]
    --gnn_dropout=<float>                   gnn dropout [default: 0.2]
    --token_embedder=<choice>               type of embedder, char or word [default: word]
    --clip_grad=<float>                     gradient clipping [default: 5.0]
    --max_change_sequence_length=<int>      maximum length of change sequences allowed [default: 70]
    --max_action_sequence_length=<int>      maximum length of action sequences allowed [default: 200]
    --work_dir=<dir>                        work dir [default: exp_runs/]
    --log_every=<int>                       log every [default: 10]
    --max_epoch=<int>                       max epoch [default: 50]
    --patience=<int>                        [default: 5]
    --max_num_trial=<int>                   [default: 5]
    --lr_decay=<float>                      [default: 0.5]
    --reset_optimizer                       Reset the optimizer
    --no_change_vector                      No change vector
    --debug                                 Debug mode
    --no_unchanged_token_encoding_in_diff_seq   Do not use the encoding of unchanged tokens, but only
                                                using their tags in the diff seq
    --num_data_load_worker=<int>            Number of worker processes for parallel data loading [default: 10]
    --word_predict_loss_weight=<float>      Word prediction loss weight [default: 0.]
    --init_decode_vec_encoder_state_dropout=<float>       Dropout encoding vector [default: 0.]
    --sample_size=<int>                     Sample size [default: 1]
    --beam_size=<int>                       Beam size [default: 5]
    --save_each_epoch                       Save results from each epoch
    --filter_dataset                        Filter the dataset when decoding
    --use_syntax_token_rnn                  Use word rnn
    --node_embed_method=<str>               How to embed syntax nodes, choose between type and type_and_field [default: type]
    --no_penalize_apply_tree_when_copy_subtree  Simple trick to not penalizing applytree actions
    --encode_change_vec_in_syntax_token_rnn     Encode change vector in syntax token RNN
    --feed_in_token_rnn_state_to_rule_rnn   Feed the state of syntax token RNN to rulle RNN
    --fuse_rule_and_token_rnns              Concatenate two RNN states for attention
    --gnn_no_token_connection               No bidirectional token connections in GNN
    --gnn_no_top_down_connection            No top-down connection
    --gnn_no_bottom_up_connection           No bottom-up connection
    --gnn_next_sibling_connection           Use next sibling connection
    --gnn_prev_sibling_connection           Use previous sibling connection
    --gnn_use_bias_for_message_linear       Use bias in message linears for GNNs
    --K=<int>                               K for nearest neighbors search [default: 10]
    --decoder_init_method=<str>             How to initialize the deocder [default: avg_pooling]
    --no_copy_identifier                    Do not copy identifiers
    --change_encoder_type=<str>             Type of change encoder: word, graph, bag [default: word]
    --change_encoder_master_node_option=<str>   Type of master nodes used in the change encoder (None|single_node|double_node) [default: None]
    --no_copy                               Do not copy terminal syntax tokens
    --no_tensorization                      Do not perform pre-tensorization
    --evaluate_ppl                          Evaluate perplexity as well
    --compute_upper_bound                   Compute the upper bound for zero shot learning
"""
import json
import os
import shutil
import sys
import time
import gc
import subprocess
import urllib.request as request
import pickle
from collections import OrderedDict

from docopt import docopt
from tqdm import tqdm

from diff_representation.asdl.grammar import ASDLGrammar
from diff_representation.asdl.transition_system.transition import TransitionSystem
from diff_representation.model.autoencoder import SequentialAutoEncoder, TreeBasedAutoEncoder, \
    TreeBasedAutoEncoderWithGraphEncoder, WordPredictionMultiTask, ChangedWordPredictionMultiTask
from diff_representation.dataset import DataSet
from diff_representation.change_entry import ChangeExample
from diff_representation.model.encoder import *
from diff_representation.model.sequential_decoder import *
from diff_representation.vocab import Vocab, VocabEntry
from diff_representation.evaluate import evaluate_nll
from diff_representation.utils.decode import *
from diff_representation.utils.utils import get_entry_str
from diff_representation.utils.relevance import get_nn


def parse_arg(arg_val):
    if arg_val == 'None': return None
    else: return arg_val


def train(args):
    mode = args['--mode']

    vocab = VocabEntry.load(args['--vocab'])
    print('vocab: %s' % repr(vocab), file=sys.stderr)

    if not os.path.exists(args['--work_dir']):
        os.makedirs(args['--work_dir'])

    # save arguments to work dir
    json.dump(OrderedDict([(key, args[key]) for key in sorted(args)]), open(os.path.join(args['--work_dir'], 'args.json'), 'w'), indent=2)

    auto_encoder_args = dict(token_embed_size=int(args['--token_embed_size']),
                             token_encoding_size=int(args['--token_encoding_size']),
                             change_vector_size=int(args['--change_vector_size']),
                             change_tag_embed_size=int(args['--change_tag_embed_size']),
                             decoder_hidden_size=int(args['--decoder_hidden_size']),
                             decoder_dropout=float(args['--decoder_dropout']),
                             vocab=vocab,
                             no_change_vector=args['--no_change_vector'],
                             no_unchanged_token_encoding_in_diff_seq=args['--no_unchanged_token_encoding_in_diff_seq'],
                             token_embedder=args['--token_embedder'],
                             init_decode_vec_encoder_state_dropout=float(args['--init_decode_vec_encoder_state_dropout']),
                             no_copy=args['--no_copy'])

    device = torch.device("cuda:0" if args['--cuda'] else "cpu")
    print('use device: %s' % device, file=sys.stderr)

    if mode == 'seq2seq':
        print('Loading datasets...', file=sys.stderr)
        train_set = DataSet.load_from_jsonl(args['--train_file'],
                                            max_workers=int(args['--num_data_load_worker']),
                                            parallel=int(args['--num_data_load_worker']) > 1,
                                            tensorization=not args['--no_tensorization'],
                                            vocab=vocab,
                                            no_copy=args['--no_copy'])
        dev_set = DataSet.load_from_jsonl(args['--dev_file'],
                                          max_workers=int(args['--num_data_load_worker']),
                                          parallel=int(args['--num_data_load_worker']) > 1,
                                          tensorization=not args['--no_tensorization'],
                                          vocab=vocab,
                                          no_copy=args['--no_copy'])
        auto_encoder_args['change_encoder_type'] = args['--change_encoder_type']
        model = SequentialAutoEncoder(**auto_encoder_args)
    elif mode in ('seq2tree', 'tree2seq', 'tree2tree', 'tree2tree_subtree_copy'):
        print('Loading grammar model...', file=sys.stderr)
        if args['--grammar_path']:
            grammar_text = open(args['--grammar_path']).read()
        else:
            grammar_text = request.urlopen('https://raw.githubusercontent.com/dotnet/roslyn/master/src/Compilers'
                                           '/CSharp/Portable/Syntax/Syntax.xml').read()

        grammar = ASDLGrammar.from_roslyn_xml(grammar_text, pruning=args['--prune_grammar'])
        transition_system = TransitionSystem(grammar)

        print('Loading datasets...', file=sys.stderr)
        train_set = ParallelLazyDataLoader(args['--train_file'],
                                           type=mode, transition_system=transition_system,
                                           max_workers=int(args['--num_data_load_worker']),
                                           debug=args['--debug'],
                                           copy_identifier=not args['--no_copy_identifier'],
                                           annotate_tree_change=args['--change_encoder_type'] != 'word',
                                           tensorization=not args['--no_tensorization'],
                                           vocab=vocab,
                                           no_copy=args['--no_copy'])
        dev_set = ParallelLazyDataLoader(args['--dev_file'],
                                         type=mode, transition_system=transition_system,
                                         max_workers=int(args['--num_data_load_worker']),
                                         debug=args['--debug'],
                                         copy_identifier=not args['--no_copy_identifier'],
                                         annotate_tree_change=args['--change_encoder_type'] != 'word',
                                         tensorization=not args['--no_tensorization'],
                                         vocab=vocab,
                                         no_copy=args['--no_copy'])

        print('Completed!', file=sys.stderr)

        auto_encoder_args['action_embed_size'] = int(args['--action_embed_size'])
        auto_encoder_args['field_embed_size'] = int(args['--field_embed_size'])
        auto_encoder_args['grammar'] = grammar
        auto_encoder_args['mode'] = mode
        auto_encoder_args['use_syntax_token_rnn'] = args['--use_syntax_token_rnn']
        auto_encoder_args['encode_change_vec_in_syntax_token_rnn'] = args['--encode_change_vec_in_syntax_token_rnn']
        auto_encoder_args['feed_in_token_rnn_state_to_rule_rnn'] = args['--feed_in_token_rnn_state_to_rule_rnn']
        auto_encoder_args['fuse_rule_and_token_rnns'] = args['--fuse_rule_and_token_rnns']

        if mode in ('tree2tree', 'tree2tree_subtree_copy', 'tree2seq'):
            auto_encoder_args['gnn_dropout'] = float(args['--gnn_dropout'])
            auto_encoder_args['gnn_layer_timesteps'] = eval(args['--gnn_layer_timesteps'])
            auto_encoder_args['gnn_residual_connections'] = eval(args['--gnn_residual_connections'])
            auto_encoder_args['node_embed_method'] = args['--node_embed_method']
            auto_encoder_args['no_penalize_apply_tree_when_copy_subtree'] = args['--no_penalize_apply_tree_when_copy_subtree']

            auto_encoder_args['gnn_no_token_connection'] = args['--gnn_no_token_connection']
            auto_encoder_args['gnn_no_top_down_connection'] = args['--gnn_no_top_down_connection']
            auto_encoder_args['gnn_no_bottom_up_connection'] = args['--gnn_no_bottom_up_connection']
            auto_encoder_args['gnn_prev_sibling_connection'] = args['--gnn_prev_sibling_connection']
            auto_encoder_args['gnn_next_sibling_connection'] = args['--gnn_next_sibling_connection']
            auto_encoder_args['decoder_init_method'] = args['--decoder_init_method']
            auto_encoder_args['copy_identifier'] = not args['--no_copy_identifier']
            auto_encoder_args['change_encoder_type'] = args['--change_encoder_type']
            auto_encoder_args['gnn_use_bias_for_message_linear'] = args['--gnn_use_bias_for_message_linear']
            auto_encoder_args['change_encoder_master_node_option'] = parse_arg(args['--change_encoder_master_node_option'])

            model = TreeBasedAutoEncoderWithGraphEncoder(**auto_encoder_args)
        elif mode == 'seq2tree':
            model = TreeBasedAutoEncoder(**auto_encoder_args)
    else:
        raise ValueError('unknown mode')

    model = model.to(device)
    model.train()

    batch_size = int(args['--batch_size'])
    epoch = train_iter = num_trial = 0
    # accumulate statistics on the device
    report_loss = 0.
    report_word_predict_loss = 0.
    report_examples = 0
    patience = 0
    max_change_sequence_length = int(args['--max_change_sequence_length'])
    max_action_sequence_length = int(args['--max_action_sequence_length'])
    history_dev_scores = []

    # if args['--filter_dataset']:
    #     print('filter dataset before training', file=sys.stderr)
    #     # remove weired examples from training set
    #     train_examples = [e for e in train_set.examples if len(e.change_seq) <= max_change_sequence_length
    #                     and len(e.context) <= max_change_sequence_length and len(e.tgt_actions) <= max_action_sequence_length]
    #     train_set = DataSet(train_examples)

    print('loaded train file at [%s] (size=%d), dev file at [%s] (size=%d)' % (
                                                                        args['--train_file'], len(train_set),
                                                                        args['--dev_file'], len(dev_set)), file=sys.stderr)

    word_predict_loss_weight = float(args['--word_predict_loss_weight'])

    parameters = list(model.parameters())

    if word_predict_loss_weight > 0.:
        word_predict_multi_task = ChangedWordPredictionMultiTask(model.code_change_encoder.change_vector_size, model.vocab, model.device)
        word_predict_multi_task = word_predict_multi_task.to(device)
        parameters.extend(word_predict_multi_task.parameters())

    optimizer = torch.optim.Adam(parameters, lr=0.001)

    while True:
        epoch += 1
        epoch_begin = time.time()
        epoch_cum_examples = 0.

        for batch_examples in train_set.batch_iter(batch_size=batch_size, shuffle=True):
            # print(f'batch size: {len(batch_examples)}, total change_seq_len: {sum(len(e.change_seq) for e in batch_examples)}, total context_len: {sum(len(e.context) for e in batch_examples)}, total decode_time_steps: {sum(len(e.tgt_actions) for e in batch_examples)}, max change_seq_len: {max(len(e.change_seq) for e in batch_examples)}, max context len: {max(len(e.context) for e in batch_examples)}, max decode_time_steps: {max(len(e.tgt_actions) for e in batch_examples)}', file=sys.stderr)

            train_iter += 1

            try:
                optimizer.zero_grad()

                log_probs, change_vecs = model(batch_examples, return_change_vectors=True)
                loss = -log_probs.mean()

                total_loss_val = (-log_probs).sum().item()
                report_loss += total_loss_val
                report_examples += len(batch_examples)
                epoch_cum_examples += len(batch_examples)

                if word_predict_loss_weight > 0:
                    word_predict_loss = word_predict_multi_task(batch_examples, change_vecs)
                    if word_predict_loss is not None:
                        report_word_predict_loss += -word_predict_loss.sum().item()

                        word_predict_loss = -word_predict_loss.mean()
                        loss = loss + word_predict_loss_weight * word_predict_loss

                loss.backward()

                # clip gradient
                grad_norm = torch.nn.utils.clip_grad_norm_(parameters, float(args['--clip_grad']))

                optimizer.step()
            except RuntimeError as e:
                err_message = getattr(e, 'message', str(e))
                if 'out of memory' in err_message:
                    print('OOM exception encountered, will skip this batch with examples:', file=sys.stderr)
                    for example in batch_examples:
                        print('\t%s' % example.id, file=sys.stderr)

                    try:
                        del loss, log_probs, change_vecs
                    except: pass

                    if word_predict_loss_weight > 0.: del word_predict_loss
                    gc_start = time.time()
                    gc.collect()
                    gc_time = time.time() - gc_start
                    print(f'gc took {gc_time}s', file=sys.stderr)
                    torch.cuda.empty_cache()
                    continue
                else: raise e

            del loss, log_probs, change_vecs
            if word_predict_loss_weight > 0. and word_predict_loss is not None: del word_predict_loss

            if train_iter % int(args['--log_every']) == 0:
                print('[Iter %d] encoder loss=%.5f, word prediction loss=%.5f, %.2fs/epoch' %
                      (train_iter,
                       report_loss / report_examples, report_word_predict_loss / report_examples, (time.time() - epoch_begin) / epoch_cum_examples * len(train_set)),
                      file=sys.stderr)

                report_loss = 0.
                report_examples = 0.
                report_word_predict_loss = 0.

        print('[Epoch %d] epoch elapsed %ds' % (epoch, time.time() - epoch_begin), file=sys.stderr)

        # perform validation
        print('[Epoch %d] begin validation' % epoch, file=sys.stderr)
        eval_start = time.time()
        # evaluate ppl
        dev_nll, dev_ppl = evaluate_nll(model, dev_set, batch_size=batch_size)
        print('[Epoch %d] average negative log likelihood=%.5f, average ppl=%.5f took %ds' % (epoch, dev_nll, dev_ppl, time.time() - eval_start), file=sys.stderr)

        # try: print(subprocess.check_output(['nvidia-smi']), file=sys.stderr)
        # except: pass
        # print('[Epoch %d] manual gc...' % epoch, file=sys.stderr)
        # gc_start = time.time()
        # gc.collect()
        # print('[Epoch %d] gc took %ds' % (epoch, time.time() - gc_start), file=sys.stderr)
        # try: print(subprocess.check_output(['nvidia-smi']), file=sys.stderr)
        # except: pass

        dev_score = -dev_nll
        is_better = history_dev_scores == [] or dev_score > max(history_dev_scores)
        history_dev_scores.append(dev_score)

        if args['--save_each_epoch']:
            save_path = os.path.join(args['--work_dir'], f'model.epoch{epoch}.bin')
            print('[Epoch %d] save model to [%s]' % (epoch, save_path), file=sys.stderr)
            model.save(save_path)

        if is_better:
            patience = 0
            save_path = os.path.join(args['--work_dir'], 'model.bin')
            print('save currently the best model to [%s]' % save_path, file=sys.stderr)
            model.save(save_path)

            # also save the optimizers' state
            torch.save(optimizer.state_dict(), os.path.join(args['--work_dir'], 'optim.bin'))
        elif patience < int(args['--patience']):
            patience += 1
            print('hit patience %d' % patience, file=sys.stderr)

        if patience == int(args['--patience']):
            num_trial += 1
            print('hit #%d trial' % num_trial, file=sys.stderr)
            if num_trial == int(args['--max_num_trial']):
                print('early stop!', file=sys.stderr)
                exit(0)

            # decay lr, and restore from previously best checkpoint
            lr = optimizer.param_groups[0]['lr'] * float(args['--lr_decay'])
            print('load previously best model and decay learning rate to %f' % lr, file=sys.stderr)

            # load model
            params = torch.load(os.path.join(args['--work_dir'], 'model.bin'),
                                map_location=lambda storage, loc: storage)
            model.load_state_dict(params['state_dict'])
            if args['--cuda']: model = model.cuda()

            # load optimizers
            if args['--reset_optimizer']:
                print('reset optimizer', file=sys.stderr)
                optimizer = torch.optim.Adam(model.inference_model.parameters(), lr=lr)
            else:
                print('restore parameters of the optimizers', file=sys.stderr)
                optimizer.load_state_dict(torch.load(os.path.join(args['--work_dir'], 'optim.bin')))

            # set new lr
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # reset patience
            patience = 0

        if epoch == int(args['--max_epoch']):
            print('reached maximum number of epochs!', file=sys.stderr)
            exit(0)


def train_simple(args):
    mode = args['--mode']
    from diff_representation.model.autoencoder import SyntaxTreeEmbedder, GraphCodeEncoder
    vocab = torch.load(open(args['--vocab'], 'rb'))
    print('vocab: %s' % repr(vocab), file=sys.stderr)

    if not os.path.exists(args['--work_dir']):
        os.makedirs(args['--work_dir'])

    # save arguments to work dir
    json.dump(args, open(os.path.join(args['--work_dir'], 'args.json'), 'w'), indent=2)

    auto_encoder_args = dict(token_embed_size=int(args['--token_embed_size']),
                             token_encoding_size=int(args['--token_encoding_size']),
                             change_vector_size=int(args['--change_vector_size']),
                             change_tag_embed_size=int(args['--change_tag_embed_size']),
                             decoder_hidden_size=int(args['--decoder_hidden_size']),
                             decoder_dropout=float(args['--decoder_dropout']),
                             vocab=vocab,
                             no_change_vector=args['--no_change_vector'],
                             no_unchanged_token_encoding_in_diff_seq=args['--no_unchanged_token_encoding_in_diff_seq'])

    device = torch.device("cuda:0" if args['--cuda'] else "cpu")
    print('use device: %s' % device)

    print('Loading grammar model...', file=sys.stderr)
    if args['--grammar_path']:
        grammar_text = open(args['--grammar_path']).read()
    else:
        grammar_text = request.urlopen('https://raw.githubusercontent.com/dotnet/roslyn/master/src/Compilers'
                                        '/CSharp/Portable/Syntax/Syntax.xml').read()

    grammar = ASDLGrammar.from_roslyn_xml(grammar_text, pruning=args['--prune_grammar'])
    transition_system = TransitionSystem(grammar)

    print('Loading datasets...', file=sys.stderr)
    train_set = original_train_set = DataSet.load_from_jsonl(args['--train_file'],
                                                                type='tree', transition_system=transition_system,
                                                                max_workers=int(args['--num_data_load_worker']))
    dev_set = original_dev_set = DataSet.load_from_jsonl(args['--dev_file'],
                                                            type='tree', transition_system=transition_system,
                                                            max_workers=int(args['--num_data_load_worker']))

    print('Completed!', file=sys.stderr)

    auto_encoder_args['action_embed_size'] = int(args['--action_embed_size'])
    auto_encoder_args['field_embed_size'] = int(args['--field_embed_size'])
    auto_encoder_args['grammar'] = grammar

    syntax_tree_embedder = SyntaxTreeEmbedder(auto_encoder_args['token_embed_size'], vocab, grammar, device=device)
    ast_encoder = GraphCodeEncoder(syntax_tree_embedder=syntax_tree_embedder,
                                   vocab=vocab, grammar=grammar, device=device)

    ast_encoder = ast_encoder.to(device)
    ast_encoder.train()

    optimizer = torch.optim.Adam(ast_encoder.parameters(), lr=0.001)

    epoch = train_iter = num_trial = 0
    # accumulate statistics on the device
    report_loss = 0.
    report_examples = 0
    patience = 0
    max_change_sequence_length = int(args['--max_change_sequence_length'])
    max_action_sequence_length = int(args['--max_action_sequence_length'])
    history_dev_scores = []

    # remove weired examples from datasets
    train_examples = [e for e in train_set.examples if len(e.change_seq) <= max_change_sequence_length
                      and len(e.context) <= max_change_sequence_length and len(e.tgt_actions) <= max_action_sequence_length]
    train_set.examples = train_examples

    dev_examples = [e for e in dev_set.examples if len(e.change_seq) <= max_change_sequence_length
                      and len(e.context) <= max_change_sequence_length and len(e.tgt_actions) <= max_action_sequence_length]
    dev_set.examples = dev_examples

    print('load train file at [%s] (size=%d), dev file at [%s] (size=%d)' % (
                                                                        args['--train_file'], len(original_train_set),
                                                                        args['--dev_file'], len(original_dev_set)), file=sys.stderr)

    while True:
        epoch += 1
        epoch_begin = time.time()

        for batch_examples in train_set.batch_iter(batch_size=int(args['--batch_size']), shuffle=True):
            print(f'batch size: {len(batch_examples)}, total change_seq_len: {sum(len(e.change_seq) for e in batch_examples)}, total context_len: {sum(len(e.context) for e in batch_examples)}, total decode_time_steps: {sum(len(e.tgt_actions) for e in batch_examples)}, max change_seq_len: {max(len(e.change_seq) for e in batch_examples)}, max context len: {max(len(e.context) for e in batch_examples)}, max decode_time_steps: {max(len(e.tgt_actions) for e in batch_examples)}', file=sys.stderr)

            train_iter += 1
            optimizer.zero_grad()

            # [batch_size, max_node_num, embedding_size]
            batch_node_encoding, batch_node_encoding_mask = ast_encoder([e.prev_code_ast for e in batch_examples])
            loss = batch_node_encoding.sum(dim=1).mean(dim=-1).sum()

            total_loss_val = loss.item()
            report_loss = report_loss + total_loss_val
            report_examples += len(batch_examples)

            loss.backward()

            # clip gradient
            grad_norm = torch.nn.utils.clip_grad_norm_(ast_encoder.parameters(), float(args['--clip_grad']))

            optimizer.step()
            del loss

            if train_iter % int(args['--log_every']) == 0:
                print('[Iter %d] encoder loss=%.5f' %
                      (train_iter,
                       report_loss / report_examples),
                      file=sys.stderr)

                report_loss = 0.
                report_examples = 0.

        print('[Epoch %d] epoch elapsed %ds' % (epoch, time.time() - epoch_begin), file=sys.stderr)

        print(subprocess.check_output(['nvidia-smi']), file=sys.stderr)
        print('Manual gc...', file=sys.stderr)
        gc.collect()
        print(subprocess.check_output(['nvidia-smi']), file=sys.stderr)


def test_ppl(args):
    model_path = args['MODEL_PATH']
    test_set_path = args['TEST_SET_PATH']
    mode = args['--mode']

    if mode.startswith('tree2tree'):
        model_cls = TreeBasedAutoEncoderWithGraphEncoder
    else:
        model_cls = TreeBasedAutoEncoder

    print(f'loading model from [{model_path}]', file=sys.stderr)
    model = model_cls.load(model_path)

    transition_system = TransitionSystem(model.grammar)

    # load dataset
    print(f'loading dataset from [{test_set_path}]', file=sys.stderr)
    test_set = DataSet.load_from_jsonl(test_set_path,
                                       type=mode, transition_system=transition_system,
                                       max_workers=int(args['--num_data_load_worker']),
                                       parallel=False if args['--debug'] else True,
                                       debug=args['--debug'])

    if args['--filter_dataset']:
        retained_examples = [e for e in test_set.examples if len(e.change_seq) <= 70
                             and len(e.context) <= 70 and len(e.tgt_actions) <= 100]
        test_set = DataSet(retained_examples)

    if mode == 'tree2tree_subtree_copy' and int(args['--sample_size']) > 1:
        from scipy.special import logsumexp
        # sample generation path
        cum_nll = 0.
        cum_ppl = 0.
        for example in test_set.examples:
            example_decoding_action_paths = transition_system.get_all_decoding_action_paths(target_ast=example.updated_code_ast,
                                                                                            prev_ast=example.prev_code_ast,
                                                                                            sample_size=int(args['--sample_size']))
            print(f'working on {example.id}, {len(example_decoding_action_paths)} paths')
            sampled_action_paths_examples = []
            for sample_id, action_path in enumerate(example_decoding_action_paths):
                sampled_example = ChangeExample(id=example.id + f'-sample{sample_id}',
                                                previous_code_chunk=example.previous_code_chunk,
                                                updated_code_chunk=example.updated_code_chunk,
                                                context=example.context,
                                                prev_code_ast=example.prev_code_ast,
                                                updated_code_ast=example.updated_code_ast,
                                                tgt_actions=action_path)
                sampled_action_paths_examples.append(sampled_example)

            example_avg_nll, example_avg_ppl, example_path_nlls = evaluate_nll(model, DataSet(sampled_action_paths_examples),
                                                                               return_nll_list=True,
                                                                               batch_size=int(args['--batch_size']))

            example_nll = -logsumexp(-np.asarray(list(example_path_nlls.values())))
            cum_nll += example_nll
            cum_ppl += example_nll / len(example.updated_code_chunk)

        avg_nll = cum_nll / len(test_set)
        avg_ppl = np.exp(cum_ppl / len(test_set))
    else:
        avg_nll, avg_ppl, nlls = evaluate_nll(model, test_set, batch_size=int(args['--batch_size']), return_nll_list=True)
        pickle.dump(nlls, open(model_path + '.nll.pkl', 'wb'))

    print(f'average negative log likelihood=%.5f, average ppl=%.5f' % (avg_nll, avg_ppl), file=sys.stderr)


def decode_updated_code(args):
    sys.setrecursionlimit(7000)
    model_path = args['MODEL_PATH']
    test_set_path = args['TEST_SET_PATH']
    mode = args['--mode']
    beam_size = int(args['--beam_size'])

    if mode.startswith('tree2tree') or mode.startswith('tree2seq'):
        model_cls = TreeBasedAutoEncoderWithGraphEncoder
    elif mode.startswith('seq2tree'):
        model_cls = TreeBasedAutoEncoder
    else:
        model_cls = SequentialAutoEncoder

    print(f'loading model from [{model_path}]', file=sys.stderr)
    model = model_cls.load(model_path, use_cuda=args['--cuda'])
    model.eval()

    if '2tree' in mode or 'tree2' in mode:
        transition_system = TransitionSystem(model.grammar)
    else:
        transition_system = None

    def _is_correct(_hyp, _example):
        if '2seq' in mode:
            return _hyp.code == _example.updated_code_chunk
        else:
            return _hyp.tree == _example.updated_code_ast.root_node

    is_graph_change_encoder = isinstance(model.code_change_encoder, GraphChangeEncoder)

    # load dataset
    print(f'loading dataset from [{test_set_path}]', file=sys.stderr)
    test_set = DataSet.load_from_jsonl(test_set_path,
                                       type='sequential' if mode.startswith('seq2seq') else mode,
                                       transition_system=transition_system,
                                       max_workers=1,
                                       parallel=False,
                                       annotate_tree_change=is_graph_change_encoder,
                                       tensorization=True,
                                       vocab=model.vocab,
                                       no_copy=model.args['no_copy'])

    hits = []
    oracle_hits = []
    decode_results = []
    with torch.no_grad():
        # decode change vectors
        change_vecs = model.code_change_encoder.encode_code_changes(test_set.examples,
                                                                    code_encoder=model.sequential_code_encoder,
                                                                    batch_size=256)
        print(f'decoded {change_vecs.shape[0]} entries')

        for e_idx, example in enumerate(tqdm(test_set.examples, file=sys.stdout)):
            change_vec = change_vecs[e_idx]
            if isinstance(model, (TreeBasedAutoEncoder, TreeBasedAutoEncoderWithGraphEncoder)):
                hypotheses = model.decode_updated_code(example,
                                                       change_vec=change_vec,
                                                       transition_system=transition_system,
                                                       beam_size=beam_size,
                                                       debug=args['--debug'])
            else:
                hypotheses = model.decode_updated_code(example,
                                                       change_vec=change_vec,
                                                       beam_size=beam_size,
                                                       debug=args['--debug'])

            if hypotheses:
                oracle_hit = any(_is_correct(hyp, example) for hyp in hypotheses)
                hit = _is_correct(hypotheses[0], example)
            else:
                oracle_hit = hit = False

            hits.append(float(hit))
            oracle_hits.append(float(oracle_hit))

            # if hit:
            #     # print(example.id)
            #     # print('Prev:')
            #     # print(example.untokenized_previous_code_chunk)
            #     # print('Updated:')
            #     # print(example.untokenized_updated_code_chunk)
            #     if '2tree' in mode and args['--debug']:
            #         log_prob, debug_log = model([example], debug=True)
            #         top_hyp_score = hypotheses[0].score
            #         top_hyp_log = hypotheses[0].action_log
            #         if np.abs(top_hyp_score.item() - log_prob.item()) > 0.0001:
            #             print(f'Warning: hyp score is different: {example.id}, hyp: {top_hyp_score.item()}, train: {log_prob[0].item()}', file=sys.stderr)
            #     elif 'tree2seq' in mode and args['--debug']:
            #         log_prob, debug_log = model([example], debug=True)
            #         top_hyp_score = hypotheses[0].score
            #         if np.abs(top_hyp_score - log_prob.item()) > 0.0001:
            #             print(
            #                 f'Warning: hyp score is different: {example.id}, hyp: {top_hyp_score}, train: {log_prob[0].item()}', file=sys.stderr)

            # f_log.write(f'*' * 20 +
            #             f'\nSource:\n{example.untokenized_previous_code_chunk}\n' +
            #             f'Target:\n{example.untokenized_updated_code_chunk}\n\n')

            hypotheses_logs = []
            for hyp in hypotheses:
                entry = {
                    'code': hyp.code if '2seq' in mode else [token.value for token in hyp.tree.descendant_tokens],
                    'score': float(hyp.score),
                    'is_correct': _is_correct(hyp, example)}
                if args['--debug']:
                    entry['action_log'] = hyp.action_log
                hypotheses_logs.append(entry)

            del hypotheses
            decode_results.append((example, hypotheses_logs))

    print('', file=sys.stderr)
    print(f'acc@{beam_size}={sum(hits)}/{len(hits)}={sum(hits) / len(hits)}', file=sys.stderr)
    print(f'oracle acc@{beam_size}={sum(oracle_hits)}/{len(oracle_hits)}={sum(oracle_hits) / len(oracle_hits)}',
          file=sys.stderr)

    if args['--evaluate_ppl']:
        avg_nll, avg_ppl = evaluate_nll(model, test_set, batch_size=128)
        print(f'average nll={avg_nll}, average ppl={avg_ppl}', file=sys.stderr)

    if args['--evaluate_ppl']:
        avg_nll, avg_ppl = evaluate_nll(model, test_set, batch_size=128)
        print(f'average nll={avg_nll}, average ppl={avg_ppl}', file=sys.stderr)

    save_decode_path = model_path + '.decode.bin'
    pickle.dump(decode_results, open(save_decode_path, 'bw'))
    print(f'saved decoding results to {save_decode_path}', file=sys.stderr)


def decode_updated_code_for_neighbors(args):
    model_path = args['MODEL_PATH']
    test_set_path = args['TEST_SET_PATH']
    mode = args['--mode']
    beam_size = int(args['--beam_size'])

    if mode.startswith('tree2tree'):
        model_cls = TreeBasedAutoEncoderWithGraphEncoder
    elif mode.startswith('seq2tree'):
        model_cls = TreeBasedAutoEncoder
    else:
        model_cls = SequentialAutoEncoder

    print(f'loading model from [{model_path}]', file=sys.stderr)
    model = model_cls.load(model_path, use_cuda=args['--cuda'])
    model.eval()

    if '2tree' in mode:
        transition_system = TransitionSystem(model.grammar)
    else:
        transition_system = None

    def _is_correct(_hyp, _example):
        if '2seq' in mode:
            return _hyp.code == _example.updated_code_chunk
        else:
            return _hyp.tree == _example.updated_code_ast.root_node

    def _decode_and_compute_acc(_example, _change_vec):
        if isinstance(model, (TreeBasedAutoEncoder, TreeBasedAutoEncoderWithGraphEncoder)):
            hypotheses = model.decode_updated_code(_example,
                                                   transition_system=transition_system,
                                                   change_vec=_change_vec,
                                                   beam_size=beam_size)
        else:
            hypotheses = model.decode_updated_code(_example,
                                                   change_vec=_change_vec,
                                                   beam_size=beam_size)

        if hypotheses:
            recall_hit = any(_is_correct(hyp, _example) for hyp in hypotheses)
            hit = _is_correct(hypotheses[0], _example)
        else:
            recall_hit = hit = False

        del hypotheses

        return hit, recall_hit

    # load dataset
    print(f'loading dataset from [{test_set_path}]', file=sys.stderr)
    test_set = DataSet.load_from_jsonl(test_set_path,
                                       type='sequential' if mode.startswith('seq2seq') else mode,
                                       transition_system=transition_system,
                                       max_workers=int(args['--num_data_load_worker']),
                                       parallel=int(args['--num_data_load_worker']) > 1,
                                       debug=args['--debug'],
                                       annotate_tree_change=True if 'tree2' in mode else False,
                                       tensorization=True,
                                       vocab=model.vocab,
                                       no_copy=model.args['no_copy'])

    seed_query_file = args['SEED_QUERIES']
    print(f'loading seed queries from [{seed_query_file}]', file=sys.stderr)
    seed_query_ids = list([line.strip() for line in open(seed_query_file)])

    # decode change vectors
    change_vecs = model.code_change_encoder.encode_code_changes(test_set.examples, code_encoder=model.sequential_code_encoder, batch_size=256)
    print(f'decoded {change_vecs.shape[0]} entries')

    # index change vectors
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=int(args['--K']), metric='cosine')
    nn.fit(change_vecs)

    # for each seed query, decode the updated code for its neighbors
    decode_results = OrderedDict()
    with torch.no_grad():
        for seed_query_id in tqdm(seed_query_ids, file=sys.stdout):
            seed_query_idx = test_set.example_id_to_index[seed_query_id]
            seed_query_vec = change_vecs[seed_query_idx]
            dist_array, indices = nn.kneighbors(seed_query_vec.reshape(1, -1), n_neighbors=min(int(args['--K']) + 1, len(test_set)))

            # assert seed_query_idx in indices
            indices = indices[0].tolist()
            dist_array = dist_array[0].tolist()

            seed_query_pos = [i for i, _idx in enumerate(indices) if _idx == seed_query_idx]
            if seed_query_pos:
                seed_query_pos = seed_query_pos[0]
                del indices[seed_query_pos]
                del dist_array[seed_query_pos]
            else:
                indices = indices[:-1]
                dist_array = dist_array[:-1]

            if len(indices) != int(args['--K']):
                print(f'Number of retrieved neighbors smalled than K for example [{seed_query_id}]', file=sys.stderr)

            neighbor_decoding_results = []
            for dist, nbr_idx in zip(dist_array, indices):
                if nbr_idx == seed_query_idx:
                    continue

                neighbor_example = test_set.examples[nbr_idx]
                nbr_change_vec = change_vecs[nbr_idx]

                hit, recall = _decode_and_compute_acc(neighbor_example, seed_query_vec)
                upper_bound_hit, upper_bound_recall = _decode_and_compute_acc(neighbor_example, nbr_change_vec)

                neighbor_decoding_results.append(dict(id=neighbor_example.id,
                                                      hit=hit,
                                                      recall=recall,
                                                      upper_bound_hit=upper_bound_hit,
                                                      upper_bound_recall=upper_bound_recall,
                                                      dist=dist))

            decode_results[seed_query_id] = neighbor_decoding_results

    # aggregate results
    avg_decode_acc = np.average([np.average([nbr['hit'] for nbr in neighbors]) for seed_query_id, neighbors in decode_results.items()])
    avg_decode_recall = np.average([np.average([nbr['recall'] for nbr in neighbors]) for seed_query_id, neighbors in decode_results.items()])

    avg_decode_upper_bound_acc = np.average(
        [np.average([nbr['upper_bound_hit'] for nbr in neighbors]) for seed_query_id, neighbors in decode_results.items()])
    avg_decode_upper_bound_recall = np.average(
        [np.average([nbr['upper_bound_recall'] for nbr in neighbors]) for seed_query_id, neighbors in
         decode_results.items()])

    print('', file=sys.stderr)
    print(f'acc@{beam_size}={avg_decode_acc}', file=sys.stderr)
    print(f'recall@{beam_size}={avg_decode_recall}', file=sys.stderr)

    print('Upper bound performance', file=sys.stderr)
    print(f'upper-bound acc@{beam_size}={avg_decode_upper_bound_acc}', file=sys.stderr)
    print(f'upper-bound recall@{beam_size}={avg_decode_upper_bound_recall}', file=sys.stderr)

    save_decode_path = model_path + '.decode.bin'
    pickle.dump(decode_results, open(save_decode_path, 'bw'))
    print(f'saved decoding results to {save_decode_path}', file=sys.stderr)


if __name__ == '__main__':
    args = docopt(__doc__)

    # seed the RNG
    seed = int(args['--seed'])
    torch.manual_seed(seed)
    if args['--cuda']:
        torch.cuda.manual_seed(seed)
    np.random.seed(seed * 13 // 7)

    if args['train']:
        train(args)
    elif args['train_simple']:
        train_simple(args)
    elif args['test_ppl']:
        test_ppl(args)
    elif args['decode_updated_code']:
        decode_updated_code(args)
    elif args['decode_updated_code_for_neighbors']:
        decode_updated_code_for_neighbors(args)
    else:
        raise RuntimeError(f'invalid run mode')
