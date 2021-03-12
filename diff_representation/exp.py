#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Learning to represent edits

Usage:
    exp.py train [options] CONFIG_FILE
    exp.py test_ppl [options] MODEL_PATH TEST_SET_PATH
    exp.py decode_updated_data [options] MODEL_PATH TEST_SET_PATH

Options:
    -h --help                                   Show this screen
    --cuda                                      Use GPU
    --debug                                     Debug mode
    --seed=<int>                                Seed [default: 0]
    --work_dir=<dir>                            work dir [default: exp_runs/]
    --sample_size=<int>                         Sample size [default: 1]
    --beam_size=<int>                           Beam size [default: 5]
    --evaluate_ppl                              Evaluate perplexity as well
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

import torch
import torch.nn as nn

from diff_representation.asdl.grammar import ASDLGrammar
from diff_representation.asdl.transition_system.transition import TransitionSystem
from diff_representation.common.config import Arguments
from diff_representation.dataset import DataSet
from diff_representation.change_entry import ChangeExample
from diff_representation.model.encdec import *
from diff_representation.vocab import Vocab, VocabEntry
from diff_representation.evaluate import evaluate_nll
from diff_representation.utils.decode import *
from diff_representation.utils.utils import get_entry_str
from diff_representation.utils.relevance import get_nn


def train(cmd_args):
    args = Arguments.from_file(cmd_args['CONFIG_FILE'], cmd_args=cmd_args)

    work_dir = cmd_args['--work_dir']
    use_cuda = cmd_args['--cuda']
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)

    # save arguments to work dir
    f = open(os.path.join(work_dir, 'config.json'), 'w')
    f.write(args.to_string())
    f.close()

    device = torch.device("cuda:0" if use_cuda else "cpu")
    print('use device: %s' % device, file=sys.stderr)

    model = NeuralEditor.build(args)
    print('Loading datasets...', file=sys.stderr)

    train_set = DataSet.load_from_jsonl(args['dataset']['train_file'],
                                        editor=model,
                                        max_workers=args['dataset']['num_data_load_worker'],
                                        tensorization=args['dataset']['tensorization'])

    dev_set = DataSet.load_from_jsonl(args['dataset']['dev_file'],
                                      editor=model,
                                      max_workers=args['dataset']['num_data_load_worker'],
                                      tensorization=args['dataset']['tensorization'])

    print('loaded train file at [%s] (size=%d), dev file at [%s] (size=%d)' % (
        args['dataset']['train_file'], len(train_set),
        args['dataset']['dev_file'], len(dev_set)), file=sys.stderr)

    model = model.to(device)
    model.train()

    batch_size = args['trainer']['batch_size']
    epoch = train_iter = num_trial = 0
    # accumulate statistics on the device
    report_loss = 0.
    report_word_predict_loss = 0.
    report_examples = 0
    patience = 0
    word_predict_loss_weight = args['loss']['word_predict_loss_weight']
    history_dev_scores = []

    parameters = list(model.parameters())

    if word_predict_loss_weight > 0.:
        word_predict_multi_task = ChangedWordPredictionMultiTask(model.edit_encoder.change_vector_size, model.vocab, model.device)
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

                results = model(batch_examples, return_change_vectors=True)
                log_probs, change_vecs = results['log_probs'], results['edit_encoding']
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
                grad_norm = torch.nn.utils.clip_grad_norm_(parameters, args['trainer']['clip_grad'])

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

            if train_iter % args['trainer']['log_every'] == 0:
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

        if args['trainer']['save_each_epoch']:
            save_path = os.path.join(work_dir, f'model.epoch{epoch}.bin')
            print('[Epoch %d] save model to [%s]' % (epoch, save_path), file=sys.stderr)
            model.save(save_path)

        if is_better:
            patience = 0
            save_path = os.path.join(work_dir, 'model.bin')
            print('save currently the best model to [%s]' % save_path, file=sys.stderr)
            model.save(save_path)

            # also save the optimizers' state
            torch.save(optimizer.state_dict(), os.path.join(work_dir, 'optim.bin'))
        elif patience < args['trainer']['patience']:
            patience += 1
            print('hit patience %d' % patience, file=sys.stderr)

        if patience == args['trainer']['patience']:
            num_trial += 1
            print('hit #%d trial' % num_trial, file=sys.stderr)
            if num_trial == args['trainer']['max_num_trial']:
                print('early stop!', file=sys.stderr)
                exit(0)

            # decay lr, and restore from previously best checkpoint
            lr = optimizer.param_groups[0]['lr'] * args['trainer']['lr_decay']
            print('load previously best model and decay learning rate to %f' % lr, file=sys.stderr)

            # load model
            params = torch.load(os.path.join(work_dir, 'model.bin'),
                                map_location=lambda storage, loc: storage)
            model.load_state_dict(params['state_dict'])
            if use_cuda: model = model.cuda()

            # load optimizers
            if args['trainer']['reset_optimizer']:
                print('reset optimizer', file=sys.stderr)
                optimizer = torch.optim.Adam(model.inference_model.parameters(), lr=lr)
            else:
                print('restore parameters of the optimizers', file=sys.stderr)
                optimizer.load_state_dict(torch.load(os.path.join(work_dir, 'optim.bin')))

            # set new lr
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # reset patience
            patience = 0

        if epoch == args['trainer']['max_epoch']:
            print('reached maximum number of epochs!', file=sys.stderr)
            exit(0)


def test_ppl(args):
    sys.setrecursionlimit(7000)
    model_path = args['MODEL_PATH']
    test_set_path = args['TEST_SET_PATH']

    print(f'loading model from [{model_path}]', file=sys.stderr)
    model = NeuralEditor.load(model_path, use_cuda=args['--cuda'])
    model.eval()

    # load dataset
    print(f'loading dataset from [{test_set_path}]', file=sys.stderr)
    test_set = DataSet.load_from_jsonl(test_set_path, editor=model)

    avg_nll, avg_ppl = evaluate_nll(model, test_set, batch_size=128)
    print(f'average nll={avg_nll}, average ppl={avg_ppl}', file=sys.stderr)


def decode_updated_code(args):
    sys.setrecursionlimit(7000)
    model_path = args['MODEL_PATH']
    test_set_path = args['TEST_SET_PATH']
    beam_size = int(args['--beam_size'])

    print(f'loading model from [{model_path}]', file=sys.stderr)
    model = NeuralEditor.load(model_path, use_cuda=args['--cuda'])
    model.eval()

    def _is_correct(_hyp, _example):
        if isinstance(model, Seq2SeqEditor):
            return _hyp.code == _example.updated_data
        elif isinstance(model, Graph2TreeEditor):
            return _hyp.tree == _example.updated_code_ast.root_node
        else:
            raise RuntimeError()

    # load dataset
    print(f'loading dataset from [{test_set_path}]', file=sys.stderr)
    test_set = DataSet.load_from_jsonl(test_set_path, editor=model)

    hits = []
    oracle_hits = []
    decode_results = []
    with torch.no_grad():
        # decode change vectors
        change_vecs = model.get_edit_encoding_by_batch(test_set.examples, batch_size=256)
        print(f'decoded {change_vecs.shape[0]} entries')

        for e_idx, example in enumerate(tqdm(test_set.examples, file=sys.stdout, total=len(test_set))):
            change_vec = change_vecs[e_idx]
            hypotheses = model.decode_updated_data(example, edit_encoding=change_vec, beam_size=beam_size, debug=args['--debug'])

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
            #     # print(example.raw_prev_data)
            #     # print('Updated:')
            #     # print(example.raw_updated_data)
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
            #             f'\nSource:\n{example.raw_prev_data}\n' +
            #             f'Target:\n{example.raw_updated_data}\n\n')

            hypotheses_logs = []
            for hyp in hypotheses:
                entry = {
                    'code': hyp.code if isinstance(model, Seq2SeqEditor) else [token.value for token in hyp.tree.descendant_tokens],
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

    save_decode_path = model_path + '.decode.bin'
    pickle.dump(decode_results, open(save_decode_path, 'bw'))
    print(f'saved decoding results to {save_decode_path}', file=sys.stderr)


if __name__ == '__main__':
    cmd_args = docopt(__doc__)

    # seed the RNG
    seed = int(cmd_args['--seed'])
    print(f'use random seed {seed}', file=sys.stderr)
    torch.manual_seed(seed)

    use_cuda = cmd_args['--cuda']
    if use_cuda:
        torch.cuda.manual_seed(seed)
    np.random.seed(seed * 13 // 7)

    if cmd_args['train']:
        train(cmd_args)
    elif cmd_args['test_ppl']:
        test_ppl(cmd_args)
    elif cmd_args['decode_updated_data']:
        decode_updated_code(cmd_args)
    else:
        raise RuntimeError(f'invalid run mode')
