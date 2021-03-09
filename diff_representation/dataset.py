# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import multiprocessing
from functools import partial
import numpy as np
import sys
import json
import concurrent.futures

from torch.multiprocessing import Manager, Process, Queue

from diff_representation.asdl.transition_system.hypothesis import *
from diff_representation.asdl.transition_system.transition import TransitionSystem
from diff_representation.model.sequential_change_encoder import SequentialChangeEncoder
from diff_representation.model.sequential_decoder import SequentialDecoder
from .change_entry import ChangeExample
from .utils.utils import *


def _encode(word_list):
    return [w.replace('\n', '-NEWLINE-') for w in word_list]


def load_one_change_entry(json_str, type, transition_system, debug=False, copy_identifier=True,
                          annotate_tree_change=False, tensorization=True, vocab=None, no_copy=False):
    entry = json.loads(json_str)
    previous_code_chunk = _encode(entry['PrevCodeChunkTokens'])
    updated_code_chunk = _encode(entry['UpdatedCodeChunkTokens'])
    context = _encode(entry['PrecedingContext'] + ['|||'] + entry['SucceedingContext'])

    if type == 'sequential':
        example = ChangeExample(id=entry['Id'],
                                previous_code_chunk=previous_code_chunk,
                                updated_code_chunk=updated_code_chunk,
                                untokenized_previous_code_chunk=entry['PrevCodeChunk'],
                                untokenized_updated_code_chunk=entry['UpdatedCodeChunk'],
                                context=context)

        # preform tensorization
        if tensorization:
            SequentialChangeEncoder.populate_aligned_token_index_and_mask(example)
            SequentialDecoder.populate_gen_and_copy_index_and_mask(example, vocab, no_copy=no_copy)
    elif type == 'tree2seq':
        updated_code_ast_json = entry['UpdatedCodeAST']
        updated_code_ast = transition_system.grammar.get_ast_from_json_obj(updated_code_ast_json)
        prev_code_ast_json = entry['PrevCodeAST']
        prev_code_ast = transition_system.grammar.get_ast_from_json_obj(prev_code_ast_json)

        example = ChangeExample(id=entry['Id'],
                                previous_code_chunk=previous_code_chunk,
                                updated_code_chunk=updated_code_chunk,
                                untokenized_previous_code_chunk=entry['PrevCodeChunk'],
                                untokenized_updated_code_chunk=entry['UpdatedCodeChunk'],
                                context=context,
                                prev_code_ast=prev_code_ast,
                                updated_code_ast=updated_code_ast)
    elif type in ('seq2tree', 'tree2tree', 'tree2tree_subtree_copy'):
        updated_code_ast_json = entry['UpdatedCodeAST']
        updated_code_ast = transition_system.grammar.get_ast_from_json_obj(updated_code_ast_json)

        prev_code_ast_json = entry['PrevCodeAST']
        prev_code_ast = transition_system.grammar.get_ast_from_json_obj(prev_code_ast_json)

        tgt_actions = transition_system.get_decoding_actions(target_ast=updated_code_ast,
                                                             prev_ast=prev_code_ast if type == 'tree2tree_subtree_copy' else None,
                                                             copy_identifier=copy_identifier)

        if debug:
            action_paths = [tgt_actions]
            for tgt_actions in action_paths:
                # sanity check target decoding actions
                hyp = Hypothesis()
                for decode_action in tgt_actions:
                    assert any(isinstance(decode_action, cls) for cls in transition_system.get_valid_continuation_types(hyp))
                    if isinstance(decode_action, ApplyRuleAction):
                        assert decode_action.production in transition_system.get_valid_continuating_productions(hyp)

                    if isinstance(decode_action, ApplySubTreeAction) and hyp.frontier_field:
                        assert decode_action.tree.production.type in transition_system.grammar.descendant_types[hyp.frontier_field.type]

                    if hyp.frontier_node:
                        assert hyp.frontier_field == decode_action.frontier_field
                        assert hyp.frontier_node.production == decode_action.frontier_prod

                    hyp.apply_action(decode_action)
                assert hyp.tree.to_string() == updated_code_ast.root_node.to_string()
                assert hyp.tree == updated_code_ast.root_node
                assert hyp.completed

        example = ChangeExample(id=entry['Id'],
                                previous_code_chunk=previous_code_chunk,
                                updated_code_chunk=updated_code_chunk,
                                untokenized_previous_code_chunk=entry['PrevCodeChunk'],
                                untokenized_updated_code_chunk=entry['UpdatedCodeChunk'],
                                context=context,
                                prev_code_ast=prev_code_ast,
                                updated_code_ast=updated_code_ast,
                                tgt_actions=tgt_actions)

        if annotate_tree_change:
            from diff_representation.model.graph_change_encoder import GraphChangeEncoder
            example.change_edges = GraphChangeEncoder.compute_change_edges(example)

        # preform tensorization
        if tensorization:
            SequentialChangeEncoder.populate_aligned_token_index_and_mask(example)
    else:
        raise ValueError('unknown dataset type')

    return example


def _create_batched_examples(input_queue: Queue,
                             output_queue: Queue,
                             arg_dict: Dict,
                             index: int):
    item = input_queue.get()
    while item is not None:
        batch_examples = []
        batch_id, serialized_batch_examples = item
        for serialized_example in serialized_batch_examples:
            example = load_one_change_entry(serialized_example, **arg_dict)
            batch_examples.append(example)

        # sort by the length of the change sequence in descending order
        batch_examples.sort(key=lambda e: -len(e.change_seq))
        output_queue.put((batch_id, batch_examples))

        item = input_queue.get()

    output_queue.put(index)


class ParallelLazyDataLoader(object):
    def __init__(self, json_file_path, type='sequential', transition_system=None,
                 max_workers=1, debug=False, tensorization=False,
                 annotate_tree_change=False, copy_identifier=True, vocab=None, no_copy=False):

        all_lines = [l.strip() for l in open(json_file_path)]
        self._serialized_examples = all_lines
        print(f'read {len(all_lines)} lines from the dataset [{json_file_path}]', file=sys.stderr)

        self.worker_arg = dict(
            type=type,
            transition_system=transition_system,
            debug=debug,
            tensorization=tensorization,
            copy_identifier=copy_identifier,
            annotate_tree_change=annotate_tree_change,
            vocab=vocab,
            no_copy=no_copy
        )

        self.processes = []
        self.output_queue_size = 500
        self.num_workers = max_workers

    def __len__(self):
        return len(self._serialized_examples)

    def __iter__(self):
        raise NotImplementedError

    def batch_iter(self, batch_size, shuffle=False):
        index_arr = np.arange(len(self._serialized_examples))
        if shuffle:
            np.random.shuffle(index_arr)

        manager = Manager()
        output_queue = manager.Queue(self.output_queue_size)
        input_queue = manager.Queue()

        batch_num = int(np.ceil(len(self._serialized_examples) / float(batch_size)))
        for batch_id in range(batch_num):
            batch_ids = index_arr[batch_size * batch_id: batch_size * (batch_id + 1)]
            serialized_examples = [self._serialized_examples[i] for i in batch_ids]
            input_queue.put((batch_id, serialized_examples))
        for i in range(self.num_workers):
            input_queue.put(None)

        print('finished putting all examples to the input queue', file=sys.stderr)

        for i in range(self.num_workers):
            args = (input_queue, output_queue, self.worker_arg, i)
            process = Process(target=_create_batched_examples, args=args)
            process.daemon = True
            process.start()
            self.processes.append(process)

        num_finished = 0
        send_idx = 0
        reorder_dict = {}
        super_fetch_batch_num = 10

        for i in range(super_fetch_batch_num):
            if num_finished < self.num_workers:
                item = output_queue.get()
                if isinstance(item, int):
                    num_finished += 1
                else:
                    idx, batch_examples = item
                    reorder_dict[idx] = batch_examples
                    # print(f'pre fetch batch {idx}')

        while True:
            if num_finished < self.num_workers:
                item = output_queue.get()
                if isinstance(item, int):
                    num_finished += 1
                else:
                    idx, batch_examples = item
                    reorder_dict[idx] = batch_examples
                    # print(f'read in batch {idx}')
            elif len(reorder_dict) == 0:
                break

            if send_idx in reorder_dict:
                batch_examples = reorder_dict.pop(send_idx)
                # print(f'output batch {send_idx}')
                send_idx += 1
                yield batch_examples

        for process in self.processes:
            process.join()
        self.processes.clear()


class DataSet:
    def __init__(self, examples):
        self.examples = examples
        self.example_id_to_index = OrderedDict([(e.id, idx) for idx, e in enumerate(self.examples)])

    def batch_iter(self, batch_size, shuffle=False):
        index_arr = np.arange(len(self.examples))
        if shuffle:
            np.random.shuffle(index_arr)

        batch_num = int(np.ceil(len(self.examples) / float(batch_size)))
        for batch_id in range(batch_num):
            batch_ids = index_arr[batch_size * batch_id: batch_size * (batch_id + 1)]
            batch_examples = [self.examples[i] for i in batch_ids]
            # sort by the length of the change sequence in descending order
            batch_examples.sort(key=lambda e: -len(e.change_seq))

            yield batch_examples

    def __len__(self):
        return len(self.examples)

    def __iter__(self):
        return iter(self.examples)

    def get_example_by_id(self, eid):
        idx = self.example_id_to_index[eid]
        return self.examples[idx]

    @staticmethod
    def load_from_jsonl(file_path, type='sequential', transition_system: TransitionSystem=None,
                        parallel=True, from_ipython=False, max_workers=None, debug=False, copy_identifier=True,
                        annotate_tree_change=False, tensorization=True, vocab=None, no_copy=False):
        examples = []
        with open(file_path) as f:
            print('reading all lines from the dataset', file=sys.stderr)
            all_lines = [l for l in f]
            print('%d lines. Done' % len(all_lines), file=sys.stderr)

            if from_ipython:
                from tqdm import tqdm_notebook
                iter_log_func = partial(tqdm_notebook, total=len(all_lines), desc='loading dataset')
            else:
                from tqdm import tqdm
                iter_log_func = partial(tqdm, total=len(all_lines), desc='loading dataset', file=sys.stdout)

            if parallel:
                with multiprocessing.Pool(max_workers) as pool:
                    processed_examples = pool.imap(partial(load_one_change_entry, type=type, transition_system=transition_system,
                                                           debug=debug, copy_identifier=copy_identifier,
                                                           annotate_tree_change=annotate_tree_change,
                                                           tensorization=tensorization, vocab=vocab, no_copy=no_copy),
                                                   iterable=all_lines,
                                                   chunksize=5000)
                    for example in iter_log_func(processed_examples):
                        examples.append(example)
            else:
                for line in iter_log_func(all_lines):
                    example = load_one_change_entry(line, type, transition_system, debug=debug,
                                                    copy_identifier=copy_identifier,
                                                    annotate_tree_change=annotate_tree_change,
                                                    tensorization=tensorization,
                                                    vocab=vocab,
                                                    no_copy=no_copy)
                    examples.append(example)

        data_set = DataSet([e for e in examples if e])

        if type == 'tree':
            print('average action length: %.2f' % np.average([len(e.tgt_actions) for e in data_set.examples]), file=sys.stderr)

        return data_set
