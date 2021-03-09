# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import List
import difflib

from diff_representation.diff_utils import TokenLevelDiffer


class ChangeExample:
    def __init__(self, previous_code_chunk: List[str], updated_code_chunk: List[str],
                 context: List[str],
                 untokenized_previous_code_chunk: str=None, untokenized_updated_code_chunk: str=None,
                 id: str='default_id', **kwargs):
        self.id = id

        self.previous_code_chunk = previous_code_chunk
        self.updated_code_chunk = updated_code_chunk

        self.untokenized_previous_code_chunk = untokenized_previous_code_chunk
        self.untokenized_updated_code_chunk = untokenized_updated_code_chunk

        self.context = context

        diff_hunk = '\n'.join(list(x.strip('\n') if x.startswith('@') else x
                                   for x in difflib.unified_diff(a=previous_code_chunk, b=updated_code_chunk,
                                                                 n=len(self.previous_code_chunk) + len(self.updated_code_chunk),
                                                                 lineterm=''))[2:])
        self.diff_hunk = diff_hunk

        self._init_change_seq()

        self.__dict__.update(kwargs)

    def _init_change_seq(self):
        differ = TokenLevelDiffer()
        diff_result = differ.unified_format(dict(diff=self.diff_hunk))
        change_seq = []

        prev_token_ptr = updated_token_ptr = 0
        for i, (added, removed, same) in enumerate(zip(diff_result.added, diff_result.removed, diff_result.same)):
            if same is not None:
                tag = 'SAME'
                token = same

                assert self.previous_code_chunk[prev_token_ptr] == self.updated_code_chunk[updated_token_ptr] == token

                prev_token_ptr += 1
                updated_token_ptr += 1
            elif added is not None and removed is not None:
                tag = 'REPLACE'
                token = (removed, added)

                assert self.previous_code_chunk[prev_token_ptr] == removed
                assert self.updated_code_chunk[updated_token_ptr] == added

                prev_token_ptr += 1
                updated_token_ptr += 1
            elif added is not None and removed is None:
                tag = 'ADD'
                token = added

                assert self.updated_code_chunk[updated_token_ptr] == added

                updated_token_ptr += 1
            elif added is None and removed is not None:
                tag = 'DEL'
                token = removed

                assert self.previous_code_chunk[prev_token_ptr] == removed

                prev_token_ptr += 1
            else:
                raise ValueError('unknown change entry')

            change_seq.append((tag, token))

        setattr(self, 'change_seq', change_seq)
