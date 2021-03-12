# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from difflib import SequenceMatcher
from typing import List, Dict, Set, Sequence, Tuple, Union
import re
from collections import namedtuple

from docopt import docopt

from utils.dataloading import load_json_gz


DIFF_OP_RE = re.compile(r'(equal,)?replace(,equal)?')


def detect_contiguous_change(prev_code_chunk:List[str], updated_code_chunk:List[str], num_contiguous_line:int=1):
    matcher = SequenceMatcher(a=prev_code_chunk, b=updated_code_chunk)
    trans_ops = matcher.get_opcodes()
    trans_op_str = ','.join(x[0] for x in trans_ops)
    m = DIFF_OP_RE.match(trans_op_str)

    if m:
        tag, i1, i2, j1, j2 = [x for x in trans_ops if x[0] == 'replace'][0]
        if i2 - i1 <= num_contiguous_line and j1 - j2 <= num_contiguous_line:
            return i1, i2, j1, j2

    return None


def is_valid_change(change):
    return change['prev_code_chunk'].strip() != change['updated_data'].strip()
