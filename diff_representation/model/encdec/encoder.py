# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from collections import namedtuple

EncodingResult = namedtuple('EncodingResult', ['data', 'encoding', 'last_state', 'last_cell', 'mask'])
