# coding=utf-8
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import OrderedDict
from typing import List, Tuple, Dict, Union

try:
    from cStringIO import StringIO
except:
    from io import StringIO
import json

from .grammar import *


class AbstractSyntaxNode(object):
    def __init__(self, production, realized_fields=None, id=-1):
        # this field is a unique identification of the node, but it's not used
        # when comparing two ASTs.
        self.id = id
        self.production = production

        # a child is essentially a *realized_field*
        self.fields = []

        # record its parent field to which it's attached
        self.parent_field = None

        # used in decoding, record the time step when this node was created
        self.created_time = 0

        if realized_fields:
            assert len(realized_fields) == len(self.production.fields)

            for field in realized_fields:
                self.add_child(field)
        else:
            for field in self.production.fields:
                self.add_child(RealizedField(field))

    def add_child(self, realized_field):
        # if isinstance(realized_field.value, AbstractSyntaxTree):
        #     realized_field.value.parent = self
        self.fields.append(realized_field)
        realized_field.parent_node = self

    def __getitem__(self, field_name):
        for field in self.fields:
            if field.name == field_name: return field
        raise KeyError

    @property
    def is_pre_terminal(self):
        return all(not f.type.is_composite for f in self.fields)

    def sanity_check(self):
        if len(self.production.fields) != len(self.fields):
            raise ValueError('filed number must match')
        for field, realized_field in zip(self.production.fields, self.fields):
            assert field == realized_field.field
        for child in self.fields:
            for child_val in child.as_value_list:
                if isinstance(child_val, AbstractSyntaxNode):
                    child_val.sanity_check()

    def copy(self):
        new_tree = AbstractSyntaxNode(self.production, id=self.id)
        new_tree.created_time = self.created_time
        for i, old_field in enumerate(self.fields):
            new_field = new_tree.fields[i]
            new_field._not_single_cardinality_finished = old_field._not_single_cardinality_finished
            if old_field.type.is_composite:
                for value in old_field.as_value_list:
                    new_field.add_value(value.copy())
            else:
                for value in old_field.as_value_list:
                    new_field.add_value(value)

        return new_tree

    def to_string(self, sb=None):
        is_root = False
        if sb is None:
            is_root = True
            sb = StringIO()

        sb.write('(')
        sb.write(self.production.constructor.name)

        for field in self.fields:
            sb.write(' ')
            sb.write('(')
            sb.write(field.type.name)
            sb.write(Field.get_cardinality_repr(field.cardinality))
            sb.write('-')
            sb.write(field.name)

            if field.value is not None:
                for val_node in field.as_value_list:
                    sb.write(' ')
                    if field.type.is_composite:
                        val_node.to_string(sb)
                    else:
                        sb.write(str(val_node).replace(' ', '-SPACE-'))

            sb.write(')')  # of field

        sb.write(')')  # of node

        if is_root:
            return sb.getvalue()

    def __hash__(self):
        code = hash(self.production)
        for field in self.fields:
            code = code + 37 * hash(field)

        return code

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False

        # if self.created_time != other.created_time:
        #     return False

        if self.production != other.production:
            return False

        if len(self.fields) != len(other.fields):
            return False

        for i in range(len(self.fields)):
            if self.fields[i] != other.fields[i]: return False

        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return repr(self.production)

    @property
    def descendant_nodes(self):
        def _visit(node):
            if isinstance(node, AbstractSyntaxNode):
                yield node

                for field in node.fields:
                    for field_val in field.as_value_list:
                        yield from _visit(field_val)

        yield from _visit(self)

    @property
    def descendant_nodes_and_tokens(self):
        def _visit(node):
            if isinstance(node, AbstractSyntaxNode):
                yield node

                for field in node.fields:
                    for field_val in field.as_value_list:
                        yield from _visit(field_val)
            else:
                yield node

        yield from _visit(self)

    @property
    def descendant_tokens(self):
        def _visit(node):
            if isinstance(node, AbstractSyntaxNode):
                for field in node.fields:
                    for field_val in field.as_value_list:
                        yield from _visit(field_val)
            else:
                yield node

        yield from _visit(self)

    @property
    def size(self):
        node_num = 1
        for field in self.fields:
            for val in field.as_value_list:
                if isinstance(val, AbstractSyntaxNode):
                    node_num += val.size
                else: node_num += 1

        return node_num

    @property
    def depth(self):
        return 1 + max(max(val.depth) for val in field.as_value_list for field in self.fields)


class SyntaxToken(object):
    """represent a terminal token on an AST"""
    def __init__(self, type, value, position=-1, id=-1):
        self.id = id
        self.type = type
        self.value = value
        self.position = position

        # record its parent field to which it's attached
        self.parent_field = None

    @property
    def size(self):
        return 1

    @property
    def depth(self):
        return 0

    def copy(self):
        return SyntaxToken(self.type, self.value, position=self.position, id=self.id)

    def __hash__(self):
        code = hash(self.type) + 37 * hash(self.value)

        return code

    def __repr__(self):
        return repr(self.value)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False

        return self.type == other.type and self.value == other.value

    def __ne__(self, other):
        return not self.__eq__(other)


class AbstractSyntaxTree(object):
    def __init__(self, root_node: AbstractSyntaxNode):
        self.root_node = root_node

        self.adjacency_list: List[Tuple[int, int]] = None
        self.id2node: Dict[int, Union[AbstractSyntaxNode, SyntaxToken]] = None
        self.syntax_tokens_and_ids: List[Tuple[int, SyntaxToken]] = None

        self._get_properties()

    def _get_properties(self):
        """assign numerical indices to index each node"""
        id2nodes = OrderedDict()
        syntax_token_position2id = OrderedDict()
        terminal_tokens_list = []
        adj_list = []

        def _index_sub_tree(root_node, parent_node):
            if parent_node:
                adj_list.append((parent_node.id, root_node.id))

            id2nodes[root_node.id] = root_node
            if isinstance(root_node, AbstractSyntaxNode):
                for field in root_node.fields:
                    for field_val in field.as_value_list:
                        _index_sub_tree(field_val, root_node)
            else:
                # it's a syntax token
                terminal_tokens_list.append((root_node.id, root_node))
                syntax_token_position2id[root_node.position] = root_node.id

        _index_sub_tree(self.root_node, None)

        self.adjacency_list = adj_list
        self.id2node = id2nodes
        self.syntax_tokens_and_ids = terminal_tokens_list
        self.syntax_token_position2id = syntax_token_position2id
        self.syntax_tokens_set = {token: id for id, token in terminal_tokens_list}
        self.node_num = len(id2nodes)

        # this property are used for training and beam search, to get ids of syntax tokens
        # given their surface values
        syntax_token_value2ids = dict()
        for id, token in self.syntax_tokens_and_ids:
            syntax_token_value2ids.setdefault(token.value, []).append(id)
        self.syntax_token_value2ids = syntax_token_value2ids

        self._init_sibling_adjacency_list()

    def _init_sibling_adjacency_list(self):
        next_siblings = []

        def _travel(node):
            if isinstance(node, AbstractSyntaxNode):
                child_nodes = []
                for field in node.fields:
                    for val in field.as_value_list:
                        child_nodes.append(val)
                for i in range(len(child_nodes) - 1):
                    left_node = child_nodes[i]
                    right_node = child_nodes[i + 1]
                    next_siblings.append((left_node.id, right_node.id))

                for child_node in child_nodes:
                    _travel(child_node)

        _travel(self.root_node)
        setattr(self, 'next_siblings_adjacency_list', next_siblings)

    @property
    def syntax_tokens(self) -> List[SyntaxToken]:
        return [token for id, token in self.syntax_tokens_and_ids]

    @property
    def descendant_nodes(self) -> List[AbstractSyntaxNode]:
        for node_id, node in self.id2node.items():
            if isinstance(node, AbstractSyntaxNode):
                yield node_id, node

    def is_syntax_token(self, token):
        if isinstance(token, int):
            return isinstance(self.id2node[token], SyntaxToken)
        else:
            return token in self.syntax_tokens_set

    def find_node(self, query_node: AbstractSyntaxNode, return_id=True):
        search_results = []
        for node_id, node in self.descendant_nodes:
            if node.production == query_node.production:
                if node == query_node:
                    if return_id:
                        search_results.append((node_id, node))
                    else:
                        search_results.append(node)

        return search_results

    def copy(self):
        ast_copy = AbstractSyntaxTree(root_node=self.root_node.copy())

        return ast_copy


class RealizedField(Field):
    """wrapper of field realized with values"""
    def __init__(self, field, value=None, parent=None):
        super(RealizedField, self).__init__(field.name, field.type, field.cardinality)

        # record its parent AST node
        self.parent_node = None

        # FIXME: hack, return the field as a property
        self.field = field

        # initialize value to correct type
        if self.cardinality == 'multiple':
            self.value = []
            if value is not None:
                for child_node in value:
                    self.add_value(child_node)
        else:
            self.value = None
            # note the value could be 0!
            if value is not None: self.add_value(value)

        # properties only used in decoding, record if the field is finished generating
        # when card in [optional, multiple]
        self._not_single_cardinality_finished = False

    def add_value(self, value):
        value.parent_field = self

        if self.cardinality == 'multiple':
            self.value.append(value)
        else:
            self.value = value

    def remove(self, value):
        """remove a value from the field"""
        if self.cardinality in ('single', 'optional'):
            if self.value == value:
                self.value = None
            else:
                raise ValueError(f'{value} is not a value of the field {self}')
        else:
            tgt_idx = self.value.index(value)
            self.value.pop(tgt_idx)

    def replace(self, value, new_value):
        """replace an old field value with a new one"""
        if self.cardinality == 'multiple':
            tgt_idx = self.value.index(value)

            new_value.parent_field = self
            self.value[tgt_idx] = new_value
        else:
            assert self.value == value

            new_value.parent_field = self
            self.value = new_value

    @property
    def as_value_list(self):
        """get value as an iterable"""
        if self.cardinality == 'multiple': return self.value
        elif self.value is not None: return [self.value]
        else: return []

    @property
    def value_count(self):
        return len(self.as_value_list)

    @property
    def finished(self):
        if self.cardinality == 'single':
            if self.value is None: return False
            else: return True
        elif self.cardinality == 'optional' and self.value is not None:
            return True
        else:
            if self._not_single_cardinality_finished: return True
            else: return False

    def set_finish(self):
        # assert self.cardinality in ('optional', 'multiple')
        self._not_single_cardinality_finished = True

    def __eq__(self, other):
        if super(RealizedField, self).__eq__(other):
            if type(other) == Field: return True  # FIXME: hack, Field and RealizedField can compare!
            if self.value == other.value: return True
            else: return False
        else: return False
