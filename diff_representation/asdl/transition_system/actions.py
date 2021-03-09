# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

class Action(object):
    pass


class ApplyRuleAction(Action):
    def __init__(self, production):
        self.production = production

    def __hash__(self):
        return hash(self.production)

    def __eq__(self, other):
        return isinstance(other, ApplyRuleAction) and self.production == other.production

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return 'ApplyRule[%s]' % self.production.__repr__()


class GenTokenAction(Action):
    def __init__(self, token):
        self.token = token

    def is_stop_signal(self):
        return self.token == '</primitive>'

    def __repr__(self):
        return 'GenToken[%s]' % self.token


class ReduceAction(Action):
    def __repr__(self):
        return 'Reduce'


class ApplySubTreeAction(Action):
    def __init__(self, tree, tree_node_ids=-1):
        self.tree = tree
        self.tree_node_ids = tree_node_ids

    def __repr__(self):
        return 'ApplySubTree[%s], Node[%d]' % (repr(self.tree), self.tree.id)


class DecodingAction:
    def __init__(self, t, parent_t, frontier_prod, frontier_field, preceding_syntax_token_index):
        self.t = t
        self.parent_t = parent_t
        self.frontier_prod = frontier_prod
        self.frontier_field = frontier_field
        self.preceding_syntax_token_index = preceding_syntax_token_index


class ApplyRuleDecodingAction(ApplyRuleAction, DecodingAction):
    def __init__(self, t, parent_t, frontier_prod, frontier_field, production, preceding_syntax_token_index=None):
        ApplyRuleAction.__init__(self, production)
        DecodingAction.__init__(self, t, parent_t, frontier_prod, frontier_field, preceding_syntax_token_index)


class ApplySubTreeDecodingAction(ApplySubTreeAction, DecodingAction):
    def __init__(self, t, parent_t, frontier_prod, frontier_field, tree, tree_node_ids, preceding_syntax_token_index=None):
        ApplySubTreeAction.__init__(self, tree, tree_node_ids)
        DecodingAction.__init__(self, t, parent_t, frontier_prod, frontier_field, preceding_syntax_token_index)


class ReduceDecodingAction(ReduceAction, DecodingAction):
    def __init__(self, t, parent_t, frontier_prod, frontier_field, preceding_syntax_token_index=None):
        DecodingAction.__init__(self, t, parent_t, frontier_prod, frontier_field, preceding_syntax_token_index)


class GenTokenDecodingAction(GenTokenAction, DecodingAction):
    def __init__(self, t, parent_t, frontier_prod, frontier_field, token, preceding_syntax_token_index=None):
        GenTokenAction.__init__(self, token)
        DecodingAction.__init__(self, t, parent_t, frontier_prod, frontier_field, preceding_syntax_token_index)
