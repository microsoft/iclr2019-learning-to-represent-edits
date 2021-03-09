# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import List
from urllib import request

from diff_representation.asdl.syntax_tree import AbstractSyntaxNode
from diff_representation.dataset import *
from diff_representation.asdl.transition_system import *
from diff_representation.asdl.grammar import *


def extract_unary_closure(data_file):
    csharp_grammar_text = request.urlopen('https://raw.githubusercontent.com/dotnet/roslyn/master/src/Compilers'
                                          '/CSharp/Portable/Syntax/Syntax.xml').read()
    grammar = ASDLGrammar.from_roslyn_xml(csharp_grammar_text, pruning=True)
    transition_system = TransitionSystem(grammar)

    DataSet.load_from_jsonl(data_file, type='tree', transition_system=transition_system)


def get_unary_closure_syntax_sub_tree(ast_root: AbstractSyntaxNode,
                                      unary_closure_root: AbstractSyntaxNode,
                                      unary_closure_last_node: AbstractSyntaxNode) -> List[AbstractSyntaxNode]:

    if ast_root.is_pre_terminal and len(ast_root.fields) <= 1:
        if unary_closure_root and not unary_closure_root.is_pre_terminal:  # has at least one intermediate production
            return [unary_closure_root]
        else:
            return []
    else:
        # case 1: only one child field has instantiated values
        instantiated_fields = [(i, field) for i, field in enumerate(ast_root.fields) if field.value_count > 1]
        if len(instantiated_fields) == 1:
            idx, instantiated_field = instantiated_fields[0]
            tgt_field = unary_closure_last_node.fields[idx]
            if tgt_field.type.is_composite and instantiated_field.value_count == 1:
                child_node = instantiated_field.as_value_list[0]
                cloned_node = AbstractSyntaxNode(child_node.production)
                tgt_field.add_value(cloned_node)
                unary_closure_last_node = cloned_node

                results = get_unary_closure_syntax_sub_tree(child_node, unary_closure_root, unary_closure_last_node)
                return results

        # other cases
        results = []
        if unary_closure_root and not unary_closure_root.is_pre_terminal:
            results.append(unary_closure_root)

        for field_id, instantiated_field in instantiated_fields:
            tgt_field = unary_closure_last_node.fields[field_id]
            if instantiated_field.field.is_composite and instantiated_field.value_count == 1:
                pass

        if len(ast_root.fields) > 1:
            unary_closures = []
            if unary_closure_root and not unary_closure_root.is_pre_terminal:
                unary_closures.append(unary_closure_root)

            for field in ast_root.fields:
                if field.type.is_composite:
                    pass
                else:
                    pass

