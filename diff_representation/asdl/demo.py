# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from diff_representation.asdl.grammar import ASDLGrammar
from diff_representation.asdl.syntax_tree import AbstractSyntaxTree
from diff_representation.asdl.transition_system.hypothesis import Hypothesis
from diff_representation.asdl.transition_system.transition import *

if __name__ == '__main__':
    import urllib.request as request
    csharp_grammar_text = request.urlopen('https://raw.githubusercontent.com/dotnet/roslyn/master/src/Compilers'
                                          '/CSharp/Portable/Syntax/Syntax.xml').read()

    fields_to_ignore = ['SemicolonToken', 'OpenBraceToken', 'CloseBraceToken', 'CommaToken', 'ColonToken', 'StartQuoteToken', 'EndQuoteToken', 'OpenBracketToken', 'CloseBracketToken', 'NewKeyword']

    grammar = ASDLGrammar.from_roslyn_xml(csharp_grammar_text, pruning=True)

    open('grammar.json', 'w').write(grammar.to_json())
    ast_json = open('C:\\Users\\t-peyi\\Research\\best-practices\\mining\\IdiomaticChangeMining\\bin\Debug\\netcoreapp2.0\\ast.json').read()
    ast_root = grammar.get_ast_from_json_str(ast_json)
    syntax_tree = AbstractSyntaxTree(ast_root)
    print(ast_root.to_string())
    print(ast_root.size)

    transition = TransitionSystem(grammar)
    actions = transition.get_actions(ast_root)
    decode_actions = transition.get_decoding_actions(ast_root)

    print('Len actions:', len(decode_actions))

    with open('actions.txt', 'w') as f:
        for action in actions:
            f.write(str(action) + '\n')

    hyp = Hypothesis()
    for action, decode_action in zip(actions, decode_actions):
        assert action.__class__ in transition.get_valid_continuation_types(hyp)
        if isinstance(action, ApplyRuleAction):
            assert action.production in transition.get_valid_continuating_productions(hyp)
            assert action.production == decode_action.production
        elif isinstance(action, GenTokenAction):
            assert action.token == decode_action.token

        if hyp.frontier_node:
            assert hyp.frontier_field == decode_action.frontier_field
            assert hyp.frontier_node.production == decode_action.frontier_prod

        hyp.apply_action(action)
    print(hyp.tree.to_string() == ast_root.to_string())


