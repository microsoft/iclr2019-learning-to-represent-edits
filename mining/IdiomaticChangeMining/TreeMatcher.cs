// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Microsoft.CodeAnalysis.Differencing;
using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace IdiomaticChangeMining
{
    public class TreeMatcher
    {
        public static Dictionary<SyntaxNode, SyntaxNode> ComputeMatch(SyntaxNode leftRoot, SyntaxNode rightRoot)
        {
            var matchedNodes = new Dictionary<SyntaxNode, SyntaxNode>();

            var m = new BlockTreeComparer();
            var match = m.ComputeMatch(leftRoot, rightRoot);
            foreach (var nodeMatch in match.Matches)
            {
                matchedNodes[nodeMatch.Key] = nodeMatch.Value;
                SyntaxNode leftBody = nodeMatch.Key;
                SyntaxNode rightBody = nodeMatch.Value;
                if (nodeMatch.Key is BaseMethodDeclarationSyntax leftMatch && nodeMatch.Value is BaseMethodDeclarationSyntax rightMatch)
                {

                    leftBody = leftMatch.Body != null ? (SyntaxNode)leftMatch.Body : (SyntaxNode)leftMatch.ExpressionBody;
                    rightBody = rightMatch.Body != null ? (SyntaxNode)rightMatch.Body : (SyntaxNode)rightMatch.ExpressionBody;
                }
                try
                {
                    var b = new StatementTreeDistance();
                    var statementMatch = b.ComputeMatch(leftBody, rightBody);
                    foreach (var stMatch in statementMatch.Matches)
                    {
                        matchedNodes[stMatch.Key] = stMatch.Value;

                        // Recurse into Anonymous functions
                        if (stMatch.Key is AnonymousFunctionExpressionSyntax leftAnonMatch && stMatch.Value is AnonymousFunctionExpressionSyntax rightAnonMatch)
                        {
                            foreach (var (left, right) in ComputeMatch(leftAnonMatch.Body, rightAnonMatch.Body))
                            {
                                matchedNodes[left] = right;
                            }
                        }
                    }
                } catch (Exception)
                {
                    // ignore if we cannot descend
                }
            }

            return matchedNodes;
        }

        public static List<Edit<SyntaxNode>> GetTreeEdits(SyntaxTree left, SyntaxTree right)
        {
            var m = new BlockTreeComparer();
            var match = m.ComputeMatch(left.GetRoot(), right.GetRoot());

            var edits = new List<Edit<SyntaxNode>>();

            foreach (var edit in match.GetTreeEdits().Edits)
            {
                if (edit.Kind == EditKind.None)
                {
                    edits.Add(edit);
                }
                else
                {
                    // TODO
                }
            }
            throw new NotImplementedException();
        }

        public static void Main(string[] args)
        {
            var beforeAst = CSharpSyntaxTree.ParseText(File.ReadAllText(args[0]));
            var afterAst = CSharpSyntaxTree.ParseText(File.ReadAllText(args[1]));

            var match = ComputeMatch(beforeAst.GetRoot(), afterAst.GetRoot());
            foreach (var (left, right) in match)
            {
                Console.WriteLine($"From {left} to {right}.");
            }

        }
    }
}
