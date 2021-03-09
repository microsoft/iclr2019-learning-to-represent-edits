// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Microsoft.CodeAnalysis.Differencing;
using Microsoft.CodeAnalysis.Text;
using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Diagnostics;
using System.Text;

namespace IdiomaticChangeMining
{
    public abstract class AbstractCSharpTreeMatch : TreeComparer<SyntaxNode>
    {
        internal const int IgnoredNode = -1;

        protected const double ExactMatchDist = 0.0;
        protected const double EpsilonDist = 0.00001;

        protected override bool TreesEqual(SyntaxNode oldNode, SyntaxNode newNode)
        {
            return oldNode.SyntaxTree == newNode.SyntaxTree;
        }

        protected override TextSpan GetSpan(SyntaxNode node)
        {
            return node.Span;
        }

        #region Comparison

        public sealed override double GetDistance(SyntaxNode oldNode, SyntaxNode newNode)
        {
            Debug.Assert(GetLabel(oldNode) == GetLabel(newNode) && GetLabel(oldNode) != IgnoredNode);

            if (oldNode == newNode)
            {
                return ExactMatchDist;
            }

            if (TryComputeWeightedDistance(oldNode, newNode, out var weightedDistance))
            {
                if (weightedDistance == ExactMatchDist && !SyntaxFactory.AreEquivalent(oldNode, newNode))
                {
                    weightedDistance = EpsilonDist;
                }

                return weightedDistance;
            }

            return ComputeValueDistance(oldNode, newNode);
        }

        protected abstract bool TryComputeWeightedDistance(SyntaxNode oldNode, SyntaxNode newNode, out double weightedDistance);

        protected static double ComputeValueDistance(SyntaxNode oldNode, SyntaxNode newNode)
        {
            if (SyntaxFactory.AreEquivalent(oldNode, newNode))
            {
                return ExactMatchDist;
            }

            double distance = ComputeDistance(oldNode, newNode);

            // We don't want to return an exact match, because there
            // must be something different, since we got here 
            return (distance == ExactMatchDist) ? EpsilonDist : distance;
        }

        public static double ComputeDistance(SyntaxNodeOrToken oldNodeOrToken, SyntaxNodeOrToken newNodeOrToken)
        {
            Debug.Assert(newNodeOrToken.IsToken == oldNodeOrToken.IsToken);

            double distance;
            if (oldNodeOrToken.IsToken)
            {
                var leftToken = oldNodeOrToken.AsToken();
                var rightToken = newNodeOrToken.AsToken();

                distance = ComputeDistance(leftToken, rightToken);
                Debug.Assert(!SyntaxFactory.AreEquivalent(leftToken, rightToken) || distance == ExactMatchDist);
            }
            else
            {
                var leftNode = oldNodeOrToken.AsNode();
                var rightNode = newNodeOrToken.AsNode();

                distance = ComputeDistance(leftNode, rightNode);
                Debug.Assert(!SyntaxFactory.AreEquivalent(leftNode, rightNode) || distance == ExactMatchDist);
            }

            return distance;
        }

        /// <summary>
        /// Enumerates tokens of all nodes in the list. Doesn't include separators.
        /// </summary>
        public static IEnumerable<SyntaxToken> GetDescendantTokensIgnoringSeparators<TSyntaxNode>(SeparatedSyntaxList<TSyntaxNode> list)
            where TSyntaxNode : SyntaxNode
        {
            foreach (var node in list)
            {
                foreach (var token in node.DescendantTokens())
                {
                    yield return token;
                }
            }
        }

        /// <summary>
        /// Calculates the distance between two syntax nodes, disregarding trivia. 
        /// </summary>
        /// <remarks>
        /// Distance is a number within [0, 1], the smaller the more similar the nodes are. 
        /// </remarks>
        public static double ComputeDistance(SyntaxNode oldNode, SyntaxNode newNode)
        {
            if (oldNode == null || newNode == null)
            {
                return (oldNode == newNode) ? 0.0 : 1.0;
            }

            return ComputeDistance(oldNode.DescendantTokens(), newNode.DescendantTokens());
        }

        /// <summary>
        /// Calculates the distance between two syntax tokens, disregarding trivia. 
        /// </summary>
        /// <remarks>
        /// Distance is a number within [0, 1], the smaller the more similar the tokens are. 
        /// </remarks>
        public static double ComputeDistance(SyntaxToken oldToken, SyntaxToken newToken)
        {
            return LongestCommonSubstring.ComputeDistance(oldToken.Text, newToken.Text);
        }

        /// <summary>
        /// Calculates the distance between two sequences of syntax tokens, disregarding trivia. 
        /// </summary>
        /// <remarks>
        /// Distance is a number within [0, 1], the smaller the more similar the sequences are. 
        /// </remarks>
        public static double ComputeDistance(IEnumerable<SyntaxToken> oldTokens, IEnumerable<SyntaxToken> newTokens)
        {
            return LcsTokens.Instance.ComputeDistance(oldTokens.AsImmutableOrEmpty(), newTokens.AsImmutableOrEmpty());
        }

        /// <summary>
        /// Calculates the distance between two sequences of syntax tokens, disregarding trivia. 
        /// </summary>
        /// <remarks>
        /// Distance is a number within [0, 1], the smaller the more similar the sequences are. 
        /// </remarks>
        public static double ComputeDistance(ImmutableArray<SyntaxToken> oldTokens, ImmutableArray<SyntaxToken> newTokens)
        {
            return LcsTokens.Instance.ComputeDistance(oldTokens.NullToEmpty(), newTokens.NullToEmpty());
        }

        /// <summary>
        /// Calculates the distance between two sequences of syntax nodes, disregarding trivia. 
        /// </summary>
        /// <remarks>
        /// Distance is a number within [0, 1], the smaller the more similar the sequences are. 
        /// </remarks>
        public static double ComputeDistance(IEnumerable<SyntaxNode> oldNodes, IEnumerable<SyntaxNode> newNodes)
        {
            return LcsNodes.Instance.ComputeDistance(oldNodes.AsImmutableOrEmpty(), newNodes.AsImmutableOrEmpty());
        }

        /// <summary>
        /// Calculates the distance between two sequences of syntax tokens, disregarding trivia. 
        /// </summary>
        /// <remarks>
        /// Distance is a number within [0, 1], the smaller the more similar the sequences are. 
        /// </remarks>
        public static double ComputeDistance(ImmutableArray<SyntaxNode> oldNodes, ImmutableArray<SyntaxNode> newNodes)
        {
            return LcsNodes.Instance.ComputeDistance(oldNodes.NullToEmpty(), newNodes.NullToEmpty());
        }

        /// <summary>
        /// Calculates the edits that transform one sequence of syntax nodes to another, disregarding trivia.
        /// </summary>
        public static IEnumerable<SequenceEdit> GetSequenceEdits(IEnumerable<SyntaxNode> oldNodes, IEnumerable<SyntaxNode> newNodes)
        {
            return LcsNodes.Instance.GetEdits(oldNodes.AsImmutableOrEmpty(), newNodes.AsImmutableOrEmpty());
        }

        /// <summary>
        /// Calculates the edits that transform one sequence of syntax nodes to another, disregarding trivia.
        /// </summary>
        public static IEnumerable<SequenceEdit> GetSequenceEdits(ImmutableArray<SyntaxNode> oldNodes, ImmutableArray<SyntaxNode> newNodes)
        {
            return LcsNodes.Instance.GetEdits(oldNodes.NullToEmpty(), newNodes.NullToEmpty());
        }

        /// <summary>
        /// Calculates the edits that transform one sequence of syntax tokens to another, disregarding trivia.
        /// </summary>
        public static IEnumerable<SequenceEdit> GetSequenceEdits(IEnumerable<SyntaxToken> oldTokens, IEnumerable<SyntaxToken> newTokens)
        {
            return LcsTokens.Instance.GetEdits(oldTokens.AsImmutableOrEmpty(), newTokens.AsImmutableOrEmpty());
        }

        /// <summary>
        /// Calculates the edits that transform one sequence of syntax tokens to another, disregarding trivia.
        /// </summary>
        public static IEnumerable<SequenceEdit> GetSequenceEdits(ImmutableArray<SyntaxToken> oldTokens, ImmutableArray<SyntaxToken> newTokens)
        {
            return LcsTokens.Instance.GetEdits(oldTokens.NullToEmpty(), newTokens.NullToEmpty());
        }

        private sealed class LcsTokens : LongestCommonImmutableArraySubsequence<SyntaxToken>
        {
            internal static readonly LcsTokens Instance = new LcsTokens();

            protected override bool Equals(SyntaxToken oldElement, SyntaxToken newElement)
            {
                return SyntaxFactory.AreEquivalent(oldElement, newElement);
            }
        }

        private sealed class LcsNodes : LongestCommonImmutableArraySubsequence<SyntaxNode>
        {
            internal static readonly LcsNodes Instance = new LcsNodes();

            protected override bool Equals(SyntaxNode oldElement, SyntaxNode newElement)
            {
                return SyntaxFactory.AreEquivalent(oldElement, newElement);
            }
        }

        /// <summary>
        /// Calculates longest common substring using Wagner algorithm.
        /// </summary>
        internal sealed class LongestCommonSubstring : LongestCommonSubsequence<string>
        {
            private static readonly LongestCommonSubstring s_instance = new LongestCommonSubstring();

            private LongestCommonSubstring()
            {
            }

            protected override bool ItemsEqual(string oldSequence, int oldIndex, string newSequence, int newIndex)
            {
                return oldSequence[oldIndex] == newSequence[newIndex];
            }

            public static double ComputeDistance(string oldValue, string newValue)
            {
                return s_instance.ComputeDistance(oldValue, oldValue.Length, newValue, newValue.Length);
            }

            public static IEnumerable<SequenceEdit> GetEdits(string oldValue, string newValue)
            {
                return s_instance.GetEdits(oldValue, oldValue.Length, newValue, newValue.Length);
            }
        }
        #endregion
    }

    class StatementTreeDistance : AbstractCSharpTreeMatch
    {
        private readonly SyntaxNode _oldRootChild;
        private readonly SyntaxNode _newRootChild;
        private readonly SyntaxNode _oldRoot;
        private readonly SyntaxNode _newRoot;

        #region Tree Traversal

        protected override bool TryGetParent(SyntaxNode node, out SyntaxNode parent)
        {
            parent = node.Parent;
            while (parent != null && !HasLabel(parent))
            {
                parent = parent.Parent;
            }

            return parent != null;
        }

        protected override IEnumerable<SyntaxNode> GetChildren(SyntaxNode node)
        {
            Debug.Assert(HasLabel(node));

            if (node == _oldRoot || node == _newRoot)
            {
                return EnumerateRootChildren(node);
            }

            return IsLeaf(node) ? null : EnumerateNonRootChildren(node);
        }

        private IEnumerable<SyntaxNode> EnumerateNonRootChildren(SyntaxNode node)
        {
            foreach (var child in node.ChildNodes())
            {
                if (IsLambdaBodyStatementOrExpression(child))
                {
                    continue;
                }

                if (HasLabel(child))
                {
                    yield return child;
                }
                else
                {
                    foreach (var descendant in child.DescendantNodes(DescendIntoChildren))
                    {
                        if (HasLabel(descendant))
                        {
                            yield return descendant;
                        }
                    }
                }
            }
        }

        private IEnumerable<SyntaxNode> EnumerateRootChildren(SyntaxNode root)
        {
            Debug.Assert(_oldRoot != null && _newRoot != null);

            var child = (root == _oldRoot) ? _oldRootChild : _newRootChild;

            if (GetLabelImpl(child) != Label.Ignored)
            {
                yield return child;
            }
            else
            {
                foreach (var descendant in child.DescendantNodes(DescendIntoChildren))
                {
                    if (HasLabel(descendant))
                    {
                        yield return descendant;
                    }
                }
            }
        }

        private bool DescendIntoChildren(SyntaxNode node)
        {
            return !IsLambdaBodyStatementOrExpression(node) && !HasLabel(node);
        }

        protected sealed override IEnumerable<SyntaxNode> GetDescendants(SyntaxNode node)
        {
            if (node == _oldRoot || node == _newRoot)
            {
                Debug.Assert(_oldRoot != null && _newRoot != null);

                var rootChild = (node == _oldRoot) ? _oldRootChild : _newRootChild;

                if (HasLabel(rootChild))
                {
                    yield return rootChild;
                }

                node = rootChild;
            }

            // TODO: avoid allocation of closure
            foreach (var descendant in node.DescendantNodes(descendIntoChildren: c => !IsLeaf(c) && (c == node || !IsLambdaBodyStatementOrExpression(c))))
            {
                if (!IsLambdaBodyStatementOrExpression(descendant) && HasLabel(descendant))
                {
                    yield return descendant;
                }
            }
        }

        private static bool IsLeaf(SyntaxNode node)
        {
            Classify(node.Kind(), node, out var isLeaf);
            return isLeaf;
        }

        #endregion

        #region Labels

        // Assumptions:
        // - Each listed label corresponds to one or more syntax kinds.
        // - Nodes with same labels might produce Update edits, nodes with different labels don't. 
        // - If IsTiedToParent(label) is true for a label then all its possible parent labels must precede the label.
        //   (i.e. both MethodDeclaration and TypeDeclaration must precede TypeParameter label).
        internal enum Label
        {
            ConstructorDeclaration,
            Block,
            CheckedStatement,
            UnsafeStatement,

            TryStatement,
            CatchClause,                      // tied to parent
            CatchDeclaration,                 // tied to parent
            CatchFilterClause,                // tied to parent
            FinallyClause,                    // tied to parent
            ForStatement,
            ForStatementPart,                 // tied to parent
            ForEachStatement,
            UsingStatement,
            FixedStatement,
            LockStatement,
            WhileStatement,
            DoStatement,
            IfStatement,
            ElseClause,                        // tied to parent 

            SwitchStatement,
            SwitchSection,
            CasePatternSwitchLabel,            // tied to parent
            WhenClause,

            YieldStatement,                    // tied to parent
            GotoStatement,
            GotoCaseStatement,
            BreakContinueStatement,
            ReturnThrowStatement,
            ExpressionStatement,

            LabeledStatement,

            // TODO: 
            // Ideally we could declare LocalVariableDeclarator tied to the first enclosing node that defines local scope (block, foreach, etc.)
            // Also consider handling LocalDeclarationStatement as just a bag of variable declarators,
            // so that variable declarators contained in one can be matched with variable declarators contained in the other.
            LocalDeclarationStatement,         // tied to parent
            LocalVariableDeclaration,          // tied to parent
            LocalVariableDeclarator,           // tied to parent

            SingleVariableDesignation,
            AwaitExpression,
            NestedFunction,

            FromClause,
            QueryBody,
            FromClauseLambda,                 // tied to parent
            LetClauseLambda,                  // tied to parent
            WhereClauseLambda,                // tied to parent
            OrderByClause,                    // tied to parent
            OrderingLambda,                   // tied to parent
            SelectClauseLambda,               // tied to parent
            JoinClauseLambda,                 // tied to parent
            JoinIntoClause,                   // tied to parent
            GroupClauseLambda,                // tied to parent
            QueryContinuation,                // tied to parent

            // helpers:
            Count,
            Ignored = IgnoredNode
        }

        private static int TiedToAncestor(Label label)
        {
            switch (label)
            {
                case Label.LocalDeclarationStatement:
                case Label.LocalVariableDeclaration:
                case Label.LocalVariableDeclarator:
                case Label.GotoCaseStatement:
                case Label.BreakContinueStatement:
                case Label.ElseClause:
                case Label.CatchClause:
                case Label.CatchDeclaration:
                case Label.CatchFilterClause:
                case Label.FinallyClause:
                case Label.ForStatementPart:
                case Label.YieldStatement:
                case Label.FromClauseLambda:
                case Label.LetClauseLambda:
                case Label.WhereClauseLambda:
                case Label.OrderByClause:
                case Label.OrderingLambda:
                case Label.SelectClauseLambda:
                case Label.JoinClauseLambda:
                case Label.JoinIntoClause:
                case Label.GroupClauseLambda:
                case Label.QueryContinuation:
                case Label.CasePatternSwitchLabel:
                    return 1;

                default:
                    return 0;
            }
        }

        /// <summary>
        /// <paramref name="nodeOpt"/> is null only when comparing value equality of a tree node.
        /// </summary>
        internal static Label Classify(SyntaxKind kind, SyntaxNode nodeOpt, out bool isLeaf)
        {
            // Notes:
            // A descendant of a leaf node may be a labeled node that we don't want to visit if 
            // we are comparing its parent node (used for lambda bodies).
            // 
            // Expressions are ignored but they may contain nodes that should be matched by tree comparer.
            // (e.g. lambdas, declaration expressions). Descending to these nodes is handled in EnumerateChildren.

            isLeaf = false;

            // If the node is a for loop Initializer, Condition, or Incrementor expression we label it as "ForStatementPart".
            // We need to capture it in the match since these expressions can be "active statements" and as such we need to map them.
            //
            // The parent is not available only when comparing nodes for value equality.
            if (nodeOpt != null && nodeOpt.Parent.IsKind(SyntaxKind.ForStatement) && nodeOpt is ExpressionSyntax)
            {
                return Label.ForStatementPart;
            }

            switch (kind)
            {
                case SyntaxKind.ConstructorDeclaration:
                    // Root when matching constructor bodies.
                    return Label.ConstructorDeclaration;

                case SyntaxKind.Block:
                    return Label.Block;

                case SyntaxKind.LocalDeclarationStatement:
                    return Label.LocalDeclarationStatement;

                case SyntaxKind.VariableDeclaration:
                    return Label.LocalVariableDeclaration;

                case SyntaxKind.VariableDeclarator:
                    return Label.LocalVariableDeclarator;

                case SyntaxKind.SingleVariableDesignation:
                    return Label.SingleVariableDesignation;

                case SyntaxKind.LabeledStatement:
                    return Label.LabeledStatement;

                case SyntaxKind.EmptyStatement:
                    isLeaf = true;
                    return Label.Ignored;

                case SyntaxKind.GotoStatement:
                    isLeaf = true;
                    return Label.GotoStatement;

                case SyntaxKind.GotoCaseStatement:
                case SyntaxKind.GotoDefaultStatement:
                    isLeaf = true;
                    return Label.GotoCaseStatement;

                case SyntaxKind.BreakStatement:
                case SyntaxKind.ContinueStatement:
                    isLeaf = true;
                    return Label.BreakContinueStatement;

                case SyntaxKind.ReturnStatement:
                case SyntaxKind.ThrowStatement:
                    return Label.ReturnThrowStatement;

                case SyntaxKind.ExpressionStatement:
                    return Label.ExpressionStatement;

                case SyntaxKind.YieldBreakStatement:
                case SyntaxKind.YieldReturnStatement:
                    return Label.YieldStatement;

                case SyntaxKind.DoStatement:
                    return Label.DoStatement;

                case SyntaxKind.WhileStatement:
                    return Label.WhileStatement;

                case SyntaxKind.ForStatement:
                    return Label.ForStatement;

                case SyntaxKind.ForEachVariableStatement:
                case SyntaxKind.ForEachStatement:
                    return Label.ForEachStatement;

                case SyntaxKind.UsingStatement:
                    return Label.UsingStatement;

                case SyntaxKind.FixedStatement:
                    return Label.FixedStatement;

                case SyntaxKind.CheckedStatement:
                case SyntaxKind.UncheckedStatement:
                    return Label.CheckedStatement;

                case SyntaxKind.UnsafeStatement:
                    return Label.UnsafeStatement;

                case SyntaxKind.LockStatement:
                    return Label.LockStatement;

                case SyntaxKind.IfStatement:
                    return Label.IfStatement;

                case SyntaxKind.ElseClause:
                    return Label.ElseClause;

                case SyntaxKind.SwitchStatement:
                    return Label.SwitchStatement;

                case SyntaxKind.SwitchSection:
                    return Label.SwitchSection;

                case SyntaxKind.CaseSwitchLabel:
                case SyntaxKind.DefaultSwitchLabel:
                    // Switch labels are included in the "value" of the containing switch section.
                    // We don't need to analyze case expressions.
                    isLeaf = true;
                    return Label.Ignored;

                case SyntaxKind.WhenClause:
                    return Label.WhenClause;

                case SyntaxKind.CasePatternSwitchLabel:
                    return Label.CasePatternSwitchLabel;

                case SyntaxKind.TryStatement:
                    return Label.TryStatement;

                case SyntaxKind.CatchClause:
                    return Label.CatchClause;

                case SyntaxKind.CatchDeclaration:
                    // the declarator of the exception variable
                    return Label.CatchDeclaration;

                case SyntaxKind.CatchFilterClause:
                    return Label.CatchFilterClause;

                case SyntaxKind.FinallyClause:
                    return Label.FinallyClause;

                case SyntaxKind.LocalFunctionStatement:
                case SyntaxKind.ParenthesizedLambdaExpression:
                case SyntaxKind.SimpleLambdaExpression:
                case SyntaxKind.AnonymousMethodExpression:
                    return Label.NestedFunction;

                case SyntaxKind.FromClause:
                    // The first from clause of a query is not a lambda.
                    // We have to assign it a label different from "FromClauseLambda"
                    // so that we won't match lambda-from to non-lambda-from.
                    // 
                    // Since FromClause declares range variables we need to include it in the map,
                    // so that we are able to map range variable declarations.
                    // Therefore we assign it a dedicated label.
                    // 
                    // The parent is not available only when comparing nodes for value equality.
                    // In that case it doesn't matter what label the node has as long as it has some.
                    if (nodeOpt == null || nodeOpt.Parent.IsKind(SyntaxKind.QueryExpression))
                    {
                        return Label.FromClause;
                    }

                    return Label.FromClauseLambda;

                case SyntaxKind.QueryBody:
                    return Label.QueryBody;

                case SyntaxKind.QueryContinuation:
                    return Label.QueryContinuation;

                case SyntaxKind.LetClause:
                    return Label.LetClauseLambda;

                case SyntaxKind.WhereClause:
                    return Label.WhereClauseLambda;

                case SyntaxKind.OrderByClause:
                    return Label.OrderByClause;

                case SyntaxKind.AscendingOrdering:
                case SyntaxKind.DescendingOrdering:
                    return Label.OrderingLambda;

                case SyntaxKind.SelectClause:
                    return Label.SelectClauseLambda;

                case SyntaxKind.JoinClause:
                    return Label.JoinClauseLambda;

                case SyntaxKind.JoinIntoClause:
                    return Label.JoinIntoClause;

                case SyntaxKind.GroupClause:
                    return Label.GroupClauseLambda;

                case SyntaxKind.IdentifierName:
                case SyntaxKind.QualifiedName:
                case SyntaxKind.GenericName:
                case SyntaxKind.TypeArgumentList:
                case SyntaxKind.AliasQualifiedName:
                case SyntaxKind.PredefinedType:
                case SyntaxKind.ArrayType:
                case SyntaxKind.ArrayRankSpecifier:
                case SyntaxKind.PointerType:
                case SyntaxKind.NullableType:
                case SyntaxKind.TupleType:
                case SyntaxKind.RefType:
                case SyntaxKind.OmittedTypeArgument:
                case SyntaxKind.NameColon:
                case SyntaxKind.StackAllocArrayCreationExpression:
                case SyntaxKind.OmittedArraySizeExpression:
                case SyntaxKind.ThisExpression:
                case SyntaxKind.BaseExpression:
                case SyntaxKind.ArgListExpression:
                case SyntaxKind.NumericLiteralExpression:
                case SyntaxKind.StringLiteralExpression:
                case SyntaxKind.CharacterLiteralExpression:
                case SyntaxKind.TrueLiteralExpression:
                case SyntaxKind.FalseLiteralExpression:
                case SyntaxKind.NullLiteralExpression:
                case SyntaxKind.TypeOfExpression:
                case SyntaxKind.SizeOfExpression:
                case SyntaxKind.DefaultExpression:
                case SyntaxKind.ConstantPattern:
                case SyntaxKind.DiscardDesignation:
                    // can't contain a lambda/await/anonymous type:
                    isLeaf = true;
                    return Label.Ignored;

                case SyntaxKind.AwaitExpression:
                    return Label.AwaitExpression;

                default:
                    // any other node may contain a lambda:
                    return Label.Ignored;
            }
        }

        protected override int GetLabel(SyntaxNode node)
        {
            return (int)GetLabelImpl(node);
        }

        internal static Label GetLabelImpl(SyntaxNode node)
        {
            return Classify(node.Kind(), node, out var isLeaf);
        }

        internal static bool HasLabel(SyntaxNode node)
        {
            return GetLabelImpl(node) != Label.Ignored;
        }

        protected override int LabelCount
        {
            get { return (int)Label.Count; }
        }

        protected override int TiedToAncestor(int label)
        {
            return TiedToAncestor((Label)label);
        }

        #endregion

        #region Comparisons

        internal static bool IgnoreLabeledChild(SyntaxKind kind)
        {
            // In most cases we can determine Label based on child kind.
            // The only cases when we can't are
            // - for Initializer, Condition and Incrementor expressions in ForStatement.
            // - first from clause of a query expression.
            return Classify(kind, null, out var isLeaf) != Label.Ignored;
        }

        public override bool ValuesEqual(SyntaxNode left, SyntaxNode right)
        {
            // only called from the tree matching alg, which only operates on nodes that are labeled.
            Debug.Assert(HasLabel(left));
            Debug.Assert(HasLabel(right));

            Func<SyntaxKind, bool> ignoreChildNode;
            switch (left.Kind())
            {
                case SyntaxKind.SwitchSection:
                    return Equal((SwitchSectionSyntax)left, (SwitchSectionSyntax)right);

                case SyntaxKind.ForStatement:
                    // The only children of ForStatement are labeled nodes and punctuation.
                    return true;

                default:
                    // When comparing the value of a node with its partner we are deciding whether to add an Update edit for the pair.
                    // If the actual change is under a descendant labeled node we don't want to attribute it to the node being compared,
                    // so we skip all labeled children when recursively checking for equivalence.
                    if (IsLeaf(left))
                    {
                        ignoreChildNode = null;
                    }
                    else
                    {
                        ignoreChildNode = IgnoreLabeledChild;
                    }

                    break;
            }

            return SyntaxFactory.AreEquivalent(left, right, ignoreChildNode);
        }

        private bool Equal(SwitchSectionSyntax left, SwitchSectionSyntax right)
        {
            return SyntaxFactory.AreEquivalent(left.Labels, right.Labels, null)
                && SyntaxFactory.AreEquivalent(left.Statements, right.Statements, ignoreChildNode: IgnoreLabeledChild);
        }

        protected override bool TryComputeWeightedDistance(SyntaxNode leftNode, SyntaxNode rightNode, out double distance)
        {
            switch (leftNode.Kind())
            {
                case SyntaxKind.VariableDeclarator:
                    distance = ComputeDistance(
                        ((VariableDeclaratorSyntax)leftNode).Identifier,
                        ((VariableDeclaratorSyntax)rightNode).Identifier);
                    return true;

                case SyntaxKind.ForStatement:
                    var leftFor = (ForStatementSyntax)leftNode;
                    var rightFor = (ForStatementSyntax)rightNode;
                    distance = ComputeWeightedDistance(leftFor, rightFor);
                    return true;

                case SyntaxKind.ForEachStatement:
                case SyntaxKind.ForEachVariableStatement:
                    {

                        var leftForEach = (CommonForEachStatementSyntax)leftNode;
                        var rightForEach = (CommonForEachStatementSyntax)rightNode;
                        distance = ComputeWeightedDistance(leftForEach, rightForEach);
                        return true;
                    }

                case SyntaxKind.UsingStatement:
                    var leftUsing = (UsingStatementSyntax)leftNode;
                    var rightUsing = (UsingStatementSyntax)rightNode;

                    if (leftUsing.Declaration != null && rightUsing.Declaration != null)
                    {
                        distance = ComputeWeightedDistance(
                            leftUsing.Declaration,
                            leftUsing.Statement,
                            rightUsing.Declaration,
                            rightUsing.Statement);
                    }
                    else
                    {
                        distance = ComputeWeightedDistance(
                            (SyntaxNode)leftUsing.Expression ?? leftUsing.Declaration,
                            leftUsing.Statement,
                            (SyntaxNode)rightUsing.Expression ?? rightUsing.Declaration,
                            rightUsing.Statement);
                    }

                    return true;

                case SyntaxKind.LockStatement:
                    var leftLock = (LockStatementSyntax)leftNode;
                    var rightLock = (LockStatementSyntax)rightNode;
                    distance = ComputeWeightedDistance(leftLock.Expression, leftLock.Statement, rightLock.Expression, rightLock.Statement);
                    return true;

                case SyntaxKind.FixedStatement:
                    var leftFixed = (FixedStatementSyntax)leftNode;
                    var rightFixed = (FixedStatementSyntax)rightNode;
                    distance = ComputeWeightedDistance(leftFixed.Declaration, leftFixed.Statement, rightFixed.Declaration, rightFixed.Statement);
                    return true;

                case SyntaxKind.WhileStatement:
                    var leftWhile = (WhileStatementSyntax)leftNode;
                    var rightWhile = (WhileStatementSyntax)rightNode;
                    distance = ComputeWeightedDistance(leftWhile.Condition, leftWhile.Statement, rightWhile.Condition, rightWhile.Statement);
                    return true;

                case SyntaxKind.DoStatement:
                    var leftDo = (DoStatementSyntax)leftNode;
                    var rightDo = (DoStatementSyntax)rightNode;
                    distance = ComputeWeightedDistance(leftDo.Condition, leftDo.Statement, rightDo.Condition, rightDo.Statement);
                    return true;

                case SyntaxKind.IfStatement:
                    var leftIf = (IfStatementSyntax)leftNode;
                    var rightIf = (IfStatementSyntax)rightNode;
                    distance = ComputeWeightedDistance(leftIf.Condition, leftIf.Statement, rightIf.Condition, rightIf.Statement);
                    return true;

                case SyntaxKind.Block:
                    BlockSyntax leftBlock = (BlockSyntax)leftNode;
                    BlockSyntax rightBlock = (BlockSyntax)rightNode;
                    return TryComputeWeightedDistance(leftBlock, rightBlock, out distance);

                case SyntaxKind.CatchClause:
                    distance = ComputeWeightedDistance((CatchClauseSyntax)leftNode, (CatchClauseSyntax)rightNode);
                    return true;

                case SyntaxKind.ParenthesizedLambdaExpression:
                case SyntaxKind.SimpleLambdaExpression:
                case SyntaxKind.AnonymousMethodExpression:
                case SyntaxKind.LocalFunctionStatement:
                    distance = ComputeWeightedDistanceOfNestedFunctions(leftNode, rightNode);
                    return true;

                case SyntaxKind.YieldBreakStatement:
                case SyntaxKind.YieldReturnStatement:
                    // Ignore the expression of yield return. The structure of the state machine is more important than the yielded values.
                    distance = (leftNode.RawKind == rightNode.RawKind) ? 0.0 : 0.1;
                    return true;

                case SyntaxKind.SingleVariableDesignation:
                    distance = ComputeWeightedDistance((SingleVariableDesignationSyntax)leftNode, (SingleVariableDesignationSyntax)rightNode);
                    return true;

                default:
                    distance = 0;
                    return false;
            }
        }

        private static double ComputeWeightedDistanceOfNestedFunctions(SyntaxNode leftNode, SyntaxNode rightNode)
        {
            GetNestedFunctionsParts(leftNode, out var leftParameters, out var leftAsync, out var leftBody, out var leftModifiers, out var leftReturnType, out var leftIdentifier, out var leftTypeParameters);
            GetNestedFunctionsParts(rightNode, out var rightParameters, out var rightAsync, out var rightBody, out var rightModifiers, out var rightReturnType, out var rightIdentifier, out var rightTypeParameters);

            if ((leftAsync.Kind() == SyntaxKind.AsyncKeyword) != (rightAsync.Kind() == SyntaxKind.AsyncKeyword))
            {
                return 1.0;
            }

            double modifierDistance = ComputeDistance(leftModifiers, rightModifiers);
            double returnTypeDistance = ComputeDistance(leftReturnType, rightReturnType);
            double identifierDistance = ComputeDistance(leftIdentifier, rightIdentifier);
            double typeParameterDistance = ComputeDistance(leftTypeParameters, rightTypeParameters);
            double parameterDistance = ComputeDistance(leftParameters, rightParameters);
            double bodyDistance = ComputeDistance(leftBody, rightBody);

            return
                modifierDistance * 0.1 +
                returnTypeDistance * 0.1 +
                identifierDistance * 0.2 +
                typeParameterDistance * 0.2 +
                parameterDistance * 0.2 +
                bodyDistance * 0.2;
        }

        private static void GetNestedFunctionsParts(
            SyntaxNode nestedFunction,
            out IEnumerable<SyntaxToken> parameters,
            out SyntaxToken asyncKeyword,
            out SyntaxNode body,
            out SyntaxTokenList modifiers,
            out TypeSyntax returnType,
            out SyntaxToken identifier,
            out TypeParameterListSyntax typeParameters)
        {
            switch (nestedFunction.Kind())
            {
                case SyntaxKind.SimpleLambdaExpression:
                    var simple = (SimpleLambdaExpressionSyntax)nestedFunction;
                    parameters = simple.Parameter.DescendantTokens();
                    asyncKeyword = simple.AsyncKeyword;
                    body = simple.Body;
                    modifiers = default;
                    returnType = default;
                    identifier = default;
                    typeParameters = default;
                    break;

                case SyntaxKind.ParenthesizedLambdaExpression:
                    var parenthesized = (ParenthesizedLambdaExpressionSyntax)nestedFunction;
                    parameters = GetDescendantTokensIgnoringSeparators(parenthesized.ParameterList.Parameters);
                    asyncKeyword = parenthesized.AsyncKeyword;
                    body = parenthesized.Body;
                    modifiers = default;
                    returnType = default;
                    identifier = default;
                    typeParameters = default;
                    break;

                case SyntaxKind.AnonymousMethodExpression:
                    var anonymous = (AnonymousMethodExpressionSyntax)nestedFunction;
                    if (anonymous.ParameterList != null)
                    {
                        parameters = GetDescendantTokensIgnoringSeparators(anonymous.ParameterList.Parameters);
                    }
                    else
                    {
                        parameters = new SyntaxToken[0];
                    }

                    asyncKeyword = anonymous.AsyncKeyword;
                    body = anonymous.Block;
                    modifiers = default;
                    returnType = default;
                    identifier = default;
                    typeParameters = default;
                    break;

                case SyntaxKind.LocalFunctionStatement:
                    var localFunction = (LocalFunctionStatementSyntax)nestedFunction;
                    parameters = GetDescendantTokensIgnoringSeparators(localFunction.ParameterList.Parameters);
                    asyncKeyword = default;
                    body = (SyntaxNode)localFunction.Body ?? localFunction.ExpressionBody;
                    modifiers = localFunction.Modifiers;
                    returnType = localFunction.ReturnType;
                    identifier = localFunction.Identifier;
                    typeParameters = localFunction.TypeParameterList;
                    break;

                default:
                    throw new Exception($"Unexpected value {nestedFunction.Kind()}");
            }
        }

        private bool TryComputeWeightedDistance(BlockSyntax leftBlock, BlockSyntax rightBlock, out double distance)
        {
            // No block can be matched with the root block.
            // Note that in constructors the root is the constructor declaration, since we need to include 
            // the constructor initializer in the match.
            if (leftBlock.Parent == null ||
                rightBlock.Parent == null ||
                leftBlock.Parent.IsKind(SyntaxKind.ConstructorDeclaration) ||
                rightBlock.Parent.IsKind(SyntaxKind.ConstructorDeclaration))
            {
                distance = 0.0;
                return true;
            }

            if (GetLabel(leftBlock.Parent) != GetLabel(rightBlock.Parent))
            {
                distance = 0.2 + 0.8 * ComputeWeightedBlockDistance(leftBlock, rightBlock);
                return true;
            }

            switch (leftBlock.Parent.Kind())
            {
                case SyntaxKind.IfStatement:
                case SyntaxKind.ForEachStatement:
                case SyntaxKind.ForEachVariableStatement:
                case SyntaxKind.ForStatement:
                case SyntaxKind.WhileStatement:
                case SyntaxKind.DoStatement:
                case SyntaxKind.FixedStatement:
                case SyntaxKind.LockStatement:
                case SyntaxKind.UsingStatement:
                case SyntaxKind.SwitchSection:
                case SyntaxKind.ParenthesizedLambdaExpression:
                case SyntaxKind.SimpleLambdaExpression:
                case SyntaxKind.AnonymousMethodExpression:
                    // value distance of the block body is included:
                    distance = GetDistance(leftBlock.Parent, rightBlock.Parent);
                    return true;

                case SyntaxKind.CatchClause:
                    var leftCatch = (CatchClauseSyntax)leftBlock.Parent;
                    var rightCatch = (CatchClauseSyntax)rightBlock.Parent;
                    if (leftCatch.Declaration == null && leftCatch.Filter == null &&
                        rightCatch.Declaration == null && rightCatch.Filter == null)
                    {
                        var leftTry = (TryStatementSyntax)leftCatch.Parent;
                        var rightTry = (TryStatementSyntax)rightCatch.Parent;

                        distance = 0.5 * ComputeValueDistance(leftTry.Block, rightTry.Block) +
                                   0.5 * ComputeValueDistance(leftBlock, rightBlock);
                    }
                    else
                    {
                        // value distance of the block body is included:
                        distance = GetDistance(leftBlock.Parent, rightBlock.Parent);
                    }

                    return true;

                case SyntaxKind.Block:
                case SyntaxKind.LabeledStatement:
                    distance = ComputeWeightedBlockDistance(leftBlock, rightBlock);
                    return true;

                case SyntaxKind.UnsafeStatement:
                case SyntaxKind.CheckedStatement:
                case SyntaxKind.UncheckedStatement:
                case SyntaxKind.ElseClause:
                case SyntaxKind.FinallyClause:
                case SyntaxKind.TryStatement:
                    distance = 0.2 * ComputeValueDistance(leftBlock, rightBlock);
                    return true;

                default:
                    throw new Exception($"Unexpected value {leftBlock.Parent.Kind()}");
            }
        }

        private double ComputeWeightedDistance(SingleVariableDesignationSyntax leftNode, SingleVariableDesignationSyntax rightNode)
        {
            double distance = ComputeDistance(leftNode, rightNode);
            double parentDistance;

            if (leftNode.Parent != null &&
                rightNode.Parent != null &&
                GetLabel(leftNode.Parent) == GetLabel(rightNode.Parent))
            {
                parentDistance = ComputeDistance(leftNode.Parent, rightNode.Parent);
            }
            else
            {
                parentDistance = 1;
            }

            return 0.5 * parentDistance + 0.5 * distance;
        }

        private static double ComputeWeightedBlockDistance(BlockSyntax leftBlock, BlockSyntax rightBlock)
        {
            if (TryComputeLocalsDistance(leftBlock, rightBlock, out var distance))
            {
                return distance;
            }

            return ComputeValueDistance(leftBlock, rightBlock);
        }

        private static double ComputeWeightedDistance(CatchClauseSyntax left, CatchClauseSyntax right)
        {
            double blockDistance = ComputeDistance(left.Block, right.Block);
            double distance = CombineOptional(blockDistance, left.Declaration, right.Declaration, left.Filter, right.Filter);
            return AdjustForLocalsInBlock(distance, left.Block, right.Block, localsWeight: 0.3);
        }

        private static double ComputeWeightedDistance(
            CommonForEachStatementSyntax leftCommonForEach,
            CommonForEachStatementSyntax rightCommonForEach)
        {
            double statementDistance = ComputeDistance(leftCommonForEach.Statement, rightCommonForEach.Statement);
            double expressionDistance = ComputeDistance(leftCommonForEach.Expression, rightCommonForEach.Expression);

            List<SyntaxToken> leftLocals = null;
            List<SyntaxToken> rightLocals = null;
            GetLocalNames(leftCommonForEach, ref leftLocals);
            GetLocalNames(rightCommonForEach, ref rightLocals);

            double localNamesDistance = ComputeDistance(leftLocals, rightLocals);

            double distance = localNamesDistance * 0.6 + expressionDistance * 0.2 + statementDistance * 0.2;
            return AdjustForLocalsInBlock(distance, leftCommonForEach.Statement, rightCommonForEach.Statement, localsWeight: 0.6);
        }

        private static double ComputeWeightedDistance(ForStatementSyntax left, ForStatementSyntax right)
        {
            double statementDistance = ComputeDistance(left.Statement, right.Statement);
            double conditionDistance = ComputeDistance(left.Condition, right.Condition);

            double incDistance = ComputeDistance(
                GetDescendantTokensIgnoringSeparators(left.Incrementors), GetDescendantTokensIgnoringSeparators(right.Incrementors));

            double distance = conditionDistance * 0.3 + incDistance * 0.3 + statementDistance * 0.4;
            if (TryComputeLocalsDistance(left.Declaration, right.Declaration, out var localsDistance))
            {
                distance = distance * 0.4 + localsDistance * 0.6;
            }

            return distance;
        }

        private static double ComputeWeightedDistance(
            VariableDeclarationSyntax leftVariables,
            StatementSyntax leftStatement,
            VariableDeclarationSyntax rightVariables,
            StatementSyntax rightStatement)
        {
            double distance = ComputeDistance(leftStatement, rightStatement);
            // Put maximum weight behind the variables declared in the header of the statement.
            if (TryComputeLocalsDistance(leftVariables, rightVariables, out var localsDistance))
            {
                distance = distance * 0.4 + localsDistance * 0.6;
            }

            // If the statement is a block that declares local variables, 
            // weight them more than the rest of the statement.
            return AdjustForLocalsInBlock(distance, leftStatement, rightStatement, localsWeight: 0.2);
        }

        private static double ComputeWeightedDistance(
            SyntaxNode leftHeaderOpt,
            StatementSyntax leftStatement,
            SyntaxNode rightHeaderOpt,
            StatementSyntax rightStatement)
        {
            Debug.Assert(leftStatement != null);
            Debug.Assert(rightStatement != null);

            double headerDistance = ComputeDistance(leftHeaderOpt, rightHeaderOpt);
            double statementDistance = ComputeDistance(leftStatement, rightStatement);
            double distance = headerDistance * 0.6 + statementDistance * 0.4;

            return AdjustForLocalsInBlock(distance, leftStatement, rightStatement, localsWeight: 0.5);
        }

        private static double AdjustForLocalsInBlock(
            double distance,
            StatementSyntax leftStatement,
            StatementSyntax rightStatement,
            double localsWeight)
        {
            // If the statement is a block that declares local variables, 
            // weight them more than the rest of the statement.
            if (leftStatement.Kind() == SyntaxKind.Block && rightStatement.Kind() == SyntaxKind.Block)
            {
                if (TryComputeLocalsDistance((BlockSyntax)leftStatement, (BlockSyntax)rightStatement, out var localsDistance))
                {
                    return localsDistance * localsWeight + distance * (1 - localsWeight);
                }
            }

            return distance;
        }

        private static bool TryComputeLocalsDistance(VariableDeclarationSyntax leftOpt, VariableDeclarationSyntax rightOpt, out double distance)
        {
            List<SyntaxToken> leftLocals = null;
            List<SyntaxToken> rightLocals = null;

            if (leftOpt != null)
            {
                GetLocalNames(leftOpt, ref leftLocals);
            }

            if (rightOpt != null)
            {
                GetLocalNames(rightOpt, ref rightLocals);
            }

            if (leftLocals == null || rightLocals == null)
            {
                distance = 0;
                return false;
            }

            distance = ComputeDistance(leftLocals, rightLocals);
            return true;
        }

        private static bool TryComputeLocalsDistance(BlockSyntax left, BlockSyntax right, out double distance)
        {
            List<SyntaxToken> leftLocals = null;
            List<SyntaxToken> rightLocals = null;

            GetLocalNames(left, ref leftLocals);
            GetLocalNames(right, ref rightLocals);

            if (leftLocals == null || rightLocals == null)
            {
                distance = 0;
                return false;
            }

            distance = ComputeDistance(leftLocals, rightLocals);
            return true;
        }

        // doesn't include variables declared in declaration expressions
        private static void GetLocalNames(BlockSyntax block, ref List<SyntaxToken> result)
        {
            foreach (var child in block.ChildNodes())
            {
                if (child.IsKind(SyntaxKind.LocalDeclarationStatement))
                {
                    GetLocalNames(((LocalDeclarationStatementSyntax)child).Declaration, ref result);
                }
            }
        }

        // doesn't include variables declared in declaration expressions
        private static void GetLocalNames(VariableDeclarationSyntax localDeclaration, ref List<SyntaxToken> result)
        {
            foreach (var local in localDeclaration.Variables)
            {
                GetLocalNames(local.Identifier, ref result);
            }
        }

        internal static void GetLocalNames(CommonForEachStatementSyntax commonForEach, ref List<SyntaxToken> result)
        {
            switch (commonForEach.Kind())
            {
                case SyntaxKind.ForEachStatement:
                    GetLocalNames(((ForEachStatementSyntax)commonForEach).Identifier, ref result);
                    return;

                case SyntaxKind.ForEachVariableStatement:
                    var forEachVariable = (ForEachVariableStatementSyntax)commonForEach;
                    GetLocalNames(forEachVariable.Variable, ref result);
                    return;

                default: throw new Exception($"Unexpected value {commonForEach.Kind()}");
            }
        }

        private static void GetLocalNames(ExpressionSyntax expression, ref List<SyntaxToken> result)
        {
            switch (expression.Kind())
            {
                case SyntaxKind.DeclarationExpression:
                    var declarationExpression = (DeclarationExpressionSyntax)expression;
                    var localDeclaration = declarationExpression.Designation;
                    GetLocalNames(localDeclaration, ref result);
                    return;

                case SyntaxKind.TupleExpression:
                    var tupleExpression = (TupleExpressionSyntax)expression;
                    foreach (var argument in tupleExpression.Arguments)
                    {
                        GetLocalNames(argument.Expression, ref result);
                    }
                    return;

                default:
                    // Do nothing for node that cannot have variable declarations inside.
                    return;
            }
        }

        private static void GetLocalNames(VariableDesignationSyntax designation, ref List<SyntaxToken> result)
        {
            switch (designation.Kind())
            {
                case SyntaxKind.SingleVariableDesignation:
                    GetLocalNames(((SingleVariableDesignationSyntax)designation).Identifier, ref result);
                    return;

                case SyntaxKind.ParenthesizedVariableDesignation:
                    var parenthesizedVariableDesignation = (ParenthesizedVariableDesignationSyntax)designation;
                    foreach (var variableDesignation in parenthesizedVariableDesignation.Variables)
                    {
                        GetLocalNames(variableDesignation, ref result);
                    }
                    return;

                default: throw new Exception($"Unexpected value {designation.Kind()}");
            }
        }

        private static void GetLocalNames(SyntaxToken syntaxToken, ref List<SyntaxToken> result)
        {
            if (result == null)
            {
                result = new List<SyntaxToken>();
            }

            result.Add(syntaxToken);
        }

        private static double CombineOptional(
            double distance0,
            SyntaxNode leftOpt1,
            SyntaxNode rightOpt1,
            SyntaxNode leftOpt2,
            SyntaxNode rightOpt2,
            double weight0 = 0.8,
            double weight1 = 0.5)
        {
            bool one = leftOpt1 != null || rightOpt1 != null;
            bool two = leftOpt2 != null || rightOpt2 != null;

            if (!one && !two)
            {
                return distance0;
            }

            double distance1 = ComputeDistance(leftOpt1, rightOpt1);
            double distance2 = ComputeDistance(leftOpt2, rightOpt2);

            double d;
            if (one && two)
            {
                d = distance1 * weight1 + distance2 * (1 - weight1);
            }
            else if (one)
            {
                d = distance1;
            }
            else
            {
                d = distance2;
            }

            return distance0 * weight0 + d * (1 - weight0);
        }

        #endregion

        #region LambdaUtils
        /// <remarks>
        /// In C# lambda bodies are expressions or block statements. In both cases it's a single node.
        /// In VB a lambda body might be a sequence of nodes (statements). 
        /// We define this function to minimize differences between C# and VB implementation.
        /// </remarks>
        public static bool IsLambdaBodyStatementOrExpression(SyntaxNode node)
        {
            return IsLambdaBody(node);
        }

        /// <summary>
        /// Returns true if the specified <paramref name="node"/> represents a body of a lambda.
        /// </summary>
        public static bool IsLambdaBody(SyntaxNode node, bool allowReducedLambdas = false)
        {
            var parent = node?.Parent;
            if (parent == null)
            {
                return false;
            }

            switch (parent.Kind())
            {
                case SyntaxKind.ParenthesizedLambdaExpression:
                case SyntaxKind.SimpleLambdaExpression:
                case SyntaxKind.AnonymousMethodExpression:
                    var anonymousFunction = (AnonymousFunctionExpressionSyntax)parent;
                    return anonymousFunction.Body == node;

                case SyntaxKind.LocalFunctionStatement:
                    var localFunction = (LocalFunctionStatementSyntax)parent;
                    return localFunction.Body == node;

                case SyntaxKind.ArrowExpressionClause:
                    var arrowExpressionClause = (ArrowExpressionClauseSyntax)parent;
                    return arrowExpressionClause.Expression == node && arrowExpressionClause.Parent is LocalFunctionStatementSyntax;

                case SyntaxKind.FromClause:
                    var fromClause = (FromClauseSyntax)parent;
                    return fromClause.Expression == node && fromClause.Parent is QueryBodySyntax;

                case SyntaxKind.JoinClause:
                    var joinClause = (JoinClauseSyntax)parent;
                    return joinClause.LeftExpression == node || joinClause.RightExpression == node;

                case SyntaxKind.LetClause:
                    var letClause = (LetClauseSyntax)parent;
                    return letClause.Expression == node;

                case SyntaxKind.WhereClause:
                    var whereClause = (WhereClauseSyntax)parent;
                    return whereClause.Condition == node;

                case SyntaxKind.AscendingOrdering:
                case SyntaxKind.DescendingOrdering:
                    var ordering = (OrderingSyntax)parent;
                    return ordering.Expression == node;

                case SyntaxKind.SelectClause:
                    var selectClause = (SelectClauseSyntax)parent;
                    return selectClause.Expression == node && (allowReducedLambdas || !IsReducedSelectOrGroupByClause(selectClause, selectClause.Expression));

                case SyntaxKind.GroupClause:
                    var groupClause = (GroupClauseSyntax)parent;
                    return (groupClause.GroupExpression == node && (allowReducedLambdas || !IsReducedSelectOrGroupByClause(groupClause, groupClause.GroupExpression))) ||
                           groupClause.ByExpression == node;
            }

            return false;
        }

        /// <summary>
        /// When queries are translated into expressions select and group-by expressions such that
        /// 1) select/group-by expression is the same identifier as the "source" identifier and
        /// 2) at least one Where or OrderBy clause but no other clause is present in the contained query body or
        ///    the expression in question is a group-by expression and the body has no clause
        /// 
        /// do not translate into lambdas.
        /// By "source" identifier we mean the identifier specified in the from clause that initiates the query or the query continuation that includes the body.
        /// 
        /// The above condition can be derived from the language specification (chapter 7.16.2) as follows:
        /// - In order for 7.16.2.5 "Select clauses" to be applicable the following conditions must hold:
        ///   - There has to be at least one clause in the body, otherwise the query is reduced into a final form by 7.16.2.3 "Degenerate query expressions".
        ///   - Only where and order-by clauses may be present in the query body, otherwise a transformation in 7.16.2.4 "From, let, where, join and orderby clauses"
        ///     produces pattern that doesn't match the requirements of 7.16.2.5.
        ///   
        /// - In order for 7.16.2.6 "Groupby clauses" to be applicable the following conditions must hold:
        ///   - Only where and order-by clauses may be present in the query body, otherwise a transformation in 7.16.2.4 "From, let, where, join and orderby clauses"
        ///     produces pattern that doesn't match the requirements of 7.16.2.5.
        /// </summary>
        private static bool IsReducedSelectOrGroupByClause(SelectOrGroupClauseSyntax selectOrGroupClause, ExpressionSyntax selectOrGroupExpression)
        {
            if (!selectOrGroupExpression.IsKind(SyntaxKind.IdentifierName))
            {
                return false;
            }

            var selectorIdentifier = ((IdentifierNameSyntax)selectOrGroupExpression).Identifier;

            SyntaxToken sourceIdentifier;
            QueryBodySyntax containingBody;

            var containingQueryOrContinuation = selectOrGroupClause.Parent.Parent;
            if (containingQueryOrContinuation.IsKind(SyntaxKind.QueryExpression))
            {
                var containingQuery = (QueryExpressionSyntax)containingQueryOrContinuation;
                containingBody = containingQuery.Body;
                sourceIdentifier = containingQuery.FromClause.Identifier;
            }
            else
            {
                var containingContinuation = (QueryContinuationSyntax)containingQueryOrContinuation;
                sourceIdentifier = containingContinuation.Identifier;
                containingBody = containingContinuation.Body;
            }

            if (!SyntaxFactory.AreEquivalent(sourceIdentifier, selectorIdentifier))
            {
                return false;
            }

            if (selectOrGroupClause.IsKind(SyntaxKind.SelectClause) && containingBody.Clauses.Count == 0)
            {
                return false;
            }

            foreach (var clause in containingBody.Clauses)
            {
                if (!clause.IsKind(SyntaxKind.WhereClause) && !clause.IsKind(SyntaxKind.OrderByClause))
                {
                    return false;
                }
            }

            return true;
        }
        #endregion
    }

    class BlockTreeComparer : AbstractCSharpTreeMatch
    {
        #region Tree Traversal

        protected override bool TryGetParent(SyntaxNode node, out SyntaxNode parent)
        {
            var parentNode = node.Parent;
            parent = parentNode;
            return parentNode != null;
        }

        protected override IEnumerable<SyntaxNode> GetChildren(SyntaxNode node)
        {
            Debug.Assert(GetLabel(node) != IgnoredNode);
            return HasChildren(node) ? EnumerateChildren(node) : null;
        }

        private IEnumerable<SyntaxNode> EnumerateChildren(SyntaxNode node)
        {
            foreach (var child in node.ChildNodesAndTokens())
            {
                var childNode = child.AsNode();
                if (childNode != null && GetLabel(childNode) != IgnoredNode)
                {
                    yield return childNode;
                }
            }
        }

        protected override IEnumerable<SyntaxNode> GetDescendants(SyntaxNode node)
        {
            foreach (var descendant in node.DescendantNodesAndTokens(
                descendIntoChildren: HasChildren,
                descendIntoTrivia: false))
            {
                var descendantNode = descendant.AsNode();
                if (descendantNode != null && GetLabel(descendantNode) != IgnoredNode)
                {
                    yield return descendantNode;
                }
            }
        }

        private static bool HasChildren(SyntaxNode node)
        {
            // Leaves are labeled statements that don't have a labeled child.
            // We also return true for non-labeled statements.
            Label label = Classify(node.Kind(), out var isLeaf);

            // ignored should always be reported as leaves
            Debug.Assert(label != Label.Ignored || isLeaf);

            return !isLeaf;
        }

        #endregion

        #region Labels

        // Assumptions:
        // - Each listed label corresponds to one or more syntax kinds.
        // - Nodes with same labels might produce Update edits, nodes with different labels don't. 
        // - If IsTiedToParent(label) is true for a label then all its possible parent labels must precede the label.
        //   (i.e. both MethodDeclaration and TypeDeclaration must precede TypeParameter label).
        // - All descendants of a node whose kind is listed here will be ignored regardless of their labels
        internal enum Label
        {
            CompilationUnit,

            NamespaceDeclaration,
            ExternAliasDirective,              // tied to parent 
            UsingDirective,                    // tied to parent

            TypeDeclaration,
            EnumDeclaration,
            DelegateDeclaration,

            FieldDeclaration,                  // tied to parent
            FieldVariableDeclaration,          // tied to parent
            FieldVariableDeclarator,           // tied to parent

            MethodDeclaration,                 // tied to parent
            OperatorDeclaration,               // tied to parent
            ConversionOperatorDeclaration,     // tied to parent
            ConstructorDeclaration,            // tied to parent
            DestructorDeclaration,             // tied to parent
            PropertyDeclaration,               // tied to parent
            IndexerDeclaration,                // tied to parent
            EventDeclaration,                  // tied to parent
            EnumMemberDeclaration,             // tied to parent

            AccessorList,                      // tied to parent
            AccessorDeclaration,               // tied to parent
            TypeParameterList,                 // tied to parent
            TypeParameterConstraintClause,     // tied to parent
            TypeParameter,                     // tied to parent
            ParameterList,                     // tied to parent
            BracketedParameterList,            // tied to parent
            Parameter,                         // tied to parent
            AttributeList,                     // tied to parent
            Attribute,                         // tied to parent

            // helpers:
            Count,
            Ignored = IgnoredNode
        }

        /// <summary>
        /// Return 1 if it is desirable to report two edits (delete and insert) rather than a move edit
        /// when the node changes its parent.
        /// </summary>
        private static int TiedToAncestor(Label label)
        {
            switch (label)
            {
                case Label.ExternAliasDirective:
                case Label.UsingDirective:
                case Label.FieldDeclaration:
                case Label.FieldVariableDeclaration:
                case Label.FieldVariableDeclarator:
                case Label.MethodDeclaration:
                case Label.OperatorDeclaration:
                case Label.ConversionOperatorDeclaration:
                case Label.ConstructorDeclaration:
                case Label.DestructorDeclaration:
                case Label.PropertyDeclaration:
                case Label.IndexerDeclaration:
                case Label.EventDeclaration:
                case Label.EnumMemberDeclaration:
                case Label.AccessorDeclaration:
                case Label.AccessorList:
                case Label.TypeParameterList:
                case Label.TypeParameter:
                case Label.TypeParameterConstraintClause:
                case Label.ParameterList:
                case Label.BracketedParameterList:
                case Label.Parameter:
                case Label.AttributeList:
                case Label.Attribute:
                    return 1;

                default:
                    return 0;
            }
        }

        // internal for testing
        internal static Label Classify(SyntaxKind kind, out bool isLeaf)
        {
            switch (kind)
            {
                case SyntaxKind.CompilationUnit:
                    isLeaf = false;
                    return Label.CompilationUnit;

                case SyntaxKind.GlobalStatement:
                    // TODO:
                    isLeaf = true;
                    return Label.Ignored;

                case SyntaxKind.ExternAliasDirective:
                    isLeaf = true;
                    return Label.ExternAliasDirective;

                case SyntaxKind.UsingDirective:
                    isLeaf = true;
                    return Label.UsingDirective;

                case SyntaxKind.NamespaceDeclaration:
                    isLeaf = false;
                    return Label.NamespaceDeclaration;

                case SyntaxKind.ClassDeclaration:
                case SyntaxKind.StructDeclaration:
                case SyntaxKind.InterfaceDeclaration:
                    isLeaf = false;
                    return Label.TypeDeclaration;

                case SyntaxKind.EnumDeclaration:
                    isLeaf = false;
                    return Label.EnumDeclaration;

                case SyntaxKind.DelegateDeclaration:
                    isLeaf = false;
                    return Label.DelegateDeclaration;

                case SyntaxKind.FieldDeclaration:
                case SyntaxKind.EventFieldDeclaration:
                    isLeaf = false;
                    return Label.FieldDeclaration;

                case SyntaxKind.VariableDeclaration:
                    isLeaf = false;
                    return Label.FieldVariableDeclaration;

                case SyntaxKind.VariableDeclarator:
                    isLeaf = true;
                    return Label.FieldVariableDeclarator;

                case SyntaxKind.MethodDeclaration:
                    isLeaf = false;
                    return Label.MethodDeclaration;

                case SyntaxKind.ConversionOperatorDeclaration:
                    isLeaf = false;
                    return Label.ConversionOperatorDeclaration;

                case SyntaxKind.OperatorDeclaration:
                    isLeaf = false;
                    return Label.OperatorDeclaration;

                case SyntaxKind.ConstructorDeclaration:
                    isLeaf = false;
                    return Label.ConstructorDeclaration;

                case SyntaxKind.DestructorDeclaration:
                    isLeaf = true;
                    return Label.DestructorDeclaration;

                case SyntaxKind.PropertyDeclaration:
                    isLeaf = false;
                    return Label.PropertyDeclaration;

                case SyntaxKind.IndexerDeclaration:
                    isLeaf = false;
                    return Label.IndexerDeclaration;

                case SyntaxKind.EventDeclaration:
                    isLeaf = false;
                    return Label.EventDeclaration;

                case SyntaxKind.EnumMemberDeclaration:
                    isLeaf = false; // attribute may be applied
                    return Label.EnumMemberDeclaration;

                case SyntaxKind.AccessorList:
                    isLeaf = false;
                    return Label.AccessorList;

                case SyntaxKind.GetAccessorDeclaration:
                case SyntaxKind.SetAccessorDeclaration:
                case SyntaxKind.AddAccessorDeclaration:
                case SyntaxKind.RemoveAccessorDeclaration:
                    isLeaf = true;
                    return Label.AccessorDeclaration;

                case SyntaxKind.TypeParameterList:
                    isLeaf = false;
                    return Label.TypeParameterList;

                case SyntaxKind.TypeParameterConstraintClause:
                    isLeaf = false;
                    return Label.TypeParameterConstraintClause;

                case SyntaxKind.TypeParameter:
                    isLeaf = false; // children: attributes
                    return Label.TypeParameter;

                case SyntaxKind.ParameterList:
                    isLeaf = false;
                    return Label.ParameterList;

                case SyntaxKind.BracketedParameterList:
                    isLeaf = false;
                    return Label.BracketedParameterList;

                case SyntaxKind.Parameter:
                    // We ignore anonymous methods and lambdas, 
                    // we only care about parameters of member declarations.
                    isLeaf = false; // children: attributes
                    return Label.Parameter;

                case SyntaxKind.AttributeList:
                    isLeaf = false;
                    return Label.AttributeList;

                case SyntaxKind.Attribute:
                    isLeaf = true;
                    return Label.Attribute;

                default:
                    isLeaf = true;
                    return Label.Ignored;
            }
        }

        protected override int GetLabel(SyntaxNode node)
        {
            return (int)GetLabel(node.Kind());
        }

        internal static Label GetLabel(SyntaxKind kind)
        {
            return Classify(kind, out var isLeaf);
        }

        // internal for testing
        internal static bool HasLabel(SyntaxKind kind)
        {
            return Classify(kind, out var isLeaf) != Label.Ignored;
        }

        protected override int LabelCount
        {
            get { return (int)Label.Count; }
        }

        protected override int TiedToAncestor(int label)
        {
            return TiedToAncestor((Label)label);
        }

        #endregion

        #region Comparisons

        public override bool ValuesEqual(SyntaxNode left, SyntaxNode right)
        {
            Func<SyntaxKind, bool> ignoreChildFunction;
            switch (left.Kind())
            {
                // all syntax kinds with a method body child:
                case SyntaxKind.MethodDeclaration:
                case SyntaxKind.ConversionOperatorDeclaration:
                case SyntaxKind.OperatorDeclaration:
                case SyntaxKind.ConstructorDeclaration:
                case SyntaxKind.DestructorDeclaration:
                case SyntaxKind.GetAccessorDeclaration:
                case SyntaxKind.SetAccessorDeclaration:
                case SyntaxKind.AddAccessorDeclaration:
                case SyntaxKind.RemoveAccessorDeclaration:
                    // When comparing method bodies we need to NOT ignore VariableDeclaration and VariableDeclarator children,
                    // but when comparing field definitions we should ignore VariableDeclarations children.

                    var leftBody = GetBody(left);
                    var rightBody = GetBody(right);

                    if (!SyntaxFactory.AreEquivalent(leftBody, rightBody, null))
                    {
                        return false;
                    }

                    ignoreChildFunction = childKind => childKind == SyntaxKind.Block || childKind == SyntaxKind.ArrowExpressionClause || HasLabel(childKind);
                    break;

                default:
                    if (HasChildren(left))
                    {
                        ignoreChildFunction = childKind => HasLabel(childKind);
                    }
                    else
                    {
                        ignoreChildFunction = null;
                    }

                    break;
            }

            return SyntaxFactory.AreEquivalent(left, right, ignoreChildFunction);
        }

        private static SyntaxNode GetBody(SyntaxNode node)
        {
            switch (node)
            {
                case BaseMethodDeclarationSyntax baseMethodDeclarationSyntax: return baseMethodDeclarationSyntax.Body ?? (SyntaxNode)baseMethodDeclarationSyntax.ExpressionBody?.Expression;
                case AccessorDeclarationSyntax accessorDeclarationSyntax: return accessorDeclarationSyntax.Body ?? (SyntaxNode)accessorDeclarationSyntax.ExpressionBody?.Expression;
                default: throw new Exception($"Unexpected value {node}");
            }
        }

        protected override bool TryComputeWeightedDistance(SyntaxNode leftNode, SyntaxNode rightNode, out double distance)
        {
            SyntaxNodeOrToken? leftName = TryGetName(leftNode);
            SyntaxNodeOrToken? rightName = TryGetName(rightNode);
            Debug.Assert(rightName.HasValue == leftName.HasValue);

            if (leftName.HasValue)
            {
                distance = ComputeDistance(leftName.Value, rightName.Value);
                return true;
            }
            else
            {
                distance = 0;
                return false;
            }
        }

        private static SyntaxNodeOrToken? TryGetName(SyntaxNode node)
        {
            switch (node.Kind())
            {
                case SyntaxKind.ExternAliasDirective:
                    return ((ExternAliasDirectiveSyntax)node).Identifier;

                case SyntaxKind.UsingDirective:
                    return ((UsingDirectiveSyntax)node).Name;

                case SyntaxKind.NamespaceDeclaration:
                    return ((NamespaceDeclarationSyntax)node).Name;

                case SyntaxKind.ClassDeclaration:
                case SyntaxKind.StructDeclaration:
                case SyntaxKind.InterfaceDeclaration:
                    return ((TypeDeclarationSyntax)node).Identifier;

                case SyntaxKind.EnumDeclaration:
                    return ((EnumDeclarationSyntax)node).Identifier;

                case SyntaxKind.DelegateDeclaration:
                    return ((DelegateDeclarationSyntax)node).Identifier;

                case SyntaxKind.FieldDeclaration:
                case SyntaxKind.EventFieldDeclaration:
                case SyntaxKind.VariableDeclaration:
                    return null;

                case SyntaxKind.VariableDeclarator:
                    return ((VariableDeclaratorSyntax)node).Identifier;

                case SyntaxKind.MethodDeclaration:
                    return ((MethodDeclarationSyntax)node).Identifier;

                case SyntaxKind.ConversionOperatorDeclaration:
                    return ((ConversionOperatorDeclarationSyntax)node).Type;

                case SyntaxKind.OperatorDeclaration:
                    return ((OperatorDeclarationSyntax)node).OperatorToken;

                case SyntaxKind.ConstructorDeclaration:
                    return ((ConstructorDeclarationSyntax)node).Identifier;

                case SyntaxKind.DestructorDeclaration:
                    return ((DestructorDeclarationSyntax)node).Identifier;

                case SyntaxKind.PropertyDeclaration:
                    return ((PropertyDeclarationSyntax)node).Identifier;

                case SyntaxKind.IndexerDeclaration:
                    return null;

                case SyntaxKind.EventDeclaration:
                    return ((EventDeclarationSyntax)node).Identifier;

                case SyntaxKind.EnumMemberDeclaration:
                    return ((EnumMemberDeclarationSyntax)node).Identifier;

                case SyntaxKind.GetAccessorDeclaration:
                case SyntaxKind.SetAccessorDeclaration:
                    return null;

                case SyntaxKind.TypeParameterConstraintClause:
                    return ((TypeParameterConstraintClauseSyntax)node).Name.Identifier;

                case SyntaxKind.TypeParameter:
                    return ((TypeParameterSyntax)node).Identifier;

                case SyntaxKind.TypeParameterList:
                case SyntaxKind.ParameterList:
                case SyntaxKind.BracketedParameterList:
                    return null;

                case SyntaxKind.Parameter:
                    return ((ParameterSyntax)node).Identifier;

                case SyntaxKind.AttributeList:
                    return ((AttributeListSyntax)node).Target;

                case SyntaxKind.Attribute:
                    return ((AttributeSyntax)node).Name;

                default:
                    return null;
            }
        }

        #endregion
    }
}
