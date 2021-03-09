// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Dynamic;
using System.IO;
using System.Text;
using System.Linq;
using System.Runtime.InteropServices;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Microsoft.CodeAnalysis.Text;
using Microsoft.VisualBasic.CompilerServices;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace IdiomaticChangeMining
{
    class FixerDataPipeline
    {

        public static IEnumerable<SyntaxNode> GetNodesByLineSpan(SyntaxNode astNode, SourceText sourceText, int startLine, int endLine)
        {
            //var nodeVisitor = new LineContiguousNodeFinder(startLine, endLine);
            //nodeVisitor.Visit(astNode);

            //return nodeVisitor.Nodes;

            //for (var lineId = startLine;
            //    lineId <= endLine;
            //    lineId++)
            //{
            //    var lineSource = sourceText.Lines[lineId];
            //    if (string.IsNullOrWhiteSpace(lineSource.ToString()))
            //        continue;

            //    var lineSpan = lineSource.Span;
            //    if (lineSpan.IsEmpty)
            //        continue;

            //    var node = astNode.FindNode(lineSpan);
            //    yield return node;
            //}

            TextSpan wholeSpan = TextSpan.FromBounds(sourceText.Lines[startLine].Start, sourceText.Lines[endLine].End);
            var nodeContainingSpan = astNode.FindNode(wholeSpan);
            return nodeContainingSpan
                .DescendantNodesAndSelf(descendIntoChildren: n => n.Span.IntersectsWith(wholeSpan) && !wholeSpan.Contains(n.Span))
                .Where(n => n.Span.IntersectsWith(wholeSpan));
        }

        public IEnumerable<ChangeSample> ProcessRevision(string previousFile, string updatedFile)
        {
            var prevAst = CSharpSyntaxTree.ParseText(previousFile);
            var updatedAst = CSharpSyntaxTree.ParseText(updatedFile);

            var changesWithContext = DiffInfo.GetChangesWithContext(prevAst, updatedAst);
            return changesWithContext;
        }

        public static IEnumerable<ChangeSample> GetChangesBetweenAsts(SyntaxTree previousFileAst, SyntaxTree updatedFileAst)
        {
            var changesWithContext = DiffInfo.GetChangesWithContext(previousFileAst, updatedFileAst);
            return changesWithContext;
        }

        public static object ProcessSingleRevision(string fixerId, string fileId, string previousFile, string updatedFile, JsonSyntaxTreeHelper jsonSyntaxTreeHelper)
        {
            // File.WriteAllText("a.original.cs", previousFile);
            // File.WriteAllText("b.original.cs", updatedFile);

            var previousFileAst = CSharpSyntaxTree.ParseText(previousFile);
            var updatedFileAst = CSharpSyntaxTree.ParseText(updatedFile);

            (SyntaxNode canonicalPrevFileAst, Dictionary<string, string> prevFileVariableNameMap) = Canonicalization.CanonicalizeSyntaxNode(previousFileAst.GetRoot(), extractAllVariablesFirst:true);
            (SyntaxNode canonicalUpdatedFileAst, Dictionary<string, string> updatedFileVariableNameMap) = Canonicalization.CanonicalizeSyntaxNode(updatedFileAst.GetRoot(), prevFileVariableNameMap);

            var prevCodeFile = canonicalPrevFileAst.GetText();
            var updatedCodeFile = canonicalUpdatedFileAst.GetText();

            var prevFileTokens = canonicalPrevFileAst.DescendantTokens().ToList();
            var updatedFileTokens = canonicalUpdatedFileAst.DescendantTokens().ToList();

            var changesInRevision = GetChangesBetweenAsts(canonicalPrevFileAst.SyntaxTree, canonicalUpdatedFileAst.SyntaxTree);

            // File.WriteAllText("a.canonical.cs", canonicalPrevFileAst.GetText().ToString());
            // File.WriteAllText("b.canonical.cs", canonicalUpdatedFileAst.GetText().ToString());

            var prevTokenIndex = new TokenIndex(prevFileTokens);
            var updatedTokenIndex = new TokenIndex(updatedFileTokens);

            if (changesInRevision.Count() == 0)
                return null;

            var change = changesInRevision.First();

            var prevCodeChunkLineSpan = canonicalPrevFileAst.SyntaxTree.GetLineSpan(change.BeforeSpan.ChangeSpan);
            var updatedCodeChunkLineSpan = canonicalUpdatedFileAst.SyntaxTree.GetLineSpan(change.AfterSpan.ChangeSpan);

            var prevCodeChunkLineSpanStart = prevCodeFile.Lines[prevCodeChunkLineSpan.StartLinePosition.Line].Span.Start;
            var prevCodeChunkSpanEnd = prevCodeFile.Lines[prevCodeChunkLineSpan.EndLinePosition.Line].Span.End;

            var updatedCodeChunkLineSpanStart = updatedCodeFile.Lines[updatedCodeChunkLineSpan.StartLinePosition.Line].Span.Start;
            var updatedCodeChunkSpanEnd = updatedCodeFile.Lines[updatedCodeChunkLineSpan.EndLinePosition.Line].Span.End;

            // TODO: remove trivial change

            // only consider SyntaxKind in allowedSytaxKinds
            var prevCodeChunkNodes = GetNodesByLineSpan(canonicalPrevFileAst, prevCodeFile, 
                prevCodeChunkLineSpan.StartLinePosition.Line, prevCodeChunkLineSpan.EndLinePosition.Line);
            if (prevCodeChunkNodes.Any(node => !allowedSytaxKinds.Contains(node.Kind())))
                return null;

            var updatedCodeChunkNodes = GetNodesByLineSpan(canonicalUpdatedFileAst, updatedCodeFile, 
                updatedCodeChunkLineSpan.StartLinePosition.Line, updatedCodeChunkLineSpan.EndLinePosition.Line);
            if (updatedCodeChunkNodes.Any(node => !allowedSytaxKinds.Contains(node.Kind())))
                return null;

            var previousCodeChunkTokens = prevTokenIndex
                .GetTokensInSpan(prevCodeChunkLineSpanStart, prevCodeChunkSpanEnd)
                .Select(token => token.ValueText)
                .Where(token => !string.IsNullOrWhiteSpace(token) && !string.IsNullOrEmpty(token))
                .ToArray();

            var updatedsCodeChunkTokens = updatedTokenIndex
                .GetTokensInSpan(updatedCodeChunkLineSpanStart, updatedCodeChunkSpanEnd)
                .Select(token => token.ValueText)
                .Where(token => !string.IsNullOrWhiteSpace(token) && !string.IsNullOrEmpty(token))
                .ToArray();

            if (previousCodeChunkTokens.Length > 0 && updatedsCodeChunkTokens.Length > 0 &&
                IsValidCodeChunkTokens(previousCodeChunkTokens) && IsValidCodeChunkTokens(updatedsCodeChunkTokens) &&
                !previousCodeChunkTokens.SequenceEqual(updatedsCodeChunkTokens))
            {
                var prevCodeChunkBlockStmt = SyntaxFactory.Block(prevCodeChunkNodes.Select(node => (StatementSyntax)node));
                var updatedCodeChunkBlockStmt = SyntaxFactory.Block(updatedCodeChunkNodes.Select(node => (StatementSyntax)node));

                IDictionary<string, string> zeroIndexedVariableNameMap;
                    (prevCodeChunkBlockStmt, updatedCodeChunkBlockStmt, zeroIndexedVariableNameMap) =
                    zeroIndexVariableNames(prevCodeChunkBlockStmt, updatedCodeChunkBlockStmt);

                var prevCodeChunkBlockStmtTokens = prevCodeChunkBlockStmt.DescendantTokens().Skip(1).SkipLast(1).ToArray();
                var prevCodeChunkBlackStmtTokensIndex = new TokenIndex(prevCodeChunkBlockStmtTokens).InitInvertedIndex();

                var updatedCodeChunkBlockStmtTokens = updatedCodeChunkBlockStmt.DescendantTokens().Skip(1).SkipLast(1).ToArray();
                var updatedCodeChunkBlockStmtTokensIndex = new TokenIndex(updatedCodeChunkBlockStmtTokens).InitInvertedIndex();

                var prevCodeBlockJObject = jsonSyntaxTreeHelper.GetJObjectForSyntaxNode(prevCodeChunkBlockStmt, prevCodeChunkBlackStmtTokensIndex);
                var updatedCodeBlockJObject = jsonSyntaxTreeHelper.GetJObjectForSyntaxNode(updatedCodeChunkBlockStmt, updatedCodeChunkBlockStmtTokensIndex);

                var precedingContextTokens = prevTokenIndex.GetTokensInSpan(change.BeforeSpan.SpanOfPrecedingContext);
                var succeedingContextTokens = updatedTokenIndex.GetTokensInSpan(change.BeforeSpan.SpanOfSucceedingContext);

                precedingContextTokens = zeroIndexVariableNames(precedingContextTokens, zeroIndexedVariableNameMap);
                succeedingContextTokens = zeroIndexVariableNames(succeedingContextTokens, zeroIndexedVariableNameMap);

                var prevCodeChunkBlockStmtTextTokens =
                    prevCodeChunkBlockStmtTokens.Select(token => token.ValueText).ToArray();
                var updatedCodeChunkBlockStmtTextTokens =
                    updatedCodeChunkBlockStmtTokens.Select(token => token.ValueText).ToArray();

                var prevCodeTextChunk = Utils.ExtractCodeTextFromBraces(prevCodeChunkBlockStmt.GetText().ToString());
                prevCodeTextChunk = Utils.RemoveLeadingWhiteSpace(prevCodeTextChunk, naive:true);

                var updatedCodeTextChunk = Utils.ExtractCodeTextFromBraces(updatedCodeChunkBlockStmt.GetText().ToString());
                updatedCodeTextChunk = Utils.RemoveLeadingWhiteSpace(updatedCodeTextChunk, naive:true);

                var precedingContextTextTokens = precedingContextTokens.Select(token => token.ValueText).ToArray();
                var succeedingContextTextTokens = succeedingContextTokens.Select(token => token.ValueText).ToArray();

                var result = new
                {
                    Id = fixerId + "_" + fileId,
                    PrevCodeChunk = prevCodeTextChunk,
                    UpdatedCodeChunk = updatedCodeTextChunk,
                    PrevCodeChunkTokens = prevCodeChunkBlockStmtTextTokens,
                    UpdatedCodeChunkTokens = updatedCodeChunkBlockStmtTextTokens,
                    PrevCodeAST = prevCodeBlockJObject,
                    UpdatedCodeAST = updatedCodeBlockJObject,
                    PrecedingContext = precedingContextTextTokens,
                    SucceedingContext = succeedingContextTextTokens,
                };

                return result;
            }

            return null;
        }

        private static string changeEntryDatumToJsonString(dynamic entry, bool withCommitMessage=false)
        {
            if (entry == null)
                return null;

            var jsonObj = new JObject();
            jsonObj["Id"] = entry.Id;
            jsonObj["PrevCodeChunk"] = entry.PrevCodeChunk;
            jsonObj["UpdatedCodeChunk"] = entry.UpdatedCodeChunk;

            jsonObj["PrevCodeChunkTokens"] = new JArray(entry.PrevCodeChunkTokens);
            jsonObj["UpdatedCodeChunkTokens"] = new JArray(entry.UpdatedCodeChunkTokens);

            jsonObj["PrevCodeAST"] = entry.PrevCodeAST;
            jsonObj["UpdatedCodeAST"] = entry.UpdatedCodeAST;

            jsonObj["PrecedingContext"] = new JArray(entry.PrecedingContext);
            jsonObj["SucceedingContext"] = new JArray(entry.SucceedingContext);

            if (withCommitMessage)
                jsonObj["CommitMessage"] = entry.CommitMessage;

            var json = JsonConvert.SerializeObject(jsonObj, Formatting.None);

            return json;
        }

        private static IEnumerable<SyntaxToken> zeroIndexVariableNames(
            IEnumerable<SyntaxToken> tokens, IDictionary<string, string> varNameMap)
        {
            SyntaxToken generateNewTokenName(SyntaxToken token)
            {
                if (token.IsKind(SyntaxKind.IdentifierToken))
                {
                    var tokenName = token.ValueText;
                    if (tokenName.StartsWith("VAR"))
                    {
                        string newTokenName;
                        if (varNameMap.ContainsKey(tokenName))
                            newTokenName = varNameMap[tokenName];
                        else
                        {
                            newTokenName = "VAR" + varNameMap.Count;
                            varNameMap[tokenName] = newTokenName;
                        }

                        return SyntaxFactory.Identifier(newTokenName);
                    }
                }

                return token;
            }

            var renamedTokens = tokens.Select(generateNewTokenName);

            return renamedTokens;
        }

        private static (BlockSyntax, BlockSyntax, IDictionary<string, string>) zeroIndexVariableNames(SyntaxNode prevCodeNode, SyntaxNode updatedCodeNode)
        {
            var varNameMap = new Dictionary<string, string>();

            SyntaxToken generateNewTokenName(SyntaxToken token)
            {
                if (token.IsKind(SyntaxKind.IdentifierToken))
                {
                    var tokenName = token.ValueText;
                    if (tokenName.StartsWith("VAR"))
                    {
                        string newTokenName;
                        if (varNameMap.ContainsKey(tokenName))
                            newTokenName = varNameMap[tokenName];
                        else
                        {
                            newTokenName = "VAR" + varNameMap.Count;
                            varNameMap[tokenName] = newTokenName;
                        }

                        return SyntaxFactory.Identifier(newTokenName);
                    }
                }
                
                return token;
            }

            var newPrevCodeNode = prevCodeNode.ReplaceTokens(prevCodeNode.DescendantTokens(), (token, _) => generateNewTokenName(token));
            var newUpdatedCodeNode = updatedCodeNode.ReplaceTokens(updatedCodeNode.DescendantTokens(), (token, _) => generateNewTokenName(token));

            return ((BlockSyntax)newPrevCodeNode, (BlockSyntax)newUpdatedCodeNode, varNameMap);
        }

        private static SyntaxNode renameIdentifiersOnAST(SyntaxNode rootNode, IDictionary<string, string> zeroIndexedVariableNameMap)
        {
            var newRootNode = rootNode.ReplaceTokens(rootNode.DescendantTokens(), (token, _) =>
            {
                if (zeroIndexedVariableNameMap.ContainsKey(token.ValueText))
                {
                    var newName = zeroIndexedVariableNameMap[token.ValueText];
                    return SyntaxFactory.Token(token.LeadingTrivia, token.Kind(), newName, newName,
                        token.TrailingTrivia);
                }

                return token;
            });

            return newRootNode;
        }

        private static (string[], string[], string[], string[], IDictionary<string, string>) zeroIndexVariables(string[] prevCodeTokens, string[] updatedCodeTokens, string[] precedingContextTokens, string[] succeedingContextTokens)
        {
            var varNameMap = new Dictionary<string, string>();

            string processToken(string token)
            {
                if (token.StartsWith("VAR"))
                {
                    if (varNameMap.ContainsKey(token))
                        return varNameMap[token];
                    else
                    {
                        var newTokenName = "VAR" + varNameMap.Count;
                        varNameMap[token] = newTokenName;
                        return newTokenName;
                    }
                }

                return token;
            }

            var newPrevCodeTokens = prevCodeTokens.Select(processToken).ToArray();
            var newUpdatedCodeTokens = updatedCodeTokens.Select(processToken).ToArray();
            var newPrecedingContextTokens = precedingContextTokens.Select(processToken).ToArray();
            var newSucceedingContextTokens = succeedingContextTokens.Select(processToken).ToArray();

            return (newPrevCodeTokens, 
                newUpdatedCodeTokens, 
                newPrecedingContextTokens, 
                newSucceedingContextTokens,
                varNameMap);
        }

        public static void DumpRevisionDataForNeuralTraining(string revisionDataFilePath, string outputFilePath, string grammarPath)
        {
            int entryProcessed = 0;
            var syntaxHelper = new JsonSyntaxTreeHelper(grammarPath);

            using(var logFile = new StreamWriter("err.log"))
            using(var fs = File.Open(outputFilePath, FileMode.Create))
            using(var sw = new StreamWriter(fs, Encoding.UTF8))
            {
                Stopwatch stopwatch = new Stopwatch();
                stopwatch.Start();

                Console.WriteLine($"Processing Fixer Root Dir {revisionDataFilePath}");
                foreach (var fixerDir in Directory.GetDirectories(revisionDataFilePath))  // .Where(x => x.EndsWith("CA2007"))
                {
                    var changeEntries = ReadRevisionData(fixerDir)
                        .AsParallel()
                        .Select(x => ProcessSingleRevision(x.Item1, x.Item2, x.Item3, x.Item4, syntaxHelper))
                        .Select(x => changeEntryDatumToJsonString(x)).Where(x => x != null);

                    int instCount = 0;

                    foreach (var changeStr in changeEntries)
                    {
                        entryProcessed++;
                        instCount++;
                        try
                        {
                            sw.WriteLine(changeStr);
                        }
                        catch (Exception) { }

                        if (entryProcessed % 10 == 0)
                            Console.Write($"\rEntry processed: {entryProcessed}");
                    }

                    logFile.WriteLine($"Processing {Path.GetFileName(fixerDir)}, {Directory.GetFiles(fixerDir).Count() / 2} Files, {instCount} Extractions");
                    logFile.Flush();
                }

                stopwatch.Stop();
                Console.WriteLine();
                Console.WriteLine("Time elapsed: {0}", stopwatch.Elapsed);
            }
        }

        public static IEnumerable<(string, string, string, string)> ReadRevisionData(string fixerDirPath)
        {
            var fixerId = Path.GetFileName(fixerDirPath);
            var allFiles = Directory.EnumerateFiles(Path.Combine(fixerDirPath), "*.cs")
                    .OrderBy(x => {
                        var fileName = Path.GetFileName(x);
                        var id = fileName.Substring(0, fileName.IndexOf('_'));
                        return int.Parse(id);
                    }).ToList();

            for (int i = 0; i < allFiles.Count; i += 2)
            {
                var fileName = Path.GetFileName(allFiles[i]);
                var fileId = fileName.Substring(0, fileName.IndexOf('_'));

                var beforeFile = fileName.Contains("before") ? allFiles[i] : allFiles[i + 1];
                var afterFile = fileName.Contains("after") ? allFiles[i] : allFiles[i + 1];

                Console.WriteLine($"{beforeFile} --> {afterFile}");

                var beforeFileContent = File.ReadAllText(beforeFile);
                var afterFileContent = File.ReadAllText(afterFile);

                yield return (fixerId, fileId, beforeFileContent, afterFileContent);
            }
        }

        public static IEnumerable<(string, string, string, string)> ReadFixerRootDir(string revisionDataFilePath)
        {
            Console.WriteLine($"Processing Fixer Root Dir {revisionDataFilePath}");
            foreach(var fixerId in Directory.GetDirectories(revisionDataFilePath).Where(x => x.EndsWith("RCS1015")))
            {
                var allFiles = Directory.EnumerateFiles(Path.Combine(revisionDataFilePath, fixerId), "*.cs")
                    .OrderBy(x => {
                        var fileName = Path.GetFileName(x);
                        var id = fileName.Substring(0, fileName.IndexOf('_'));
                        return int.Parse(id);
                    }).ToList();
                for(int i = 0; i < allFiles.Count; i+=2)
                {
                    var fileName = Path.GetFileName(allFiles[i]);
                    var fileId = fileName.Substring(0, fileName.IndexOf('_'));

                    var beforeFile = fileName.Contains("before") ? allFiles[i] : allFiles[i + 1];
                    var afterFile = fileName.Contains("after") ? allFiles[i] : allFiles[i + 1];

                    Console.WriteLine($"{beforeFile} --> {afterFile}");

                    var beforeFileContent = File.ReadAllText(beforeFile);
                    var afterFileContent = File.ReadAllText(afterFile);

                    yield return (fixerId, fileId, beforeFileContent, afterFileContent);
                }
            }
            
        }

        static readonly HashSet<SyntaxKind> allowedSytaxKinds = new HashSet<SyntaxKind>()
        {
            SyntaxKind.LocalDeclarationStatement,
            SyntaxKind.ExpressionStatement
        };

        private static bool IsValidRevision(IList<SyntaxNode> prevCodeChunkNodes, IList<SyntaxNode> updatedCodeChunkNodes)
        {
            if (String.Join('\n', prevCodeChunkNodes.Select(node => node.GetText())) ==
                String.Join('\n', updatedCodeChunkNodes.Select(node => node.GetText())))
                return false;

            if (prevCodeChunkNodes.All(node => node.IsStructuredTrivia))
                return false;

            if (updatedCodeChunkNodes.All(node => node.IsStructuredTrivia))
                return false;

            return prevCodeChunkNodes.All(node => allowedSytaxKinds.Contains(node.Kind())) &&
                   updatedCodeChunkNodes.All(node => allowedSytaxKinds.Contains(node.Kind()));
        }

        private static readonly HashSet<string> keywords =
            new HashSet<string>() {"VAR0", "int", "long", "string", "float", "LITERAL", "var"};

        private static bool IsValidCodeChunkTokens(IEnumerable<string> tokens)
        {
            var validTokenCount = tokens.Count(token => !keywords.Contains(token) && !token.All(char.IsPunctuation));

            return validTokenCount > 0;
        }
    }
}
