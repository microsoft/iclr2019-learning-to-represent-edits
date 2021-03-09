// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Reflection.Metadata;
using System.Text;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Mono.Options;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace IdiomaticChangeMining
{
    class SyntheticExp
    {
        public abstract class ToyExpression
        {
            public abstract ToyExpression Copy();
        }

        public class ToyFunctionCall : ToyExpression
        {
            public string Name { get; set; }
            public List<ToyExpression> Arguments { get; set; } = new List<ToyExpression>();

            public ToyFunctionCall(string funcName, List<ToyExpression> argList = null)
            {
                this.Name = funcName;

                if (argList != null)
                    this.Arguments = argList;
            }

            public ToyFunctionCall(ToyFunctionCall other)
            {
                this.Name = other.Name;
                this.Arguments = new List<ToyExpression>(other.Arguments.Select(x => x.Copy()).ToList());
            }

            public override string ToString()
            {
                return Name + "(" + string.Join(",", Arguments.Select(arg => arg.ToString()).ToArray()) + ")";
            }

            public override int GetHashCode()
            {
                return 31 * Name.GetHashCode() + 17 * Arguments.GetHashCode();
            }

            public override bool Equals(object obj)
            {
                return obj is ToyFunctionCall && Equals((ToyFunctionCall)obj);
            }

            public bool Equals(ToyFunctionCall other)
            {
                return other.Name == Name && other.Arguments == Arguments;
            }

            public override ToyExpression Copy()
            {
                var copiedArgs = Arguments.Select(arg => arg.Copy()).ToList();
                return new ToyFunctionCall(Name, copiedArgs);
            }

        }

        public class ToyVariable : ToyExpression
        {
            public string Name { get; set; }

            public ToyVariable(string name)
            {
                this.Name = name;
            }

            public override string ToString()
            {
                return Name;
            }

            public override int GetHashCode()
            {
                return 31 * Name.GetHashCode();
            }

            public override bool Equals(object obj)
            {
                return obj is ToyVariable && Equals((ToyVariable)obj);
            }

            public bool Equals(ToyVariable other)
            {
                return other.Name == Name;
            }

            public override ToyExpression Copy()
            {
                return new ToyVariable(Name);
            }
        }

        public List<string> Members { get; set; } = new List<string>();

        public List<string> Variables { get; set; } = new List<string>();

        public SyntheticExp()
        {
            Members = File.ReadAllLines("members.txt").ToList();
            for (char v = 'a'; v <= 'z'; v++)
            {
                Variables.Add(v.ToString());
            }
        }

        public ToyFunctionCall SampleFunctionCall(Random rand, int nestLevel, int maxArgNum, int minArgNum=0)
        {
            // FunctionCall(A) = Foo(.Bar)*(A)
            // Example = FunctionCall((,FunctionCall())*)

            string funcName = SampleFunctionName(rand);
            if (nestLevel == 0)
            {
                return new ToyFunctionCall(funcName);
            }

            int argNum = rand.Next(minArgNum, maxArgNum + 1);
            var argList = new List<ToyExpression>();
            for (int i = 0; i < argNum; i++)
            {
                int simpleArg = rand.Next(2);
                if (simpleArg == 1)
                {
                    var varName = SampleVariableName(rand);
                    argList.Add(varName);
                }
                else
                {
                    var arg = SampleFunctionCall(rand, nestLevel - 1, maxArgNum);
                    argList.Add(arg);
                }
            }

            var func = new ToyFunctionCall(funcName, argList);
            return func;
        }

        public string SampleFunctionName(Random rand)
        {
            int maxMemberAccessNum = 3;
            int memberNum = rand.Next(1, maxMemberAccessNum + 1);
            string name = "";
            for (int i = 0; i < memberNum; i++)
            {
                var sampledName = Members[rand.Next(Members.Count)];
                name = name + "." + sampledName;
            }

            return name.Substring(1);
        }

        public ToyVariable SampleVariableName(Random rand)
        {
            return new ToyVariable(Variables[rand.Next(Variables.Count)]);
        }

        private static string changeEntryDatumToJsonString(dynamic entry)
        {
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

            var json = JsonConvert.SerializeObject(jsonObj, Formatting.None);

            return json;
        }

        public HashSet<(ToyFunctionCall OldCode, ToyFunctionCall NewCode)> GenerateArgSwapDataset()
        {
            var rand = new Random(Seed: 1);

            var samples = new HashSet<(ToyFunctionCall OldCode, ToyFunctionCall NewCode)>();
            var sampleKeys = new HashSet<string>();
            for (int i = 0; i < 10000; i++)
            {
                var oldCall = SampleFunctionCall(rand, nestLevel: 2, maxArgNum: 5, minArgNum: 2);
                var newCallWithSwappedArgs = new ToyFunctionCall(oldCall);

                var t = oldCall.Arguments[0];
                oldCall.Arguments[0] = oldCall.Arguments[oldCall.Arguments.Count - 1];
                oldCall.Arguments[oldCall.Arguments.Count - 1] = t;

                Console.WriteLine("******************************");
                Console.WriteLine(oldCall.ToString());
                Console.WriteLine();
                Console.WriteLine(newCallWithSwappedArgs.ToString());

                var sample = (oldCall, newCallWithSwappedArgs);
                var sampleKey = oldCall.ToString() + "|||" + newCallWithSwappedArgs.ToString();
                if (!sampleKeys.Contains(sampleKey) && oldCall.ToString() != newCallWithSwappedArgs.ToString())
                {
                    samples.Add(sample);
                    sampleKeys.Add(sampleKey);
                }
            }

            return samples;
        }

        public HashSet<(ToyFunctionCall OldCode, ToyFunctionCall NewCode)> GenerateMemeberAceessReplaceDataset()
        {
            var rand = new Random(Seed: 1);

            var samples = new HashSet<(ToyFunctionCall OldCode, ToyFunctionCall NewCode)>();
            var sampleKeys = new HashSet<string>();
            while (samples.Count < 8000)
            {
                var oldCall = SampleFunctionCall(rand, nestLevel: 2, maxArgNum: 5, minArgNum: 2);
                var funcCallArgs = oldCall.Arguments.Select((x, idx) => new {Arg = x, Id = idx}).Where(x => x.Arg is ToyFunctionCall)
                    .Select(x => x.Id).ToList();
                if (funcCallArgs.Count >= 1)
                {
                    int firstFuncCallArg = funcCallArgs.First();
                    var newCall = oldCall.Copy() as ToyFunctionCall;
                    newCall.Name = ((ToyFunctionCall) newCall.Arguments[firstFuncCallArg]).Name;
                    ((ToyFunctionCall) newCall.Arguments[firstFuncCallArg]).Name = oldCall.Name;

                    Console.WriteLine("******************************");
                    Console.WriteLine(oldCall.ToString());
                    Console.WriteLine();
                    Console.WriteLine(newCall.ToString());

                    var sample = (oldCall, newCall);
                    var sampleKey = oldCall.ToString() + "|||" + newCall.ToString();
                    if (!sampleKeys.Contains(sampleKey) && oldCall.ToString() != newCall.ToString())
                    {
                        samples.Add(sample);
                        sampleKeys.Add(sampleKey);
                    }
                }
            }

            return samples;
        }

        public static void Main(string[] args)
        {
            var syntaxHelper = new JsonSyntaxTreeHelper(@"../../../../../data/grammar.full.json");

            var exp = new SyntheticExp();

            var samples = exp.GenerateMemeberAceessReplaceDataset();

            var id = 0;
            using(var sw = new StreamWriter(@"toy_exp.member_access_replace.jsonl"))
            foreach (var sample in samples)
            {
                var oldCodeBlockNode = SyntaxTreeHelper.GetBlockSyntaxNodeForLinesOfCode(new string[] { sample.OldCode.ToString() + ";" });
                var newCodeBlockNode = SyntaxTreeHelper.GetBlockSyntaxNodeForLinesOfCode(new string[] { sample.NewCode.ToString() + ";" });

                var prevCodeChunkBlockStmtTokens = oldCodeBlockNode.DescendantTokens().Skip(1).SkipLast(1).ToArray();
                var prevCodeChunkBlackStmtTokensIndex = new TokenIndex(prevCodeChunkBlockStmtTokens).InitInvertedIndex();

                var updatedCodeChunkBlockStmtTokens = newCodeBlockNode.DescendantTokens().Skip(1).SkipLast(1).ToArray();
                var updatedCodeChunkBlockStmtTokensIndex = new TokenIndex(updatedCodeChunkBlockStmtTokens).InitInvertedIndex();

                var prevCodeBlockJObject = syntaxHelper.GetJObjectForSyntaxNode(oldCodeBlockNode, prevCodeChunkBlackStmtTokensIndex);
                var updatedCodeBlockJObject = syntaxHelper.GetJObjectForSyntaxNode(newCodeBlockNode, updatedCodeChunkBlockStmtTokensIndex);

                if (prevCodeChunkBlockStmtTokens.Length <= 70)
                {
                    var entry = new
                    {
                        Id = id,
                        PrevCodeChunk = sample.OldCode.ToString(),
                        UpdatedCodeChunk = sample.NewCode.ToString(),
                        PrevCodeChunkTokens = prevCodeChunkBlockStmtTokens.Select(token => token.ValueText).ToArray(),
                        UpdatedCodeChunkTokens = updatedCodeChunkBlockStmtTokens.Select(token => token.ValueText).ToArray(),
                        PrevCodeAST = prevCodeBlockJObject,
                        UpdatedCodeAST = updatedCodeBlockJObject,
                        PrecedingContext = new string[] { },
                        SucceedingContext = new string[] { }
                    };

                    var jsonStr = changeEntryDatumToJsonString(entry);
                    sw.WriteLine(jsonStr);

                    id += 1;
                }
            }
        }
    }
}
