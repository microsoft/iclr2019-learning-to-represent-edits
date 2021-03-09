// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.Text;
using Mono.Options;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using DocoptNet;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace IdiomaticChangeMining
{
    class Program
    {
        private const string usage = @"Idiomatic Change Mining.

    Usage:
      IdiomaticChangeMining.exe get_python_input <input_file> <output_file> <grammar_file>
      IdiomaticChangeMining.exe get_python_input_for_fixer_data <input_file> <output_file> <grammar_file>
      IdiomaticChangeMining.exe get_python_input_single_file_change <input_file> <output_file> <grammar_file>
      IdiomaticChangeMining.exe get_embedding_train_file <input_folder> <output_file>
      IdiomaticChangeMining.exe get_toy_exp_data      

    Options:
      -h --help     Show this screen.

    ";

        private static void Main(string[] args)
        {
            var arguments = new Docopt().Apply(usage, args, version: "0.1", exit: true);
            foreach (var argument in arguments)
            {
                Console.WriteLine("{0} = {1}", argument.Key, argument.Value);
            }

            if (arguments["get_python_input"].IsTrue)
            {
                Pipeline.DumpRevisionDataForNeuralTraining(arguments["<input_file>"].ToString(),
                    arguments["<output_file>"].ToString(),
                    arguments["<grammar_file>"].ToString());
            }
            else if (arguments["get_python_input_for_fixer_data"].IsTrue)
            {
                FixerDataPipeline.DumpRevisionDataForNeuralTraining(arguments["<input_file>"].ToString(),
                    arguments["<output_file>"].ToString(),
                    arguments["<grammar_file>"].ToString());
            }
            else if (arguments["get_python_input_single_file_change"].IsTrue)
            {
                Pipeline.DumpRevisionDataForCommitMessagePrediction(arguments["<input_file>"].ToString(),
                    arguments["<output_file>"].ToString(),
                    arguments["<grammar_file>"].ToString());
            }
            else if (arguments["get_embedding_train_file"].IsTrue)
            {
                EmbeddingHelper.TokenizeCSharpCodeFolder(arguments["<input_folder>"].ToString(), arguments["<output_file>"].ToString());
            }
            else if (arguments["get_toy_exp_data"].IsTrue)
            {
                SyntheticExp.Main(null);
            }
        }
    }
}
