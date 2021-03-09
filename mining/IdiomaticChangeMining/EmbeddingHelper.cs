// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;

namespace IdiomaticChangeMining
{
    class EmbeddingHelper
    {
        public static void TokenizeCSharpCodeFolder(string folderPath, string outputFile)
        {
            IEnumerable<string> GetAllCSharpCodeFileContentsInDirectory()
            {
                foreach (var fileName in Directory.EnumerateFiles(folderPath, "*.cs", SearchOption.AllDirectories))
                {
                    var content = File.ReadAllText(fileName);
                    yield return content;
                }
            }

            GetAllCSharpCodeFileContentsInDirectory();

            var resultStream = GetAllCSharpCodeFileContentsInDirectory().AsParallel()
                .Select(SyntaxTreeHelper.GetSyntaxTokenStringsFromSourceCode);

            using (var fs = File.Open(outputFile, FileMode.Create))
            using (var sw = new StreamWriter(fs, Encoding.UTF8))
            {
                int numProcessed = 0;
                foreach (var fileTokens in resultStream)
                {
                    var str = String.Join(' ', fileTokens);
                    if (string.IsNullOrEmpty(str))
                        continue;

                    try
                    {
                        sw.WriteLine();
                        sw.WriteLine(str);
                    }
                    catch (Exception) { }

                    numProcessed += 1;
                    if (numProcessed % 10 == 0)
                    {
                        Console.Write("\rFile processed: {0}", numProcessed);
                    }
                }
            }
        }
    }
}
