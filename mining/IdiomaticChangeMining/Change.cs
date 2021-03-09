// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace IdiomaticChangeMining
{
    public class CodeChange
    {
        public string Id { get; set; }

        public string PreviousCodeFile { get; set; }

        public string UpdatedCodeFile { get; set; }

        public int PrevFileChangeStartIdx { get; set; }

        public int PrevFileChangeEndIdx { get; set; }

        public int UpdatedFileChangeStartIdx { get; set; }

        public int UpdatedFileChangeEndIdx { get; set; }

        public string CanonicalPrevCodeChunk { get; set; }

        public string CanonicalUpdatedCodeChunk { get; set; }

        public string PreviousCodeChunk
        {
            get
            {
                return String.Join('\n', this.PreviousCodeFile.Split('\n').Skip(this.PrevFileChangeStartIdx).Take(this.PrevFileChangeEndIdx - this.PrevFileChangeStartIdx + 1));
            }
        }

        public string UpdatedCodeChunk
        {
            get
            {
                return String.Join('\n', this.UpdatedCodeFile.Split('\n').Skip(this.UpdatedFileChangeStartIdx).Take(this.UpdatedFileChangeEndIdx - this.UpdatedFileChangeStartIdx + 1));
            }
        }

        public Feature Feature { get; set; }

        [JsonIgnore]
        public List<SyntaxNode> CanonicalPrevCodeChunkNodes { get; internal set; }

        [JsonIgnore]
        public List<SyntaxNode> CanonicalUpdatedCodeChunkNodes { get; internal set; }

        public CodeChange() { }

        static readonly HashSet<SyntaxKind> allowedSytaxKinds = new HashSet<SyntaxKind>()
        {
            SyntaxKind.LocalDeclarationStatement,
            SyntaxKind.ExpressionStatement
        };

        public static bool IsValidChangeGivenAstList(IList<SyntaxNode> prevCodeChunkAsts, IList<SyntaxNode> updatedCodeChunkAsts)
        {
            if (prevCodeChunkAsts.All(node => node.IsStructuredTrivia))
                return false;

            if (updatedCodeChunkAsts.All(node => node.IsStructuredTrivia))
                return false;

            return prevCodeChunkAsts.All(node => allowedSytaxKinds.Contains(node.Kind())) &&
                   updatedCodeChunkAsts.All(node => allowedSytaxKinds.Contains(node.Kind()));
        }
    }


}