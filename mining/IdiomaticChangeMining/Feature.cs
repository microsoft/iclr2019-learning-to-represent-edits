// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace IdiomaticChangeMining
{
    [JsonConverter(typeof(FeatureJsonConverter))]
    public class Feature
    {
        public IList<SyntaxToken> RemovedTokens { get; set; }

        public IList<SyntaxToken> AddedTokens { get; set; }

        public IList<SyntaxToken> PrevCodeTokens { get; set; }

        public IList<SyntaxToken> UpdatedCodeTokens { get; set; }

        public List<SyntaxToken> CommonTokens { get; set; }

        public Feature()
        {

        }

        public override string ToString()
        {
            return $"Feature[Removed: {String.Join(", ", RemovedTokens)} ||| Added: {String.Join(", ", AddedTokens)} ||| Common: {String.Join(", ", CommonTokens)}]";
        }

        static readonly HashSet<SyntaxKind> _omittedTokenKinds = new HashSet<SyntaxKind>() { SyntaxKind.SemicolonToken, SyntaxKind.CommaToken, SyntaxKind.OpenParenToken, SyntaxKind.CloseParenToken };

        public static Feature GetFeature(CodeChange change)
        {
            var canonicalPrevCodeTokens = change.CanonicalPrevCodeChunkNodes.SelectMany(node => node.DescendantTokens()).Where(tok => ! _omittedTokenKinds.Contains(tok.Kind())).ToList(); // Where(tok => tok.IsKind(SyntaxKind.IdentifierToken))
            var canonicalUpdatedCodeTokens = change.CanonicalUpdatedCodeChunkNodes.SelectMany(node => node.DescendantTokens()).Where(tok => !_omittedTokenKinds.Contains(tok.Kind())).ToList();

            var canonicalPrevCodeTokensValSet = canonicalPrevCodeTokens.Select(tok => tok.ValueText).ToHashSet();
            var canonicalUpdatedCodeTokensValSet = canonicalUpdatedCodeTokens.Select(tok => tok.ValueText).ToHashSet();

            var feat = new Feature()
            {
                RemovedTokens = canonicalPrevCodeTokens.Where(tok => !canonicalUpdatedCodeTokensValSet.Contains(tok.ValueText)).ToList(),
                AddedTokens = canonicalUpdatedCodeTokens.Where(tok => !canonicalPrevCodeTokensValSet.Contains(tok.ValueText)).ToList(),
                CommonTokens = canonicalPrevCodeTokens.Where(tok => canonicalUpdatedCodeTokensValSet.Contains(tok.ValueText)).ToList(),
                PrevCodeTokens = canonicalPrevCodeTokens,
                UpdatedCodeTokens = canonicalUpdatedCodeTokens
            };

            return feat;
        }
    }

    public class FeatureJsonConverter : JsonConverter
    {
        public override bool CanConvert(Type objectType)
        {
            return true;
        }

        public override object ReadJson(JsonReader reader, Type objectType, object existingValue, JsonSerializer serializer)
        {
            throw new NotImplementedException();
        }

        public override void WriteJson(JsonWriter writer, object value, JsonSerializer serializer)
        {
            var featObj = value as Feature;
            var jsonObject = new JObject(
                new JProperty("PrevCodeTokens", featObj.PrevCodeTokens.Select(tok => tok.ValueText).ToList()),
                new JProperty("UpdatedCodeTokens", featObj.UpdatedCodeTokens.Select(tok => tok.ValueText).ToList()),
                new JProperty("AddedTokens", featObj.AddedTokens.Select(tok => tok.ValueText).ToList()),
                new JProperty("RemovedTokens", featObj.RemovedTokens.Select(tok => tok.ValueText).ToList()),
                new JProperty("CommonTokens", featObj.CommonTokens.Select(tok => tok.ValueText).ToList())
            );

            jsonObject.WriteTo(writer);
        }
    }
}
