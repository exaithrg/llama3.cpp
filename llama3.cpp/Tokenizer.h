#pragma once

#include <list>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include "Tensor.h"

// ----------------------------------------------------------------------------
// The Byte Pair Encoding (BPE) Tokenizer that translates strings <-> tokens

class TokenQueue : public std::list<int>
{
public:
    void push(int token);
    int pop();
};

class Tokenizer
{
public:
    Tokenizer(std::string path, int vocabSize);

    std::optional<std::string> decode(int token) const;
    TokenQueue encode(std::string text, bool bos, bool eos) const;

private:
    struct TokenIndex
    {
        std::string const *str;
        int id;

        bool operator<(Tokenizer::TokenIndex const &second) const;
        bool operator==(Tokenizer::TokenIndex const &second) const;
    };

    std::optional<int> strLookUp(std::string str) const;

    void merge(TokenQueue &tokens) const;

private:
    int vocabSize;
    FloatTensor vocabScores;

    unsigned char bytePieces[512];
    unsigned int maxTokenLength;

    std::vector<std::string> vocab;
    std::set<TokenIndex> sortedVocab;
};