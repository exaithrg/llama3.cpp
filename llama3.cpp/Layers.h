#pragma once

#include <fstream>
#include <vector>

#include "Tensor.h"

class RMSNorm
{
public:
    RMSNorm(size_t dim);

    void forward(Tensor &x, Tensor &out);
    void loadWeights(std::ifstream &inputStream);

private:
    size_t dim;
    Tensor weight;
};

class Linear
{
public:
    Linear(size_t inDim, size_t outDim);

    void forward(Tensor &x, Tensor &out);
    void loadWeights(std::ifstream &inputStream);

    template <typename T> void setWeights(T const &w);

private:
    size_t inDim;
    size_t outDim;
    Tensor weight;
};

class CausalAttention
{
public:
    CausalAttention(size_t seqLength, size_t dim, size_t nHeads, size_t nKVHeads);

    void forward(Tensor &x, Tensor &out);
    void loadWeights(std::ifstream &inputStream);

private:
    size_t pos;
    size_t dim;
    size_t nHeads;
    size_t nKVHeads;

    Linear wq;
    Linear wk;
    Linear wv;
    Linear wo;

    Tensor query;
    std::vector<Tensor> keyCache;
    std::vector<Tensor> valueCache;

    std::vector<Tensor> att; // buffer for scores/attention values (n_heads, seq_len)

    Tensor xb;
};

class FFN
{
public:
    FFN(size_t dim, size_t hiddenDim);

    void forward(Tensor &x, Tensor &out);
    void loadWeights(std::ifstream &inputStream);

private:
    size_t dim;
    size_t hiddenDim;

    Linear w1;
    Linear w2;
    Linear w3;

    Tensor hb;
    Tensor hb2;
};

class TransformerBlock
{
public:
    TransformerBlock(size_t seqLength, size_t dim, size_t nHeads, size_t nKVHeads, size_t hiddenDim);

    void forward(Tensor &x, Tensor &out);
    void loadWeights(std::ifstream &inputStream);

private:
    RMSNorm attentionNorm;
    CausalAttention attention;
    RMSNorm ffnNorm;
    FFN ffn;

    // buffer 1
    Tensor xb;
    // buffer 2. why so mamy buffer neededd?
    Tensor xb2;
};
