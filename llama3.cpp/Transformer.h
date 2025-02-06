#pragma once

#include <cstdint>
#include <fstream>
#include <vector>

#include "Tensor.h"
#include "Layers.h"

struct Config
{
    int dim;       // transformer dimension
    int hiddenDim; // for ffn layers
    int nLayers;   // number of layers
    int nHeads;    // number of query heads
    int nKVHeads;  // number of key/value heads (can be < query heads because of multiquery)
    int vocabSize; // vocabulary size, usually 4096 (byte-level)
    int seqLength; // max sequence length
    uint8_t sharedClassifier;
    uint8_t padding[3];
};

class Transformer
{
public:
    Transformer(Config config);

    void loadWeights(std::ifstream &inputStream);
    void forward(int token, Tensor &logits);

    Config const &getConfig();

private:
    Config config;

    FloatTensor tokenEmbeddingTable; // (vocab_size, dim)

    std::vector<TransformerBlock> layers;
    RMSNorm finalNorm;
    Linear output;

    // x means input
    Tensor x;
    // xb is the input buffer (or output buffer, anyway)
    Tensor xb;
    Tensor logits;
};
