#pragma once

#include <cstdint>
#include <fstream>
#include <vector>

#include "Tensor.h"
#include "layers.h"

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
    // 构造函数，但是实际用的是Transformer build_transformer(std::string const &checkpoint_path)
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

    Tensor x;
    Tensor xb;
    Tensor logits;
};
