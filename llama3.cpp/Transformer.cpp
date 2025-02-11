#include <algorithm>
#include <cmath>
#include <numeric>
#include <cassert>
// #include <list>

#include "Transformer.h"
#include "Tokenizer.h"
#include "Logger.h"

Transformer::Transformer(Config config)
    : config(std::move(config)),
      tokenEmbeddingTable(config.vocabSize * config.dim),
      layers(config.nLayers,
             TransformerBlock(config.seqLength, config.dim, config.nHeads, config.nKVHeads, config.hiddenDim)),
      finalNorm(config.dim),
      output(config.dim, config.vocabSize),
      x(config.dim),
      xb(config.dim),
      logits(config.vocabSize)
{
    // You can directly use (gdb) p getConfig() to get the follwing config
    // logger(Logger::DEBUG) << std::endl;
    logger(Logger::DEBUG) << "==DEBUG== --------------------------------------------------------" << std::endl;
    logger(Logger::DEBUG) << "==DEBUG== Transformer model config:" << std::endl;
    logger(Logger::DEBUG) << "==DEBUG== config.dim = " << config.dim << std::endl;
    logger(Logger::DEBUG) << "==DEBUG== config.hiddenDim = " << config.hiddenDim << std::endl;
    logger(Logger::DEBUG) << "==DEBUG== config.nLayers = " << config.nLayers << std::endl;
    logger(Logger::DEBUG) << "==DEBUG== config.nHeads = " << config.nHeads << std::endl;
    logger(Logger::DEBUG) << "==DEBUG== config.nKVHeads = " << config.nKVHeads << std::endl;
    logger(Logger::DEBUG) << "==DEBUG== config.vocabSize = " << config.vocabSize << std::endl;
    logger(Logger::DEBUG) << "==DEBUG== config.seqLength = " << config.seqLength << std::endl;
    // logger(Logger::DEBUG) << "==DEBUG== (uint8_t) config.sharedClassifier = 0x" << std::hex
    //     << static_cast<unsigned int>(config.sharedClassifier) 
    //     << std::dec << std::endl;
    // for (int i=0; i<3; ++i){
    //     logger(Logger::DEBUG) << "==DEBUG== (uint8_t) config.padding[i] = 0x" << std::hex
    //     << static_cast<unsigned int>(config.padding[i]) 
    //     << std::dec << std::endl;
    // }
}

void Transformer::loadWeights(std::ifstream &inputStream)
{
    // Read tokenEmbeddingTable which might serve as the weights of
    // the classifier when sharedClassifier is true.
    Tensor tet(config.vocabSize * config.dim);
    tet.readFromFile(inputStream);

    // We always use f32 for tokenEmbeddingTable, but we might use the
    // quantized values for the classifier so keep them valid by
    // using the const version of f().
    this->tokenEmbeddingTable = tet.cf();

    for (int i = 0; i < config.nLayers; ++i)
        layers[i].loadWeights(inputStream);

    finalNorm.loadWeights(inputStream);

    if (config.sharedClassifier)
        if (tet.isQuantizedValid())
            output.setWeights(tet.cq());
        else
            output.setWeights(tet.f());
    else
        output.loadWeights(inputStream);
}

void Transformer::forward(TokenQueue tokens, Tensor &logits)
{

    // original version:
    // assert(tokens.size() == 1);
    // std::copy(tokenEmbeddingTable.data() + tokens.front() * config.dim, tokenEmbeddingTable.data() + (1 + tokens.front()) * config.dim, x.f().data());

    // prefill version:
    // b Transformer.cpp:78
    x = Tensor(tokens.size() * config.dim);
    float* x_data = x.f().data();
    int i=0;
    for (int token : tokens) {
        int start_idx = token * config.dim;
        int end_idx = start_idx + config.dim;
        std::copy(
            tokenEmbeddingTable.data() + start_idx,
            tokenEmbeddingTable.data() + end_idx,
            x_data + i * config.dim
        );
        ++i;
    }

    std::reference_wrapper<Tensor> t1 = x;
    std::reference_wrapper<Tensor> t2 = xb;

    // forward all the layers
    for (int l = 0; l < config.nLayers; l++)
    {
        logger(Logger::TRACE) << "==TRACE== layers[" << l << "].forward" << std::endl;
        layers[l].forward(t1, t2);
        // works like a ping-pong buffer, let the last layer output as the curr layer input
        std::swap(t1, t2);
    }

    // final rmsnorm
    finalNorm.forward(t1, xb);

    // classifier into logits
    output.forward(xb, logits);

}

Config const &Transformer::getConfig() { return config; }
