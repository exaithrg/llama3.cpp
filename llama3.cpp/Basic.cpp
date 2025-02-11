#include <chrono>
#include <fstream>
#include <iostream>

#include "Transformer.h"
#include "Tokenizer.h"
#include "Logger.h"
#include "Sampler.h"

// ----------------------------------------------------------------------------
// utilities: time
auto time_in_ms()
{
    // return time in milliseconds, for benchmarking the model speed
    return duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
}

void check_header(std::ifstream &inputStream)
{
    using namespace std::string_literals;

    uint32_t magic_number;
    inputStream.read((char *)(&magic_number), sizeof(magic_number));

    // read in magic number (uint32), has to be 0x616b3432, i.e. "ak42" in ASCII
    if (magic_number != 0x616b3432)
        throw std::runtime_error("Bad magic number");

    int version;
    inputStream.read((char *)(&version), sizeof(version));

    if (version != 1)
        throw std::runtime_error("Bad version "s + std::to_string(version) + "need version 2"s);
}

Transformer build_transformer(std::string const &checkpoint_path)
{
    std::ifstream inputStream(checkpoint_path, std::ios::binary);

    if (!inputStream.is_open())
        throw std::runtime_error("Can not open file!");

    check_header(inputStream);

    // read in the Config and the Weights from the checkpoint
    Config config;
    inputStream.read((char *)(&config), sizeof(Config));

    inputStream.seekg(256);

    Transformer transformer(config);
    transformer.loadWeights(inputStream);

    return transformer;
}

// ----------------------------------------------------------------------------
// generation loop
void generate(Transformer &transformer, Tokenizer const &tokenizer, Sampler &sampler, std::string const &prompt, size_t numSteps)
{
    // encode the (string) prompt into tokens sequence
    auto prompt_tokens = tokenizer.encode(prompt, 1, 0);

    if (prompt_tokens.size() < 1){
        logger(Logger::ERROR) << "==ERROR== something is wrong, expected at least 1 prompt token" << std::endl;
        throw std::runtime_error("==ERROR== something is wrong, expected at least 1 prompt token");
        return;
    }

    logger(Logger::DEBUG) << "==DEBUG== --------------------------------------------------------" << std::endl;
    logger(Logger::DEBUG) << "==DEBUG== GENERATION start with " << prompt_tokens.size() << " tokens" << std::endl;

    // start the main loop
    logger(Logger::INFO) << "==INFO== --------------------------------------------------------" << std::endl;
    logger(Logger::INFO) << "==INFO== GENERATION LOOP" << std::endl;
    logger(Logger::INFO) << "==INFO== --------------------------------------------------------" << std::endl;
    std::optional<std::chrono::milliseconds> start; // used to time our code, only initialized after first iteration
    size_t steps = 0;
    int next_token;

    // b Basic.cpp:82
    logger(Logger::DEBUG) << "==DEBUG== INPUT TOKENS:" << std::endl;
    for(auto it = prompt_tokens.begin(); it != prompt_tokens.end(); ++it){
        auto p = tokenizer.decode(*it);
        std::cout << *p << std::flush;
    }
    logger(Logger::DEBUG) << std::endl;

    // PREFILL STAGE
    logger(Logger::DEBUG) << "==DEBUG== PREFILL STAGE start with " << prompt_tokens.size() <<" tokens" << std::endl;
    Tensor logits(transformer.getConfig().vocabSize);
    // b Basic.cpp:93
    transformer.forward(prompt_tokens, logits);
    next_token = sampler.sample(logits.f());
    if (next_token == 128001 || next_token == 128009){
        std::cout << std::endl;
        logger(Logger::DEBUG) << "==DEBUG== --------------------------------------------------------" << std::endl;
        logger(Logger::DEBUG) << "==DEBUG== GENERATION end at PREFILL STAGE." << std::endl;
        return;
    }
    if (auto p = tokenizer.decode(next_token)){
        logger(Logger::DEBUG) << "==DEBUG== PREFILL STAGE end with generated token: [";
        std::cout << *p << std::flush;
        logger(Logger::DEBUG) << "]" << std::endl;
    }

    // DECODE STAGE
    TokenQueue tokens = {next_token};
    // 0 means infinity
    while (0 == numSteps || steps < numSteps)
    {
        // forward the transformer to get logits for the next token
        logger(Logger::DEBUG) << "==DEBUG== DECODE STAGE " << steps << " start with token=" << next_token << std::endl;
        transformer.forward(tokens, logits);

        next_token = sampler.sample(logits.f());

        // data-dependent terminating condition: the BOS (=1) token delimits sequences
        if (next_token == 128001 || next_token == 128009)
            break;

        // print the token as string, decode it with the Tokenizer object
        if (auto p = tokenizer.decode(next_token)){
            logger(Logger::DEBUG) << "==DEBUG== Step " << steps << " with generated token: [";
            std::cout << *p << std::flush;
            logger(Logger::DEBUG) << "]" << std::endl;
        }

        // init the timer here because the first iteration can be slower
        if (!start.has_value())
            start = time_in_ms();

        ++steps;
    }
    std::cout << std::endl;

    // report achieved tok/s (pos-1 because the timer starts after first iteration)
    if (start.has_value())
    {
        auto end = time_in_ms();
        auto elapsed = (end - *start).count();
        if (0 < elapsed){
            logger(Logger::DEBUG) << "==DEBUG== --------------------------------------------------------" << std::endl;
            logger(Logger::DEBUG) << "==DEBUG== achieved tok/s: " << static_cast<double>(steps - 1) / elapsed * 1000 << std::endl;
        }
    }

    logger(Logger::DEBUG) << "==DEBUG== --------------------------------------------------------" << std::endl;
    logger(Logger::DEBUG) << "==DEBUG== GENERATION end with " << steps << " tokens" << std::endl;

}

