/* Inference for Llama-3 Transformer model in C++ */

#include <fstream>
#include <iostream>
#include <string>

#include "argparse/argparse.hpp"

#include "Logger.h"
#include "Sampler.h"
#include "Tokenizer.h"
#include "Transformer.h"
#include "Basic.h"

int main(int argc, char *argv[])
{
    auto args = argparse::parse<MyArgs>(argc, argv);

    if (args.debug)
        logger.setLevel(Logger::DEBUG);

    logger(Logger::DEBUG) << "--------------------------------------------------------" << std::endl;
    logger(Logger::DEBUG) << "Building the Transformer via the model .bin file..." << std::endl;

    // build the Transformer via the model .bin file
    Transformer transformer = build_transformer(args.checkpoint_path);
    Tokenizer tokenizer(args.tokenizerPath, transformer.getConfig().vocabSize);
    NucleusSampler sampler(transformer.getConfig().vocabSize, args.temperature, args.topP, args.rngSeed);
    // run!
    if (args.mode == "generate"){
        logger(Logger::DEBUG) << "--------------------------------------------------------" << std::endl;
        logger(Logger::DEBUG) << "Model building ok, generation start..." << std::endl;
        generate(transformer, tokenizer, sampler, args.prompt, args.steps);
    }
    else if (args.mode == "chat")
        chat(transformer, tokenizer, sampler, args.systemPrompt, args.steps);
    else
        std::cerr << "unknown mode: " << args.mode << std::endl;

    return 0;
}
