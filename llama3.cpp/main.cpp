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

    if (args.debug){
        std::map<std::string, Logger::Level> levelMap = {
            {"FATAL", Logger::Level::FATAL},
            {"ERROR", Logger::Level::ERROR},
            {"WARN", Logger::Level::WARN},
            {"INFO", Logger::Level::INFO},
            {"DEBUG", Logger::Level::DEBUG},
            {"TRACE", Logger::Level::TRACE}
        };
        auto it = levelMap.find(args.debuglevel);
        if (it != levelMap.end())
            logger.setLevel(it->second);
        else {
            logger(Logger::WARN) << "==WARN== Invalid debug level: " << args.debuglevel << ". Setting to WARN." << std::endl;
            logger.setLevel(Logger::Level::WARN);
        }
    }

    logger(Logger::INFO) << "==INFO== --------------------------------------------------------" << std::endl;
    logger(Logger::INFO) << "==INFO== Building the Transformer via model: " << args.checkpoint_path << " ..." << std::endl;
    logger(Logger::INFO) << "==INFO== Building the Transformer via tokenizer: " << args.tokenizerPath << " ..." << std::endl;

    // build the Transformer via the model .bin file
    Transformer transformer = build_transformer(args.checkpoint_path);
    Tokenizer tokenizer(args.tokenizerPath, transformer.getConfig().vocabSize);
    NucleusSampler sampler(transformer.getConfig().vocabSize, args.temperature, args.topP, args.rngSeed);
    // run!
    if (args.mode == "generate"){
        logger(Logger::INFO) << "==INFO== --------------------------------------------------------" << std::endl;
        logger(Logger::INFO) << "==INFO== Model building ok, generation start..." << std::endl;
        generate(transformer, tokenizer, sampler, args.prompt, args.steps);
    }
    else
        std::cerr << "==ERROR== unknown mode: " << args.mode << std::endl;

    logger(Logger::INFO) << "==INFO== --------------------------------------------------------" << std::endl;
    logger(Logger::INFO) << "==INFO== DONE" << std::endl;

    return 0;
}
