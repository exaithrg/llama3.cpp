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
    logger(Logger::INFO) << "==INFO== --------------------------------------------------------" << std::endl;
    logger(Logger::INFO) << "==INFO== GENERATION LOOP" << std::endl;
    logger(Logger::INFO) << "==INFO== --------------------------------------------------------" << std::endl;

    // encode the (string) prompt into tokens sequence
    auto prompt_tokens = tokenizer.encode(prompt, 1, 0);
    // b Basic.cpp:67
    if (prompt_tokens.size() < 1)
        throw std::runtime_error("==ERROR== something is wrong, expected at least 1 prompt token");

    // start the main loop
    std::optional<std::chrono::milliseconds> start; // used to time our code, only initialized after first iteration
    size_t steps = 0;

    int token = prompt_tokens.pop();
    Tensor logits(transformer.getConfig().vocabSize);

    // 0 means infinity
    while (0 == numSteps || steps < numSteps)
    {
        // forward the transformer to get logits for the next token
        logger(Logger::DEBUG) << "==DEBUG== Transformer::forward " << steps << " start with token=" << token << std::endl;
        transformer.forward(token, logits);

        // advance the state machine
        if (!prompt_tokens.empty())
            // if we are still processing the input prompt, force the next prompt token
            token = prompt_tokens.pop();
        else
            // otherwise sample the next token from the logits
            token = sampler.sample(logits.f());

        // data-dependent terminating condition: the BOS (=1) token delimits sequences
        if ((token == 128001 || token == 128009) && prompt_tokens.empty())
            break;

        // print the token as string, decode it with the Tokenizer object
        if (auto p = tokenizer.decode(token)){
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
}

// ----------------------------------------------------------------------------
// chat loop
// I manually inspected the tokens for a few chat conversations compared to
// python reference and that seemed ok, but this was not thoroughly tested and
// is not safely implemented, it's more a proof of concept atm.

void chat(Transformer &transformer, Tokenizer const &tokenizer, Sampler &sampler, std::string system_prompt, size_t numSteps)
{
    if (system_prompt == "")
    {
        std::cout << "Enter system prompt (optional): ";
        getline(std::cin, system_prompt);
    }

    TokenQueue prompt_tokens;

    // start the main loop
    int8_t turn = 0; // user in even turns
    size_t steps = 0;
    int token = 0; // stores the current token to feed into the transformer
    Tensor logits(transformer.getConfig().vocabSize);

    while (0 == numSteps || steps < numSteps)
    {
        // when it is the user's turn to contribute tokens to the dialog...
        if (0 == turn % 2)
        {

            if (0 == turn)
            {
                // "<|begin_of_text|>" "<|start_header_id|>" "system" "<|end_header_id|>" "\n\n"
                prompt_tokens.insert(prompt_tokens.end(), {128000, 128006, 9125, 128007, 271});

                if (!system_prompt.empty())
                {
                    auto system_prompt_tokens = tokenizer.encode(system_prompt, 0, 0);
                    std::copy(system_prompt_tokens.begin(), system_prompt_tokens.end(),
                              std::back_inserter(prompt_tokens));
                }

                prompt_tokens.push_back(128009); // "<|eot_id|>"
            }

            // "<|start_header_id|>" "user" "<|end_header_id|>" "\n\n"
            prompt_tokens.insert(prompt_tokens.end(), {128006, 882, 128007, 271});

            // otherwise get user prompt from stdin
            std::cout << "User (or exit): ";
            std::string user_prompt;
            getline(std::cin, user_prompt);

            // encode the user prompt into tokens
            auto user_prompt_tokens = tokenizer.encode(user_prompt, 0, 0);
            std::copy(user_prompt_tokens.begin(), user_prompt_tokens.end(), std::back_inserter(prompt_tokens));

            // "<|eot_id|>" "<|start_header_id|>" "assistant" "<|end_header_id|>" "\n\n"
            prompt_tokens.insert(prompt_tokens.end(), {128009, 128006, 78191, 128007, 271});

            ++turn;
            std::cout << "Assistant: ";
        }

        // determine the token to pass into the transformer next
        if (!prompt_tokens.empty())
        {
            // if we are still processing the input prompt, force the next
            // prompt token
            token = prompt_tokens.pop();
        }

        // EOS (=128009) token ends the Assistant turn
        if (prompt_tokens.empty() && (token == 128009 || token == 128001))
            ++turn;

        // forward the transformer to get logits for the next token
        transformer.forward(token, logits);
        token = sampler.sample(logits.f());

        if ((prompt_tokens.empty() && token != 128009) && token != 128001 && token != 128006)
        {
            // the Assistant is responding, so print its output
            if (auto p = tokenizer.decode(token))
                std::cout << *p << std::flush;
        }
        if ((prompt_tokens.empty() && token == 128009) || token == 128001)
            std::cout << std::endl;

        ++steps;
    }
    std::cout << std::endl;
}

