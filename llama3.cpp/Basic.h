#pragma once

struct MyArgs : public argparse::Args
{
    std::string &checkpoint_path = arg("checkpoint", "Model checkpoint");

    float &temperature = kwarg("t", "temperature in [0,inf], default 1.0").set_default(1.0f);
    float &topP = kwarg("p", "p value in top-p (nucleus) sampling in [0,1]").set_default(0.9f);
    int &rngSeed = kwarg("s", "random seed, default time(NULL)").set_default(static_cast<unsigned int>(time(NULL)));
    int &steps = kwarg("n", "number of steps to run for, default 128. 0 = infinite").set_default(128);
    std::string &prompt = kwarg("i", "input prompt").set_default("");
    std::string &tokenizerPath = kwarg("z", "optional path to custom tokenizer").set_default("tokenizer.bin");
    std::string &mode = kwarg("m", "mode: generate|chat, default: generate").set_default("generate");
    std::string &systemPrompt = kwarg("y", "(optional) system prompt in chat mode").set_default("");
    bool &debug = flag("d", "debug");
};

auto time_in_ms();
void check_header(std::ifstream &inputStream);
Transformer build_transformer(std::string const &checkpoint_path);

void generate(Transformer &transformer, Tokenizer const &tokenizer, Sampler &sampler, std::string const &prompt, size_t numSteps);

void chat(Transformer &transformer, Tokenizer const &tokenizer, Sampler &sampler, std::string system_prompt, size_t numSteps);
