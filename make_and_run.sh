#!/bin/bash

# quit if make fail
set -e

cd ./llama3.cpp/build/
make || exit 1
cd ../..

./llama3.cpp/build/llama3 ./models/Llama3.2-1B.bin -n 16 -s 1234 -z "./tokenizer/tokenizer.bin" -i "Welcome to the most" -d -l "DEBUG"

# ./llama3 ./models/Llama3.1-8B-q80.bin -i "<HERE GOES THE PROMPT>"

# ./llama3.cpp/build/llama3 ./models/Llama3.2-1B.bin -n 32 -s 6515 -z "./tokenizer/tokenizer.bin" -i "I think Lava(?) is an accessible, open large language model (LLM) designed for"

# debug
# ./llama3.cpp/build/llama3 ./models/Llama3.2-1B.bin -n 56 -s 6515 -z "./tokenizer/tokenizer.bin" -i "I think Lava(?) is an accessible, open large language model (LLM) designed for" -d | tee -i run.log

# ./llama3.cpp/build/llama3 ./models/Llama3.2-1B.bin -n 16 -s 1234 -z "./tokenizer/tokenizer.bin" -i "Welcome to the most" -d | tee -i run.log

# ./llama3.cpp/build/llama3 ./models/stories15M.bin -i "Once upon a time, in a bright forest, little Bunny found a magical flower that"

# ./llama3.cpp/build/llama3 ./models/stories42M.bin -i "Once upon a time, in a bright forest, little Bunny found a magical flower that"

# compeletely failed
# ./llama3.cpp/build/llama3 ./models/Llama3.2-1B.bin -z "./tokenizer/tokenizer.bin" -m chat
