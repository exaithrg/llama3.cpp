#!/bin/bash

# quit if make fail
set -e

cd ./llama3.cpp/build/
make || exit 1
cd ../..

EXECUTABLE="./llama3.cpp/build/llama3"

ARGS=(
    "./models/Llama3.2-1B.bin"
    "-n" "16"
    "-s" "1234"
    "-z" "./tokenizer/tokenizer.bin"
    "-i" "Once upon a time, there"
    "-d"
    "-l" "DEBUG"
)

# gdb --args "$EXECUTABLE" "${ARGS[@]}" | tee -i debug.log
gdb --args "$EXECUTABLE" "${ARGS[@]}"
