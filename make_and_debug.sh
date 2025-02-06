#!/bin/bash

cd ./llama3.cpp/build/
make
cd ../..

EXECUTABLE="./llama3.cpp/build/llama3"

ARGS=(
    "./models/Llama3.2-1B.bin"
    "-n" "16"
    "-s" "1234"
    "-z" "./tokenizer/tokenizer.bin"
    "-i" "Welcome to the most"
    "-d"
)

# gdb --args "$EXECUTABLE" "${ARGS[@]}" | tee -i debug.log
gdb --args "$EXECUTABLE" "${ARGS[@]}"
