#!/bin/bash

# commands
# sh rebuild_debug.sh
# sh remake.sh
# sh debug.sh
# In GDB:
# info sources
# break Basic.cpp:66


EXECUTABLE="./llama3.cpp/build/llama3"

ARGS=(
    "./models/Llama3.2-1B.bin"
    "-n" "56"
    "-s" "6515"
    "-z" "./tokenizer/tokenizer.bin"
    "-i" "I think Lava(?) is an accessible, open large language model (LLM) designed for"
    "-d"
)

# gdb --args "$EXECUTABLE" "${ARGS[@]}" | tee -i debug.log
gdb --args "$EXECUTABLE" "${ARGS[@]}"
