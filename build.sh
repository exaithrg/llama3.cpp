#!/bin/bash

rm -rf ./llama3.cpp/build/
mkdir -p ./llama3.cpp/build/
cd ./llama3.cpp/build/
cmake ..
make
