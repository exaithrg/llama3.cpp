cmake_minimum_required(VERSION 3.5.0)
project(llama3 VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_FLAGS "-O3 -Wall -Wextra -Werror -fopenmp -march=native")
set(CMAKE_CXX_FLAGS_DEBUG "-g -Wall -Wextra -Werror -fopenmp -march=native")

add_executable(llama3 Logger.cpp Sampler.cpp Tensor.cpp Tokenizer.cpp Transformer.cpp layers.cpp main.cpp)

set_property(TARGET llama3 PROPERTY CXX_STANDARD 20)
