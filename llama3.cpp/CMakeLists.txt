cmake_minimum_required(VERSION 3.5.0)
project(llama3 VERSION 0.1.0 LANGUAGES C CXX)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS "-g -Wall -Wextra -Werror -fopenmp -march=native")
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CXX_FLAGS "-O3 -Wall -Wextra -Werror -fopenmp -march=native")
endif()

add_executable(llama3 
    Logger.cpp 
    Sampler.cpp 
    Tensor.cpp 
    Tokenizer.cpp 
    Transformer.cpp 
    Layers.cpp 
    Basic.cpp 
    main.cpp
)

set_property(TARGET llama3 PROPERTY CXX_STANDARD 20)
