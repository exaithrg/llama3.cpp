# README

This is a modification of [llama3.cpp](https://github.com/jonemeth/llama3.cpp) by [jonemeth](https://github.com/jonemeth).

## How to run:

Make sure that `tokenizer.bin` is in folder `./tokenizer`/ and sth like `Llama3.2-1B.bin` is in folder `./models/`, then

```bash
bash rebuild_debug.sh
bash make_and_run.sh
```

you will see sth like:

```
➜ sh rebuild_debug.sh:
-- The C compiler identification is GNU 11.4.0
-- The CXX compiler identification is GNU 11.4.0
-- Detecting C compiler ABI info
-- Detecting C compiler ABI info - done
-- Check for working C compiler: /usr/bin/cc - skipped
-- Detecting C compile features
-- Detecting C compile features - done
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Check for working CXX compiler: /usr/bin/c++ - skipped
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- Configuring done (9.0s)
-- Generating done (0.0s)
-- Build files have been written to: /home/geng/github/exaithrg/llama3.cpp/llama3.cpp/build
[ 11%] Building CXX object CMakeFiles/llama3.dir/Logger.cpp.o
[ 22%] Building CXX object CMakeFiles/llama3.dir/Sampler.cpp.o
[ 33%] Building CXX object CMakeFiles/llama3.dir/Tensor.cpp.o
[ 44%] Building CXX object CMakeFiles/llama3.dir/Tokenizer.cpp.o
[ 55%] Building CXX object CMakeFiles/llama3.dir/Transformer.cpp.o
[ 66%] Building CXX object CMakeFiles/llama3.dir/Layers.cpp.o
[ 77%] Building CXX object CMakeFiles/llama3.dir/Basic.cpp.o
[ 88%] Building CXX object CMakeFiles/llama3.dir/main.cpp.o
[100%] Linking CXX executable llama3
[100%] Built target llama3

➜ sh make_and_run.sh
[100%] Built target llama3
==INFO== --------------------------------------------------------
==INFO== Building the Transformer via model: ./models/Llama3.2-1B.bin ...
==INFO== Building the Transformer via tokenizer: ./tokenizer/tokenizer.bin ...
==INFO== --------------------------------------------------------
==INFO== Model building ok, generation start...
==INFO== --------------------------------------------------------
==INFO== GENERATION LOOP
==INFO== --------------------------------------------------------
Once upon a time, there was a man who lived in a big city, and he was very busy with his work. He had to work very hard,
==INFO== --------------------------------------------------------
==INFO== GENERATION DONE
```



## What changes we made:

1. [OK] automate cmake/run/debug flow with shell scripts
2. [OK] add more debug infos
3. [DROPPED] add prefill stage to class Transformer
4. [WIP] delete int8 implementation
5. [WIP] rewrite Tensor class with c++ template 



# Original README by jonemeth

https://github.com/jonemeth/llama3.cpp

# llama3.cpp

Llama3 inference in pure C++.

This is a C++ port of [llama3.c](https://github.com/jameswdelancey/llama3.c) by [James Delancey](https://github.com/jameswdelancey), which is a modified version of [llama2.c](https://github.com/karpathy/llama2.c): by [Andrej Karpathy](https://github.com/karpathy).

Unlike the single-file C implementation, here the source code is split into multiple source files. Algorithms from the standard C++ library have been applied in many cases, but most of the algorithms and methods can be identified with their counterpart in the original C version. The original comments have also been kept unchanged where it was appropriate.

However, in the case of llama3.cpp, the same executable can be used with both float32 and int8 weights: the `Tensor` class automatically converts between float and quantized values, while the `Linear` layer executes quantized matrix multiplication when its weight tensor contains quantized values. Note that it required to change the order in which `export.py` exports the model weights.

llama3.cpp uses the [argparse](https://github.com/morrisfranken/argparse) library by [Morris Franken](https://github.com/morrisfranken).


## Usage
The code is written in C++20 and tested under Linux.

### Build
After cloning the repository, the executable can be compiled either using `cmake` or by:
```
g++ -std=c++20 -Wall -Wextra -Werror -Ofast -fopenmp -march=native *.cpp -o llama3
```

### Convert models

The first step is to download one of the Llama3 models by following the instructions on the [Llama repository](https://github.com/meta-llama/llama-models), or by cloning the [huggingface]()https://huggingface.co/meta-llama Llama repositories, e.g.:
```
git clone https://huggingface.co/meta-llama/Llama-3.2-1B
```

To use the Llama3 checkpoints, they need to be converted into a format that can be used with llama3.cpp. To export float32 weights using:

```
mkdir models
python3 export.py ./models/Llama3.2-1B.bin --meta-llama <PATH-TO>/Llama3.2-1B/
```

or quantized weights with the `--quantize` flag:
```
python3 export.py ./models/Llama3.2-1B-q80.bin --meta-llama <PATH-TO/Llama3.2-1B/ --quantize
```

To convert models downloaded from huggingface use the `--hf` argument instead of `--meta-llama`.

The following models have been tried:
 * Llama3.1-8B and Llama3.1-8B-Instruct
 * Llama3.2-1B and Llama3.2-1B-Instruct
 * Llama3.2-3B and Llama3.2-3B-Instruct

Converting a 8B model with quantization required ~45G memory (e.g.: 16G RAM and 48G swap) and took 1.5 hours.

### Run llama3
The compiled executable can be used with both float32 and quantized models, for example:
```
./llama3 ./models/Llama3.1-8B-q80.bin -i "<HERE GOES THE PROMPT>"
```

For chat mode set the `-m chat` argument, and use an "Instruct" model:
```
./llama3 ./models/Llama3.1-8B-Instruct-q80.bin -m chat
```



# END

