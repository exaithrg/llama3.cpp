# llama3.cpp

toolchain:

```
cmake g++
```

How to run:

```bash
bash rebuild_debug.sh
bash make_and_run.sh
```

reference console output:

```bash
➜  llama3.cpp git:(submit250212) sh rebuild_debug.sh
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
➜  llama3.cpp git:(submit250212) sh make_and_run.sh
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



# END

