# Fast MNIST NN

High-performance C++ neural network for MNIST digit recognition.

## Build

### CMake (recommended)

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

Optional flags:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DFAST_MNIST_ENABLE_OPENMP=ON \
  -DFAST_MNIST_ENABLE_NATIVE=ON
```

### Manual compile

```sh
g++ -g -Wall -std=c++17 -O3 -march=native -Iinclude \
  src/Matrix.cpp src/NeuralNet.cpp apps/fast_mnist_cli.cpp -o fast_mnist_nn
```

## Run

```sh
./fast_mnist_nn <data_root> [train_count] [epochs] \
  [train_list] [test_list]
```

Example:

```sh
./fast_mnist_nn data 5000 10 TrainingSetList.txt TestingSetList.txt
```

## Tests

```sh
cmake -S . -B build -DBUILD_TESTING=ON
cmake --build build
ctest --test-dir build
```

## Data

The MNIST PGM files are not included in this repository. The
`TrainingSetList.txt` and `TestingSetList.txt` files expect relative
paths like `TrainingSet/digit_10000_7.pgm` under your data root.
