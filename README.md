# Fast MNIST NN

High-performance C++ neural network for MNIST digit recognition.

## Build

```sh
g++ -g -Wall -std=c++17 -O3 -march=native \
  Matrix.cpp NeuralNet.cpp main.cpp -o fast_mnist_nn
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

## Data

The MNIST PGM files are not included in this repository. The
`TrainingSetList.txt` and `TestingSetList.txt` files expect relative
paths like `TrainingSet/digit_10000_7.pgm` under your data root.
