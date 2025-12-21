#include <benchmark/benchmark.h>

#include "fast_mnist/Matrix.h"
#include "fast_mnist/NeuralNet.h"

#include <random>

namespace {

Matrix makeMatrix(std::size_t rows, std::size_t cols, double value) {
    Matrix m(rows, cols, 0.0);
    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t c = 0; c < cols; ++c) {
            m[r][c] = value;
        }
    }
    return m;
}

Matrix makeInput(std::size_t rows) {
    Matrix input(rows, 1, Matrix::NoInit{});
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    for (std::size_t i = 0; i < rows; ++i) {
        input[i][0] = dist(rng);
    }
    return input;
}

Matrix makeExpected(int label) {
    Matrix expected(10, 1, 0.0);
    expected[label][0] = 1.0;
    return expected;
}

void benchDot(benchmark::State& state) {
    const std::size_t n = static_cast<std::size_t>(state.range(0));
    Matrix a = makeMatrix(n, n, 1.0);
    Matrix b = makeMatrix(n, n, 1.0);

    for (auto _ : state) {
        Matrix c = a.dot(b);
        benchmark::DoNotOptimize(c[0][0]);
    }
    state.SetItemsProcessed(state.iterations() * n * n * n);
}

void benchTranspose(benchmark::State& state) {
    const std::size_t n = static_cast<std::size_t>(state.range(0));
    Matrix a = makeMatrix(n, n, 1.0);

    for (auto _ : state) {
        Matrix t = a.transpose();
        benchmark::DoNotOptimize(t[0][0]);
    }
    state.SetItemsProcessed(state.iterations() * n * n);
}

void benchAxpy(benchmark::State& state) {
    const std::size_t n = static_cast<std::size_t>(state.range(0));
    Matrix a = makeMatrix(n, n, 1.0);
    Matrix b = makeMatrix(n, n, 2.0);

    for (auto _ : state) {
        a.axpy(0.5, b);
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * n * n);
}

void benchLearn(benchmark::State& state) {
    NeuralNet net({784, 30, 10});
    Matrix input = makeInput(784);
    Matrix expected = makeExpected(3);

    for (auto _ : state) {
        net.learn(input, expected, 0.3);
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations());
}

void benchClassify(benchmark::State& state) {
    NeuralNet net({784, 30, 10});
    Matrix input = makeInput(784);

    for (auto _ : state) {
        Matrix out = net.classify(input);
        benchmark::DoNotOptimize(out[0][0]);
    }
    state.SetItemsProcessed(state.iterations());
}

} // namespace

BENCHMARK(benchDot)->Arg(64)->Arg(128);
BENCHMARK(benchTranspose)->Arg(256)->Arg(512);
BENCHMARK(benchAxpy)->Arg(256)->Arg(512);
BENCHMARK(benchLearn);
BENCHMARK(benchClassify);

BENCHMARK_MAIN();
