# Repository Guidelines

## Project Structure & Module Organization
- `include/fast_mnist/` holds public headers (`Matrix.h`, `NeuralNet.h`).
- `src/` contains library implementation (`Matrix.cpp`, `NeuralNet.cpp`).
- `apps/fast_mnist_cli.cpp` is the CLI entry point used for training
  and evaluation.
- `tests/` contains Catch2 unit tests.
- `TrainingSetList.txt` and `TestingSetList.txt` are generated locally
  by `tools/prepare_mnist.py` and store relative PGM paths (for example,
  `TrainingSet/digit_10000_7.pgm`).

## Build, Test, and Development Commands
Recommended CMake workflow:

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

Run the CLI:

```sh
./build/fast_mnist_cli data 5000 10 TrainingSetList.txt TestingSetList.txt
```

Tests (Catch2 via CTest):

```sh
cmake -S . -B build -DBUILD_TESTING=ON
cmake --build build
ctest --test-dir build
```

## Coding Style & Naming Conventions
- 4-space indentation; keep lines at or below 80 characters.
- Prefer `const` correctness and avoid hidden globals.
- Use Doxygen-style block comments for public APIs.
- Types use `PascalCase` (`Matrix`, `NeuralNet`); functions and locals
  use `camelCase` (`loadPGM`, `maxElemIndexCol`).

## Testing Guidelines
- Use Catch2 for unit tests in `tests/` and keep them focused and fast.
- Name tests by capability (for example, `Matrix dot multiplies`).
- Add coverage for new math kernels and data loading changes.

## Commit & Pull Request Guidelines
- Use short, imperative commit subjects (for example, "Add NEON kernel").
- PRs should include a brief summary, key commands run, and relevant
  performance notes or benchmark outputs when applicable.
