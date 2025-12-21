<p align="center">
  <img src="docs/branding/readme-light.svg#gh-light-mode-only" width="700"
       alt="Fast MNIST NN">
  <img src="docs/branding/readme-dark.svg#gh-dark-mode-only" width="700"
       alt="Fast MNIST NN">
</p>

---

[![ci][ci-badge]][ci-url]
[![license][license-badge]][license-url]
[![c++][cpp-badge]][cpp-url]

High-performance C++ neural network for MNIST digit recognition with
SIMD kernels, OpenMP, and reproducible benchmarks.

## Highlights

- SIMD-accelerated matrix ops (AVX2/AVX-512/NEON) with aligned storage.
- OpenMP-aware hot paths for dot, transpose, and axpy.
- P2 PGM parser with in-memory + on-disk cache for repeat runs.
- CLI training + evaluation pipeline with configurable epochs.
- Catch2 tests wired to CTest.
- Google Benchmark suite with published results + charts.
- Doxygen docs target and clang-format config.
- CI on Linux/macOS/Windows via GitHub Actions.

## Quickstart

```sh
python3 tools/run.py
```

This downloads MNIST, builds the project, and runs a training pass.
Use `python3 tools/run.py --help` for flags.

## Benchmarks

Run file: `docs/benchmarks/runs/bench-20251221-132025.json`

- Apple M2, macOS 15.5
- Apple clang 17.0.0
- Release (`-O3`, `-march=native`, OpenMP on)

Matrix ops (ns/op):

| Case | ns/op |
| --- | --- |
| dot 64 | `96972` |
| dot 128 | `379712` |
| transpose 256 | `28198` |
| transpose 512 | `101070` |
| axpy 256 | `26254` |
| axpy 512 | `36403` |

Training/inference throughput:

| Case | Images/sec |
| --- | --- |
| learn step | `45821` |
| classify | `79473` |

![Matrix benchmark chart][matrix-light]
![Matrix benchmark chart][matrix-dark]

![Throughput chart][throughput-light]
![Throughput chart][throughput-dark]

See `docs/benchmarks/benchmarks.md` for methodology and scripts.

### Run Benchmarks

```sh
python3 tools/run_benchmarks.py --openmp --native
```

## Build and Test

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build
```

macOS quickstart:

```sh
./tools/bootstrap_macos.sh
```

## Run

```sh
./build/fast_mnist_cli data 5000 10 TrainingSetList.txt TestingSetList.txt
```

## Formatting

```sh
clang-format -i src/*.cpp include/fast_mnist/*.h apps/*.cpp
```

## Documentation

```sh
cmake -S . -B build -DFAST_MNIST_ENABLE_DOXYGEN=ON
cmake --build build --target docs
```

## Data

```sh
python3 tools/prepare_mnist.py --output data --list-dir .
```

The script auto-installs `tqdm` for progress bars; pass
`--no-auto-install` to skip that step.

## License

MIT -- see `LICENSE`.

[ci-badge]: https://github.com/ShreeChaturvedi/fast-mnist-nn/actions/workflows/ci.yml/badge.svg
[ci-url]: https://github.com/ShreeChaturvedi/fast-mnist-nn/actions/workflows/ci.yml
[license-badge]: https://img.shields.io/badge/license-MIT-blue.svg
[license-url]: LICENSE
[cpp-badge]: https://img.shields.io/badge/C%2B%2B-17-blue.svg
[cpp-url]: https://isocpp.org/
[matrix-light]: docs/benchmarks/charts/matrix-light.svg#gh-light-mode-only
[matrix-dark]: docs/benchmarks/charts/matrix-dark.svg#gh-dark-mode-only
[throughput-light]: docs/benchmarks/charts/throughput-light.svg#gh-light-mode-only
[throughput-dark]: docs/benchmarks/charts/throughput-dark.svg#gh-dark-mode-only
