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

Run files:
- `docs/benchmarks/runs/bench-20251226-154121-baseline.json`
- `docs/benchmarks/runs/bench-20251226-154121-native.json`
- `docs/benchmarks/runs/bench-20251226-154121-openmp-native.json`

Configs:
- baseline: OpenMP off, native off
- native: OpenMP off, native on
- openmp+native: OpenMP on, native on

Environment:
- Apple M2, macOS 15.5
- Apple clang 17.0.0
- Release (`-O3`, OpenMP on/off, `-march=native` on/off)

Matrix ops (ns/op, lower is better):

| Case | baseline | native | openmp+native |
| --- | --- | --- | --- |
| dot 32 | `6165` | `6229` | `6287` |
| dot 64 | `65252` | `57222` | `89130` |
| dot 128 | `575281` | `587767` | `374400` |
| dot 256 | `4835360` | `4759132` | `1379835` |
| transpose 128 | `5441` | `5292` | `23662` |
| transpose 256 | `23098` | `22104` | `31108` |
| transpose 512 | `198735` | `178676` | `87914` |
| transpose 1024 | `978383` | `861078` | `502426` |
| axpy 128 | `3486` | `3477` | `23917` |
| axpy 256 | `13886` | `13896` | `26335` |
| axpy 512 | `55848` | `55441` | `35846` |
| axpy 1024 | `230626` | `229230` | `114910` |

Training/inference throughput (img/s, higher is better):

| Case | baseline | native | openmp+native |
| --- | --- | --- | --- |
| learn step | `48755` | `49399` | `48636` |
| classify | `81628` | `80712` | `69994` |

OpenMP overhead shows up on smaller sizes; the line charts illustrate where
parallelism starts to pay off.

<p align="center">
  <img src="docs/benchmarks/charts/dot-light.svg#gh-light-mode-only" width="760"
       alt="Dot scaling">
  <img src="docs/benchmarks/charts/dot-dark.svg#gh-dark-mode-only" width="760"
       alt="Dot scaling">
</p>

<p align="center">
  <img src="docs/benchmarks/charts/transpose-light.svg#gh-light-mode-only"
       width="760" alt="Transpose scaling">
  <img src="docs/benchmarks/charts/transpose-dark.svg#gh-dark-mode-only"
       width="760" alt="Transpose scaling">
</p>

<p align="center">
  <img src="docs/benchmarks/charts/axpy-light.svg#gh-light-mode-only" width="760"
       alt="Axpy scaling">
  <img src="docs/benchmarks/charts/axpy-dark.svg#gh-dark-mode-only" width="760"
       alt="Axpy scaling">
</p>

<p align="center">
  <img src="docs/benchmarks/charts/throughput-compare-light.svg#gh-light-mode-only"
       width="760" alt="Throughput comparison">
  <img src="docs/benchmarks/charts/throughput-compare-dark.svg#gh-dark-mode-only"
       width="760" alt="Throughput comparison">
</p>

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
