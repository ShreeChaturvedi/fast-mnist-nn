## Benchmark Methodology

Benchmarks are run with Google Benchmark in `benchmarks/bench_matrix.cpp`.
Results are recorded as JSON and summarized into CSV for README tables.

### Run

```sh
python3 tools/run_benchmarks.py --openmp --native
```

### Notes

- `benchDot`, `benchTranspose`, `benchAxpy` use square matrices.
- `benchLearn` and `benchClassify` operate on a 784-30-10 network with
  deterministic input values.
- Results are machine-specific; see `docs/benchmarks/bench_env.md`.
