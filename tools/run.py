#!/usr/bin/env python3
"""Cross-platform runner for data prep, build, and CLI execution."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def run_cmd(cmd: list[str], cwd: Path) -> None:
    print("+", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(cwd), check=False)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def detect_cli(build_dir: Path) -> Path:
    exe = "fast_mnist_cli.exe" if os.name == "nt" else "fast_mnist_cli"
    candidates = [
        build_dir / exe,
        build_dir / "Release" / exe,
        build_dir / "Debug" / exe,
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("fast_mnist_cli not found in build dir")


def prepare_data(root: Path, data_root: Path, list_dir: Path) -> None:
    train_dir = data_root / "TrainingSet"
    test_dir = data_root / "TestingSet"
    train_list = list_dir / "TrainingSetList.txt"
    test_list = list_dir / "TestingSetList.txt"
    if train_dir.exists() and test_dir.exists() and train_list.exists():
        if test_list.exists():
            return
    prepare = root / "tools" / "prepare_mnist.py"
    run_cmd(
        [
            sys.executable,
            str(prepare),
            "--output",
            str(data_root),
            "--list-dir",
            str(list_dir),
        ],
        root,
    )


def build_project(root: Path, build_dir: Path, config: str,
                  openmp: bool, native: bool) -> None:
    args = [
        "cmake",
        "-S",
        str(root),
        "-B",
        str(build_dir),
        f"-DCMAKE_BUILD_TYPE={config}",
        f"-DFAST_MNIST_ENABLE_OPENMP={'ON' if openmp else 'OFF'}",
        f"-DFAST_MNIST_ENABLE_NATIVE={'ON' if native else 'OFF'}",
    ]
    run_cmd(args, root)
    build_args = ["cmake", "--build", str(build_dir)]
    if os.name == "nt":
        build_args.extend(["--config", config])
    run_cmd(build_args, root)


def run_cli(cli: Path, cwd: Path, data_root: Path, train_count: int,
            epochs: int, train_list: str, test_list: str) -> None:
    cmd = [
        str(cli),
        str(data_root),
        str(train_count),
        str(epochs),
        train_list,
        test_list,
    ]
    run_cmd(cmd, cwd)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare, build, run.")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--build-dir", default="build")
    parser.add_argument("--config", default="Release")
    parser.add_argument("--openmp", action="store_true")
    parser.add_argument("--native", action="store_true")
    parser.add_argument("--train-count", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--train-list", default="TrainingSetList.txt")
    parser.add_argument("--test-list", default="TestingSetList.txt")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = repo_root()
    data_root = (root / args.data_root).resolve()
    list_dir = root
    build_dir = (root / args.build_dir).resolve()

    explicit = args.prepare or args.build or args.run
    if explicit:
        do_prepare = args.prepare
        do_build = args.build
        do_run = args.run
    else:
        do_prepare = True
        do_build = True
        do_run = True

    if do_prepare:
        prepare_data(root, data_root, list_dir)
    if do_build:
        build_project(root, build_dir, args.config, args.openmp, args.native)
    if do_run:
        cli = detect_cli(build_dir)
        run_cli(
            cli,
            root,
            data_root,
            args.train_count,
            args.epochs,
            args.train_list,
            args.test_list,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
