#!/usr/bin/env python3
"""Build and run benchmarks, then write summaries and charts."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def run_cmd(cmd: list[str], cwd: Path) -> None:
    print("+", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(cwd), check=False)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def time_unit_scale(unit: str) -> float:
    return {"ns": 1e-9, "us": 1e-6, "ms": 1e-3, "s": 1.0}.get(unit, 1e-9)


def write_svg_chart(
    items: list[tuple[str, float]],
    out_path: Path,
    title: str,
    unit: str,
    theme: str,
) -> None:
    width = 820
    height = 360
    margin_left = 140
    margin_right = 20
    margin_top = 50
    margin_bottom = 60
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    max_val = max(value for _, value in items) if items else 1.0
    bar_w = plot_w / max(len(items), 1)

    if theme == "dark":
        text = "#e5e7eb"
        grid = "#374151"
        bar = "#60a5fa"
    else:
        text = "#111827"
        grid = "#e5e7eb"
        bar = "#3b82f6"

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">',
        f'<text x="{margin_left}" y="28" fill="{text}" '
        'font-family="Avenir Next, Avenir, Segoe UI, Helvetica, Arial, '
        'sans-serif" font-size="18" font-weight="600">'
        f'{title}</text>',
    ]

    for i in range(5):
        y = margin_top + plot_h * i / 4
        lines.append(
            f'<line x1="{margin_left}" y1="{y:.1f}" '
            f'x2="{width - margin_right}" y2="{y:.1f}" '
            f'stroke="{grid}" stroke-width="1" />'
        )

    for i, (label, value) in enumerate(items):
        bar_h = (value / max_val) * plot_h
        x = margin_left + i * bar_w + bar_w * 0.15
        y = margin_top + (plot_h - bar_h)
        w = bar_w * 0.7
        lines.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" '
            f'height="{bar_h:.1f}" fill="{bar}" rx="4" />'
        )
        lines.append(
            f'<text x="{x + w / 2:.1f}" y="{margin_top + plot_h + 22}" '
            f'fill="{text}" text-anchor="middle" font-size="11" '
            'font-family="Avenir Next, Avenir, Segoe UI, Helvetica, Arial, '
            f'sans-serif">{label}</text>'
        )
        lines.append(
            f'<text x="{x + w / 2:.1f}" y="{y - 6:.1f}" '
            f'fill="{text}" text-anchor="middle" font-size="11" '
            'font-family="Avenir Next, Avenir, Segoe UI, Helvetica, Arial, '
            f'sans-serif">{value:.2f}</text>'
        )

    lines.append(
        f'<text x="{margin_left}" y="{height - 18}" fill="{text}" '
        'font-family="Avenir Next, Avenir, Segoe UI, Helvetica, Arial, '
        f'sans-serif" font-size="12">{unit}</text>'
    )
    lines.append("</svg>")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def parse_benchmarks(path: Path) -> dict[str, dict[str, float | str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    results: dict[str, dict[str, float | str]] = {}
    for entry in data.get("benchmarks", []):
        if "aggregate_name" in entry:
            continue
        name = entry.get("name", "")
        if not name:
            continue
        results[name] = {
            "real_time": float(entry.get("real_time", 0.0)),
            "cpu_time": float(entry.get("cpu_time", 0.0)),
            "time_unit": str(entry.get("time_unit", "ns")),
        }
    return results


def write_summary_csv(
    out_path: Path,
    rows: Iterable[tuple[str, str, float, str]],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["benchmark", "metric", "value", "unit"])
        for row in rows:
            writer.writerow(row)


def compiler_version() -> str:
    try:
        result = subprocess.run(
            ["c++", "--version"],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return "unknown"
    line = (result.stdout or result.stderr).splitlines()
    return line[0].strip() if line else "unknown"


def write_env(path: Path, run_id: str, openmp: bool, native: bool) -> None:
    lines = [
        f"Run: {run_id}",
        f"OS: {platform.platform()}",
        f"Arch: {platform.machine()}",
        f"CPU: {platform.processor() or 'unknown'}",
        f"Compiler: {compiler_version()}",
        f"OpenMP: {'on' if openmp else 'off'}",
        f"Native: {'on' if native else 'off'}",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run benchmarks + charts.")
    parser.add_argument("--build-dir", default="build-bench")
    parser.add_argument("--openmp", action="store_true")
    parser.add_argument("--native", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = repo_root()
    build_dir = root / args.build_dir

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    runs_dir = root / "docs" / "benchmarks" / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    run_path = runs_dir / f"bench-{run_id}.json"

    run_cmd(
        [
            "cmake",
            "-S",
            str(root),
            "-B",
            str(build_dir),
            "-DCMAKE_BUILD_TYPE=Release",
            "-DFAST_MNIST_ENABLE_BENCHMARKS=ON",
            "-DBUILD_TESTING=OFF",
            f"-DFAST_MNIST_ENABLE_OPENMP={'ON' if args.openmp else 'OFF'}",
            f"-DFAST_MNIST_ENABLE_NATIVE={'ON' if args.native else 'OFF'}",
        ],
        root,
    )
    run_cmd(["cmake", "--build", str(build_dir)], root)

    bench_bin = build_dir / ("fast_mnist_benchmarks.exe"
                             if sys.platform == "win32"
                             else "fast_mnist_benchmarks")
    run_cmd(
        [
            str(bench_bin),
            f"--benchmark_out={run_path}",
            "--benchmark_out_format=json",
        ],
        root,
    )

    results = parse_benchmarks(run_path)
    summary_rows: list[tuple[str, str, float, str]] = []

    matrix_items: list[tuple[str, float]] = []
    for name in [
        "benchDot/64",
        "benchDot/128",
        "benchTranspose/256",
        "benchTranspose/512",
        "benchAxpy/256",
        "benchAxpy/512",
    ]:
        if name not in results:
            continue
        data = results[name]
        time_ns = float(data["real_time"])
        summary_rows.append((name, "time", time_ns, "ns/op"))
        label = name.replace("bench", "").replace("/", " ")
        matrix_items.append((label, time_ns))

    throughput_items: list[tuple[str, float]] = []
    for name in ["benchLearn", "benchClassify"]:
        if name not in results:
            continue
        data = results[name]
        unit = str(data["time_unit"])
        seconds = float(data["real_time"]) * time_unit_scale(unit)
        images_per_sec = (1.0 / seconds) if seconds > 0 else 0.0
        summary_rows.append((name, "throughput", images_per_sec, "img/s"))
        label = name.replace("bench", "").lower()
        throughput_items.append((label, images_per_sec))

    summary_path = root / "docs" / "benchmarks" / "bench_summary.csv"
    write_summary_csv(summary_path, summary_rows)

    charts_dir = root / "docs" / "benchmarks" / "charts"
    write_svg_chart(
        matrix_items,
        charts_dir / "matrix-light.svg",
        "Matrix ops (ns/op)",
        "Lower is better",
        "light",
    )
    write_svg_chart(
        matrix_items,
        charts_dir / "matrix-dark.svg",
        "Matrix ops (ns/op)",
        "Lower is better",
        "dark",
    )
    write_svg_chart(
        throughput_items,
        charts_dir / "throughput-light.svg",
        "Training/inference throughput",
        "Higher is better (img/s)",
        "light",
    )
    write_svg_chart(
        throughput_items,
        charts_dir / "throughput-dark.svg",
        "Training/inference throughput",
        "Higher is better (img/s)",
        "dark",
    )

    env_path = root / "docs" / "benchmarks" / "bench_env.md"
    write_env(env_path, run_id, args.openmp, args.native)
    print(f"Wrote {run_path}")
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
