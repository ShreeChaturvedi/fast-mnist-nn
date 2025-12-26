#!/usr/bin/env python3
"""Build and run benchmarks, then write summaries and charts."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
from dataclasses import dataclass
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


def time_unit_to_ns(unit: str) -> float:
    return {"ns": 1.0, "us": 1e3, "ms": 1e6, "s": 1e9}.get(unit, 1.0)


def format_compact(value: float) -> str:
    if value >= 1e9:
        return f"{value / 1e9:.1f}G"
    if value >= 1e6:
        return f"{value / 1e6:.1f}M"
    if value >= 1e3:
        return f"{value / 1e3:.1f}k"
    return f"{value:.0f}"


@dataclass(frozen=True)
class BenchConfig:
    name: str
    label: str
    openmp: bool
    native: bool


def write_svg_line_chart(
    series: list[tuple[str, list[tuple[int, float]]]],
    out_path: Path,
    title: str,
    unit: str,
    theme: str,
) -> None:
    width = 820
    height = 360
    margin_left = 80
    margin_right = 30
    margin_top = 50
    margin_bottom = 60
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    sizes = sorted({size for _, points in series for size, _ in points})
    max_val = max(
        (value for _, points in series for _, value in points),
        default=1.0,
    )
    max_val *= 1.1
    if max_val <= 0:
        max_val = 1.0

    if theme == "dark":
        text = "#e5e7eb"
        grid = "#374151"
        colors = ["#9ca3af", "#60a5fa", "#f59e0b"]
    else:
        text = "#111827"
        grid = "#e5e7eb"
        colors = ["#94a3b8", "#2563eb", "#f97316"]

    if sizes:
        if len(sizes) == 1:
            x_positions = {sizes[0]: margin_left + plot_w / 2}
        else:
            x_positions = {
                size: margin_left + (plot_w * idx / (len(sizes) - 1))
                for idx, size in enumerate(sizes)
            }
    else:
        x_positions = {}

    def x_for(size: int) -> float:
        return x_positions.get(size, margin_left)

    def y_for(value: float) -> float:
        return margin_top + plot_h * (1.0 - value / max_val)

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
        value = max_val * (1.0 - i / 4)
        lines.append(
            f'<line x1="{margin_left}" y1="{y:.1f}" '
            f'x2="{width - margin_right}" y2="{y:.1f}" '
            f'stroke="{grid}" stroke-width="1" />'
        )
        lines.append(
            f'<text x="{margin_left - 10}" y="{y + 4:.1f}" '
            f'fill="{text}" text-anchor="end" font-size="11" '
            'font-family="Avenir Next, Avenir, Segoe UI, Helvetica, Arial, '
            f'sans-serif">{format_compact(value)}</text>'
        )

    for size in sizes:
        x = x_for(size)
        lines.append(
            f'<text x="{x:.1f}" y="{margin_top + plot_h + 22}" '
            f'fill="{text}" text-anchor="middle" font-size="11" '
            'font-family="Avenir Next, Avenir, Segoe UI, Helvetica, Arial, '
            f'sans-serif">{size}</text>'
        )

    for idx, (label, points) in enumerate(series):
        color = colors[idx % len(colors)]
        sorted_points = sorted(points, key=lambda item: item[0])
        if not sorted_points:
            continue
        poly_points = " ".join(
            f"{x_for(size):.1f},{y_for(value):.1f}"
            for size, value in sorted_points
        )
        lines.append(
            f'<polyline fill="none" stroke="{color}" '
            f'stroke-width="2.5" points="{poly_points}" />'
        )
        for size, value in sorted_points:
            lines.append(
                f'<circle cx="{x_for(size):.1f}" cy="{y_for(value):.1f}" '
                f'r="4" fill="{color}" />'
            )

    legend_x = width - margin_right - 160
    legend_y = margin_top + 6
    for idx, (label, _) in enumerate(series):
        color = colors[idx % len(colors)]
        y = legend_y + idx * 18
        lines.append(
            f'<rect x="{legend_x}" y="{y - 10}" width="12" height="12" '
            f'fill="{color}" rx="2" />'
        )
        lines.append(
            f'<text x="{legend_x + 18}" y="{y}" fill="{text}" '
            'font-size="11" font-family="Avenir Next, Avenir, Segoe UI, '
            f'Helvetica, Arial, sans-serif">{label}</text>'
        )

    lines.append(
        f'<text x="{margin_left}" y="{height - 18}" fill="{text}" '
        'font-family="Avenir Next, Avenir, Segoe UI, Helvetica, Arial, '
        f'sans-serif" font-size="12">{unit}</text>'
    )
    lines.append("</svg>")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_svg_grouped_bar_chart(
    groups: list[str],
    series: list[tuple[str, list[float]]],
    out_path: Path,
    title: str,
    unit: str,
    theme: str,
) -> None:
    width = 820
    height = 360
    margin_left = 110
    margin_right = 30
    margin_top = 50
    margin_bottom = 60
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    max_val = max((val for _, values in series for val in values), default=1.0)
    max_val *= 1.1
    if max_val <= 0:
        max_val = 1.0
    group_w = plot_w / max(len(groups), 1)
    series_count = max(len(series), 1)

    if theme == "dark":
        text = "#e5e7eb"
        grid = "#374151"
        colors = ["#9ca3af", "#60a5fa", "#f59e0b"]
    else:
        text = "#111827"
        grid = "#e5e7eb"
        colors = ["#94a3b8", "#2563eb", "#f97316"]

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
        value = max_val * (1.0 - i / 4)
        lines.append(
            f'<line x1="{margin_left}" y1="{y:.1f}" '
            f'x2="{width - margin_right}" y2="{y:.1f}" '
            f'stroke="{grid}" stroke-width="1" />'
        )
        lines.append(
            f'<text x="{margin_left - 10}" y="{y + 4:.1f}" '
            f'fill="{text}" text-anchor="end" font-size="11" '
            'font-family="Avenir Next, Avenir, Segoe UI, Helvetica, Arial, '
            f'sans-serif">{format_compact(value)}</text>'
        )

    for group_idx, group in enumerate(groups):
        base_x = margin_left + group_idx * group_w + group_w * 0.15
        bar_area = group_w * 0.7
        bar_w = bar_area / series_count
        for series_idx, (_, values) in enumerate(series):
            value = values[group_idx] if group_idx < len(values) else 0.0
            bar_h = (value / max_val) * plot_h
            x = base_x + series_idx * bar_w
            y = margin_top + (plot_h - bar_h)
            color = colors[series_idx % len(colors)]
            lines.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" '
                f'height="{bar_h:.1f}" fill="{color}" rx="4" />'
            )
        lines.append(
            f'<text x="{margin_left + group_idx * group_w + group_w / 2:.1f}" '
            f'y="{margin_top + plot_h + 22}" fill="{text}" '
            'text-anchor="middle" font-size="11" '
            'font-family="Avenir Next, Avenir, Segoe UI, Helvetica, Arial, '
            f'sans-serif">{group}</text>'
        )

    legend_x = width - margin_right - 160
    legend_y = margin_top + 6
    for idx, (label, _) in enumerate(series):
        color = colors[idx % len(colors)]
        y = legend_y + idx * 18
        lines.append(
            f'<rect x="{legend_x}" y="{y - 10}" width="12" height="12" '
            f'fill="{color}" rx="2" />'
        )
        lines.append(
            f'<text x="{legend_x + 18}" y="{y}" fill="{text}" '
            'font-size="11" font-family="Avenir Next, Avenir, Segoe UI, '
            f'Helvetica, Arial, sans-serif">{label}</text>'
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


def collect_series(
    results: dict[str, dict[str, float | str]],
    prefix: str,
) -> list[tuple[int, float]]:
    items: list[tuple[int, float]] = []
    for name, data in results.items():
        if not name.startswith(f"{prefix}/"):
            continue
        size_str = name.split("/")[-1]
        if not size_str.isdigit():
            continue
        size = int(size_str)
        unit = str(data["time_unit"])
        time_ns = float(data["real_time"]) * time_unit_to_ns(unit)
        items.append((size, time_ns))
    return sorted(items, key=lambda item: item[0])


def write_summary_csv(
    out_path: Path,
    rows: Iterable[tuple[str, str, str, float, str]],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["config", "benchmark", "metric", "value", "unit"])
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


def write_env(
    path: Path,
    run_id: str,
    configs: list[BenchConfig],
    run_paths: dict[str, Path],
) -> None:
    lines = [
        f"Run: {run_id}",
        f"OS: {platform.platform()}",
        f"Arch: {platform.machine()}",
        f"CPU: {platform.processor() or 'unknown'}",
        f"Compiler: {compiler_version()}",
    ]
    for config in configs:
        run_path = run_paths.get(config.name)
        lines.extend(
            [
                "",
                f"Config: {config.label}",
                f"OpenMP: {'on' if config.openmp else 'off'}",
                f"Native: {'on' if config.native else 'off'}",
                f"Run file: {run_path.name if run_path else 'unknown'}",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run benchmarks + charts.")
    parser.add_argument("--build-dir", default="build-bench")
    parser.add_argument(
        "--openmp",
        action="store_true",
        help="Enable OpenMP for the optimized config.",
    )
    parser.add_argument(
        "--native",
        action="store_true",
        help="Enable -march=native for the optimized config.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = repo_root()
    configs = [BenchConfig("baseline", "baseline", False, False)]
    if args.native:
        configs.append(BenchConfig("native", "native", False, True))
    if args.openmp:
        openmp_native = args.native
        openmp_label = "openmp+native" if openmp_native else "openmp"
        openmp_name = "openmp-native" if openmp_native else "openmp"
        configs.append(BenchConfig(openmp_name, openmp_label, True, openmp_native))

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    runs_dir = root / "docs" / "benchmarks" / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    run_paths: dict[str, Path] = {}
    results_by_config: dict[str, dict[str, dict[str, float | str]]] = {}

    for config in configs:
        build_dir = root / f"{args.build_dir}-{config.name}"
        run_path = runs_dir / f"bench-{run_id}-{config.name}.json"
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
                f"-DFAST_MNIST_ENABLE_OPENMP={'ON' if config.openmp else 'OFF'}",
                f"-DFAST_MNIST_ENABLE_NATIVE={'ON' if config.native else 'OFF'}",
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

        run_paths[config.name] = run_path
        results_by_config[config.name] = parse_benchmarks(run_path)

    summary_rows: list[tuple[str, str, str, float, str]] = []
    ops = {
        "dot": ("benchDot", "Dot"),
        "transpose": ("benchTranspose", "Transpose"),
        "axpy": ("benchAxpy", "Axpy"),
    }
    series_by_op: dict[str, list[tuple[str, list[tuple[int, float]]]]] = {
        key: [] for key in ops
    }
    throughput_by_config: dict[str, list[float]] = {}

    for config in configs:
        results = results_by_config.get(config.name, {})
        for key, (prefix, _) in ops.items():
            points = collect_series(results, prefix)
            series_by_op[key].append((config.label, points))
            for size, time_ns in points:
                summary_rows.append(
                    (config.label, f"{prefix}/{size}", "time", time_ns, "ns/op")
                )

        throughput_values: list[float] = []
        for name in ["benchLearn", "benchClassify"]:
            data = results.get(name)
            if not data:
                throughput_values.append(0.0)
                continue
            unit = str(data["time_unit"])
            seconds = float(data["real_time"]) * time_unit_scale(unit)
            images_per_sec = (1.0 / seconds) if seconds > 0 else 0.0
            summary_rows.append(
                (config.label, name, "throughput", images_per_sec, "img/s")
            )
            throughput_values.append(images_per_sec)
        throughput_by_config[config.label] = throughput_values

    summary_path = root / "docs" / "benchmarks" / "bench_summary.csv"
    write_summary_csv(summary_path, summary_rows)

    charts_dir = root / "docs" / "benchmarks" / "charts"
    for key, (_, label) in ops.items():
        write_svg_line_chart(
            series_by_op[key],
            charts_dir / f"{key}-light.svg",
            f"{label} scaling",
            "Lower is better (ns/op)",
            "light",
        )
        write_svg_line_chart(
            series_by_op[key],
            charts_dir / f"{key}-dark.svg",
            f"{label} scaling",
            "Lower is better (ns/op)",
            "dark",
        )

    throughput_series = [
        (config.label, throughput_by_config.get(config.label, [0.0, 0.0]))
        for config in configs
    ]
    write_svg_grouped_bar_chart(
        ["learn", "classify"],
        throughput_series,
        charts_dir / "throughput-compare-light.svg",
        "Training/inference throughput",
        "Higher is better (img/s)",
        "light",
    )
    write_svg_grouped_bar_chart(
        ["learn", "classify"],
        throughput_series,
        charts_dir / "throughput-compare-dark.svg",
        "Training/inference throughput",
        "Higher is better (img/s)",
        "dark",
    )

    env_path = root / "docs" / "benchmarks" / "bench_env.md"
    write_env(env_path, run_id, configs, run_paths)
    for run_path in run_paths.values():
        print(f"Wrote {run_path}")
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
