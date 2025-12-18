#!/usr/bin/env python3
"""Download MNIST and convert to P2 PGM plus list files."""

from __future__ import annotations

import argparse
import gzip
import shutil
import struct
import urllib.request
from pathlib import Path
from typing import BinaryIO, Iterable, Tuple

MNIST_BASE = "http://yann.lecun.com/exdb/mnist/"
FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


class MnistError(RuntimeError):
    """Raised when MNIST files are missing or malformed."""


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return
    with urllib.request.urlopen(url) as resp, dest.open("wb") as out:
        shutil.copyfileobj(resp, out)


def read_u32_be(fp: BinaryIO) -> int:
    data = fp.read(4)
    if len(data) != 4:
        raise MnistError("Unexpected end of file")
    return struct.unpack(">I", data)[0]


def read_labels(path: Path) -> bytes:
    with gzip.open(path, "rb") as fp:
        magic = read_u32_be(fp)
        if magic != 2049:
            raise MnistError(f"Bad label magic: {magic}")
        count = read_u32_be(fp)
        data = fp.read(count)
        if len(data) != count:
            raise MnistError("Label file truncated")
        return data


def iter_images(path: Path) -> Tuple[int, int, Iterable[bytes]]:
    with gzip.open(path, "rb") as fp:
        magic = read_u32_be(fp)
        if magic != 2051:
            raise MnistError(f"Bad image magic: {magic}")
        count = read_u32_be(fp)
        rows = read_u32_be(fp)
        cols = read_u32_be(fp)
        image_size = rows * cols
        images = (fp.read(image_size) for _ in range(count))
        return rows, cols, images


def write_pgm(path: Path, rows: int, cols: int, pixels: bytes) -> None:
    with path.open("w", encoding="ascii") as out:
        out.write("P2\n")
        out.write(f"{cols} {rows}\n")
        out.write("255\n")
        for r in range(rows):
            start = r * cols
            row = pixels[start : start + cols]
            out.write(" ".join(str(v) for v in row))
            out.write("\n")


def convert_split(
    images_path: Path,
    labels_path: Path,
    out_dir: Path,
    list_path: Path,
    limit: int | None,
) -> None:
    labels = read_labels(labels_path)
    rows, cols, images = iter_images(images_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    list_path.parent.mkdir(parents=True, exist_ok=True)

    max_count = len(labels) if limit is None else min(limit, len(labels))
    with list_path.open("w", encoding="utf-8") as list_file:
        for idx, pixels in enumerate(images):
            if idx >= max_count:
                break
            label = labels[idx]
            name = f"digit_{idx}_{label}.pgm"
            rel_path = f"{out_dir.name}/{name}"
            write_pgm(out_dir / name, rows, cols, pixels)
            list_file.write(rel_path + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download MNIST and generate PGM files + lists."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data"),
        help="Output data root (default: ./data)",
    )
    parser.add_argument(
        "--list-dir",
        type=Path,
        default=Path("."),
        help="Directory for TrainingSetList.txt and TestingSetList.txt",
    )
    parser.add_argument(
        "--limit-train",
        type=int,
        default=None,
        help="Optional cap on training images",
    )
    parser.add_argument(
        "--limit-test",
        type=int,
        default=None,
        help="Optional cap on test images",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    download_root = args.output / "downloads"

    for key, name in FILES.items():
        url = MNIST_BASE + name
        download_file(url, download_root / name)

    convert_split(
        download_root / FILES["train_images"],
        download_root / FILES["train_labels"],
        args.output / "TrainingSet",
        args.list_dir / "TrainingSetList.txt",
        args.limit_train,
    )
    convert_split(
        download_root / FILES["test_images"],
        download_root / FILES["test_labels"],
        args.output / "TestingSet",
        args.list_dir / "TestingSetList.txt",
        args.limit_test,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
