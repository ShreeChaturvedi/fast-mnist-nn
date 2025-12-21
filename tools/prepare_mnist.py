#!/usr/bin/env python3
"""Download MNIST and convert to P2 PGM plus list files."""

from __future__ import annotations

import argparse
import gzip
import shutil
import struct
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import BinaryIO, Iterable, Tuple

MNIST_BASES = [
    "http://yann.lecun.com/exdb/mnist/",
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
]
FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


class MnistError(RuntimeError):
    """Raised when MNIST files are missing or malformed."""


try:
    from tqdm import tqdm

    HAVE_TQDM = True
except ModuleNotFoundError:
    tqdm = None
    HAVE_TQDM = False


def enable_tqdm(auto_install: bool) -> bool:
    global tqdm
    global HAVE_TQDM
    if HAVE_TQDM:
        return True
    if not auto_install:
        return False
    print("tqdm not found; installing with pip...")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--user",
            "--break-system-packages",
            "--no-warn-script-location",
            "tqdm",
        ],
        check=False,
    )
    if result.returncode != 0:
        print("tqdm install failed; using basic progress output.")
        return False
    try:
        from tqdm import tqdm as tqdm_mod
    except ModuleNotFoundError:
        print("tqdm install incomplete; using basic progress output.")
        return False
    tqdm = tqdm_mod
    HAVE_TQDM = True
    return True


def update_progress(prefix: str, current: int, total: int | None,
                    last_percent: int) -> int:
    if total is None or total <= 0:
        return last_percent
    percent = int(current * 100 / total)
    if percent == last_percent:
        return last_percent
    bar = "#" * (percent // 5)
    bar = bar.ljust(20, "-")
    sys.stdout.write(f"\r{prefix} [{bar}] {percent:3d}%")
    sys.stdout.flush()
    return percent


def download_file(url: str, dest: Path, desc: str | None = None) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return
    with urllib.request.urlopen(url) as resp, dest.open("wb") as out:
        if not HAVE_TQDM:
            total = resp.headers.get("Content-Length")
            total_size = int(total) if total is not None else None
            prefix = f"Downloading {desc or dest.name}"
            downloaded = 0
            last_percent = -1
            if total_size is None:
                print(f"{prefix} ...")
            while True:
                chunk = resp.read(1024 * 128)
                if not chunk:
                    break
                out.write(chunk)
                downloaded += len(chunk)
                last_percent = update_progress(
                    prefix,
                    downloaded,
                    total_size,
                    last_percent,
                )
            if total_size is not None:
                sys.stdout.write("\n")
            sys.stdout.flush()
            return
        total = resp.headers.get("Content-Length")
        total_size = int(total) if total is not None else None
        with tqdm(
            total=total_size,
            unit="B",
            unit_scale=True,
            desc=desc or dest.name,
        ) as bar:
            while True:
                chunk = resp.read(1024 * 128)
                if not chunk:
                    break
                out.write(chunk)
                bar.update(len(chunk))

def download_with_fallback(name: str, dest: Path) -> None:
    last_err = None
    for base in MNIST_BASES:
        url = base + name
        try:
            download_file(url, dest, desc=name)
            return
        except (urllib.error.HTTPError, urllib.error.URLError) as err:
            last_err = err
    raise MnistError(f"Failed to download {name}: {last_err}")

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
    fp = gzip.open(path, "rb")
    magic = read_u32_be(fp)
    if magic != 2051:
        fp.close()
        raise MnistError(f"Bad image magic: {magic}")
    count = read_u32_be(fp)
    rows = read_u32_be(fp)
    cols = read_u32_be(fp)
    image_size = rows * cols

    def image_iter() -> Iterable[bytes]:
        try:
            for _ in range(count):
                data = fp.read(image_size)
                if len(data) != image_size:
                    raise MnistError("Image file truncated")
                yield data
        finally:
            fp.close()

    return rows, cols, image_iter()


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
    iterator = enumerate(images)
    if HAVE_TQDM:
        iterator = tqdm(
            iterator,
            total=max_count,
            desc=f"Converting {out_dir.name}",
        )
    else:
        print(f"Converting {out_dir.name} ...")
        last_percent = -1
    with list_path.open("w", encoding="utf-8") as list_file:
        for idx, pixels in iterator:
            if idx >= max_count:
                break
            label = labels[idx]
            name = f"digit_{idx}_{label}.pgm"
            rel_path = f"{out_dir.name}/{name}"
            write_pgm(out_dir / name, rows, cols, pixels)
            list_file.write(rel_path + "\n")
            if not HAVE_TQDM:
                last_percent = update_progress(
                    f"Converting {out_dir.name}",
                    idx + 1,
                    max_count,
                    last_percent,
                )
    if not HAVE_TQDM and max_count > 0:
        sys.stdout.write("\n")
        sys.stdout.flush()


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
    parser.add_argument(
        "--no-auto-install",
        action="store_true",
        help="Disable auto-install of tqdm for progress bars",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    download_root = args.output / "downloads"

    auto_install = not args.no_auto_install
    enable_tqdm(auto_install)
    if not HAVE_TQDM:
        print("Using basic progress output (install tqdm for bars).")

    for name in FILES.values():
        download_with_fallback(name, download_root / name)

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
