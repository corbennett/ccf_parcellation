# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "numpy>=2.0",
#   "pyarrow>=18.0",
# ]
# ///
"""Add Unified Atlas v2 labels to the consolidated units parquet.

Run directly with uv; no checkout or environment setup is required:

    uv run append_unified_atlas_labels.py

The input and output may be local paths or S3 URIs. AWS credentials are read
through the normal AWS credential chain. The atlas itself is downloaded once
from its Figshare release and retained in a local cache.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import struct
import sys
from pathlib import Path
from typing import BinaryIO
from urllib.request import urlopen

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pyarrow import fs


DEFAULT_INPUT = (
    "s3://aind-scratch-data/dynamic-routing/cache/nwb_components/"
    "v0.0.289/consolidated/units.parquet"
)
DEFAULT_OUTPUT = "units_with_unified_atlas.parquet"

ATLAS_FILENAME = "UnifiedAtlas_Label_v2_20um-isotropic.nii"
ONTOLOGY_FILENAME = "UnifiedAtlas_Label_ontology_v2.csv"
FIGSHARE_FILES_URL = "https://api.figshare.com/v2/articles/25750983/files"
SHAPE = (570, 400, 660)  # Source NIfTI axes: right/left, dorsal/ventral, posterior/anterior.
RESOLUTION_UM = 20
HEADER_BYTES = 352
REQUEST_TIMEOUT_SECONDS = 60
ALLEN_FALLBACK_LABELS = {728: ("arbor vitae", "arb")}
NEW_COLUMNS = (
    "unified_atlas_name",
    "unified_atlas_acronym",
    "unified_atlas_status",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT, help="input parquet path or S3 URI")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="output parquet path or S3 URI")
    parser.add_argument(
        "--atlas-cache",
        type=Path,
        default=Path(os.environ.get("CCF_ATLAS_DIR", Path.home() / ".cache" / "ccf-parcellation")),
        help="directory in which to cache the Unified Atlas files",
    )
    parser.add_argument("--overwrite", action="store_true", help="replace an existing output")
    return parser.parse_args()


def arrow_location(uri: str) -> tuple[fs.FileSystem, str]:
    """Resolve a local path or S3 URI to a PyArrow filesystem and path."""
    if uri.lower().startswith("s3://"):
        bucket, separator, key = uri[5:].partition("/")
        if not separator or not bucket or not key:
            raise ValueError(f"S3 URI must include a bucket and key: {uri}")
        region = fs.resolve_s3_region(bucket)
        return fs.S3FileSystem(region=region), f"{bucket}/{key}"
    path = Path(uri).expanduser().resolve()
    return fs.LocalFileSystem(), str(path)


def ensure_output_available(filesystem: fs.FileSystem, path: str, overwrite: bool) -> None:
    info = filesystem.get_file_info(path)
    if info.type != fs.FileType.NotFound and not overwrite:
        raise FileExistsError(f"output already exists (pass --overwrite to replace it): {path}")
    parent = str(Path(path).parent) if isinstance(filesystem, fs.LocalFileSystem) else path.rpartition("/")[0]
    if parent and parent != ".":
        filesystem.create_dir(parent, recursive=True)


def figshare_files() -> dict[str, dict]:
    with urlopen(FIGSHARE_FILES_URL, timeout=REQUEST_TIMEOUT_SECONDS) as response:
        return {item["name"]: item for item in json.load(response)}


def download(response: BinaryIO, destination: Path, expected_size: int) -> None:
    temporary = destination.with_name(f".{destination.name}.download")
    bytes_written = 0
    try:
        with temporary.open("wb") as handle:
            while chunk := response.read(1024 * 1024):
                bytes_written += len(chunk)
                if bytes_written > expected_size:
                    raise ValueError(f"download exceeded expected size for {destination.name}")
                handle.write(chunk)
        if bytes_written != expected_size:
            raise ValueError(
                f"incomplete download for {destination.name}: "
                f"expected {expected_size} bytes, received {bytes_written}"
            )
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def validate_atlas(path: Path, expected_size: int) -> None:
    with path.open("rb") as handle:
        header = handle.read(HEADER_BYTES)
    if (
        path.stat().st_size != expected_size
        or len(header) != HEADER_BYTES
        or struct.unpack_from("<i", header, 0)[0] != 348
        or struct.unpack_from("<4h", header, 40) != (3, *SHAPE)
        or struct.unpack_from("<2h", header, 70) != (512, 16)
        or struct.unpack_from("<f", header, 108)[0] != HEADER_BYTES
        or header[344:348] != b"n+1\x00"
    ):
        raise ValueError(f"unexpected atlas NIfTI layout: {path}")


def read_ontology(path: Path) -> dict[int, tuple[str, str]]:
    regions: dict[int, tuple[str, str]] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            raw_id = row["id"].strip()
            if not raw_id:
                continue
            region_id = int(raw_id)
            if region_id in regions:
                raise ValueError(f"duplicate ontology ID {region_id} in {path}")
            regions[region_id] = (row["name"].strip(), row["acronym"].strip())
    if not regions:
        raise ValueError(f"ontology contains no regions: {path}")
    return regions


def prepare_atlas(cache_dir: Path) -> tuple[Path, dict[int, tuple[str, str]]]:
    cache_dir = cache_dir.expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    metadata = figshare_files()

    for filename in (ATLAS_FILENAME, ONTOLOGY_FILENAME):
        if filename not in metadata:
            raise ValueError(f"Figshare release is missing {filename}")
        item = metadata[filename]
        destination = cache_dir / filename
        expected_size = int(item["size"])
        try:
            if filename == ATLAS_FILENAME:
                validate_atlas(destination, expected_size)
            elif destination.stat().st_size != expected_size:
                raise ValueError("wrong ontology size")
        except (OSError, ValueError):
            print(f"Downloading {filename} to {cache_dir} ...", file=sys.stderr)
            with urlopen(item["download_url"], timeout=REQUEST_TIMEOUT_SECONDS) as response:
                download(response, destination, expected_size)

    atlas_path = cache_dir / ATLAS_FILENAME
    ontology_path = cache_dir / ONTOLOGY_FILENAME
    validate_atlas(atlas_path, int(metadata[ATLAS_FILENAME]["size"]))
    return atlas_path, read_ontology(ontology_path)


def atlas_columns(
    table: pa.Table, atlas_path: Path, regions: dict[int, tuple[str, str]]
) -> tuple[pa.Array, pa.Array, pa.Array]:
    required = ("ccf_ap", "ccf_dv", "ccf_ml")
    missing = [name for name in required if name not in table.column_names]
    if missing:
        raise ValueError(f"input parquet is missing coordinate columns: {', '.join(missing)}")

    coordinates = [
        table[name].combine_chunks().to_numpy(zero_copy_only=False, writable=False)
        for name in required
    ]
    ap, dv, ml = (np.asarray(values, dtype=np.float64) for values in coordinates)
    finite = np.isfinite(ap) & np.isfinite(dv) & np.isfinite(ml)
    in_bounds = (
        finite
        & (ap >= 0)
        & (ap < SHAPE[2] * RESOLUTION_UM)
        & (dv >= 0)
        & (dv < SHAPE[1] * RESOLUTION_UM)
        & (ml >= 0)
        & (ml < SHAPE[0] * RESOLUTION_UM)
    )

    region_ids = np.zeros(len(table), dtype=np.uint16)
    valid_rows = np.flatnonzero(in_bounds)
    if valid_rows.size:
        ap_index = np.floor_divide(ap[valid_rows], RESOLUTION_UM).astype(np.intp)
        dv_index = np.floor_divide(dv[valid_rows], RESOLUTION_UM).astype(np.intp)
        ml_index = np.floor_divide(ml[valid_rows], RESOLUTION_UM).astype(np.intp)
        atlas = np.memmap(
            atlas_path,
            dtype="<u2",
            mode="r",
            offset=HEADER_BYTES,
            shape=SHAPE,
            order="F",
        )
        region_ids[valid_rows] = atlas[
            SHAPE[0] - 1 - ml_index,
            dv_index,
            SHAPE[2] - 1 - ap_index,
        ]

    names: list[str | None] = [None] * len(table)
    acronyms: list[str | None] = [None] * len(table)
    statuses = np.full(len(table), "missing_coordinates", dtype=object)
    statuses[finite & ~in_bounds] = "out_of_bounds"

    for row in valid_rows:
        region_id = int(region_ids[row])
        if region_id == 0:
            statuses[row] = "outside_brain"
        elif region_id in regions:
            names[row], acronyms[row] = regions[region_id]
            statuses[row] = "labeled"
        elif region_id in ALLEN_FALLBACK_LABELS:
            names[row], acronyms[row] = ALLEN_FALLBACK_LABELS[region_id]
            statuses[row] = "allen_ontology_fallback"
        else:
            statuses[row] = "missing_from_ontology"

    return pa.array(names), pa.array(acronyms), pa.array(statuses)


def main() -> int:
    args = parse_args()
    input_fs, input_path = arrow_location(args.input)
    output_fs, output_path = arrow_location(args.output)
    ensure_output_available(output_fs, output_path, args.overwrite)

    print(f"Reading {args.input} ...", file=sys.stderr)
    table = pq.read_table(input_path, filesystem=input_fs)
    collisions = [name for name in NEW_COLUMNS if name in table.column_names]
    if collisions:
        raise ValueError(f"input already contains output columns: {', '.join(collisions)}")

    atlas_path, regions = prepare_atlas(args.atlas_cache)
    print(f"Labeling {len(table):,} units ...", file=sys.stderr)
    columns = atlas_columns(table, atlas_path, regions)
    for name, column in zip(NEW_COLUMNS, columns):
        table = table.append_column(name, column)

    print(f"Writing {args.output} ...", file=sys.stderr)
    pq.write_table(table, output_path, filesystem=output_fs, compression="zstd")
    print(f"Wrote {len(table):,} rows to {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, pa.ArrowException) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
