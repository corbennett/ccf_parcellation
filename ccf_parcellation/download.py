"""Download the files required for local Unified Atlas lookups."""

import json
from pathlib import Path
from typing import Dict, Tuple, Union
from urllib.request import urlopen

from .lookup import (
    ATLAS_FILENAME,
    FIGSHARE_FILES_URL,
    HEADER_BYTES,
    ONTOLOGY_FILENAME,
    REQUEST_TIMEOUT_SECONDS,
    _read_ontology,
    _validate_nifti,
)


def download_atlas(atlas_dir: Union[str, Path]) -> Tuple[Path, Path]:
    """Download and validate the label volume and ontology into ``atlas_dir``.

    Files that already match the Figshare release are reused. Downloads are
    written to temporary files and moved into place only after validation.

    Returns:
        The paths to the label NIfTI and ontology CSV, respectively.
    """
    atlas_dir = Path(atlas_dir).expanduser()
    atlas_dir.mkdir(parents=True, exist_ok=True)
    files = _figshare_files()

    paths = []
    for filename in (ATLAS_FILENAME, ONTOLOGY_FILENAME):
        try:
            metadata = files[filename]
        except KeyError as exc:
            raise ValueError(f"Figshare release is missing {filename}") from exc

        destination = atlas_dir / filename
        expected_size = int(metadata["size"])
        if _is_valid_file(destination, filename, expected_size):
            paths.append(destination)
            continue

        temporary = destination.with_name(f".{destination.name}.download")
        try:
            _download_file(metadata["download_url"], temporary, expected_size)
            _validate_file(temporary, filename, expected_size)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
        paths.append(destination)

    return paths[0], paths[1]


def _figshare_files() -> Dict[str, dict]:
    with urlopen(FIGSHARE_FILES_URL, timeout=REQUEST_TIMEOUT_SECONDS) as response:
        return {item["name"]: item for item in json.load(response)}


def _download_file(url: str, destination: Path, expected_size: int) -> None:
    bytes_written = 0
    with urlopen(url, timeout=REQUEST_TIMEOUT_SECONDS) as response:
        with destination.open("wb") as handle:
            while chunk := response.read(1024 * 1024):
                bytes_written += len(chunk)
                if bytes_written > expected_size:
                    raise ValueError(f"Download is larger than expected: {url}")
                handle.write(chunk)
    if bytes_written != expected_size:
        raise ValueError(
            f"Incomplete download from {url}: expected {expected_size} bytes, "
            f"received {bytes_written}"
        )


def _is_valid_file(path: Path, filename: str, expected_size: int) -> bool:
    try:
        _validate_file(path, filename, expected_size)
    except (OSError, ValueError, KeyError):
        return False
    return True


def _validate_file(path: Path, filename: str, expected_size: int) -> None:
    if path.stat().st_size != expected_size:
        raise ValueError(f"Unexpected file size: {path}")
    if filename == ATLAS_FILENAME:
        with path.open("rb") as handle:
            _validate_nifti(handle.read(HEADER_BYTES), expected_size, str(path))
    else:
        regions = _read_ontology(path)
        if not regions:
            raise ValueError(f"Ontology contains no regions: {path}")
