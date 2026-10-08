"""Memory-mapped lookup into the Figshare Unified Atlas v2 label NIfTI.

The released label NIfTI has no qform/sform and says pixdim=(1, 1, 1), so its
header alone cannot locate voxels in CCF space. The matching template and
dataset description establish a 20 µm grid of 570 x 400 x 660 voxels. The
released array has RSP orientation (right, superior, posterior at its origin).
Allen CCF coordinates use the ASL corner and AP, DV, ML axis order.
"""

import argparse
import csv
import json
import math
import mmap
import os
import struct
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import BinaryIO, Dict, Optional, Tuple, Union


ATLAS_FILENAME = "UnifiedAtlas_Label_v2_20um-isotropic.nii"
ONTOLOGY_FILENAME = "UnifiedAtlas_Label_ontology_v2.csv"
SHAPE = (570, 400, 660)  # R/L, D/V, P/A
RESOLUTION_UM = 20
HEADER_BYTES = 352
# The v2 volume uses ID 728 but its CSV omits it. Allen CCF identifies it as
# arbor vitae (arb); keep this separate from labels in the release ontology.
ALLEN_FALLBACK_LABELS = {728: ("arbor vitae", "arb")}


@dataclass(frozen=True)
class LookupResult:
    """A label and its provenance for one CCF point."""

    ccf_um: Tuple[float, float, float]  # AP, DV, ML
    atlas_voxel: Tuple[int, int, int]  # file axes: R/L, D/V, P/A
    region_id: int
    name: Optional[str]
    acronym: Optional[str]
    status: str  # labeled, allen_ontology_fallback, outside_brain, or missing_from_ontology


class AtlasLookup:
    """Open the local atlas once and perform constant-time point lookups.

    Coordinates are in micrometers from the Allen CCFv3 anterior, superior,
    left corner, ordered (AP, DV, ML). Increasing AP goes posterior, increasing
    DV goes ventral, and increasing ML goes right. Each position selects the
    containing 20 µm voxel; boundaries belong to the next voxel.
    """

    def __init__(self, atlas_dir: Union[str, Path]):
        atlas_dir = Path(atlas_dir).expanduser()
        atlas_path = atlas_dir / ATLAS_FILENAME
        ontology_path = atlas_dir / ONTOLOGY_FILENAME
        self._regions = _read_ontology(ontology_path)
        self._file = atlas_path.open("rb")
        try:
            _validate_nifti(self._file, atlas_path)
            self._data = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        except Exception:
            self._file.close()
            raise

    def close(self) -> None:
        self._data.close()
        self._file.close()

    def __enter__(self) -> "AtlasLookup":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def lookup(self, ap_um: float, dv_um: float, ml_um: float) -> LookupResult:
        """Return the finest released label at one CCF position.

        Raises ValueError for a coordinate outside the CCF volume or nonfinite
        input. ID 0 means outside annotated brain tissue. ID 728 uses an
        explicitly marked Allen CCF label missing from the release ontology.
        Other unknown IDs retain their raw ID and have null label fields.
        """
        coords = (ap_um, dv_um, ml_um)
        limits = (SHAPE[2], SHAPE[1], SHAPE[0])
        for value, size, axis in zip(coords, limits, ("AP", "DV", "ML")):
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{axis} must be a finite number of micrometers")
            if not 0 <= value < size * RESOLUTION_UM:
                raise ValueError(
                    f"{axis} must be in [0, {size * RESOLUTION_UM}) µm; got {value}"
                )

        # The source NIfTI is RSP, while the input coordinates are AP/DV/ML
        # from the CCF ASL corner. NIfTI axis 0 is the fastest-changing axis.
        ap_index = int(ap_um // RESOLUTION_UM)
        dv_index = int(dv_um // RESOLUTION_UM)
        ml_index = int(ml_um // RESOLUTION_UM)
        voxel = (SHAPE[0] - 1 - ml_index, dv_index, SHAPE[2] - 1 - ap_index)
        offset = HEADER_BYTES + 2 * (
            voxel[0] + SHAPE[0] * (voxel[1] + SHAPE[1] * voxel[2])
        )
        region_id = struct.unpack_from("<H", self._data, offset)[0]

        if region_id == 0:
            name = acronym = None
            status = "outside_brain"
        elif region_id in self._regions:
            name, acronym = self._regions[region_id]
            status = "labeled"
        elif region_id in ALLEN_FALLBACK_LABELS:
            name, acronym = ALLEN_FALLBACK_LABELS[region_id]
            status = "allen_ontology_fallback"
        else:
            name = acronym = None
            status = "missing_from_ontology"
        return LookupResult(coords, voxel, region_id, name, acronym, status)


def _read_ontology(path: Path) -> Dict[int, Tuple[str, str]]:
    regions: Dict[int, Tuple[str, str]] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            raw_id = row["id"].strip()
            if not raw_id:  # The released CSV has trailing blank rows.
                continue
            region_id = int(raw_id)
            if region_id in regions:
                raise ValueError(f"Duplicate ontology ID {region_id} in {path}")
            regions[region_id] = (row["name"].strip(), row["acronym"].strip())
    return regions


def _validate_nifti(handle: BinaryIO, path: Path) -> None:
    header = handle.read(HEADER_BYTES)
    expected_size = HEADER_BYTES + 2 * math.prod(SHAPE)
    if (
        len(header) != HEADER_BYTES
        or struct.unpack_from("<i", header, 0)[0] != 348
        or struct.unpack_from("<4h", header, 40) != (3, *SHAPE)
        or struct.unpack_from("<2h", header, 70) != (512, 16)
        or struct.unpack_from("<f", header, 108)[0] != HEADER_BYTES
        or header[344:348] != b"n+1\x00"
        or path.stat().st_size != expected_size
    ):
        raise ValueError(f"Unexpected atlas NIfTI layout: {path}")


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Look up a Unified Atlas v2 area at Allen CCFv3 AP DV ML coordinates."
    )
    parser.add_argument("ap", type=float, help="anterior to posterior, µm")
    parser.add_argument("dv", type=float, help="dorsal to ventral, µm")
    parser.add_argument("ml", type=float, help="left to right, µm")
    parser.add_argument(
        "--atlas-dir",
        type=Path,
        default=Path(os.environ.get("CCF_ATLAS_DIR", "~/Downloads/25750983")),
        help="directory with the downloaded label NIfTI and ontology CSV",
    )
    args = parser.parse_args(argv)
    try:
        with AtlasLookup(args.atlas_dir) as atlas:
            result = atlas.lookup(args.ap, args.dv, args.ml)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(json.dumps(asdict(result), ensure_ascii=False))
    return 0
