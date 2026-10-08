"""Compare shared anatomical regions in Unified Atlas v2 and Allen CCF 2017.

Run with: uv run --extra viz python scripts/validate_alignment.py

Both volumes are sampled on the Allen 25 µm grid at voxel centers. The
unified labels are aggregated through their ontology so that, for example,
V1 includes its layers and its new monocular/binocular subdivisions.
"""

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import nrrd
import numpy as np


REGIONS = {"V1": 385, "M1": 985, "S1BF": 329, "CA1": 382, "CP": 672}
NEW_SHAPE = (660, 400, 570)  # raw NIfTI: posterior-to-anterior, DV, right-to-left
OLD_SHAPE = (528, 320, 456)  # Allen NRRD: anterior-to-posterior, DV, right-to-left


def descendants_by_region(ontology_path: Path) -> dict:
    children = defaultdict(list)
    with ontology_path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if row["id"].strip() and row["structure_id_path"].strip():
                children[int(row["structure_id_path"])].append(int(row["id"]))
    result = {}
    for name, root in REGIONS.items():
        ids = {root}
        stack = [root]
        while stack:
            for child in children[stack.pop()]:
                if child not in ids:
                    ids.add(child)
                    stack.append(child)
        result[name] = np.array(sorted(ids), dtype=np.uint32)
    return result


def dice_on_allen_grid(old, new, region_ids, ap_flip=True, dv_flip=False, ml_flip=False):
    dv = np.floor((np.arange(320) + 0.5) * 25 / 20).astype(int)
    ml = np.floor((np.arange(456) + 0.5) * 25 / 20).astype(int)
    if dv_flip:
        dv = 399 - dv
    if ml_flip:
        ml = 569 - ml
    intersection = old_count = new_count = 0
    for old_ap in range(528):
        new_ap = int(np.floor((old_ap + 0.5) * 25 / 20))
        if ap_flip:
            new_ap = 659 - new_ap
        old_mask = np.isin(old[old_ap], region_ids)
        new_mask = np.isin(new[new_ap][np.ix_(dv, ml)], region_ids)
        intersection += np.count_nonzero(old_mask & new_mask)
        old_count += np.count_nonzero(old_mask)
        new_count += np.count_nonzero(new_mask)
    dice = 2 * intersection / (old_count + new_count)
    return {"dice": round(dice, 4), "old_voxels": old_count, "unified_voxels_on_25um_grid": new_count}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unified-dir", type=Path, default=Path("~/Downloads/25750983"))
    parser.add_argument("--allen-labels", type=Path, default=Path("~/Downloads/annotation_25.nrrd"))
    parser.add_argument("--output", type=Path, default=Path("figures/alignment_validation.json"))
    args = parser.parse_args()
    unified_dir = args.unified_dir.expanduser()
    allen_path = args.allen_labels.expanduser()
    ontology = descendants_by_region(unified_dir / "UnifiedAtlas_Label_ontology_v2.csv")
    old, _ = nrrd.read(str(allen_path), index_order="F")
    if old.shape != OLD_SHAPE:
        raise ValueError(f"Expected Allen 25 µm shape {OLD_SHAPE}, got {old.shape}")
    new_path = unified_dir / "UnifiedAtlas_Label_v2_20um-isotropic.nii"
    if new_path.stat().st_size != 352 + 2 * np.prod(NEW_SHAPE):
        raise ValueError(f"Unexpected unified label NIfTI size: {new_path}")
    new = np.memmap(new_path, dtype="<u2", mode="r", offset=352, shape=NEW_SHAPE)

    results = {name: dice_on_allen_grid(old, new, ids) for name, ids in ontology.items()}
    alternatives = {
        "AP_unreversed": dice_on_allen_grid(old, new, ontology["V1"], ap_flip=False),
        "DV_reversed": dice_on_allen_grid(old, new, ontology["V1"], dv_flip=True),
        "ML_reversed": dice_on_allen_grid(old, new, ontology["V1"], ml_flip=True),
    }
    report = {
        "allen_annotation_sha256": hashlib.sha256(allen_path.read_bytes()).hexdigest(),
        "sampling": "Allen 25 µm voxel centers; nearest containing unified 20 µm voxel",
        "region_dice": results,
        "V1_alternative_orientations": alternatives,
        "interpretation": (
            "Shared regions strongly validate AP and DV alignment. "
            "Bilateral symmetry makes the ML sign hard to determine from these masks alone."
        ),
    }
    output = args.output.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for name, result in results.items():
        print(f"{name:5} Dice {result['dice']:.4f}")
    for name, result in alternatives.items():
        print(f"V1 {name:15} Dice {result['dice']:.4f}")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
