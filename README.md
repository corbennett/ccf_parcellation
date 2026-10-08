# CCF area lookup

Look up the [Unified Mouse Brain Atlas v2](https://figshare.com/articles/dataset/Unified_mouse_brain_atlas_v2/25750983) label at an Allen CCFv3 position. The atlas is from [Chon et al., 2019](https://www.nature.com/articles/s41467-019-13057-w). Point lookups can use a downloaded copy or read directly from Figshare without loading the 287 MB image into memory.

## Atlas data

For local use, download the Unified Mouse Brain Atlas v2 data from the [Unified Atlas website on Figshare](https://figshare.com/articles/dataset/Unified_mouse_brain_atlas_v2/25750983) and extract it. The lookup expects `UnifiedAtlas_Label_v2_20um-isotropic.nii` and `UnifiedAtlas_Label_ontology_v2.csv` in the extracted directory. By default, it looks in `~/Downloads/25750983`; use `--atlas-dir /path/to/25750983` or set `CCF_ATLAS_DIR` for another location. Keep the large atlas files outside this repository.

For point lookups without a local copy, add `--remote-if-missing`. If both files are present locally, the lookup uses them. Otherwise it reads the small ontology CSV from Figshare and requests only the NIfTI header and the two bytes for each queried voxel. Remote lookups require an internet connection and incur one request per point. The figure and alignment scripts still require local atlas files.

## Run

From this directory, with [uv](https://docs.astral.sh/uv/) installed:

```sh
uv run python main.py 5180 3000 7380
```

To use Figshare when the local files are absent:

```sh
uv run python main.py 5180 3000 7380 --remote-if-missing
```

The three numbers are **AP, DV, ML in micrometers**, measured from the anterior, superior, left corner of Allen CCFv3. AP increases toward the posterior, DV toward the ventral side, and ML toward the right. The example returns:

```json
{"ccf_um": [5180.0, 3000.0, 7380.0], "atlas_voxel": [200, 150, 400], "region_id": 2380, "name": "Caudoputamen- intermediate, dorsomedial, dorsal tip", "acronym": "CPi, dm, dt", "status": "labeled"}
```

## Use from Python

```python
from ccf_parcellation import AtlasLookup

with AtlasLookup("~/Downloads/25750983") as atlas:
    result = atlas.lookup(5180, 3000, 7380)
    print(result.name, result.acronym)
```

Pass `remote_if_missing=True` to `AtlasLookup` for the same fallback in Python.

Reuse one `AtlasLookup` instance for many points. Each call returns the atlas voxel, raw region ID, label, and status. Coordinates must be within AP `[0, 13200)`, DV `[0, 8000)`, and ML `[0, 11400)` µm. Each point selects the containing 20 µm voxel. `outside_brain` means the volume contains ID 0 at an otherwise valid CCF position.

The released label NIfTI has no usable spatial transform, and its header records `pixdim=(1, 1, 1)` without spatial units despite the 20 µm atlas description. The lookup therefore uses the known 20 µm grid and the atlas' RSP axis orientation, as documented in [BrainGlobe's packaging script](https://github.com/brainglobe/brainglobe-atlasapi/blob/main/atlas_scripts/kim_mouse_isotropic.py), to map Allen's [PIR CCF coordinates](https://brain-map.org/support/documentation/api-allen-brain-connectivity-atlas) to the file's axes. Input coordinates must already be registered to Allen CCFv3; stereotaxic coordinates relative to bregma are a different frame.

The supplied ontology omits nonzero label ID **728**, which appears in the volume. [Allen's CCF reference](https://s3.amazonaws.com/webflow-prod-assets/689cfbd308fa7373b604d290/68e8e1748df5868fa28754dc_MouseCCF.pdf) identifies it as **arbor vitae** (`arb`). At those points the result includes that name and `status: "allen_ontology_fallback"` to distinguish its source from the released v2 CSV. Any other unknown ID returns `missing_from_ontology` with the raw ID and null label fields.

Run the tests with `uv run python -m unittest discover -s tests`.

## Coronal comparison figures

Generate matched coronal sections from the unified atlas and the standard Allen CCF 2017 annotation:

```sh
uv run --extra viz python scripts/plot_coronal_comparison.py
```

The script uses the downloaded unified atlas plus `~/Downloads/annotation_25.nrrd` and `~/Downloads/average_template_25.nrrd`. The local annotation file was verified to match [Allen's official CCF 2017 25 µm volume](https://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/annotation/ccf_2017/annotation_25.nrrd) by SHA-256. Pass `--unified-dir`, `--allen-labels`, or `--allen-template` for other locations. `--ap-mm` selects positions measured posteriorly from the CCF anterior origin; the default figure uses 3, 5, 7, 9, and 11 mm.

The output is [one comparison sheet](figures/coronal_parcellation_comparison.png), a [PDF](figures/coronal_parcellation_comparison.pdf), and separate PNGs for each AP position in `figures/`. Both columns use the **same Allen 25 µm template slice**. The unified 20 µm labels are sampled at Allen voxel centers onto that grid before either overlay is drawn, so corresponding pixels refer to the same CCF position. This comparison uses a 25 µm display grid and therefore does not show every 20 µm detail in the unified volume. Colors are assigned consistently by numeric region ID; they are visual aids, not anatomical categories.

## Validate the atlas alignment with shared regions

```sh
uv run --extra viz python scripts/validate_alignment.py
```

This compares the complete 25 µm Allen CCF 2017 annotation with the unified atlas sampled at the same voxel centers. It groups every unified region with its descendants, so **V1** includes the original primary visual layers plus the new monocular and binocular subdivisions. The resulting whole-volume Dice overlaps are **0.863 for V1**, **0.785 for CA1**, and **0.931 for caudoputamen**; other regions differ more substantially because the unified atlas redraws boundaries. Reversing AP or DV gives **zero V1 overlap**, supporting those two axis directions. The [machine-readable report](figures/alignment_validation.json) includes all tested regions and orientation alternatives.

The V1 mask is nearly bilateral: reversing ML changes its Dice only from **0.8633 to 0.8622**. Shared-region overlap therefore does not independently establish the left–right sign. The ML reversal in the coordinate lookup follows the [source atlas RSP orientation](https://github.com/brainglobe/brainglobe-atlasapi/blob/main/atlas_scripts/kim_mouse_isotropic.py) and [Allen's CCF coordinate convention](https://brain-map.org/support/documentation/api-allen-brain-connectivity-atlas).
