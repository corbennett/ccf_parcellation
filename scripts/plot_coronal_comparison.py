"""Render both parcellations over the same Allen CCF coronal template slices.

Run with: uv run --extra viz python scripts/plot_coronal_comparison.py
"""

import argparse
import colorsys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nrrd
import numpy as np


NEW_SHAPE = (660, 400, 570)  # raw NIfTI as (posterior-to-anterior, DV, right-to-left)
OLD_SHAPE = (528, 320, 456)  # Allen NRRD as (anterior-to-posterior, DV, right-to-left)
DEFAULT_AP_MM = (3.0, 5.0, 7.0, 9.0, 11.0)
EXTENT_MM = (0, 11.4, 8.0, 0)


def read_nii_volume(path: Path) -> np.memmap:
    """Map the 570 x 400 x 660 label NIfTI in source array order."""
    expected_size = 352 + 2 * np.prod(NEW_SHAPE)
    if path.stat().st_size != expected_size:
        raise ValueError(f"Unexpected NIfTI size: {path}")
    return np.memmap(path, dtype="<u2", mode="r", offset=352, shape=NEW_SHAPE)


def unified_on_allen_grid(volume: np.memmap, allen_ap: int) -> np.ndarray:
    """Sample unified labels at centers of one Allen 25 µm coronal slice."""
    ap_um = (allen_ap + 0.5) * 25
    unified_ap = 659 - int(ap_um // 20)
    dv = np.floor((np.arange(320) + 0.5) * 25 / 20).astype(int)
    ml = np.floor((np.arange(456) + 0.5) * 25 / 20).astype(int)
    # Both source arrays begin at anatomical right across a coronal plane.
    return volume[unified_ap][np.ix_(dv, ml)]


def region_overlay(labels: np.ndarray) -> np.ndarray:
    """Give each integer ID a stable pastel color; ID 0 stays transparent."""
    ids, inverse = np.unique(labels, return_inverse=True)
    colors = np.zeros((len(ids), 4), dtype=np.float32)
    for index, region_id in enumerate(ids):
        if region_id:
            # Shared IDs get the same color in the two parcellations.
            hue = (int(region_id) * 0.61803398875) % 1
            colors[index, :3] = colorsys.hsv_to_rgb(hue, 0.47, 0.96)
            colors[index, 3] = 0.55
    return colors[inverse].reshape((*labels.shape, 4))


def boundary_overlay(labels: np.ndarray) -> np.ndarray:
    """Draw a one-pixel line where adjacent nonzero labels differ."""
    edges = np.zeros(labels.shape, dtype=bool)
    edges[1:, :] |= (labels[1:, :] != labels[:-1, :]) & (labels[1:, :] != 0)
    edges[:, 1:] |= (labels[:, 1:] != labels[:, :-1]) & (labels[:, 1:] != 0)
    rgba = np.zeros((*labels.shape, 4), dtype=np.float32)
    rgba[..., :3] = (0.07, 0.12, 0.18)
    rgba[..., 3] = edges * 0.86
    return rgba


def draw_panel(ax, reference: np.ndarray, labels: np.ndarray, show_scale: bool = False) -> None:
    if labels.shape != reference.shape:
        raise ValueError("Labels and Allen template slice must use the same 25 µm grid")
    ax.imshow(reference, cmap="gray", vmin=0, vmax=516, extent=EXTENT_MM, interpolation="nearest")
    ax.imshow(region_overlay(labels), extent=EXTENT_MM, interpolation="nearest")
    ax.imshow(boundary_overlay(labels), extent=EXTENT_MM, interpolation="nearest")
    ax.set_xlim(0, 11.4)
    ax.set_ylim(8, 0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if show_scale:
        ax.plot([0.55, 2.55], [7.6, 7.6], color="white", linewidth=3, solid_capstyle="butt")
        ax.plot([0.55, 2.55], [7.6, 7.6], color="#243446", linewidth=1.1, solid_capstyle="butt")
        ax.text(1.55, 7.39, "2 mm", ha="center", va="bottom", fontsize=8, color="#243446",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1.5})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unified-dir", type=Path, default=Path("~/Downloads/25750983"))
    parser.add_argument("--allen-labels", type=Path, default=Path("~/Downloads/annotation_25.nrrd"))
    parser.add_argument("--allen-template", type=Path, default=Path("~/Downloads/average_template_25.nrrd"))
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    parser.add_argument("--ap-mm", nargs="+", type=float, default=DEFAULT_AP_MM)
    args = parser.parse_args()

    unified_dir = args.unified_dir.expanduser()
    allen_labels_path = args.allen_labels.expanduser()
    allen_template_path = args.allen_template.expanduser()
    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    unified_labels = read_nii_volume(unified_dir / "UnifiedAtlas_Label_v2_20um-isotropic.nii")
    allen_labels, _ = nrrd.read(str(allen_labels_path), index_order="F")
    allen_template, _ = nrrd.read(str(allen_template_path), index_order="F")
    if allen_labels.shape != OLD_SHAPE or allen_template.shape != OLD_SHAPE:
        raise ValueError("Allen NRRD files must have shape (528, 320, 456) at 25 µm")

    slices = []
    for ap_mm in args.ap_mm:
        if not 0 <= ap_mm < 13.2:
            raise ValueError(f"AP position outside CCF extent: {ap_mm} mm")
        # Use the Allen voxel containing the requested AP point for both panels.
        allen_ap = int(ap_mm * 1000 // 25)
        common_ref = allen_template[allen_ap]
        new_lab = unified_on_allen_grid(unified_labels, allen_ap)
        old_lab = allen_labels[allen_ap]
        slices.append((ap_mm, common_ref, new_lab, old_lab))

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(len(slices), 2, figsize=(11.5, 3.05 * len(slices) + 1.3),
                             layout="constrained")
    if len(slices) == 1:
        axes = np.array([axes])
    fig.suptitle("Coronal parcellations on the Allen CCF template", fontsize=17, weight="bold")
    for col, title in enumerate(("Unified Atlas v2 labels", "Allen CCF 2017 labels")):
        axes[0, col].set_title(title, fontsize=12, weight="bold", pad=12)
    for row, (ap_mm, common_ref, new_lab, old_lab) in enumerate(slices):
        draw_panel(axes[row, 0], common_ref, new_lab, show_scale=row == len(slices) - 1)
        draw_panel(axes[row, 1], common_ref, old_lab, show_scale=row == len(slices) - 1)
        axes[row, 0].text(-0.06, 0.5, f"AP {ap_mm:g} mm", transform=axes[row, 0].transAxes,
                          rotation=90, va="center", ha="right", fontsize=10, weight="bold")
    fig.supxlabel("Same Allen 25 µm template and grid in both columns  ·  anatomical right at image left", fontsize=9)
    composite = output_dir / "coronal_parcellation_comparison.png"
    fig.savefig(composite, dpi=220, facecolor="white")
    fig.savefig(output_dir / "coronal_parcellation_comparison.pdf", facecolor="white")
    plt.close(fig)

    for ap_mm, common_ref, new_lab, old_lab in slices:
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.1), layout="constrained")
        fig.suptitle(f"Coronal parcellation at CCF AP {ap_mm:g} mm", fontsize=15, weight="bold")
        axes[0].set_title("Unified Atlas v2 labels", fontsize=11)
        axes[1].set_title("Allen CCF 2017 labels", fontsize=11)
        draw_panel(axes[0], common_ref, new_lab, show_scale=True)
        draw_panel(axes[1], common_ref, old_lab, show_scale=True)
        fig.savefig(output_dir / f"coronal_ap_{ap_mm:g}mm.png", dpi=220, facecolor="white")
        plt.close(fig)
    print(f"Wrote {composite} and {len(slices)} individual slice comparisons")


if __name__ == "__main__":
    main()
