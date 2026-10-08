import math
import struct
import tempfile
import unittest
from pathlib import Path

from ccf_parcellation import AtlasLookup
from ccf_parcellation.lookup import ATLAS_FILENAME, ONTOLOGY_FILENAME, SHAPE


class AtlasLookupTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        directory = Path(self.temp_dir.name)
        (directory / ONTOLOGY_FILENAME).write_text(
            "id,name,acronym\n42,Anterior left,AL\n43,Next ML voxel,NM\n\n",
            encoding="utf-8",
        )
        volume = directory / ATLAS_FILENAME
        header = bytearray(352)
        struct.pack_into("<i", header, 0, 348)
        struct.pack_into("<4h", header, 40, 3, *SHAPE)
        struct.pack_into("<2h", header, 70, 512, 16)
        struct.pack_into("<f", header, 108, 352)
        header[344:348] = b"n+1\x00"
        with volume.open("wb") as handle:
            handle.write(header)
            handle.truncate(352 + 2 * math.prod(SHAPE))
            for voxel, region_id in [
                ((569, 0, 659), 42),
                ((568, 0, 659), 43),
                ((0, 399, 0), 728),
            ]:
                offset = 352 + 2 * (voxel[0] + SHAPE[0] * (voxel[1] + SHAPE[1] * voxel[2]))
                handle.seek(offset)
                handle.write(struct.pack("<H", region_id))
        self.directory = directory

    def test_axes_and_20_um_boundaries(self):
        with AtlasLookup(self.directory) as atlas:
            first = atlas.lookup(0, 0, 19.99)
            self.assertEqual(first.atlas_voxel, (569, 0, 659))
            self.assertEqual((first.region_id, first.name, first.status), (42, "Anterior left", "labeled"))
            self.assertEqual(atlas.lookup(0, 0, 20).region_id, 43)
            opposite = atlas.lookup(13199, 7999, 11399)
            self.assertEqual(opposite.atlas_voxel, (0, 399, 0))
            self.assertEqual(opposite.region_id, 728)
            self.assertEqual(opposite.status, "allen_ontology_fallback")
            self.assertEqual((opposite.name, opposite.acronym), ("arbor vitae", "arb"))
            self.assertEqual(atlas.lookup(100, 100, 100).status, "outside_brain")

    def test_rejects_out_of_bounds_and_nonfinite_positions(self):
        with AtlasLookup(self.directory) as atlas:
            for coords in [(-1, 0, 0), (13200, 0, 0), (0, 8000, 0), (0, 0, 11400), (math.nan, 0, 0)]:
                with self.subTest(coords=coords), self.assertRaises(ValueError):
                    atlas.lookup(*coords)


if __name__ == "__main__":
    unittest.main()
