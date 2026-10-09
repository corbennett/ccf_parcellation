import io
import json
import math
import struct
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch
from urllib.request import Request

from ccf_parcellation import AtlasLookup, main
from ccf_parcellation.lookup import (
    ATLAS_FILENAME,
    FIGSHARE_FILES_URL,
    ONTOLOGY_FILENAME,
    SHAPE,
)


class FakeResponse(io.BytesIO):
    def __init__(self, data, status=200, headers=None):
        super().__init__(data)
        self.status = status
        self.headers = headers or {}


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

    def _remote_response(self, request, timeout):
        atlas_path = self.directory / ATLAS_FILENAME
        if request == FIGSHARE_FILES_URL:
            files = [
                {"name": ATLAS_FILENAME, "size": atlas_path.stat().st_size,
                 "download_url": "https://example.org/atlas.nii"},
                {"name": ONTOLOGY_FILENAME, "size": (self.directory / ONTOLOGY_FILENAME).stat().st_size,
                 "download_url": "https://example.org/ontology.csv"},
            ]
            return FakeResponse(json.dumps(files).encode())
        if request == "https://example.org/ontology.csv":
            return FakeResponse((self.directory / ONTOLOGY_FILENAME).read_bytes())
        self.assertIsInstance(request, Request)
        self.assertEqual(request.full_url, "https://example.org/atlas.nii")
        byte_range = request.get_header("Range")
        self.remote_ranges.append(byte_range)
        start, end = map(int, byte_range.removeprefix("bytes=").split("-"))
        with atlas_path.open("rb") as handle:
            handle.seek(start)
            data = handle.read(end - start + 1)
        return FakeResponse(
            data, status=206,
            headers={"Content-Range": f"bytes {start}-{end}/{atlas_path.stat().st_size}"},
        )

    def test_remote_fallback_reads_only_requested_ranges(self):
        self.remote_ranges = []
        with patch("ccf_parcellation.lookup.urlopen", side_effect=self._remote_response):
            with AtlasLookup(self.directory / "absent", remote_if_missing=True) as atlas:
                self.assertEqual(atlas.lookup(0, 0, 0).region_id, 42)
                self.assertEqual(atlas.lookup(0, 0, 20).region_id, 43)
                self.assertEqual(atlas.lookup(13199, 7999, 11399).status, "allen_ontology_fallback")
        self.assertEqual(self.remote_ranges[0], "bytes=0-351")
        self.assertEqual(len(self.remote_ranges), 4)
        for byte_range in self.remote_ranges[1:]:
            start, end = map(int, byte_range.removeprefix("bytes=").split("-"))
            self.assertEqual(end, start + 1)

    def test_missing_atlas_file_uses_remote_copy(self):
        partial_dir = self.directory / "partial"
        partial_dir.mkdir()
        (partial_dir / ONTOLOGY_FILENAME).write_bytes(
            (self.directory / ONTOLOGY_FILENAME).read_bytes()
        )
        self.remote_ranges = []
        with patch("ccf_parcellation.lookup.urlopen", side_effect=self._remote_response):
            with AtlasLookup(partial_dir, remote_if_missing=True) as atlas:
                self.assertEqual(atlas.lookup(0, 0, 0).region_id, 42)
        self.assertEqual(len(self.remote_ranges), 2)

    def test_missing_local_copy_requires_opt_in(self):
        with self.assertRaises(FileNotFoundError):
            AtlasLookup(self.directory / "absent")

    def test_remote_option_prefers_complete_local_copy(self):
        with patch("ccf_parcellation.lookup.urlopen") as urlopen:
            with AtlasLookup(self.directory, remote_if_missing=True) as atlas:
                self.assertEqual(atlas.lookup(0, 0, 0).region_id, 42)
            urlopen.assert_not_called()

    def test_cli_uses_remote_copy_by_default_when_local_files_are_missing(self):
        self.remote_ranges = []
        output = io.StringIO()
        with patch("ccf_parcellation.lookup.urlopen", side_effect=self._remote_response):
            with redirect_stdout(output):
                exit_code = main([
                    "0", "0", "0", "--atlas-dir", str(self.directory / "absent")
                ])
        self.assertEqual(exit_code, 0)
        self.assertEqual(json.loads(output.getvalue())["region_id"], 42)

    def test_cli_local_only_fails_when_local_files_are_missing(self):
        error = io.StringIO()
        with redirect_stderr(error), self.assertRaisesRegex(SystemExit, "2"):
            main([
                "0", "0", "0", "--atlas-dir", str(self.directory / "absent"),
                "--local-only",
            ])
        self.assertIn("No such file or directory", error.getvalue())

    def test_remote_fallback_rejects_server_ignoring_range(self):
        self.remote_ranges = []

        def ignored_range(request, timeout):
            response = self._remote_response(request, timeout)
            if isinstance(request, Request):
                response.status = 200
            return response

        with patch("ccf_parcellation.lookup.urlopen", side_effect=ignored_range):
            with self.assertRaisesRegex(ValueError, "did not honor byte range"):
                AtlasLookup(self.directory / "absent", remote_if_missing=True)


if __name__ == "__main__":
    unittest.main()
