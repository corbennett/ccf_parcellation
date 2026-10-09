import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ccf_parcellation import download_atlas
from ccf_parcellation.lookup import ATLAS_FILENAME, ONTOLOGY_FILENAME


class DownloadAtlasTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.directory = Path(self.temp_dir.name)
        self.contents = {
            ATLAS_FILENAME: b"atlas bytes",
            ONTOLOGY_FILENAME: b"id,name,acronym\n1,Area,A\n",
        }
        self.urls = {
            "https://example.org/atlas": self.contents[ATLAS_FILENAME],
            "https://example.org/ontology": self.contents[ONTOLOGY_FILENAME],
        }
        self.metadata = [
            {
                "name": ATLAS_FILENAME,
                "size": len(self.contents[ATLAS_FILENAME]),
                "download_url": "https://example.org/atlas",
            },
            {
                "name": ONTOLOGY_FILENAME,
                "size": len(self.contents[ONTOLOGY_FILENAME]),
                "download_url": "https://example.org/ontology",
            },
        ]

    def _response(self, request, timeout):
        if isinstance(request, str) and request.startswith("https://api.figshare.com"):
            return io.BytesIO(json.dumps(self.metadata).encode())
        return io.BytesIO(self.urls[request])

    @patch("ccf_parcellation.download._validate_nifti")
    def test_downloads_both_required_files(self, validate_nifti):
        with patch("ccf_parcellation.download.urlopen", side_effect=self._response):
            paths = download_atlas(self.directory / "atlas")

        self.assertEqual([path.name for path in paths], [ATLAS_FILENAME, ONTOLOGY_FILENAME])
        for path in paths:
            self.assertEqual(path.read_bytes(), self.contents[path.name])
        validate_nifti.assert_called_once()

    @patch("ccf_parcellation.download._validate_nifti")
    def test_reuses_valid_existing_files(self, validate_nifti):
        atlas_dir = self.directory / "atlas"
        atlas_dir.mkdir()
        for filename, content in self.contents.items():
            (atlas_dir / filename).write_bytes(content)

        requested = []

        def response(request, timeout):
            requested.append(request)
            return self._response(request, timeout)

        with patch("ccf_parcellation.download.urlopen", side_effect=response):
            download_atlas(atlas_dir)

        self.assertEqual(len(requested), 1)
        validate_nifti.assert_called_once()

    @patch("ccf_parcellation.download._validate_nifti")
    def test_failed_download_does_not_replace_existing_file(self, validate_nifti):
        atlas_dir = self.directory / "atlas"
        atlas_dir.mkdir()
        destination = atlas_dir / ATLAS_FILENAME
        destination.write_bytes(b"old")
        self.urls["https://example.org/atlas"] = b"short"

        with patch("ccf_parcellation.download.urlopen", side_effect=self._response):
            with self.assertRaisesRegex(ValueError, "Incomplete download"):
                download_atlas(atlas_dir)

        self.assertEqual(destination.read_bytes(), b"old")
        self.assertFalse((atlas_dir / f".{ATLAS_FILENAME}.download").exists())


if __name__ == "__main__":
    unittest.main()
