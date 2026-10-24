import json
import unittest
from pathlib import Path

import api


class OpenApiSnapshotTest(unittest.TestCase):
    def test_openapi_schema_matches_committed_snapshot(self):
        snapshot_path = (
            Path(__file__).resolve().parents[1] / "contracts" / "openapi.json"
        )
        self.assertTrue(
            snapshot_path.exists(),
            "OpenAPI snapshot is missing; run scripts/openapi_snapshot.py",
        )
        expected = json.loads(snapshot_path.read_text(encoding="utf-8"))
        self.assertEqual(api.app.openapi(), expected)


if __name__ == "__main__":
    unittest.main()
