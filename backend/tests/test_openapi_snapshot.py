import json
import os
import unittest
from pathlib import Path

import api

SNAPSHOT_PATH = Path(__file__).resolve().parent / "openapi_snapshot.json"
UPDATE_VARIABLE = "UPDATE_OPENAPI_SNAPSHOT"


class OpenApiSnapshotTest(unittest.TestCase):
    def test_openapi_schema_matches_committed_snapshot(self):
        schema = api.app.openapi()
        if os.environ.get(UPDATE_VARIABLE) == "1":
            SNAPSHOT_PATH.write_text(
                json.dumps(schema, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        self.assertTrue(
            SNAPSHOT_PATH.exists(),
            f"OpenAPI snapshot is missing; run with {UPDATE_VARIABLE}=1 to generate it",
        )
        expected = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
        self.assertEqual(
            schema,
            expected,
            f"OpenAPI schema changed; review it, then run with {UPDATE_VARIABLE}=1 to update",
        )


if __name__ == "__main__":
    unittest.main()
