import json
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np

from src.artifacts.collections import derive_content_means, load_unit_means
from src.artifacts.store import ArtifactStore, write_json


class ContentCollectionTests(unittest.TestCase):
    def test_selection_preserves_metadata_pooling_and_checkpoint_lineage(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = ArtifactStore(temporary)
            recipe = {"checkpoint_id": "synthetic", "sources": [],
                      "embedding_layout": {"rate_coordinate": 0, "content_coordinates": [1, 2, 3]}}
            values = np.arange(12, dtype=np.float32).reshape(3, 4)
            with store.build("collections", recipe) as (stage, parent):
                shard = stage / "session-00000"
                shard.mkdir()
                write_json(shard / "units.json", [{"unit_uid": str(i), "firing_rate_hz": float(i)} for i in range(3)])
                write_json(shard / "windows.json", [{"start": 0, "end": 20}])
                with h5py.File(shard / "embeddings.h5", "w") as out:
                    out["unit_mean"] = values
                    out["observation_count"] = np.ones(3, dtype=np.int64)
                write_json(stage / "index.json", {"dimension": 4, "sessions": [{"shard": shard.name}]})
            content = derive_content_means(store, parent)
            matrix, rows = load_unit_means(store, [content])
            np.testing.assert_array_equal(matrix, values[:, 1:])
            self.assertEqual([r["firing_rate_hz"] for r in rows], [0., 1., 2.])
            folder = store.path("collections", content)
            self.assertEqual(json.loads((folder / "index.json").read_text())["dimension"], 3)
            with h5py.File(folder / "session-00000/embeddings.h5") as data:
                np.testing.assert_array_equal(data["observation_count"], np.ones(3))
            self.assertEqual(content, derive_content_means(store, parent))
            np.testing.assert_array_equal(load_unit_means(store, [parent])[0], values)
            with self.assertRaisesRegex(ValueError, "content coordinates"):
                derive_content_means(store, content)


if __name__ == "__main__":
    unittest.main()
