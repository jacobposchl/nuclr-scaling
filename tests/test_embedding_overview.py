import csv
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from src.artifacts.embedding_overview import (
    add_coding_labels, add_population_metadata, analyze_embedding_overview,
    analyze_visual_cortex_overview, categorical_palettes, overview_datasets,
)
from src.artifacts.store import ArtifactStore, write_json


class PopulationMetadataTests(unittest.TestCase):
    def test_recording_means_and_probe_sizes_use_namespaced_groups(self):
        rows = [dict(unit_uid=str(i), session_uid=session, probe_uid=probe, firing_rate_hz=rate)
                for i, (session, probe, rate) in enumerate([
                    ("a/1", "a/1/7", 1.), ("a/1", "a/1/7", 3.), ("a/1", "a/1/8", 8.),
                    ("b/1", "b/1/7", 20.),
                ])]
        result = add_population_metadata(rows)
        self.assertEqual([r["population_mean_firing_rate_hz"] for r in result], [4., 4., 4., 20.])
        self.assertEqual([r["population_size"] for r in result], [3, 3, 3, 1])
        self.assertEqual([r["probe_population_size"] for r in result], [2, 2, 1, 1])
        self.assertTrue(all(r["cell_type"] == "Unknown" for r in result))
        self.assertNotIn("population_size", rows[0])

    def test_missing_rate_and_cross_recording_probe_collisions_are_rejected(self):
        row = dict(unit_uid="1", session_uid="a", probe_uid="probe7", firing_rate_hz=float("nan"))
        with self.assertRaisesRegex(ValueError, "finite"):
            add_population_metadata([row])
        row["firing_rate_hz"] = 1.
        with self.assertRaisesRegex(ValueError, "namespaced"):
            add_population_metadata([row, dict(row, unit_uid="2", session_uid="b")])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            add_population_metadata([row, row])


class CodingLabelTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "labels.csv"
        self.rows = [dict(dataset_id="allen_visual_coding", session_uid=f"allen_visual_coding/{session}",
                          unit_uid=f"allen_visual_coding/{session}/7", probe_uid=f"allen_visual_coding/{session}/8",
                          brain_region="VISp", cell_type="Unknown", joint_umap_1="1.234")
                     for session in (1, 2)]
        self.annotations = [dict(session_id=str(session), unit_id="7", probe_id="8",
                                 ecephys_structure_acronym="VISp", lolcat_celltype=label)
                            for session, label in ((2, "wt"), (1, "Pvalb"))]

    def write_annotations(self, annotations):
        with self.path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.annotations[0]))
            writer.writeheader()
            writer.writerows(annotations)

    def test_session_unit_join_preserves_behavior_and_coordinate_values(self):
        self.write_annotations(self.annotations)
        behavior = dict(self.rows[0], dataset_id="allen_visual_behavior", cell_type="Sst")
        result = add_coding_labels([*self.rows, behavior], self.path)
        self.assertEqual([r["cell_type"] for r in result], ["Pvalb", "Unlabeled", "Sst"])
        self.assertEqual([r["lolcat_celltype"] for r in result], ["Pvalb", "wt", ""])
        self.assertTrue(all(r["joint_umap_1"] == "1.234" for r in result))
        self.assertEqual(self.rows[0]["cell_type"], "Unknown")

    def test_incomplete_ambiguous_or_inconsistent_annotations_are_rejected(self):
        for annotations, message in [
            (self.annotations[:1], "Missing"),
            (self.annotations + self.annotations[:1], "Duplicate"),
            ([dict(r, probe_id="99") for r in self.annotations], "mismatch"),
            ([dict(r, ecephys_structure_acronym="VISl") for r in self.annotations], "mismatch"),
            ([dict(r, lolcat_celltype="") for r in self.annotations], "Unrecognized"),
        ]:
            with self.subTest(message=message):
                self.write_annotations(annotations)
                with self.assertRaisesRegex(ValueError, message):
                    add_coding_labels(self.rows, self.path)

    def test_all_region_probe_channel_unit_ids_join_numeric_csv_ids(self):
        self.write_annotations(self.annotations)
        rows = [dict(row, unit_uid=row["unit_uid"].rsplit("/", 1)[0] + "/8%2F12%2F7")
                for row in self.rows]
        result = add_coding_labels(rows, self.path)
        self.assertEqual([row["cell_type"] for row in result], ["Pvalb", "Unlabeled"])
        rows[0]["unit_uid"] = "allen_visual_coding/1/99%2F12%2F7"
        with self.assertRaisesRegex(ValueError, "probe prefix"):
            add_coding_labels(rows, self.path)

    def test_many_brain_regions_have_distinct_colors(self):
        rows = [dict(dataset_id="allen_visual_behavior", subject_uid="mouse", cell_type="Unknown",
                     brain_region=f"region_{i}", probe_uid="probe") for i in range(65)]
        palette = categorical_palettes(rows)["brain_region"]
        self.assertEqual(len(set(palette.values())), 65)

    def test_ibl_labels_remain_unavailable_and_have_a_dataset_color(self):
        self.write_annotations(self.annotations)
        row = dict(self.rows[0], dataset_id="ibl_brain_wide_bench_2026", subject_uid="ibl/mouse")
        result = add_coding_labels([row], self.path)
        self.assertEqual(result[0]["cell_type"], "Unlabeled")
        self.assertEqual(result[0]["cell_type_source"], "not_available")
        self.assertEqual(overview_datasets(result), {"ibl_brain_wide_bench_2026": "IBL"})
        self.assertIn("ibl_brain_wide_bench_2026", categorical_palettes(result)["dataset_id"])


class ThreeDatasetOverviewTests(unittest.TestCase):
    def test_full_row_limit_and_all_three_separate_fits(self):
        datasets = ["allen_visual_behavior", "allen_visual_coding", "ibl_brain_wide_bench_2026"]
        rows = [dict(dataset_id=dataset, unit_uid=f"{dataset}/{i}", session_uid=dataset,
                     subject_uid=dataset, probe_uid=dataset, brain_region="VISp",
                     cell_type="Unknown", firing_rate_hz=float(i + 1))
                for dataset in datasets for i in range(3)]
        embeddings = np.arange(18, dtype=np.float32).reshape(9, 2)
        cfg = {"fit_selection": {}, "show_selection": {}, "standardize": False,
               "max_rows": 200000, "seed": 42, "umap": {"n_neighbors": 2}}
        with tempfile.TemporaryDirectory() as temporary:
            store = ArtifactStore(temporary)
            with store.build("collections", {"operation": "extract_v1", "sources": []}) as (_, collection):
                pass
            with store.build("analyses", {"operation": "visualization_v1", "config": cfg,
                                          "collections": [collection]}) as (stage, source):
                write_json(stage / "index.json", {"methods": {"umap": {"coordinates": "coordinates.csv"}}})
                with (stage / "coordinates.csv").open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=["unit_uid", "firing_rate_hz", "coordinate_1", "coordinate_2"])
                    writer.writeheader()
                    writer.writerows(dict(unit_uid=row["unit_uid"], firing_rate_hz=row["firing_rate_hz"],
                                          coordinate_1=xy[0], coordinate_2=xy[1])
                                     for row, xy in zip(rows, embeddings))
            umap = Mock(side_effect=lambda **kwargs: SimpleNamespace(fit_transform=lambda x: x))
            with patch("src.artifacts.embedding_overview.load_unit_means", return_value=(embeddings, rows)) as load, \
                    patch("src.artifacts.embedding_overview.plot_overview") as plot, \
                    patch.dict("sys.modules", {"umap": SimpleNamespace(UMAP=umap)}):
                analysis = analyze_embedding_overview(store, source)
            load.assert_called_once_with(store, [collection], max_rows=200000)
            self.assertEqual(umap.call_count, 3)
            self.assertEqual([len(call.args[2]) for call in plot.call_args_list], [9, 3, 3, 3])
            index = json.loads((store.path("analyses", analysis) / "index.json").read_text())
            self.assertEqual(index["datasets"], dict.fromkeys(datasets, 3))
            self.assertEqual(index["plots"][-1], "ibl_brain_wide_bench_2026_umap_panels.png")
            # Earlier two-dataset figures retain the same dataset ordering.
            self.assertEqual(list(overview_datasets(rows[:6])), datasets[:2])

    def test_visual_cortex_refit_excludes_visceral_and_preserves_population_metadata(self):
        regions = ["VISp", "VISC5", "LGd", "VISp6a", "VIS", "SCig", "VISrl2/3"]
        rows = [dict(unit_uid=str(i), dataset_id="ibl_brain_wide_bench_2026", brain_region=region,
                     population_size=100, probe_population_size=50, population_mean_firing_rate_hz=7.,
                     joint_umap_1=-999, joint_umap_2=-999, within_dataset_umap_1=-999)
                for i, region in enumerate(regions)]
        expected = [6, 4, 3, 0]
        embeddings = np.arange(14, dtype=np.float32).reshape(7, 2)
        with tempfile.TemporaryDirectory() as temporary:
            store = ArtifactStore(temporary)
            recipe = {"operation": "embedding_metadata_overview_v1", "collections": ["a" * 64],
                      "seed": 42, "umap": {"n_neighbors": 2}}
            with store.build("analyses", recipe) as (stage, source):
                write_json(stage / "index.json", {
                    "coordinates": "coordinates.csv", "category_colors": "category_colors.csv",
                    "neurons": 7, "population_definition": "All recording units",
                    "probe_definition": "All probe units", "pooling": {}})
                with (stage / "coordinates.csv").open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
                (stage / "category_colors.csv").write_text("field,category,color\n")

            def load_selected(_store, collections, max_rows, row_filter):
                self.assertEqual(max_rows, 7)
                selected = [i for i in reversed(range(7)) if row_filter(rows[i])]
                self.assertEqual(selected, expected)
                return embeddings[selected], [rows[i] for i in selected]

            fit = Mock(side_effect=lambda x: x + 100)
            umap = Mock(return_value=SimpleNamespace(fit_transform=fit))
            with patch("src.artifacts.embedding_overview.load_unit_means", side_effect=load_selected), \
                    patch("src.artifacts.embedding_overview.plot_overview") as plot, \
                    patch.dict("sys.modules", {"umap": SimpleNamespace(UMAP=umap)}):
                analysis = analyze_visual_cortex_overview(store, source)
            np.testing.assert_array_equal(fit.call_args.args[0], embeddings[expected])
            self.assertEqual(umap.call_args.kwargs["n_jobs"], 1)
            self.assertEqual(len(plot.call_args.args[2]), 4)
            self.assertEqual(len(plot.call_args.args[4]), 7)
            with (store.path("analyses", analysis) / "coordinates.csv").open() as stream:
                saved = list(csv.DictReader(stream))
            self.assertEqual([r["unit_uid"] for r in saved], [str(i) for i in expected])
            self.assertTrue(all(r["population_size"] == "100" and r["probe_population_size"] == "50"
                                and r["population_mean_firing_rate_hz"] == "7.0" for r in saved))
            self.assertNotIn("within_dataset_umap_1", saved[0])
            np.testing.assert_array_equal([float(r["joint_umap_1"]) for r in saved],
                                          embeddings[expected, 0] + 100)


if __name__ == "__main__":
    unittest.main()
