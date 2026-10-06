"""
Unit Tests for the Exploratory Data Analysis (EDA) Module (src.eda).
"""

import json
import os
import shutil
import tempfile
import unittest
import numpy as np
import pandas as pd

from src.eda import (
    compute_shannon_entropy,
    analyze_partition,
    summarize_partitions,
    diagnose_data_quality,
    analyze_features,
    analyze_dataset,
    plot_class_distribution,
    plot_partition_heterogeneity,
    plot_feature_skewness,
    export_client_eda,
    export_scenario_eda_summary,
    export_dataset_audit,
)


class TestEDAModule(unittest.TestCase):
    def setUp(self):
        # 8 classes with clear skew
        self.sample_labels = np.array(
            [0] * 50 + [1] * 30 + [2] * 10 + [3] * 5 + [4] * 3 + [5] * 2 + [6] * 0 + [7] * 0,
            dtype=np.int64
        )
        self.class_names = ["Benign", "DDoS", "DoS", "Recon", "Spoof", "Mirai", "Web-based", "Brute-force"]

    def test_compute_shannon_entropy(self):
        # Uniform counts: maximum entropy
        uniform_counts = [10, 10, 10, 10]
        entropy, norm_entropy = compute_shannon_entropy(uniform_counts)
        self.assertAlmostEqual(norm_entropy, 1.0, places=2)

        # Single-class counts: zero entropy
        single_counts = [100, 0, 0, 0]
        entropy, norm_entropy = compute_shannon_entropy(single_counts)
        self.assertEqual(entropy, 0.0)
        self.assertEqual(norm_entropy, 0.0)

    def test_analyze_partition(self):
        eda = analyze_partition(self.sample_labels, class_names=self.class_names, client_id=0)

        self.assertEqual(eda["client_id"], 0)
        self.assertEqual(eda["total_samples"], 100)
        self.assertEqual(eda["num_classes"], 8)
        self.assertEqual(eda["dominant_class"], "Benign")
        self.assertEqual(eda["dominant_class_pct"], 50.0)
        self.assertIn("Web-based", eda["missing_classes"])
        self.assertIn("Brute-force", eda["missing_classes"])
        self.assertEqual(eda["class_counts"]["Benign"], 50)
        self.assertEqual(eda["class_percentages"]["DDoS"], 30.0)
        self.assertTrue(0.0 < eda["normalized_entropy"] < 1.0)

    def test_summarize_partitions(self):
        client_partitions = {
            0: np.arange(50),
            1: np.arange(50, 100)
        }
        df = summarize_partitions(client_partitions, self.sample_labels, class_names=self.class_names)

        self.assertEqual(len(df), 2)
        self.assertIn("client_id", df.columns)
        self.assertIn("total_samples", df.columns)
        self.assertIn("dominant_class", df.columns)
        self.assertIn("entropy", df.columns)
        self.assertIn("Benign_count", df.columns)
        self.assertIn("Benign_pct", df.columns)

    def test_diagnose_data_quality(self):
        # Clean data
        X_clean = np.random.randn(100, 5).astype(np.float32)
        report_clean = diagnose_data_quality(X_clean)
        self.assertTrue(report_clean["is_training_safe"])

        # Data with NaNs and Infs
        X_dirty = X_clean.copy()
        X_dirty[0, 0] = np.nan
        X_dirty[1, 1] = np.inf
        report_dirty = diagnose_data_quality(X_dirty)
        self.assertFalse(report_dirty["is_training_safe"])
        self.assertTrue(report_dirty["has_missing_values"])
        self.assertTrue(report_dirty["has_infinite_values"])

    def test_analyze_features(self):
        np.random.seed(42)
        X = np.random.exponential(scale=2.0, size=(500, 3)).astype(np.float32)
        stats_df = analyze_features(X, feature_names=["feat_a", "feat_b", "feat_c"])

        self.assertEqual(len(stats_df), 3)
        self.assertIn("skewness", stats_df.columns)
        self.assertIn("is_heavy_tailed", stats_df.columns)

    def test_analyze_dataset(self):
        X = np.random.randn(len(self.sample_labels), 4).astype(np.float32)
        report = analyze_dataset(X, self.sample_labels, class_names=self.class_names)

        self.assertIn("data_quality", report)
        self.assertIn("target_distribution", report)
        self.assertIn("heavy_tailed_features", report)

    def test_export_client_eda(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            client_dir = os.path.join(tmpdir, "client_0")
            eda = export_client_eda(
                client_dir=client_dir,
                y=self.sample_labels,
                class_names=self.class_names,
                client_id=0,
                generate_plot=True
            )

            self.assertTrue(os.path.exists(os.path.join(client_dir, "eda.json")))
            self.assertTrue(os.path.exists(os.path.join(client_dir, "class_distribution.png")))

            with open(os.path.join(client_dir, "eda.json")) as f:
                saved_eda = json.load(f)
            self.assertEqual(saved_eda["total_samples"], 100)
            self.assertEqual(saved_eda["dominant_class"], "Benign")

    def test_visualizations(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            eda = analyze_partition(self.sample_labels, class_names=self.class_names, client_id=1)
            plot_path = os.path.join(tmpdir, "dist.png")
            fig1 = plot_class_distribution(eda, save_path=plot_path)
            self.assertTrue(os.path.exists(plot_path))

            # Cross-client heterogeneity plot
            client_partitions = {0: np.arange(50), 1: np.arange(50, 100)}
            summary_df = summarize_partitions(client_partitions, self.sample_labels, class_names=self.class_names)
            het_path = os.path.join(tmpdir, "het.png")
            fig2 = plot_partition_heterogeneity(summary_df, class_names=self.class_names, save_path=het_path)
            self.assertTrue(os.path.exists(het_path))

            # Skewness plot
            X = np.random.randn(100, 5)
            stats_df = analyze_features(X)
            skew_path = os.path.join(tmpdir, "skew.png")
            fig3 = plot_feature_skewness(stats_df, top_k=3, save_path=skew_path)
            self.assertTrue(os.path.exists(skew_path))


if __name__ == "__main__":
    unittest.main()
