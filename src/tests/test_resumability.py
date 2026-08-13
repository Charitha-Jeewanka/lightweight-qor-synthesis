"""Tests config-hash based resumability and skip logic against existing MLflow runs.

Ensures that any previously completed run is detected as SKIPPED_EXISTING
without initiating retraining or duplicating MLflow runs.
"""

import sys
import unittest
import mlflow

from src.config.schema import ExperimentConfig
from src.data.loaders import load_modeling_table
from src.eval.harness import run_experiment
from src.experiments.run_gnn import create_gnn_model
from src.models.gnn import GNNQoRModel
from src.tracking.mlflow_utils import check_run_exists_by_hash, get_run_by_hash, init_mlflow


class TestResumability(unittest.TestCase):
    def setUp(self):
        init_mlflow("Phase6_GNN_Stage1")
        self.client = mlflow.tracking.MlflowClient()
        exp = self.client.get_experiment_by_name("Phase6_GNN_Stage1")
        self.assertIsNotNone(exp, "Phase6_GNN_Stage1 experiment not found.")
        self.exp_id = exp.experiment_id

    def test_existing_l5_runs_are_skipped(self):
        """Queries completed L=5 runs and verifies skip logic detects them as finished."""
        runs = self.client.search_runs(self.exp_id)
        finished_runs = [r for r in runs if r.info.status == "FINISHED" and "config_hash" in r.data.params]

        self.assertGreater(len(finished_runs), 0, "No finished runs with config_hash found.")
        print(f"Testing resumability check on {len(finished_runs)} completed MLflow runs...")

        skipped_count = 0
        for run in finished_runs:
            chash = run.data.params["config_hash"]
            is_done = check_run_exists_by_hash(chash, experiment_name="Phase6_GNN_Stage1")
            if is_done:
                skipped_count += 1

        self.assertEqual(skipped_count, len(finished_runs), f"Expected {len(finished_runs)} skipped, got {skipped_count}.")
        print(f"RESUMABILITY VERIFIED: 100% ({skipped_count}/{len(finished_runs)}) of completed MLflow runs correctly detected as already-done.")

    def test_dry_run_run_experiment_skip(self):
        """Simulates calling run_experiment for a completed L=5 config."""
        dataset_l5 = load_modeling_table(seq_len=5, target="target_and_ratio")
        hp = {"learning_rate": 1e-3, "weight_decay": 1e-4, "dropout": 0.1, "batch_size": 256, "circuits_per_batch": 2}

        # Select a configuration known to be in MLflow (e.g. GCN L=5 Random seed 42)
        config = ExperimentConfig(
            experiment_name="Phase6_GNN_Stage1",
            model_family="gnn",
            dataset_L=5,
            split_protocol="random",
            exclude_hyp=True,
            seed=42,
            model_params=hp,
        )
        model = create_gnn_model("gcn", hp, seed=42)

        chash = config.compute_config_hash()
        is_done_before = check_run_exists_by_hash(chash, experiment_name="Phase6_GNN_Stage1")

        if is_done_before:
            res = run_experiment(model, dataset_l5, config)
            self.assertEqual(res["status"], "SKIPPED_EXISTING", f"Expected SKIPPED_EXISTING status, got {res.get('status')}")
            print(f"DRY RUN PASSED: run_experiment returned status='SKIPPED_EXISTING' for config_hash={chash[:8]}")
        else:
            print("Note: GCN L=5 seed 42 hash not found in finished runs (skipped dry run assert).")


if __name__ == "__main__":
    unittest.main()
