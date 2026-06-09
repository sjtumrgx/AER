"""Tests for optional Weights & Biases observability wiring."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_wandb_logging_module():
    module_path = ROOT / "gym_learn" / "utils" / "wandb_logging.py"
    spec = importlib.util.spec_from_file_location("aer_wandb_logging", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FakeWandb:
    def __init__(self):
        self.init_calls = []
        self.log_calls = []
        self.finish_calls = 0
        self.config = {}

    def init(self, **kwargs):
        self.init_calls.append(kwargs)
        return self

    def log(self, metrics, step=None):
        self.log_calls.append((metrics, step))

    def finish(self):
        self.finish_calls += 1


class WandbObservabilityTest(unittest.TestCase):
    def test_train_help_exposes_wandb_flags_without_importing_isaacgym(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "train.py"), "--help"],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertIn("--wandb", result.stdout)
        self.assertIn("--wandb_project", result.stdout)
        self.assertIn("--wandb_entity", result.stdout)
        self.assertIn("--wandb_mode", result.stdout)

    def test_wandb_logger_initializes_only_when_enabled_and_logs_scalar_metrics(self):
        module = load_wandb_logging_module()
        fake = FakeWandb()
        logger = module.WandbLogger(enabled=True, wandb_module=fake)

        logger.init(
            project="aer-go2",
            entity="unitree",
            name="seed-0",
            mode="offline",
            config={"seed": 0, "nested": {"cfg": "adaptive_en"}},
            dir="checkpoints/train/seed-0",
        )
        logger.log({"train/rew_total": 1.5, "non_scalar": object()}, step=7)
        logger.finish()

        self.assertEqual(1, len(fake.init_calls))
        self.assertEqual("aer-go2", fake.init_calls[0]["project"])
        self.assertEqual("unitree", fake.init_calls[0]["entity"])
        self.assertEqual("seed-0", fake.init_calls[0]["name"])
        self.assertEqual("offline", fake.init_calls[0]["mode"])
        self.assertEqual({"seed": 0, "nested": {"cfg": "adaptive_en"}}, fake.init_calls[0]["config"])
        self.assertEqual({"train/rew_total": 1.5}, fake.log_calls[0][0])
        self.assertEqual(7, fake.log_calls[0][1])
        self.assertEqual(1, fake.finish_calls)

    def test_wandb_config_converts_non_serializable_values(self):
        module = load_wandb_logging_module()
        fake = FakeWandb()
        logger = module.WandbLogger(enabled=True, wandb_module=fake)
        opaque = object()

        logger.init(
            project="aer-go2",
            config={"seed": 0, "opaque": opaque, "items": ["ok", opaque]},
        )

        config = fake.init_calls[0]["config"]
        self.assertEqual(0, config["seed"])
        self.assertIsInstance(config["opaque"], str)
        self.assertEqual("ok", config["items"][0])
        self.assertIsInstance(config["items"][1], str)

    def test_training_entrypoint_wires_wandb_logger_into_runner(self):
        train_source = (ROOT / "scripts" / "train.py").read_text()
        runner_source = (ROOT / "gym_learn" / "ppo_cse" / "__init__.py").read_text()
        setup_source = (ROOT / "setup.py").read_text()
        readme = (ROOT / "README.md").read_text()

        self.assertIn("wandb_logger = WandbLogger(enabled=args.wandb)", train_source)
        self.assertIn("metrics_logger=wandb_logger", train_source)
        self.assertIn("self.metrics_logger.log_prefixed", runner_source)
        self.assertIn("ppo/", runner_source)
        self.assertIn("'wandb'", setup_source)
        self.assertIn("--wandb", readme)
        self.assertIn("wandb login", readme)

    def test_disabled_wandb_logger_is_noop(self):
        module = load_wandb_logging_module()
        fake = FakeWandb()
        logger = module.WandbLogger(enabled=False, wandb_module=fake)

        logger.init(project="aer-go2", config={"seed": 0})
        logger.log({"train/rew_total": 1.5}, step=1)
        logger.finish()

        self.assertEqual([], fake.init_calls)
        self.assertEqual([], fake.log_calls)
        self.assertEqual(0, fake.finish_calls)


if __name__ == "__main__":
    unittest.main(verbosity=2)
