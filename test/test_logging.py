#!/usr/bin/env python
# Copyright (c) 2026 GitHub@Apex_rl Developer
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Regression tests for training log structure."""

import importlib.util
import sys
import time
import types
import warnings
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

if "gymnasium" not in sys.modules and importlib.util.find_spec("gymnasium") is None:
    gymnasium_stub = types.ModuleType("gymnasium")
    gymnasium_stub.spaces = SimpleNamespace(Space=object)
    sys.modules["gymnasium"] = gymnasium_stub

if "tensordict" not in sys.modules and importlib.util.find_spec("tensordict") is None:
    tensordict_stub = types.ModuleType("tensordict")

    class TensorDict:  # pragma: no cover - import shim only
        pass

    tensordict_stub.TensorDict = TensorDict
    sys.modules["tensordict"] = tensordict_stub

from apexrl.agent.on_policy_runner import OnPolicyRunner
from apexrl.algorithms.ppo.config import PPOConfig
from apexrl.utils.logger import Logger, SwanLabLogger, WandbLogger


class DummyLogger:
    """Minimal logger that records scalar batches."""

    def __init__(self):
        self.calls = []

    def log_scalars(self, scalars, step):
        self.calls.append((dict(scalars), step))


class _ConfigRecorder:
    def __init__(self):
        self.updates = []

    def update(self, config):
        self.updates.append(dict(config))


class FakeWandbModule(types.ModuleType):
    def __init__(self):
        super().__init__("wandb")
        self.init_calls = []
        self.log_calls = []
        self.finish_calls = 0
        self.config = _ConfigRecorder()

    def init(self, **kwargs):
        self.init_calls.append(dict(kwargs))

    def log(self, data, step=None):
        self.log_calls.append((dict(data), step))

    def finish(self):
        self.finish_calls += 1

    class Histogram:
        def __init__(self, values):
            self.values = values

    class Image:
        def __init__(self, image):
            self.image = image

    class Video:
        def __init__(self, video, fps=30):
            self.video = video
            self.fps = fps


class FakeSwanLabModule(types.ModuleType):
    def __init__(self):
        super().__init__("swanlab")
        self.init_calls = []
        self.log_calls = []
        self.finish_calls = 0
        self.config = _ConfigRecorder()

    def init(self, **kwargs):
        self.init_calls.append(dict(kwargs))

    def log(self, data, step=None):
        self.log_calls.append((dict(data), step))

    def finish(self):
        self.finish_calls += 1

    class Image:
        def __init__(self, image):
            self.image = image


def test_on_policy_runner_logging_defaults_avoid_redundant_metrics():
    """Default runner logging should avoid duplicated iteration-axis metrics."""
    runner = OnPolicyRunner.__new__(OnPolicyRunner)
    runner.cfg = PPOConfig()
    runner.env = SimpleNamespace(num_envs=2)
    runner.logger = DummyLogger()
    runner.total_timesteps = 128
    runner.iteration = 3
    runner.log_interval = runner.cfg.log_interval
    runner.reward_components = {}
    runner.log_buffers = {}
    runner.agent = SimpleNamespace(
        rollout_buffer=SimpleNamespace(
            advantages=torch.tensor([1.0, 2.0]),
            values=torch.tensor([3.0, 4.0]),
            returns=torch.tensor([5.0, 6.0]),
        ),
        episode_rewards=[10.0, 20.0],
        episode_lengths=[30.0, 40.0],
    )

    runner._log_iteration(
        iteration=3,
        total_iters=100,
        rollout_stats={
            "rollout/mean_reward": 1.5,
            "rollout/mean_episode_reward": 15.0,
            "rollout/completed_episodes": 2.0,
        },
        update_stats={
            "train/policy_loss": 0.1,
            "train/value_loss": 0.2,
        },
        last_log_time=time.time() - 1.0,
    )

    logged_keys = {key for scalars, _step in runner.logger.calls for key in scalars}

    assert "time/iteration" not in logged_keys
    assert "train_vs_iter/policy_loss" not in logged_keys
    assert "episode_vs_iter/mean_reward" not in logged_keys
    assert "rollout/mean_episode_reward" not in logged_keys
    assert "episode/mean_reward" in logged_keys
    assert "rollout/completed_episodes" in logged_keys


def test_wandb_logger_backend_integration(monkeypatch, tmp_path):
    """Logger factory should initialize and drive the wandb backend."""
    fake_wandb = FakeWandbModule()
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    logger = Logger.create(
        "wandb",
        experiment_name="ppo_exp",
        log_dir=str(tmp_path),
        project="apexrl",
        entity="team",
        tags=["ppo"],
    )

    assert isinstance(logger, WandbLogger)
    assert fake_wandb.init_calls == [
        {
            "project": "apexrl",
            "entity": "team",
            "name": "ppo_exp",
            "dir": str(tmp_path),
            "tags": ["ppo"],
            "resume": None,
        }
    ]

    logger.log_scalars({"train/loss": 0.1, "episode/reward": 10.0}, step=12)
    logger.log_config({"lr": 3e-4, "algo": "ppo"})
    logger.close()

    assert fake_wandb.log_calls == [
        ({"train/loss": 0.1, "episode/reward": 10.0}, 12)
    ]
    assert fake_wandb.config.updates == [{"lr": 3e-4, "algo": "ppo"}]
    assert fake_wandb.finish_calls == 1


def test_swanlab_logger_backend_integration(monkeypatch, tmp_path):
    """Logger factory should initialize and drive the SwanLab backend."""
    fake_swanlab = FakeSwanLabModule()
    monkeypatch.setitem(sys.modules, "swanlab", fake_swanlab)

    logger = Logger.create(
        "swanlab",
        experiment_name="dqn_exp",
        log_dir=str(tmp_path),
        project="apexrl",
        experiment_description="debug run",
    )

    assert isinstance(logger, SwanLabLogger)
    assert fake_swanlab.init_calls == [
        {
            "project": "apexrl",
            "experiment_name": "dqn_exp",
            "description": "debug run",
            "logdir": str(tmp_path),
        }
    ]

    logger.log_scalars({"train/q_loss": 0.2}, step=7)
    logger.log_config({"buffer_size": 1024})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        logger.log_video("rollout/video", "clip.mp4", step=7, fps=24)
    logger.close()

    assert fake_swanlab.log_calls[0] == ({"train/q_loss": 0.2}, 7)
    assert fake_swanlab.config.updates == [{"buffer_size": 1024}]
    assert fake_swanlab.log_calls[1] == (
        {"rollout/video": "[Video at step 7]"},
        7,
    )
    assert fake_swanlab.finish_calls == 1
    assert caught
    assert "not fully supported" in str(caught[0].message)


if __name__ == "__main__":
    test_on_policy_runner_logging_defaults_avoid_redundant_metrics()
