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


class DummyLogger:
    """Minimal logger that records scalar batches."""

    def __init__(self):
        self.calls = []

    def log_scalars(self, scalars, step):
        self.calls.append((dict(scalars), step))


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


if __name__ == "__main__":
    test_on_policy_runner_logging_defaults_avoid_redundant_metrics()
