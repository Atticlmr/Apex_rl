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

"""Configuration dataclass for recurrent PPO."""

from __future__ import annotations

from dataclasses import dataclass

from apexrl.algorithms.ppo.config import PPOConfig


@dataclass
class RecurrentPPOConfig(PPOConfig):
    """Configuration for PPO with recurrent actor and critic networks.

    Recurrent PPO keeps rollout data in time order and updates the networks on
    fixed-length sequence minibatches. All standard :class:`PPOConfig` fields are
    still available; this subclass adds sequence and recurrent-network settings.

    Args:
        sequence_length: Number of contiguous time steps per training sequence.
            ``num_steps`` must be divisible by this value.
        recurrent_minibatch_size: Number of sequences per optimizer minibatch.
            If ``None``, it is derived from ``minibatch_size // sequence_length``.
        rnn_hidden_size: Hidden dimension used by the built-in GRU networks.
        rnn_num_layers: Number of recurrent layers used by the built-in GRU
            networks.
        rnn_feature_dim: Optional feature dimension before the recurrent layer.
            If ``None``, the last value in ``actor_hidden_dims`` /
            ``critic_hidden_dims`` is used by the built-in GRU networks.
    """

    sequence_length: int = 16
    recurrent_minibatch_size: int | None = None
    rnn_hidden_size: int = 256
    rnn_num_layers: int = 1
    rnn_feature_dim: int | None = None

    def __post_init__(self):
        """Validate recurrent PPO configuration."""
        super().__post_init__()
        assert self.sequence_length > 0, "sequence_length must be positive"
        assert self.num_steps % self.sequence_length == 0, (
            "num_steps must be divisible by sequence_length"
        )
        assert self.rnn_hidden_size > 0, "rnn_hidden_size must be positive"
        assert self.rnn_num_layers > 0, "rnn_num_layers must be positive"
        if self.recurrent_minibatch_size is not None:
            assert self.recurrent_minibatch_size > 0, (
                "recurrent_minibatch_size must be positive"
            )

    def get_recurrent_minibatch_size(self, num_envs: int) -> int:
        """Return optimizer minibatch size measured in sequences.

        Args:
            num_envs: Number of parallel environments used to derive the default
                minibatch size.

        Returns:
            Number of sequences sampled per optimizer step.
        """
        if self.recurrent_minibatch_size is not None:
            return self.recurrent_minibatch_size
        transitions = self.get_minibatch_size(num_envs)
        return max(1, transitions // self.sequence_length)
