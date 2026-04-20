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

"""Unified logger interface supporting multiple backends.

This module provides a unified logging interface that supports:
- TensorBoard
- Weights & Biases (wandb)
- SwanLab

Usage:
    >>> from apexrl.utils.logger import Logger
    >>> logger = Logger(
    ...     backend="tensorboard", log_dir="./runs", experiment_name="ppo_exp"
    ... )
    >>> logger.log_scalar("reward/mean", 100.0, step=1000)
    >>> logger.log_scalars({"loss/policy": 0.1, "loss/value": 0.05}, step=1000)
    >>> logger.close()
"""

import os
import warnings
from abc import ABC, abstractmethod
from typing import Any


class BaseLogger(ABC):
    """Abstract base class for loggers."""

    def __init__(self, experiment_name: str, log_dir: str, **kwargs):
        """Initialize the logger.

        Args:
            experiment_name: Name of the experiment.
            log_dir: Directory to save logs.
            **kwargs: Additional arguments for specific backends.
        """
        self.experiment_name = experiment_name
        self.log_dir = log_dir
        self.kwargs = kwargs

    @abstractmethod
    def log_scalar(self, key: str, value: int | float, step: int | None = None) -> None:
        """Log a scalar value.

        Args:
            key: Metric name.
            value: Metric value.
            step: Global step (e.g., training iteration).
        """
        pass

    @abstractmethod
    def log_scalars(
        self, scalars: dict[str, int | float], step: int | None = None
    ) -> None:
        """Log multiple scalar values.

        Args:
            scalars: Dictionary of metric names and values.
            step: Global step (e.g., training iteration).
        """
        pass

    @abstractmethod
    def log_histogram(self, key: str, values: Any, step: int | None = None) -> None:
        """Log a histogram of values.

        Args:
            key: Metric name.
            values: Array of values.
            step: Global step (e.g., training iteration).
        """
        pass

    @abstractmethod
    def log_image(self, key: str, image: Any, step: int | None = None) -> None:
        """Log an image.

        Args:
            key: Image name.
            image: Image data (format depends on backend).
            step: Global step (e.g., training iteration).
        """
        pass

    @abstractmethod
    def log_video(
        self, key: str, video: Any, step: int | None = None, fps: int = 30
    ) -> None:
        """Log a video.

        Args:
            key: Video name.
            video: Video data (format depends on backend).
            step: Global step (e.g., training iteration).
            fps: Frames per second.
        """
        pass

    @abstractmethod
    def log_config(self, config: dict[str, Any]) -> None:
        """Log configuration/hyperparameters.

        Args:
            config: Dictionary of configuration parameters.
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Close the logger and release resources."""
        pass

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


class TensorBoardLogger(BaseLogger):
    """TensorBoard logger implementation."""

    def __init__(self, experiment_name: str, log_dir: str = "./runs", **kwargs):
        """Initialize TensorBoard logger.

        Args:
            experiment_name: Name of the experiment.
            log_dir: Directory to save TensorBoard logs.
            **kwargs: Additional arguments passed to SummaryWriter.
        """
        super().__init__(experiment_name, log_dir, **kwargs)

        from torch.utils.tensorboard import SummaryWriter

        self.writer = SummaryWriter(
            log_dir=os.path.join(log_dir, experiment_name), **kwargs
        )

    def log_scalar(self, key: str, value: int | float, step: int | None = None) -> None:
        """Log a scalar value to TensorBoard."""
        self.writer.add_scalar(key, value, global_step=step)

    def log_scalars(
        self, scalars: dict[str, int | float], step: int | None = None
    ) -> None:
        """Log multiple scalar values to TensorBoard."""
        for key, value in scalars.items():
            self.writer.add_scalar(key, value, global_step=step)

    def log_histogram(self, key: str, values: Any, step: int | None = None) -> None:
        """Log a histogram to TensorBoard."""
        self.writer.add_histogram(key, values, global_step=step)

    def log_image(self, key: str, image: Any, step: int | None = None) -> None:
        """Log an image to TensorBoard.

        Args:
            key: Image name.
            image: Image tensor of shape (C, H, W) or (N, C, H, W).
            step: Global step.
        """
        self.writer.add_image(key, image, global_step=step)

    def log_video(
        self, key: str, video: Any, step: int | None = None, fps: int = 30
    ) -> None:
        """Log a video to TensorBoard.

        Args:
            key: Video name.
            video: Video tensor of shape (N, T, C, H, W) or (T, C, H, W).
            step: Global step.
            fps: Frames per second.
        """
        self.writer.add_video(key, video, global_step=step, fps=fps)

    def log_config(self, config: dict[str, Any]) -> None:
        """Log configuration to TensorBoard as hyperparameters."""
        # Filter out non-serializable values
        hparams = {}
        metrics = {}
        for k, v in config.items():
            if isinstance(v, (int, float, str, bool)):
                hparams[k] = v
            elif (
                isinstance(v, (list, tuple))
                and len(v) > 0
                and isinstance(v[0], (int, float))
            ):
                hparams[k] = str(v)
            else:
                hparams[k] = str(v)

        # Add dummy metric for hparams to show in TensorBoard
        self.writer.add_hparams(hparams, metrics)

    def close(self) -> None:
        """Close the TensorBoard writer."""
        self.writer.close()


class WandbLogger(BaseLogger):
    """Weights & Biases (wandb) logger implementation."""

    def __init__(
        self,
        experiment_name: str,
        log_dir: str = "./wandb",
        project: str | None = None,
        entity: str | None = None,
        tags: list | None = None,
        resume: str | None = None,
        **kwargs,
    ):
        """Initialize wandb logger.

        Args:
            experiment_name: Name of the run.
            log_dir: Directory to save wandb logs.
            project: Wandb project name.
            entity: Wandb entity (username or team).
            tags: List of tags for the run.
            resume: Resume mode ("allow", "must", "never", or run_id).
            **kwargs: Additional arguments passed to wandb.init.
        """
        super().__init__(experiment_name, log_dir, **kwargs)

        import wandb

        self.wandb = wandb
        self.wandb.init(
            project=project,
            entity=entity,
            name=experiment_name,
            dir=log_dir,
            tags=tags,
            resume=resume,
            **kwargs,
        )

    def log_scalar(self, key: str, value: int | float, step: int | None = None) -> None:
        """Log a scalar value to wandb."""
        self.wandb.log({key: value}, step=step)

    def log_scalars(
        self, scalars: dict[str, int | float], step: int | None = None
    ) -> None:
        """Log multiple scalar values to wandb."""
        self.wandb.log(scalars, step=step)

    def log_histogram(self, key: str, values: Any, step: int | None = None) -> None:
        """Log a histogram to wandb."""
        import numpy as np

        if not isinstance(values, np.ndarray):
            values = np.array(values)

        self.wandb.log({key: self.wandb.Histogram(values)}, step=step)

    def log_image(self, key: str, image: Any, step: int | None = None) -> None:
        """Log an image to wandb.

        Args:
            key: Image name.
            image: PIL Image or numpy array (H, W, C) or (H, W).
            step: Global step.
        """
        self.wandb.log({key: self.wandb.Image(image)}, step=step)

    def log_video(
        self, key: str, video: Any, step: int | None = None, fps: int = 30
    ) -> None:
        """Log a video to wandb.

        Args:
            key: Video name.
            video: numpy array (T, H, W, C) or path to video file.
            step: Global step.
            fps: Frames per second.
        """
        self.wandb.log({key: self.wandb.Video(video, fps=fps)}, step=step)

    def log_config(self, config: dict[str, Any]) -> None:
        """Log configuration/hyperparameters to wandb."""
        self.wandb.config.update(config)

    def close(self) -> None:
        """Finish the wandb run."""
        self.wandb.finish()


class SwanLabLogger(BaseLogger):
    """SwanLab logger implementation."""

    def __init__(
        self,
        experiment_name: str,
        log_dir: str = "./swanlab",
        project: str | None = None,
        experiment_description: str | None = None,
        **kwargs,
    ):
        """Initialize SwanLab logger.

        Args:
            experiment_name: Name of the experiment.
            log_dir: Directory to save SwanLab logs.
            project: SwanLab project name.
            experiment_description: Description of the experiment.
            **kwargs: Additional arguments passed to swanlab.init.
        """
        super().__init__(experiment_name, log_dir, **kwargs)

        import swanlab

        self.swanlab = swanlab
        self.swanlab.init(
            project=project,
            experiment_name=experiment_name,
            description=experiment_description,
            logdir=log_dir,
            **kwargs,
        )

    def log_scalar(self, key: str, value: int | float, step: int | None = None) -> None:
        """Log a scalar value to SwanLab."""
        self.swanlab.log({key: value}, step=step)

    def log_scalars(
        self, scalars: dict[str, int | float], step: int | None = None
    ) -> None:
        """Log multiple scalar values to SwanLab."""
        self.swanlab.log(scalars, step=step)

    def log_histogram(self, key: str, values: Any, step: int | None = None) -> None:
        """Log a histogram to SwanLab."""
        import numpy as np

        if not isinstance(values, np.ndarray):
            values = np.array(values)

        # SwanLab supports histogram via summary statistics
        self.swanlab.log(
            {
                f"{key}/mean": values.mean(),
                f"{key}/std": values.std(),
                f"{key}/min": values.min(),
                f"{key}/max": values.max(),
                f"{key}/median": np.median(values),
            },
            step=step,
        )

    def log_image(self, key: str, image: Any, step: int | None = None) -> None:
        """Log an image to SwanLab.

        Args:
            key: Image name.
            image: PIL Image or numpy array (H, W, C).
            step: Global step.
        """
        self.swanlab.log({key: self.swanlab.Image(image)}, step=step)

    def log_video(
        self, key: str, video: Any, step: int | None = None, fps: int = 30
    ) -> None:
        """Log a video to SwanLab.

        Args:
            key: Video name.
            video: numpy array (T, H, W, C) or path to video file.
            step: Global step.
            fps: Frames per second.
        """
        # SwanLab may not support video directly, so log a text placeholder.
        warnings.warn(
            "SwanLab video logging is not fully supported. "
            f"Video '{key}' logged as text."
        )
        self.swanlab.log({key: f"[Video at step {step}]"}, step=step)

    def log_config(self, config: dict[str, Any]) -> None:
        """Log configuration/hyperparameters to SwanLab."""
        self.swanlab.config.update(config)

    def close(self) -> None:
        """Finish the SwanLab run."""
        self.swanlab.finish()


class MultiLogger(BaseLogger):
    """Logger that logs to multiple backends simultaneously."""

    def __init__(self, loggers: list):
        """Initialize with a list of loggers.

        Args:
            loggers: List of BaseLogger instances.
        """
        self.loggers = loggers

    def log_scalar(self, key: str, value: int | float, step: int | None = None) -> None:
        """Log a scalar value to all loggers."""
        for logger in self.loggers:
            logger.log_scalar(key, value, step)

    def log_scalars(
        self, scalars: dict[str, int | float], step: int | None = None
    ) -> None:
        """Log multiple scalar values to all loggers."""
        for logger in self.loggers:
            logger.log_scalars(scalars, step)

    def log_histogram(self, key: str, values: Any, step: int | None = None) -> None:
        """Log a histogram to all loggers."""
        for logger in self.loggers:
            logger.log_histogram(key, values, step)

    def log_image(self, key: str, image: Any, step: int | None = None) -> None:
        """Log an image to all loggers."""
        for logger in self.loggers:
            logger.log_image(key, image, step)

    def log_video(
        self, key: str, video: Any, step: int | None = None, fps: int = 30
    ) -> None:
        """Log a video to all loggers."""
        for logger in self.loggers:
            logger.log_video(key, video, step, fps)

    def log_config(self, config: dict[str, Any]) -> None:
        """Log configuration to all loggers."""
        for logger in self.loggers:
            logger.log_config(config)

    def close(self) -> None:
        """Close all loggers."""
        for logger in self.loggers:
            logger.close()


class Logger:
    """Factory class for creating loggers.

    This is the main entry point for logging in the library.
    """

    BACKENDS = {
        "tensorboard": TensorBoardLogger,
        "wandb": WandbLogger,
        "swanlab": SwanLabLogger,
    }

    @classmethod
    def create(
        cls,
        backend: str | list,
        experiment_name: str,
        log_dir: str = "./logs",
        **kwargs,
    ) -> BaseLogger:
        """Create a logger instance.

        Args:
            backend: Backend name(s). Can be "tensorboard", "wandb", "swanlab",
                     or a list for multiple backends.
            experiment_name: Name of the experiment.
            log_dir: Directory to save logs.
            **kwargs: Additional backend-specific arguments.

        Returns:
            A logger instance.

        Examples:
            >>> # Single backend
            >>> logger = Logger.create("tensorboard", "my_exp", "./runs")

            >>> # Multiple backends
            >>> logger = Logger.create(["tensorboard", "wandb"], "my_exp", "./logs")

            >>> # With backend-specific config
            >>> logger = Logger.create(
            ...     "wandb",
            ...     "my_exp",
            ...     project="rl_project",
            ...     entity="my_team"
            ... )
        """
        if isinstance(backend, list):
            loggers = []
            for b in backend:
                if b not in cls.BACKENDS:
                    raise ValueError(
                        f"Unknown backend: {b}. Available: {list(cls.BACKENDS.keys())}"
                    )
                try:
                    logger = cls.BACKENDS[b](experiment_name, log_dir, **kwargs)
                    loggers.append(logger)
                except ImportError as e:
                    warnings.warn(f"Failed to import {b}: {e}. Skipping.")

            if len(loggers) == 0:
                raise RuntimeError("No loggers were successfully created.")
            elif len(loggers) == 1:
                return loggers[0]
            else:
                return MultiLogger(loggers)
        else:
            if backend not in cls.BACKENDS:
                raise ValueError(
                    "Unknown backend: "
                    f"{backend}. Available: {list(cls.BACKENDS.keys())}"
                )
            return cls.BACKENDS[backend](experiment_name, log_dir, **kwargs)

    @classmethod
    def register_backend(cls, name: str, logger_class: type) -> None:
        """Register a custom backend.

        Args:
            name: Name of the backend.
            logger_class: Logger class inheriting from BaseLogger.
        """
        if not issubclass(logger_class, BaseLogger):
            raise ValueError("Logger class must inherit from BaseLogger")
        cls.BACKENDS[name] = logger_class


def get_logger(
    backend: str | list = "tensorboard",
    experiment_name: str = "experiment",
    log_dir: str = "./logs",
    **kwargs,
) -> BaseLogger:
    """Convenience function to create a logger.

    Args:
        backend: Backend name(s). Can be "tensorboard", "wandb", "swanlab",
                 or a list for multiple backends.
        experiment_name: Name of the experiment.
        log_dir: Directory to save logs.
        **kwargs: Additional backend-specific arguments.

    Returns:
        A logger instance.

    Examples:
        >>> from apexrl.utils.logger import get_logger
        >>>
        >>> # TensorBoard (default)
        >>> logger = get_logger()
        >>>
        >>> # Weights & Biases
        >>> logger = get_logger("wandb", project="my_project")
        >>>
        >>> # SwanLab
        >>> logger = get_logger("swanlab", project="my_project")
        >>>
        >>> # Multiple backends
        >>> logger = get_logger(["tensorboard", "wandb"])
        >>>
        >>> # Log metrics
        >>> logger.log_scalar("reward", 100.0, step=1000)
        >>> logger.log_scalars({"loss": 0.5, "entropy": 0.01}, step=1000)
        >>>
        >>> # Log hyperparameters
        >>> logger.log_config({"lr": 3e-4, "gamma": 0.99})
        >>>
        >>> # Close logger
        >>> logger.close()
    """
    return Logger.create(backend, experiment_name, log_dir, **kwargs)
