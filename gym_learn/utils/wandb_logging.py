"""Optional Weights & Biases logging helpers for Go2 training.

The training stack must keep working without wandb installed unless the user
explicitly passes ``--wandb``.  This module therefore imports wandb lazily and
keeps all tensor/numpy handling duck-typed so tests can exercise it without the
Isaac Gym/PyTorch runtime.
"""

from __future__ import annotations

import importlib
import numbers
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, Optional, Union


class WandbLogger:
    """Small optional adapter around ``wandb.init`` and ``wandb.log``."""

    def __init__(self, enabled: bool = False, wandb_module: Any = None) -> None:
        self.enabled = bool(enabled)
        self._wandb = wandb_module
        self._run = None
        self._active = False

    @property
    def active(self) -> bool:
        return self._active

    def init(
        self,
        *,
        project: str,
        entity: Optional[str] = None,
        name: Optional[str] = None,
        group: Optional[str] = None,
        mode: str = "online",
        config: Optional[Mapping[str, Any]] = None,
        dir: Optional[str] = None,
    ) -> "WandbLogger":
        """Initialize a W&B run when enabled; otherwise stay a no-op."""

        if not self.enabled or mode == "disabled":
            return self

        wandb = self._wandb or self._import_wandb()
        init_kwargs = {
            "project": project,
            "name": name,
            "config": self._sanitize_config(config or {}),
            "mode": mode,
            "dir": dir,
        }
        if entity:
            init_kwargs["entity"] = entity
        if group:
            init_kwargs["group"] = group

        self._run = wandb.init(**init_kwargs)
        self._wandb = wandb
        self._active = True
        return self

    def log(self, metrics: Mapping[str, Any], step: Optional[int] = None) -> None:
        """Log scalar metrics to W&B, silently skipping non-scalars."""

        if not self._active:
            return

        scalar_metrics = self._sanitize_metrics(metrics)
        if not scalar_metrics:
            return

        target = self._run if hasattr(self._run, "log") else self._wandb
        target.log(scalar_metrics, step=step)

    def log_prefixed(self, prefix: str, metrics: Mapping[str, Any], step: Optional[int] = None) -> None:
        """Log metrics with a slash-separated namespace prefix."""

        clean_prefix = prefix.strip("/")
        self.log({f"{clean_prefix}/{key}": value for key, value in metrics.items()}, step=step)

    def finish(self) -> None:
        """Close the W&B run if one was started."""

        if not self._active:
            return

        if hasattr(self._run, "finish"):
            self._run.finish()
        elif self._wandb is not None and hasattr(self._wandb, "finish"):
            self._wandb.finish()
        self._active = False

    @classmethod
    def _sanitize_config(cls, value: Any) -> Any:
        if value is None or isinstance(value, (str, bool, numbers.Integral, numbers.Real)):
            return value
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, Mapping):
            return {str(key): cls._sanitize_config(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls._sanitize_config(item) for item in value]
        if isinstance(value, set):
            return [cls._sanitize_config(item) for item in sorted(value, key=str)]
        if hasattr(value, "item"):
            try:
                return cls._sanitize_config(value.item())
            except (TypeError, ValueError, RuntimeError):
                pass
        return repr(value)

    @staticmethod
    def _import_wandb() -> Any:
        try:
            return importlib.import_module("wandb")
        except ImportError as exc:
            raise RuntimeError(
                "--wandb was passed, but the wandb package is not installed. "
                "Install it with `pip install wandb` or run without --wandb."
            ) from exc

    @classmethod
    def _sanitize_metrics(cls, metrics: Mapping[str, Any]) -> Dict[str, Union[float, int, bool]]:
        sanitized: Dict[str, Union[float, int, bool]] = {}
        cls._flatten_scalars(sanitized, "", metrics)
        return sanitized

    @classmethod
    def _flatten_scalars(cls, out: Dict[str, Union[float, int, bool]], prefix: str, metrics: Mapping[str, Any]) -> None:
        for key, value in metrics.items():
            name = f"{prefix}/{key}" if prefix else str(key)
            if isinstance(value, Mapping):
                cls._flatten_scalars(out, name, value)
                continue
            scalar = cls._to_scalar(value)
            if scalar is not None:
                out[name] = scalar

    @staticmethod
    def _to_scalar(value: Any) -> Optional[Union[float, int, bool]]:
        if isinstance(value, bool):
            return value
        if isinstance(value, numbers.Integral):
            return int(value)
        if isinstance(value, numbers.Real):
            return float(value)

        # PyTorch tensors and numpy arrays/scalars are handled by duck typing to
        # avoid importing heavy runtime packages in --help and unit tests.
        try:
            if hasattr(value, "detach"):
                value = value.detach()
            if hasattr(value, "float") and hasattr(value, "mean"):
                value = value.float().mean()
            elif hasattr(value, "mean") and not hasattr(value, "item"):
                value = value.mean()
            if hasattr(value, "item"):
                item = value.item()
                if isinstance(item, bool):
                    return item
                if isinstance(item, numbers.Integral):
                    return int(item)
                if isinstance(item, numbers.Real):
                    return float(item)
        except (TypeError, ValueError, RuntimeError):
            return None

        return None
