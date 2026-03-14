"""
Load and resolve config.yaml.

Usage:
    from utils.config import load_config
    cfg = load_config()               # finds config.yaml walking up from cwd
    cfg = load_config("path/to/config.yaml")

All ${ENV_VAR:-default} placeholders in string values are resolved at load time.
"""

import os
import re
from pathlib import Path

import yaml


def _resolve(value: str) -> str:
    """Expand ${VAR:-default} shell-style placeholders."""
    def replacer(match):
        var, _, default = match.group(1).partition(":-")
        return os.environ.get(var, default)
    return re.sub(r"\$\{([^}]+)\}", replacer, value)


def _resolve_recursive(obj):
    if isinstance(obj, str):
        return _resolve(obj)
    if isinstance(obj, dict):
        return {k: _resolve_recursive(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_recursive(v) for v in obj]
    return obj


def _find_config(start: Path) -> Path:
    """Walk up from start looking for config.yaml."""
    for parent in [start, *start.parents]:
        candidate = parent / "config.yaml"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "config.yaml not found. Run from the project root or pass an explicit path."
    )


class Config:
    """Dot-access wrapper around a nested dict."""
    def __init__(self, data: dict):
        self._data = data

    def __getattr__(self, key):
        if key.startswith("_"):
            return super().__getattribute__(key)
        try:
            val = self._data[key]
        except KeyError:
            raise AttributeError(f"Config has no key '{key}'")
        return Config(val) if isinstance(val, dict) else val

    def __getitem__(self, key):
        return self._data[key]

    def get(self, key, default=None):
        val = self._data.get(key, default)
        return Config(val) if isinstance(val, dict) else val

    def as_dict(self) -> dict:
        return self._data

    # ── Derived paths (computed from config so nothing is hardcoded) ──────────

    @property
    def workspace(self) -> Path:
        return Path(self._data["paths"]["workspace"])

    @property
    def data_dir(self) -> Path:
        return self.workspace / "data"

    @property
    def models_dir(self) -> Path:
        return self.workspace / "models"

    @property
    def model_name(self) -> str:
        return self._data["model"]["name"]

    @property
    def base_model(self) -> str:
        return self._data["model"]["base"]

    @property
    def adapter_dir(self) -> Path:
        return self.models_dir / self.model_name

    @property
    def merged_dir(self) -> Path:
        return self.models_dir / f"{self.model_name}_merged"

    @property
    def gguf_dir(self) -> Path:
        return self.models_dir / f"{self.model_name}_gguf"

    @property
    def gguf_file(self) -> Path:
        quant = self._data["convert"]["quantization"].upper()
        return self.gguf_dir / f"{self.model_name}.{quant}.gguf"

    @property
    def ollama_model_name(self) -> str:
        """Ollama model tag — defaults to model.name, override with OLLAMA_MODEL env var."""
        return os.environ.get("OLLAMA_MODEL", self.model_name)

    @property
    def train_file(self) -> Path:
        return self.data_dir / "train.jsonl"

    @property
    def val_file(self) -> Path:
        return self.data_dir / "val.jsonl"

    @property
    def train_sft_file(self) -> Path:
        return self.data_dir / "train_sft.jsonl"

    @property
    def val_sft_file(self) -> Path:
        return self.data_dir / "val_sft.jsonl"


def load_config(path: str | Path | None = None) -> Config:
    if path is None:
        path = _find_config(Path.cwd())
    raw = yaml.safe_load(Path(path).read_text())
    resolved = _resolve_recursive(raw)
    return Config(resolved)
