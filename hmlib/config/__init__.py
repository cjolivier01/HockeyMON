"""Configuration helpers and YAML-backed defaults for HockeyMON.

This package intentionally re-exports the public API from `hmlib/config.py`.

Reason: the repository contains both `hmlib/config.py` (Python code) and the
`hmlib/config/` directory (YAML and model config files). Some build/test
environments (notably Bazel with generated `__init__.py` files) will resolve
`import hmlib.config` to the directory, shadowing the `config.py` module.

By making `hmlib/config/` a real package and explicitly loading `config.py`
as an implementation module, `from hmlib.config import ...` stays stable.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

_IMPL_MODULE_NAME = "hmlib._config_impl"


def _load_impl() -> ModuleType:
    existing = sys.modules.get(_IMPL_MODULE_NAME)
    if isinstance(existing, ModuleType):
        return existing

    impl_path = Path(__file__).resolve().parent.parent / "config.py"
    spec = importlib.util.spec_from_file_location(_IMPL_MODULE_NAME, impl_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load config implementation module at {impl_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_IMPL_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


_impl = _load_impl()


def __getattr__(name: str) -> Any:
    return getattr(_impl, name)


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(dir(_impl)))
