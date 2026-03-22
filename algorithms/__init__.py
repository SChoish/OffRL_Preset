"""알고리즘 구현 + `--algo` 레지스트리. 규약(`AlgorithmBase`)은 `offrl.core`."""

from importlib import import_module
from types import ModuleType
from typing import Dict

_ALGO_MODULES: Dict[str, str] = {
    "td3bc": "algorithms.td3bc",
}


def load_algo(name: str) -> ModuleType:
    key = name.strip().lower().replace("-", "_")
    if key not in _ALGO_MODULES:
        known = ", ".join(sorted(_ALGO_MODULES))
        raise ValueError(f"알 수 없는 알고리즘: {name!r}. 사용 가능: {known}")
    return import_module(_ALGO_MODULES[key])


def list_algorithms() -> tuple[str, ...]:
    return tuple(sorted(_ALGO_MODULES.keys()))


__all__ = ["load_algo", "list_algorithms"]
