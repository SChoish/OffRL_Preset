from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def load_yaml_config(path: Optional[Path]) -> Dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return dict(raw) if isinstance(raw, dict) else {}


__all__ = ["load_yaml_config"]
