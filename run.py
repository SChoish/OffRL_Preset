#!/usr/bin/env python3
"""레포 루트에서 학습 실행. `pip install -e .` 없이 `python run.py ...` 만으로 동작."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from offrl.cli.main import main

if __name__ == "__main__":
    raise SystemExit(main())
