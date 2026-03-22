"""한 번의 학습 실행(run)에 대한 결과 디렉터리·체크포인트·CSV/JSONL 정리.

Linux 전제: eval 직후 `training.log` / `eval.csv` 는 flush + os.fsync 로 다른 프로세스(tail 등)에 바로 보이게 한다.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch


def _flush_os(f) -> None:
    """유저 공간 버퍼 → 커널. Linux에서 tail -f / 다른 프로세스가 바로 보게."""
    f.flush()
    try:
        os.fsync(f.fileno())
    except OSError:
        pass


def parse_d4rl_env_name(env_id: str) -> Tuple[str, str]:
    """예: halfcheetah-medium-v2 -> (halfcheetah, medium_v2)."""
    parts = env_id.split("-")
    if len(parts) < 2:
        return parts[0] if parts else "env", "default"
    return parts[0], "_".join(parts[1:])


def default_run_slug(algo: str) -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + f"_{algo}"


def resolve_run_root(
    preset_root: Path,
    algo: str,
    d4rl_id: str,
    seed: int,
    *,
    output_root: Optional[str],
    log_dir: Optional[str],
    run_name: Optional[str],
) -> Optional[Path]:
    """
    단일 run_root 아래에 checkpoints/, logs/ 만 둔다.
    - log_dir 가 있으면 (구버전 호환): {preset_root}/{log_dir}/{slug}/
    - 없으면: {preset_root}/{output_root}/{algo}/{env}/{task}/seed_{seed}/{slug}/
    """
    slug = (run_name or "").strip() or default_run_slug(algo)
    oroot = (output_root or "").strip()

    if log_dir and str(log_dir).strip():
        p = Path(log_dir.strip())
        base = p.resolve() if p.is_absolute() else (preset_root / p).resolve()
        run_root = (base / slug).resolve()
    elif oroot:
        env_base, task = parse_d4rl_env_name(d4rl_id)
        run_root = (
            preset_root
            / oroot
            / algo
            / env_base
            / task
            / f"seed_{seed}"
            / slug
        ).resolve()
    else:
        return None

    (run_root / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_root / "logs").mkdir(parents=True, exist_ok=True)
    return run_root


def trainer_state_dict(algo: str, trainer: Any) -> Dict[str, Any]:
    if algo == "td3bc":
        return {
            "algo": algo,
            "step": int(getattr(trainer, "step", 0)),
            "actor": trainer.actor.state_dict(),
            "actor_target": trainer.actor_target.state_dict(),
            "q1": trainer.q1.state_dict(),
            "q2": trainer.q2.state_dict(),
            "q1_target": trainer.q1_target.state_dict(),
            "q2_target": trainer.q2_target.state_dict(),
        }
    return {"algo": algo, "note": "checkpoint not defined for this algo"}


@dataclass
class RunArtifacts:
    run_root: Path
    metrics_path: Path
    eval_csv_path: Path
    training_log_path: Path

    @classmethod
    def create(cls, run_root: Path) -> RunArtifacts:
        logs = run_root / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        return cls(
            run_root=run_root,
            metrics_path=logs / "metrics.jsonl",
            eval_csv_path=logs / "eval.csv",
            training_log_path=logs / "training.log",
        )

    def append_metrics(self, row: Dict[str, Any]) -> None:
        with self.metrics_path.open("a", encoding="utf-8", buffering=1) as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()

    def append_eval_row(
        self,
        step: int,
        return_mean: float,
        return_std: float,
        d4rl_norm: Optional[float],
    ) -> None:
        write_header = not self.eval_csv_path.is_file()
        with self.eval_csv_path.open("a", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["step", "return_mean", "return_std", "d4rl_norm"])
            w.writerow(
                [
                    step,
                    f"{return_mean:.6f}",
                    f"{return_std:.6f}",
                    "" if d4rl_norm is None else f"{d4rl_norm:.6f}",
                ]
            )
            _flush_os(f)

    def log_line(self, line: str) -> None:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        with self.training_log_path.open("a", encoding="utf-8", buffering=1) as f:
            f.write(f"[{ts}] {line}\n")
            _flush_os(f)

    def save_checkpoint(self, filename: str, algo: str, trainer: Any) -> Path:
        ckpt_dir = self.run_root / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        path = ckpt_dir / filename
        torch.save(trainer_state_dict(algo, trainer), path)
        return path

    def write_summary(self, payload: Dict[str, Any]) -> Path:
        path = self.run_root / "logs" / "summary.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
            _flush_os(f)
        return path
