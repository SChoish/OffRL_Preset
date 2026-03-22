"""통합 학습 CLI. 구현체는 최상위 `algorithms.<algo>`, 규약은 `offrl.core`."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from algorithms import list_algorithms, load_algo

from offrl import ReplayBuffer, eval_actor, set_seed
from offrl.cli.artifacts import RunArtifacts, parse_d4rl_env_name, resolve_run_root
from offrl.config import load_yaml_config
from offrl.core import TransitionBatch


def _reexec_under_nohup_linux() -> None:
    """Linux 기본: nohup으로 실행. OFFRL_NO_NOHUP=1이면 생략."""
    if sys.platform != "linux":
        return
    if os.environ.get("OFFRL_NO_NOHUP") == "1":
        return
    if os.environ.get("OFFRL_INSIDE_NOHUP") == "1":
        return
    nohup = shutil.which("nohup") or "/usr/bin/nohup"
    if not os.path.isfile(nohup):
        return
    m = sys.modules.get("__main__")
    spec = getattr(m, "__spec__", None) if m else None
    if spec and spec.parent and spec.name and spec.name.endswith("__main__"):
        argv = [nohup, sys.executable, "-m", spec.parent, *sys.argv[1:]]
    else:
        argv = [nohup, sys.executable, *sys.argv]
    env = os.environ.copy()
    env["OFFRL_INSIDE_NOHUP"] = "1"
    os.execve(nohup, argv, env)


def find_preset_root() -> Path:
    if os.environ.get("OFFRL_ROOT"):
        return Path(os.environ["OFFRL_ROOT"]).resolve()
    cwd = Path.cwd()
    if (cwd / "config").is_dir():
        return cwd.resolve()
    for p in cwd.resolve().parents:
        if (p / "pyproject.toml").is_file():
            return p
    return cwd.resolve()


def numpy_batch_to_torch(batch: Tuple[np.ndarray, ...], device: torch.device) -> TransitionBatch:
    s, a, r, s2, d = batch
    return TransitionBatch(
        states=torch.as_tensor(s, device=device, dtype=torch.float32),
        actions=torch.as_tensor(a, device=device, dtype=torch.float32),
        rewards=torch.as_tensor(r, device=device, dtype=torch.float32),
        next_states=torch.as_tensor(s2, device=device, dtype=torch.float32),
        dones=torch.as_tensor(d, device=device, dtype=torch.float32),
    )


def _type_for_default(key: str, default: object):
    if key in ("device", "d4rl", "log_dir", "run_name", "output_root"):
        return str
    if isinstance(default, bool):
        return lambda x: str(x).lower() in ("1", "true", "yes")
    if isinstance(default, int):
        return int
    if isinstance(default, float):
        return float
    return str


def _add_algo_args(p: argparse.ArgumentParser, mod: Any) -> None:
    for key in mod.CLI_KEYS:
        default = mod.TRAIN_DEFAULTS[key]
        typ = _type_for_default(key, default)
        flag = f"--{key.replace('_', '-')}"
        p.add_argument(flag, dest=key, type=typ, default=argparse.SUPPRESS)


def _should_eval(step: int, total_steps: int, eval_every: int) -> bool:
    if eval_every <= 0:
        return step == total_steps
    return step % eval_every == 0 or step == total_steps


def _merge_cfg(mod: Any, config_path: Path, ns: Dict[str, Any]) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {**mod.TRAIN_DEFAULTS, **load_yaml_config(config_path)}
    for k in mod.CLI_KEYS:
        if k in ns:
            cfg[k] = ns[k]
    return cfg


def run_training(algo_name: str, cfg: Dict[str, Any], root: Path) -> None:
    mod = load_algo(algo_name)
    dev = cfg["device"]
    if dev is None or dev == "":
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(str(dev))
    steps = int(cfg["steps"])
    batch_size = int(cfg["batch_size"])
    seed = int(cfg["seed"])
    d4rl_id = str(cfg.get("d4rl") or "").strip()
    if not d4rl_id:
        raise SystemExit("d4rl 환경 ID가 비어 있습니다. config의 d4rl 또는 --d4rl 로 지정하세요.")
    set_seed(seed)

    out_raw = cfg.get("output_root")
    out_root_s = "" if out_raw is None else str(out_raw).strip()
    log_dir_raw = cfg.get("log_dir")
    log_dir_s = str(log_dir_raw).strip() if log_dir_raw else ""
    run_name_raw = cfg.get("run_name")
    run_name_opt = str(run_name_raw).strip() if run_name_raw else None

    run_root = resolve_run_root(
        root,
        algo_name,
        d4rl_id,
        seed,
        output_root=out_root_s if out_root_s else None,
        log_dir=log_dir_s if log_dir_s else None,
        run_name=run_name_opt,
    )
    art: Optional[RunArtifacts] = RunArtifacts.create(run_root) if run_root else None
    eval_history: List[Dict[str, Any]] = []
    checkpoint_every = max(0, int(cfg.get("checkpoint_every", 0)))

    try:
        import d4rl  # noqa: F401
        import gym
    except ImportError as e:
        raise SystemExit("pip install gym d4rl") from e
    env = gym.make(d4rl_id)
    ds = d4rl.qlearning_dataset(env)
    state_dim = ds["observations"].shape[1]
    action_dim = ds["actions"].shape[1]
    max_action = float(env.action_space.high[0])
    buf = ReplayBuffer(state_dim, action_dim, len(ds["observations"]))
    buf.load_d4rl_dataset(ds)
    env.close()
    eval_env_id = d4rl_id

    trainer = mod.build_trainer(
        state_dim,
        action_dim,
        max_action,
        device,
        gamma=float(cfg["gamma"]),
        tau=float(cfg["tau"]),
        policy_noise=float(cfg["policy_noise"]),
        noise_clip=float(cfg["noise_clip"]),
        policy_delay=int(cfg["policy_delay"]),
        alpha=float(cfg["alpha"]),
        lr_actor=float(cfg["lr_actor"]),
        lr_critic=float(cfg["lr_critic"]),
    )
    algo = mod.Algorithm()
    log_every_ratio = max(1, int(cfg.get("log_every_ratio", 20)))
    log_every = max(1, steps // log_every_ratio)
    eval_episodes = int(cfg.get("eval_episodes", 0))
    eval_every = int(cfg.get("eval_every", 0))

    env_base, task = parse_d4rl_env_name(d4rl_id)
    if art:
        art.log_line(
            f"start algo={algo_name} env={d4rl_id} seed={seed} steps={steps} "
            f"eval_every={eval_every} eval_episodes={eval_episodes}"
        )

    def maybe_eval(step: int) -> None:
        if eval_episodes <= 0 or not eval_env_id:
            return
        import gym

        env = gym.make(eval_env_id)
        rews = eval_actor(env, trainer.actor, str(device), eval_episodes, seed + step)
        raw_mean = float(rews.mean())
        raw_std = float(rews.std()) if rews.size > 1 else 0.0
        d4rl_norm: Optional[float] = None
        if hasattr(env, "get_normalized_score"):
            try:
                d4rl_norm = float(env.get_normalized_score(raw_mean) * 100.0)
            except Exception:
                d4rl_norm = None
        env.close()
        metrics = {
            "eval_return_mean": raw_mean,
            "eval_return_std": raw_std,
        }
        if d4rl_norm is not None:
            metrics["eval_d4rl_normalized"] = d4rl_norm

        print("---------------------------------------")
        print(f"[Eval] step {step}")
        print(f"  return_mean={raw_mean:.3f}  return_std={raw_std:.3f}")
        if d4rl_norm is not None:
            print(f"  d4rl_normalized={d4rl_norm:.1f} (x100 scale)")
        print("---------------------------------------")

        row = {"step": step, **metrics}
        eval_history.append(dict(row))
        if art:
            art.append_metrics(row)
            art.append_eval_row(step, raw_mean, raw_std, d4rl_norm)
            art.log_line(
                f"eval step={step} mean={raw_mean:.4f} std={raw_std:.4f}"
                + (f" d4rl_norm={d4rl_norm:.4f}" if d4rl_norm is not None else "")
            )

    for t in range(1, steps + 1):
        raw = buf.sample(batch_size)
        batch = numpy_batch_to_torch(raw, device)
        lq, lp = mod.train_step(algo, trainer, batch)
        if checkpoint_every > 0 and art and t % checkpoint_every == 0:
            p = art.save_checkpoint(f"step_{t}.pth", algo_name, trainer)
            art.log_line(f"checkpoint {p.name}")
            print(f"step {t:7d}  checkpoint -> {p}")
        if _should_eval(t, steps, eval_every):
            maybe_eval(t)
        if t % log_every == 0:
            m: Dict[str, Any] = {"step": t, "loss_q": float(lq)}
            if lp is not None:
                m["loss_pi"] = float(lp)
            extra = f"  L_pi={lp:.4f}" if lp is not None else ""
            print(f"step {t:6d}  L_q={lq:.4f}{extra}")
            if art:
                art.append_metrics(m)

    if art:
        final_p = art.save_checkpoint("final.pth", algo_name, trainer)
        art.log_line(f"final checkpoint {final_p.name}")
        summary = {
            "algo": algo_name,
            "d4rl": d4rl_id,
            "env_base": env_base,
            "task": task,
            "seed": seed,
            "steps": steps,
            "eval_every": eval_every,
            "eval_episodes": eval_episodes,
            "checkpoint_every": checkpoint_every,
            "run_root": str(run_root),
            "eval_history": eval_history,
            "checkpoints": {
                "final": str(final_p.relative_to(run_root)) if run_root else str(final_p),
            },
        }
        art.write_summary(summary)
        print(f"summary.json -> {(run_root / 'logs' / 'summary.json')}")
        print(f"final.pth    -> {final_p}")

    print("done — preset root:", root)


def main(argv: Optional[List[str]] = None) -> int:
    if argv is None:
        _reexec_under_nohup_linux()
    root = find_preset_root()
    argv = list(sys.argv[1:] if argv is None else argv)

    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--algo", type=str, default="td3bc")
    pre.add_argument("--config", type=Path, default=None)
    pre_args, _ = pre.parse_known_args(argv)

    try:
        mod = load_algo(pre_args.algo)
    except ValueError as e:
        print(e, file=sys.stderr)
        return 2

    cfg_default = root / "config" / f"{pre_args.algo}.yaml"
    default_config = pre_args.config if pre_args.config is not None else cfg_default

    p = argparse.ArgumentParser(
        description="offrl 통합 학습 실행",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--algo",
        type=str,
        default=pre_args.algo,
        choices=list_algorithms(),
        help="알고리즘 이름",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=default_config,
        help=f"YAML (기본: {cfg_default})",
    )
    _add_algo_args(p, mod)
    args = p.parse_args(argv)

    ns = vars(args).copy()
    config_path = ns.pop("config")
    algo = ns.pop("algo")
    cfg = _merge_cfg(mod, config_path, ns)
    run_training(algo, cfg, root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
