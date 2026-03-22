# offrl-preset

D4RL 형식 오프라인 데이터와 PyTorch 정책·Q·버퍼·학습 유틸을 묶은 **로컬 워크스페이스**입니다.

- **규약**: `offrl.core` — `AlgorithmBase`, `TransitionBatch` 등  
- **알고리즘**: 레포 루트 `algorithms/` (`td3bc` 등), `offrl`과 분리

## 실행

레포 루트에서:

```bash
pip install -r requirements.txt
python run.py --algo td3bc
python run.py --algo td3bc --d4rl halfcheetah-medium-v2
```

- 기본 태스크는 `config/td3bc.yaml`의 `d4rl`(`hopper-medium-v2`). 바꾸려면 `--d4rl` 또는 YAML 수정.
- **`config/<algo>.yaml`**이 있으면 로드하고, **CLI 인자가 YAML보다 우선**합니다. YAML이 없으면 각 알고리즘 모듈의 `TRAIN_DEFAULTS`만 씁니다.

### Linux / nohup

`python run.py`와 `python -m offrl`은 기본으로 **같은 명령을 `nohup`으로 한 번 더 실행**합니다. SSH를 끊어도(SIGHUP) 학습이 이어지고, 터미널에 stdout이 붙어 있으면 출력이 **`./nohup.out`**으로 이어지는 경우가 많습니다.

IDE 디버그·포그라운드만 쓰려면:

```bash
OFFRL_NO_NOHUP=1 python run.py --algo td3bc
```

### import 경로

`run.py`가 레포 루트를 `sys.path`에 넣으므로 **`pip install -e .` 없이** `import offrl`이 됩니다. VS Code로 이 폴더를 열면 `.vscode/settings.json`의 `PYTHONPATH`로 `python -m offrl`도 동일하게 동작합니다. 다른 경로에서 쓰려면 `pip install -e .` 또는 `OFFRL_ROOT`로 루트를 고정하세요. CLI는 `config/`·`pyproject.toml`을 기준으로 루트를 찾고, 서브디렉터리에서 실행해도 상위를 탐색합니다.

## 산출물

기본은 **`results/`** 아래 run당 한 폴더(`.gitignore`에 포함).

```
results/<algo>/<env>/<task>/seed_<seed>/<run_name>/
├── checkpoints/     # step_{N}.pth (checkpoint_every), final.pth
└── logs/
    ├── metrics.jsonl
    ├── eval.csv
    ├── training.log
    └── summary.json
```

- **`log_dir`**를 쓰면 `{preset_root}/{log_dir}/{run_name}/` 아래에 같은 `checkpoints/`, `logs/` 구조.
- **`output_root`가 null/빈 값**이고 `log_dir`도 없으면 **디스크에 저장하지 않고** 콘솔만 출력합니다.

## 디렉터리 구조

```
├── run.py
├── requirements.txt
├── pyproject.toml
├── config/
│   └── td3bc.yaml
├── algorithms/
│   ├── __init__.py    # --algo 레지스트리
│   └── td3bc.py
└── offrl/
    ├── __main__.py    # python -m offrl
    ├── cli/
    ├── core/
    ├── config/
    ├── data/
    ├── envs/
    ├── models/
    └── policy/
```

## 모듈 개요

| 위치 | 역할 |
|------|------|
| `offrl.core` | `AlgorithmBase`, `TransitionBatch`, `action_for_loss` |
| `offrl.data` | `ReplayBuffer` (NumPy) |
| `offrl.envs` | `set_seed`, `eval_actor`, 정규화·래핑 |
| `offrl.models` | MLP, 정책·Q·V |
| `offrl.policy` | `get_action`, `sample_actions`, `act_for_eval` |
| `offrl.config` | `load_yaml_config` |
| `offrl.cli` | 데이터 적재, 학습 루프, 평가, 로그 |
| `algorithms` | `load_algo`, `list_algorithms` + 구현 모듈 |

새 알고리즘: `algorithms/<이름>.py` 추가 후 `algorithms/__init__.py`의 `_ALGO_MODULES`에 등록. TD3+BC 참고: `algorithms/td3bc.py`, 규약: `offrl.core.AlgorithmBase`.

## D4RL → 버퍼 (예시)

```python
import d4rl, gym
from offrl import ReplayBuffer

env = gym.make("hopper-medium-v2")
ds = d4rl.qlearning_dataset(env)
buf = ReplayBuffer(
    state_dim=ds["observations"].shape[1],
    action_dim=ds["actions"].shape[1],
    buffer_size=len(ds["observations"]),
)
buf.load_d4rl_dataset(ds)
```

(레포 루트에서 `run.py`와 동일하게 `PYTHONPATH`가 잡힌 상태에서 실행.)

## 요구 사항

| 항목 | 버전 |
|------|------|
| Python | ≥ 3.9 |
| PyTorch | ≥ 1.13 |
| NumPy | ≥ 1.20 |

의존성 목록은 `pyproject.toml` / `requirements.txt` 참고.

## 라이선스

별도 `LICENSE` 파일은 없습니다.
