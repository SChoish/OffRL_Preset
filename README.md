# offrl-preset

D4RL 형식 오프라인 데이터와 **PyTorch** 정책·Q/V·버퍼·학습 유틸을 묶어 **이 레포 위에서** 알고리즘을 붙이고 실험하기 위한 워크스페이스입니다. PyPI 패키지로 설치해 쓰는 것을 전제로 두지 않습니다.

- **규약**: `offrl.core` — `AlgorithmBase`, `TransitionBatch` 등  
- **구현**: 레포 루트 **`algorithms/`** — `td3bc` 등 (프리셋 `offrl`과 분리)

---

## 빠른 시작 (레포 루트 기준)

```bash
cd Preset   # 이 레포 루트
pip install -r requirements.txt

python run.py --algo td3bc
python run.py --algo td3bc --d4rl halfcheetah-medium-v2
```

**Linux:** `python run.py` / `python -m offrl` 은 기본으로 **`nohup`으로 한 번 더 실행**되어, SSH 세션을 끊어도(SIGHUP) 학습이 계속됩니다. 터미널에 stdout이 붙어 있으면 출력은 보통 현재 디렉터리의 **`nohup.out`** 으로 이어집니다.

포그라운드로만 돌리거나(IDE 디버그 등) 터미널에 로그를 그대로 두려면:

```bash
OFFRL_NO_NOHUP=1 python run.py --algo td3bc
```

기본 `config/td3bc.yaml`의 `d4rl`은 `hopper-medium-v2`입니다. 다른 태스크는 `--d4rl` 또는 YAML로 바꿉니다.

### 실행 결과 저장 (MPI 스타일 정리)

기본으로 **`results/`** 아래 한 run당 한 폴더만 씁니다 (`.gitignore`에 포함).

```
results/<algo>/<env>/<task>/seed_<seed>/<run_name>/
├── checkpoints/          # step_{N}.pth (checkpoint_every마다), final.pth
└── logs/
    ├── metrics.jsonl       # 학습 loss + eval 지표 한 줄씩 JSON
    ├── eval.csv            # step, return_mean, return_std, d4rl_norm
    ├── training.log        # 평가·체크포인트 요약 타임스탬프 로그
    └── summary.json        # 메타 + eval_history 전체
```

- **`log_dir`** 를 쓰면 (예전 방식) `{preset_root}/{log_dir}/{run_name}/` 아래에 같은 `checkpoints/`, `logs/` 구조를 둡니다.
- **`output_root: null`** (YAML)이고 `log_dir` 도 없으면 디스크에 저장하지 않고 콘솔만 출력합니다.

`run.py`가 레포 루트를 `sys.path`에 넣으므로 **`pip install -e .` 없이** `import offrl`이 됩니다.

Cursor/VS Code로 이 폴더를 연 경우 `.vscode/settings.json`에서 터미널에 `PYTHONPATH`를 잡아 두었으므로, 같은 루트에서 `python -m offrl` / `python -c "from offrl import …"`도 동작합니다.

### 선택: editable 설치

다른 디렉터리에서도 `import offrl`을 쓰고 싶을 때만:

```bash
pip install -e .
```

콘솔 스크립트 `offrl-run`은 넣지 않았습니다. 항상 **`python run.py`** 또는 **`python -m offrl`**을 쓰면 됩니다.

### 프리셋 루트 찾기

CLI는 `config/`, `pyproject.toml`을 기준으로 레포 루트를 잡습니다. 서브폴더에서 실행해도 상위로 올라가며 찾습니다. 환경 변수 **`OFFRL_ROOT`**로 고정할 수도 있습니다.

---

## 디렉터리 구조

```
Preset/
├── run.py                 # 학습 진입점 (설치 없이 실행)
├── results/               # 기본 실험 산출물 (gitignore)
├── requirements.txt
├── pyproject.toml         # 메타/선택적 editable용
├── .vscode/settings.json  # PYTHONPATH (선택)
├── config/
│   └── td3bc.yaml
├── algorithms/            # 알고리즘 구현 (--algo 레지스트리)
│   ├── __init__.py
│   └── td3bc.py
└── offrl/
    ├── __init__.py
    ├── __main__.py        # python -m offrl
    ├── py.typed
    ├── cli/
    ├── config/
    ├── core/
    ├── data/
    ├── envs/
    ├── models/
    └── policy/
```

---

## 설정 YAML

- 기본: **`config/<algo>.yaml`**. 없으면 해당 모듈의 `TRAIN_DEFAULTS`만 사용.
- CLI 인자가 YAML보다 우선.

---

## 요구 사항

| 항목 | 버전 |
|------|------|
| Python | ≥ 3.9 |
| PyTorch | ≥ 1.13 |
| NumPy | ≥ 1.20 |

---

## 모듈

| 모듈 | 내용 |
|------|------|
| `offrl.core` | `AlgorithmBase`, `TransitionBatch`, `TensorBatch`, `ActorConfig`, `action_for_loss` |
| `offrl.data` | `ReplayBuffer` (CPU/NumPy) |
| `offrl.envs` | `set_seed`, `eval_actor`, `wrap_env`, `compute_mean_std`, `normalize_states` |
| `offrl.models` | MLP 유틸, 정책(`actors`), Q/V(`critics`) |
| `offrl.policy` | `get_action`, `sample_actions`, `act_for_eval` … |
| `offrl.config` | `load_yaml_config` (`config/*.yaml`) |
| `algorithms` | `load_algo`, `list_algorithms`; 구현 모듈(`td3bc` …) |
| `offrl.cli` | `main` — 데이터 적재·학습 루프·평가·`metrics.jsonl` |

새 알고리즘: **`algorithms/<이름>.py`** 추가 후 **`algorithms/__init__.py`** 의 `_ALGO_MODULES`에 등록.

---

## 데이터 → 버퍼

```python
import d4rl
import gym
from offrl import ReplayBuffer

env = gym.make("hopper-medium-v2")
ds = d4rl.qlearning_dataset(env)
buf = ReplayBuffer(
    state_dim=ds["observations"].shape[1],
    action_dim=ds["actions"].shape[1],
    buffer_size=len(ds["observations"]),
)
buf.load_d4rl_dataset(ds)
s, a, r, s2, d = buf.sample(256)
```

(위 스크립트는 레포 루트에서 `PYTHONPATH=.` 또는 `run.py`와 동일한 방식으로 경로가 잡힌 상태에서 실행.)

---

## AlgorithmBase · TD3+BC

`update_critic` / `compute_actor_loss` / `update_target_networks` 규약은 **`offrl.core.AlgorithmBase`** 에 정의됩니다. TD3+BC 구현은 **`algorithms.td3bc`** 를 참고하면 됩니다.

---

## 라이선스

별도 `LICENSE`가 없으면 사용 전 정책을 확인하세요.
