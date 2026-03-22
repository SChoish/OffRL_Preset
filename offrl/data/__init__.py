from typing import Dict, Optional, Tuple

import numpy as np


class ReplayBuffer:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        buffer_size: int,
        seed: Optional[int] = None,
    ):
        self._buffer_size = buffer_size
        self._pointer = 0
        self._size = 0
        self._states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self._actions = np.zeros((buffer_size, action_dim), dtype=np.float32)
        self._rewards = np.zeros((buffer_size, 1), dtype=np.float32)
        self._next_states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self._dones = np.zeros((buffer_size, 1), dtype=np.float32)
        self._rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()

    def load_d4rl_dataset(self, data: Dict[str, np.ndarray]) -> None:
        if self._size != 0:
            raise ValueError("Trying to load data into non-empty replay buffer")
        n = data["observations"].shape[0]
        if n > self._buffer_size:
            raise ValueError("Replay buffer is smaller than the dataset")
        self._states[:n] = np.asarray(data["observations"], dtype=np.float32)
        self._actions[:n] = np.asarray(data["actions"], dtype=np.float32)
        self._rewards[:n] = np.asarray(data["rewards"][..., None], dtype=np.float32)
        self._next_states[:n] = np.asarray(data["next_observations"], dtype=np.float32)
        self._dones[:n] = np.asarray(data["terminals"][..., None], dtype=np.float32)
        self._size += n
        self._pointer = min(self._size, n)
        print(f"Dataset size: {n}")

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        max_idx = min(self._size, self._pointer)
        idx = self._rng.integers(0, max_idx, size=batch_size)
        return (
            self._states[idx],
            self._actions[idx],
            self._rewards[idx],
            self._next_states[idx],
            self._dones[idx],
        )

    def add_transition(self) -> None:
        raise NotImplementedError
