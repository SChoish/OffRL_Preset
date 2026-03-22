from typing import Optional, Tuple, Union

import gym
import numpy as np
import torch
import torch.nn as nn


def set_seed(
    seed: int, env: Optional[gym.Env] = None, deterministic_torch: bool = False
) -> None:
    import os
    import random

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(deterministic_torch)
    if env is not None:
        try:
            env.seed(seed)
        except AttributeError:
            pass
        try:
            env.action_space.seed(seed)
        except AttributeError:
            pass
        try:
            env.observation_space.seed(seed)
        except AttributeError:
            pass


@torch.no_grad()
def eval_actor(
    env: gym.Env, actor: nn.Module, device: str, n_episodes: int, seed: int
) -> np.ndarray:
    actor.eval()
    try:
        env.seed(seed)
        env.action_space.seed(seed)
        if hasattr(env, "observation_space"):
            env.observation_space.seed(seed)
    except Exception:
        pass
    rewards = []
    for ep in range(n_episodes):
        es = seed + ep
        try:
            env.seed(es)
            env.action_space.seed(es)
        except Exception:
            pass
        state, done = env.reset(), False
        if isinstance(state, tuple):
            state = state[0]
        R = 0.0
        while not done:
            action = actor.act(state, device)
            out = env.step(action)
            if len(out) == 5:
                state, r, term, trunc, _ = out
                done = term or trunc
            else:
                state, r, done, _ = out
            R += r
        rewards.append(R)
    actor.train()
    return np.asarray(rewards)


def compute_mean_std(states: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    return states.mean(0), states.std(0) + eps


def normalize_states(states: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (states - mean) / std


def wrap_env(
    env: gym.Env,
    state_mean: Union[np.ndarray, float] = 0.0,
    state_std: Union[np.ndarray, float] = 1.0,
    reward_scale: float = 1.0,
) -> gym.Env:
    env = gym.wrappers.TransformObservation(env, lambda s: (s - state_mean) / state_std)
    if reward_scale != 1.0:
        env = gym.wrappers.TransformReward(env, lambda r: reward_scale * r)
    return env
