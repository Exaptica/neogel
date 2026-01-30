from __future__ import annotations

import numpy as np
from neogel.core.types import EvalRecord

try:
    import gymnasium as gym
except ImportError as e:
    raise ImportError("pip install 'gymnasium[box2d]'") from e


class BipedalWalkerLinearPolicyQD:
    def __init__(
        self,
        env_id: str = "BipedalWalker-v3",
        obs_dim: int = 24,
        act_dim: int = 4,
        episode_len: int = 1600,
        seed: int | None = None,
    ):
        self.env_id = env_id
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.episode_len = episode_len
        self.seed = seed

    @property
    def genome_dim(self) -> int:
        return self.act_dim * self.obs_dim + self.act_dim

    def _decode(self, g: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        g = np.asarray(g, dtype=np.float32)
        W = g[: self.act_dim * self.obs_dim].reshape(self.act_dim, self.obs_dim)
        b = g[self.act_dim * self.obs_dim :].reshape(self.act_dim)
        return W, b

    def evaluate(self, genotype: np.ndarray) -> EvalRecord:
        W, b = self._decode(genotype)

        env = gym.make(self.env_id)
        obs, _ = env.reset(seed=self.seed)

        total_reward = 0.0
        steps = 0
        act_mag = 0.0

        for _ in range(self.episode_len):
            act = np.tanh(W @ obs + b)
            act_mag += float(np.mean(np.abs(act)))
            obs, r, terminated, truncated, _ = env.step(act)
            total_reward += float(r)
            steps += 1
            if terminated or truncated:
                break

        env.close()

        extras = {
            "steps": steps,
            "mean_abs_action": act_mag / max(1, steps),
        }

        return EvalRecord(objectives=np.array([total_reward], dtype=float), extras=extras)