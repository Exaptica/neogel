from __future__ import annotations

import numpy as np
from neogel.core.types import EvalRecord

try:
    import gymnasium as gym
except ImportError as e:
    raise ImportError("Install gymnasium to use BipedalWalker: pip install gymnasium") from e


class BipedalWalkerLinearPolicy:
    """Evolve a linear policy for BipedalWalker.

    Genotype: flat vector of length (act_dim * obs_dim + act_dim)
      first act_dim*obs_dim entries: W (act_dim x obs_dim)
      last act_dim entries: b (act_dim,)
    """

    def __init__(
        self,
        env_id: str = "BipedalWalker-v3",
        obs_dim: int = 24,
        act_dim: int = 4,
        episode_len: int = 1600,
        n_episodes: int = 1,
        render: bool = False,
        seed: int | None = None,
    ):
        self.env_id = env_id
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.episode_len = episode_len
        self.n_episodes = n_episodes
        self.render = render
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

        total_return = 0.0
        for ep in range(self.n_episodes):
            env = gym.make(self.env_id, render_mode="human" if self.render else None)
            obs, _ = env.reset(seed=self.seed)
            ep_ret = 0.0

            for _ in range(self.episode_len):
                # linear policy
                a = np.tanh(W @ obs + b)
                obs, r, terminated, truncated, _ = env.step(a)
                ep_ret += float(r)
                if terminated or truncated:
                    break

            env.close()
            total_return += ep_ret

        avg_return = total_return / float(self.n_episodes)
        return EvalRecord(objectives=np.array([avg_return], dtype=float), extras=None)