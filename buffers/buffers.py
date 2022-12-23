import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, Generator, List, Optional, NamedTuple, Union

import numpy as np
import torch


try:
    import psutil
except ImportError:
    psutil = None
    
class ReplayBufferSamples(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    next_observations: torch.Tensor
    dones: torch.Tensor
    rewards: torch.Tensor

class RolloutBufferSamples(NamedTuple):
    observations: torch.Tensor
    surroundings: torch.Tensor
    actions: torch.Tensor
    old_i_values: torch.Tensor
    old_n_values: torch.Tensor
    old_g_values: torch.Tensor
    old_log_prob: torch.Tensor
    i_advantages: torch.Tensor
    n_advantages: torch.Tensor
    g_advantages: torch.Tensor
    i_returns: torch.Tensor
    n_returns: torch.Tensor
    g_returns: torch.Tensor
    lcf: torch.Tensor
    


class BaseBuffer(ABC):
    """
    Base class that represent a buffer (rollout or replay)
    :param device: PyTorch device
        to which the values will be converted
    :param n_envs: Number of parallel environments
    """

    def __init__(
        self,
        buffer_size: int,
        state_dim: int,
        action_dim: int,
        device: Union[torch.device, str] = "cpu",
        n_envs: int = 1,
    ):
        super().__init__()
        self.buffer_size = buffer_size

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.pos = 0
        self.full = False
        self.device = device
        self.n_envs = n_envs

    @staticmethod
    def swap_and_flatten(arr: np.ndarray) -> np.ndarray:
        """
        Swap and then flatten axes 0 (buffer_size) and 1 (n_envs)
        to convert shape from [n_steps, n_envs, ...] (when ... is the shape of the features)
        to [n_steps * n_envs, ...] (which maintain the order)

        :param arr:
        :return:
        """
        shape = arr.shape
        if len(shape) < 3:
            shape = shape + (1,)
        return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])

    def size(self) -> int:
        """
        :return: The current size of the buffer
        """
        if self.full:
            return self.buffer_size
        return self.pos

    def add(self, *args, **kwargs) -> None:
        """
        Add elements to the buffer.
        """
        raise NotImplementedError()

    def extend(self, *args, **kwargs) -> None:
        """
        Add a new batch of transitions to the buffer
        """
        # Do a for loop along the batch axis
        for data in zip(*args):
            self.add(*data)

    def reset(self) -> None:
        """
        Reset the buffer.
        """
        self.pos = 0
        self.full = False

    def sample(self, batch_size: int):
        """
        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        upper_bound = self.buffer_size if self.full else self.pos
        batch_inds = np.random.randint(0, upper_bound, size=batch_size)
        return self._get_samples(batch_inds)

    @abstractmethod
    def _get_samples(
        self, batch_inds: np.ndarray
    ) -> Union[ReplayBufferSamples, RolloutBufferSamples]:
        """
        :param batch_inds:
        :param env:
        :return:
        """
        raise NotImplementedError()

    def to_torch(self, array: np.ndarray, copy: bool = True) -> torch.Tensor:
        """
        Convert a numpy array to a PyTorch tensor.
        Note: it copies the data by default

        :param array:
        :param copy: Whether to copy or not the data
            (may be useful to avoid changing things be reference)
        :return:
        """
        if copy:
            return torch.tensor(array).to(self.device)
        return torch.as_tensor(array).to(self.device)


class RolloutBuffer(BaseBuffer):
    """
    Rollout buffer used in on-policy algorithms like A2C/PPO.
    It corresponds to ``buffer_size`` transitions collected
    using the current policy.
    This experience will be discarded after the policy update.
    In order to use PPO objective, we also store the current value of each state
    and the log probability of each taken action.

    The term rollout here refers to the model-free notion and should not
    be used with the concept of rollout used in model-based RL or planning.
    Hence, it is only involved in policy and value function training but not action selection.

    :param buffer_size: Max number of element in the buffer
    :param device:
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to classic advantage when set to 1.
    :param gamma: Discount factor
    :param n_envs: Number of parallel environments
    """

    def __init__(
        self,
        buffer_size: int,
        state_dim: int,
        img_size: int,
        action_dim: int,
        device: Union[torch.device, str] = "cpu",
        gae_lambda: float = 0.95,
        gamma: float = 0.999,
        n_envs: int = 1,
    ):

        super().__init__(buffer_size, state_dim, action_dim, device, n_envs=n_envs)
        self.img_size = img_size
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.observations, self.actions, self.rewards, self.advantages = None, None, None, None
        self.surroundings, self.n_rewards, self.n_advantages, self.g_rewards, self.g_advantages = None, None, None, None, None
        self.returns, self.episode_starts, self.values, self.log_probs = None, None, None, None
        self.n_returns, self.g_returns, self.n_values, self.g_values = None, None, None, None
        self.lcf = None
        self.generator_ready = False
        self.reset()

    def reset(self) -> None:

        self.observations = np.zeros((self.buffer_size, self.n_envs, self.state_dim), dtype=np.float32)
        self.surroundings = np.zeros((self.buffer_size, self.n_envs) + (3, self.img_size, self.img_size), dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.episode_starts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.n_rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.n_returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.n_values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.n_advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.g_rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.g_returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.g_values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.g_advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.lcf = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.generator_ready = False
        super().reset()

    def compute_returns_and_advantage(self, last_values: torch.Tensor, last_n_values: torch.Tensor, last_g_values: torch.Tensor, dones: np.ndarray) -> None:
        """
        Post-processing step: compute the lambda-return (TD(lambda) estimate)
        and GAE(lambda) advantage.

        Uses Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)
        to compute the advantage. To obtain Monte-Carlo advantage estimate (A(s) = R - V(S))
        where R is the sum of discounted reward with value bootstrap
        (because we don't always have full episode), set ``gae_lambda=1.0`` during initialization.

        The TD(lambda) estimator has also two special cases:
        - TD(1) is Monte-Carlo estimate (sum of discounted rewards)
        - TD(0) is one-step estimate with bootstrapping (r_t + gamma * v(s_{t+1}))

        For more information, see discussion in https://github.com/DLR-RM/stable-baselines3/pull/375.

        :param last_values: state value estimation for the last step (one for each env)
        :param dones: if the last step was a terminal step (one bool for each env).
        """
        # Convert to numpy
        last_values = last_values.clone().cpu().numpy().flatten()
        last_n_values = last_n_values.clone().cpu().numpy().flatten()
        last_g_values = last_g_values.clone().cpu().numpy().flatten()

        last_gae_lam = 0
        last_n_gae_lam = 0
        last_g_gae_lam = 0
        
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones
                next_values = last_values
                next_n_values = last_n_values
                next_g_values = last_g_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]
                next_n_values = self.n_values[step + 1]
                next_g_values = self.g_values[step + 1]
                
            delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam
            
            n_delta = self.n_rewards[step] + self.gamma * next_n_values * next_non_terminal - self.n_values[step]
            last_n_gae_lam = n_delta + self.gamma * self.gae_lambda * next_non_terminal * last_n_gae_lam
            self.n_advantages[step] = last_n_gae_lam
            
            g_delta = self.g_rewards[step] + self.gamma * next_g_values * next_non_terminal - self.g_values[step]
            last_g_gae_lam = g_delta + self.gamma * self.gae_lambda * next_non_terminal * last_g_gae_lam
            self.g_advantages[step] = last_g_gae_lam
        # TD(lambda) estimator, see Github PR #375 or "Telescoping in TD(lambda)"
        # in David Silver Lecture 4: https://www.youtube.com/watch?v=PnHCvfgC_ZA
        self.returns = self.advantages + self.values
        self.n_returns = self.n_advantages + self.n_values
        self.g_returns = self.g_advantages + self.g_values

    def add(
        self,
        obs: np.ndarray,
        imgs:np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        n_reward: np.ndarray,
        g_reward: np.ndarray,
        episode_start: np.ndarray,
        value: torch.Tensor,
        n_value: torch.Tensor,
        g_value: torch.Tensor,
        log_prob: torch.Tensor,
        lcf: np.ndarray
    ) -> None:
        """
        :param obs: Observation
        :param action: Action
        :param reward:
        :param episode_start: Start of episode signal.
        :param value: estimated value of the current state
            following the current policy.
        :param log_prob: log probability of the action
            following the current policy.
        """
        if len(log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = log_prob.reshape(-1, 1)

        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)

        self.observations[self.pos] = np.array(obs).copy()
        self.surroundings[self.pos] = np.array(imgs).copy()
        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.n_rewards[self.pos] = np.array(n_reward).copy()
        self.g_rewards[self.pos] = np.array(g_reward).copy()
        self.episode_starts[self.pos] = np.array(episode_start).copy()
        self.values[self.pos] = value.clone().cpu().numpy().flatten()
        self.n_values[self.pos] = n_value.clone().cpu().numpy().flatten()
        self.g_values[self.pos] = g_value.clone().cpu().numpy().flatten()
        self.log_probs[self.pos] = log_prob.clone().cpu().numpy()
        self.lcf[self.pos] = np.array(lcf).copy()
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def get(self, batch_size: Optional[int] = None) -> Generator[RolloutBufferSamples, None, None]:
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        # Prepare the data
        if not self.generator_ready:

            _tensor_names = [
                "observations",
                "surroundings",
                "actions",
                "values",
                "n_values",
                "g_values",
                "log_probs",
                "advantages",
                "n_advantages",
                "g_advantages",
                "returns",
                "n_returns",
                "g_returns",
                "lcf"
            ]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(self, batch_inds: np.ndarray) -> RolloutBufferSamples:
        data = (
            self.observations[batch_inds],
            self.surroundings[batch_inds],
            self.actions[batch_inds],
            self.values[batch_inds].flatten(),
            self.n_values[batch_inds].flatten(),
            self.g_values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.n_advantages[batch_inds].flatten(),
            self.g_advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
            self.n_returns[batch_inds].flatten(),
            self.g_returns[batch_inds].flatten(),
            self.lcf[batch_inds].flatten()
        )
        return RolloutBufferSamples(*tuple(map(self.to_torch, data)))