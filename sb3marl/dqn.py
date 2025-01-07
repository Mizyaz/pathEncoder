import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from stable_baselines3.common.utils import explained_variance
from stable_baselines3.common.vec_env import VecEnv
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional, Any, Union, Type
import os

class MultiAgentEnvWrapper(gym.Wrapper):
    """
    Wrapper that converts dictionary obs/action spaces to flat spaces for compatibility with SB3.
    """
    def __init__(self, env):
        super().__init__(env)
        self.num_agents = len(env.action_space.spaces)
        
        # Store original spaces
        self.orig_obs_space = env.observation_space
        self.orig_action_space = env.action_space
        
        # Create flattened observation space
        obs_spaces = env.observation_space.spaces
        total_obs_dim = sum(np.prod(space.shape) for space in obs_spaces.values())
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(total_obs_dim,),
            dtype=np.float32
        )
        
        # Create flattened action space (assuming all agents have same action space)
        single_action_dim = env.action_space["agent_0"].n
        self.action_space = spaces.MultiDiscrete([single_action_dim] * self.num_agents)
        
        # Store agent IDs for reconstruction
        self.agent_ids = list(env.action_space.spaces.keys())

    def _flatten_obs(self, obs_dict):
        """Convert dictionary observation to flat array."""
        return np.concatenate([obs_dict[k].flatten() for k in self.agent_ids])

    def _unflatten_obs(self, obs):
        """Convert flat array to dictionary observation."""
        obs_dict = {}
        start_idx = 0
        for agent_id in self.agent_ids:
            space = self.orig_obs_space[agent_id]
            flat_dim = int(np.prod(space.shape))
            obs_dict[agent_id] = obs[start_idx:start_idx + flat_dim].reshape(space.shape)
            start_idx += flat_dim
        return obs_dict

    def _unflatten_action(self, action):
        """Convert flat action array to dictionary."""
        return {
            agent_id: int(action[i])  # Convert to int for discrete actions
            for i, agent_id in enumerate(self.agent_ids)
        }

    def step(self, action):
        """Convert flat action to dict, step env, convert obs to flat."""
        action_dict = self._unflatten_action(action)
        obs_dict, reward, done, truncated, info = self.env.step(action_dict)
        return self._flatten_obs(obs_dict), reward, done, truncated, info

    def reset(self, **kwargs):
        """Reset env and convert dict obs to flat."""
        obs_dict, info = self.env.reset(**kwargs)
        return self._flatten_obs(obs_dict), info

class MultiAgentFeaturesExtractor(BaseFeaturesExtractor):
    """
    Features extractor that handles flattened observations for multiple agents.
    """
    def __init__(self, observation_space: spaces.Box, features_dim: int = 64):
        super().__init__(observation_space, features_dim=features_dim)
        
        n_input = int(np.prod(observation_space.shape))
        
        self.net = nn.Sequential(
            nn.Linear(n_input, 128),
            nn.ReLU(),
            nn.Linear(128, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(observations)

class MultiAgentPolicy(ActorCriticPolicy):
    """
    Policy that handles flattened multi-agent observations and actions.
    """
    def __init__(
        self,
        observation_space: spaces.Box,
        action_space: spaces.MultiDiscrete,
        lr_schedule: Schedule,
        *args,
        **kwargs
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            *args,
            features_extractor_class=MultiAgentFeaturesExtractor,
            features_extractor_kwargs=dict(features_dim=128),
            **kwargs,
        )

class DQN:
    """
    Multi-agent DQN trainer using PPO.
    """
    def __init__(
        self,
        env,
        model_name: str = "ma_ppo",
        device: str = "auto",
        verbose: int = 1,
        tensorboard_log: Optional[str] = "logs",
    ):
        # Wrap the environment
        self.env = MultiAgentEnvWrapper(env)
        self.model_name = model_name
        self.device = device
        self.verbose = verbose
        self.tensorboard_log = tensorboard_log
        
        # Create PPO model with custom policy
        self.model = PPO(
            MultiAgentPolicy,
            self.env,
            verbose=verbose,
            tensorboard_log=tensorboard_log,
            device=device,
            # PPO specific parameters
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            learning_rate=3e-4,
            ent_coef=0.01,
            vf_coef=0.5,
            clip_range=0.2,
            gae_lambda=0.95,
            max_grad_norm=0.5,
            use_sde=False,
            sde_sample_freq=-1,
        )

    def learn(self, total_timesteps: int, progress_bar: bool = True):
        """Train the model."""
        self.model.learn(
            total_timesteps=total_timesteps,
            progress_bar=progress_bar,
        )

    def save(self, path: Optional[str] = None):
        """Save the model."""
        if path is None:
            path = os.path.join("models", self.model_name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)

    def load(self, path: str):
        """Load a trained model."""
        self.model = PPO.load(path, env=self.env)

    def predict(self, observation, deterministic: bool = True):
        """Get model prediction for an observation."""
        # Convert dict observation to flat array
        if isinstance(observation, dict):
            observation = self.env._flatten_obs(observation)
        action, state = self.model.predict(observation, deterministic=deterministic)
        # Convert flat action back to dict if needed
        if isinstance(self.env.env.action_space, spaces.Dict):
            action = self.env._unflatten_action(action)
        return action, state
    
from env import MultiAgentGridCoverage
if __name__ == "__main__":
    env = MultiAgentGridCoverage(num_agents=8, grid_size=8, plot_coverage=True, max_steps=20)
    model = DQN(env)
    model.learn(total_timesteps=100000)
