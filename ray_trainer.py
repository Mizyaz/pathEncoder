from pprint import pprint
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray import tune
import ray
import gymnasium as gym
from gymnasium import spaces

#!/usr/bin/env python3

import warnings
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import sys
import time
from typing import Dict, List, Tuple, Optional, Any, Union


from ray_env import MultiAgentGridCoverage, environment_trace

from ray.rllib.algorithms.callbacks import DefaultCallbacks

import plotext as plt

class CoverageCallback(DefaultCallbacks):
    def on_episode_start(self, *, episode, env_runner, **kwargs):
        # Called at the beginning of each episode.
        pass
    def on_episode_end(self, *, episode, env_runner, **kwargs):
        # Called at the end of each episode.
        pass

    def on_train_result(self, *, algorithm, result, **kwargs):
        # Called after each training iteration.

        coverages_ = ray.get_actor("global_coverages")
        timesteps_ = ray.get_actor("global_timesteps")
        
        print(f"mean coverage: {ray.get(coverages_.mean.remote()):.2f}, max coverage: {ray.get(coverages_.max.remote()):.2f}, min coverage: {ray.get(coverages_.min.remote()):.2f}, std coverage: {ray.get(coverages_.std.remote()):.2f}")
        print(f"mean timesteps: {ray.get(timesteps_.mean.remote()):.2f}, max timesteps: {ray.get(timesteps_.max.remote()):.2f}, min timesteps: {ray.get(timesteps_.min.remote()):.2f}, std timesteps: {ray.get(timesteps_.std.remote()):.2f}")
        plt.clear_data()
        plt.plot(ray.get(coverages_.get.remote()), label="coverage")
        plt.show()
        plt.clear_data()
        plt.plot(ray.get(timesteps_.get.remote()), label="timesteps")
        plt.show()

@ray.remote
class Counter:
    def __init__(self):
        self.count = []

    def inc(self, n):
        self.count.append(n)

    def get(self):
        return self.count

    def mean(self):
        return np.mean(self.count)
    
    def max(self):
        return np.max(self.count)
    
    def min(self):
        return np.min(self.count)
    
    def std(self):
        return np.std(self.count)

import pathlib
from ray.rllib.algorithms.ppo import PPOConfig

def train():
    config = {
        "env": MultiAgentGridCoverage,
        "env_config": {
            "num_agents": 8,
            "grid_size": 8,
            "plot_coverage": True,
            "max_steps": 20,
            "observation_radius": 2
        },
    }
    env = MultiAgentGridCoverage(**config["env_config"])



    config = (PPOConfig()
        .environment(MultiAgentGridCoverage)
        .framework("torch")
        .callbacks(CoverageCallback)
        .multi_agent(
            policies={"default": (None, env.observation_space["agent_0"], env.action_space["agent_0"], {})},
            policy_mapping_fn=lambda agent_id, *args, **kwargs: "default",
        )
        .env_runners(num_env_runners=0)
        .training(
            train_batch_size=512,
            num_sgd_iter=10,
            lr=3e-4,
            gamma=0.99,
            num_epochs=10,
            model={"fcnet_hiddens": [256, 256], "fcnet_activation": "relu"},
        )

    )

    algo = config.build()
    import time
    start_time = time.time()

    for i in range(5):
        result = algo.train()
        result.pop("config")
        #pprint(result)

        if i % 5 == 0:
            checkpoint_dir = algo.save_checkpoint(pathlib.Path("./checkpoinRay").resolve())
            print(f"Checkpoint saved in directory {checkpoint_dir}")

    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.2f} seconds")

from ray.rllib.core.rl_module import RLModule
import torch

def evaluate(checkpoint_dir):
    config = {
        "env": MultiAgentGridCoverage,
        "env_config": {
            "num_agents": 8,
            "grid_size": 8,
            "plot_coverage": True,
            "max_steps": 20,
            "observation_radius": 2
        },
    }

    env = MultiAgentGridCoverage(**config["env_config"])
    checkpoint_path = pathlib.Path(checkpoint_dir).resolve() / "learner_group" / "learner" / "rl_module"
    rl_modules = RLModule.from_checkpoint(checkpoint_path)

    episode_return = {agent_id: 0 for agent_id in env.agents}
    terminateds = truncateds = {"__all__": False}

    obs, info = env.reset()

    while not terminateds["__all__"] and not truncateds["__all__"]:
        actions = {}
        for agent_id, agent_obs in obs.items():
            policy_id = "default"
            rl_module = rl_modules[policy_id]
            torch_obs = torch.from_numpy(np.array([agent_obs]))
            action_logits = rl_module.forward_inference({"obs": torch_obs})["action_dist_inputs"]
            actions[agent_id] = torch.argmax(action_logits[0]).numpy()
        print(f"actions made: {actions}")

        obs, rewards, terminateds, truncateds, info = env.step(actions)
        for agent_id, reward in rewards.items():
            episode_return[agent_id] += reward
        print(f"rewards: {rewards}")

    print(f"Reached episode return of {episode_return}.")



if __name__ == "__main__":
    # on the driver
    coverages = Counter.options(name="global_coverages").remote()
    timesteps = Counter.options(name="global_timesteps").remote()
    #train()
    evaluate("./checkpoinRay")