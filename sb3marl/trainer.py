import numpy as np
import torch
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.ppo import PPO
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional, Type, Union, Any
from env import MultiAgentGridCoverage

class MultiAgentFeaturesExtractor(BaseFeaturesExtractor):
    """
    Feature extractor that handles multiple agent observations independently.
    """
    def __init__(
        self,
        observation_space: spaces.Dict,
        num_agents: int,
        features_dim: int = 64,
        agent_feature_dim: int = 32
    ):
        super().__init__(observation_space, features_dim=features_dim*num_agents)
        
        self.num_agents = num_agents
        self.agent_feature_dim = agent_feature_dim
        
        # Create separate feature networks for each agent
        self.agent_networks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(np.prod(observation_space[f"agent_{i}"].shape), agent_feature_dim),
                nn.ReLU(),
                nn.Linear(agent_feature_dim, features_dim),
                nn.ReLU()
            ) for i in range(num_agents)
        ])

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Process each agent's observations
        features = []
        for i in range(self.num_agents):
            agent_obs = observations[f"agent_{i}"].float()
            agent_features = self.agent_networks[i](agent_obs.flatten(start_dim=1))
            features.append(agent_features)
            
        # Concatenate all agent features
        return torch.cat(features, dim=1)

class MultiAgentActorCriticPolicy(ActorCriticPolicy):
    """
    Policy network that handles multiple agents with separate action spaces.
    """
    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.MultiDiscrete,
        lr_schedule: callable,
        num_agents: int,
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        features_extractor_class: Type[BaseFeaturesExtractor] = MultiAgentFeaturesExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        sde_net_arch: Optional[List[int]] = None,
        share_features_extractor: bool = True,
        squash_output: bool = False,
    ):
        self.num_agents = num_agents
        
        if features_extractor_kwargs is None:
            features_extractor_kwargs = {"num_agents": num_agents}
        else:
            features_extractor_kwargs["num_agents"] = num_agents

        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
        )

class MultiAgentPPO(PPO):
    """
    Multi-agent version of PPO algorithm.
    """
    def __init__(
        self,
        policy: Union[str, Type[MultiAgentActorCriticPolicy]],
        env,
        num_agents: int,
        learning_rate: Union[float, callable] = 3e-4,
        n_steps: int = 2048,
        batch_size: Optional[int] = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, callable] = 0.2,
        clip_range_vf: Union[None, float, callable] = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        target_kl: Optional[float] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[torch.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        if policy_kwargs is None:
            policy_kwargs = {}
        policy_kwargs["num_agents"] = num_agents
        
        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            clip_range_vf=clip_range_vf,
            normalize_advantage=normalize_advantage,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            target_kl=target_kl,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model,
        )
        
        self.num_agents = num_agents

import optuna
import time

def optuna_train(trial, params=None):
    # Create environment

    start_time = time.time()
    env = MultiAgentGridCoverage(
        grid_size=8,
        num_agents=8,
        observation_radius=2,
        max_steps=20,
        render_mode=None,  # Disable rendering for training
        plot_coverage=False if params is None else True
    )
    
    if params is None:
        network_1 = trial.suggest_int("network_1", 32, 256, step=32)
        network_2 = trial.suggest_int("network_2", 32, 256, step=32)
        agent_feature_dim = trial.suggest_int("agent_feature_dim", 32, 256, step=32)
        features_dim = trial.suggest_int("features_dim", 32, 256, step=32)

        n_steps = trial.suggest_int("n_steps", 64, 2048, step=64)
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        n_epochs = trial.suggest_int("n_epochs", 1, 10, step=1)
        gamma = trial.suggest_float("gamma", 0.8, 0.99, step=0.05)
        gae_lambda = trial.suggest_float("gae_lambda", 0.8, 0.99, step=0.05)
    else:
        network_1 = params["network_1"]
        network_2 = params["network_2"]
        agent_feature_dim = params["agent_feature_dim"]
        features_dim = params["features_dim"]

        n_steps = params["n_steps"]
        learning_rate = params["learning_rate"]
        n_epochs = params["n_epochs"]
        gamma = params["gamma"]
        gae_lambda = params["gae_lambda"]
    
    # Initialize Multi-Agent PPO
    model = MultiAgentPPO(
        policy=MultiAgentActorCriticPolicy,
        env=env,
        num_agents=8,
        verbose=0,
        n_steps=n_steps,
        batch_size=64,
        learning_rate=learning_rate,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        policy_kwargs={"net_arch": [network_1, network_2],
                    "features_extractor_kwargs": {"agent_feature_dim": agent_feature_dim, 
                                                    "features_dim": features_dim}},
    )
    
    # Train the model
    model.learn(total_timesteps=10000 if params is None else 200000)

    """print(f"Time taken for network size {network_1} and {network_2} and agent feature dim {agent_feature_dim} and features dim {features_dim}: {time.time() - start_time:.2f} seconds")
    print(f"max coverage rate: {max(env.coverage_rates):.2f}%")
    print(f"mean coverage rate: {np.mean(env.coverage_rates):.2f}%")
    print(f"std coverage rate: {np.std(env.coverage_rates):.2f}%")"""
    """
    # Test the trained model
    env = MultiAgentGridCoverage(  # Create a new env for testing with rendering
        grid_size=8,
        num_agents=8,
        observation_radius=2,
        max_steps=20,
        render_mode="rgb_array"
    )
    
    obs, _ = env.reset()
    for _ in range(1000):
        actions, _ = model.predict(obs, deterministic=True)
        obs, rewards, terminated, truncated, info = env.step(actions)
        
        if all(terminated.values()) or all(truncated.values()):
            obs, _ = env.reset()
            
    env.close()"""

    return np.mean(env.coverage_rates)

if __name__ == "__main__":
    """study = optuna.create_study(direction="maximize")
    study.optimize(optuna_train, n_trials=200)
    params = study.best_trial.params"""
    params = {'network_1': 128, 'network_2': 64, 'agent_feature_dim': 128, 'features_dim': 128, 'n_steps': 128, 'learning_rate': 0.0002417489436267694, 'n_epochs': 1, 'gamma': 0.8, 'gae_lambda': 0.8}
    print(params)
    optuna_train(None, params)