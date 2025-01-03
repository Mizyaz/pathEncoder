#################################
# multiagent_env.py
#################################

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional
import plotext as plt

class MultiAgentGridCoverage(gym.Env):
    """
    Multi-agent grid coverage environment with optional telemetry-based communication.
    Refined so that observation shapes remain stable and consistent.
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": 4}

    def __init__(
        self,
        grid_size: int = 20,
        num_agents: int = 3,
        observation_radius: int = 2,
        max_steps: int = 500,
        render_mode: Optional[str] = None,
        plot_coverage: bool = False,
        Rcomm: float = 5.0,
        telemetry_config: Optional[Dict] = None
    ):
        super().__init__()
        self.grid_size = grid_size
        self.num_agents = num_agents
        self.obs_radius = observation_radius
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.plot_coverage = plot_coverage
        self.Rcomm = Rcomm
        self.telemetry_config = telemetry_config if telemetry_config else {'position': True}

        self.total_cells = grid_size * grid_size

        # We'll define the observation space after the first reset,
        # because we rely on the actual shape at runtime.
        # For now, just a placeholder single-dimension.
        self.observation_space = spaces.Dict({
            f"agent_{i}": spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
            for i in range(num_agents)
        })

        # Action space: each agent can move in 4 directions.
        self.action_space = spaces.MultiDiscrete([4] * num_agents)

        # Communication helper stubs
        self.telemetry_factory = self.TelemetryFactory(self.telemetry_config)
        self.knowledge_bases = [
            self.Knowledge(i, num_agents) for i in range(num_agents)
        ]

        # For tracking coverage
        self.coverage_rates = []
        self.finish_steps = []

        # Make initial reset to define actual observation_space
        self.reset()
        self._define_observation_space()

    class Telemetry:
        """Simple telemetry object storing data from sender to receivers."""
        def __init__(self, timestamp: int, sender_id: int, data: Dict):
            self.timestamp = timestamp
            self.sender_id = sender_id
            self.data = data
            self.receivers: List[int] = []

    class TelemetryFactory:
        """Factory for creating and managing telemetry objects."""
        def __init__(self, config: Dict):
            self.config = config
            self.telemetries: List[MultiAgentGridCoverage.Telemetry] = []

        def reset(self):
            self.telemetries = []

        def create_telemetry(self, timestamp: int, sender_id: int, data: Dict) -> 'MultiAgentGridCoverage.Telemetry':
            t = MultiAgentGridCoverage.Telemetry(timestamp, sender_id, data)
            self.telemetries.append(t)
            return t

    class Knowledge:
        """Stores aggregated knowledge for each agent."""
        def __init__(self, agent_id: int, num_agents: int):
            self.agent_id = agent_id
            self.num_agents = num_agents
            self.knowledge: Dict[int, Dict] = {i: {} for i in range(num_agents)}

        def add_knowledge(self, telemetry: 'MultiAgentGridCoverage.Telemetry'):
            sender = telemetry.sender_id
            if sender != self.agent_id:
                for key, value in telemetry.data.items():
                    self.knowledge[sender][key] = value

    def _define_observation_space(self):
        # We just did a reset and got an actual observation.
        # Let's define the shape properly now.
        test_obs, _ = self.reset()
        space_dict = {}
        for agent_id, obs in test_obs.items():
            shape_ = obs.shape
            # Observations are clamped to [-1, 1] to help neural nets
            space_dict[agent_id] = spaces.Box(
                low=-1.0, high=1.0, shape=shape_, dtype=np.float32
            )
        self.observation_space = spaces.Dict(space_dict)

        # revert to actual initial obs
        return test_obs, {}

    def reset(self, seed=None, options=None) -> Tuple[Dict[str, np.ndarray], Dict]:
        super().reset(seed=seed)
        self.steps = 0

        # Reset grid coverage
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        # Start all agents at (0,0)
        self.agent_positions = np.zeros((self.num_agents, 2), dtype=int)
        # Mark initial coverage
        self.grid[0, 0] = 1

        # Reset telemetry
        self.telemetry_factory.reset()
        for kb in self.knowledge_bases:
            kb.knowledge = {i: {} for i in range(self.num_agents)}

        coverage = np.mean(self.grid)

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def _get_local_view(self, agent_idx: int) -> np.ndarray:
        r = self.obs_radius
        pos = self.agent_positions[agent_idx]
        padded = np.pad(self.grid, r, mode='constant', constant_values=0)
        center_x, center_y = pos + r
        view = padded[
            center_x - r:center_x + r + 1,
            center_y - r:center_y + r + 1
        ]
        return view

    def _get_obs(self) -> Dict[str, np.ndarray]:
        coverage = np.mean(self.grid)
        obs = {}
        for i in range(self.num_agents):
            local_view = self._get_local_view(i).flatten()
            relative_pos = self.agent_positions[i] / self.grid_size

            # Flatten everything into one array
            # [local_view, rel_x, rel_y, coverage]
            combined = np.concatenate([
                local_view,
                relative_pos,
                [coverage]
            ], axis=0).astype(np.float32)

            # Scale to [-1,1] (roughly)
            combined = 2.0 * (combined - 0.5)

            obs[f"agent_{i}"] = combined
        return obs

    def _get_info(self) -> Dict:
        coverage = np.mean(self.grid)
        return {
            "coverage": coverage,
            "steps": self.steps,
            "coverage_rate": coverage * 100
        }

    def step(self, actions: np.ndarray):
        old_coverage = np.mean(self.grid)
        agent_rewards = {}

        # Process actions
        for i in range(self.num_agents):
            action = actions[i]
            pos = self.agent_positions[i]
            new_pos = pos.copy()

            if action == 0:  # Up
                new_pos[0] = max(0, pos[0] - 1)
            elif action == 1:  # Right
                new_pos[1] = min(self.grid_size - 1, pos[1] + 1)
            elif action == 2:  # Down
                new_pos[0] = min(self.grid_size - 1, pos[0] + 1)
            elif action == 3:  # Left
                new_pos[1] = max(0, pos[1] - 1)

            # Check collision
            collision = False
            for j, other_pos in enumerate(self.agent_positions):
                if i != j and np.array_equal(new_pos, other_pos):
                    collision = True
                    break

            if not collision:
                self.agent_positions[i] = new_pos
                was_covered = self.grid[new_pos[0], new_pos[1]] == 1
                self.grid[new_pos[0], new_pos[1]] = 1
                agent_rewards[f"agent_{i}"] = 1.0 if not was_covered else -0.5
            else:
                agent_rewards[f"agent_{i}"] = -0.5

        self.steps += 1
        new_coverage = np.mean(self.grid)
        coverage_improv = new_coverage - old_coverage
        coverage_bonus = 100.0 if np.all(self.grid == 1) else 0.0

        for i in range(self.num_agents):
            agent_rewards[f"agent_{i}"] += coverage_improv * 10.0 + coverage_bonus

        # Communication
        self._handle_communication()

        # Mean reward for SB3
        mean_reward = np.mean(list(agent_rewards.values()))

        terminated = {f"agent_{i}": bool(coverage_bonus > 0) for i in range(self.num_agents)}
        truncated = {f"agent_{i}": self.steps >= self.max_steps for i in range(self.num_agents)}

        info = self._get_info()
        info["agent_rewards"] = agent_rewards
        info["mean_reward"] = mean_reward

        if any(terminated.values()) or any(truncated.values()):
            end_type = "terminated" if any(terminated.values()) else "truncated"
            self.coverage_rates.append(info['coverage_rate'])
            self.finish_steps.append(self.steps)
            if self.plot_coverage:
                plt.clear_data()
                if self.max_steps == 20:
                    plt.hline(100, color="red")
                    plt.plot(self.coverage_rates)
                else:
                    plt.hline(20, color="red")
                    plt.plot(self.finish_steps)
                plt.show()

        obs = self._get_obs()
        done = any(terminated.values())
        trunc = any(truncated.values())

        return obs, mean_reward, done, trunc, info

    def _handle_communication(self):
        self.telemetry_factory.reset()
        for i in range(self.num_agents):
            data = {}
            if self.telemetry_config.get('position', False):
                data['position'] = tuple(self.agent_positions[i])
            telemetry = self.telemetry_factory.create_telemetry(
                timestamp=self.steps,
                sender_id=i,
                data=data
            )
            # broadcast to neighbors in range
            for j in range(self.num_agents):
                if i == j:
                    continue
                dist = np.linalg.norm(self.agent_positions[i] - self.agent_positions[j])
                if dist <= self.Rcomm:
                    telemetry.receivers.append(j)

        # Distribute
        for t in self.telemetry_factory.telemetries:
            for rcv in t.receivers:
                self.knowledge_bases[rcv].add_knowledge(t)

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        frame = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.uint8)
        # covered cells
        frame[self.grid == 1] = [200, 200, 200]
        # agents
        colors = [
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
            [255, 165, 0],
            [128, 0, 128]
        ]
        for i, pos in enumerate(self.agent_positions):
            c = colors[i % len(colors)]
            frame[pos[0], pos[1]] = c
        return frame

    def close(self):
        pass

#################################
# multiagent_policy.py
#################################

import torch
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.ppo import PPO
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional, Type, Union, Any
import math

class MultiAgentFeaturesExtractor(BaseFeaturesExtractor):
    """
    A configurable multi-agent feature extractor.

    If use_transformer=True, we treat each agent's observation as one "token"
    and use multi-head self-attention to create a combined representation.
    Otherwise, we fall back to a standard MLP-based approach for each agent.
    """
    def __init__(
        self,
        observation_space: spaces.Dict,
        num_agents: int,
        features_dim: int = 128,
        agent_feature_dim: int = 64,
        use_transformer: bool = False,
        n_heads: int = 4,
        n_layers: int = 2
    ):
        # We'll produce a final feature vector of size `features_dim`.
        # But if use_transformer=True, we might have an internal dimension
        # that is bigger to handle multi-head attention.
        super().__init__(observation_space, features_dim=features_dim)
        self.num_agents = num_agents
        self.agent_feature_dim = agent_feature_dim
        self.use_transformer = use_transformer

        # Determine the dimension of each agent's raw observation
        # e.g. from shape(...) in observation_space[f"agent_i"]
        example_key = list(observation_space.spaces.keys())[0]
        single_agent_shape = observation_space.spaces[example_key].shape[0]

        # Basic embedding for each agent
        self.agent_embed = nn.Sequential(
            nn.Linear(single_agent_shape, agent_feature_dim),
            nn.ReLU()
        )

        if use_transformer:
            # Transformer-based approach
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=agent_feature_dim,
                nhead=n_heads,
                dim_feedforward=2 * agent_feature_dim,
                dropout=0.1,
                activation='relu',
                batch_first=True
            )
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
            self.final_proj = nn.Linear(agent_feature_dim, features_dim)
        else:
            # MLP approach (one per agent, then aggregate)
            self.agent_mlps = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(agent_feature_dim, agent_feature_dim),
                    nn.ReLU()
                ) for _ in range(num_agents)
            ])
            self.agg = nn.Linear(num_agents * agent_feature_dim, features_dim)

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        observations: Dict with keys "agent_0", "agent_1", ...
                      each is a (batch_size, obs_dim) tensor
        Returns a (batch_size, features_dim) tensor
        """
        # Convert to a list for consistent ordering
        batch_size = None
        agent_tensors = []
        for i in range(self.num_agents):
            x = observations[f"agent_{i}"].float()  # (batch_size, obs_dim)
            if batch_size is None:
                batch_size = x.shape[0]
            # embed each agent
            embedded = self.agent_embed(x)  # (batch_size, agent_feature_dim)
            agent_tensors.append(embedded)

        if self.use_transformer:
            # Transformer expects (batch_size, sequence_length, d_model)
            # We'll treat agents as sequence_length dimension
            seq = torch.stack(agent_tensors, dim=1)  # (batch_size, num_agents, agent_feature_dim)
            transformed = self.transformer_encoder(seq)  # same shape
            # Possibly we just take the mean across tokens
            pooled = torch.mean(transformed, dim=1)  # (batch_size, agent_feature_dim)
            out = self.final_proj(pooled)            # (batch_size, features_dim)
            return out
        else:
            # MLP approach
            final_reps = []
            for i in range(self.num_agents):
                rep = self.agent_mlps[i](agent_tensors[i])  # (batch_size, agent_feature_dim)
                final_reps.append(rep)
            concat = torch.cat(final_reps, dim=1)           # (batch_size, num_agents*agent_feature_dim)
            out = self.agg(concat)                          # (batch_size, features_dim)
            return out

class MultiAgentActorCriticPolicy(ActorCriticPolicy):
    """
    Slightly adjusted to accommodate multi-agent dictionaries.
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
        normalize_images: bool = False,  # We handle scaling in the env
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
            features_extractor_kwargs = {}
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
    Same PPO structure, just passing num_agents around.
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
            policy=policy,
            env=env,
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

#################################
# train_script.py
#################################

import optuna
import time
import numpy as np

def optuna_train(trial=None, params=None):
    start_time = time.time()

    # Create environment
    env = MultiAgentGridCoverage(
        grid_size=8,
        num_agents=8,
        observation_radius=2,
        max_steps=20,
        render_mode=None,
        plot_coverage=False if params is None else True
    )

    if params is None and trial is not None:
        network_1 = trial.suggest_int("network_1", 32, 256, step=32)
        network_2 = trial.suggest_int("network_2", 32, 256, step=32)
        agent_feature_dim = trial.suggest_int("agent_feature_dim", 32, 256, step=32)
        features_dim = trial.suggest_int("features_dim", 32, 256, step=32)

        n_steps = trial.suggest_int("n_steps", 64, 2048, step=64)
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        n_epochs = trial.suggest_int("n_epochs", 1, 10, step=1)
        gamma = trial.suggest_float("gamma", 0.8, 0.99, step=0.05)
        gae_lambda = trial.suggest_float("gae_lambda", 0.8, 0.99, step=0.05)

        use_transformer = trial.suggest_categorical("use_transformer", [False, True])
        n_heads = trial.suggest_int("n_heads", 1, 4, step=1)
        n_layers = trial.suggest_int("n_layers", 1, 4, step=1)
    else:
        # Hard-coded parameters
        network_1 = params["network_1"]
        network_2 = params["network_2"]
        agent_feature_dim = params["agent_feature_dim"]
        features_dim = params["features_dim"]
        n_steps = params["n_steps"]
        learning_rate = params["learning_rate"]
        n_epochs = params["n_epochs"]
        gamma = params["gamma"]
        gae_lambda = params["gae_lambda"]
        # Additional
        use_transformer = params.get("use_transformer", False)
        n_heads = params.get("n_heads", 2)
        n_layers = params.get("n_layers", 2)

    policy_kwargs = {
        "net_arch": [network_1, network_2],
        "features_extractor_kwargs": {
            "agent_feature_dim": agent_feature_dim,
            "features_dim": features_dim,
            "use_transformer": use_transformer,
            "n_heads": n_heads,
            "n_layers": n_layers
        }
    }

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
        policy_kwargs=policy_kwargs
    )

    # Train (timesteps might need to be bigger for a stable result)
    total_timesteps = 10000 if params is None else 200000
    model.learn(total_timesteps=total_timesteps)

    if params is None:
        # For optuna
        return np.mean(env.coverage_rates)
    else:
        # Log results
        elapsed = time.time() - start_time
        print(f"Time taken for network sizes {network_1} and {network_2}; agent_feat {agent_feature_dim}; features_dim {features_dim}: {elapsed:.2f}s")
        if len(env.coverage_rates) > 0:
            print(f"Max coverage: {max(env.coverage_rates):.2f}%")
            print(f"Mean coverage: {np.mean(env.coverage_rates):.2f}%")
            print(f"Std coverage: {np.std(env.coverage_rates):.2f}%")
        else:
            print("No coverage data logged... possibly ended too early?")
        return np.mean(env.coverage_rates) if len(env.coverage_rates) > 0 else 0.0, elapsed

if __name__ == "__main__":
    # Example usage with fixed hyperparams
    fixed_params = {
        'network_1': 128,
        'network_2': 128,
        'agent_feature_dim': 128,
        'features_dim': 128,
        'n_steps': 1024,
        'learning_rate': 0.001,
        'n_epochs': 10,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        # Additional
        'use_transformer': True,
        'n_heads': 4,
        'n_layers': 4
    }
    print(fixed_params)
    res = optuna_train(None, fixed_params)
    print("Training finished. Coverage stats:", res)
