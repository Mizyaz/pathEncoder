############################################
# single_file_multiagent.py
############################################

import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional, Union, Any
import math
import plotext as plt

from env import MultiAgentGridCoverage

############################################
# FEATURE EXTRACTORS
############################################

class BaseFeaturesExtractor(nn.Module):
    """
    Minimal class to keep some consistency.
    """
    def __init__(self, features_dim: int):
        super().__init__()
        self.features_dim = features_dim

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError

class FastNet(BaseFeaturesExtractor):
    """
    Ultra-fast network with minimal operations and precomputed tensors
    """
    def __init__(self, obs_dim: int, num_agents: int, hidden_dim: int, out_dim: int):
        super().__init__(features_dim=out_dim)
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim

        # Single linear layer per agent with minimal processing
        self.agent_nets = nn.ModuleList([
            nn.Linear(obs_dim, hidden_dim, bias=False)  # Remove bias for speed
            for _ in range(num_agents)
        ])
        
        # Fast projection to output
        self.output_net = nn.Linear(num_agents * hidden_dim, out_dim, bias=False)
        
        # Initialize with orthogonal weights for better gradient flow
        for net in self.agent_nets:
            nn.init.orthogonal_(net.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.output_net.weight, gain=np.sqrt(2))

    def forward(self, x: Dict[str, torch.Tensor]):
        # Process all agents in parallel
        batch_size = x["agent_0"].shape[0]
        
        # Pre-allocate output tensor
        hidden_features = torch.empty(
            batch_size, 
            self.num_agents * self.hidden_dim, 
            device=x["agent_0"].device
        )
        
        # Fast processing of each agent
        for i in range(self.num_agents):
            start_idx = i * self.hidden_dim
            end_idx = start_idx + self.hidden_dim
            
            # Linear + Fast ReLU
            h = self.agent_nets[i](x[f"agent_{i}"])
            hidden_features[:, start_idx:end_idx] = torch.maximum(h, torch.zeros_like(h))
        
        # Final projection
        return self.output_net(hidden_features)

class MLPExtractor(BaseFeaturesExtractor):
    """
    Enhanced MLP approach with better feature processing and normalization
    """
    def __init__(self, obs_dim: int, num_agents: int, hidden_dim: int, out_dim: int):
        super().__init__(features_dim=out_dim)
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim

        # Enhanced per-agent networks with layer normalization
        self.agent_embed = nn.ModuleList([
            nn.Sequential(
                nn.Linear(obs_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU()
            ) for _ in range(num_agents)
        ])

        # Fast projection to output
        self.final_proj = nn.Sequential(
            nn.Linear(num_agents * hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU()
        )

    def forward(self, x: Dict[str, torch.Tensor]):
        # Process each agent's observation
        embedded_agents = []
        for i in range(self.num_agents):
            obs = x[f"agent_{i}"]
            feat = self.agent_embed[i](obs)
            embedded_agents.append(feat)

        # Concatenate and project
        concat = torch.cat(embedded_agents, dim=1)
        return self.final_proj(concat)

############################################
# POLICY
############################################

class MultiAgentPolicy(nn.Module):
    """
    We'll define a basic ActorCritic. 
    """
    def __init__(
        self,
        obs_dim: int,
        num_agents: int,
        action_dim: int,
        feature_extractor: BaseFeaturesExtractor
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.num_agents = num_agents
        self.action_dim = action_dim
        self.features_extractor = feature_extractor

        feat_dim = feature_extractor.features_dim
        
        # Enhanced policy network with better gradient flow
        self.shared_net = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.ReLU()
        )
        
        # Separate policy heads for each agent
        self.policy_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feat_dim, feat_dim // 2),
                nn.LayerNorm(feat_dim // 2),
                nn.ReLU(),
                nn.Linear(feat_dim // 2, action_dim)
            ) for _ in range(num_agents)
        ])
        
        # Enhanced value network
        self.value_net = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, feat_dim // 2),
            nn.LayerNorm(feat_dim // 2),
            nn.ReLU(),
            nn.Linear(feat_dim // 2, 1)
        )

    def forward(self, obs: Dict[str, torch.Tensor]):
        feats = self.features_extractor(obs)
        shared_feats = self.shared_net(feats)
        
        # Generate logits for each agent
        all_logits = []
        for i in range(self.num_agents):
            agent_logits = self.policy_heads[i](shared_feats)
            all_logits.append(agent_logits)
            
        # Concatenate logits and compute value
        logits = torch.cat(all_logits, dim=1)
        value = self.value_net(feats)
        
        return logits, value

    def evaluate_actions(self, obs: Dict[str, torch.Tensor], actions: torch.Tensor):
        logits, values = self.forward(obs)
        batch_size = actions.shape[0]
        
        # Split logits for each agent
        action_probs = []
        for i in range(self.num_agents):
            start_idx = i * self.action_dim
            end_idx = start_idx + self.action_dim
            agent_logits = logits[:, start_idx:end_idx]
            dist = torch.distributions.Categorical(logits=agent_logits)
            action_probs.append(dist.log_prob(actions[:, i]))
        
        # Sum log probs across agents
        log_prob = torch.stack(action_probs, dim=1).sum(dim=1)
        return log_prob, values

############################################
# ROLL OUT STORAGE & PPO
############################################

class RolloutBuffer:
    """
    Minimal roll-out buffer for PPO, storing observations in a dictionary manner.
    """
    def __init__(self, buffer_size, num_agents, obs_dim, action_dim, device="cpu"):
        self.buffer_size = buffer_size
        self.device = device
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # We store dictionary obs. We'll keep it as a list of dicts.
        self.observations: List[Dict[str, torch.Tensor]] = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []
        self.next_observations: List[Dict[str, torch.Tensor]] = []

    def add(
        self,
        obs: Dict[str, np.ndarray],
        action: np.ndarray,
        reward: float,
        done: bool,
        value: float,
        log_prob: float
    ):
        # We'll store them for each step
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
        self.log_probs.append(log_prob)

    def compute_returns_and_advantage(self, next_value, gamma, lam):
        # We'll do GAE-lambda
        size = len(self.rewards)
        self.advantages = np.zeros(size, dtype=np.float32)
        self.returns = np.zeros(size, dtype=np.float32)

        gae = 0.0
        for step in reversed(range(size)):
            delta = self.rewards[step] + gamma * (0 if self.dones[step] else self.values[step + 1 if step + 1 < size else -1]) - self.values[step]
            gae = delta + gamma * lam * (0 if self.dones[step] else gae)
            self.advantages[step] = gae
            self.returns[step] = gae + self.values[step]

    def clear(self):
        self.observations = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []

class MultiAgentPPO:
    """
    Basic PPO. We won't do full multi-agent training in separate ways, 
    but we treat the environment's observation as a dictionary, 
    and produce multiple discrete actions from one combined policy.
    """
    def __init__(
        self,
        env: MultiAgentGridCoverage,
        train_mode: str = "normal",
        gamma: float = 0.99,
        lam: float = 0.95,
        lr: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        device: str = "cpu",
        seed: Optional[int] = None,
        ent_coef: float = 0.01,
        clip_range: float = 0.2,
        max_grad_norm: float = 0.5,
        feature_extractor: Optional[BaseFeaturesExtractor] = None
    ):
        self.env = env
        self.num_agents = env.num_agents
        self.obs_dim = len(env._get_obs()[f"agent_0"])
        self.action_dim = 4

        self.gamma = gamma
        self.lam = lam
        self.lr = lr
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.ent_coef = ent_coef
        self.clip_range = clip_range
        self.max_grad_norm = max_grad_norm

        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        self.device = torch.device(device)
        
        # Use provided feature extractor or create default
        if feature_extractor is None:
            if train_mode == "fast":
                feature_extractor = FastNet(
                    obs_dim=self.obs_dim,
                    num_agents=self.num_agents,
                    hidden_dim=32,
                    out_dim=64
                )
            else:
                feature_extractor = MLPExtractor(
                    obs_dim=self.obs_dim,
                    num_agents=self.num_agents,
                    hidden_dim=64,
                    out_dim=128
                )

        self.policy = MultiAgentPolicy(
            obs_dim=self.obs_dim,
            num_agents=self.num_agents,
            action_dim=self.action_dim,
            feature_extractor=feature_extractor
        ).to(device)

        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)
        
        # Pre-allocate tensors for faster training
        self.advantages = torch.zeros(self.n_steps, device=self.device)
        self.returns = torch.zeros(self.n_steps, device=self.device)
        self.value_preds = torch.zeros(self.n_steps, device=self.device)
        self.action_log_probs = torch.zeros(self.n_steps, device=self.device)
        
        # Initialize rollout buffer
        self.rollout_buffer = RolloutBuffer(
            buffer_size=self.n_steps,
            num_agents=self.num_agents,
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            device=device
        )

    def _process_rollout(self):
        """Process rollout data efficiently"""
        with torch.no_grad():
            last_obs = {
                k: torch.as_tensor(v, device=self.device).float().unsqueeze(0)
                for k, v in self.rollout_buffer.observations[-1].items()
            }
            _, last_value = self.policy(last_obs)
            last_value = last_value.squeeze(-1)

            # Convert rewards and values to tensor
            rewards = torch.tensor(self.rollout_buffer.rewards, device=self.device)
            values = torch.tensor(self.rollout_buffer.values, device=self.device)
            dones = torch.tensor(self.rollout_buffer.dones, device=self.device)

            # Compute GAE and returns in one go
            gae = 0
            for t in reversed(range(self.n_steps)):
                if t == self.n_steps - 1:
                    next_value = last_value
                else:
                    next_value = values[t + 1]
                
                delta = rewards[t] + self.gamma * (1.0 - dones[t]) * next_value - values[t]
                gae = delta + self.gamma * self.lam * (1.0 - dones[t]) * gae
                self.returns[t] = gae + values[t]
            
            self.advantages = self.returns - values
            self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    def _update_policy(self, obs_batch, actions_batch, old_values, old_log_probs, returns_batch, advantages_batch):
        # Convert numpy arrays to tensors
        actions = torch.as_tensor(actions_batch, device=self.device)
        old_values = torch.as_tensor(old_values, device=self.device)
        old_log_probs = torch.as_tensor(old_log_probs, device=self.device)
        advantages = torch.as_tensor(advantages_batch, device=self.device)
        returns = torch.as_tensor(returns_batch, device=self.device)

        # Evaluate actions
        log_probs, values = self.policy.evaluate_actions(obs_batch, actions)
        values = values.squeeze(-1)

        # Policy loss
        ratio = torch.exp(log_probs - old_log_probs)
        policy_loss1 = advantages * ratio
        policy_loss2 = advantages * torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)
        policy_loss = -torch.min(policy_loss1, policy_loss2).mean()

        # Value loss
        value_loss = 0.5 * (returns - values).pow(2).mean()

        # Total loss
        loss = policy_loss + value_loss

        # Optimization step
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return policy_loss.item(), value_loss.item()

    def learn(self, total_timesteps=10000):
        timesteps_so_far = 0
        
        while timesteps_so_far < total_timesteps:
            self.rollout_buffer.clear()
            
            # Collect rollout
            obs_dict, _ = self.env.reset()
            episode_steps = 0
            
            while episode_steps < self.n_steps:
                # Convert observations to tensors
                tensor_obs = {
                    k: torch.as_tensor(v, device=self.device).float().unsqueeze(0)
                    for k, v in obs_dict.items()
                }
                
                # Get action
                with torch.no_grad():
                    logits, value = self.policy(tensor_obs)
                    value = value.squeeze(-1).cpu().numpy()
                    
                    # Sample actions for all agents
                    actions = []
                    log_prob_sum = 0
                    for i in range(self.num_agents):
                        start_idx = i * self.action_dim
                        end_idx = start_idx + self.action_dim
                        agent_logits = logits[0, start_idx:end_idx]
                        dist = torch.distributions.Categorical(logits=agent_logits)
                        action = dist.sample()
                        log_prob_sum += dist.log_prob(action).cpu().numpy()
                        actions.append(action.cpu().numpy())
                
                actions = np.array(actions)
                next_obs, reward, done, truncated, info = self.env.step(actions)
                
                self.rollout_buffer.add(
                    obs_dict, actions, reward, done or truncated,
                    value[0], log_prob_sum
                )
                
                obs_dict = next_obs
                episode_steps += 1
                timesteps_so_far += 1
                
                if done or truncated:
                    obs_dict, _ = self.env.reset()
            
            # Process rollout
            self._process_rollout()
            
            # Update policy
            inds = np.arange(self.n_steps)
            for _ in range(self.n_epochs):
                np.random.shuffle(inds)
                for start in range(0, self.n_steps, self.batch_size):
                    end = start + self.batch_size
                    mb_inds = inds[start:end]
                    
                    obs_batch = {
                        k: torch.stack([
                            torch.as_tensor(self.rollout_buffer.observations[i][k], device=self.device)
                            for i in mb_inds
                        ])
                        for k in self.rollout_buffer.observations[0].keys()
                    }
                    
                    self._update_policy(
                        obs_batch,
                        np.array([self.rollout_buffer.actions[i] for i in mb_inds]),
                        np.array([self.rollout_buffer.values[i] for i in mb_inds]),
                        np.array([self.rollout_buffer.log_probs[i] for i in mb_inds]),
                        self.returns[mb_inds].cpu().numpy(),
                        self.advantages[mb_inds].cpu().numpy()
                    )


############################################
# TRAIN SCRIPT (MAIN)
############################################

def main():
    """
    We define a single-file runner that can take a train_mode param.
    Possible values: 'normal', 'hyperfast', 'experimental'
    This is where we criticize ourselves and tweak solutions:
    - If your coverage saturates, try bigger net, or adjust gamma upwards (0.99 or 0.995).
    - If training is too slow, use 'hyperfast' or reduce n_steps.
    - If you want to test new structures, use 'experimental'.
    """
    train_mode = "fast"       # Using our new fast mode
    gamma = 0.99              # Standard discount
    lam = 0.95               # Standard GAE
    lr = 1e-3                # Slightly higher learning rate
    n_steps = 256            # Reduced for faster updates
    batch_size = 64          # Smaller batches for speed
    n_epochs = 5             # Fewer epochs
    ent_coef = 0.01         # Standard entropy
    clip_range = 0.2

    print(f"Running train_mode={train_mode}, gamma={gamma}, n_steps={n_steps}")
    env = MultiAgentGridCoverage(
        grid_size=8,
        num_agents=8,
        observation_radius=2,  # Reduced radius
        max_steps=20,         # Shorter episodes
        render_mode=None,
        plot_coverage=True,
        Rcomm=3.0,
        telemetry_config={'position': True},
        telemetry_shapes={'position': (2,)}
    )

    # Create feature extractor based on mode
    if train_mode == "fast":
        feature_extractor = FastNet(
            obs_dim=len(env._get_obs()["agent_0"]),
            num_agents=env.num_agents,
            hidden_dim=32,    # Smaller hidden dim
            out_dim=64        # Smaller output dim
        )
    else:
        feature_extractor = MLPExtractor(
            obs_dim=len(env._get_obs()["agent_0"]),
            num_agents=env.num_agents,
            hidden_dim=64,
            out_dim=128
        )

    ppo = MultiAgentPPO(
        env=env,
        train_mode=train_mode,
        gamma=gamma,
        lam=lam,
        lr=lr,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        ent_coef=ent_coef,
        clip_range=clip_range,
        feature_extractor=feature_extractor  # Pass feature extractor directly
    )

    start_time = time.time()
    ppo.learn(total_timesteps=10000)
    elapsed = time.time() - start_time
    print(f"Training completed in {elapsed:.2f} seconds.")

if __name__ == "__main__":
    main()
