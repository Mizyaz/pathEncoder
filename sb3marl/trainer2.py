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

# OPTIONAL: If you truly want to test PyTorch Geometric, you'd install it
# But here's a minimal demonstration without the real import, to keep this code run-safe.
# from torch_geometric.nn import MessagePassing  # an example usage

############################################
# ENVIRONMENT
############################################

class MultiAgentGridCoverage:
    """
    Multi-agent coverage environment. 
    No structure is broken from previous versions, but we can refine minor details.
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
        self.grid_size = grid_size
        self.num_agents = num_agents
        self.obs_radius = observation_radius
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.plot_coverage = plot_coverage
        self.Rcomm = Rcomm
        self.telemetry_config = telemetry_config if telemetry_config else {"position": True}

        # Internals
        self.total_cells = grid_size * grid_size
        self.steps = 0
        self.coverage_rates: List[float] = []
        self.finish_steps: List[int] = []

        # We'll define action and observation spaces conceptually
        # (no gym spaces usage for a single-file approach, but we store them anyway)
        self.action_space = (4,) * self.num_agents  # Each agent has discrete(4)
        # For observation shapes, we define them on first reset:
        # Each agent sees local_view + 2-dim position + coverage => shape T.B.D.

        # Communication system
        self.telemetry_factory = self.TelemetryFactory(self.telemetry_config)
        self.knowledge_bases = [self.Knowledge(i, num_agents) for i in range(num_agents)]

        # Internal states
        self.grid = None
        self.agent_positions = None
        self.coverage_rates = []
        self.finish_steps = []

        # Do an initial reset so we can figure out observation shapes
        self.reset()

    class Telemetry:
        def __init__(self, timestamp: int, sender_id: int, data: Dict):
            self.timestamp = timestamp
            self.sender_id = sender_id
            self.data = data
            self.receivers: List[int] = []

    class TelemetryFactory:
        def __init__(self, config: Dict):
            self.config = config
            self.telemetries: List[MultiAgentGridCoverage.Telemetry] = []

        def reset(self):
            self.telemetries = []

        def create_telemetry(self, timestamp: int, sender_id: int, data: Dict):
            t = MultiAgentGridCoverage.Telemetry(timestamp, sender_id, data)
            self.telemetries.append(t)
            return t

    class Knowledge:
        """Stores knowledge about other agents."""
        def __init__(self, agent_id: int, num_agents: int):
            self.agent_id = agent_id
            self.num_agents = num_agents
            self.knowledge: Dict[int, Dict] = {i: {} for i in range(num_agents)}

        def add_knowledge(self, telemetry: 'MultiAgentGridCoverage.Telemetry'):
            sender = telemetry.sender_id
            if sender != self.agent_id:
                for k, v in telemetry.data.items():
                    self.knowledge[sender][k] = v

    def reset(self):
        self.steps = 0


        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        self.agent_positions = np.zeros((self.num_agents, 2), dtype=int)
        self.grid[0, 0] = 1

        self.telemetry_factory.reset()
        for kb in self.knowledge_bases:
            kb.knowledge = {i: {} for i in range(self.num_agents)}

        obs = self._get_obs()
        return obs, self._get_info()

    def _get_local_view(self, i: int):
        pos = self.agent_positions[i]
        r = self.obs_radius
        pad_grid = np.pad(self.grid, ((r, r), (r, r)), mode='constant', constant_values=0)
        center_x, center_y = pos + r
        local_view = pad_grid[center_x - r:center_x + r + 1, center_y - r:center_y + r + 1]
        return local_view

    def _get_obs(self):
        # Each agent's obs: local_view.flatten + 2 pos + coverage
        coverage = np.mean(self.grid)
        obs_dict = {}
        for i in range(self.num_agents):
            local_view = self._get_local_view(i).flatten()
            rel_pos = self.agent_positions[i] / self.grid_size
            combined = np.concatenate([local_view, rel_pos, [coverage]]).astype(np.float32)
            # scale roughly [-1,1]
            combined = 2.0 * (combined - 0.5)
            obs_dict[f"agent_{i}"] = combined
        return obs_dict

    def _get_info(self):
        coverage = np.mean(self.grid)
        return {
            "coverage": coverage,
            "steps": self.steps,
            "coverage_rate": coverage * 100.0
        }

    def step(self, actions: np.ndarray):
        old_coverage = np.mean(self.grid)
        agent_rewards = {}
        for i in range(self.num_agents):
            action = actions[i]
            pos = self.agent_positions[i].copy()
            if action == 0:  # Up
                pos[0] = max(0, pos[0] - 1)
            elif action == 1:  # Right
                pos[1] = min(self.grid_size - 1, pos[1] + 1)
            elif action == 2:  # Down
                pos[0] = min(self.grid_size - 1, pos[0] + 1)
            elif action == 3:  # Left
                pos[1] = max(0, pos[1] - 1)

            collision = False
            for j, other_pos in enumerate(self.agent_positions):
                if j != i and np.array_equal(pos, other_pos):
                    collision = True
                    break

            if not collision:
                self.agent_positions[i] = pos
                was_covered = self.grid[pos[0], pos[1]] == 1
                self.grid[pos[0], pos[1]] = 1
                agent_rewards[f"agent_{i}"] = 1.0 if not was_covered else -0.5
            else:
                agent_rewards[f"agent_{i}"] = -0.5

        self.steps += 1
        new_coverage = np.mean(self.grid)
        coverage_improv = new_coverage - old_coverage
        coverage_bonus = 100.0 if np.all(self.grid == 1) else 0.0

        for k in agent_rewards:
            agent_rewards[k] += coverage_improv * 10.0 + coverage_bonus

        # Communication
        self._handle_communication()

        mean_reward = np.mean(list(agent_rewards.values()))
        term = coverage_bonus > 0
        trunc = self.steps >= self.max_steps
        info = self._get_info()
        info["agent_rewards"] = agent_rewards
        info["mean_reward"] = mean_reward

        if term or trunc:
            self.coverage_rates.append(info["coverage_rate"])
            self.finish_steps.append(self.steps)
            print(f"coverage rate: {info['coverage_rate']:.2f}%")
            plt.clear_data()
            plt.plot(self.coverage_rates)
            plt.show()

        obs = self._get_obs()
        done = term
        truncated = trunc
        return obs, mean_reward, done, truncated, info

    def _handle_communication(self):
        self.telemetry_factory.reset()
        for i in range(self.num_agents):
            data = {}
            if self.telemetry_config.get("position", False):
                data["position"] = tuple(self.agent_positions[i])
            t = self.telemetry_factory.create_telemetry(self.steps, i, data)
            # broadcast
            for j in range(self.num_agents):
                if i == j:
                    continue
                dist = np.linalg.norm(self.agent_positions[i] - self.agent_positions[j])
                if dist <= self.Rcomm:
                    t.receivers.append(j)

        for t in self.telemetry_factory.telemetries:
            for rcv in t.receivers:
                self.knowledge_bases[rcv].add_knowledge(t)

    def render(self):
        # If you had a rendering solution, you'd do it here
        pass

    def close(self):
        pass

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

class MLPExtractor(BaseFeaturesExtractor):
    """
    Normal MLP approach, each agent observation is embedded separately, then concatenated.
    """
    def __init__(self, obs_dim: int, num_agents: int, hidden_dim: int, out_dim: int):
        super().__init__(features_dim=out_dim)
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim

        self.agent_embed = nn.ModuleList([
            nn.Sequential(
                nn.Linear(obs_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            ) for _ in range(num_agents)
        ])
        self.final_proj = nn.Linear(num_agents * hidden_dim, out_dim)

    def forward(self, x: Dict[str, torch.Tensor]):
        # x is {f"agent_0": tensor, f"agent_1": tensor, ...}
        embedded_agents = []
        for i in range(self.num_agents):
            obs = x[f"agent_{i}"]
            feat = self.agent_embed[i](obs)
            embedded_agents.append(feat)

        concat = torch.cat(embedded_agents, dim=1)
        return self.final_proj(concat)

class HyperFastExtractor(BaseFeaturesExtractor):
    """
    Hyper-fast mode: fewer layers, smaller dims. 
    This is basically a 'minimized' version to run quickly.
    """
    def __init__(self, obs_dim: int, num_agents: int, hidden_dim: int, out_dim: int):
        super().__init__(features_dim=out_dim)
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim

        # Very small net: just one linear per agent
        self.agent_embed = nn.ModuleList([
            nn.Sequential(
                nn.Linear(obs_dim, hidden_dim),
                nn.ReLU()
            ) for _ in range(num_agents)
        ])
        self.final_proj = nn.Linear(num_agents * hidden_dim, out_dim)

    def forward(self, x: Dict[str, torch.Tensor]):
        embedded_agents = []
        for i in range(self.num_agents):
            obs = x[f"agent_{i}"]
            feat = self.agent_embed[i](obs)
            embedded_agents.append(feat)
        concat = torch.cat(embedded_agents, dim=1)
        return self.final_proj(concat)

class ExperimentalGraphExtractor(BaseFeaturesExtractor):
    """
    A pretend graph-based feature extractor (for demonstration).
    We simulate how you'd handle adjacency from multiple agents.

    Real usage might require torch_geometric, but we won't rely on it fully here.
    We'll do a naive adjacency approach.
    """
    def __init__(self, obs_dim: int, num_agents: int, hidden_dim: int, out_dim: int):
        super().__init__(features_dim=out_dim)
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim

        # naive per-agent embed
        self.agent_embed = nn.Linear(obs_dim, hidden_dim)
        # adjacency aggregator
        self.edge_lin = nn.Linear(hidden_dim, hidden_dim)
        self.final_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: Dict[str, torch.Tensor]):
        """
        We'll do:
        1) embed each agent individually
        2) produce adjacency matrix (all-to-all for demonstration)
        3) sum neighbor embeddings
        4) pass it through final projection
        """
        # (batch_size, obs_dim)
        # We'll store them in a list, then stack
        agent_feats = []
        for i in range(self.num_agents):
            obs_i = x[f"agent_{i}"]
            # single linear embed
            f_i = self.agent_embed(obs_i)
            agent_feats.append(f_i)

        # We'll have shape [num_agents, batch_size, hidden_dim]
        # but let's reorder it to [batch_size, num_agents, hidden_dim]
        batch_feats = torch.stack(agent_feats, dim=1)  # (batch_size, n_agents, hidden_dim)

        # adjacency matrix: for demonstration, let’s do full adjacency except self
        # shape (n_agents, n_agents) => 1 or 0
        # we do it in batch sense, but let's keep it simple
        A = (torch.ones(self.num_agents, self.num_agents) - torch.eye(self.num_agents))
        # Move to device
        A = A.to(batch_feats.device)

        # naive message passing: out_feat_i = sum_j (A_ij * MLP(agent_j_feat))
        # This is extremely simplified
        # shape => (batch_size, n_agents, hidden_dim)
        msg_pass = []
        for i in range(self.num_agents):
            # gather neighbors
            neighbors = []
            for j in range(self.num_agents):
                if A[i, j] > 0:
                    neighbors.append(batch_feats[:, j, :])  # (batch_size, hidden_dim)
            if len(neighbors) > 0:
                sum_neighbors = torch.stack(neighbors, dim=2).sum(dim=2)
            else:
                sum_neighbors = torch.zeros_like(batch_feats[:, i, :])
            # transform
            sum_neighbors = self.edge_lin(sum_neighbors)
            # combine self feature + neighbor
            new_i = batch_feats[:, i, :] + sum_neighbors
            msg_pass.append(new_i)

        # shape => (batch_size, n_agents, hidden_dim)
        final_feats = torch.stack(msg_pass, dim=1)
        # mean pooling across agents
        pooled = final_feats.mean(dim=1)  # (batch_size, hidden_dim)
        return self.final_proj(pooled)

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
        action_dim: int,  # 4 for each agent, but we treat MultiDiscrete => separate heads
        feature_extractor: BaseFeaturesExtractor
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.num_agents = num_agents
        self.action_dim = action_dim
        self.features_extractor = feature_extractor

        # We produce a single latent dimension from the features
        feat_dim = feature_extractor.features_dim

        # policy & value heads
        # Because we have multi-discrete(4) for each agent, we do something like:
        # each agent gets a 4-dim distribution => total param = num_agents * 4
        # We'll produce them from a single network, but chunk them per agent.
        self.policy_net = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, num_agents * action_dim)  # we chunk
        )
        self.value_net = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, 1)
        )

    def forward(self, obs: Dict[str, torch.Tensor]):
        # produce policy logits & value
        feats = self.features_extractor(obs)  # (batch_size, feat_dim)
        logits = self.policy_net(feats)       # (batch_size, num_agents*action_dim)
        value = self.value_net(feats)         # (batch_size, 1)
        return logits, value

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
        max_grad_norm: float = 0.5
    ):
        self.env = env
        self.num_agents = env.num_agents
        self.obs_dim = len(env._get_obs()[f"agent_0"])  # shape
        self.action_dim = 4  # fixed discrete(4)

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

        # Decide which feature extractor to use, based on train_mode
        if train_mode == "hyperfast":
            # smaller dimension
            feature_extractor = HyperFastExtractor(
                obs_dim=self.obs_dim,
                num_agents=self.num_agents,
                hidden_dim=16,   # small
                out_dim=32       # small
            )
        elif train_mode == "experimental":
            # pretend graph-based
            feature_extractor = ExperimentalGraphExtractor(
                obs_dim=self.obs_dim,
                num_agents=self.num_agents,
                hidden_dim=64,
                out_dim=64
            )
        else:
            # normal
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
        self.rollout_buffer = RolloutBuffer(
            buffer_size=self.n_steps,
            num_agents=self.num_agents,
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            device=device
        )
        self.device = device

    def sample_action(self, obs: Dict[str, torch.Tensor]):
        """
        Forward pass, sample from categorical distribution for each agent
        """
        with torch.no_grad():
            logits, value = self.policy(obs)
        # logits shape: (batch, num_agents*action_dim)
        # we want to slice for each agent: 0:4, 4:8, etc.
        # But here we have a single batch, so shape is (1, n_agents*4).
        logits = logits[0]
        value = value[0]

        # We'll do a separate categorical distribution for each agent
        actions = []
        log_probs = []
        offset = 0
        for i in range(self.num_agents):
            agent_logits = logits[offset:offset+self.action_dim]
            offset += self.action_dim

            dist = torch.distributions.Categorical(logits=agent_logits)
            a = dist.sample()
            lp = dist.log_prob(a)
            actions.append(a.item())
            log_probs.append(lp.item())

        return np.array(actions), sum(log_probs), value.item()

    def train_on_batch(self, obs_batch, actions_batch, old_values, old_log_probs, returns, advantages):
        # obs_batch is a list of dictionaries
        # We'll process them in a single forward pass
        # shape => (batch_size,) of dict
        # we want something => dict of (batch_size, obs_dim)

        batch_dict = {}
        for i in range(self.num_agents):
            batch_dict[f"agent_{i}"] = []

        for ob in obs_batch:
            for i in range(self.num_agents):
                batch_dict[f"agent_{i}"].append(ob[f"agent_{i}"])

        # Now convert each to tensor
        for k in batch_dict:
            batch_dict[k] = torch.tensor(np.stack(batch_dict[k]), dtype=torch.float32, device=self.device)

        actions_batch = torch.tensor(np.array(actions_batch), device=self.device)
        old_values = torch.tensor(np.array(old_values), device=self.device)
        old_log_probs = torch.tensor(np.array(old_log_probs), device=self.device)
        returns = torch.tensor(np.array(returns), device=self.device)
        advantages = torch.tensor(np.array(advantages), device=self.device)

        logits, values = self.policy(batch_dict)
        # shape => (batch_size, n_agents*action_dim), (batch_size, 1)

        dist_list = []
        offset = 0
        for i in range(self.num_agents):
            agent_logits = logits[:, offset:offset+self.action_dim]
            offset += self.action_dim
            dist = torch.distributions.Categorical(logits=agent_logits)
            dist_list.append(dist)

        # Now we must compute the log_probs for the given actions
        # actions_batch is shape (batch_size, num_agents)
        # but we stored them as a 1D array. Let's reshape:
        actions_batch = actions_batch.view(-1, self.num_agents)
        log_probs_list = []
        ent_list = []

        for i, dist in enumerate(dist_list):
            a_i = actions_batch[:, i]
            lp_i = dist.log_prob(a_i)
            ent_i = dist.entropy()
            log_probs_list.append(lp_i)
            ent_list.append(ent_i)

        new_log_probs = torch.stack(log_probs_list, dim=1).sum(dim=1)
        entropy = torch.stack(ent_list, dim=1).mean()  # average

        values = values.squeeze(-1)
        # ratio
        ratio = torch.exp(new_log_probs - old_log_probs)

        # clipped surrogate
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        value_loss = nn.functional.mse_loss(returns, values)

        loss = policy_loss + 0.5 * value_loss - self.ent_coef * entropy

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return policy_loss.item(), value_loss.item(), entropy.item()

    def learn(self, total_timesteps=10000):
        obs_dict, info = self.env.reset()
        done = False
        truncated = False
        step_count = 0

        while step_count < total_timesteps:
            self.rollout_buffer.clear()

            for _ in range(self.n_steps):
                # convert obs_dict => dict of shape (1, obs_dim)
                # forward -> sample actions
                tensor_dict = {}
                for i in range(self.num_agents):
                    obs = obs_dict[f"agent_{i}"]
                    obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                    tensor_dict[f"agent_{i}"] = obs_t

                with torch.no_grad():
                    logits, val = self.policy(tensor_dict)

                actions, logp, value_est = self.sample_action(tensor_dict)
                next_obs, reward, d, t, info = self.env.step(actions)

                self.rollout_buffer.add(obs_dict, actions, reward, d, value_est, logp)

                obs_dict = next_obs
                done = d
                truncated = t
                step_count += 1
                if done or truncated:
                    obs_dict, info = self.env.reset()
                    break

            # Now we have n_steps in the buffer, compute returns
            # For the last step, get next value
            if not done and not truncated:
                # get next value from the policy
                tensor_dict_ = {}
                for i in range(self.num_agents):
                    obs = obs_dict[f"agent_{i}"]
                    obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                    tensor_dict_[f"agent_{i}"] = obs_t
                with torch.no_grad():
                    _, val_ = self.policy(tensor_dict_)
                next_value = val_.item()
            else:
                next_value = 0.0

            self.rollout_buffer.compute_returns_and_advantage(next_value, self.gamma, self.lam)

            # optimize
            # We'll do a naive approach: random mini-batches
            total_steps = len(self.rollout_buffer.rewards)
            indices = np.arange(total_steps)
            for epoch_i in range(self.n_epochs):
                np.random.shuffle(indices)
                for start in range(0, total_steps, self.batch_size):
                    end = start + self.batch_size
                    mb_idx = indices[start:end]
                    obs_b = [self.rollout_buffer.observations[i] for i in mb_idx]
                    actions_b = [self.rollout_buffer.actions[i] for i in mb_idx]
                    old_vals_b = [self.rollout_buffer.values[i] for i in mb_idx]
                    old_log_b = [self.rollout_buffer.log_probs[i] for i in mb_idx]
                    returns_b = [self.rollout_buffer.returns[i] for i in mb_idx]
                    adv_b = [self.rollout_buffer.advantages[i] for i in mb_idx]

                    p_loss, v_loss, ent = self.train_on_batch(
                        obs_b, actions_b, old_vals_b, old_log_b, returns_b, adv_b
                    )

        print("Finished training. Steps used:", step_count)
        if self.env.coverage_rates:
            print("Max coverage:", max(self.env.coverage_rates))
            print("Mean coverage:", np.mean(self.env.coverage_rates))
        else:
            print("No coverage recorded; might not have triggered coverage logging.")


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
    train_mode = "normal"      # "normal", "hyperfast", or "experimental"
    gamma = 0.99               # Not too low: 0.8 is a hack
    lam = 0.95
    lr = 1e-3
    n_steps = 512
    batch_size = 64
    n_epochs = 8
    ent_coef = 0.01
    clip_range = 0.2

    print(f"Running train_mode={train_mode}, gamma={gamma}, n_steps={n_steps}")
    env = MultiAgentGridCoverage(
        grid_size=8,
        num_agents=8,
        observation_radius=2,
        max_steps=20,
        render_mode=None,
        plot_coverage=False,
        Rcomm=3.0
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
        clip_range=clip_range
    )

    start_time = time.time()
    ppo.learn(total_timesteps=10000)
    elapsed = time.time() - start_time
    print(f"Training completed in {elapsed:.2f} seconds.")

if __name__ == "__main__":
    main()
