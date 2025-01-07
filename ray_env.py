from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray import tune

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

import ray

try:
    import plotext as plt
    PLOTEXT_AVAILABLE = True
except ImportError:
    PLOTEXT_AVAILABLE = False

# Optional: graph-based features if torch_geometric installed
GEO_AVAILABLE = False
try:
    import torch_geometric.nn as tgnn
    GEO_AVAILABLE = True
except ImportError:
    pass

####################
# Multi-Agent Environment
####################

class EnvironmentTrace:
    def __init__(self):
        self.envs = []
        self.coverage_rates = []
        self.finish_steps = []

    def get_overall_of_attr(self, attribute_name):
        initial_list = []
        for env in self.envs:
            initial_list += getattr(env, attribute_name)
        setattr(self, attribute_name, initial_list)
    
    def get_overall_average(self, attribute_name):
        return np.mean(getattr(self, attribute_name))

environment_trace = EnvironmentTrace()


class MultiAgentGridCoverage(MultiAgentEnv):
    """
    Multi-agent grid coverage environment with optional telemetry-based communication.
    Each agent has 5 possible actions: [0: Up, 1: Right, 2: Down, 3: Left, 4: Stay].
    Observations and actions are returned as dictionaries with keys like "agent_0", "agent_1", etc.
    Supports different operational modes: coverage, coverage with GCS connectivity, and target detection.
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": 4}

    def __init__(
        self,
        config=None,
        **kwargs
    ):
        super().__init__()
        # Handle both dictionary config and direct kwargs
        if config is None:
            config = kwargs
        elif isinstance(config, dict):
            config = {**config, **kwargs}  # Merge both configs, kwargs take precedence

        # Extract configuration with defaults
        self.grid_size = int(config.get("grid_size", 8))
        self.n_agents = int(config.get("num_agents", 8))
        self.obs_radius = int(config.get("observation_radius", 2))
        self.max_steps = int(config.get("max_steps", 50))
        self.render_mode = config.get("render_mode", None)
        self.plot_coverage = bool(config.get("plot_coverage", False))
        self.Rcomm = float(config.get("Rcomm", 8 * np.sqrt(2)))
        self.telemetry_config = config.get("telemetry_config", {"position": True})
        self.telemetry_shapes = config.get("telemetry_shapes", {"position": (2,)})
        self.enable_connectivity = bool(config.get("enable_connectivity", False))
        self.enable_target_detection = bool(config.get("enable_target_detection", False))
        self.n_targets = int(config.get("n_targets", 0))
        self.target_types = config.get("target_types", [1] * self.n_targets)
        self.reward_weights = config.get("reward_weights", {
            "coverage": 1.0,
            "collision": -0.5,
            "coverage_bonus": 100.0,
            "target_detection": 2.0,
            "target_completion": 5.0,
        })

        environment_trace.envs.append(self)


        self.agents = [f"agent_{i}" for i in range(self.n_agents)]
        self.possible_agents = self.agents.copy()
        self.total_cells = self.grid_size * self.grid_size
        self.steps = 0

        # Calculate observation space size
        local_view_size = (2 * self.obs_radius + 1) ** 2  # Local grid view
        position_size = 2  # Agent's relative position
        coverage_size = 1  # Overall coverage
        other_agents_size = (self.n_agents - 1) * 2  # Other agents' positions from telemetry
        self.obs_size = local_view_size + position_size + coverage_size + other_agents_size

        # Action space: Dict of Discrete(5) for each agent
        self.action_space = spaces.Dict({
            agent_id: spaces.Discrete(5) for agent_id in self.agents
        })

        # Observation space: Dict of Box spaces for each agent
        self.observation_space = spaces.Dict({
            agent_id: spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(self.obs_size,),
                dtype=np.float32
            ) for agent_id in self.agents
        })

        # Communication/telemetry system
        self.telemetry_factory = self.TelemetryFactory(self.telemetry_config)
        self.knowledge_bases = [
            self.Knowledge(i, self.n_agents, self.telemetry_shapes)
            for i in range(self.n_agents)
        ]

        # Tracking coverage
        self.coverage_rates = []
        self.finish_steps = []

        # Initialize targets if target detection is enabled
        if self.enable_target_detection:
            self.target_positions = self._initialize_targets()

        self.reset()

    class Telemetry:
        """Telemetry data structure."""
        def __init__(self, timestamp: int, sender_id: int, data: Dict[str, Any]):
            self.timestamp = timestamp
            self.sender_id = sender_id
            self.data = data
            self.receivers: List[int] = []

    class TelemetryFactory:
        """Factory for creating telemetry objects."""
        def __init__(self, config: Dict[str, Any]):
            self.config = config
            self.telemetries: List['MultiAgentGridCoverage.Telemetry'] = []

        def reset(self):
            self.telemetries = []

        def create_telemetry(self, timestamp: int, sender_id: int, data: Dict[str, Any]) -> 'MultiAgentGridCoverage.Telemetry':
            telemetry = MultiAgentGridCoverage.Telemetry(timestamp, sender_id, data)
            self.telemetries.append(telemetry)
            return telemetry

    class Knowledge:
        """Aggregated knowledge base for each agent."""
        def __init__(self, agent_id: int, n_agents: int, telemetry_shapes: Dict[str, Tuple[int, ...]]):
            self.agent_id = agent_id
            self.n_agents = n_agents
            self.knowledge: Dict[int, Dict[str, np.ndarray]] = {i: {} for i in range(n_agents)}

            for n in range(n_agents):
                for key, shape in telemetry_shapes.items():
                    self.knowledge[n][key] = np.zeros(shape, dtype=np.float32)

        def add_knowledge(self, telemetry: 'MultiAgentGridCoverage.Telemetry'):
            sender = telemetry.sender_id
            if sender != self.agent_id:
                for key, value in telemetry.data.items():
                    self.knowledge[sender][key] = np.array(value, dtype=np.float32)

    def _initialize_targets(self) -> List[Tuple[int, int]]:
        """Randomly initialize target positions on the grid."""
        np.random.seed()  # Ensure randomness
        positions = set()
        while len(positions) < self.n_targets:
            pos = tuple(np.random.randint(0, self.grid_size, size=2))
            if pos != (0, 0):  # Avoid starting position
                positions.add(pos)
        return list(positions)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        # Reset grid coverage
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        # Start all agents at random positions (ensure no overlap)
        self.agent_positions = self._initialize_agents()
        # Mark initial coverage
        for pos in self.agent_positions:
            self.grid[pos[0], pos[1]] = 1.0

        # Reset telemetry
        self.telemetry_factory.reset()
        for kb in self.knowledge_bases:
            kb.knowledge = {i: {} for i in range(self.n_agents)}
            for n in range(self.n_agents):
                for key, shape in self.telemetry_shapes.items():
                    kb.knowledge[n][key] = np.zeros(shape, dtype=np.float32)

        return self._get_obs(), self._get_info()

    def _initialize_agents(self) -> List[Tuple[int, int]]:
        """Initialize agents at unique random positions."""
        positions = set()
        while len(positions) < self.n_agents:
            pos = tuple(np.random.randint(0, self.grid_size, size=2))
            positions.add(pos)
        return list(positions)

    def step(self, action_dict):
        """
        Execute actions for all agents.
        Args:
            action_dict (Dict[str, int]): Actions for each agent.
        Returns:
            Tuple[Dict[str, np.ndarray], Dict[str, float], Dict[str, bool], Dict[str, bool], Dict[str, Any]]
        """
        old_coverage = np.mean(self.grid)
        agent_rewards = {agent_id: 0.0 for agent_id in self.agents}
        collision_penalty = self.reward_weights.get("collision", -0.5)

        # Process each agent's action
        new_positions = self.agent_positions.copy()
        for i, agent_id in enumerate(self.agents):
            action = action_dict[agent_id]  # Get action for current agent
            pos = self.agent_positions[i]
            new_pos = list(pos)

            # 0: Up, 1: Right, 2: Down, 3: Left, 4: Stay
            if action == 0 and pos[0] > 0:
                new_pos[0] -= 1
            elif action == 1 and pos[1] < self.grid_size - 1:
                new_pos[1] += 1
            elif action == 2 and pos[0] < self.grid_size - 1:
                new_pos[0] += 1
            elif action == 3 and pos[1] > 0:
                new_pos[1] -= 1
            # action == 4 is 'Stay'

            new_positions[i] = tuple(new_pos)

        # Check for collisions
        position_counts = {}
        for pos in new_positions:
            position_counts[pos] = position_counts.get(pos, 0) + 1

        # Apply movements and assign rewards
        for i, agent_id in enumerate(self.agents):
            old_pos = self.agent_positions[i]
            new_pos = new_positions[i]

            if position_counts[new_pos] > 1:
                # Collision occurred
                agent_rewards[agent_id] = collision_penalty
                new_positions[i] = old_pos  # Revert to old position
            else:
                # Move agent
                self.agent_positions[i] = new_pos
                # Update coverage
                was_covered = self.grid[new_pos[0], new_pos[1]] == 1.0
                self.grid[new_pos[0], new_pos[1]] = 1.0
                if not was_covered:
                    agent_rewards[agent_id] = self.reward_weights.get("coverage", 1.0)
                else:
                    # Revisiting an already covered cell
                    agent_rewards[agent_id] = self.reward_weights.get("revisit", -0.1)

        # Calculate coverage improvement
        new_coverage = np.mean(self.grid)
        coverage_improv = new_coverage - old_coverage

        # Coverage bonus
        coverage_bonus = 0.0
        if np.all(self.grid == 1.0):
            coverage_bonus = self.reward_weights.get("coverage_bonus", 100.0)

        # Assign coverage improvement and bonus to each agent
        for agent_id in self.agents:
            agent_rewards[agent_id] += coverage_improv * 10.0 + coverage_bonus

        # Target detection and completion
        if self.enable_target_detection:
            detected_targets, completed_targets = self._handle_targets()
            for i, agent_id in enumerate(self.agents):
                if detected_targets[i]:
                    agent_rewards[agent_id] += self.reward_weights.get("target_detection", 2.0)
                if completed_targets[i]:
                    agent_rewards[agent_id] += self.reward_weights.get("target_completion", 5.0)

        self.steps += 1

        # Communication and knowledge sharing
        self._handle_communication()

        # Check termination
        done = False
        if coverage_bonus > 0.0 or self.steps >= self.max_steps:
            done = True
            self.coverage_rates.append(new_coverage * 100.0)
            self.finish_steps.append(self.steps)
            #print(f"Coverage rate: {new_coverage * 100.0}, average coverage rate: {np.mean(self.coverage_rates)}, max coverage rate: {np.max(self.coverage_rates)}")
            coverages_ = ray.get_actor("global_coverages")
            timesteps_ = ray.get_actor("global_timesteps")
            coverages_.inc.remote(new_coverage * 100.0)
            timesteps_.inc.remote(self.steps)
            if self.plot_coverage and PLOTEXT_AVAILABLE and self.max_steps == 20:
                plt.clear_data()
                plt.plot(self.coverage_rates, label="Coverage Rate")
                plt.hline(100, color="red")
                plt.show()
            elif self.plot_coverage and PLOTEXT_AVAILABLE:
                plt.clear_data()
                plt.plot(self.finish_steps, label="Finish Steps")
                plt.hline(20, color="red")
                plt.show()

        # Prepare return values
        obs = self._get_obs()
        dones = {agent_id: done for agent_id in self.agents}
        dones["__all__"] = done
        truncateds = {agent_id: False for agent_id in self.agents}
        truncateds["__all__"] = False
        infos = self._get_info()

        return obs, agent_rewards, dones, truncateds, infos

    def _handle_targets(self) -> Tuple[List[bool], List[bool]]:
        """
        Handle target detection and completion.
        Returns:
            detected_targets (List[bool]): Whether each agent detected a target.
            completed_targets (List[bool]): Whether each agent completed a target mission.
        """
        detected_targets = [False] * self.n_agents
        completed_targets = [False] * self.n_agents

        for agent_id, pos in enumerate(self.agent_positions):
            for t_id, target_pos in enumerate(self.target_positions):
                if pos == target_pos:
                    detected_targets[agent_id] = True
                    # Remove the target to mark it as completed
                    self.target_positions[t_id] = (-1, -1)  # Invalidate position
                    completed_targets[agent_id] = True

        return detected_targets, completed_targets

    def _get_local_view(self, agent_idx: int) -> np.ndarray:
        """Retrieve the local grid view around an agent."""
        r = self.obs_radius
        pos = self.agent_positions[agent_idx]
        padded = np.pad(self.grid, r, mode='constant', constant_values=0)
        cx, cy = pos[0] + r, pos[1] + r
        view = padded[cx - r:cx + r + 1, cy - r:cy + r + 1]
        return view

    def _get_obs(self) -> Dict[str, np.ndarray]:
        """Generate observations for all agents."""
        coverage = np.mean(self.grid)
        obs = {}
        for i in range(self.n_agents):
            local_view = self._get_local_view(i).flatten()
            relative_pos = np.array(self.agent_positions[i]) / self.grid_size
            # Knowledge from telemetry
            other_positions = [
                self.knowledge_bases[i].knowledge[n]["position"]
                for n in range(self.n_agents) if n != i
            ]
            combined = np.concatenate([
                local_view,
                relative_pos,
                [coverage],
                np.array(other_positions).flatten()
            ], axis=0).astype(np.float32)

            # Scale to ~[-1, 1]
            combined = 2.0 * (combined - 0.5)
            obs[f"agent_{i}"] = combined
        return obs

    def _get_info(self) -> Dict[str, Any]:
        """Gather additional information about the current state."""
        coverage = np.mean(self.grid)
        # Create per-agent info dictionaries
        info = {
            agent_id: {
                "coverage": coverage,
                "coverage_rate": coverage * 100.0,
                "steps": self.steps,
                "position": self.agent_positions[i],
            } for i, agent_id in enumerate(self.agents)
        }
        # Add common info fields
        info["__common__"] = {
            "coverage": coverage,
            "coverage_rate": coverage * 100.0,
            "steps": self.steps,
            "agent_positions": self.agent_positions.copy(),
        }
        return info

    def _handle_communication(self):
        """Handle telemetry-based communication between agents."""
        self.telemetry_factory.reset()
        for i in range(self.n_agents):
            data = {}
            if self.telemetry_config.get("position", False):
                data["position"] = tuple(self.agent_positions[i])
            telemetry = self.telemetry_factory.create_telemetry(
                timestamp=self.steps,
                sender_id=i,
                data=data
            )
            # Broadcast to neighbors within Rcomm
            for j in range(self.n_agents):
                if i == j:
                    continue
                dist = np.linalg.norm(np.array(self.agent_positions[i]) - np.array(self.agent_positions[j]))
                if dist <= self.Rcomm:
                    telemetry.receivers.append(j)

        # Distribute telemetry data
        for telemetry in self.telemetry_factory.telemetries:
            for receiver in telemetry.receivers:
                self.knowledge_bases[receiver].add_knowledge(telemetry)

    def render(self):
        """Render the current state as an RGB array."""
        if self.render_mode == "rgb_array":
            return self._render_frame()
        return None

    def _render_frame(self):
        """Generate an RGB frame of the current grid and agent positions."""
        frame = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.uint8)
        frame[self.grid == 1.0] = [200, 200, 200]  # Covered cells

        # Define colors for agents
        colors = [
            [255, 0, 0],      # Red
            [0, 255, 0],      # Green
            [0, 0, 255],      # Blue
            [255, 165, 0],    # Orange
            [128, 0, 128],    # Purple
            [0, 255, 255],    # Cyan
            [255, 192, 203],  # Pink
            [255, 255, 0],    # Yellow
            # Add more colors if needed
        ]

        for i, pos in enumerate(self.agent_positions):
            color = colors[i % len(colors)]
            frame[pos[0], pos[1]] = color

        return frame

    def close(self):
        pass