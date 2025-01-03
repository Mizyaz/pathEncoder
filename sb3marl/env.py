import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional, Any
import plotext as plt

class MultiAgentGridCoverage(gym.Env):
    """
    Multi-agent grid coverage environment with enhanced communication capabilities.
    Agents communicate via Telemetry when within a specified communication radius (Rcomm).
    This communication allows agents to share knowledge, improving coverage efficiency.
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": 4}

    class Telemetry:
        """Represents a telemetry message from one agent to others."""
        def __init__(self, timestamp: int, sender_id: int, data: Dict[str, Any]):
            self.timestamp = timestamp
            self.sender_id = sender_id
            self.data = data  # e.g., {'position': (x, y)}
            self.receivers: List[int] = []

    class TelemetryFactory:
        """Factory to create and manage telemetry messages."""
        def __init__(self, config: Dict[str, Any]):
            self.config = config
            self.telemetries: List[MultiAgentGridCoverage.Telemetry] = []

        def reset(self):
            self.telemetries = []

        def create_telemetry(self, timestamp: int, sender_id: int, data: Dict[str, Any]) -> 'MultiAgentGridCoverage.Telemetry':
            telemetry = MultiAgentGridCoverage.Telemetry(timestamp, sender_id, data)
            self.telemetries.append(telemetry)
            return telemetry

    class Knowledge:
        """Stores the most recent knowledge for an agent."""
        def __init__(self, agent_id: int, num_agents: int):
            self.agent_id = agent_id
            self.num_agents = num_agents
            # Initialize knowledge with None for other agents
            self.knowledge: Dict[int, Dict[str, Any]] = {i: {} for i in range(num_agents)}
            self.knowledge[agent_id] = {}  # Own knowledge can be handled separately if needed

        def add_knowledge(self, telemetry: 'MultiAgentGridCoverage.Telemetry'):
            sender = telemetry.sender_id
            if sender != self.agent_id:
                # Merge telemetry data into knowledge
                for key, value in telemetry.data.items():
                    self.knowledge[sender][key] = value

    def __init__(
        self,
        grid_size: int = 20,
        num_agents: int = 3,
        observation_radius: int = 2,
        max_steps: int = 500,
        render_mode: Optional[str] = None,
        plot_coverage: bool = False,
        Rcomm: float = 5.0,  # Communication radius
        telemetry_config: Optional[Dict[str, Any]] = None  # Config for Telemetry data
    ):
        super().__init__()

        self.grid_size = grid_size
        self.num_agents = num_agents
        self.obs_radius = observation_radius
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.plot_coverage = plot_coverage
        self.Rcomm = Rcomm
        self.telemetry_config = telemetry_config if telemetry_config is not None else {'position': True}

        # Calculate total cells for coverage calculation
        self.total_cells = grid_size * grid_size

        # Placeholder for observation_space; will be defined after first reset
        self.observation_space = spaces.Dict({
            f"agent_{i}": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32) for i in range(num_agents)
        })

        # Action space: Discrete movement in 4 directions for each agent
        self.action_space = spaces.MultiDiscrete([4] * num_agents)

        # Initialize communication components
        self.telemetry_factory = self.TelemetryFactory(self.telemetry_config)
        self.knowledge_bases: List['MultiAgentGridCoverage.Knowledge'] = [
            self.Knowledge(agent_id=i, num_agents=num_agents) for i in range(num_agents)
        ]

        # Initialize Telemetry history
        self.reset_telemetries()

        # Initialize coverage tracking
        self.coverage_rates: List[float] = []
        self.finish_steps: List[int] = []

        # Perform initial reset to set up observation_space
        initial_obs, initial_info = self.reset()

        # Define observation_space based on initial observation
        obs_space_dict = {}
        for agent_key, obs in initial_obs.items():
            obs_space_dict[agent_key] = spaces.Box(
                low=0,
                high=1,
                shape=(obs.shape[0],),
                dtype=np.float32
            )
        self.observation_space = spaces.Dict(obs_space_dict)

    def reset_telemetries(self):
        """Reset telemetry factory and telemetries."""
        self.telemetry_factory.reset()

    def reset(self, seed=None, options=None) -> Tuple[Dict, Dict]:
        super().reset(seed=seed)

        # Initialize grid (0: uncovered, 1: covered)
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)

        # Place all agents at (0,0)
        self.agent_positions = np.zeros((self.num_agents, 2), dtype=int)

        # Mark initial position as covered
        self.grid[0, 0] = 1

        self.steps = 0

        # Reset communication components
        self.reset_telemetries()
        for kb in self.knowledge_bases:
            kb.knowledge = {i: {} for i in range(self.num_agents)}
            kb.knowledge[kb.agent_id] = {}

        coverage = np.mean(self.grid)

        return self._get_obs(), self._get_info()

    def _get_local_view(self, agent_idx: int) -> np.ndarray:
        """Get local grid view centered on agent."""
        pos = self.agent_positions[agent_idx]
        r = self.obs_radius
        padded_grid = np.pad(self.grid, r, mode='constant')
        center_x, center_y = pos + r

        return padded_grid[
            center_x - r:center_x + r + 1,
            center_y - r:center_y + r + 1
        ]

    def _get_obs(self) -> Dict:
        """Get observations for all agents, including knowledge from Telemetry."""
        obs = {}
        coverage = np.mean(self.grid)

        for i in range(self.num_agents):
            local_view = self._get_local_view(i).flatten()
            relative_pos = self.agent_positions[i] / self.grid_size

            # Integrate knowledge into the observation if needed
            # Currently, keeping the original observation structure
            # Further integration can be done based on specific requirements

            # Concatenate local view, relative position, and global coverage
            agent_obs = np.concatenate([
                local_view,
                relative_pos,
                [coverage]
            ]).astype(np.float32)

            obs[f"agent_{i}"] = agent_obs

        return obs

    def _get_info(self) -> Dict:
        """Get environment info."""
        coverage = np.mean(self.grid)
        return {
            "coverage": coverage,
            "steps": self.steps,
            "coverage_rate": coverage * 100  # As percentage
        }

    def step(self, actions: np.ndarray) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        Process actions for all agents.
        Actions: 0: up, 1: right, 2: down, 3: left
        Returns scalar reward (mean of all agents) for compatibility with Monitor wrapper
        Episode terminates when full coverage is achieved or max steps is reached
        """
        agent_rewards = {}
        old_coverage = np.mean(self.grid)

        # Process each agent's action
        for i in range(self.num_agents):
            action = actions[i]
            pos = self.agent_positions[i]
            new_pos = pos.copy()

            # Update position based on action
            if action == 0:  # Up
                new_pos[0] = max(0, pos[0] - 1)
            elif action == 1:  # Right
                new_pos[1] = min(self.grid_size - 1, pos[1] + 1)
            elif action == 2:  # Down
                new_pos[0] = min(self.grid_size - 1, pos[0] + 1)
            elif action == 3:  # Left
                new_pos[1] = max(0, pos[1] - 1)

            # Check for collisions with other agents
            collision = False
            for j, other_pos in enumerate(self.agent_positions):
                if j != i and np.array_equal(new_pos, other_pos):
                    collision = True
                    break

            if not collision:
                self.agent_positions[i] = new_pos
                # Mark new position as covered
                was_covered = self.grid[new_pos[0], new_pos[1]] == 1
                self.grid[new_pos[0], new_pos[1]] = 1

                # Calculate reward based on new coverage
                agent_rewards[f"agent_{i}"] = 1.0 if not was_covered else -1.0
            else:
                agent_rewards[f"agent_{i}"] = -0.5  # Penalty for collision

        self.steps += 1

        # Calculate new coverage and additional reward based on global improvement
        new_coverage = np.mean(self.grid)
        coverage_improvement = new_coverage - old_coverage

        # Bonus reward for achieving full coverage
        is_full_coverage = np.all(self.grid == 1)
        coverage_bonus = 100.0 if is_full_coverage else 0.0

        for i in range(self.num_agents):
            agent_rewards[f"agent_{i}"] += coverage_improvement * 10.0 + coverage_bonus

        # Handle communication between agents
        self._handle_communication()

        # Calculate mean reward across all agents for the monitor
        mean_reward = np.mean([reward for reward in agent_rewards.values()])

        # Episode terminates if full coverage is achieved
        terminated = {
            f"agent_{i}": is_full_coverage
            for i in range(self.num_agents)
        }

        truncated = {
            f"agent_{i}": self.steps >= self.max_steps
            for i in range(self.num_agents)
        }

        # Store individual agent rewards in info dict
        info = self._get_info()
        info["agent_rewards"] = agent_rewards
        info["mean_reward"] = mean_reward

        # Print coverage rate when episode ends
        if any(terminated.values()) or any(truncated.values()):
            end_type = "terminated" if any(terminated.values()) else "truncated"
            self.coverage_rates.append(info['coverage_rate'])
            self.finish_steps.append(self.steps)
            #print(f"Episode {end_type} after {self.steps} steps with {info['coverage_rate']:.2f}% coverage")
            if self.plot_coverage:
                plt.clear_data()
                if self.max_steps == 20:
                    plt.hline(100, color="red")
                    plt.plot(self.coverage_rates)
                else:
                    plt.hline(20, color="red")
                    plt.plot(self.finish_steps)
                plt.show()

        return (
            self._get_obs(),
            mean_reward,  # Return scalar reward for monitor
            any(terminated.values()),  # Episode ends if any agent terminates
            any(truncated.values()),   # Episode truncates if any agent truncates
            info
        )

    def _handle_communication(self):
        """Handle telemetry sharing between agents within communication range."""
        # Reset telemetries for the current step
        self.reset_telemetries()

        # Generate Telemetry for each agent
        for i in range(self.num_agents):
            data = {}
            if self.telemetry_config.get('position', False):
                data['position'] = tuple(self.agent_positions[i])
            telemetry = self.telemetry_factory.create_telemetry(
                timestamp=self.steps,
                sender_id=i,
                data=data
            )

            # Check for agents within Rcomm to receive telemetry
            for j in range(self.num_agents):
                if i == j:
                    continue
                distance = np.linalg.norm(self.agent_positions[i] - self.agent_positions[j])
                if distance <= self.Rcomm:
                    telemetry.receivers.append(j)

        # Distribute Telemetries to receivers
        for telemetry in self.telemetry_factory.telemetries:
            for receiver_id in telemetry.receivers:
                self.knowledge_bases[receiver_id].add_knowledge(telemetry)

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        """Simple numpy-based rendering"""
        frame = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.uint8)

        # Draw covered cells
        frame[self.grid == 1] = [200, 200, 200]  # Gray for covered cells

        # Draw agents
        colors = [
            [255, 0, 0],    # Red
            [0, 255, 0],    # Green
            [0, 0, 255],    # Blue
            [255, 165, 0],  # Orange
            [128, 0, 128]   # Purple
        ]

        for i, pos in enumerate(self.agent_positions):
            color = colors[i % len(colors)]
            frame[pos[0], pos[1]] = color

        return frame

    def close(self):
        pass
