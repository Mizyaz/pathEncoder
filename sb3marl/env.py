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
        telemetry_config: Optional[Dict] = None,
        telemetry_shapes: Optional[Dict] = None
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
        self.telemetry_shapes = telemetry_shapes if telemetry_shapes else {'position': (2,)}

        self.total_cells = grid_size * grid_size

        # We'll define the observation space after the first reset,
        # because we rely on the actual shape at runtime.
        # For now, just a placeholder single-dimension.
        self.observation_space = spaces.Dict({
            f"agent_{i}": spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
            for i in range(num_agents)
        })

        # Action space: each agent can move in 4 directions.
        self.action_space = spaces.Dict({
            f"agent_{i}": spaces.Discrete(4)
            for i in range(num_agents)
        })

        # Communication helper stubs
        self.telemetry_factory = self.TelemetryFactory(self.telemetry_config)
        self.knowledge_bases = [
            self.Knowledge(i, num_agents, self.telemetry_shapes) for i in range(num_agents)
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
        def __init__(self, agent_id: int, num_agents: int, telemetry_shapes: Dict):
            self.agent_id = agent_id
            self.num_agents = num_agents
            self.knowledge: Dict[int, Dict] = {i: {} for i in range(num_agents)}

            for n in range(num_agents):
                for key, shape in telemetry_shapes.items():
                    self.knowledge[n][key] = np.zeros(shape)

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
            for n in range(self.num_agents):
                for key, shape in self.telemetry_shapes.items():
                    kb.knowledge[n][key] = np.zeros(shape)

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
            other_pos = [self.knowledge_bases[i].knowledge[n]['position'] for n in range(self.num_agents) if n != i]

            # Flatten everything into one array
            # [local_view, rel_x, rel_y, coverage]
            combined = np.concatenate([
                local_view,
                relative_pos,
                [coverage],
                np.array(other_pos).flatten()
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

    def step(self, actions: Dict[str, int]):
        old_coverage = np.mean(self.grid)
        agent_rewards = {}

        # Process actions
        for i in range(self.num_agents):
            action = actions[f"agent_{i}"]
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
