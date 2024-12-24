import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Dict, Tuple, List, Optional, Any
import networkx as nx
from .RewardManager import RewardManager
from .ObservationManager import ObservationManager
from .Utils import DEFAULT_INITIAL_POSITIONS
class RobustMultiUAVTargetEnv(gym.Env):
    """
    Enhanced Multi-UAV environment with robust state handling and consistent typing
    for target search, coverage, and information relay missions.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        # Default configuration with type annotations
        self.config: Dict[str, Any] = {
            "n_agents": 8,
            "grid_size": (10, 10),
            "max_steps": 100,
            "comm_dist": 2.8284,
            "gcs_pos": (0, 0),
            "enable_connectivity": True,
            "n_targets": 3,
            "target_types": [2, 2, 2],
            "reward_weights": {
                "coverage": 1.0,
                "revisit": -0.1,
                "connectivity": 0.5,
                "target_detection": 2.0,
                "target_relay": 1.5,
                "target_completion": 5.0
            },
            "selected_obs": ["visited_cells", "agent_positions", "detected_targets"]
        }
        
        if config:
            self.config.update(config)
            
        # Extract and validate configuration
        self.n_agents = int(self.config["n_agents"])
        self.G1, self.G2 = map(int, self.config["grid_size"])
        self.max_steps = int(self.config["max_steps"])
        self.n_targets = int(self.config["n_targets"])
        
        if isinstance(self.config["target_types"], (list, tuple)):
            self.target_types = np.array(self.config["target_types"][:self.n_targets])
        else:
            self.target_types = np.ones(self.n_targets, dtype=np.int32) * 2
        # Action and observation spaces
        self.action_space = spaces.MultiDiscrete([5] * self.n_agents)
        
        # Observation space with explicit shapes and types
        self.observation_manager = ObservationManager(self)
        self.observation_space = spaces.Dict(self.observation_manager.observation_spaces)
        
        # Movement vectors with explicit typing
        self.action_to_move = {
            0: np.array([0, 0], dtype=np.float32),   # hover
            1: np.array([0, 1], dtype=np.float32),   # up
            2: np.array([0, -1], dtype=np.float32),  # down
            3: np.array([-1, 0], dtype=np.float32),  # left
            4: np.array([1, 0], dtype=np.float32)    # right
        }
        
        self.timestep = 0

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Initialize environment state with explicit typing"""
        super().reset(seed=seed)
        self.timestep = 0

        self.reward_manager = RewardManager(self)
        
        # Initialize drone positions
        if options and "initial_positions" in options:
            self.drone_positions = np.array(options["initial_positions"], dtype=np.float32)
        else:
            self.drone_positions = DEFAULT_INITIAL_POSITIONS[:self.n_agents]
        
        # Initialize target positions and states
        if options and "target_positions" in options:
            self.target_positions = np.array(options["target_positions"], dtype=np.float32)
        else:
            self.target_positions = np.array([
                [self.np_random.integers(0, self.G1), self.np_random.integers(0, self.G2)]
                for _ in range(self.n_targets)
            ], dtype=np.float32)
            
        # Reset state arrays
        self.visited_cells = np.zeros((self.G1, self.G2), dtype=np.int32)
        self.detected_targets = np.zeros(self.n_targets, dtype=np.int32)
        self.target_completion = np.zeros(self.n_targets, dtype=np.int32)
        self.target_visits = np.zeros(self.n_targets, dtype=np.int32)
        self.target_connected_visits = np.zeros(self.n_targets, dtype=np.int32)
        self.target_consecutive_visits = np.zeros(self.n_targets, dtype=np.int32)
        self.paths = np.zeros((self.n_agents, self.max_steps, 2), dtype=np.float32)
        self.adjacency_matrix = np.zeros((self.n_agents + 1, self.n_agents + 1), dtype=np.float32)
        self.previous_coverage = np.sum(self.visited_cells > 0)

        # Mark initial positions
        for pos in self.drone_positions:
            self.visited_cells[int(pos[0]), int(pos[1])] += 1
            
        return self._get_obs(), self._get_info()

    def _get_obs(self) -> Dict[str, np.ndarray]:
        """Construct observation dictionary with consistent types"""
        return self.observation_manager.get_observation(self)

    def _get_info(self) -> Dict[str, Any]:
        """Construct info dictionary with consistent types"""
        return {
            "timestep": self.timestep,
            "n_targets": self.n_targets,
            "detected_targets": self.detected_targets.copy(),
            "completed_targets": self.target_completion.copy(),
            "target_visits": self.target_visits.copy(),
            "target_connected_visits": self.target_connected_visits.copy(),
            "connectivity": self._compute_connectivity(),
            "coverage_percentage": np.sum(self.visited_cells > 0) / (self.G1 * self.G2),
            "unique_cells_covered": np.sum(self.visited_cells > 0),
            "total_revisits": np.sum(self.visited_cells > 1),
            "paths": self.paths.copy()
        }

    def compute_connectivity(env) -> float:
        """Compute connectivity metric with proper matrix conversion"""
            
        G = nx.Graph()
        gcs_idx = env.n_agents
        G.add_node(gcs_idx, pos=np.array(env.config["gcs_pos"], dtype=np.float32))
        
        # Add UAV nodes and edges
        for i in range(env.n_agents):
            G.add_node(i, pos=env.drone_positions[i])
            if np.linalg.norm(env.drone_positions[i] - np.array(env.config["gcs_pos"], dtype=np.float32)) <= env.config["comm_dist"]:
                G.add_edge(i, gcs_idx)
        
        # Add UAV-UAV edges
        for i in range(env.n_agents):
            for j in range(i+1, env.n_agents):
                if np.linalg.norm(env.drone_positions[i] - env.drone_positions[j]) <= env.config["comm_dist"]:
                    G.add_edge(i, j)
        
        # Handle adjacency matrix conversion explicitly
        adj_matrix = nx.adjacency_matrix(G)
        env.adjacency_matrix = adj_matrix.toarray().astype(np.float32)
        
        # Count connected UAVs
        connected_count = 0
        for i in range(env.n_agents):
            try:
                if nx.has_path(G, i, gcs_idx):
                    connected_count += 1
            except nx.NetworkXNoPath:
                continue
                
        return connected_count / env.n_agents

    def step(self, actions: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """Execute actions with proper type handling and state updates"""
        # Track previous states
        
        # Process movements with explicit typing
        for agent_idx, action in enumerate(actions):
            move = self.action_to_move[action]
            new_pos = self.drone_positions[agent_idx] + move
            
            # Record path
            self.paths[agent_idx, self.timestep] = new_pos
            
            # Boundary check
            if (0 <= new_pos[0] < self.G1 and 0 <= new_pos[1] < self.G2):
                self.drone_positions[agent_idx] = new_pos
                
        # Update visited cells
        for pos in self.drone_positions:
            self.visited_cells[int(pos[0]), int(pos[1])] += 1
                    
        # Total reward
        reward = self.reward_manager.compute_reward()
        
        self.timestep += 1
        
        done, truncated = self.get_done_and_truncated()

        self.previous_coverage = np.sum(self.visited_cells > 0)
        self.previous_target_completion = self.target_completion.copy()
        self.previous_detected_targets = self.detected_targets.copy()
        self.previous_connectivity = self._compute_connectivity()
        self.previous_coverage_percentage = np.sum(self.visited_cells > 0) / (self.G1 * self.G2)
        self.previous_unique_cells_covered = np.sum(self.visited_cells > 0)
        self.previous_total_revisits = np.sum(self.visited_cells > 1)
        self.previous_actions = actions.copy()

        return self._get_obs(), float(reward), bool(done), bool(truncated), self._get_info()

    def get_done_and_truncated(self):
        if self.n_targets > 0:
            done = np.all(self.target_completion == 1)
        else:
            coverage_percentage = np.sum(self.visited_cells > 0) / (self.G1 * self.G2)
            done = coverage_percentage >= 0.99  # Allow for small numerical errors
            
        truncated = self.timestep >= self.max_steps
        return done, truncated

    def update_target_states(self) -> float:
        """Update target states and compute related rewards"""
        reward = 0.0
        weights = self.config["reward_weights"]
        
        for i, target_pos in enumerate(self.target_positions):
            target_detected = False
            target_connected = False
            
            # Check target detection
            for drone_pos in self.drone_positions:
                if np.array_equal(drone_pos.astype(int), target_pos.astype(int)):
                    target_detected = True
                    break
            
            if target_detected:
                # New detection reward
                if not self.detected_targets[i]:
                    self.detected_targets[i] = 1
                    reward += weights["target_detection"]
                    
                # Check connectivity for information relay
                if self._compute_connectivity() > 0:
                    target_connected = True
                    self.target_connected_visits[i] += 1
                    reward += weights["target_relay"]
                    
                    # Handle different target types
                    if self.target_types[i] == 1:  # Consecutive
                        self.target_consecutive_visits[i] += 1
                        if self.target_consecutive_visits[i] >= 3:
                            if not self.target_completion[i]:
                                self.target_completion[i] = 1
                                reward += weights["target_completion"]
                    elif self.target_types[i] == 2:  # Multiple updates
                        if self.target_connected_visits[i] >= 3:
                            if not self.target_completion[i]:
                                self.target_completion[i] = 1
                                reward += weights["target_completion"]
                    else:  # One-time
                        if not self.target_completion[i]:
                            self.target_completion[i] = 1
                            reward += weights["target_completion"]
            
            # Reset consecutive counter if not connected
            if not target_connected and self.target_types[i] == 1:
                self.target_consecutive_visits[i] = 0
                
        return reward