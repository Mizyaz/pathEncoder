import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Dict, Tuple, List, Optional, Any
import networkx as nx

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
            "target_priorities": [1.0, 0.7, 0.3],  # High to low priority
            "inform_times": 3,  # Number of visits needed for target completion
            "time_slots": {},  # Dict mapping target_id to list of timesteps
            "obs_config": {
                "known_target_positions": False,  # If True, agents know target positions from start
                "known_target_priorities": False,  # If True, agents know target priorities from start
                "known_time_slots": False,  # If True, agents know when targets will appear
            },
            "reward_weights": {
                "coverage": 1.0,
                "revisit": -0.1,
                "connectivity": 0.5,
                "target_detection": 2.0,
                "target_relay": 1.5,
                "target_completion": 5.0,
                "early_completion": 3.0  # Bonus for completing high priority targets early
            }
        }
        
        if config:
            self.config.update(config)
            
        # Extract and validate configuration
        self.n_agents = int(self.config["n_agents"])
        self.G1, self.G2 = map(int, self.config["grid_size"])
        self.max_steps = int(self.config["max_steps"])
        self.n_targets = int(self.config["n_targets"])
        self.inform_times = int(self.config["inform_times"])
        
        # Initialize target priorities
        if isinstance(self.config["target_priorities"], (list, tuple)):
            self.target_priorities = np.array(self.config["target_priorities"][:self.n_targets])
        else:
            self.target_priorities = np.ones(self.n_targets, dtype=np.float32)
            
        # Initialize time slots
        self.time_slots = self.config["time_slots"]
        
        # Initialize observation flags
        self.obs_config = self.config["obs_config"]
        
        # Initialize state variables with explicit types
        self.drone_positions: np.ndarray = np.zeros((self.n_agents, 2), dtype=np.float32)
        self.target_positions: np.ndarray = np.zeros((self.n_targets, 2), dtype=np.float32)
        self.visited_cells: np.ndarray = np.zeros((self.G1, self.G2), dtype=np.int32)
        self.detected_targets: np.ndarray = np.zeros(self.n_targets, dtype=np.int32)
        self.target_completion: np.ndarray = np.zeros(self.n_targets, dtype=np.int32)
        self.target_visits: np.ndarray = np.zeros(self.n_targets, dtype=np.int32)
        self.target_connected_visits: np.ndarray = np.zeros(self.n_targets, dtype=np.int32)
        self.target_knowledge: np.ndarray = np.zeros(self.n_targets, dtype=np.int32)  # Tracks if target info is known
        self.paths: np.ndarray = np.zeros((self.n_agents, self.max_steps, 2), dtype=np.float32)
        self.adjacency_matrix: np.ndarray = np.zeros((self.n_agents + 1, self.n_agents + 1), dtype=np.float32)

        self.coverage_percentages = []
        self.count = 0
        
        # Action and observation spaces
        self.action_space = spaces.MultiDiscrete([5] * self.n_agents)
        
        # Observation space with explicit shapes and types
        obs_spaces = {
            "visited_cells": spaces.Box(0, np.inf, shape=(self.G1, self.G2), dtype=np.float32),
            "agent_positions": spaces.Box(0, max(self.G1, self.G2), shape=(self.n_agents, 2), dtype=np.float32),
            "detected_targets": spaces.Box(0, 1, shape=(self.n_targets,), dtype=np.float32),
            "target_completion": spaces.Box(0, 1, shape=(self.n_targets,), dtype=np.float32),
            "target_priorities": spaces.Box(0, 1, shape=(self.n_targets,), dtype=np.float32),
            "target_knowledge": spaces.Box(0, 1, shape=(self.n_targets,), dtype=np.float32),
            "target_availability": spaces.Box(0, 1, shape=(self.n_targets,), dtype=np.float32),
            "target_positions": spaces.Box(0, max(self.G1, self.G2), shape=(self.n_targets, 2), dtype=np.float32),
        }

        self.observation_space = spaces.Dict(obs_spaces)
        
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
        
        # Initialize drone positions
        if options and "initial_positions" in options:
            self.drone_positions = np.array(options["initial_positions"], dtype=np.float32)
        else:
            self.drone_positions = np.array([[0, 0], [3, 0], [3, 0], [7, 0], [0, 7], [5, 0], [0, 5], [2, 0], [0, 2], [0, 4], [4, 0], [1, 0], [0, 1]])[:self.n_agents]
        
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
        self.target_knowledge = np.zeros(self.n_targets, dtype=np.int32)
        self.paths = np.zeros((self.n_agents, self.max_steps, 2), dtype=np.float32)
        self.adjacency_matrix = np.zeros((self.n_agents + 1, self.n_agents + 1), dtype=np.float32)
        
        # Mark initial positions
        for pos in self.drone_positions:
            self.visited_cells[int(pos[0]), int(pos[1])] += 1
            
        return self._get_obs(), self._get_info()

    def _get_obs(self) -> Dict[str, np.ndarray]:
        """Construct observation dictionary with consistent types"""
        obs = {
            "visited_cells": self.visited_cells.astype(np.float32),
            "agent_positions": self.drone_positions,
            "detected_targets": self.detected_targets.astype(np.float32),
            "target_completion": self.target_completion.astype(np.float32),
            "target_knowledge": self.target_knowledge.astype(np.float32),
            "target_availability": np.zeros(self.n_targets, dtype=np.float32),
            "target_priorities": np.zeros(self.n_targets, dtype=np.float32),
            "target_positions": np.zeros((self.n_targets, 2), dtype=np.float32),
        }
        
        # Add target positions if known
        if self.obs_config["known_target_positions"] or any(self.target_knowledge):
            for i in range(self.n_targets):
                if self.target_knowledge[i] == 1:
                    obs["target_positions"][i] = self.target_positions[i]
                else:
                    obs["target_positions"][i] = np.zeros(2, dtype=np.float32)
        else:
            obs["target_positions"] = np.zeros((self.n_targets, 2), dtype=np.float32)
            
        # Add target priorities if known
        if self.obs_config["known_target_priorities"] or any(self.target_knowledge):
            for i in range(self.n_targets):
                if self.target_knowledge[i] == 1:
                    obs["target_priorities"][i] = self.target_priorities[i]
                else:
                    obs["target_priorities"][i] = 0.0
        else:
            obs["target_priorities"] = np.zeros(self.n_targets, dtype=np.float32)
            
        # Add target availability if known
        if self.obs_config["known_time_slots"] or any(self.target_knowledge):
            availability = np.ones(self.n_targets, dtype=np.float32)
            for target_id, slots in self.time_slots.items():
                if target_id < self.n_targets and self.target_knowledge[target_id] == 1:
                    availability[target_id] = 1 if self.timestep in slots else 0
                else:
                    availability[target_id] = 0
            obs["target_availability"] = availability
        else:
            obs["target_availability"] = np.zeros(self.n_targets, dtype=np.float32)
            

        # add target knowledge
        obs["target_knowledge"] = self.target_knowledge

        return obs

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

    def _compute_connectivity(self) -> float:
        """Compute connectivity metric with proper matrix conversion"""
        if not self.config["enable_connectivity"]:
            return 0.0
            
        G = nx.Graph()
        gcs_idx = self.n_agents
        G.add_node(gcs_idx, pos=np.array(self.config["gcs_pos"], dtype=np.float32))
        
        # Add UAV nodes and edges
        for i in range(self.n_agents):
            G.add_node(i, pos=self.drone_positions[i])
            if np.linalg.norm(self.drone_positions[i] - np.array(self.config["gcs_pos"], dtype=np.float32)) <= self.config["comm_dist"]:
                G.add_edge(i, gcs_idx)
        
        # Add UAV-UAV edges
        for i in range(self.n_agents):
            for j in range(i+1, self.n_agents):
                if np.linalg.norm(self.drone_positions[i] - self.drone_positions[j]) <= self.config["comm_dist"]:
                    G.add_edge(i, j)
        
        # Handle adjacency matrix conversion explicitly
        adj_matrix = nx.adjacency_matrix(G)
        self.adjacency_matrix = adj_matrix.toarray().astype(np.float32)
        
        # Count connected UAVs
        connected_count = 0
        for i in range(self.n_agents):
            try:
                if nx.has_path(G, i, gcs_idx):
                    connected_count += 1
            except nx.NetworkXNoPath:
                continue
                
        return connected_count / self.n_agents

    def step(self, actions: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """Execute actions with proper type handling and state updates"""
        # Track previous states
        previous_coverage = np.sum(self.visited_cells > 0)
        
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
            
        # Update target states and compute rewards
        target_reward = self._update_target_states()
        
        # Compute base rewards
        weights = self.config["reward_weights"]
        coverage_reward = weights["coverage"] * (np.sum(self.visited_cells > 0) - previous_coverage)
        revisit_penalty = weights["revisit"] * np.sum(self.visited_cells > 1)
        
        # Compute connectivity reward
        connectivity = self._compute_connectivity()
        connectivity_reward = weights["connectivity"] * connectivity if self.config["enable_connectivity"] else 0
        
        # Total reward
        reward = coverage_reward + revisit_penalty + connectivity_reward + target_reward
        
        self.timestep += 1
        
        # Check termination
        if self.n_targets > 0:
            done = np.all(self.target_completion == 1)
        else:
            coverage_percentage = np.sum(self.visited_cells > 0) / (self.G1 * self.G2)
            done = coverage_percentage >= 0.99  # Allow for small numerical errors

        truncated = self.timestep >= self.max_steps

        if done or truncated:
            self.coverage_percentages.append(np.sum(self.visited_cells > 0) / (self.G1 * self.G2))
            print(max(self.coverage_percentages), "coverage percentage", self.coverage_percentages[-1], np.round(np.mean(self.coverage_percentages[-10:]), 2)) if self.count % 10 == 0 else None
            self.count += 1
            
        
        return self._get_obs(), float(reward), bool(done), bool(truncated), self._get_info()

    def _update_target_states(self) -> float:
        """Update target states and compute related rewards"""
        reward = 0.0
        weights = self.config["reward_weights"]
        
        for i, target_pos in enumerate(self.target_positions):
            # Check if target is available at current timestep
            is_available = True
            if i in self.time_slots:
                is_available = self.timestep in self.time_slots[i]
            
            if not is_available:
                continue
                
            target_detected = False
            target_connected = False
            
            # Check target detection
            for drone_pos in self.drone_positions:
                if np.array_equal(drone_pos.astype(int), target_pos.astype(int)):
                    target_detected = True
                    self.target_knowledge[i] = 1  # Target info becomes known once detected
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
                    
                    # Check completion based on inform_times
                    if self.target_connected_visits[i] >= self.inform_times:
                        if not self.target_completion[i]:
                            self.target_completion[i] = 1
                            completion_reward = weights["target_completion"]
                            
                            # Add early completion bonus based on priority and timestep
                            early_factor = 1.0 - (self.timestep / self.max_steps)
                            priority_bonus = self.target_priorities[i] * weights["early_completion"] * early_factor
                            reward += completion_reward + priority_bonus
                    
        return reward