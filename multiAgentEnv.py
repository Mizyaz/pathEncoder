import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Dict, Tuple, List, Optional
import networkx as nx

class MultiUAVCoverageEnv(gym.Env):
    """
    Specialized environment for optimizing grid coverage with multiple UAVs.
    Implements optional connectivity constraints and modular reward mechanisms.
    """
    
    def __init__(self, config: Dict = None):
        super().__init__()
        
        # Configuration parameters
        self.config = {
            "n_agents": 8,
            "grid_size": (10, 10),
            "max_steps": 1000,
            "comm_dist": 2.8284,  # 2√2 for connectivity if enabled
            "gcs_pos": (0, 0),
            "enable_connectivity": False,  # Optional connectivity constraints
            "reward_weights": {
                "coverage": 1.0,         # Weight for new cell coverage
                "revisit": -0.1,         # Penalty for revisiting cells
                "connectivity": 0.5      # Optional connectivity reward weight
            },
            "image_observation": False
        }
        
        if config:
            self.config.update(config)
            
        self.n_agents = self.config["n_agents"]
        self.G1, self.G2 = self.config["grid_size"]
        self.max_steps = self.config["max_steps"]
        
        # State variables
        self.drone_positions = None  # (n_agents, 2)
        self.visited_cells = None    # (G1, G2)
        self.timestep = 0
        

        if self.config["image_observation"]:
            grid_status = spaces.Box(0, np.inf, shape=(1, self.G1, self.G2), dtype=np.uint8)
        else:
            grid_status = spaces.Box(0, np.inf, shape=(self.G1, self.G2), dtype=np.uint8)

        obs_keys = {
            "visited_cells": grid_status,
            "revisited_cells": grid_status,
            "agent_positions": spaces.Box(0, max(self.G1, self.G2), shape=(self.n_agents, 2)),
            "prev_agent_positions": spaces.Box(0, max(self.G1, self.G2), shape=(self.n_agents, 2)),
            "local_empty_cells": spaces.Box(0, np.inf, shape=(self.n_agents, 3, 3), dtype=np.float32),
            "closest_unvisited_cells": spaces.Box(0, max(self.G1, self.G2), shape=(self.n_agents, 2), dtype=np.float32),
            "prev_actions": spaces.Box(0, 4, shape=(self.n_agents, 1), dtype=np.float32),
            "paths": spaces.Box(0, 1, shape=(self.n_agents, self.max_steps, 2), dtype=np.float32),
            "action_history": spaces.Box(0, 4, shape=(self.n_agents, self.max_steps), dtype=np.float32),
            "adjacency_matrix": spaces.Box(0, 1, shape=(self.n_agents+1, self.n_agents+1), dtype=np.float32)
        }

        self.used_obs_keys = ["visited_cells", 
                         "revisited_cells", 
                         "agent_positions", 
                         "prev_agent_positions", 
                         #"local_empty_cells", 
                         #"closest_unvisited_cells", 
                         "prev_actions",
                         #"paths",
                         #"action_history",
                         #"adjacency_matrix"
                         ]

        # Observation space: grid status and agent positions
        self.observation_space = spaces.Dict({key: obs_keys[key] for key in self.used_obs_keys})
        
        # Action space: movement directions for each agent
        self.action_space = spaces.MultiDiscrete([5] * self.n_agents)  # hover, up, down, left, right
        
        # Movement vectors
        self.action_to_move = {
            0: (0, 0),   # hover
            1: (0, 1),   # up
            2: (0, -1),  # down
            3: (-1, 0),  # left
            4: (1, 0)    # right
        }

    def _get_local_empty_cells(self) -> np.ndarray:
        """
        Extract 3x3 windows around each drone showing visitation patterns.
        
        Returns:
            np.ndarray: Array of shape (n_agents, 3, 3) containing local visitation patterns
                       -1 indicates out-of-bounds cells
        """
        local_windows = np.zeros((self.n_agents, 3, 3))
        
        for i in range(self.n_agents):
            x, y = self.drone_positions[i].astype(int)
            
            # Initialize window with -1 (out-of-bounds indicator)
            window = np.full((3, 3), -1.0)
            
            # Calculate valid ranges for the local window
            x_start, x_end = max(0, x-1), min(self.G1, x+2)
            y_start, y_end = max(0, y-1), min(self.G2, y+2)
            
            # Calculate corresponding indices in the 3x3 window
            w_x_start = 1 - (x - x_start)  # If x=0, this will be 1
            w_x_end = 1 + (x_end - x)      # If x=G1-1, this will be 2
            w_y_start = 1 - (y - y_start)
            w_y_end = 1 + (y_end - y)
            
            # Fill in the valid part of the window with actual visitation values
            window[w_x_start:w_x_end, w_y_start:w_y_end] = \
                self.visited_cells[x_start:x_end, y_start:y_end]
            
            local_windows[i] = window
            
        return local_windows > 0

    def _get_closest_unvisited(self) -> np.ndarray:
        """
        Compute closest unvisited cell coordinates for each drone.
        
        Returns:
            np.ndarray: Array of shape (n_agents, 2) containing closest unvisited x,y coordinates
                       Returns [-1,-1] for drones when all cells are visited
        """
        # Initialize output array
        closest_coords = np.zeros((self.n_agents, 2), dtype=np.float32)
        
        # If all cells are visited, return -1 coordinates
        if np.all(self.visited_cells > 0):
            return np.full((self.n_agents, 2), -1)
        
        # Get unvisited cell coordinates
        unvisited_y, unvisited_x = np.where(self.visited_cells == 0)
        unvisited_coords = np.column_stack((unvisited_x, unvisited_y))
        
        # Compute for each drone
        for i in range(self.n_agents):
            drone_pos = self.drone_positions[i]
            
            # Compute distances to all unvisited cells
            distances = np.linalg.norm(unvisited_coords - drone_pos, axis=1)
            
            # Find closest unvisited cell
            closest_idx = np.argmin(distances)
            closest_coords[i] = unvisited_coords[closest_idx]
        
        return closest_coords

    def reset(self, seed: Optional[int] = None, options: Dict = None) -> Tuple:
        """Initialize environment state with optional configuration"""
        super().reset(seed=seed)
        self.timestep = 0
        
        # Initialize drone positions
        self.drone_positions = np.zeros((self.n_agents, 2))
        self.prev_agent_positions = np.zeros((self.n_agents, 2))
        self.paths = np.zeros((self.n_agents, self.max_steps, 2))
        self.prev_actions = np.zeros((self.n_agents, 1))
        self.action_history = np.zeros((self.n_agents, self.max_steps))
        self.adjacency_matrix = np.zeros((self.n_agents+1, self.n_agents+1))

        if options and "initial_positions" in options:
            self.drone_positions = options["initial_positions"]
        else:
            for i in range(self.n_agents):
                self.drone_positions[i] = [
                    self.np_random.integers(0, self.G1),
                    self.np_random.integers(0, self.G2)
                ]
        
        self.drone_positions = np.array([[0, 0], [3, 0], [3, 0], [7, 0], [0, 7], [5, 0], [0, 5], [2, 0], [0, 2], [0, 4], [4, 0], [1, 0], [0, 1]])[:self.n_agents]

        
        self.visited_cells = np.zeros((self.G1, self.G2))
        # Mark initial positions as visited
        for pos in self.drone_positions:
            self.visited_cells[int(pos[0]), int(pos[1])] = 1
            
        return self._get_obs(), {}

    def _get_obs(self) -> Dict:
        """Construct observation dictionary"""


        complete_obs = {"visited_cells": self.visited_cells > 0,
                "revisited_cells": self.visited_cells > 1,
                "agent_positions": self.drone_positions / (self.G1 + self.G2),
                "prev_agent_positions": self.prev_agent_positions / (self.G1 + self.G2),
                "local_empty_cells": self._get_local_empty_cells(),
                "closest_unvisited_cells": self._get_closest_unvisited() / (self.G1 + self.G2),
                "prev_actions": self.prev_actions / 5,
                "paths": self.paths / (self.G1 + self.G2),
                "action_history": self.action_history / 5,
                "adjacency_matrix": self.adjacency_matrix
                }
        if self.config["image_observation"]:
            visited_pixels = (self.visited_cells.reshape(1, self.G1, self.G2)/self.visited_cells.max() > 0)*255 
            revisited_pixels = (self.visited_cells.reshape(1, self.G1, self.G2)/self.visited_cells.max() > 1)*255
            complete_obs["visited_cells"] = visited_pixels.astype(np.uint8)
            complete_obs["revisited_cells"] = revisited_pixels.astype(np.uint8)
            # returns only the used keys
            return {key: complete_obs[key] for key in self.used_obs_keys}

        else:
            return {key: complete_obs[key] for key in self.used_obs_keys}

    def _compute_connectivity(self) -> float:
        """Optional connectivity metric computation"""
        if not self.config["enable_connectivity"]:
            return 0.0
            
        G = nx.Graph()
        gcs_idx = self.n_agents
        G.add_node(gcs_idx, pos=np.array(self.config["gcs_pos"]))
        
        # Add UAV nodes and edges
        for i in range(self.n_agents):
            G.add_node(i, pos=self.drone_positions[i])
            # Check GCS connection
            if np.linalg.norm(self.drone_positions[i] - np.array(self.config["gcs_pos"])) <= self.config["comm_dist"]:
                G.add_edge(i, gcs_idx)
        
        # Add UAV-UAV edges
        for i in range(self.n_agents):
            for j in range(i+1, self.n_agents):
                if np.linalg.norm(self.drone_positions[i] - self.drone_positions[j]) <= self.config["comm_dist"]:
                    G.add_edge(i, j)
        
        self.adjacency_matrix = nx.adjacency_matrix(G).toarray()
        # Count UAVs with path to GCS
        connected_count = 0
        for i in range(self.n_agents):
            try:
                if nx.has_path(G, i, gcs_idx):
                    connected_count += 1
            except nx.NetworkXNoPath:
                continue
                
        return connected_count / self.n_agents

    def step(self, actions) -> Tuple:
        """Execute actions and return next state"""
        
        # Track newly covered cells this step
        previous_coverage = np.sum(self.visited_cells > 0)
        previous_revisits = np.sum(self.visited_cells > 1)
        self.prev_agent_positions = self.drone_positions.copy()
        self.prev_actions = np.array(actions).reshape(self.n_agents, 1)
        
        # Process movements
        for agent_idx, action in enumerate(actions):
            move = self.action_to_move[action]
            new_pos = self.drone_positions[agent_idx] + move
            self.paths[agent_idx, self.timestep] = new_pos
            self.action_history[agent_idx, self.timestep] = action
            # Boundary check
            if (0 <= new_pos[0] < self.G1 and 
                0 <= new_pos[1] < self.G2):
                self.drone_positions[agent_idx] = new_pos
                
        # Update visited cells
        for pos in self.drone_positions:
            self.visited_cells[int(pos[0]), int(pos[1])] += 1
            
        # Compute rewards
        weights = self.config["reward_weights"]

        self.suggested_steps = self.G1 * self.G2 / self.n_agents

        # Modified time bonus calculation
        if self.timestep < self.suggested_steps:
            self.time_bonus = 1.0
        else:
            # Cap the minimum time bonus at -1.0 to prevent runaway negative rewards
            self.time_bonus = max(-1.0, 1.0 - 0.1 * ((self.timestep - self.suggested_steps) / self.suggested_steps))
        
        # Coverage reward
        new_coverage = np.sum(self.visited_cells > 0) - previous_coverage
        reward = weights["coverage"] * new_coverage + self.time_bonus
        
        # Revisit penalty
        revisits = np.sum(self.visited_cells > 1) - previous_revisits
        reward += weights["revisit"] * revisits
        
        connectivity = self._compute_connectivity()
        # Optional connectivity reward
        if self.config["enable_connectivity"]:
            reward += weights["connectivity"] * connectivity
        
        self.timestep += 1

        # Check termination
        coverage_percentage = np.sum(self.visited_cells > 0) / (self.G1 * self.G2)
        done = False

        if coverage_percentage == 1.0:  # All cells covered
            reward += 10
            done = True
        elif np.any(self.visited_cells > 3) or np.sum(self.visited_cells > 2) > 10:
            reward -= 10
            done = True

        truncated = self.timestep >= self.max_steps
        
        # Information dictionary
        info = {
            "unique_cells_covered": np.sum(self.visited_cells > 0),
            "coverage_percentage": np.sum(self.visited_cells > 0) / (self.G1 * self.G2),
            "total_revisits": np.sum(self.visited_cells > 1),
            "adjacency_matrix": self.adjacency_matrix,
            "connectivity": connectivity
        }
        
        return self._get_obs(), reward, done, truncated, info

    def render(self):
        """Optional visualization method"""
        pass