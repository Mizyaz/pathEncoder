from typing import Dict, Any, Tuple, Optional
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from PyFlyt.core.aviary import Aviary
from PyFlyt.core.drones.quadx import QuadX

class RobustMultiUAVTargetEnv(gym.Env):
    """
    Enhanced Multi-UAV environment with PyFlyt physics for target search missions.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        # Default configuration with type annotations
        self.config = {
            "n_agents": 3,
            "grid_size": (20, 20),
            "n_targets": 5, 
            "target_types": [0, 1],  # 0: stationary, 1: mobile
            "enable_connectivity": True,
            "comm_radius": 5.0,
            "max_steps": 500,
            "reward_weights": {
                "coverage": 1.0,
                "target": 2.0,
                "connectivity": 0.5
            }
        }
        self.config.update(config)
        
        # Extract config values
        self.n_agents = self.config["n_agents"]
        self.G1, self.G2 = self.config["grid_size"]
        self.max_steps = self.config["max_steps"]
        
        # Initialize PyFlyt environment
        start_positions = np.zeros((self.n_agents, 3))
        start_orientations = np.zeros((self.n_agents, 3))
        
        # Randomly distribute initial positions
        for i in range(self.n_agents):
            start_positions[i] = [
                np.random.uniform(0, self.G1),
                np.random.uniform(0, self.G2),
                2.0  # Fixed height
            ]
        
        # Initialize PyFlyt aviary
        self.aviary = Aviary(
            start_positions,
            start_orientations,
            drone_type='quadx',
        )
        
        # Action and observation spaces
        self.action_space = spaces.Box(
            low=-1,
            high=1, 
            shape=(self.n_agents, 4),  # vx, vy, vz, yaw_rate
            dtype=np.float32
        )
        
        # Observation space with explicit shapes and types
        obs_spaces = {
            "position": spaces.Box(
                low=0, 
                high=max(self.G1, self.G2),
                shape=(self.n_agents, 3),
                dtype=np.float32
            ),
            "velocity": spaces.Box(
                low=-5,
                high=5,
                shape=(self.n_agents, 3), 
                dtype=np.float32
            ),
            "target_positions": spaces.Box(
                low=0,
                high=max(self.G1, self.G2),
                shape=(self.config["n_targets"], 2),
                dtype=np.float32
            )
        }
        self.observation_space = spaces.Dict(obs_spaces)
        
        # Initialize state variables
        self.timestep = 0
        self.visited_cells = np.zeros((self.G1, self.G2))
        self.target_positions = None
        self.target_types = None
        self.target_consecutive_visits = None

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Initialize environment state with explicit typing"""
        super().reset(seed=seed)
        self.timestep = 0
        
        # Reset PyFlyt aviary
        self.aviary.reset()
        
        # Reset visited cells grid
        self.visited_cells = np.zeros((self.G1, self.G2))
        
        # Initialize target positions and types
        self.target_positions = np.random.uniform(
            0, [self.G1, self.G2], 
            size=(self.config["n_targets"], 2)
        )
        self.target_types = np.random.choice(
            self.config["target_types"],
            size=self.config["n_targets"]
        )
        self.target_consecutive_visits = np.zeros(self.config["n_targets"])
        
        # Mark initial UAV positions as visited
        for drone in self.aviary.drones:
            print(drone)
            pos = drone.state["position"][:2]
            self.visited_cells[int(pos[0]), int(pos[1])] += 1
            
        return self._get_obs(), self._get_info()

    def step(self, actions: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """Execute actions with proper type handling and state updates"""
        # Track previous states
        previous_coverage = np.sum(self.visited_cells > 0)
        
        # Apply actions to PyFlyt drones
        for i, drone in enumerate(self.aviary.drones):
            # Convert normalized actions to drone commands
            vx, vy, vz, yaw_rate = actions[i]
            drone.setpoint = np.array([
                vx * 2.0,  # Scale to reasonable velocity
                vy * 2.0,
                vz * 2.0,
                yaw_rate * np.pi
            ])
        
        # Step PyFlyt simulation
        self.aviary.step()
        
        # Update visited cells and compute rewards
        reward = 0.0
        for drone in self.aviary.drones:
            pos = drone.state["position"][:2]
            self.visited_cells[int(pos[0]), int(pos[1])] += 1
            
        # Update rewards based on components
        weights = self.config["reward_weights"]
        
        # Coverage reward
        new_coverage = np.sum(self.visited_cells > 0)
        reward += weights["coverage"] * (new_coverage - previous_coverage)
        
        # Target detection reward
        reward += weights["target"] * self._update_target_states()
        
        # Connectivity reward
        if self.config["enable_connectivity"]:
            reward += weights["connectivity"] * self._compute_connectivity()
            
        # Update timestep and check termination
        self.timestep += 1
        
        # Check termination conditions
        done = False
        if self.config.get("coverage_termination", True):
            coverage_percentage = np.sum(self.visited_cells > 0) / (self.G1 * self.G2)
            done = coverage_percentage >= 0.99
            
        truncated = self.timestep >= self.max_steps
        
        return self._get_obs(), float(reward), bool(done), bool(truncated), self._get_info()

    def _get_obs(self) -> Dict[str, np.ndarray]:
        """Get current observation dictionary"""
        return {
            "position": np.array([drone.state["position"] for drone in self.aviary.drones]),
            "velocity": np.array([drone.state["velocity"] for drone in self.aviary.drones]),
            "target_positions": self.target_positions
        }

    def _get_info(self) -> Dict[str, Any]:
        """Get current info dictionary"""
        return {
            "timestep": self.timestep,
            "coverage": np.sum(self.visited_cells > 0) / (self.G1 * self.G2),
            "connectivity": self._compute_connectivity()
        }

    def render(self):
        """Render the environment"""
        return self.aviary.render()

    def close(self):
        """Clean up environment resources"""
        self.aviary.disconnect()