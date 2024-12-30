from typing import Tuple, Dict, Any, Optional, List
import numpy as np
from enum import Enum
import scipy.stats as stats

class DroneType(Enum):
    STANDARD = "standard"
    LONG_RANGE = "long_range"
    HIGH_SPEED = "high_speed"
    ENERGY_EFFICIENT = "energy_efficient"
    SENSOR_FOCUSED = "sensor_focused"

class SensorType(Enum):
    DETERMINISTIC = "deterministic"
    GAUSSIAN = "gaussian"
    RAYLEIGH = "rayleigh"
    RICIAN = "rician"

class Drone:
    """Base class for all drone types with heterogeneous capabilities"""
    def __init__(self, config: Dict[str, Any]):
        # Basic properties
        self.position = np.array(config.get("initial_position", [0, 0]), dtype=np.float32)
        self.drone_type = DroneType(config.get("drone_type", "standard"))
        self.grid_size = config.get("grid_size", (10, 10))
        
        # Movement capabilities
        self.speed = float(config.get("speed", 1.0))
        self.energy = float(config.get("initial_energy", 100.0))
        self.energy_capacity = float(config.get("energy_capacity", 100.0))
        self.energy_consumption_rate = float(config.get("energy_consumption_rate", 0.1))
        self.charging_rate = float(config.get("charging_rate", 0.5))
        
        # Communication capabilities
        self.comm_range = float(config.get("comm_range", 2.8284))
        self.comm_reliability = float(config.get("comm_reliability", 0.9))
        self.max_relay_hops = int(config.get("max_relay_hops", 2))
        
        # Sensor capabilities
        self.sensor_range = float(config.get("sensor_range", 1.0))
        self.sensor_type = SensorType(config.get("sensor_type", "deterministic"))
        self.sensor_reliability = float(config.get("sensor_reliability", 0.8))
        self.sensor_noise_std = float(config.get("sensor_noise_std", 0.1))
        
        # For Rician fading
        self.k_factor = float(config.get("k_factor", 4.0))  # Rician K-factor
        
        # Path tracking
        self.path = []
        self.path_capacity = int(config.get("path_capacity", 1000))
        
        # Initialize type-specific parameters
        self._initialize_drone_type()
        
    def _initialize_drone_type(self) -> None:
        """Set drone-specific parameters based on type"""
        if self.drone_type == DroneType.LONG_RANGE:
            self.comm_range *= 2.0
            self.energy_consumption_rate *= 1.5
            
        elif self.drone_type == DroneType.HIGH_SPEED:
            self.speed *= 1.5
            self.energy_consumption_rate *= 2.0
            
        elif self.drone_type == DroneType.ENERGY_EFFICIENT:
            self.energy_consumption_rate *= 0.5
            self.speed *= 0.8
            
        elif self.drone_type == DroneType.SENSOR_FOCUSED:
            self.sensor_range *= 1.5
            self.sensor_reliability *= 1.2
            self.energy_consumption_rate *= 1.2
            
    def move(self, action: int) -> bool:
        """Execute movement action and return success status"""
        if self.energy <= 0:
            return False
            
        action_to_move = {
            0: np.array([0, 0], dtype=np.float32),   # hover
            1: np.array([0, 1], dtype=np.float32),   # up
            2: np.array([0, -1], dtype=np.float32),  # down
            3: np.array([-1, 0], dtype=np.float32),  # left
            4: np.array([1, 0], dtype=np.float32)    # right
        }
        
        move = action_to_move[action]
        new_pos = self.position + move * self.speed
        
        # Check boundaries
        if not (0 <= new_pos[0] < self.grid_size[0] and 0 <= new_pos[1] < self.grid_size[1]):
            return False
            
        # Update position and consume energy
        self.position = new_pos
        self._consume_energy(action != 0)  # More energy for movement than hovering
        
        # Update path
        self.path.append(self.position.copy())
        if len(self.path) > self.path_capacity:
            self.path.pop(0)
            
        return True
        
    def detect_target(self, target_pos: np.ndarray, prior_probability: float = 0.5) -> Tuple[bool, float]:
        """Detect target using configured sensor model"""
        if self.energy <= 0:
            return False, 0.0
            
        distance = np.linalg.norm(self.position - target_pos)
        if distance > self.sensor_range:
            return False, 0.0
            
        # Base detection probability
        detection_prob = self.sensor_reliability * np.exp(-distance / self.sensor_range)
        
        if self.sensor_type == SensorType.DETERMINISTIC:
            return detection_prob > 0.5, detection_prob
            
        elif self.sensor_type == SensorType.GAUSSIAN:
            noisy_prob = detection_prob + np.random.normal(0, self.sensor_noise_std)
            noisy_prob = np.clip(noisy_prob, 0, 1)
            
        elif self.sensor_type == SensorType.RAYLEIGH:
            scale = detection_prob / np.sqrt(2)
            noisy_prob = stats.rayleigh.rvs(scale=scale)
            noisy_prob = np.clip(noisy_prob, 0, 1)
            
        elif self.sensor_type == SensorType.RICIAN:
            nu = np.sqrt(self.k_factor / (1 + self.k_factor)) * detection_prob
            sigma = np.sqrt(detection_prob**2 / (2 * (1 + self.k_factor)))
            noisy_prob = stats.rice.rvs(nu/sigma, scale=sigma)
            noisy_prob = np.clip(noisy_prob, 0, 1)
            
        # Combine with prior using Bayes rule
        posterior = (noisy_prob * prior_probability) / \
                   (noisy_prob * prior_probability + (1 - noisy_prob) * (1 - prior_probability))
                   
        detection = np.random.random() < posterior
        return detection, posterior
        
    def can_communicate_with(self, other_pos: np.ndarray) -> Tuple[bool, float]:
        """Check if can communicate with another position and return signal strength"""
        if self.energy <= 0:
            return False, 0.0
            
        distance = np.linalg.norm(self.position - other_pos)
        if distance > self.comm_range:
            return False, 0.0
            
        signal_strength = self.comm_reliability * np.exp(-distance / self.comm_range)
        return signal_strength > 0.2, signal_strength
        
    def _consume_energy(self, is_moving: bool) -> None:
        """Update energy levels based on action"""
        base_consumption = self.energy_consumption_rate
        if is_moving:
            base_consumption *= 2.0
            
        self.energy = max(0.0, self.energy - base_consumption)
        
    def charge(self) -> None:
        """Charge drone's energy"""
        self.energy = min(self.energy_capacity, self.energy + self.charging_rate)
        
    def get_state(self) -> Dict[str, Any]:
        """Get current drone state"""
        return {
            "position": self.position.copy(),
            "energy": self.energy,
            "drone_type": self.drone_type.value,
            "sensor_type": self.sensor_type.value,
            "comm_range": self.comm_range,
            "sensor_range": self.sensor_range,
            "path": self.path.copy() if self.path else []
        } 