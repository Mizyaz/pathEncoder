from typing import Tuple, Dict, Any, Optional, List
import numpy as np
from enum import Enum

class TargetType(Enum):
    STATIC = "static"
    RANDOM_WALK = "random_walk"
    PATROL = "patrol"
    WAYPOINT = "waypoint"

class Target:
    """Base class for all target types"""
    def __init__(self, config: Dict[str, Any]):
        self.position = np.array(config.get("initial_position", [0, 0]), dtype=np.float32)
        self.priority = float(config.get("priority", 1.0))
        self.target_type = TargetType(config.get("target_type", "static"))
        self.grid_size = config.get("grid_size", (10, 10))
        self.movement_pattern = config.get("movement_pattern", [])
        self.current_waypoint = 0
        self.direction = 0  # Maps to action space: 0=hover, 1=up, 2=down, 3=left, 4=right
        self.speed = float(config.get("speed", 1.0))
        self.time_slots = config.get("time_slots", [])
        self.is_active = True
        
    def step(self) -> None:
        """Update target position based on movement type"""
        if self.target_type == TargetType.STATIC:
            return
            
        if self.target_type == TargetType.RANDOM_WALK:
            self._random_walk()
        elif self.target_type == TargetType.PATROL:
            self._patrol()
        elif self.target_type == TargetType.WAYPOINT:
            self._follow_waypoints()
            
    def _random_walk(self) -> None:
        """Random walk movement pattern"""
        if np.random.random() < 0.3:  # 30% chance to change direction
            self.direction = np.random.randint(0, 5)
            
        move = self._direction_to_move()
        new_pos = self.position + move * self.speed
        
        # Boundary check and update
        if (0 <= new_pos[0] < self.grid_size[0] and 0 <= new_pos[1] < self.grid_size[1]):
            self.position = new_pos
            
    def _patrol(self) -> None:
        """Patrol between predefined points"""
        if len(self.movement_pattern) < 2:
            return
            
        target_pos = np.array(self.movement_pattern[self.current_waypoint])
        if np.allclose(self.position, target_pos, atol=self.speed):
            self.current_waypoint = (self.current_waypoint + 1) % len(self.movement_pattern)
            target_pos = np.array(self.movement_pattern[self.current_waypoint])
            
        direction = target_pos - self.position
        norm = np.linalg.norm(direction)
        if norm > 0:
            self.position += direction / norm * min(self.speed, norm)
            # Update direction for observation
            self.direction = self._get_closest_action(direction)
            
    def _follow_waypoints(self) -> None:
        """Follow waypoints in sequence"""
        if len(self.movement_pattern) == 0:
            return
            
        target_pos = np.array(self.movement_pattern[self.current_waypoint])
        if np.allclose(self.position, target_pos, atol=self.speed):
            if self.current_waypoint < len(self.movement_pattern) - 1:
                self.current_waypoint += 1
            else:
                return  # End of waypoints
                
        direction = target_pos - self.position
        norm = np.linalg.norm(direction)
        if norm > 0:
            self.position += direction / norm * min(self.speed, norm)
            self.direction = self._get_closest_action(direction)
            
    def _direction_to_move(self) -> np.ndarray:
        """Convert action space direction to movement vector"""
        action_to_move = {
            0: np.array([0, 0], dtype=np.float32),   # hover
            1: np.array([0, 1], dtype=np.float32),   # up
            2: np.array([0, -1], dtype=np.float32),  # down
            3: np.array([-1, 0], dtype=np.float32),  # left
            4: np.array([1, 0], dtype=np.float32)    # right
        }
        return action_to_move[self.direction]
        
    def _get_closest_action(self, direction: np.ndarray) -> int:
        """Convert continuous direction to closest discrete action"""
        angles = {
            1: np.pi/2,    # up
            2: -np.pi/2,   # down
            3: np.pi,      # left
            4: 0,          # right
        }
        
        angle = np.arctan2(direction[1], direction[0])
        if np.allclose(direction, 0):
            return 0  # hover
            
        closest_action = min(angles.items(), key=lambda x: abs(x[1] - angle))[0]
        return closest_action
        
    def is_available(self, timestep: int) -> bool:
        """Check if target is available at given timestep"""
        if not self.time_slots:
            return self.is_active
        return timestep in self.time_slots and self.is_active
        
    def get_state(self) -> Dict[str, Any]:
        """Get current target state"""
        return {
            "position": self.position.copy(),
            "priority": self.priority,
            "target_type": self.target_type.value,
            "direction": self.direction,
            "is_active": self.is_active,
            "speed": self.speed
        } 