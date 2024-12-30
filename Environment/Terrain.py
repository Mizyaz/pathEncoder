from typing import Tuple, Dict, Any, List, Set, Optional
import numpy as np
from enum import Enum

class TerrainType(Enum):
    EMPTY = 0
    OBSTACLE = 1
    CHARGING_STATION = 2
    COMMUNICATION_RELAY = 3
    HIGH_INTERFERENCE = 4
    LOW_INTERFERENCE = 5

class Terrain:
    """Manages terrain features and obstacles in the environment"""
    def __init__(self, grid_size: Tuple[int, int], config: Optional[Dict[str, Any]] = None):
        self.grid_size = grid_size
        self.terrain_map = np.zeros(grid_size, dtype=np.int32)
        self.interference_map = np.zeros(grid_size, dtype=np.float32)
        
        if config is None:
            config = {}
            
        # Initialize static obstacles
        self.obstacles = []
        for pos in config.get("obstacles", []):
            self.terrain_map[pos[0], pos[1]] = TerrainType.OBSTACLE.value
            self.obstacles.append({"position": np.array(pos, dtype=np.float32)})
            
        # Initialize charging stations
        self.charging_stations = []
        for pos in config.get("charging_stations", []):
            self.terrain_map[pos[0], pos[1]] = TerrainType.CHARGING_STATION.value
            self.charging_stations.append({"position": np.array(pos, dtype=np.float32)})
            
        # Initialize communication relays
        self.comm_relays = []
        for pos in config.get("comm_relays", []):
            self.terrain_map[pos[0], pos[1]] = TerrainType.COMMUNICATION_RELAY.value
            self.comm_relays.append({"position": np.array(pos, dtype=np.float32)})
            
        # Initialize dynamic obstacles
        self.dynamic_obstacles = []
        for obs in config.get("dynamic_obstacles", []):
            obstacle = {
                "position": np.array(obs["initial_position"], dtype=np.float32),
                "pattern": obs["pattern"],
                "speed": obs["speed"],
                "waypoints": [np.array(wp, dtype=np.float32) for wp in obs.get("waypoints", [])],
                "current_waypoint": 0
            }
            self.dynamic_obstacles.append(obstacle)
            
        # Initialize interference
        interference_config = config.get("interference", {})
        self.base_interference = interference_config.get("base_level", 0.0)
        self.interference_map.fill(self.base_interference)
        
        # High interference zones
        for zone in interference_config.get("high_interference_zones", []):
            center = np.array(zone["center"])
            radius = zone["radius"]
            strength = zone["strength"]
            
            # Create circular interference zone
            y, x = np.ogrid[-center[0]:grid_size[0]-center[0], -center[1]:grid_size[1]-center[1]]
            mask = x*x + y*y <= radius*radius
            self.interference_map[mask] = np.maximum(self.interference_map[mask], strength)
            
        # Low interference corridors
        for corridor in interference_config.get("low_interference_corridors", []):
            start = np.array(corridor["start"])
            end = np.array(corridor["end"])
            width = corridor["width"]
            
            # Create line of low interference
            length = np.linalg.norm(end - start)
            direction = (end - start) / length
            normal = np.array([-direction[1], direction[0]])
            
            for i in range(grid_size[0]):
                for j in range(grid_size[1]):
                    point = np.array([i, j])
                    # Calculate distance to line
                    dist = abs(np.dot(point - start, normal))
                    if dist <= width:
                        self.interference_map[i, j] = self.base_interference
                        
    def update(self) -> None:
        """Update dynamic elements of the terrain"""
        # Update dynamic obstacles
        for obstacle in self.dynamic_obstacles:
            if obstacle["pattern"] == "patrol":
                # Get current and next waypoint
                current = obstacle["waypoints"][obstacle["current_waypoint"]]
                next_idx = (obstacle["current_waypoint"] + 1) % len(obstacle["waypoints"])
                next_point = obstacle["waypoints"][next_idx]
                
                # Calculate direction and move
                direction = next_point - obstacle["position"]
                distance = np.linalg.norm(direction)
                
                if distance > 0:
                    normalized_direction = direction / distance
                    new_position = obstacle["position"] + normalized_direction * obstacle["speed"]
                    
                    # Check if we've reached the next waypoint
                    if np.linalg.norm(new_position - next_point) < obstacle["speed"]:
                        obstacle["position"] = next_point.copy()
                        obstacle["current_waypoint"] = next_idx
                    else:
                        obstacle["position"] = new_position
                        
                # Update terrain map
                old_pos = np.floor(obstacle["position"]).astype(np.int32)
                if 0 <= old_pos[0] < self.grid_size[0] and 0 <= old_pos[1] < self.grid_size[1]:
                    self.terrain_map[old_pos[0], old_pos[1]] = TerrainType.EMPTY.value
                    
                new_pos = np.floor(obstacle["position"]).astype(np.int32)
                if 0 <= new_pos[0] < self.grid_size[0] and 0 <= new_pos[1] < self.grid_size[1]:
                    self.terrain_map[new_pos[0], new_pos[1]] = TerrainType.OBSTACLE.value
                    
    def is_obstacle(self, position: Tuple[int, int]) -> bool:
        """Check if position contains an obstacle"""
        if not (0 <= position[0] < self.grid_size[0] and 0 <= position[1] < self.grid_size[1]):
            return True
        return self.terrain_map[position[0], position[1]] == TerrainType.OBSTACLE.value
        
    def is_charging_station(self, position: Tuple[int, int]) -> bool:
        """Check if position is a charging station"""
        if not (0 <= position[0] < self.grid_size[0] and 0 <= position[1] < self.grid_size[1]):
            return False
        return self.terrain_map[position[0], position[1]] == TerrainType.CHARGING_STATION.value
        
    def is_comm_relay(self, position: Tuple[int, int]) -> bool:
        """Check if position is a communication relay"""
        if not (0 <= position[0] < self.grid_size[0] and 0 <= position[1] < self.grid_size[1]):
            return False
        return self.terrain_map[position[0], position[1]] == TerrainType.COMMUNICATION_RELAY.value
        
    def get_interference(self, position: Tuple[int, int]) -> float:
        """Get interference level at position"""
        if not (0 <= position[0] < self.grid_size[0] and 0 <= position[1] < self.grid_size[1]):
            return 1.0
        return float(self.interference_map[position[0], position[1]])
        
    def get_state(self) -> Dict[str, Any]:
        """Get current state of terrain"""
        return {
            "terrain_map": self.terrain_map.copy(),
            "interference_map": self.interference_map.copy(),
            "dynamic_obstacles": [
                {
                    "position": obs["position"].copy(),
                    "pattern": obs["pattern"],
                    "speed": obs["speed"],
                    "current_waypoint": obs["current_waypoint"]
                }
                for obs in self.dynamic_obstacles
            ]
        } 