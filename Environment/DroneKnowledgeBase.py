from typing import Tuple, Dict, Any, Optional, List
import numpy as np
from collections import defaultdict

class TargetInfo:
    """Stores detailed information about a detected target"""
    def __init__(self):
        self.is_detected = False
        self.position = np.array([-1, -1], dtype=np.float32)
        self.confidence = 0.0
        self.last_seen_time = -1
        self.is_moving = False
        self.movement_direction = 0  # Maps to action space
        self.speed = 0.0
        self.priority = 0.0
        self.type = "unknown"
        self.predicted_positions = []  # List of predicted future positions
        self.observation_history = []  # List of (time, position, confidence) tuples
        
class DroneKnowledgeBase:
    """Enhanced knowledge base with detailed target tracking"""
    def __init__(self, grid_size: Tuple[int, int], n_targets: int):
        self.grid_size = grid_size
        self.n_targets = n_targets
        
        # Basic maps
        self.visited_map = np.zeros(grid_size, dtype=np.int32)
        self.target_belief_map = np.zeros(grid_size, dtype=np.float32)
        self.obstacle_map = np.zeros(grid_size, dtype=np.int32)
        self.energy_map = np.zeros(grid_size, dtype=np.float32)  # For tracking energy-rich areas
        
        # Target tracking
        self.target_info = [TargetInfo() for _ in range(n_targets)]
        self.unidentified_targets = []  # List of potential targets not yet matched
        
        

        # Communication tracking
        self.shared_knowledge = defaultdict(list)  # Tracks knowledge shared by other drones
        self.communication_history = defaultdict(list)  # Tracks communication events
        
        # Exploration metrics
        self.frontier_cells = set()  # Unexplored cells at the boundary of explored area
        self.information_gain_map = np.zeros(grid_size, dtype=np.float32)
        
    def update_target(self, target_idx: int, position: np.ndarray, confidence: float,
                     timestep: int, target_info: Optional[Dict[str, Any]] = None) -> None:
        """Update knowledge about a specific target"""
        if not 0 <= target_idx < self.n_targets:
            return
            
        target = self.target_info[target_idx]
        target.is_detected = True
        target.confidence = max(target.confidence, confidence)
        target.last_seen_time = timestep
        
        # Update position and history
        old_pos = target.position.copy()
        target.position = position.copy()
        target.observation_history.append((timestep, position.copy(), confidence))
        
        # Limit history length
        if len(target.observation_history) > 100:
            target.observation_history.pop(0)
            
        # Update movement information if we have multiple observations
        if len(target.observation_history) > 1 and not np.array_equal(old_pos, position):
            target.is_moving = True
            movement = position - old_pos
            target.speed = np.linalg.norm(movement)
            target.movement_direction = self._get_closest_action(movement)
            
            # Predict future positions
            target.predicted_positions = self._predict_future_positions(target)
            
        # Update additional target information if provided
        if target_info:
            target.type = target_info.get("target_type", target.type)
            target.priority = target_info.get("priority", target.priority)
            
        # Update belief map
        pos_int = position.astype(int)
        if 0 <= pos_int[0] < self.grid_size[0] and 0 <= pos_int[1] < self.grid_size[1]:
            self.target_belief_map[pos_int[0], pos_int[1]] = confidence
            
    def update_from_communication(self, source_drone_id: int, shared_info: Dict[str, Any],
                                timestep: int) -> None:
        """Update knowledge base with information shared by another drone"""
        # Record communication event
        self.communication_history[source_drone_id].append((timestep, shared_info))
        
        # Update target information
        for target_idx, target_data in shared_info.get("targets", {}).items():
            if target_idx >= self.n_targets:
                continue
                
            # Only update if the shared information is newer or more confident
            current_info = self.target_info[target_idx]
            shared_confidence = target_data.get("confidence", 0.0)
            shared_timestamp = target_data.get("last_seen_time", -1)
            
            if (shared_timestamp > current_info.last_seen_time or 
                (shared_timestamp == current_info.last_seen_time and 
                 shared_confidence > current_info.confidence)):
                self.update_target(
                    target_idx,
                    np.array(target_data["position"]),
                    shared_confidence,
                    shared_timestamp,
                    target_data
                )
                
        # Update maps
        for map_name, map_data in shared_info.get("maps", {}).items():
            if map_name == "visited":
                self.visited_map = np.maximum(self.visited_map, map_data)
            elif map_name == "obstacles":
                self.obstacle_map = np.maximum(self.obstacle_map, map_data)
            elif map_name == "energy":
                self.energy_map = np.maximum(self.energy_map, map_data)
                
        # Update frontier cells
        self._update_frontier_cells()
        
    def get_shareable_knowledge(self) -> Dict[str, Any]:
        """Get knowledge that can be shared with other drones"""
        shareable = {
            "targets": {},
            "maps": {
                "visited": self.visited_map.copy(),
                "obstacles": self.obstacle_map.copy(),
                "energy": self.energy_map.copy()
            }
        }
        
        # Add target information
        for idx, target in enumerate(self.target_info):
            if target.is_detected:
                shareable["targets"][idx] = {
                    "position": target.position.tolist(),
                    "confidence": target.confidence,
                    "last_seen_time": target.last_seen_time,
                    "is_moving": target.is_moving,
                    "movement_direction": target.movement_direction,
                    "speed": target.speed,
                    "type": target.type,
                    "priority": target.priority,
                    "predicted_positions": [pos.tolist() for pos in target.predicted_positions]
                }
                
        return shareable
        
    def _predict_future_positions(self, target: TargetInfo, 
                                prediction_steps: int = 5) -> List[np.ndarray]:
        """Predict future positions of a moving target"""
        if not target.is_moving:
            return []
            
        predictions = []
        current_pos = target.position.copy()
        movement = self._action_to_move(target.movement_direction) * target.speed
        
        for _ in range(prediction_steps):
            next_pos = current_pos + movement
            # Check boundaries
            if (0 <= next_pos[0] < self.grid_size[0] and 
                0 <= next_pos[1] < self.grid_size[1] and
                self.obstacle_map[int(next_pos[0]), int(next_pos[1])] == 0):
                predictions.append(next_pos.copy())
                current_pos = next_pos
            else:
                break
                
        return predictions
        
    def _update_frontier_cells(self) -> None:
        """Update the set of frontier cells (unexplored cells adjacent to explored ones)"""
        self.frontier_cells.clear()
        visited = self.visited_map > 0
        
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                if not visited[i, j] and self.obstacle_map[i, j] == 0:
                    # Check if adjacent to visited cell
                    for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        ni, nj = i + di, j + dj
                        if (0 <= ni < self.grid_size[0] and 
                            0 <= nj < self.grid_size[1] and 
                            visited[ni, nj]):
                            self.frontier_cells.add((i, j))
                            break
                            
        # Update information gain map
        self.information_gain_map.fill(0)
        for i, j in self.frontier_cells:
            self.information_gain_map[i, j] = 1.0
            
    def _action_to_move(self, action: int) -> np.ndarray:
        """Convert action to movement vector"""
        action_to_move = {
            0: np.array([0, 0], dtype=np.float32),   # hover
            1: np.array([0, 1], dtype=np.float32),   # up
            2: np.array([0, -1], dtype=np.float32),  # down
            3: np.array([-1, 0], dtype=np.float32),  # left
            4: np.array([1, 0], dtype=np.float32)    # right
        }
        return action_to_move[action]
        
    def _get_closest_action(self, movement: np.ndarray) -> int:
        """Convert movement vector to closest discrete action"""
        if np.allclose(movement, 0):
            return 0
            
        angle = np.arctan2(movement[1], movement[0])
        angles = {
            1: np.pi/2,    # up
            2: -np.pi/2,   # down
            3: np.pi,      # left
            4: 0,          # right
        }
        
        return min(angles.items(), key=lambda x: abs(x[1] - angle))[0] 