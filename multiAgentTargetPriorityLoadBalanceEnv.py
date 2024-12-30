import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Dict, Tuple, List, Optional, Any
import networkx as nx
from collections import defaultdict

from Environment.Target import Target, TargetType
from Environment.Drone import Drone, DroneType, SensorType
from Environment.DroneKnowledgeBase import DroneKnowledgeBase
from Environment.Terrain import Terrain, TerrainType

class RobustMultiUAVTargetLoadBalanceEnv(gym.Env):
    """
    Enhanced Multi-UAV environment with sensor uncertainties, communication tracking,
    heterogeneous agents, and dynamic targets.
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
            "target_priorities": [1.0, 0.7, 0.3],
            "inform_times": 3,
            "time_slots": {},
            "sensor_config": {
                "detection_range": 1.0,
                "base_reliability": 1.0,
                "noise_std": 0,
                "sensor_type": "deterministic"
            },
            "drone_configs": [
                {"drone_type": "standard"},
                {"drone_type": "standard"},
                {"drone_type": "standard"},
                {"drone_type": "standard"},
                {"drone_type": "standard"},
                {"drone_type": "standard"},
                {"drone_type": "standard"},
                {"drone_type": "standard"}
            ],
            "target_configs": [
                {"target_type": "static", "priority": 1.0},
                {"target_type": "random_walk", "priority": 0.7, "speed": 0.5},
                {"target_type": "patrol", "priority": 0.3, "movement_pattern": [[1, 1], [8, 1], [8, 8], [1, 8]]}
            ],
            "terrain_config": {
                "obstacles": [],
                "charging_stations": [],
                "comm_relays": [],
                "interference": {
                    "base_level": 0.1,
                    "high_interference_zones": [
                        {"center": (3, 3), "radius": 2, "strength": 0.8}
                    ],
                    "low_interference_corridors": [
                        {"start": (0, 0), "end": (9, 9), "width": 1}
                    ]
                },
                "dynamic_obstacles": [

                ]
            },
            "obs_config": {
                "known_target_positions": False,
                "known_target_priorities": False,
                "known_time_slots": False,
            },
            "reward_weights": {
                "coverage": 1.0,
                "revisit": -0.1,
                "connectivity": 0.5,
                "target_detection": 2.0,
                "target_relay": 1.5,
                "target_completion": 5.0,
                "early_completion": 3.0,
                "load_balance": 0.5,
                "energy_efficiency": 0.3,
                "collision_penalty": -2.0
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
        
        # Initialize terrain
        self.terrain = Terrain((self.G1, self.G2), self.config["terrain_config"])
        
        # Initialize drones with heterogeneous capabilities
        self.drones = []
        for i in range(self.n_agents):
            drone_config = self.config["drone_configs"][i].copy()
            drone_config.update({
                "grid_size": (self.G1, self.G2),
                "initial_position": self._get_initial_position(i),
                "sensor_config": self.config["sensor_config"]
            })
            self.drones.append(Drone(drone_config))
            
        # Initialize targets with different behaviors
        self.targets = []
        for i in range(self.n_targets):
            target_config = self.config["target_configs"][i].copy()
            target_config.update({
                "grid_size": (self.G1, self.G2),
                "initial_position": self._get_random_position()
            })
            self.targets.append(Target(target_config))
            
        # Initialize knowledge bases for each drone
        self.knowledge_bases = [
            DroneKnowledgeBase((self.G1, self.G2), self.n_targets)
            for _ in range(self.n_agents)
        ]
        
        # Communication tracking
        self.comm_matrix = np.zeros((self.n_agents + 1, self.n_agents + 1), dtype=np.int32)  # +1 for GCS
        
        # Initialize observation flags
        self.obs_config = self.config["obs_config"]
        
        # Action space remains the same
        self.action_space = spaces.MultiDiscrete([5] * self.n_agents)
        
        # Enhanced observation space with per-drone observations
        if self.n_targets > 0:
            obs_spaces = {
                "visited_cells": spaces.Box(0, np.inf, shape=(self.n_agents, self.G1, self.G2), dtype=np.float32),
                "agent_positions": spaces.Box(0, max(self.G1, self.G2), shape=(self.n_agents, 2), dtype=np.float32),
                "agent_energies": spaces.Box(0, 1, shape=(self.n_agents,), dtype=np.float32),
                "detected_targets": spaces.Box(0, 1, shape=(self.n_agents, self.n_targets), dtype=np.float32),
                "target_completion": spaces.Box(0, 1, shape=(self.n_targets,), dtype=np.float32),
                "target_priorities": spaces.Box(0, 1, shape=(self.n_targets,), dtype=np.float32),
                "target_knowledge": spaces.Box(0, 1, shape=(self.n_agents, self.n_targets), dtype=np.float32),
                "target_availability": spaces.Box(0, 1, shape=(self.n_targets,), dtype=np.float32),
                "target_positions": spaces.Box(0, max(self.G1, self.G2), shape=(self.n_agents, self.n_targets, 2), dtype=np.float32),
                "target_confidences": spaces.Box(0, 1, shape=(self.n_agents, self.n_targets), dtype=np.float32),
                "target_movement_info": spaces.Box(0, 1, shape=(self.n_agents, self.n_targets, 3), dtype=np.float32),  # is_moving, direction, speed
                "terrain_map": spaces.Box(0, len(TerrainType), shape=(self.G1, self.G2), dtype=np.int32),
                "interference_map": spaces.Box(0, 1, shape=(self.G1, self.G2), dtype=np.float32),
                "frontier_cells": spaces.Box(0, 1, shape=(self.G1, self.G2), dtype=np.float32),
            }
        else:
            obs_spaces = {
                "visited_cells": spaces.Box(0, np.inf, shape=(self.n_agents, self.G1, self.G2), dtype=np.float32),
                "agent_positions": spaces.Box(0, max(self.G1, self.G2), shape=(self.n_agents, 2), dtype=np.float32),
                "agent_energies": spaces.Box(0, 1, shape=(self.n_agents,), dtype=np.float32),
            }
            
            
        self.observation_space = spaces.Dict(obs_spaces)
        
        self.timestep = 0
        self.coverage_percentages = []
        self.count = 0
        
    def _get_initial_position(self, agent_idx: int) -> np.ndarray:
        """Get initial position for drone"""
        if agent_idx < 4:
            return np.array([0, agent_idx * 2], dtype=np.float32)
        else:
            return np.array([agent_idx - 4, 0], dtype=np.float32)
            
    def _get_random_position(self) -> np.ndarray:
        """Get random valid position"""
        while True:
            pos = np.array([
                self.np_random.integers(0, self.G1),
                self.np_random.integers(0, self.G2)
            ], dtype=np.float32)
            if not self.terrain.is_obstacle(tuple(pos.astype(int))):
                return pos
                
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset environment state"""
        super().reset(seed=seed)
        self.timestep = 0
        
        # Reset communication tracking
        self.comm_matrix = np.zeros((self.n_agents + 1, self.n_agents + 1), dtype=np.int32)
        
        # Reset drones
        for i, drone in enumerate(self.drones):
            if options and "initial_positions" in options:
                drone.position = np.array(options["initial_positions"][i], dtype=np.float32)
            else:
                drone.position = self._get_initial_position(i)
            drone.energy = drone.energy_capacity
            drone.path.clear()
            
        # Reset targets
        for i, target in enumerate(self.targets):
            if options and "target_positions" in options:
                target.position = np.array(options["target_positions"][i], dtype=np.float32)
            else:
                target.position = self._get_random_position()
            target.is_active = True
            
        # Reset knowledge bases
        self.knowledge_bases = [
            DroneKnowledgeBase((self.G1, self.G2), self.n_targets)
            for _ in range(self.n_agents)
        ]
        
        # Reset terrain
        self.terrain = Terrain((self.G1, self.G2), self.config["terrain_config"])
        
        return self._get_obs(), self._get_info()
        
    def _get_obs(self) -> Dict[str, np.ndarray]:
        """Construct observation dictionary with per-drone observations"""
        # Initialize base observation with non-target related items
        obs = {
            "visited_cells": np.array([kb.visited_map for kb in self.knowledge_bases]),
            "agent_positions": np.array([drone.position for drone in self.drones]),
            "agent_energies": np.array([drone.energy / drone.energy_capacity for drone in self.drones]),
        }
        
        # Handle target-related observations
        if self.n_targets > 0:
            # Initialize target arrays with proper shapes
            detected_targets = np.zeros((self.n_agents, self.n_targets), dtype=np.float32)
            target_completion = np.zeros(self.n_targets, dtype=np.float32)
            target_knowledge = np.zeros((self.n_agents, self.n_targets), dtype=np.float32)
            target_availability = np.zeros(self.n_targets, dtype=np.float32)
            target_positions = np.zeros((self.n_agents, self.n_targets, 2), dtype=np.float32)
            target_confidences = np.zeros((self.n_agents, self.n_targets), dtype=np.float32)
            target_movement_info = np.zeros((self.n_agents, self.n_targets, 3), dtype=np.float32)
            target_priorities = np.zeros(self.n_targets, dtype=np.float32)
            
            # Fill target information if available
            for i in range(self.n_targets):
                for j, kb in enumerate(self.knowledge_bases):
                    if hasattr(kb, 'target_info') and i < len(kb.target_info):
                        detected_targets[j, i] = float(kb.target_info[i].is_detected)
                        target_knowledge[j, i] = float(kb.target_info[i].is_detected)
                        target_positions[j, i] = kb.target_info[i].position
                        target_confidences[j, i] = kb.target_info[i].confidence
                        if hasattr(kb.target_info[i], 'is_moving'):
                            target_movement_info[j, i] = [
                                float(kb.target_info[i].is_moving),
                                kb.target_info[i].movement_direction / 4.0 if hasattr(kb.target_info[i], 'movement_direction') else 0.0,
                                kb.target_info[i].speed if hasattr(kb.target_info[i], 'speed') else 0.0
                            ]
                
                if i < len(self.targets):
                    target_availability[i] = float(self.targets[i].is_available(self.timestep))
                    if self.obs_config.get("known_target_priorities", False):
                        target_priorities[i] = self.targets[i].priority
                
                # Calculate target completion
                if i < len(self.targets):
                    target_completion[i] = float(all(kb.target_info[i].is_detected for kb in self.knowledge_bases if hasattr(kb, 'target_info')))
            
            # Add target-related observations
            obs.update({
                "detected_targets": detected_targets,
                "target_completion": target_completion,
                "target_knowledge": target_knowledge,
                "target_availability": target_availability,
                "target_positions": target_positions,
                "target_confidences": target_confidences,
                "target_movement_info": target_movement_info,
                "target_priorities": target_priorities
            })
        else:
            # Add empty target-related observations for zero targets
            obs.update({
                "detected_targets": np.zeros((self.n_agents, 0), dtype=np.float32),
                "target_completion": np.zeros(0, dtype=np.float32),
                "target_knowledge": np.zeros((self.n_agents, 0), dtype=np.float32),
                "target_availability": np.zeros(0, dtype=np.float32),
                "target_positions": np.zeros((self.n_agents, 0, 2), dtype=np.float32),
                "target_confidences": np.zeros((self.n_agents, 0), dtype=np.float32),
                "target_movement_info": np.zeros((self.n_agents, 0, 3), dtype=np.float32),
                "target_priorities": np.zeros(0, dtype=np.float32)
            })
        
        # Add terrain-related observations with defaults
        obs.update({
            "terrain_map": getattr(self.terrain, 'terrain_map', np.zeros((self.G1, self.G2), dtype=np.int32)),
            "interference_map": getattr(self.terrain, 'interference_map', np.zeros((self.G1, self.G2), dtype=np.float32)),
            "frontier_cells": np.array([getattr(kb, 'information_gain_map', np.zeros((self.G1, self.G2), dtype=np.float32)) for kb in self.knowledge_bases])
        })

        if self.n_targets == 0:
            final_obs = {
                "visited_cells": np.array([kb.visited_map for kb in self.knowledge_bases]),
                "agent_positions": np.array([drone.position for drone in self.drones]),
                "agent_energies": np.array([drone.energy / drone.energy_capacity for drone in self.drones]),
            }
        else:
            final_obs = obs
        
        return final_obs
        
    def _get_info(self) -> Dict[str, Any]:
        """Construct info dictionary with detailed statistics"""
        info = {
            "timestep": self.timestep,
            "n_targets": self.n_targets,
            "detected_targets": np.array([any(kb.target_info[i].is_detected for kb in self.knowledge_bases) for i in range(self.n_targets)]),
            "completed_targets": np.array([all(kb.target_info[i].is_detected for kb in self.knowledge_bases) for i in range(self.n_targets)]),
            "connectivity": self._compute_connectivity(),
            "coverage_percentage": np.mean([np.sum(kb.visited_map > 0) / (self.G1 * self.G2) for kb in self.knowledge_bases]),
            "energy_levels": np.array([drone.energy / drone.energy_capacity for drone in self.drones]),
            "communication_matrix": self.comm_matrix.copy(),
            "terrain_state": self.terrain.get_state(),
            "target_states": [target.get_state() for target in self.targets],
            "drone_states": [drone.get_state() for drone in self.drones]
        }
        
        # Calculate load distribution for each drone
        for i in range(self.n_agents):
            total_comms = np.sum(self.comm_matrix[i])
            if total_comms > 0:
                distribution = self.comm_matrix[i] / total_comms
                info[f"drone_{i}_load_distribution"] = distribution
                
        return info
        
    def _compute_connectivity(self) -> float:
        """Compute connectivity metric with interference and relays"""
        if not self.config["enable_connectivity"]:
            return 0.0
            
        G = nx.Graph()
        gcs_idx = self.n_agents
        G.add_node(gcs_idx, pos=np.array(self.config["gcs_pos"], dtype=np.float32))
        
        # Add drone nodes
        for i, drone in enumerate(self.drones):
            G.add_node(i, pos=drone.position)
            
            # Check direct connection to GCS
            can_connect, signal_strength = drone.can_communicate_with(np.array(self.config["gcs_pos"]))
            if can_connect:
                interference = self.terrain.get_interference(tuple(drone.position.astype(int)))
                effective_strength = signal_strength * (1 - interference)
                if effective_strength > 0.2:
                    G.add_edge(i, gcs_idx, weight=effective_strength)
                    self.comm_matrix[i, -1] += 1
                    
        # Add drone-drone edges
        for i in range(self.n_agents):
            for j in range(i+1, self.n_agents):
                can_connect, signal_strength = self.drones[i].can_communicate_with(self.drones[j].position)
                if can_connect:
                    pos_i = tuple(self.drones[i].position.astype(int))
                    pos_j = tuple(self.drones[j].position.astype(int))
                    interference = max(
                        self.terrain.get_interference(pos_i),
                        self.terrain.get_interference(pos_j)
                    )
                    effective_strength = signal_strength * (1 - interference)
                    if effective_strength > 0.2:
                        G.add_edge(i, j, weight=effective_strength)
                        self.comm_matrix[i, j] += 1
                        self.comm_matrix[j, i] += 1
                        
        # Count connected drones
        connected_count = 0
        for i in range(self.n_agents):
            try:
                if nx.has_path(G, i, gcs_idx):
                    connected_count += 1
            except nx.NetworkXNoPath:
                continue
                
        return connected_count / self.n_agents
        
    def step(self, actions: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """Execute actions with enhanced dynamics"""
        # Process movements and update states
        collisions = np.zeros(self.n_agents, dtype=bool)
        
        # Move drones
        for agent_idx, action in enumerate(actions):
            success = self.drones[agent_idx].move(action)
            if not success:
                collisions[agent_idx] = True
                
            # Update knowledge base
            pos_int = self.drones[agent_idx].position.astype(int)
            self.knowledge_bases[agent_idx].visited_map[pos_int[0], pos_int[1]] += 1
            
        # Update targets
        for target in self.targets:
            target.step()
            
        # Update terrain
        self.terrain.update()
        
        # Process target detection and knowledge sharing
        self._process_target_detection()
        self._share_knowledge()
        
        # Compute rewards
        reward = self._compute_reward(collisions)
        
        self.timestep += 1
        
        # Check termination
        if self.n_targets > 0:
            done = all(target.is_active == False for target in self.targets)
        else:
            coverage = np.mean([np.sum(kb.visited_map > 0) / (self.G1 * self.G2) for kb in self.knowledge_bases])
            done = coverage >= 0.99
            
        truncated = self.timestep >= self.max_steps or all(drone.energy <= 0 for drone in self.drones)
        
        if done or truncated:
            self.coverage_percentages.append(np.mean([np.sum(kb.visited_map > 0) / (self.G1 * self.G2) for kb in self.knowledge_bases]))
            if self.count % 10 == 0:
                print(f"Max coverage: {max(self.coverage_percentages):.2f}, "
                      f"Current: {self.coverage_percentages[-1]:.2f}, "
                      f"Recent avg: {np.mean(self.coverage_percentages[-10:]):.2f}")
                      
                # Print load distribution for each drone
                for i in range(self.n_agents):
                    total_comms = np.sum(self.comm_matrix[i])
                    if total_comms > 0:
                        distribution = self.comm_matrix[i] / total_comms
                        print(f"Drone {i} ({self.drones[i].drone_type.value}) load distribution:", 
                              ", ".join([f"{x:.2f}" for x in distribution]))
            self.count += 1
            
        return self._get_obs(), float(reward), bool(done), bool(truncated), self._get_info()
        
    def _process_target_detection(self) -> None:
        """Process target detection for all drones"""
        for agent_idx, drone in enumerate(self.drones):
            for target_idx, target in enumerate(self.targets):
                if target.is_available(self.timestep):
                    kb = self.knowledge_bases[agent_idx]
                    prior_prob = kb.target_belief_map[int(target.position[0]), int(target.position[1])]
                    if prior_prob == 0:
                        prior_prob = 0.5
                        
                    detection, confidence = drone.detect_target(
                        target.position,
                        prior_prob
                    )
                    
                    if detection:
                        target_info = target.get_state()
                        kb.update_target(
                            target_idx,
                            target.position,
                            confidence,
                            self.timestep,
                            target_info
                        )
                        
    def _share_knowledge(self) -> None:
        """Share knowledge between connected drones"""
        # Build connectivity graph
        G = nx.Graph()
        for i in range(self.n_agents):
            for j in range(i+1, self.n_agents):
                can_connect, _ = self.drones[i].can_communicate_with(self.drones[j].position)
                if can_connect:
                    G.add_edge(i, j)
                    
        # Share knowledge within connected components
        for component in nx.connected_components(G):
            shared_knowledge = {}
            # Merge knowledge from all drones in component
            for drone_idx in component:
                kb = self.knowledge_bases[drone_idx]
                drone_knowledge = kb.get_shareable_knowledge()
                for key, value in drone_knowledge.items():
                    if key not in shared_knowledge:
                        shared_knowledge[key] = value
                    else:
                        if isinstance(value, dict):
                            shared_knowledge[key].update(value)
                        elif isinstance(value, np.ndarray):
                            shared_knowledge[key] = np.maximum(shared_knowledge[key], value)
                            
            # Update all drones in component with merged knowledge
            for drone_idx in component:
                self.knowledge_bases[drone_idx].update_from_communication(
                    -1,  # Special ID for merged knowledge
                    shared_knowledge,
                    self.timestep
                )
                
    def _compute_reward(self, collisions: np.ndarray) -> float:
        """Compute reward with enhanced components"""
        weights = self.config["reward_weights"]
        reward = 0.0
        
        # Coverage reward
        for kb in self.knowledge_bases:
            new_cells = np.sum(kb.visited_map == 1)
            reward += weights["coverage"] * new_cells
            reward += weights["revisit"] * np.sum(kb.visited_map > 1)
            
        # Connectivity reward
        if self.config["enable_connectivity"]:
            reward += weights["connectivity"] * self._compute_connectivity()
            
        # Target detection and completion rewards
        for target_idx, target in enumerate(self.targets):
            if target.is_available(self.timestep):
                detected = False
                completed = True
                for kb in self.knowledge_bases:
                    if kb.target_info[target_idx].is_detected:
                        detected = True
                    else:
                        completed = False
                        
                if detected and not target.is_active:
                    reward += weights["target_detection"] * target.priority
                    
                if completed:
                    target.is_active = False
                    completion_reward = weights["target_completion"]
                    early_factor = 1.0 - (self.timestep / self.max_steps)
                    reward += completion_reward * target.priority * (1 + weights["early_completion"] * early_factor)
                    
        # Load balance reward
        if np.sum(self.comm_matrix) > 0:
            load_distribution = self.comm_matrix / np.sum(self.comm_matrix)
            reward += -weights["load_balance"] * np.std(load_distribution)
            
        # Energy efficiency reward
        energy_levels = np.array([drone.energy / drone.energy_capacity for drone in self.drones])
        reward += weights["energy_efficiency"] * np.mean(energy_levels)
        
        # Collision penalty
        reward += weights["collision_penalty"] * np.sum(collisions)
        
        return reward
