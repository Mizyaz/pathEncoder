import numpy as np
from multiAgentTargetPriorityLoadBalanceEnv import RobustMultiUAVTargetLoadBalanceEnv
import time
from typing import Dict, Any
import json

def print_step_info(info: Dict[str, Any], step: int) -> None:
    """Print relevant information for each step"""
    print(f"\nStep {step}:")
    print(f"Coverage: {info['coverage_percentage']:.2f}%")
    if 'energy_levels' in info:
        print(f"Energy Levels: {[f'{e:.2f}' for e in info['energy_levels']]}")
    if 'connectivity' in info:
        print(f"Connectivity: {info['connectivity']:.2f}")
    if 'detected_targets' in info:
        print(f"Detected Targets: {info['detected_targets']}")
    if 'drone_states' in info:
        for i, drone in enumerate(info['drone_states']):
            print(f"Drone {i} ({drone['drone_type']}): pos={drone['position']}")

def run_random_episode(env, max_steps: int = 100, render: bool = False) -> None:
    """Run a single episode with random actions"""
    obs, info = env.reset()
    done = False
    truncated = False
    total_reward = 0
    step = 0
    
    while not done and not truncated and step < max_steps:
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        
        if render:
            print_step_info(info, step)
            
        step += 1
    
    print(f"\nEpisode finished after {step} steps")
    print(f"Final coverage: {info['coverage_percentage']:.2f}%")
    print(f"Total reward: {total_reward:.2f}")
    return info

def test_single_agent_coverage():
    """Test basic coverage with a single agent"""
    print("\n=== Testing Single Agent Coverage ===")
    config = {
        "name": "single_agent_coverage",
        "n_agents": 1,
        "grid_size": (10, 10),
        "max_steps": 50,
        "enable_connectivity": False,
        "n_targets": 0,
        "drone_configs": [
            {"drone_type": "standard"}
        ],
        "reward_weights": {
            "coverage": 1.0,
            "revisit": -0.1,
            "connectivity": 0.0,
            "target_detection": 0.0,
            "target_relay": 0.0,
            "target_completion": 0.0,
            "early_completion": 0.0,
            "load_balance": 0.0,
            "energy_efficiency": 0.0,
            "collision_penalty": -1.0
        }
    }
    
    env = RobustMultiUAVTargetLoadBalanceEnv(config)
    run_random_episode(env, render=True)

def test_multi_agent_coverage():
    """Test coverage with multiple agents"""
    print("\n=== Testing Multi-Agent Coverage ===")
    config = {
        "name": "multi_agent_coverage",
        "n_agents": 4,
        "grid_size": (10, 10),
        "max_steps": 50,
        "enable_connectivity": False,
        "n_targets": 0,
        "drone_configs": [
            {"drone_type": "standard"},
            {"drone_type": "high_speed"},
            {"drone_type": "energy_efficient"},
            {"drone_type": "sensor_focused"}
        ],
        "reward_weights": {
            "coverage": 1.0,
            "revisit": -0.1,
            "connectivity": 0.0,
            "target_detection": 0.0,
            "target_relay": 0.0,
            "target_completion": 0.0,
            "early_completion": 0.0,
            "load_balance": 0.0,
            "energy_efficiency": 0.3,
            "collision_penalty": -1.0
        }
    }
    
    env = RobustMultiUAVTargetLoadBalanceEnv(config)
    run_random_episode(env, render=True)

def test_connectivity():
    """Test connectivity requirements"""
    print("\n=== Testing Connectivity Requirements ===")
    config = {
        "name": "connectivity_test",
        "n_agents": 4,
        "grid_size": (10, 10),
        "max_steps": 100,
        "enable_connectivity": True,
        "comm_dist": 2.0,
        "gcs_pos": (0, 0),
        "n_targets": 0,
        "drone_configs": [
            {"drone_type": "standard"},
            {"drone_type": "long_range"},
            {"drone_type": "long_range"},
            {"drone_type": "standard"}
        ],
        "reward_weights": {
            "coverage": 0.5,
            "revisit": -0.1,
            "connectivity": 1.0,
            "target_detection": 0.0,
            "target_relay": 0.0,
            "target_completion": 0.0,
            "early_completion": 0.0,
            "load_balance": 0.5,
            "energy_efficiency": 0.3,
            "collision_penalty": -1.0
        }
    }
    
    env = RobustMultiUAVTargetLoadBalanceEnv(config)
    run_random_episode(env, render=True)

def test_target_detection():
    """Test target detection with sensor uncertainty"""
    print("\n=== Testing Target Detection with Sensor Uncertainty ===")
    config = {
        "name": "target_detection",
        "n_agents": 3,
        "grid_size": (10, 10),
        "max_steps": 100,
        "enable_connectivity": True,
        "n_targets": 2,
        "drone_configs": [
            {"drone_type": "sensor_focused"},
            {"drone_type": "standard"},
            {"drone_type": "high_speed"}
        ],
        "target_configs": [
            {"target_type": "static", "priority": 1.0},
            {"target_type": "random_walk", "priority": 0.5, "speed": 0.5}
        ],
        "sensor_config": {
            "detection_range": 1.5,
            "base_reliability": 0.8,
            "noise_std": 0.2,
            "sensor_type": "gaussian"
        },
        "reward_weights": {
            "coverage": 0.3,
            "revisit": -0.1,
            "connectivity": 0.5,
            "target_detection": 1.0,
            "target_relay": 0.5,
            "target_completion": 2.0,
            "early_completion": 1.0,
            "load_balance": 0.3,
            "energy_efficiency": 0.2,
            "collision_penalty": -1.0
        }
    }
    
    env = RobustMultiUAVTargetLoadBalanceEnv(config)
    run_random_episode(env, render=True)

def test_obstacles_and_terrain():
    """Test navigation with obstacles and terrain features"""
    print("\n=== Testing Obstacles and Terrain Features ===")
    config = {
        "name": "obstacles_terrain",
        "n_agents": 3,
        "grid_size": (10, 10),
        "max_steps": 100,
        "enable_connectivity": True,
        "n_targets": 1,
        "drone_configs": [
            {"drone_type": "standard"},
            {"drone_type": "high_speed"},
            {"drone_type": "energy_efficient"}
        ],
        "target_configs": [
            {"target_type": "patrol", "priority": 1.0, "movement_pattern": [[1, 1], [3, 1], [3, 3], [1, 3]]}
        ],
        "terrain_config": {
            "obstacles": [[2, 2]],
            "charging_stations": [[0, 0], [4, 4]],
            "comm_relays": [[2, 3]],
            "interference": {
                "base_level": 0.1,
                "high_interference_zones": [
                    {"center": (3, 3), "radius": 1, "strength": 0.8}
                ],
                "low_interference_corridors": [
                    {"start": (0, 0), "end": (4, 4), "width": 1}
                ]
            }
        },
        "reward_weights": {
            "coverage": 0.3,
            "revisit": -0.1,
            "connectivity": 0.5,
            "target_detection": 1.0,
            "target_relay": 0.5,
            "target_completion": 2.0,
            "early_completion": 1.0,
            "load_balance": 0.3,
            "energy_efficiency": 0.5,
            "collision_penalty": -2.0
        }
    }
    
    env = RobustMultiUAVTargetLoadBalanceEnv(config)
    run_random_episode(env, render=True)

def test_full_features():
    """Test all features combined"""
    print("\n=== Testing All Features Combined ===")
    config = {
        "name": "full_features",
        "n_agents": 5,
        "grid_size": (10, 10),
        "max_steps": 100,
        "enable_connectivity": True,
        "comm_dist": 2.0,
        "n_targets": 3,
        "drone_configs": [
            {"drone_type": "standard"},
            {"drone_type": "long_range"},
            {"drone_type": "high_speed"},
            {"drone_type": "energy_efficient"},
            {"drone_type": "sensor_focused"}
        ],
        "target_configs": [
            {"target_type": "static", "priority": 1.0},
            {"target_type": "random_walk", "priority": 0.7, "speed": 0.5},
            {"target_type": "patrol", "priority": 0.3, "movement_pattern": [[1, 1], [3, 1], [3, 3], [1, 3]]}
        ],
        "terrain_config": {
            "obstacles": [[2, 2], [2, 3]],
            "charging_stations": [[0, 0], [4, 4]],
            "comm_relays": [[2, 4]],
            "interference": {
                "base_level": 0.1,
                "high_interference_zones": [
                    {"center": (3, 3), "radius": 1, "strength": 0.8}
                ],
                "low_interference_corridors": [
                    {"start": (0, 0), "end": (4, 4), "width": 1}
                ]
            },
            "dynamic_obstacles": [
                {
                    "initial_position": [1, 2],
                    "pattern": "patrol",
                    "waypoints": [[1, 2], [1, 4], [3, 4], [3, 2]],
                    "speed": 0.5
                }
            ]
        },
        "sensor_config": {
            "detection_range": 1.5,
            "base_reliability": 0.8,
            "noise_std": 0.2,
            "sensor_type": "gaussian"
        },
        "reward_weights": {
            "coverage": 0.3,
            "revisit": -0.1,
            "connectivity": 0.5,
            "target_detection": 1.0,
            "target_relay": 0.5,
            "target_completion": 2.0,
            "early_completion": 1.0,
            "load_balance": 0.3,
            "energy_efficiency": 0.3,
            "collision_penalty": -2.0
        }
    }
    
    env = RobustMultiUAVTargetLoadBalanceEnv(config)
    run_random_episode(env, render=True)

if __name__ == "__main__":
    print("Starting Environment Tests...")
    
    # Run tests in sequence
    test_single_agent_coverage()
    time.sleep(1)
    
    test_multi_agent_coverage()
    time.sleep(1)
    
    test_connectivity()
    time.sleep(1)
    
    test_target_detection()
    time.sleep(1)
    
    test_obstacles_and_terrain()
    time.sleep(1)
    
    test_full_features()
    
    print("\nAll tests completed!") 