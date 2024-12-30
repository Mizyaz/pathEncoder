from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
from typing import Dict, List
import wandb
from path_animator import PathAnimator
from multiAgentTargetPriorityLoadBalanceEnv import RobustMultiUAVTargetLoadBalanceEnv
from sb3_contrib import TRPO
from collections import defaultdict

ANIMATE = False

class LoadBalanceCallback(BaseCallback):
    """
    Enhanced callback for monitoring heterogeneous drone performance,
    energy efficiency, and load balancing metrics
    """
    def __init__(self, eval_env, eval_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.metrics_history = []
        
    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            metrics = self.evaluate_mission(n_eval_episodes=100)
            self.metrics_history.append(metrics)
            
            # Log metrics
            print(f"\nStep {self.n_calls}")
            print(f"Target Detection Rate: {metrics['detection_rate']:.2f}%")
            print(f"Target Completion Rate: {metrics['completion_rate']:.2f}%")
            print(f"Mean Energy Level: {metrics['mean_energy']:.2f}%")
            print(f"Coverage: {metrics['mean_coverage']:.2f}%")
            print(f"Connectivity: {metrics['connectivity']:.2f}")
            print(f"Load Balance Score: {metrics['load_balance_score']:.2f}")
            print("\nDrone Type Performance:")
            for drone_type in metrics['drone_type_stats']:
                stats = metrics['drone_type_stats'][drone_type]
                print(f"{drone_type}:")
                print(f"  Detection Rate: {stats['detection_rate']:.2f}%")
                print(f"  Energy Efficiency: {stats['energy_efficiency']:.2f}")
                print(f"  Comm Load: {stats['comm_load']:.2f}")
            
            # Log to wandb if initialized
            if wandb.run is not None:
                wandb_metrics = {
                    "detection_rate": metrics['detection_rate'],
                    "completion_rate": metrics['completion_rate'],
                    "mean_energy": metrics['mean_energy'],
                    "mean_coverage": metrics['mean_coverage'],
                    "connectivity": metrics['connectivity'],
                    "load_balance_score": metrics['load_balance_score'],
                    "collision_rate": metrics['collision_rate'],
                    "training_step": self.n_calls,
                    "mean_total_reward": metrics['mean_total_reward']
                }
                
                # Add drone type specific metrics
                for drone_type, stats in metrics['drone_type_stats'].items():
                    for key, value in stats.items():
                        wandb_metrics[f"{drone_type}_{key}"] = value
                        
                wandb.log(wandb_metrics)
                
        return True
        
    def evaluate_mission(self, n_eval_episodes: int = 5) -> Dict:
        """Evaluate current policy performance with enhanced metrics"""
        episode_metrics = defaultdict(list)
        drone_type_metrics = defaultdict(lambda: defaultdict(list))
        
        for _ in range(n_eval_episodes):
            obs, info = self.eval_env.reset()
            done = False
            truncated = False
            episode_reward = 0
            collisions = 0
            
            # Track metrics per step
            step_metrics = {
                'connectivities': [],
                'energy_levels': [],
                'detections': defaultdict(set),
                'completions': defaultdict(set),
                'comm_loads': defaultdict(int),
            }
            
            while not done and not truncated:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = self.eval_env.step(action)
                episode_reward += reward
                
                # Track basic metrics
                step_metrics['connectivities'].append(info['connectivity'])
                step_metrics['energy_levels'].append(np.mean(info['energy_levels']))
                
                # Track collisions
                collisions += np.sum([1 for drone in info['drone_states'] if not drone.get('last_move_success', True)])
                
                # Track detections and completions per drone type
                for i, drone in enumerate(info['drone_states']):
                    drone_type = drone['drone_type']
                    # Use target_info from knowledge base instead of target_detections
                    for target_idx in range(self.eval_env.n_targets):
                        if info['detected_targets'][target_idx]:
                            step_metrics['detections'][drone_type].add(target_idx)
                        if info['completed_targets'][target_idx]:
                            step_metrics['completions'][drone_type].add(target_idx)
                    
                    # Track communication load
                    step_metrics['comm_loads'][drone_type] += np.sum(info['communication_matrix'][i])
            
            # Calculate episode metrics
            episode_metrics['total_rewards'].append(episode_reward)
            episode_metrics['coverage'].append(info['coverage_percentage'] * 100)
            episode_metrics['connectivity'].append(np.mean(step_metrics['connectivities']))
            episode_metrics['energy'].append(np.mean(step_metrics['energy_levels']) * 100)
            episode_metrics['collision_rate'].append(collisions / self.eval_env.max_steps)
            
            # Calculate load balance score
            if np.sum(info['communication_matrix']) > 0:
                load_distribution = info['communication_matrix'] / np.sum(info['communication_matrix'])
                episode_metrics['load_balance'].append(-np.std(load_distribution))
            
            # Calculate per-drone-type metrics
            for drone_type in set(drone['drone_type'] for drone in info['drone_states']):
                type_drones = [i for i, d in enumerate(info['drone_states']) if d['drone_type'] == drone_type]
                n_type_drones = len(type_drones)
                if n_type_drones > 0:
                    # Detection rate for this type
                    detections = len(step_metrics['detections'][drone_type])
                    drone_type_metrics[drone_type]['detection_rate'].append(100 * detections / self.eval_env.n_targets if self.eval_env.n_targets > 0 else 100)
                    
                    # Energy efficiency
                    type_energy = np.mean([info['energy_levels'][i] for i in type_drones])
                    drone_type_metrics[drone_type]['energy_efficiency'].append(type_energy)
                    
                    # Communication load
                    type_comm_load = step_metrics['comm_loads'][drone_type] / n_type_drones
                    drone_type_metrics[drone_type]['comm_load'].append(type_comm_load)
            
            # Generate animation for successful episodes
            if info['coverage_percentage'] > 0.95 and np.mean(step_metrics['energy_levels']) > 0.3 and ANIMATE:
                animator = PathAnimator(
                    paths=[drone.get('path', []) for drone in info['drone_states']],
                    grid_size=self.eval_env.config["grid_size"],
                    animation_duration=10.0,
                    background_alpha=0.4,
                    comm_range=self.eval_env.config["comm_dist"],
                    target_positions=[target.get('position', [0, 0]) for target in info['target_states']],
                    terrain_map=info['terrain_state']['terrain_map']
                )
                animator.save(f'load_balance_animation_{self.n_calls}.gif', writer='ffmpeg')
        
        # Compute final metrics
        metrics = {
            'detection_rate': np.mean([len(step_metrics['detections'][drone_type]) for drone_type in step_metrics['detections']]) / self.eval_env.n_targets * 100,
            'completion_rate': np.mean([len(step_metrics['completions'][drone_type]) for drone_type in step_metrics['completions']]) / self.eval_env.n_targets * 100,
            'mean_coverage': np.mean(episode_metrics['coverage']),
            'mean_energy': np.mean(episode_metrics['energy']),
            'connectivity': np.mean(episode_metrics['connectivity']),
            'load_balance_score': np.mean(episode_metrics['load_balance']),
            'collision_rate': np.mean(episode_metrics['collision_rate']),
            'mean_total_reward': np.mean(episode_metrics['total_rewards']),
            'drone_type_stats': {
                drone_type: {
                    'detection_rate': np.mean(stats['detection_rate']),
                    'energy_efficiency': np.mean(stats['energy_efficiency']),
                    'comm_load': np.mean(stats['comm_load'])
                }
                for drone_type, stats in drone_type_metrics.items()
            }
        }
        
        return metrics

def train_load_balance_mission(
    config: Dict,
    total_timesteps: int = 1_000_000,
    seed: int = 42
) -> PPO:
    """
    Training loop for load-balanced heterogeneous drone missions
    """
    # Initialize wandb
    run = wandb.init(
        project="multi-uav-load-balance-mission",
        config={
            "algorithm": "PPO",
            "total_timesteps": total_timesteps,
            **config
        }
    )
    
    # Environment setup
    def make_env():
        return RobustMultiUAVTargetLoadBalanceEnv(config)
    
    env = DummyVecEnv([make_env for _ in range(1)])
    env = VecMonitor(env)
    eval_env = make_env()
    
    # PPO model with tuned parameters for heterogeneous agents
    model = PPO(
        "MultiInputPolicy",
        env,
        gamma=0.99,
        gae_lambda=0.95,
        n_steps=2048,
        batch_size=64,
        learning_rate=3e-4,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=0.02,  # Adaptive KL divergence
        verbose=1
    )
    
    # Setup callback
    mission_callback = LoadBalanceCallback(
        eval_env=eval_env,
        eval_freq=5000
    )
    
    # Train model
    model.learn(
        total_timesteps=total_timesteps,
        callback=mission_callback,
        progress_bar=True
    )
    
    # Save final model
    model.save(f"models/load_balance_{config['name']}_{run.id}")
    
    # Final evaluation
    final_metrics = mission_callback.evaluate_mission(n_eval_episodes=20)
    print("\nFinal Performance:")
    print(f"Target Detection Rate: {final_metrics['detection_rate']:.2f}%")
    print(f"Target Completion Rate: {final_metrics['completion_rate']:.2f}%")
    print(f"Mean Energy Level: {final_metrics['mean_energy']:.2f}%")
    print(f"Coverage: {final_metrics['mean_coverage']:.2f}%")
    print(f"Load Balance Score: {final_metrics['load_balance_score']:.2f}")
    print("\nDrone Type Performance:")
    for drone_type, stats in final_metrics['drone_type_stats'].items():
        print(f"{drone_type}:")
        print(f"  Detection Rate: {stats['detection_rate']:.2f}%")
        print(f"  Energy Efficiency: {stats['energy_efficiency']:.2f}")
        print(f"  Comm Load: {stats['comm_load']:.2f}")
    
    wandb.finish()
    return model

if __name__ == "__main__":
    # Configuration for load-balanced heterogeneous drone mission
    config = {
        "name": "load_balance_mission",
        "n_agents": 1,
        "grid_size": (4, 4),
        "max_steps": 20,
        "comm_dist": 2.8284,
        "enable_connectivity": False,
        "n_targets": 0,
        "drone_configs": [
            {"drone_type": "standard"},
        ],
        #"target_configs": [
        #    {"target_type": "static", "priority": 1.0},
        #    {"target_type": "random_walk", "priority": 0.7, "speed": 0.5},
        #    {"target_type": "patrol", "priority": 0.3, "movement_pattern": [[1, 1], [8, 1], [8, 8], [1, 8]]}
        #],
        #"terrain_config": {
        #    "obstacles": [[2, 2], [2, 3], [3, 2], [7, 7]],
        #    "charging_stations": [[0, 0], [9, 9]],
        #    "comm_relays": [[5, 5]],
        #    "interference": {
        #        "base_level": 0.1,
        #        "high_interference_zones": [
        #            {"center": (3, 3), "radius": 2, "strength": 0.8}
        #        ],
        #        "low_interference_corridors": [
        #            {"start": (0, 0), "end": (9, 9), "width": 1}
        #        ]
        #    }
        #},
        #"sensor_config": {
        #    "detection_range": 1.0,
        #    "base_reliability": 0.8,
        #    "noise_std": 0.1,
        #    "sensor_type": "deterministic"
        #},
        "obs_config": {
            "known_target_positions": False,
            "known_target_priorities": False,
            "known_time_slots": False
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

    train_load_balance_mission(config, total_timesteps=1_000_000) 