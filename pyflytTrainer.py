from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
from typing import Dict
import wandb
from pyflytEnv import RobustMultiUAVTargetEnv

class PyFlytMetricsCallback(BaseCallback):
    """
    Callback for monitoring PyFlyt UAV mission metrics
    """
    def __init__(self, eval_env, eval_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.metrics_history = []
        
    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            metrics = self.evaluate_mission(n_eval_episodes=5)
            self.metrics_history.append(metrics)
            
            # Log metrics
            print(f"\nStep {self.n_calls}")
            print(f"Coverage Rate: {metrics['mean_coverage']:.2f}%")
            print(f"Target Detection Rate: {metrics['detection_rate']:.2f}%")
            print(f"Connectivity Score: {metrics['connectivity']:.2f}")
            
            # Log to wandb if initialized
            if wandb.run is not None:
                wandb.log({
                    "mean_coverage": metrics['mean_coverage'],
                    "detection_rate": metrics['detection_rate'],
                    "connectivity": metrics['connectivity'],
                    "mean_reward": metrics['mean_reward'],
                    "training_step": self.n_calls
                })
                
        return True

    def evaluate_mission(self, n_eval_episodes: int = 5) -> Dict:
        """Evaluate current policy performance"""
        coverage_rates = []
        detection_rates = []
        connectivity_scores = []
        episode_rewards = []
        
        for _ in range(n_eval_episodes):
            obs, info = self.eval_env.reset()
            done = False
            truncated = False
            episode_reward = 0
            
            while not done and not truncated:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = self.eval_env.step(action)
                episode_reward += reward
                
                if done or truncated:
                    coverage_rates.append(info['coverage'] * 100)
                    connectivity_scores.append(info['connectivity'])
                    episode_rewards.append(episode_reward)
                    
                    # Calculate target detection rate
                    target_positions = obs['target_positions']
                    drone_positions = obs['position']
                    detection_range = 2.0  # Detection radius in meters
                    
                    detected = 0
                    for target_pos in target_positions:
                        for drone_pos in drone_positions:
                            if np.linalg.norm(target_pos - drone_pos[:2]) < detection_range:
                                detected += 1
                                break
                    detection_rate = 100 * detected / len(target_positions)
                    detection_rates.append(detection_rate)
        
        return {
            'mean_coverage': np.mean(coverage_rates),
            'detection_rate': np.mean(detection_rates),
            'connectivity': np.mean(connectivity_scores),
            'mean_reward': np.mean(episode_rewards)
        }

def train_pyflyt_mission(
    config: Dict,
    total_timesteps: int = 1_000_000,
    seed: int = 42
) -> PPO:
    """
    Training loop for PyFlyt UAV missions using PPO
    """
    # Initialize wandb
    run = wandb.init(
        project="pyflyt-uav-mission",
        config={
            "algorithm": "PPO",
            "total_timesteps": total_timesteps,
            **config
        }
    )
    
    # Environment setup
    def make_env():
        return RobustMultiUAVTargetEnv(config)
    
    env = DummyVecEnv([make_env for _ in range(1)])
    env = VecMonitor(env)
    eval_env = make_env()
    
    # PPO hyperparameters tuned for continuous control
    model = PPO(
        "MultiInputPolicy",
        env,
        gamma=0.99,
        n_steps=2048,
        batch_size=64,
        learning_rate=3e-4,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        seed=seed
    )
    
    # Setup callback
    metrics_callback = PyFlytMetricsCallback(
        eval_env=eval_env,
        eval_freq=5000
    )
    
    # Train model
    model.learn(
        total_timesteps=total_timesteps,
        callback=metrics_callback,
        progress_bar=True
    )
    
    # Save final model
    model.save(f"models/pyflyt_mission_{run.id}")
    
    # Final evaluation
    final_metrics = metrics_callback.evaluate_mission(n_eval_episodes=20)
    print("\nFinal Performance:")
    print(f"Coverage Rate: {final_metrics['mean_coverage']:.2f}%")
    print(f"Target Detection Rate: {final_metrics['detection_rate']:.2f}%")
    print(f"Connectivity Score: {final_metrics['connectivity']:.2f}")
    
    wandb.finish()
    return model

if __name__ == "__main__":
    # Configuration for PyFlyt UAV mission
    config = {
        "n_agents": 3,
        "grid_size": (20, 20),
        "n_targets": 5,
        "target_types": [0, 1],  # Mix of stationary and mobile targets
        "enable_connectivity": True,
        "comm_radius": 5.0,
        "max_steps": 500,
        "reward_weights": {
            "coverage": 1.0,
            "target": 2.0,
            "connectivity": 0.5
        }
    }
    
    # Train model
    model = train_pyflyt_mission(config)
