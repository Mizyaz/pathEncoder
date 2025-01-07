from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
from typing import Dict, List
import wandb
from path_animator import PathAnimator
from multiAgentTargetEnv import MultiUAVTargetEnv as MultiUAVTargetEnv
import pickle

from sb3_contrib import TRPO

class TargetMissionMetricsCallback(BaseCallback):
    """
    Enhanced callback for monitoring both coverage and target-related metrics
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
            print(f"Mean Inform Time: {metrics['mean_inform_time']:.2f}")
            print(f"Coverage: {metrics['mean_coverage']:.2f}%")
            print(f"Connectivity: {metrics['connectivity']:.2f}")
            
            # Log to wandb if initialized
            if wandb.run is not None:
                wandb.log({
                    "detection_rate": metrics['detection_rate'],
                    "completion_rate": metrics['completion_rate'],
                    "mean_inform_time": metrics['mean_inform_time'],
                    "mean_coverage": metrics['mean_coverage'],
                    "connectivity": metrics['connectivity'],
                    "training_step": self.n_calls,
                    "mean_total_reward": metrics['mean_total_reward']
                })
                
        return True
        
    def evaluate_mission(self, n_eval_episodes: int = 5) -> Dict:
        """Evaluate current policy performance on target search and connect mission"""
        detection_rates = []
        completion_rates = []
        inform_times = []
        coverage_rates = []
        connectivity_scores = []
        total_rewards = []
        
        for _ in range(n_eval_episodes):
            obs, info = self.eval_env.reset()
            done = False
            truncated = False
            episode_reward = 0
            connectivities = []
            first_detection_times = {}
            completion_times = {}
            
            while not done and not truncated:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = self.eval_env.step(action)
                episode_reward += reward
                connectivities.append(info['connectivity'])
                
                # Track detection and completion times
                for i in range(len(info['detected_targets'])):
                    if info['detected_targets'][i] and i not in first_detection_times:
                        first_detection_times[i] = self.eval_env.timestep
                    if info['completed_targets'][i] and i not in completion_times:
                        completion_times[i] = self.eval_env.timestep
                
                if done or truncated:
                    n_targets = len(info['detected_targets'])
                    if n_targets > 0:
                        detection_rate = 100 * len(first_detection_times) / n_targets
                        completion_rate = 100 * len(completion_times) / n_targets
                        mean_inform_time = np.mean(list(completion_times.values())) if completion_times else self.eval_env.max_steps
                    else:
                        detection_rate = 100
                        completion_rate = 100
                        mean_inform_time = 0
                        
                    detection_rates.append(detection_rate)
                    completion_rates.append(completion_rate)
                    inform_times.append(mean_inform_time)
                    coverage_rates.append(info['coverage_percentage'] * 100)
                    connectivity_scores.append(np.mean(connectivities))
                    total_rewards.append(episode_reward)
            
            # Generate animation for successful episodes
            if completion_rate > 95:
                """animator = PathAnimator(
                    paths=self.eval_env.paths,
                    grid_size=self.eval_env.config["grid_size"],
                    animation_duration=10.0,
                    background_alpha=0.4,
                    comm_range=self.eval_env.config["comm_dist"],
                    target_positions=self.eval_env.target_positions if hasattr(self.eval_env, 'target_positions') else None
                )
                animator.save(f'mission_animation_{self.n_calls}.gif', writer='ffmpeg')"""
                pass
        
        return {
            'detection_rate': np.mean(detection_rates),
            'completion_rate': np.mean(completion_rates),
            'mean_inform_time': np.mean(inform_times),
            'mean_coverage': np.mean(coverage_rates),
            'connectivity': np.mean(connectivity_scores),
            'mean_total_reward': np.mean(total_rewards)
        }

def compute_effective_horizon(n_steps: int, gamma: float = None, gae_lambda: float = None) -> float:
    """
    Compute effective horizon based on discount factor and GAE lambda
    Returns the missing parameter (gamma or lambda) given the other and desired horizon
    """
    if gamma is not None and gae_lambda is None:
        # Solve for lambda given gamma and horizon
        return (1 - 1/n_steps) / gamma
    elif gae_lambda is not None and gamma is None:
        # Solve for gamma given lambda and horizon
        return (1 - 1/n_steps) / gae_lambda
    else:
        raise ValueError("Provide exactly one of gamma or gae_lambda")

def train_target_mission(
    config: Dict,
    total_timesteps: int = 1_000_000,
    seed: int = 42,
    name: str = "multi-uav-target-mission"
) -> PPO:
    """
    Enhanced training loop for target search and connect missions with PPO
    """
    # Initialize wandb
    run = wandb.init(
        project="multi-uav-target-mission",
        id=name,
        config={
            "algorithm": "PPO",
            "total_timesteps": total_timesteps,
            **config
        }
    )
    
    # Environment setup
    def make_env():
        return MultiUAVTargetEnv(config)
    
    
    env = DummyVecEnv([make_env for _ in range(1)])
    env = VecMonitor(env)
    eval_env = make_env()
    print("Trying to save env")
    env.envs[0].save("multi-uav-target-env.pkl")
    print("Env saved")
    
    
    # Compute effective horizon and learning parameters
    n_steps = 2048  # PPO default
    gamma = 0.88  # From paper for inform phase
    gae_lambda = compute_effective_horizon(n_steps, gamma=gamma)
    
    # PPO model with paper-specified parameters
    model = PPO(
        "MultiInputPolicy",
        env,
        gamma=gamma,
        gae_lambda=gae_lambda,
        n_steps=n_steps,
        batch_size=64,
        learning_rate=3e-5,  # Paper specified
        ent_coef=0.01,      # For exploration
        vf_coef=0.5,        # From paper
        max_grad_norm=0.5,
        verbose=1
    )

    model = TRPO(
        "MultiInputPolicy",
        env,
        gamma=gamma,
    )
    
    # Setup callback
    mission_callback = TargetMissionMetricsCallback(
        eval_env=eval_env,
        eval_freq=5000  # Evaluate every 5k steps
    )
    
    # Train model
    model.learn(
        total_timesteps=total_timesteps,
        callback=mission_callback,
        progress_bar=True
    )
    
    # Save final model
    model.save(f"models/target_mission_{run.id}")
    
    # Final evaluation
    final_metrics = mission_callback.evaluate_mission(n_eval_episodes=20)
    print("\nFinal Performance:")
    print(f"Target Detection Rate: {final_metrics['detection_rate']:.2f}%")
    print(f"Target Completion Rate: {final_metrics['completion_rate']:.2f}%")
    print(f"Mean Inform Time: {final_metrics['mean_inform_time']:.2f}")
    print(f"Coverage: {final_metrics['mean_coverage']:.2f}%")
    
    wandb.finish()
    return model, env

if __name__ == "__main__":
    # Configuration for heterogeneous target mission
    config = {
        "n_agents": 8,
        "grid_size": (8, 8),
        "max_steps": 20,
        "comm_dist": 2*np.sqrt(2),
        "enable_connectivity": True,
        "n_targets": 3,
        "target_types": [2, 2, 2],  # One of each type
        "reward_weights": {
            "coverage": 0,
            "revisit": 0,
            "connectivity": 0.5,
            "target_detection": 2.0,
            "target_relay": 1.5,
            "target_completion": 5.0
        },
        "image_observation": False
    }
    
    # Train model
    model, env = train_target_mission(config, 
                                      name=f"multi-uav-target-mission-{config['n_targets']}-targets-{config['n_agents']}-agents-{config['grid_size']}-grid-size",
                                      total_timesteps=1_000_000)

    # Save the environment
    env.envs[0].save("multi-uav-target-env.pkl")

    reward_info = env.envs[0].reward_info

    with open("reward_info.pkl", "wb") as f:
        pickle.dump(reward_info, f)