from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
from typing import Dict

from sb3_contrib import RecurrentPPO
from sb3_contrib import TRPO

from path_animator import PathAnimator

from multiAgentEnv import MultiUAVCoverageEnv
import wandb

class CoverageMetricsCallback(BaseCallback):
    """
    Custom callback for monitoring coverage metrics during training
    """
    def __init__(self, eval_env, eval_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.coverage_history = []
        
    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            coverage_metrics = self.evaluate_coverage(n_eval_episodes=1)
            self.coverage_history.append(coverage_metrics)
            
            # Log metrics
            print(f"\nStep {self.n_calls}")
            print(f"Average Coverage: {coverage_metrics['mean_coverage']:.2f}%")
            print(f"Coverage Efficiency: {coverage_metrics['coverage_efficiency']:.2f}")
            print(f"Mean End Time: {coverage_metrics['mean_end_time']:.2f}")
            print(f"Mean Total Reward: {coverage_metrics['mean_total_reward']:.2f}")
            print(f"Connectivity: {coverage_metrics['connectivity']:.2f}")
            # Log to wandb if initialized
            if wandb.run is not None:
                wandb.log({
                    "mean_coverage": coverage_metrics['mean_coverage'],
                    "coverage_efficiency": coverage_metrics['coverage_efficiency'],
                    "training_step": self.n_calls,
                    "mean_end_time": coverage_metrics['mean_end_time'],
                    "mean_total_reward": coverage_metrics['mean_total_reward'],
                    "connectivity": coverage_metrics['connectivity']
                })
                
        return True
        
    def evaluate_coverage(self, n_eval_episodes: int = 5) -> Dict:
        """Evaluate current policy coverage performance"""
        coverage_percentages = []
        coverage_efficiencies = []
        end_times = []
        total_rewards = []
        connectivity_scores = []
        for _ in range(n_eval_episodes):
            obs, info = self.eval_env.reset()
            done = False
            truncated = False
            total_reward = 0
            connectivities = []
            while not done and not truncated:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = self.eval_env.step(action)
                total_reward += reward
                connectivities.append(info['connectivity'])
                
                if truncated or done:
                    coverage_percentages.append(info['coverage_percentage'] * 100)
                    coverage_efficiencies.append(
                        info['unique_cells_covered'] / (info['total_revisits'] + 1)
                    )
                    end_times.append(self.eval_env.timestep)
                    connectivity_scores.append(np.mean(connectivities))
            
            total_rewards.append(total_reward)

            if coverage_percentages[-1] > 99:
                # Example usage
                animator = PathAnimator(
                    paths=self.eval_env.paths,  # Your N*T*2 paths matrix
                    grid_size=(8, 8),
                    animation_duration=10.0,  # 10 seconds animation
                    background_alpha=0.4,
                    comm_range=2*np.sqrt(2) # Same as your environment
                )

                # Or save it to a file
                animator.save('drone_animation.gif', writer='ffmpeg')
            
        return {
            'mean_coverage': np.mean(coverage_percentages),
            'coverage_efficiency': np.mean(coverage_efficiencies),
            'mean_end_time': np.mean(end_times),
            'mean_total_reward': np.mean(total_rewards),
            'connectivity': np.mean(connectivity_scores)
        }

def train_coverage_optimization(
    config: Dict,
    total_timesteps: int = 500_000,
    seed: int = 42
) -> PPO:
    """
    Training loop for coverage optimization with regular evaluation
    """
    # Initialize wandb
    run = wandb.init(
        project="uav-coverage-optimization",
        config={
            "algorithm": "PPO",
            "total_timesteps": total_timesteps,
            **config
        }
    )
    
    # Create environments
    def make_env():
        return MultiUAVCoverageEnv(config)
    
    env = DummyVecEnv([make_env for _ in range(1)])  # 4 parallel environments
    env = VecMonitor(env)
    
    # Create evaluation environment
    eval_env = make_env()

    """policy_kwargs = {
        "net_arch": [dict(pi=[512, 256], vf=[512, 256])],
        "features_extractor_kwargs": {
        },
        "share_features_extractor": True
    }"""
    
    # Initialize PPO model
    model = TRPO(
        "MultiInputPolicy",
        env,
        gamma=0.99,
        n_steps=1000,
        batch_size=1000,

    )
    
    # Setup callback
    coverage_callback = CoverageMetricsCallback(
        eval_env=eval_env,
        eval_freq=1000  # Evaluate every 10k steps
    )
    
    # Train model
    model.learn(
        total_timesteps=total_timesteps,
        callback=coverage_callback,
        progress_bar=True
    )
    
    # Save final model
    model.save(f"models/coverage_optimizer_{run.id}")
    
    # Final evaluation
    final_metrics = coverage_callback.evaluate_coverage(n_eval_episodes=20)
    print("\nFinal Performance:")
    print(f"Coverage: {final_metrics['mean_coverage']:.2f}%")
    print(f"Efficiency: {final_metrics['coverage_efficiency']:.2f}")
    
    wandb.finish()
    return model

if __name__ == "__main__":
    # Configuration
    config = {
        "n_agents": 8,
        "grid_size": (8, 8),
        "max_steps": 50,
        "enable_connectivity": True,
        "comm_dist": 2*np.sqrt(2),
        "reward_weights": {
            "coverage": 2.0,
            "revisit": -0.5,
            "connectivity": 10
        },
        "image_observation": False
    }
    
    # Train model
    model = train_coverage_optimization(config)