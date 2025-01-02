import torch
import torch.optim as optim
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import wandb
from torch.distributions import Categorical
from multiAgentTargetEnv import MultiUAVTargetEnv
from generative_path_planner import Generator, Discriminator, calculate_input_dim
import torch.nn.functional as F

class ExperienceBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, trajectory: Dict[str, torch.Tensor]):
        self.buffer.append(trajectory)
        
    def sample(self, batch_size: int) -> List[Dict[str, torch.Tensor]]:
        indices = np.random.choice(len(self.buffer), min(batch_size, len(self.buffer)), replace=False)
        return [self.buffer[i] for i in indices]
    
    def __len__(self):
        return len(self.buffer)

class MultiAgentSeqGANTrainer:
    def __init__(self, env_config: Dict, gan_config: Dict):
        # Initialize environment
        self.env = MultiUAVTargetEnv(env_config)
        
        # Training configs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gan_config = gan_config
        self.state_dim = calculate_input_dim(self.env)
        
        print(f"State dimension: {self.state_dim}")
        
        # Initialize networks
        self.generator = Generator(
            num_agents=self.env.n_agents,
            action_dim=5,
            hidden_dim=gan_config['hidden_dim'],
            state_dim=self.state_dim
        ).to(self.device)
        
        self.discriminator = Discriminator(
            num_agents=self.env.n_agents,
            action_dim=5,
            hidden_dim=gan_config['hidden_dim'],
            state_dim=self.state_dim,
            max_seq_len=env_config['max_steps']
        ).to(self.device)
        
        # Optimizers
        self.g_optimizer = optim.Adam(
            self.generator.parameters(),
            lr=gan_config['generator_lr']
        )
        self.d_optimizer = optim.Adam(
            self.discriminator.parameters(),
            lr=gan_config['discriminator_lr']
        )
        
        # Experience buffer for trajectory storage
        self.buffer = ExperienceBuffer(gan_config['buffer_size'])
        
        # Initialize logging
        if gan_config.get('use_wandb', False):
            wandb.init(project="multi-uav-seqgan", config={**env_config, **gan_config})
            
    def state_to_tensor(self, state: Dict) -> torch.Tensor:
        """Convert environment state dict to tensor representation"""
        # Normalize grid coordinates
        grid_size = max(self.env.G1, self.env.G2)
        
        # Process visited cells
        visited = state['visited_cells'].astype(np.float32)
        visited = np.minimum(visited, 1.0)  # Binary occupancy
        
        # Normalize agent positions
        agent_pos = state['agent_positions'].astype(np.float32) / grid_size
        
        # Process target information
        target_pos = state['target_positions'].astype(np.float32) / grid_size
        detected = state['detected_targets'].astype(np.float32)
        completion = state['target_completion'].astype(np.float32)
        
        # Concatenate all components
        state_components = [
            visited.flatten(),
            agent_pos.flatten(),
            target_pos.flatten(),
            detected,
            completion
        ]
        
        state_tensor = np.concatenate([comp.flatten() for comp in state_components])
        return torch.FloatTensor(state_tensor).to(self.device)
    
    def collect_trajectory(self, temperature: float = 1.0) -> Dict[str, torch.Tensor]:
        """Generate a single trajectory using current generator policy"""
        state = self.env.reset()[0]  # Get initial state
        done = False
        truncated = False
        
        states, actions, rewards = [], [], []
        episode_reward = 0
        
        while not (done or truncated):
            # Convert state to tensor
            state_tensor = self.state_to_tensor(state).unsqueeze(0)
            
            # Get action probabilities from generator
            with torch.no_grad():
                action_probs = self.generator(state_tensor, temperature)
            
            # Sample actions for each agent
            actions_list = []
            log_probs_list = []
            
            for agent_idx in range(self.env.n_agents):
                dist = Categorical(action_probs[:, agent_idx])
                action = dist.sample()
                actions_list.append(action.item())
                log_probs_list.append(dist.log_prob(action))
            
            # Execute action in environment
            next_state, reward, done, truncated, info = self.env.step(np.array(actions_list))
            
            # Store step information
            states.append(state_tensor)
            actions.append(torch.tensor(actions_list, device=self.device))
            rewards.append(reward)
            
            state = next_state
            episode_reward += reward
            
        trajectory = {
            'states': torch.cat(states),
            'actions': torch.stack(actions),
            'rewards': torch.tensor(rewards, device=self.device),
            'episode_reward': episode_reward
        }
        
        # Add additional metrics
        trajectory.update({
            'coverage': info['coverage_percentage'],
            'connectivity': info['connectivity'],
            'detected_targets': info['detected_targets'].sum(),
            'completed_targets': info['completed_targets'].sum()
        })
        
        return trajectory

    def pretrain_generator(self, num_episodes: int = 1000):
        """Pretrain generator using simple heuristic or expert demonstrations"""
        print("Pretraining generator...")
        running_reward = deque(maxlen=100)
        metrics = {
            'coverage': deque(maxlen=100),
            'connectivity': deque(maxlen=100),
            'detected_targets': deque(maxlen=100),
            'completed_targets': deque(maxlen=100)
        }
        
        for episode in range(num_episodes):
            trajectory = self.collect_trajectory(temperature=1.0)
            self.buffer.push(trajectory)
            
            # Update metrics
            running_reward.append(trajectory['episode_reward'])
            for key in metrics:
                metrics[key].append(trajectory[key])
            
            if episode % 100 == 0:
                avg_reward = sum(running_reward) / len(running_reward)
                print(f"Pretraining episode {episode}, Avg reward: {avg_reward:.2f}")
                
                if self.gan_config.get('use_wandb', False):
                    wandb.log({
                        'pretrain_episode': episode,
                        'pretrain_avg_reward': avg_reward,
                        'pretrain_avg_coverage': sum(metrics['coverage']) / len(metrics['coverage']),
                        'pretrain_avg_connectivity': sum(metrics['connectivity']) / len(metrics['connectivity']),
                        'pretrain_avg_detected': sum(metrics['detected_targets']) / len(metrics['detected_targets']),
                        'pretrain_avg_completed': sum(metrics['completed_targets']) / len(metrics['completed_targets'])
                    })

    def train_discriminator(self, real_trajectories: List[Dict], fake_trajectories: List[Dict]) -> float:
        """Train discriminator on real and generated trajectories"""
        self.d_optimizer.zero_grad()
        
        # Process real trajectories
        real_states = torch.cat([t['states'] for t in real_trajectories])
        real_actions = torch.cat([t['actions'] for t in real_trajectories])
        real_traj = torch.cat([real_states, 
                             F.one_hot(real_actions.long(), 5).float().view(real_states.size(0), -1)], 
                             dim=-1)
        real_predictions = self.discriminator(real_traj)
        
        # Process fake trajectories
        fake_states = torch.cat([t['states'] for t in fake_trajectories])
        fake_actions = torch.cat([t['actions'] for t in fake_trajectories])
        fake_traj = torch.cat([fake_states, 
                             F.one_hot(fake_actions.long(), 5).float().view(fake_states.size(0), -1)], 
                             dim=-1)
        fake_predictions = self.discriminator(fake_traj)
        
        # Compute discriminator loss with label smoothing
        real_labels = torch.ones_like(real_predictions) * 0.9  # Label smoothing
        fake_labels = torch.zeros_like(fake_predictions)
        
        d_loss = -(torch.log(real_predictions + 1e-8) * real_labels).mean() - \
                 (torch.log(1 - fake_predictions + 1e-8) * fake_labels).mean()
        
        d_loss.backward()
        self.d_optimizer.step()
        
        return d_loss.item()

    def train_generator(self, trajectories: List[Dict]) -> float:
        """Train generator using policy gradient with discriminator rewards"""
        self.g_optimizer.zero_grad()
        
        # Process trajectories
        states = torch.cat([t['states'] for t in trajectories])
        actions = torch.cat([t['actions'] for t in trajectories])
        
        # Get discriminator scores
        traj = torch.cat([states, 
                         F.one_hot(actions.long(), 5).float().view(states.size(0), -1)], 
                         dim=-1)
        
        with torch.no_grad():
            d_rewards = self.discriminator(traj)
        
        # Generate action probabilities
        action_probs = self.generator(states)
        
        # Compute log probabilities for each agent's actions
        log_probs = []
        for i in range(self.env.n_agents):
            dist = Categorical(action_probs[:, i])
            log_prob = dist.log_prob(actions[:, i])
            log_probs.append(log_prob)
        
        log_probs = torch.stack(log_probs, dim=1)
        
        # Compute policy gradient loss with baseline
        baseline = d_rewards.mean()
        advantages = d_rewards - baseline
        
        g_loss = -(log_probs * advantages.unsqueeze(-1)).mean()
        
        g_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 1.0)
        self.g_optimizer.step()
        
        return g_loss.item()

    def train(self, num_episodes: int = 10000):
        """Main training loop"""
        # Pretrain generator
        self.pretrain_generator(num_episodes=0)
        
        print("Starting adversarial training...")
        running_reward = deque(maxlen=100)
        best_reward = float('-inf')
        metrics = {
            'coverage': deque(maxlen=100),
            'connectivity': deque(maxlen=100),
            'detected_targets': deque(maxlen=100),
            'completed_targets': deque(maxlen=100)
        }
        
        for episode in range(num_episodes):
            # Generate trajectories with annealed temperature
            temperature = max(0.5, 1.0 - episode / num_episodes)
            trajectory = self.collect_trajectory(temperature)
            self.buffer.push(trajectory)
            
            # Update metrics
            running_reward.append(trajectory['episode_reward'])
            for key in metrics:
                metrics[key].append(trajectory[key])
            
            # Train networks
            if len(self.buffer) >= self.gan_config['batch_size']:
                real_trajectories = self.buffer.sample(self.gan_config['batch_size'])
                fake_trajectories = [self.collect_trajectory(temperature) 
                                  for _ in range(self.gan_config['batch_size'])]
                
                d_loss = self.train_discriminator(real_trajectories, fake_trajectories)
                g_loss = self.train_generator(fake_trajectories)
                
                # Logging
                if episode % 100 == 0:
                    avg_reward = sum(running_reward) / len(running_reward)
                    avg_metrics = {
                        key: sum(metrics[key]) / len(metrics[key])
                        for key in metrics
                    }
                    
                    print(f"\nEpisode {episode}")
                    print(f"Avg reward: {avg_reward:.2f}")
                    print(f"D loss: {d_loss:.4f}, G loss: {g_loss:.4f}")
                    print(f"Coverage: {avg_metrics['coverage']:.2f}")
                    print(f"Connectivity: {avg_metrics['connectivity']:.2f}")
                    print(f"Detected targets: {avg_metrics['detected_targets']:.1f}")
                    print(f"Completed targets: {avg_metrics['completed_targets']:.1f}")
                    
                    if self.gan_config.get('use_wandb', False):
                        wandb.log({
                            'episode': episode,
                            'avg_reward': avg_reward,
                            'discriminator_loss': d_loss,
                            'generator_loss': g_loss,
                            'temperature': temperature,
                            'avg_coverage': avg_metrics['coverage'],
                            'avg_connectivity': avg_metrics['connectivity'],
                            'avg_detected_targets': avg_metrics['detected_targets'],
                            'avg_completed_targets': avg_metrics['completed_targets']
                        })
                    
                    # Save best model
                    if avg_reward > best_reward:
                        best_reward = avg_reward
                        torch.save({
                            'generator_state_dict': self.generator.state_dict(),
                            'discriminator_state_dict': self.discriminator.state_dict(),
                            'config': self.gan_config,
                            'metrics': {
                                'reward': avg_reward,
                                **avg_metrics
                            }
                        }, 'best_model.pt')

if __name__ == "__main__":
    # Environment configuration
    env_config = {
        "n_agents": 1,
        "grid_size": (8, 8),
        "max_steps": 100,
        "comm_dist": 2.8284,
        "gcs_pos": (0, 0),
        "enable_connectivity": False,
        "n_targets": 0,
        "target_types": [2, 2, 2],
        "reward_weights": {
            "coverage": 1.0,
            "revisit": -0.1,
            "connectivity": 0.5,
            "target_detection": 2.0,
            "target_relay": 1.5,
            "target_completion": 5.0
        }
    }
    
    # GAN configuration
    gan_config = {
        "hidden_dim": 128,
        "generator_lr": 1e-4,
        "discriminator_lr": 1e-4,
        "buffer_size": 100,
        "batch_size": 32,
        "use_wandb": True  # Set to False if not using W&B
    }
    
    # Initialize and train
    trainer = MultiAgentSeqGANTrainer(env_config, gan_config)
    trainer.train()