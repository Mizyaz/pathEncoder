import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import numpy as np
from PyFlyt.core import Aviary

class DroneGridEnv(gym.Env):
    def __init__(self):
        super().__init__()
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(4)  # 0: right, 1: left, 2: up, 3: down
        self.observation_space = spaces.Box(
            low=np.array([0, 0]), 
            high=np.array([7, 7]), 
            dtype=np.float32
        )
        
        # Grid parameters
        self.grid_size = 8
        self.current_pos = np.array([0, 0])
        self.target_height = 1.0
        
        # Initialize PyFlyt environment
        start_pos = np.array([[0.0, 0.0, 0.0]])
        start_orn = np.array([[0.0, 0.0, 0.0]])
        self.env = Aviary(
            start_pos=start_pos, 
            start_orn=start_orn, 
            render=True, 
            drone_type="quadx"
        )
        self.env.set_mode(7)  # Position control mode
        
        # Initialize trajectory for snake pattern
        self.snake_trajectory = self._create_snake_trajectory()
        self.trajectory_index = 0
        
    def _create_snake_trajectory(self):
        """Creates a snake-like trajectory across the 8x8 grid"""
        trajectory = []
        for y in range(self.grid_size):
            row = range(self.grid_size) if y % 2 == 0 else range(self.grid_size-1, -1, -1)
            for x in row:
                trajectory.append([x, y])
        return trajectory
    
    def reset(self, seed=None):
        super().reset(seed=seed)
        # Reset drone position to origin
        self.current_pos = np.array([0, 0])
        self.trajectory_index = 0
        self.env.reset()
        
        return self.current_pos, {}
    
    def step(self, action):
        # Get next position from snake trajectory
        if self.trajectory_index < len(self.snake_trajectory) - 1:
            self.trajectory_index += 1
            next_pos = self.snake_trajectory[self.trajectory_index]
            print(next_pos)
            
            # Convert grid coordinates to PyFlyt coordinates
            setpoint = np.array([
                float(next_pos[0]),
                float(next_pos[1]),
                0.0,  # yaw
                1.0
            ])
            
            # Command drone to move to next position
            self.env.set_setpoint(0, setpoint)

            logs = []
            
            # Simulate several steps to allow movement
            for _ in range(2):  # Adjust this value based on movement speed needed
                self.env.step()
                logs.append(self.env.state(0)[-1])

            
            self.current_pos = np.array(next_pos)
            
            # Check if we've reached the end
            done = self.trajectory_index == len(self.snake_trajectory) - 1
            reward = 1.0 if done else 0.0
            
            return self.current_pos, reward, done, False, {}
        else:
            return self.current_pos, 0.0, True, False, {}
    
    def render(self):
        # PyFlyt handles rendering internally
        pass
    
    def close(self):
        self.env.close()

# Example usage
if __name__ == "__main__":
    env = DroneGridEnv()
    obs, _ = env.reset()
    
    done = False
    trajectory = [obs.copy()]
    
    while not done:
        action = env.action_space.sample()  # Random action (not really used in snake pattern)
        obs, reward, done, _, _ = env.step(action)
        trajectory.append(obs.copy())
    
    # Plot the trajectory
    trajectory = np.array(trajectory)
    plt.figure(figsize=(10, 10))
    plt.plot(trajectory[:, 0], trajectory[:, 1], 'b-', label='Drone path')
    plt.plot(trajectory[:, 0], trajectory[:, 1], 'r.', label='Grid points')
    plt.grid(True)
    plt.xlim(-0.5, 7.5)
    plt.ylim(-0.5, 7.5)
    plt.title('Drone Snake Pattern Trajectory')
    plt.legend()
    plt.show()
