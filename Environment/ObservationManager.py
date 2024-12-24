import numpy as np
from gymnasium import spaces
class ObservationManager:
    def __init__(self, env):
        self.env = env

        self.selected_obs = env.config.get("selected_obs", ["visited_cells", "agent_positions", "detected_targets"])

        self.observation_spaces = {
            "visited_cells": spaces.Box(0, np.inf, shape=(self.env.G1, self.env.G2), dtype=np.float32),
            "agent_positions": spaces.Box(0, max(self.env.G1, self.env.G2), shape=(self.env.n_agents, 2), dtype=np.float32),
            "detected_targets": spaces.Box(0, 1, shape=(self.env.n_targets,), dtype=np.float32),
            "target_completion": spaces.Box(0, 1, shape=(self.env.n_targets,), dtype=np.float32),
            "target_types": spaces.Box(0, 2, shape=(self.env.n_targets,), dtype=np.float32),
            "target_positions": spaces.Box(0, max(self.env.G1, self.env.G2), shape=(self.env.n_targets, 2), dtype=np.float32),
            "adjacency_matrix": spaces.Box(0, 1, shape=(self.env.n_agents + 1, self.env.n_agents + 1), dtype=np.float32),
            "paths": spaces.Box(0, max(self.env.G1, self.env.G2), shape=(self.env.n_agents, self.env.max_steps, 2), dtype=np.float32),
            "timestep": spaces.Box(0, self.env.max_steps, shape=(1,), dtype=np.int32),
            "n_targets": spaces.Box(0, self.env.n_targets, shape=(1,), dtype=np.int32),
            "completed_targets": spaces.Box(0, 1, shape=(self.env.n_targets,), dtype=np.float32),
            "detected_targets": spaces.Box(0, 1, shape=(self.env.n_targets,), dtype=np.float32),
            "target_visits": spaces.Box(0, np.inf, shape=(self.env.n_targets,), dtype=np.float32),
            "target_connected_visits": spaces.Box(0, np.inf, shape=(self.env.n_targets,), dtype=np.float32),
            "connectivity": spaces.Box(0, 1, shape=(1,), dtype=np.float32),
            "coverage_percentage": spaces.Box(0, 1, shape=(1,), dtype=np.float32),
            "unique_cells_covered": spaces.Box(0, np.inf, shape=(1,), dtype=np.float32)
        }
        


        self.observation_funcs = {
            "visited_cells": lambda env: env.visited_cells.astype(np.float32),
            "agent_positions": lambda env: env.drone_positions,
            "detected_targets": lambda env: env.detected_targets.astype(np.float32),
            "target_completion": lambda env: env.target_completion.astype(np.float32),
            "target_positions": lambda env: env.target_positions,
            "adjacency_matrix": lambda env: env.adjacency_matrix,
            "paths": lambda env: env.paths,
            "timestep": lambda env: env.timestep,
            "n_targets": lambda env: env.n_targets,
            "detected_targets": lambda env: env.detected_targets.copy(),
            "completed_targets": lambda env: env.target_completion.copy(),
            "target_visits": lambda env: env.target_visits.copy(),
            "target_connected_visits": lambda env: env.target_connected_visits.copy(),
            "connectivity": lambda env: env._compute_connectivity(),
            "coverage_percentage": lambda env: np.sum(env.visited_cells > 0) / (env.G1 * env.G2),
            "unique_cells_covered": lambda env: np.sum(env.visited_cells > 0),
            "target_types": lambda env: env.target_types.copy()
        }

    def get_observation(self, env):
        return {key: self.observation_funcs[key](env) for key in self.selected_obs}
