import numpy as np

class RewardManager:
    def __init__(self, env):
        self.env = env
        self.config = env.config
        
        self.coverage_reward_func = lambda env: env.config["reward_weights"]["coverage"] * (np.sum(env.visited_cells > 0) - env.previous_coverage)
        self.revisit_penalty_func = lambda env: env.config["reward_weights"]["revisit"] * np.sum(env.visited_cells > 1)
        self.connectivity_reward_func = lambda env: env.config["reward_weights"]["connectivity"] * env.compute_connectivity() if env.config["enable_connectivity"] else 0
        self.target_reward_func = lambda env: env.config["reward_weights"]["target"] * env.update_target_states()
        self.reward_funcs = {
            "coverage": self.coverage_reward_func,
            "revisit": self.revisit_penalty_func,
            "connectivity": self.connectivity_reward_func,
            "target": self.target_reward_func
        }

    def compute_reward(self):
        return sum(func(self.env) for func in self.reward_funcs.values())