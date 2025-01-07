import pickle
import matplotlib.pyplot as plt

with open("reward_info.pkl", "rb") as f:
    reward_info = pickle.load(f)



for episode in reward_info:
    total_coverage_reward = 0
    total_target_reward = 0
    total_revisit_penalty = 0
    total_connectivity_reward = 0
    for step in episode["reward_history"]:
        #print(f"Timestep: {step['timestep']}, Connectivity Reward: {step['connectivity_reward']}, Target Reward: {step['target_reward']}, Target Completion: {step['target_completion']}, Target Visits: {step['target_visits']}, Target Connected Visits: {step['target_connected_visits']}, Target Consecutive Visits: {step['target_consecutive_visits']}, Coverage Reward: {step['coverage_reward']}, Revisit Penalty: {step['revisit_penalty']}")
        total_coverage_reward += step["coverage_reward"]
        total_target_reward += step["target_reward"]
        total_revisit_penalty += step["revisit_penalty"]
        total_connectivity_reward += step["connectivity_reward"]

    print(f"Total Coverage Reward: {total_coverage_reward}, Total Target Reward: {total_target_reward}, Total Revisit Penalty: {total_revisit_penalty}, Total Connectivity Reward: {total_connectivity_reward}")
