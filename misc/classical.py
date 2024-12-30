import numpy as np
import pulp
from typing import List, Dict, Tuple, Any
import itertools

# Define action mappings
ACTION_MAP = {
    0: (0, 0),   # hover
    1: (0, 1),   # up
    2: (0, -1),  # down
    3: (-1, 0),  # left
    4: (1, 0)    # right
}

def manhattan_distance(x1: int, y1: int, x2: int, y2: int) -> int:
    """Calculate Manhattan distance between two points."""
    return abs(x1 - x2) + abs(y1 - y2)

def integer_optimizer(
    config: Dict[str, Any],
    initial_positions: List[Tuple[int, int]],
    target_positions: List[Tuple[int, int]]
) -> List[List[int]]:
    """
    Solves for the optimal sequence of actions for each UAV using classical integer optimization.
    """
    n_agents = config["n_agents"]
    G1, G2 = config["grid_size"]
    max_steps = config["max_steps"]
    n_targets = config["n_targets"]
    target_types = config.get("target_types", [2] * n_targets)
    weights = config["reward_weights"]

    # Initialize the problem
    prob = pulp.LpProblem("UAV_Mission_Optimization", pulp.LpMaximize)

    # Decision variables
    actions = {
        (a, t): pulp.LpVariable(f"action_a{a}_t{t}", lowBound=0, upBound=4, cat='Integer')
        for a in range(n_agents)
        for t in range(max_steps)
    }

    # Positions
    positions = {
        (a, t, d): pulp.LpVariable(f"pos_a{a}_t{t}_d{d}", lowBound=0, upBound=max(G1, G2)-1, cat='Integer')
        for a in range(n_agents)
        for t in range(max_steps + 1)
        for d in range(2)
    }

    # Target detection
    target_detected = {
        (i, t): pulp.LpVariable(f"target_detected_{i}_t{t}", cat='Binary')
        for i in range(n_targets)
        for t in range(max_steps + 1)
    }

    # Initial positions constraints
    for a in range(n_agents):
        initial_x, initial_y = initial_positions[a]
        prob += positions[a, 0, 0] == initial_x, f"init_x_{a}"
        prob += positions[a, 0, 1] == initial_y, f"init_y_{a}"

    # Position update constraints
    for a in range(n_agents):
        for t in range(max_steps):
            # Update positions based on actions
            for d in range(2):
                prob += positions[a, t + 1, d] == positions[a, t, d] + \
                    pulp.lpSum([ACTION_MAP[i][d] * (actions[a, t] == i) for i in range(5)]), \
                    f"pos_update_{a}_{t}_{d}"

            # Boundary constraints
            prob += positions[a, t + 1, 0] >= 0, f"bound_min_x_{a}_{t}"
            prob += positions[a, t + 1, 0] <= G1 - 1, f"bound_max_x_{a}_{t}"
            prob += positions[a, t + 1, 1] >= 0, f"bound_min_y_{a}_{t}"
            prob += positions[a, t + 1, 1] <= G2 - 1, f"bound_max_y_{a}_{t}"

    # Target detection constraints
    for i, target_pos in enumerate(target_positions):
        tx, ty = target_pos
        for t in range(max_steps + 1):
            # Target is detected if any UAV is at target position
            for a in range(n_agents):
                # Distance to target
                dist_x = pulp.LpVariable(f"dist_x_a{a}_t{t}_i{i}", lowBound=0)
                dist_y = pulp.LpVariable(f"dist_y_a{a}_t{t}_i{i}", lowBound=0)
                
                # |x - tx| constraints
                prob += dist_x >= positions[a, t, 0] - tx, f"dist_x_ge1_a{a}_t{t}_i{i}"
                prob += dist_x >= tx - positions[a, t, 0], f"dist_x_ge2_a{a}_t{t}_i{i}"
                
                # |y - ty| constraints
                prob += dist_y >= positions[a, t, 1] - ty, f"dist_y_ge1_a{a}_t{t}_i{i}"
                prob += dist_y >= ty - positions[a, t, 1], f"dist_y_ge2_a{a}_t{t}_i{i}"
                
                # Total Manhattan distance
                total_dist = pulp.LpVariable(f"total_dist_a{a}_t{t}_i{i}", lowBound=0)
                prob += total_dist == dist_x + dist_y, f"total_dist_a{a}_t{t}_i{i}"
                
                # Target is detected if distance is 0
                prob += target_detected[i, t] >= 1 - total_dist, f"target_detected_a{a}_t{t}_i{i}"

    # Objective: Maximize target detection
    detection_obj = weights["target_detection"] * pulp.lpSum(target_detected[i, t] for i in range(n_targets) for t in range(max_steps + 1))
    
    # Add distance minimization to help guide the solution
    # Create auxiliary variables for absolute values in the objective
    dist_vars = {
        (a, t, i): pulp.LpVariable(f"obj_dist_a{a}_t{t}_i{i}", lowBound=0)
        for a in range(n_agents)
        for t in range(max_steps + 1)
        for i in range(n_targets)
    }

    # Add constraints for the absolute values
    for a in range(n_agents):
        for t in range(max_steps + 1):
            for i, (tx, ty) in enumerate(target_positions):
                # X distance
                dx = pulp.LpVariable(f"obj_dx_a{a}_t{t}_i{i}", lowBound=0)
                prob += dx >= positions[a, t, 0] - tx, f"obj_dx_ge1_a{a}_t{t}_i{i}"
                prob += dx >= tx - positions[a, t, 0], f"obj_dx_ge2_a{a}_t{t}_i{i}"
                
                # Y distance
                dy = pulp.LpVariable(f"obj_dy_a{a}_t{t}_i{i}", lowBound=0)
                prob += dy >= positions[a, t, 1] - ty, f"obj_dy_ge1_a{a}_t{t}_i{i}"
                prob += dy >= ty - positions[a, t, 1], f"obj_dy_ge2_a{a}_t{t}_i{i}"
                
                # Total distance
                prob += dist_vars[a, t, i] == dx + dy, f"obj_dist_a{a}_t{t}_i{i}"

    # Distance objective
    distance_obj = -pulp.lpSum(dist_vars[a, t, i] for a in range(n_agents) for t in range(max_steps + 1) for i in range(n_targets))

    prob += detection_obj + 0.1 * distance_obj, "objective"

    # Solve the problem
    solver = pulp.PULP_CBC_CMD(msg=True)
    result = prob.solve(solver)

    # Check if an optimal solution was found
    if pulp.LpStatus[result] != 'Optimal':
        raise ValueError("No optimal solution found.")

    # Extract the action sequences
    actions_sequence = [[] for _ in range(n_agents)]
    for a in range(n_agents):
        for t in range(max_steps):
            action_val = int(pulp.value(actions[a, t]))
            actions_sequence[a].append(action_val)

    return actions_sequence

# Example Usage
if __name__ == "__main__":
    # Example configuration matching the environment
    config = {
        "n_agents": 2,
        "grid_size": (5, 5),
        "max_steps": 10,
        "comm_dist": 2.8284,  # sqrt(8), roughly
        "gcs_pos": (0, 0),
        "enable_connectivity": True,
        "n_targets": 1,
        "target_types": [2],
        "reward_weights": {
            "coverage": 1.0,
            "revisit": -0.1,
            "connectivity": 0.5,
            "target_detection": 2.0,
            "target_relay": 1.5,
            "target_completion": 5.0
        }
    }

    # Initial positions of UAVs
    initial_positions = [
        (0, 0),  # UAV 0 starts at GCS
        (4, 4)   # UAV 1 starts at opposite corner
    ]

    # Target positions
    target_positions = [
        (2, 2)  # Single target in the center
    ]

    # Compute the optimal action sequence
    optimal_actions = integer_optimizer(config, initial_positions, target_positions)

    # Display the action sequences
    for a, actions in enumerate(optimal_actions):
        action_names = [list(ACTION_MAP.keys())[act] for act in actions]
        print(f"UAV {a}: {action_names}")
