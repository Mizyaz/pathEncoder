import numpy as np

DEFAULT_INITIAL_POSITIONS = np.array([[0, 0], 
                                      [3, 0], 
                                      [3, 0], 
                                      [7, 0], 
                                      [0, 7], 
                                      [5, 0], 
                                      [0, 5], 
                                      [2, 0], 
                                      [0, 2], 
                                      [0, 4], 
                                      [4, 0], 
                                      [1, 0], 
                                      [0, 1]])

def get_distance_matrix(positions):
    return np.linalg.norm(positions[:, np.newaxis] - positions, axis=2)

def get_adjacency_matrix(positions):
    distance_matrix = get_distance_matrix(positions)
    adjacency_matrix = (distance_matrix <= 1).astype(np.float32)
    return adjacency_matrix