import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
from typing import List, Tuple, Optional, Dict
import networkx as nx

class PathAnimator:
    """
    A class to animate drone paths with connectivity visualization.
    """
    def __init__(
        self,
        paths: np.ndarray,  # Shape: (n_agents, timesteps, 2)
        grid_size: Tuple[int, int] = (10, 10),
        animation_duration: float = 10.0,
        background_alpha: float = 0.4,
        drone_size: float = 100,
        gcs_pos: Tuple[int, int] = (0, 0),
        comm_range: float = 2.8284,
        path_colors: Optional[List[str]] = None,
        show_connectivity: bool = True,
        connectivity_style: str = ':',
        connectivity_color: str = 'gray',
        connectivity_alpha: float = 0.5,
        connected_color: str = 'green',
        disconnected_color: str = 'red',
        inner_dot_size: float = 30,
        fps: int = 30
    ):
        """
        Initialize the PathAnimator.

        Args:
            paths: Numpy array of shape (n_agents, timesteps, 2) containing drone paths
            grid_size: Tuple of grid dimensions (width, height)
            animation_duration: Total duration of animation in seconds
            background_alpha: Alpha value for background path traces
            drone_size: Size of the drone markers
            gcs_pos: Position of the Ground Control Station
            comm_range: Communication range for connectivity
            path_colors: List of colors for each drone's path
            show_connectivity: Whether to show connectivity lines
            connectivity_style: Line style for connectivity lines
            connectivity_color: Color of connectivity lines
            connectivity_alpha: Alpha value for connectivity lines
            connected_color: Color for connected drone indicators
            disconnected_color: Color for disconnected drone indicators
            inner_dot_size: Size of the connectivity status indicator
            fps: Frames per second for the animation
        """
        self.paths = paths
        self.n_agents = paths.shape[0]
        self.timesteps = paths.shape[1]
        self.grid_size = grid_size
        self.animation_duration = animation_duration
        self.background_alpha = background_alpha
        self.drone_size = drone_size
        self.gcs_pos = gcs_pos
        self.comm_range = comm_range
        self.show_connectivity = show_connectivity
        self.connectivity_style = connectivity_style
        self.connectivity_color = connectivity_color
        self.connectivity_alpha = connectivity_alpha
        self.connected_color = connected_color
        self.disconnected_color = disconnected_color
        self.inner_dot_size = inner_dot_size
        self.interval = 1000 * animation_duration / self.timesteps  # ms between frames

        # Set default colors if none provided
        if path_colors is None:
            self.path_colors = plt.cm.rainbow(np.linspace(0, 1, self.n_agents))
        else:
            self.path_colors = path_colors

        # Initialize the figure and animation
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.setup_plot()
        self.ani = None

    def setup_plot(self):
        """Set up the plot with initial settings."""
        self.ax.set_xlim(-0.5, self.grid_size[0] - 0.5)
        self.ax.set_ylim(-0.5, self.grid_size[1] - 0.5)
        self.ax.grid(True)
        self.ax.set_aspect('equal')
        
        # Plot GCS
        self.ax.plot(self.gcs_pos[0], self.gcs_pos[1], 'k^', markersize=10, label='GCS')
        
        # Plot background paths
        for i in range(self.n_agents):
            self.ax.plot(self.paths[i, :, 0], self.paths[i, :, 1], 
                        color=self.path_colors[i], alpha=self.background_alpha)

        # Initialize drone markers
        self.drone_markers = []
        self.inner_markers = []
        self.connectivity_lines = []

        for i in range(self.n_agents):
            # Main drone marker
            marker = self.ax.scatter([], [], s=self.drone_size, 
                                   color='black', zorder=5)
            self.drone_markers.append(marker)
            
            # Inner marker for connectivity status
            inner = self.ax.scatter([], [], s=self.inner_dot_size,
                                  color=self.disconnected_color, zorder=6)
            self.inner_markers.append(inner)

        # Initialize connectivity lines
        if self.show_connectivity:
            for _ in range(self.n_agents * (self.n_agents - 1) // 2 + self.n_agents):
                line = self.ax.plot([], [], self.connectivity_style,
                                  color=self.connectivity_color,
                                  alpha=self.connectivity_alpha, zorder=4)[0]
                self.connectivity_lines.append(line)

    def compute_connectivity(self, positions: np.ndarray) -> Tuple[List[bool], List[Tuple[float, float, float, float]]]:
        """
        Compute connectivity status and line coordinates.
        
        Args:
            positions: Current positions of all drones
            
        Returns:
            Tuple of (connected_status, line_coords)
        """
        connected = [False] * self.n_agents
        line_coords = []

        # Check GCS connectivity
        for i in range(self.n_agents):
            pos = positions[i]
            # Check GCS connection
            dist_to_gcs = np.linalg.norm(pos - np.array(self.gcs_pos))
            if dist_to_gcs <= self.comm_range:
                connected[i] = True
                line_coords.append((pos[0], self.gcs_pos[0], pos[1], self.gcs_pos[1]))

        # Check drone-to-drone connectivity
        for i in range(self.n_agents):
            for j in range(i + 1, self.n_agents):
                dist = np.linalg.norm(positions[i] - positions[j])
                if dist <= self.comm_range:
                    connected[i] = connected[i] or connected[j]
                    connected[j] = connected[i]
                    line_coords.append((positions[i][0], positions[j][0],
                                      positions[i][1], positions[j][1]))

        return connected, line_coords

    def update(self, frame):
        """Update function for animation."""
        current_positions = self.paths[:, frame, :]
        
        # Update drone positions
        for i, marker in enumerate(self.drone_markers):
            marker.set_offsets(current_positions[i])

        if self.show_connectivity:
            # Compute connectivity
            connected, line_coords = self.compute_connectivity(current_positions)
            
            # Update connectivity lines
            for i, line in enumerate(self.connectivity_lines):
                if i < len(line_coords):
                    x1, x2, y1, y2 = line_coords[i]
                    line.set_data([x1, x2], [y1, y2])
                else:
                    line.set_data([], [])

            # Update inner markers
            for i, (marker, is_connected) in enumerate(zip(self.inner_markers, connected)):
                marker.set_offsets(current_positions[i])
                marker.set_color(self.connected_color if is_connected else self.disconnected_color)

        return self.drone_markers + self.inner_markers + self.connectivity_lines

    def animate(self) -> FuncAnimation:
        """Create and return the animation."""
        self.ani = FuncAnimation(
            self.fig, self.update,
            frames=self.timesteps,
            interval=self.interval,
            blit=True
        )
        return self.ani

    def save(self, filename: str, **kwargs):
        """Save the animation to a file."""
        if self.ani is None:
            self.animate()
        self.ani.save(filename, **kwargs)

    def show(self):
        """Display the animation."""
        if self.ani is None:
            self.animate()
        plt.show() 