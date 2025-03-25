import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

def process_trajectory_file(file_path = None):
    data = np.loadtxt(file_path)
    if data.size == 0:
        return None, None
    timestamps = data[:, 0]
    positions = data[:, 1:4]
    return timestamps, positions

def visualize_2d_trajectory(file_path = None):
    timestamps, positions = process_trajectory_file(file_path=file_path)
    plt.figure(figsize=(10, 8))
    
    # Plot top-down view (X-Z plane)
    plt.plot(positions[:, 0], positions[:, 2], 'b-', label='Trajectory')
    plt.plot(positions[0, 0], positions[0, 2], 'go', label='Start')
    plt.plot(positions[-1, 0], positions[-1, 2], 'ro', label='End')
    
    plt.xlabel('X (meters)')
    plt.ylabel('Z (meters)')
    plt.axis('equal')  # Equal scaling
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Replace with your trajectory file path
    trajectory_file = "/home/rahul/Desktop/aylAI/ORB_SLAM3/CameraTrajectory.txt"
    
    # Create 2D visualization (top-down view)
    visualize_2d_trajectory(trajectory_file)
