import numpy as np
import open3d as o3d
import os
import matplotlib.pyplot as plt


def process_trajectory_file(file_path=None):
    data = np.loadtxt(file_path)
    if data.size == 0:
        return None, None
    timestamps = data[:, 0]
    positions = data[:, 1:4]
    return timestamps, positions

# Dictionary to store points by timestamp
points_by_time = {}

folder = input("Enter the path to the tracked_keypoints directory: ")
trajectory_file = input("Enter the path to the trajectory file: ")

# Read all tracked keypoint files
for filename in os.listdir(folder):
    if filename.startswith('keypoints_tracked_'):
        timestamp = float(filename.split('_')[-1].split('.')[0])
        file_path = os.path.join(folder, filename)
        points = np.loadtxt(file_path, encoding="utf-8")

        # Points format: [x_2d, y_2d, X_3d, Y_3d, Z_3d]
        if points.size > 0:
            points_3d = points[:, 2:5]  # Get just the 3D coordinates
            points_by_time[timestamp] = points_3d

# Combine all points into a single point cloud
all_points = np.vstack([pts for pts in points_by_time.values()])

# Create Open3D point cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(all_points)

# Save the point cloud
o3d.io.write_point_cloud("store_map.ply", pcd)

# Get trajectory data
timestamps, positions = process_trajectory_file(file_path=trajectory_file)

# Combined 2D Visualization (X-Z plane)
plt.figure(figsize=(12, 10))

# Plot point cloud
plt.scatter(all_points[:, 0], all_points[:, 2], s=1, alpha=0.3, color='gray', label='Point Cloud')

# Plot trajectory on top
if timestamps is not None and positions is not None:
    plt.plot(positions[:, 0], positions[:, 2], 'b-', linewidth=2, label='Trajectory')
    plt.plot(positions[0, 0], positions[0, 2], 'go', markersize=8, label='Start')
    plt.plot(positions[-1, 0], positions[-1, 2], 'ro', markersize=8, label='End')

plt.title('2D Point Cloud with Trajectory (X-Z Plane)')
plt.xlabel('X axis (meters)')
plt.ylabel('Z axis (meters)')
plt.axis('equal')  # Equal scale for X and Z axes
plt.grid(True)
plt.legend()
plt.savefig("combined_map_2d_xz.png", dpi=300)
plt.show()