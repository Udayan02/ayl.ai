import numpy as np
import open3d as o3d
import os
import matplotlib.pyplot as plt

# Dictionary to store points by timestamp
points_by_time = {}

folder = input("Enter the path to the tracked_keypoints directory: ")

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

# Visualize
o3d.visualization.draw_geometries([pcd])

# Save the point cloud
o3d.io.write_point_cloud("store_map.ply", pcd)


# 2D Visualization (X-Z plane)
plt.figure(figsize=(10, 8))
plt.scatter(all_points[:, 0], all_points[:, 2], s=1, alpha=0.5)
plt.title('2D Point Cloud (X-Z Plane)')
plt.xlabel('X axis')
plt.ylabel('Z axis')
plt.axis('equal')  # Equal scale for X and Z axes
plt.grid(True)
plt.savefig("store_map_2d_xz.png", dpi=300)
plt.show()