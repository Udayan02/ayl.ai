import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os

# Directory containing the tracked keypoint files
KEYPOINTS_FOLDER = "./tracked_keypoints"
OUTPUT_DIR = "./kmeans_results"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Function to perform k-means clustering with 1 centroid
def perform_kmeans(points_3d):
    if len(points_3d) < 1:
        return None
    
    kmeans = KMeans(n_clusters=1, random_state=0, n_init=10)
    kmeans.fit(points_3d)
    centroid = kmeans.cluster_centers_[0]
    return centroid

# Dictionary to store centroids by timestamp
centroids_by_time = {}

# Process all keypoint files
print("Processing keypoint files...")
for filename in os.listdir(KEYPOINTS_FOLDER):
    if filename.startswith('keypoints_tracked_'):
        # Extract timestamp from filename
        timestamp = float(filename.split('_')[-1].split('.')[0])
        file_path = os.path.join(KEYPOINTS_FOLDER, filename)
        
        try:
            # Load the keypoints file
            points = np.loadtxt(file_path, encoding="utf-8")
            
            if points.size > 0:
                # Extract 3D coordinates (columns 3-5)
                points_3d = points[:, 2:5]
                
                # Perform k-means clustering to find the centroid
                centroid = perform_kmeans(points_3d)
                
                if centroid is not None:
                    centroids_by_time[timestamp] = centroid
                    # print(f"Processed {filename}: Centroid at {centroid}")
                else:
                    print(f"Skipping {filename}: Not enough points for clustering")
            else:
                print(f"Skipping {filename}: No points found")
                
        except Exception as e:
            print(f"Error processing {filename}: {e}")

# Convert centroids to numpy array for visualization
if centroids_by_time:
    timestamps = np.array(list(centroids_by_time.keys()))
    centroids = np.array(list(centroids_by_time.values()))
    
    # Sort by timestamp
    sort_idx = np.argsort(timestamps)
    timestamps = timestamps[sort_idx]
    centroids = centroids[sort_idx]
    
    # Save centroids to file
    centroids_with_time = np.column_stack((timestamps, centroids))
    np.savetxt(os.path.join(OUTPUT_DIR, "centroids.txt"), centroids_with_time, 
               header="timestamp x y z", fmt="%.6f")
    
    # Create a 2D visualization of centroids (X-Z plane)
    plt.figure(figsize=(12, 10))
    
    # Plot all centroids
    plt.scatter(centroids[:, 0], centroids[:, 2], s=20, color='green', alpha=0.7, 
                label='Frame Centroids')
    
    # Connect centroids in sequence to show the path
    plt.plot(centroids[:, 0], centroids[:, 2], 'g-', linewidth=1, alpha=0.5)
    
    # Mark start and end points
    plt.scatter(centroids[0, 0], centroids[0, 2], s=100, color='blue', marker='o', 
                label='Start')
    plt.scatter(centroids[-1, 0], centroids[-1, 2], s=100, color='red', marker='o', 
                label='End')
    
    # Add title and labels
    plt.title('K-means Centroids from Each Frame (X-Z Plane)')
    plt.xlabel('X axis (meters)')
    plt.ylabel('Z axis (meters)')
    plt.grid(True)
    plt.legend()
    
    # Set reasonable axis limits - may need adjustment based on your data
    plt.xlim(-50, 50)  # Adjust these values based on your data
    plt.ylim(-30, 70)    # Adjust these values based on your data
    
    # Save the plot
    plt.savefig(os.path.join(OUTPUT_DIR, "centroids_trajectory_xz.png"), dpi=300)
    print(f"Saved plot to {os.path.join(OUTPUT_DIR, 'centroids_trajectory_xz.png')}")
    
    print(f"Processed {len(centroids_by_time)} frames with valid centroids")
else:
    print("No valid centroids found")

print("Done!")
