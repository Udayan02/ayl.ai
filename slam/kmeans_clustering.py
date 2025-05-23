import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import argparse
import cv2


class KeypointClustering:
    def __init__(self, keypoints, n_clusters=5, max_iter=100, tol=1e-4):
        """
        Initialize the KeypointClustering class.

        Parameters:
        -----------
        keypoints : numpy.ndarray
            Array of shape (n_points, 2) containing the (x, y) coordinates of the keypoints
        n_clusters : int, default=5
            Number of clusters to form
        max_iter : int, default=100
            Maximum number of iterations for the k-means algorithm
        tol : float, default=1e-4
            Tolerance for convergence
        """
        self.keypoints = np.array(keypoints)
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.centroids = None
        self.labels = None

    def initialize_centroids(self):
        """Initialize centroids using the k-means++ method."""
        # Select first centroid randomly
        centroids = [self.keypoints[np.random.randint(len(self.keypoints))]]

        # Select remaining centroids based on distance
        for _ in range(1, self.n_clusters):
            # Calculate distances to existing centroids
            distances = cdist(self.keypoints, np.array(centroids)).min(axis=1)
            # Normalize distances to create a probability distribution
            probs = distances / distances.sum()
            # Select next centroid based on probability distribution
            next_centroid_idx = np.random.choice(len(self.keypoints), p=probs)
            centroids.append(self.keypoints[next_centroid_idx])

        return np.array(centroids)

    def fit(self):
        """Fit k-means clustering to the keypoints."""
        # Initialize centroids
        self.centroids = self.initialize_centroids()

        for iteration in range(self.max_iter):
            # Calculate distances between points and centroids
            distances = cdist(self.keypoints, self.centroids)

            # Assign each point to the nearest centroid
            new_labels = np.argmin(distances, axis=1)

            # Store the current centroids to check for convergence
            old_centroids = self.centroids.copy()

            # Update centroids based on the mean of assigned points
            for i in range(self.n_clusters):
                if np.sum(new_labels == i) > 0:  # Avoid empty clusters
                    self.centroids[i] = np.mean(self.keypoints[new_labels == i], axis=0)

            # Check for convergence
            if np.allclose(old_centroids, self.centroids, atol=self.tol):
                break

            self.labels = new_labels

        return self

    def predict(self, points):
        """Predict cluster labels for new points."""
        points = np.array(points)
        distances = cdist(points, self.centroids)
        return np.argmin(distances, axis=1)

    def inertia(self):
        """Calculate the inertia (sum of squared distances to nearest centroid)."""
        distances = cdist(self.keypoints, self.centroids)
        min_distances = np.min(distances, axis=1)
        return np.sum(min_distances ** 2)

    def silhouette_score(self):
        """Calculate silhouette score to evaluate clustering quality."""
        from sklearn.metrics import silhouette_score
        if len(np.unique(self.labels)) < 2:
            return 0  # Cannot calculate silhouette for 1 cluster
        return silhouette_score(self.keypoints, self.labels)

    def find_optimal_clusters(self, max_clusters=10):
        """Find the optimal number of clusters using the elbow method."""
        inertias = []
        silhouette_scores = []

        for n in range(2, max_clusters + 1):
            self.n_clusters = n
            self.fit()
            inertias.append(self.inertia())
            silhouette_scores.append(self.silhouette_score())

        return inertias, silhouette_scores

    def get_cluster_bounding_boxes(self):
        """Get bounding boxes for each cluster."""
        bounding_boxes = []

        for cluster_idx in range(self.n_clusters):
            cluster_points = self.keypoints[self.labels == cluster_idx]

            if len(cluster_points) > 0:
                # Find min and max coordinates for the bounding box
                min_x = np.min(cluster_points[:, 0])
                min_y = np.min(cluster_points[:, 1])
                max_x = np.max(cluster_points[:, 0])
                max_y = np.max(cluster_points[:, 1])

                # Store as (x_min, y_min, x_max, y_max)
                bounding_boxes.append((int(min_x), int(min_y), int(max_x), int(max_y)))
            else:
                bounding_boxes.append(None)

        return bounding_boxes

    def plot_clusters(self, title="Keypoint Clusters"):
        """Plot the clusters and centroids."""
        plt.figure(figsize=(10, 8))

        # Plot the points with their cluster colors
        for cluster_idx in range(self.n_clusters):
            cluster_points = self.keypoints[self.labels == cluster_idx]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster_idx}')

            # Draw bounding box if cluster has points
            if len(cluster_points) > 0:
                min_x = np.min(cluster_points[:, 0])
                min_y = np.min(cluster_points[:, 1])
                max_x = np.max(cluster_points[:, 0])
                max_y = np.max(cluster_points[:, 1])

                # Create rectangle patch
                width = max_x - min_x
                height = max_y - min_y
                rect = plt.Rectangle((min_x, min_y), width, height,
                                     fill=False, edgecolor='gray', linestyle='--', linewidth=1.5)
                plt.gca().add_patch(rect)

        # Plot the centroids
        plt.scatter(self.centroids[:, 0], self.centroids[:, 1],
                    marker='X', s=200, c='black', label='Centroids')

        plt.title(title)
        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        return plt


parser = argparse.ArgumentParser(description="Extract keypoint neighborhoods from video")
parser.add_argument("--keypoints", required=True, help="Path to tracked keypoints file")
parser.add_argument("--video", required=True, help="Path to video file")
parser.add_argument("--timestamp", type=float, required=True, help="Timestamp in seconds")
parser.add_argument("--output", type=str, required=True, help="Output path to store image")

args = parser.parse_args()
keypoints = np.loadtxt(args.keypoints)
if keypoints.ndim == 1:  # Only one keypoint
    keypoints = keypoints.reshape(1, -1)

pixel_coords = keypoints[:, :2].astype(int)

# Determine optimal number of clusters
kmeans = KeypointClustering(pixel_coords, max_iter=1000)
inertias, silhouette_scores = kmeans.find_optimal_clusters(max_clusters=10)

# Plot to find optimal number of clusters
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(2, len(inertias) + 2), inertias, 'o-')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method')

# Choose best number of clusters from the plot (where the "elbow" occurs)
best_n_clusters = 5  # Replace with your chosen number

# Run final clustering with optimal number of clusters
kmeans = KeypointClustering(pixel_coords, n_clusters=best_n_clusters, max_iter=1000)
kmeans.fit()

# Get cluster assignments for each keypoint
cluster_labels = kmeans.labels

# Get cluster centroids
centroids = kmeans.centroids

# Get bounding boxes for each cluster
bounding_boxes = kmeans.get_cluster_bounding_boxes()

# Visualize results
kmeans.plot_clusters()
plt.show()

video_path = args.video
timestamp = args.timestamp
output_path = args.output

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
target_frame = int(timestamp * fps)

cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)

ret, frame = cap.read()
if not ret:
    print(f"Failed to read frame {target_frame}")
    cap.release()
else:
    height, width = frame.shape[:2]
    frame = cv2.resize(frame, (1280, 720))
    viz_frame = frame.copy()
    colors = [(220, 20, 60), (65, 105, 225), (34, 139, 34), (218, 165, 32), (255, 140, 0)]

    # Draw centroids
    for i, (x, y) in enumerate(kmeans.centroids):
        cv2.circle(viz_frame, (int(x), int(y)), 5, (0, 0, 255), -1)

    # Draw keypoints and bounding boxes for each cluster
    for cluster_idx in range(kmeans.n_clusters):
        cluster_points = kmeans.keypoints[kmeans.labels == cluster_idx]
        cluster_color = colors[cluster_idx % len(colors)]

        # Draw keypoints
        for point in cluster_points:
            cv2.circle(viz_frame, (int(point[0]), int(point[1])), 3, cluster_color, -1)

        # Draw bounding box
        if bounding_boxes[cluster_idx]:
            x_min, y_min, x_max, y_max = bounding_boxes[cluster_idx]
            # Add padding to make boxes more visible (optional)
            padding = 5
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(viz_frame.shape[1], x_max + padding)
            y_max = min(viz_frame.shape[0], y_max + padding)

            # Draw rectangle with thicker line
            cv2.rectangle(viz_frame, (x_min, y_min), (x_max, y_max), cluster_color, 2)

            # Optionally add cluster label text
            cv2.putText(viz_frame, f"Cluster {cluster_idx}", (x_min, y_min - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, cluster_color, 2)

    cv2.imwrite(f"{output_path}/frame_{target_frame}_centroids.png", viz_frame)
    cap.release()