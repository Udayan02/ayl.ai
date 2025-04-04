from kmeans_clustering import KeypointClustering
import argparse
import cv2
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract keypoint neighborhoods from video")
    parser.add_argument("--keypoints", required=True, help="Path to tracked keypoints file")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--timestamp", type=float, required=True, help="Timestamp in seconds")
    parser.add_argument("--output", type=str, required=True, help="Path to store image output")

    args = parser.parse_args()
    keypoints = np.loadtxt(args.keypoints)
    if keypoints.ndim == 1:  # Only one keypoint
        keypoints = keypoints.reshape(1, -1)

    pixel_coords = keypoints[:, :2].astype(int)

    # Determine optimal number of clusters
    kmeans = KeypointClustering(pixel_coords)
    inertias, silhouette_scores = kmeans.find_optimal_clusters(max_clusters=10)

    # Choose best number of clusters from the plot (where the "elbow" occurs)
    best_n_clusters = 5  # Replace with your chosen number

    # Run final clustering with optimal number of clusters
    kmeans = KeypointClustering(pixel_coords, n_clusters=best_n_clusters, max_iter=1000)
    kmeans.fit()

    # Get cluster assignments for each keypoint
    cluster_labels = kmeans.labels

    # Get cluster centroids
    centroids = kmeans.centroids

    args = parser.parse_args()
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
        for i, (x, y) in enumerate(kmeans.centroids):
            cv2.circle(viz_frame, (x, y), 5, (0, 0, 255), -1)

        cv2.imwrite(f"{output_path}/frame_{target_frame}_centroids.png", viz_frame)
        cap.release()