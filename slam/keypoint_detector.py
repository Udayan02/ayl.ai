import cv2
import numpy as np
import argparse
import os


def extract_keypoint_neighborhoods(keypoints_file, video_path, timestamp, output_dir, neighborhood_size=10):
    """
    Extract pixel neighborhoods around keypoints from a video frame.

    Args:
        keypoints_file: Path to the tracked keypoints file
        video_path: Path to the video file
        timestamp: Timestamp of the frame to process (in seconds)
        output_dir: Directory to save the extracted neighborhoods
        neighborhood_size: Size of the neighborhood (default: 10x10)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load keypoints from file
    # Format: [x_2d, y_2d, X_3d, Y_3d, Z_3d]
    keypoints = np.loadtxt(keypoints_file)
    if keypoints.ndim == 1:  # Only one keypoint
        keypoints = keypoints.reshape(1, -1)

    # Extract 2D pixel coordinates
    pixel_coords = keypoints[:, :2].astype(int)

    # Open the video
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate frame number from timestamp
    target_frame = int(timestamp * fps)

    print(f"Video FPS: {fps}")
    print(f"Total frames: {total_frames}")
    print(f"Target frame: {target_frame}")

    # Seek to the target frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)

    # Read the frame
    ret, frame = cap.read()
    if not ret:
        print(f"Failed to read frame {target_frame}")
        cap.release()
        return

    # Get frame dimensions
    height, width = frame.shape[:2]

    # For visualization
    viz_frame = frame.copy()

    # Process each keypoint
    half_size = neighborhood_size // 2
    neighborhoods = []

    for i, (x, y) in enumerate(pixel_coords):
        # Check if the keypoint is within frame boundaries
        if 0 <= x < width and 0 <= y < height:
            # Calculate neighborhood boundaries with padding
            x1 = max(0, x - half_size)
            y1 = max(0, y - half_size)
            x2 = min(width, x + half_size)
            y2 = min(height, y + half_size)

            # Extract neighborhood
            neighborhood = frame[y1:y2, x1:x2]
            neighborhoods.append((i, neighborhood))

            # Draw rectangle on visualization frame
            cv2.rectangle(viz_frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.circle(viz_frame, (x, y), 2, (0, 0, 255), -1)

            # Save individual neighborhood
            cv2.imwrite(f"{output_dir}/keypoint_{i}_neighborhood.png", neighborhood)

    # Save visualization frame
    # cv2.imwrite(f"{output_dir}/frame_{target_frame}_keypoints.png", viz_frame)
    cv2.imshow("Display", viz_frame)

    print(f"Extracted {len(neighborhoods)} neighborhoods from frame {target_frame}")
    cap.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract keypoint neighborhoods from video")
    parser.add_argument("--keypoints", required=True, help="Path to tracked keypoints file")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--timestamp", type=float, required=True, help="Timestamp in seconds")
    parser.add_argument("--output", default="neighborhoods", help="Output directory")
    parser.add_argument("--size", type=int, default=10, help="Neighborhood size")

    args = parser.parse_args()

    extract_keypoint_neighborhoods(
        args.keypoints,
        args.video,
        args.timestamp,
        args.output,
        args.size
    )