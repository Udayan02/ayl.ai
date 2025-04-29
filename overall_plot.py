import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
import os
from typing import Tuple, List, Dict, Any

def process_trajectory_file(file_path=None):
    """Process a trajectory file containing timestamps and positions."""
    data = np.loadtxt(file_path)
    if data.size == 0:
        return None, None
    timestamps = data[:, 0]
    positions = data[:, 1:4]
    return timestamps, positions

def extract_centroids_from_json(json_file):
    """Extract centroids from the JSON file along with timestamps, excluding 'unassigned' bbox."""
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Store all centroids with timestamps in a list
    all_centroids = []
    
    for frame_name, frame_data in data.items():
        timestamp = frame_data['timestamp']
        
        # Process each bounding box in the frame
        for bbox_id, bbox_data in frame_data['bounding_boxes'].items():
            # Skip "unassigned" bounding boxes
            if bbox_id == "unassigned":
                continue
                
            centroid = bbox_data.get('centroid_3d')
            if centroid is not None:
                # Store timestamp, frame name, bbox ID, and centroid coordinates
                all_centroids.append({
                    'timestamp': timestamp,
                    'frame': frame_name,
                    'bbox_id': bbox_id,
                    'centroid': centroid
                })
    
    # Sort all centroids by timestamp
    all_centroids.sort(key=lambda x: x['timestamp'])
    
    return all_centroids

def visualize_combined_2d_trajectory(trajectory_file=None, json_file=None, output_path=None, output_dir=None):
    """Create a visualization combining line trajectory and centroid points."""
    # Process the trajectory file
    if trajectory_file:
        timestamps_trajectory, positions_trajectory = process_trajectory_file(file_path=trajectory_file)
    else:
        timestamps_trajectory, positions_trajectory = None, None
    
    # Process the JSON file if provided
    centroids_data = None
    if json_file:
        centroids_data = extract_centroids_from_json(json_file)
    
    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Create the figure
    plt.figure(figsize=(12, 10))
    
    # Plot trajectory line if available
    if positions_trajectory is not None and len(positions_trajectory) > 0:
        # Plot top-down view (X-Z plane)
        plt.plot(positions_trajectory[:, 0], positions_trajectory[:, 2], 'b-', linewidth=2, label='Trajectory Line')
        plt.plot(positions_trajectory[0, 0], positions_trajectory[0, 2], 'bo', markersize=8, label='Line Start')
        plt.plot(positions_trajectory[-1, 0], positions_trajectory[-1, 2], 'bx', markersize=10, label='Line End')
    
    # Plot centroid points if available
    if centroids_data and len(centroids_data) > 0:
        # Convert centroids to numpy array for plotting
        points = np.array([c['centroid'] for c in centroids_data])
        timestamps = np.array([c['timestamp'] for c in centroids_data])
        
        # Plot centroid points (X-Z plane)
        plt.scatter(points[:, 0], points[:, 2], s=40, color='green', alpha=0.7, label='Centroids')
        
        # Mark start and end points with different colors
        plt.scatter(points[0, 0], points[0, 2], s=100, color='blue', marker='o', label='Centroid Start')
        plt.scatter(points[-1, 0], points[-1, 2], s=100, color='red', marker='o', label='Centroid End')
    
    # Add title and labels
    plt.title('Combined Trajectory Visualization (X-Z Plane)')
    plt.xlabel('X (meters)')
    plt.ylabel('Z (meters)')
    plt.grid(True)
    plt.legend()
    
    # Try to set reasonable axis limits
    all_points = []
    if positions_trajectory is not None and len(positions_trajectory) > 0:
        all_points.append(positions_trajectory[:, [0, 2]])  # X and Z coordinates
    
    if centroids_data and len(centroids_data) > 0:
        points = np.array([c['centroid'] for c in centroids_data])
        all_points.append(points[:, [0, 2]])  # X and Z coordinates
    
    if all_points:
        # Combine all points for determining axis limits
        combined_points = np.vstack(all_points)
        x_min, x_max = np.min(combined_points[:, 0]), np.max(combined_points[:, 0])
        z_min, z_max = np.min(combined_points[:, 1]), np.max(combined_points[:, 1])
        
        # Add some padding (20%)
        x_pad = (x_max - x_min) * 0.2
        z_pad = (z_max - z_min) * 0.2
        
        plt.xlim(x_min - x_pad, x_max + x_pad)
        plt.ylim(z_min - z_pad, z_max + z_pad)
    
    # Save or display the plot
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Combined trajectory visualization saved to: {output_path}")
    elif output_dir:
        output_file = os.path.join(output_dir, "combined_trajectory_xz.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Combined trajectory visualization saved to: {output_file}")
    else:
        plt.show()
    
    # Create a 3D plot if centroid data is available
    if centroids_data and len(centroids_data) > 0 and output_dir:
        create_3d_visualization(centroids_data, positions_trajectory, output_dir)

def create_3d_visualization(centroids_data, positions_trajectory=None, output_dir=None):
    """Create a 3D visualization of the trajectory and centroids."""
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot trajectory line if available
    if positions_trajectory is not None and len(positions_trajectory) > 0:
        ax.plot(positions_trajectory[:, 0], positions_trajectory[:, 1], positions_trajectory[:, 2], 
                'b-', linewidth=2, label='Trajectory Line')
        ax.scatter(positions_trajectory[0, 0], positions_trajectory[0, 1], positions_trajectory[0, 2], 
                  s=100, c='blue', marker='o', label='Line Start')
        ax.scatter(positions_trajectory[-1, 0], positions_trajectory[-1, 1], positions_trajectory[-1, 2], 
                  s=100, c='red', marker='x', label='Line End')
    
    # Plot centroid points
    points = np.array([c['centroid'] for c in centroids_data])
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='green', s=40, alpha=0.7, label='Centroids')
    
    # Mark start and end points
    ax.scatter(points[0, 0], points[0, 1], points[0, 2], s=100, c='blue', marker='o', label='Centroid Start')
    ax.scatter(points[-1, 0], points[-1, 1], points[-1, 2], s=100, c='red', marker='o', label='Centroid End')
    
    # Set labels and title
    ax.set_xlabel('X axis (meters)')
    ax.set_ylabel('Y axis (meters)')
    ax.set_zlabel('Z axis (meters)')
    ax.set_title('Combined Trajectory Visualization (3D View)')
    ax.legend()
    
    # Save the 3D plot
    if output_dir:
        output_file_3d = os.path.join(output_dir, "combined_trajectory_3d.png")
        plt.savefig(output_file_3d, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved 3D trajectory plot to {output_file_3d}")
    else:
        plt.show()

def save_centroids_to_file(centroids_data, output_dir):
    """Save centroids data to a text file."""
    if not centroids_data:
        return
    
    timestamps = np.array([c['timestamp'] for c in centroids_data])
    points = np.array([c['centroid'] for c in centroids_data])
    
    # Save centroids to file
    output_points = np.column_stack((timestamps, points))
    output_file = os.path.join(output_dir, "combined_centroids.txt")
    np.savetxt(output_file, 
               output_points, 
               fmt='%.6f %f %f %f',
               header="timestamp x y z")
    print(f"Saved centroids data to {output_file}")

if __name__ == "__main__":
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description="Visualize combined trajectory data")
    parser.add_argument("--trajectory_file", type=str, default=None, 
                        help="Path to the trajectory file (line data)")
    parser.add_argument("--json_file", type=str, default=None,
                        help="Path to the JSON file with centroid data (point data)")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Path to save the visualization image (optional)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save visualization outputs (optional)")
    
    # python overall_plot.py --trajectory_file ./KeyFrameTrajectory.txt --json_file ../vid_6_frames_final/final_output_1.json --output_path ../vid_6_frames_final/full_vis_1.png --output_dir ../vid_6_frames_final/trajectory_visualizations_2
    
    args = parser.parse_args()
    
    # Check if at least one input file is provided
    if not args.trajectory_file and not args.json_file:
        parser.error("At least one of --trajectory_file or --json_file must be specified")
    
    # Process the JSON file if provided
    centroids_data = None
    if args.json_file:
        centroids_data = extract_centroids_from_json(args.json_file)
        
        # Save centroids to file if output directory is specified
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            save_centroids_to_file(centroids_data, args.output_dir)
    
    # Create visualization
    visualize_combined_2d_trajectory(
        trajectory_file=args.trajectory_file,
        json_file=args.json_file,
        output_path=args.output_path,
        output_dir=args.output_dir
    )
    
    print("Visualization complete!")