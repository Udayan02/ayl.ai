import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import argparse
import os
import pickle
import heapq
import json
import itertools
import re
from typing import Tuple, Dict, List, Set, Optional, Any
from PIL import Image


# GraphNode class - REMAINS THE SAME
class GraphNode:
    """A node in the trajectory graph."""
    def __init__(self, id, position):
        self.id = id  # Unique identifier for the node
        self.position = position  # 3D position (x, y, z)
        self.neighbors = {}  # Dictionary mapping node IDs to (neighbor_node, distance)
    
    def add_neighbor(self, neighbor_node, distance):
        """Add a neighbor node with the distance between them."""
        self.neighbors[neighbor_node.id] = (neighbor_node, distance)
    
    def __str__(self):
        return f"Node {self.id} at {self.position} with {len(self.neighbors)} neighbors"


# TrajectoryGraph class - REMAINS THE SAME
class TrajectoryGraph:
    """A graph representation of a trajectory."""
    def __init__(self):
        self.nodes = {}  # Dictionary mapping node IDs to nodes
        self.edges = []  # List to store all edges for debugging
    
    def add_node(self, node_id, position):
        """Add a node to the graph."""
        self.nodes[node_id] = GraphNode(node_id, position)
        return self.nodes[node_id]
    
    def add_edge(self, node1_id, node2_id):
        """Add an edge between two nodes."""
        node1 = self.nodes[node1_id]
        node2 = self.nodes[node2_id]
        
        # Calculate distance between nodes
        distance = np.linalg.norm(node1.position - node2.position)
        
        # Add bidirectional connections
        node1.add_neighbor(node2, distance)
        node2.add_neighbor(node1, distance)
        
        # Store edge information for debugging
        self.edges.append((node1_id, node2_id, distance))
        
    def dijkstra(self, start_id, end_id):
        """Run Dijkstra's algorithm to find the shortest path between two nodes."""
        if start_id not in self.nodes or end_id not in self.nodes:
            return None, float('inf')
        
        # Priority queue for Dijkstra's algorithm
        queue = [(0, start_id)]
        
        # Distance from start to each node
        distances = {node_id: float('inf') for node_id in self.nodes}
        distances[start_id] = 0
        
        # Previous node in optimal path
        previous = {node_id: None for node_id in self.nodes}
        
        while queue:
            current_distance, current_id = heapq.heappop(queue)
            
            # If we've reached the target, we're done
            if current_id == end_id:
                break
            
            # If we've found a worse path, skip
            if current_distance > distances[current_id]:
                continue
            
            # Check all neighbors
            current_node = self.nodes[current_id]
            for neighbor_id, (neighbor, weight) in current_node.neighbors.items():
                distance = current_distance + weight
                
                # If we've found a better path, update
                if distance < distances[neighbor_id]:
                    distances[neighbor_id] = distance
                    previous[neighbor_id] = current_id
                    heapq.heappush(queue, (distance, neighbor_id))
        
        # Reconstruct the path
        path = []
        current_id = end_id
        
        while current_id is not None:
            path.append(current_id)
            current_id = previous[current_id]
        
        path.reverse()
        
        # Return path and total distance
        return path, distances[end_id]


# save_graph and load_graph functions - REMAIN THE SAME
def save_graph(graph, output_path: str):
    """Save the trajectory graph to a pickle file."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(graph, f)
    
    print(f"Graph saved to file: {output_path}")
    return output_path

def load_graph(input_path: str):
    """Load a trajectory graph from a pickle file."""
    with open(input_path, 'rb') as f:
        graph = pickle.load(f)
    
    print(f"Graph loaded from file: {input_path}")
    return graph


# Modified extract_centroids_from_json function to include image paths
def extract_centroids_from_json(json_file):
    """Extract centroids from the JSON file along with timestamps, tags, and image paths."""
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Store all centroids with timestamps, tags, and image paths
    all_centroids = []
    
    for frame_name, frame_data in data.items():
        timestamp = frame_data['timestamp']
        image_path = frame_data.get('image')  # Get image path if it exists
        
        # Process each bounding box in the frame
        for bbox_id, bbox_data in frame_data['bounding_boxes'].items():
            # Skip "unassigned" bounding boxes
            if bbox_id == "unassigned":
                continue
                
            centroid = bbox_data.get('centroid_3d')
            tags = bbox_data.get('tags', [])
            tag = tags[0] if tags else "Unknown"
            
            if centroid is not None:
                # Store timestamp, frame name, bbox ID, centroid coordinates, tag, and image path
                all_centroids.append({
                    'timestamp': timestamp,
                    'frame': frame_name,
                    'bbox_id': bbox_id,
                    'centroid': centroid,
                    'tag': tag,
                    'image': image_path  # Add image path
                })
    
    # Sort all centroids by timestamp
    all_centroids.sort(key=lambda x: x['timestamp'])
    
    return all_centroids


# find_centroids_by_tags function - REMAINS THE SAME
def find_centroids_by_tags(centroids_data, target_tags):
    """Find centroids that match the given tags."""
    tag_to_centroid = {}
    
    # Create a dictionary with lowercase tags for case-insensitive matching
    lowercase_target_tags = [tag.lower() for tag in target_tags]
    
    for centroid in centroids_data:
        tag = centroid['tag']
        
        # Case insensitive comparison
        if tag.lower() in lowercase_target_tags:
            # If we haven't seen this tag yet, or this centroid is more recent
            if tag not in tag_to_centroid or centroid['timestamp'] > tag_to_centroid[tag]['timestamp']:
                tag_to_centroid[tag] = centroid
    
    return tag_to_centroid


# find_closest_node_to_point function - REMAINS THE SAME
def find_closest_node_to_point(graph, point):
    """Find the graph node closest to a given 3D point by Euclidean distance."""
    closest_node_id = None
    min_distance = float('inf')
    
    for node_id, node in graph.nodes.items():
        distance = np.linalg.norm(np.array(node.position) - np.array(point))
        
        if distance < min_distance:
            min_distance = distance
            closest_node_id = node_id
    
    return closest_node_id, min_distance


# find_path_through_special_nodes function - REMAINS THE SAME
def find_path_through_special_nodes(graph, start_node, special_nodes, end_node):
    """Find an optimal path that passes through all special nodes."""
    # If no special nodes, just find direct path
    if not special_nodes:
        return graph.dijkstra(start_node, end_node)
    
    # Try all permutations of special nodes to find the best order
    # Add start and end nodes
    all_permutations = []
    for perm in itertools.permutations(special_nodes):
        all_permutations.append((start_node,) + perm + (end_node,))
    
    best_path = None
    best_distance = float('inf')
    
    # For each permutation, calculate total distance
    for nodes in all_permutations:
        total_distance = 0
        current_path = []
        
        # Connect all segments
        for i in range(len(nodes) - 1):
            from_node = nodes[i]
            to_node = nodes[i + 1]
            
            segment_path, segment_distance = graph.dijkstra(from_node, to_node)
            
            if segment_path is None:
                # This permutation is invalid, no path exists for one segment
                total_distance = float('inf')
                break
            
            # Add segment path (excluding the first node after the first segment to avoid duplicates)
            if i == 0:
                current_path.extend(segment_path)
            else:
                current_path.extend(segment_path[1:])
            
            total_distance += segment_distance
        
        # Update best path if this one is better
        if total_distance < best_distance:
            best_distance = total_distance
            best_path = current_path
    
    return best_path, best_distance


# Function to add images to JSON
def add_image_references(json_file, image_dir):
    """Add image references to the JSON data."""
    # Load the JSON data
    with open(json_file, 'r') as f:
        json_data = json.load(f)
    
    # Get all image files in the directory
    image_files = os.listdir(image_dir)
    
    # Create a mapping from frame numbers to image files
    frame_to_image = {}
    for image_file in image_files:
        # Try to extract the frame number from the image file name
        match = re.search(r'(\d{3,4})(?=\D|$)', image_file)
        if match:
            frame_digits = match.group(1)
            # Pad with zeros to match the format in JSON
            for frame_key in json_data:
                if frame_key.endswith(frame_digits.zfill(6)) or frame_key.endswith(frame_digits):
                    frame_to_image[frame_key] = image_file
    
    # Add image references to the JSON data
    for frame_key in json_data:
        if frame_key in frame_to_image:
            json_data[frame_key]["image"] = os.path.join(image_dir, frame_to_image[frame_key])
        else:
            # If no image is found, we can set a placeholder
            json_data[frame_key]["image"] = None
    
    # Save the updated JSON
    output_json_path = 'output/filtered_output_with_images.json'
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    print(f"Updated JSON saved to {output_json_path}")
    return json_data


# Modified plot_graph function to include images
def plot_graph(graph, output_path, 
               current_position=None, path=None, 
               view_mode='global', 
               zoom_center=None, zoom_radius=None,
               node_size=10, node_alpha=0.7,
               highlight_size=80, highlight_alpha=1.0,
               path_width=2, path_alpha=0.9,
               edge_width=1, edge_alpha=0.5,
               show_node_labels=True,
               centroids_data=None,
               special_nodes=None,
               show_images=False,
               image_size=0.3):
    """
    Create a static plot of the graph with customizable parameters and optional image overlays.
    
    Args:
        ... (all previous parameters)
        show_images: Whether to display images from centroids data
        image_size: Size of the displayed images (as fraction of the plot)
    """
    # Set up the plot
    plt.figure(figsize=(12, 10))
    
    # Make sure we have valid values
    if current_position is None:
        current_position = min(graph.nodes.keys())
    
    if zoom_center is None:
        zoom_center = current_position
    
    if zoom_radius is None:
        zoom_radius = 5.0  # Default 5m radius
    
    # Get the center node for zoomed view
    center_node = graph.nodes[zoom_center]
    
    # Plot all nodes - but only if they're in view range for local view
    for node_id, node in graph.nodes.items():
        x, y, z = node.position
        
        # In local view, only show nodes within radius
        if view_mode == 'local':
            distance = np.linalg.norm(node.position - center_node.position)
            if distance > zoom_radius:
                continue
        
        # Plot the node
        plt.scatter(x, z, s=node_size, c='cyan', alpha=node_alpha)
        
        # Add node ID label if requested
        if show_node_labels:
            plt.text(x, z, str(node_id), fontsize=8, ha='center', va='bottom')
    
    # Plot all edges - but only if they're in view range for local view
    for edge in graph.edges:
        node1_id, node2_id, _ = edge
        node1 = graph.nodes[node1_id]
        node2 = graph.nodes[node2_id]
        
        # In local view, only show edges if both nodes are within radius
        if view_mode == 'local':
            distance1 = np.linalg.norm(node1.position - center_node.position)
            distance2 = np.linalg.norm(node2.position - center_node.position)
            if distance1 > zoom_radius or distance2 > zoom_radius:
                continue
        
        plt.plot([node1.position[0], node2.position[0]],
                 [node1.position[2], node2.position[2]],
                 'g-', linewidth=edge_width, alpha=edge_alpha)
    
    # Highlight special nodes if provided
    if special_nodes:
        for node_id in special_nodes:
            if node_id in graph.nodes:
                node = graph.nodes[node_id]
                plt.scatter(node.position[0], node.position[2], 
                          s=highlight_size, c='purple', alpha=highlight_alpha,
                          marker='s', label='Special Node' if node_id == special_nodes[0] else "")
                
                # Add a more noticeable label
                if show_node_labels:
                    plt.text(node.position[0], node.position[2] + 0.3, 
                           f"Node {node_id}", fontsize=10, 
                           ha='center', va='bottom', weight='bold', color='purple')
    
    # Highlight the current position
    if current_position in graph.nodes:
        current_node = graph.nodes[current_position]
        plt.scatter(current_node.position[0], current_node.position[2], 
                  s=highlight_size, c='red', alpha=highlight_alpha,
                  label='Current Position')
    
    # For path, we need to handle it differently in local view
    if path and len(path) > 1:
        if view_mode == 'global':
            # Plot the full path in global view
            path_x = []
            path_z = []
            for node_id in path:
                if node_id in graph.nodes:
                    node = graph.nodes[node_id]
                    path_x.append(node.position[0])
                    path_z.append(node.position[2])
            
            plt.plot(path_x, path_z, 'r-', linewidth=path_width, alpha=path_alpha, 
                   label='Path')
        else:
            # In local view, only plot path segments that have at least one endpoint in view
            for i in range(len(path) - 1):
                if path[i] in graph.nodes and path[i+1] in graph.nodes:
                    node1 = graph.nodes[path[i]]
                    node2 = graph.nodes[path[i+1]]
                    
                    # Check if at least one endpoint is within range
                    distance1 = np.linalg.norm(node1.position - center_node.position)
                    distance2 = np.linalg.norm(node2.position - center_node.position)
                    
                    if distance1 <= zoom_radius or distance2 <= zoom_radius:
                        plt.plot([node1.position[0], node2.position[0]],
                               [node1.position[2], node2.position[2]],
                               'r-', linewidth=path_width, alpha=path_alpha,
                               label='Path' if i == 0 else "")
    
    # Plot centroids if available
    if centroids_data and len(centroids_data) > 0:
        # Extract positions and tags
        centroid_x = []
        centroid_z = []
        tags = []
        images_with_positions = []  # Store images with their positions
        
        for centroid in centroids_data:
            pos = centroid['centroid']
            x, _, z = pos  # Using only x and z for 2D visualization
            
            # In local view, check if centroid is within radius
            if view_mode == 'local':
                distance = np.linalg.norm([x, 0, z] - center_node.position)
                if distance > zoom_radius:
                    continue
            
            centroid_x.append(x)
            centroid_z.append(z)
            tags.append(centroid['tag'])
            
            # Add image information if available
            if show_images and 'image' in centroid and centroid['image'] is not None:
                images_with_positions.append((x, z, centroid['image'], centroid['tag']))
        
        # Plot centroids
        plt.scatter(centroid_x, centroid_z, s=60, c='green', alpha=0.7, marker='*', label='Object Centroids')
        
        # Add small tag labels
        for i, (x, z, tag) in enumerate(zip(centroid_x, centroid_z, tags)):
            if show_images:
                # plt.text(x, z + 1.2, tag, color='green', fontsize=7, ha='center', va='bottom', alpha=1)
                pass
            else:
                plt.text(x, z + 0.2, tag, color='green', fontsize=7, ha='center', va='bottom', alpha=1)
        
        # Add images if requested
        if show_images:
            for x, z, img_path, tag in images_with_positions:
                try:
                    # Load and display the image if it exists
                    if os.path.exists(img_path):
                        img = plt.imread(img_path)
                        imagebox = OffsetImage(img, zoom=image_size)
                        ab = AnnotationBbox(imagebox, (x, z), frameon=False, 
                                         pad=0.1, boxcoords="data")
                        plt.gca().add_artist(ab)
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")
    
    # Set axis limits for local view
    if view_mode == 'local':
        center_x, _, center_z = center_node.position
        plt.xlim(center_x - zoom_radius, center_x + zoom_radius)
        plt.ylim(center_z - zoom_radius, center_z + zoom_radius)
    
    # Add labels and title
    mode_label = "Local" if view_mode == 'local' else "Global"
    # plt.title(f"{mode_label} View - Current Position: Node {current_position}")
    # plt.xlabel('X (meters)')
    # plt.ylabel('Z (meters)')
    # plt.grid(True)
    plt.axis('equal')
    current_ax = plt.gca()

    # Remove x-axis ticks and labels
    current_ax.set_xticks([])
    current_ax.set_xticklabels([])

    # Remove y-axis ticks and labels
    current_ax.set_yticks([])
    current_ax.set_yticklabels([])

    current_ax.spines['top'].set_visible(False)
    current_ax.spines['right'].set_visible(False)
    current_ax.spines['bottom'].set_visible(False)
    current_ax.spines['left'].set_visible(False)
    
    # Handle legend - only show items actually in the plot
    # handles, labels = plt.gca().get_legend_handles_labels()
    # by_label = dict(zip(labels, handles))
    # if by_label:
    #     plt.legend(by_label.values(), by_label.keys())
    
    # Make sure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Save the figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved to: {output_path}")
    return output_path


# Modified traverse_and_plot_with_tags function to include image support
def traverse_and_plot_with_tags(graph_file: str, target_tags: List[str],
                      start_node: Optional[int] = None, 
                      target_node: Optional[int] = None,
                      view_mode: str = 'global', 
                      output_file: str = "output/traversal.png",
                      zoom_radius: float = 5.0, 
                      show_node_labels: bool = True,
                      json_file: Optional[str] = None,
                      show_images: bool = False,
                      image_size: float = 0.3):
    """
    Find centroids matching tags, map to closest nodes, and plan a path through them.
    
    Args:
        ... (all previous parameters)
        show_images: Whether to display images in the visualization
        image_size: Size factor for the displayed images
    """
    # Load the graph from file
    graph = load_graph(graph_file)
    
    # Use the minimum node ID as default start if not specified
    if start_node is None:
        start_node = min(graph.nodes.keys())
    
    # Use the maximum node ID as default target if not specified
    if target_node is None:
        target_node = max(graph.nodes.keys())
    
    # Check for valid start and target nodes
    if start_node not in graph.nodes:
        print(f"Error: Start node {start_node} not found in graph!")
        return None
    
    if target_node not in graph.nodes:
        print(f"Error: Target node {target_node} not found in graph!")
        return None
    
    # If no JSON file is provided, we can't proceed with tag matching
    if not json_file:
        print("Error: JSON file required for tag-based navigation!")
        return None
    
    # Load centroids data from JSON file
    centroids_data = extract_centroids_from_json(json_file)
    print(f"Loaded {len(centroids_data)} centroids from {json_file}")
    
    # Find centroids matching the target tags
    tag_to_centroid = find_centroids_by_tags(centroids_data, target_tags)
    
    if not tag_to_centroid:
        print(f"Warning: No centroids found for tags: {target_tags}")
        # Fall back to regular path finding
        # path, distance = graph.dijkstra(start_node, target_node)
        path, distance = graph.dijkstra(start_node, start_node)
        special_nodes = None
        if not path:
            print("Failed to create path")
            return None
    else:
        print(f"Found centroids for these tags: {list(tag_to_centroid.keys())}")
        
        # Map centroids to closest nodes in the graph
        tag_to_node = {}
        special_nodes = []
        
        for tag, centroid_data in tag_to_centroid.items():
            centroid_pos = centroid_data['centroid']
            node_id, distance = find_closest_node_to_point(graph, centroid_pos)
            
            tag_to_node[tag] = (node_id, distance)
            special_nodes.append(node_id)
            
            print(f"Tag '{tag}' mapped to node {node_id} (distance: {distance:.2f}m)")
        
        # Find path through special nodes
        path, distance = find_path_through_special_nodes(graph, start_node, special_nodes, target_node)
        
        if not path:
            print("Failed to create path through special nodes")
            return None
        
        print(f"Found path through {len(special_nodes)} special nodes with total distance {distance:.2f}m")
    
    # Plot the graph with the path, special nodes, and images if requested
    return plot_graph(
        graph=graph,
        output_path=output_file,
        current_position=start_node,
        path=path,
        view_mode=view_mode,
        zoom_center=start_node,
        zoom_radius=zoom_radius,
        node_size=30,
        highlight_size=250,
        path_width=4.5,
        show_node_labels=show_node_labels,
        centroids_data=centroids_data,
        special_nodes=special_nodes,
        show_images=show_images,
        image_size=image_size
    )


def main():
    parser = argparse.ArgumentParser(description="Navigate and visualize trajectory graphs")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Traverse command (original version)
    traverse_parser = subparsers.add_parser("traverse", help="Compute path between nodes and visualize")
    traverse_parser.add_argument("--graph_file", type=str, required=True,
                              help="Path to the saved graph pickle file")
    traverse_parser.add_argument("--target_node", type=int, required=False,
                              help="Target node ID to navigate to (defaults to highest ID node)")
    traverse_parser.add_argument("--start_node", type=int, default=None,
                              help="Starting node ID (defaults to lowest ID node)")
    traverse_parser.add_argument("--view_mode", choices=["global", "local"], default="global",
                              help="View mode for plotting")
    traverse_parser.add_argument("--zoom_radius", type=float, default=5.0,
                              help="Radius for local view (in meters)")
    traverse_parser.add_argument("--output_file", type=str, default="output/traversal.png",
                              help="Path to save the visualization image")
    traverse_parser.add_argument("--hide_labels", action="store_true",
                              help="Hide node ID labels in visualization")
    traverse_parser.add_argument("--json_file", type=str, default=None,
                              help="Path to JSON file containing object centroids and tags")
    traverse_parser.add_argument("--show_images", action="store_true",
                              help="Display images from the JSON data in the visualization")
    traverse_parser.add_argument("--image_dir", type=str, default=None,
                              help="Directory containing images to reference in the JSON")
    traverse_parser.add_argument("--image_size", type=float, default=0.3,
                              help="Size factor for displayed images (default: 0.3)")
    
    # Tag-based traverse command (with image support)
    tag_traverse_parser = subparsers.add_parser("tag_traverse", 
                                           help="Find path through nodes closest to specified tags")
    tag_traverse_parser.add_argument("--graph_file", type=str, required=True,
                                  help="Path to the saved graph pickle file")
    tag_traverse_parser.add_argument("--json_file", type=str, required=True,
                                  help="Path to JSON file containing object centroids and tags")
    tag_traverse_parser.add_argument("--tags", type=str, nargs='+', required=True,
                                  help="List of tags to find (e.g. --tags 'Apple' 'Banana' 'Orange')")
    tag_traverse_parser.add_argument("--start_node", type=int, default=None,
                                  help="Starting node ID (defaults to lowest ID node)")
    tag_traverse_parser.add_argument("--target_node", type=int, default=None,
                                  help="Final target node ID (defaults to highest ID node)")
    tag_traverse_parser.add_argument("--view_mode", choices=["global", "local"], default="global",
                                  help="View mode for plotting")
    tag_traverse_parser.add_argument("--zoom_radius", type=float, default=5.0,
                                  help="Radius for local view (in meters)")
    tag_traverse_parser.add_argument("--output_file", type=str, default="output/tag_traversal.png",
                                  help="Path to save the visualization image")
    tag_traverse_parser.add_argument("--hide_labels", action="store_true",
                                  help="Hide node ID labels in visualization")
    tag_traverse_parser.add_argument("--show_images", action="store_true",
                                  help="Display images from the JSON data in the visualization")
    tag_traverse_parser.add_argument("--image_dir", type=str, default=None,
                                  help="Directory containing images to reference in the JSON")
    tag_traverse_parser.add_argument("--image_size", type=float, default=0.3,
                                  help="Size factor for displayed images (default: 0.3)")
    
    # Add images command
    add_images_parser = subparsers.add_parser("add_images", 
                                          help="Add image references to JSON file")
    add_images_parser.add_argument("--json_file", type=str, required=True,
                                help="Path to the input JSON file")
    add_images_parser.add_argument("--image_dir", type=str, required=True,
                                help="Directory containing images to reference")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Make sure output directory exists
    os.makedirs("output", exist_ok=True)
    
    # Execute appropriate command
    if args.command == "traverse":
        
        # Change behavior to make target_node default to the highest node ID
        target_node = args.target_node
        if target_node is None:
            graph = load_graph(args.graph_file)
            target_node = max(graph.nodes.keys())
            print(f"Using default target node: {target_node}")
            
        # If image_dir is provided, update the JSON file to include image paths
        json_file = args.json_file
        if args.json_file and args.image_dir and args.show_images:
            print(f"Adding image references from {args.image_dir} to JSON...")
            # Update the JSON and get the modified data
            add_image_references(args.json_file, args.image_dir)
            # Use the updated JSON file
            json_file = 'output/filtered_output_with_images.json'
            
        # Load the graph and run the traverse_and_plot_with_tags function
        # Note: We're using the enhanced function even for regular traversal for consistency
        output_path = traverse_and_plot_with_tags(
            graph_file=args.graph_file,
            target_tags=None,
            start_node=args.start_node,
            target_node=args.target_node,
            view_mode=args.view_mode,
            output_file=args.output_file,
            zoom_radius=args.zoom_radius,
            show_node_labels=not args.hide_labels,
            json_file=json_file,
            show_images=args.show_images,
            image_size=args.image_size
        )
        pass
    elif args.command == 'tag_traverse':
        output_path = traverse_and_plot_with_tags(
            graph_file=args.graph_file,
            target_tags=args.tags,
            start_node=args.start_node,
            target_node=args.target_node,
            view_mode=args.view_mode,
            output_file=args.output_file,
            zoom_radius=args.zoom_radius,
            show_node_labels=not args.hide_labels,
            json_file=args.json_file,
            show_images=args.show_images,
            image_size=args.image_size
        )
        if output_path:
            print(f"Tag-based traversal plot saved to {output_path}")

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
