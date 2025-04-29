import numpy as np
import matplotlib.pyplot as plt
import argparse
from typing import Tuple, Dict, List, Set
import heapq  # For Dijkstra's algorithm

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
        print(f"Added edge between node {node1_id} and node {node2_id} with distance {distance:.4f}")
    
    def detect_loops(self, distance_threshold=1.0):
        """
        Detect loops in the trajectory by finding nodes that are close to each other
        but not adjacent in the sequence.
        """
        loop_connections = []
        
        # Get list of nodes sorted by ID
        node_ids = sorted(self.nodes.keys())
        
        print(f"Looking for loops with distance threshold {distance_threshold}")
        
        # Check each pair of nodes that aren't adjacent in sequence
        for i, id1 in enumerate(node_ids):
            node1 = self.nodes[id1]
            
            for j, id2 in enumerate(node_ids):
                # Skip if nodes are the same or adjacent in sequence
                if abs(i - j) <= 1 or i == j:
                    continue
                
                node2 = self.nodes[id2]
                distance = np.linalg.norm(node1.position - node2.position)
                
                # If nodes are close enough, they might be part of a loop
                if distance < distance_threshold:
                    loop_connections.append((id1, id2, distance))
                    print(f"Potential loop detected: node {id1} and node {id2} with distance {distance:.4f}")
        
        print(f"Found {len(loop_connections)} potential loop connections")
        return loop_connections
    
    def visualize(self, output_path=None, show_all_edges=True):
        """Visualize the graph in 2D (X-Z plane)."""
        plt.figure(figsize=(12, 10))
        
        # Collect node positions
        x_coords = []
        z_coords = []
        node_ids = []
        
        for node_id, node in self.nodes.items():
            x_coords.append(node.position[0])
            z_coords.append(node.position[2])
            node_ids.append(node_id)
        
        # Plot nodes
        plt.scatter(x_coords, z_coords, c='blue', s=30, label='Nodes')
        
        # Plot sequential edges
        for i in range(len(self.nodes) - 1):
            node1 = self.nodes[i]
            node2 = self.nodes[i + 1]
            plt.plot([node1.position[0], node2.position[0]],
                     [node1.position[2], node2.position[2]],
                     'g-', linewidth=1.5, alpha=0.8, label='Sequential Edge' if i == 0 else "")
        
        # Mark start and end nodes
        start_node = self.nodes[min(self.nodes.keys())]
        end_node = self.nodes[max(self.nodes.keys())]
        
        plt.plot(start_node.position[0], start_node.position[2], 'go', markersize=10, label='Start')
        plt.plot(end_node.position[0], end_node.position[2], 'ro', markersize=10, label='End')
        
        plt.xlabel('X (meters)')
        plt.ylabel('Z (meters)')
        plt.title('Trajectory Graph Visualization')
        plt.grid(True)
        plt.axis('equal')
        
        # Only show legend items once
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Graph visualization saved to: {output_path}")
        else:
            plt.show()
        
        # Print edge statistics
        print(f"\nGraph Edge Statistics:")
        print(f"Total edges: {len(self.edges)}")
        if show_all_edges:
            print("\nAll Edges (node1, node2, distance):")
            for edge in sorted(self.edges):
                print(f"  {edge[0]} -- {edge[1]}: {edge[2]:.4f}m")
    
    def dijkstra(self, start_id, end_id):
        """
        Run Dijkstra's algorithm to find the shortest path between two nodes.
        
        Args:
            start_id: ID of the starting node
            end_id: ID of the target node
            
        Returns:
            Tuple of (path, distance) where path is a list of node IDs
        """
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

def process_trajectory_file(file_path=None):
    """Process a trajectory file containing timestamps and positions."""
    data = np.loadtxt(file_path)
    if data.size == 0:
        return None, None
    timestamps = data[:, 0]
    positions = data[:, 1:4]
    return timestamps, positions

def create_trajectory_graph(timestamps, positions):
    """
    Create a graph structure from trajectory data with only sequential connections.
    
    Args:
        timestamps: Array of timestamps
        positions: Array of 3D positions
        
    Returns:
        TrajectoryGraph object
    """
    graph = TrajectoryGraph()
    
    # Add nodes for all points in the trajectory
    for i in range(len(timestamps)):
        graph.add_node(i, positions[i])
    
    # Connect sequential nodes only
    print("\nAdding sequential connections:")
    for i in range(len(timestamps) - 1):
        graph.add_edge(i, i + 1)
    
    return graph

def visualize_2d_trajectory(file_path=None, output_path=None, show_all_edges=True):
    """
    Visualize the trajectory in 2D and create a graph structure.
    
    Args:
        file_path: Path to the trajectory file
        output_path: Path to save the visualization
        show_all_edges: Whether to print all edges for debugging
    
    Returns:
        TrajectoryGraph object representing the trajectory
    """
    timestamps, positions = process_trajectory_file(file_path=file_path)
    
    # Create graph from trajectory data with only sequential connections
    graph = create_trajectory_graph(timestamps, positions)
    
    # Visualize the original trajectory
    plt.figure(figsize=(10, 8))
    
    # Plot top-down view (X-Z plane)
    plt.plot(positions[:, 0], positions[:, 2], 'b-', label='Trajectory')
    plt.plot(positions[0, 0], positions[0, 2], 'go', label='Start')
    plt.plot(positions[-1, 0], positions[-1, 2], 'ro', label='End')
    
    plt.xlabel('X (meters)')
    plt.ylabel('Z (meters)')
    plt.title('Original Trajectory')
    plt.axis('equal')  # Equal scaling
    plt.grid(True)
    plt.legend()
    
    if output_path:
        # Save the trajectory visualization
        trajectory_output = output_path.replace('.', '_trajectory.')
        plt.savefig(trajectory_output, dpi=300, bbox_inches='tight')
        print(f"Trajectory visualization saved to: {trajectory_output}")
        
        # Save the graph visualization
        graph_output = output_path.replace('.', '_graph.')
        graph.visualize(output_path=graph_output, show_all_edges=show_all_edges)
    else:
        plt.show()
        # Show the graph in a separate figure
        graph.visualize(show_all_edges=show_all_edges)
    
    return graph

def find_path(graph, start_id, end_id):
    """Find and visualize the shortest path between two nodes."""
    path, distance = graph.dijkstra(start_id, end_id)
    
    if path:
        print(f"Shortest path from {start_id} to {end_id}:")
        print(f"Path: {path}")
        print(f"Distance: {distance:.2f} meters")
        
        # Visualize the path
        plt.figure(figsize=(12, 10))
        
        # Plot all nodes
        for node_id, node in graph.nodes.items():
            plt.plot(node.position[0], node.position[2], 'bo', markersize=5, alpha=0.3)
        
        # Plot all edges
        for node_id, node in graph.nodes.items():
            for neighbor_id, (neighbor, _) in node.neighbors.items():
                if node_id < neighbor_id:  # Only draw each edge once
                    # Draw sequential edges in green
                    if abs(node_id - neighbor_id) == 1:
                        plt.plot([node.position[0], neighbor.position[0]],
                                [node.position[2], neighbor.position[2]],
                                'g-', linewidth=0.5, alpha=0.4)
                    # Draw loop edges in red dashed lines
                    else:
                        plt.plot([node.position[0], neighbor.position[0]],
                                [node.position[2], neighbor.position[2]],
                                'r--', linewidth=0.5, alpha=0.4)
        
        # Plot the shortest path
        path_x = []
        path_z = []
        for node_id in path:
            node = graph.nodes[node_id]
            path_x.append(node.position[0])
            path_z.append(node.position[2])
        
        plt.plot(path_x, path_z, 'b-', linewidth=2.5, label=f'Shortest Path ({distance:.2f}m)')
        
        # Mark start and end nodes
        plt.plot(path_x[0], path_z[0], 'go', markersize=10, label='Start')
        plt.plot(path_x[-1], path_z[-1], 'ro', markersize=10, label='End')
        
        plt.xlabel('X (meters)')
        plt.ylabel('Z (meters)')
        plt.title('Shortest Path Visualization')
        plt.grid(True)
        plt.axis('equal')
        plt.legend()
        
        plt.show()
    else:
        print(f"No path found from {start_id} to {end_id}")

if __name__ == "__main__":
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description="Create a graph from trajectory data")
    parser.add_argument("--trajectory_file", type=str, required=True, 
                        help="Path to the trajectory file")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Path to save the visualization image (optional)")
    parser.add_argument("--find_path", action="store_true",
                        help="Find the shortest path between two nodes")
    parser.add_argument("--start_node", type=int, default=0,
                        help="Starting node ID for path finding")
    parser.add_argument("--end_node", type=int, default=-1,
                        help="Ending node ID for path finding")
    parser.add_argument("--show_all_edges", action="store_true", default=True,
                        help="Print all edges for debugging")
    
    args = parser.parse_args()
    
    # Create trajectory graph
    graph = visualize_2d_trajectory(
        file_path=args.trajectory_file,
        output_path=args.output_path,
        show_all_edges=args.show_all_edges
    )
    
    # Find path if requested
    if args.find_path:
        end_node = args.end_node
        if end_node < 0:
            end_node = len(graph.nodes) - 1
        
        find_path(graph, args.start_node, end_node)