# import numpy as np
# import matplotlib.pyplot as plt
# from typing import List, Dict, Tuple, Optional
# import os

# def plot_graph(graph, output_path, 
#                current_position=None, path=None, 
#                view_mode='global', 
#                zoom_center=None, zoom_radius=None,
#                node_size=30, node_alpha=0.7,
#                highlight_size=80, highlight_alpha=1.0,
#                path_width=2, path_alpha=0.9,
#                edge_width=1, edge_alpha=0.5):
#     """
#     Create a static plot of the graph with customizable parameters.
    
#     Args:
#         graph: TrajectoryGraph object
#         output_path: Path to save the image
#         current_position: Node ID of current position to highlight
#         path: List of node IDs representing the path traversed
#         view_mode: 'global' to show the entire graph, 'local' to zoom in
#         zoom_center: Node ID to center the view on (if view_mode is 'local')
#         zoom_radius: Radius in meters for local view
#         node_size: Size of regular nodes
#         node_alpha: Alpha transparency of regular nodes
#         highlight_size: Size of highlighted nodes
#         highlight_alpha: Alpha transparency of highlighted nodes
#         path_width: Width of the path line
#         path_alpha: Alpha transparency of the path
#         edge_width: Width of graph edges
#         edge_alpha: Alpha transparency of graph edges
    
#     Returns:
#         Path to the saved image
#     """
#     # Set up the plot
#     plt.figure(figsize=(12, 10))
    
#     # Make sure we have valid values
#     if current_position is None:
#         current_position = min(graph.nodes.keys())
    
#     if zoom_center is None:
#         zoom_center = current_position
    
#     if zoom_radius is None:
#         zoom_radius = 5.0  # Default 5m radius
    
#     # Get the center node for zoomed view
#     center_node = graph.nodes[zoom_center]
    
#     # Plot all nodes
#     for node_id, node in graph.nodes.items():
#         x, y, z = node.position
        
#         # In local view, only show nodes within radius
#         if view_mode == 'local':
#             distance = np.linalg.norm(node.position - center_node.position)
#             if distance > zoom_radius:
#                 continue
        
#         # Plot the node
#         plt.scatter(x, z, s=node_size, c='blue', alpha=node_alpha)
        
#         # Add node ID label
#         plt.text(x, z, str(node_id), fontsize=8, ha='center', va='bottom')
    
#     # Plot all edges
#     for edge in graph.edges:
#         node1_id, node2_id, _ = edge
#         node1 = graph.nodes[node1_id]
#         node2 = graph.nodes[node2_id]
        
#         # In local view, only show edges if both nodes are within radius
#         if view_mode == 'local':
#             distance1 = np.linalg.norm(node1.position - center_node.position)
#             distance2 = np.linalg.norm(node2.position - center_node.position)
#             if distance1 > zoom_radius or distance2 > zoom_radius:
#                 continue
        
#         plt.plot([node1.position[0], node2.position[0]],
#                  [node1.position[2], node2.position[2]],
#                  'g-', linewidth=edge_width, alpha=edge_alpha)
    
#     # Highlight the current position
#     if current_position in graph.nodes:
#         current_node = graph.nodes[current_position]
#         plt.scatter(current_node.position[0], current_node.position[2], 
#                    s=highlight_size, c='red', alpha=highlight_alpha,
#                    label='Current Position')
    
#     # Plot the path if provided
#     if path and len(path) > 1:
#         path_x = []
#         path_z = []
#         for node_id in path:
#             if node_id in graph.nodes:
#                 node = graph.nodes[node_id]
#                 path_x.append(node.position[0])
#                 path_z.append(node.position[2])
        
#         plt.plot(path_x, path_z, 'r-', linewidth=path_width, alpha=path_alpha, 
#                 label='Path')
    
#     # Set axis limits for local view
#     if view_mode == 'local':
#         center_x, _, center_z = center_node.position
#         plt.xlim(center_x - zoom_radius, center_x + zoom_radius)
#         plt.ylim(center_z - zoom_radius, center_z + zoom_radius)
    
#     # Add labels and title
#     mode_label = "Local" if view_mode == 'local' else "Global"
#     plt.title(f"{mode_label} View - Current Position: Node {current_position}")
#     plt.xlabel('X (meters)')
#     plt.ylabel('Z (meters)')
#     plt.grid(True)
#     plt.axis('equal')
#     plt.legend()
    
#     # Make sure output directory exists
#     os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
#     # Save the figure
#     plt.savefig(output_path, dpi=300, bbox_inches='tight')
#     plt.close()
    
#     print(f"Plot saved to: {output_path}")
#     return output_path

# def traverse_to_node(graph, target_node, start_node=None):
#     """
#     Traverse the graph to a target node from a start node.
#     Returns the path taken.
    
#     Args:
#         graph: TrajectoryGraph object
#         target_node: Target node ID
#         start_node: Starting node ID (defaults to first node)
    
#     Returns:
#         List of node IDs representing the path
#     """
#     if start_node is None:
#         start_node = min(graph.nodes.keys())
    
#     if start_node not in graph.nodes or target_node not in graph.nodes:
#         print(f"Invalid start or target node: {start_node}, {target_node}")
#         return []
    
#     # For sequential traversal, just create a path between start and target
#     path = []
#     if start_node <= target_node:
#         path = list(range(start_node, target_node + 1))
#     else:
#         path = list(range(start_node, target_node - 1, -1))
    
#     # Verify all nodes in path exist
#     valid_path = [node_id for node_id in path if node_id in graph.nodes]
    
#     print(f"Traversed from node {start_node} to node {target_node}")
#     return valid_path

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
import os

def plot_graph(graph, output_path, 
               current_position=None, path=None, 
               view_mode='global', 
               zoom_center=None, zoom_radius=None,
               node_size=10, node_alpha=0.7, # node_size = 30
               highlight_size=80, highlight_alpha=1.0,
               path_width=2, path_alpha=0.9,
               edge_width=1, edge_alpha=0.5):
    """
    Create a static plot of the graph with customizable parameters.
    
    Args:
        graph: TrajectoryGraph object
        output_path: Path to save the image
        current_position: Node ID of current position to highlight
        path: List of node IDs representing the path traversed
        view_mode: 'global' to show the entire graph, 'local' to zoom in
        zoom_center: Node ID to center the view on (if view_mode is 'local')
        zoom_radius: Radius in meters for local view
        node_size: Size of regular nodes
        node_alpha: Alpha transparency of regular nodes
        highlight_size: Size of highlighted nodes
        highlight_alpha: Alpha transparency of highlighted nodes
        path_width: Width of the path line
        path_alpha: Alpha transparency of the path
        edge_width: Width of graph edges
        edge_alpha: Alpha transparency of graph edges
    
    Returns:
        Path to the saved image
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
        plt.scatter(x, z, s=node_size, c='blue', alpha=node_alpha)
        
        # Add node ID label
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
    
    # Set axis limits for local view
    if view_mode == 'local':
        center_x, _, center_z = center_node.position
        plt.xlim(center_x - zoom_radius, center_x + zoom_radius)
        plt.ylim(center_z - zoom_radius, center_z + zoom_radius)
    
    # Add labels and title
    mode_label = "Local" if view_mode == 'local' else "Global"
    plt.title(f"{mode_label} View - Current Position: Node {current_position}")
    plt.xlabel('X (meters)')
    plt.ylabel('Z (meters)')
    plt.grid(True)
    plt.axis('equal')
    
    # Handle legend - only show items actually in the plot
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if by_label:
        plt.legend(by_label.values(), by_label.keys())
    
    # Make sure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Save the figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved to: {output_path}")
    return output_path

def traverse_to_node(graph, target_node, start_node=None):
    """
    Traverse the graph to a target node from a start node.
    Returns the path taken.
    
    Args:
        graph: TrajectoryGraph object
        target_node: Target node ID
        start_node: Starting node ID (defaults to first node)
    
    Returns:
        List of node IDs representing the path
    """
    if start_node is None:
        start_node = min(graph.nodes.keys())
    
    if start_node not in graph.nodes or target_node not in graph.nodes:
        print(f"Invalid start or target node: {start_node}, {target_node}")
        return []
    
    # For sequential traversal, just create a path between start and target
    path = []
    if start_node <= target_node:
        path = list(range(start_node, target_node + 1))
    else:
        path = list(range(start_node, target_node - 1, -1))
    
    # Verify all nodes in path exist
    valid_path = [node_id for node_id in path if node_id in graph.nodes]
    
    print(f"Traversed from node {start_node} to node {target_node}")
    return valid_path

def find_optimal_path(graph, start_node, target_node):
    """
    Find the optimal path using Dijkstra's algorithm.
    
    Args:
        graph: TrajectoryGraph object
        start_node: Starting node ID
        target_node: Target node ID
        
    Returns:
        List of node IDs representing the optimal path
    """
    # Check for necessary method
    if hasattr(graph, 'dijkstra'):
        path, distance = graph.dijkstra(start_node, target_node)
        if path:
            print(f"Found optimal path from {start_node} to {target_node} with distance {distance:.2f}m")
            return path
    
    # Fallback to sequential path if dijkstra is not available
    print("Dijkstra's algorithm not available, using sequential path")
    return traverse_to_node(graph, target_node, start_node)