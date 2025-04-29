import os
import sys
import argparse
from typing import List, Dict, Tuple, Optional, Any

# Add the current directory to path
sys.path.append('.')

# Import the original trajectory code
from trajectory_graph_3 import process_trajectory_file, create_trajectory_graph

# Import our new modules
from graph_storage_simple import save_graph, load_graph
from graph_plotter import plot_graph, traverse_to_node

def generate_and_save_graph(trajectory_file: str, output_file: str = "output/graph.pkl"):
    """
    Generate a graph from trajectory data and save it to disk.
    
    Args:
        trajectory_file: Path to the trajectory data file
        output_file: Path to save the graph
    
    Returns:
        The created graph
    """
    print(f"Processing trajectory file: {trajectory_file}")
    
    # Process the trajectory file
    timestamps, positions = process_trajectory_file(file_path=trajectory_file)
    if timestamps is None or positions is None:
        print("Error: Failed to load trajectory data")
        return None
    
    print(f"Loaded {len(timestamps)} data points")
    
    # Create the graph from trajectory data
    graph = create_trajectory_graph(timestamps, positions)
    
    # Save the graph
    save_graph(graph, output_file)
    
    return graph

def traverse_and_plot(graph_file: str, target_node: int, start_node: Optional[int] = None, 
                      view_mode: str = 'global', output_file: str = "output/traversal.png",
                      zoom_radius: float = 5.0):
    """
    Load a graph, traverse to a specific node, and plot the result.
    
    Args:
        graph_file: Path to the saved graph file
        target_node: Target node ID to traverse to
        start_node: Starting node ID (defaults to the first node)
        view_mode: 'global' or 'local' view mode
        output_file: Path to save the plot
        zoom_radius: Radius for local view
    
    Returns:
        Path to the saved plot
    """
    # Load the graph
    graph = load_graph(graph_file)
    
    # Traverse to the target node
    path = traverse_to_node(graph, target_node, start_node)
    
    if not path:
        print("Failed to create path")
        return None
    
    # Create the plot
    plot_path = plot_graph(
        graph=graph,
        output_path=output_file,
        current_position=target_node,
        path=path,
        view_mode=view_mode,
        zoom_center=target_node,
        zoom_radius=zoom_radius,
        node_size=30,
        highlight_size=80,
        path_width=2
    )
    
    return plot_path

def main():
    parser = argparse.ArgumentParser(description="Generate and traverse trajectory graphs")
    
    # Command type
    parser.add_argument("command", choices=["generate", "traverse"],
                        help="Command to execute")
    
    # Generate command arguments
    parser.add_argument("--trajectory_file", type=str,
                        help="Path to the trajectory file (for 'generate' command)")
    parser.add_argument("--graph_output", type=str, default="output/graph.pkl",
                        help="Path to save the generated graph")
    
    # Traverse command arguments
    parser.add_argument("--graph_file", type=str,
                        help="Path to the graph file (for 'traverse' command)")
    parser.add_argument("--target_node", type=int,
                        help="Target node ID to traverse to")
    parser.add_argument("--start_node", type=int, default=None,
                        help="Starting node ID (defaults to first node)")
    parser.add_argument("--view_mode", choices=["global", "local"], default="global",
                        help="View mode for plotting")
    parser.add_argument("--zoom_radius", type=float, default=5.0,
                        help="Radius for local view (in meters)")
    parser.add_argument("--plot_output", type=str, default="output/traversal.png",
                        help="Path to save the plot")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs("output", exist_ok=True)
    
    if args.command == "generate":
        if not args.trajectory_file:
            print("Error: trajectory_file is required for 'generate' command")
            return
        
        graph = generate_and_save_graph(args.trajectory_file, args.graph_output)
        if graph:
            print(f"Graph successfully generated and saved to {args.graph_output}")
            
            # Create a basic global view plot of the full graph
            plot_graph(
                graph=graph,
                output_path=args.graph_output.replace(".pkl", "_global.png"),
                current_position=0,  # Highlight the first node
                view_mode='global'
            )
    
    elif args.command == "traverse":
        if not args.graph_file:
            print("Error: graph_file is required for 'traverse' command")
            return
        
        if args.target_node is None:
            print("Error: target_node is required for 'traverse' command")
            return
        
        plot_path = traverse_and_plot(
            graph_file=args.graph_file,
            target_node=args.target_node,
            start_node=args.start_node,
            view_mode=args.view_mode,
            output_file=args.plot_output,
            zoom_radius=args.zoom_radius
        )
        
        if plot_path:
            print(f"Traversal plot saved to {plot_path}")

if __name__ == "__main__":
    main()s