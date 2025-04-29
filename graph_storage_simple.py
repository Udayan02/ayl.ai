import numpy as np
import pickle
import json
import os
from typing import Dict, Any, List, Tuple

def save_graph(graph, output_path: str):
    """
    Save the trajectory graph to a pickle file.
    
    Args:
        graph: TrajectoryGraph object
        output_path: Path to save the pickle file
    
    Returns:
        Path to the saved file
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Save to pickle file
    with open(output_path, 'wb') as f:
        pickle.dump(graph, f)
    
    print(f"Graph saved to file: {output_path}")
    return output_path

def load_graph(input_path: str):
    """
    Load a trajectory graph from a pickle file.
    
    Args:
        input_path: Path to the pickle file
    
    Returns:
        TrajectoryGraph object
    """
    with open(input_path, 'rb') as f:
        graph = pickle.load(f)
    
    print(f"Graph loaded from file: {input_path}")
    return graph