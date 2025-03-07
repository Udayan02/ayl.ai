import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import open3d as o3d


def depth_to_point_cloud(img_src: str, depth_src: str, grayscale: bool, focal_length: float = None, debug: bool = True):
    # Load the image and depth map
    img = np.array(Image.open(img_src).convert('RGB'))  # Convert RGBA to RGB
    depth_map = Image.open(depth_src)
    print(f"Image Shape: {img.shape}, Depth Map Shape: {np.array(depth_map).shape}")

    # Convert depth map to grayscale if it's colored
    if depth_map.mode != 'L':
        depth_map = depth_map.convert('L')

    depth_map = np.array(depth_map)

    # Normalize the depth map
    if grayscale:
        depth_map = depth_map * 10.0 / 255.0
    else:
        # Handle colored depth maps (if needed)
        depth_map = depth_map.mean(axis=-1) * 10.0 / 255.0  # Convert to grayscale by averaging channels

    # Get the dimensions of the depth map
    h, w = depth_map.shape

    # Estimate focal length if not provided
    if focal_length is None:
        focal_length = max(w, h) * 1.2  # Rough estimate

    # Calculate the principal point (center of the image)
    cy, cx = h / 2, w / 2

    # Create meshgrid and calculate camera coordinates
    points_x, points_y = np.meshgrid(np.arange(w), np.arange(h))
    x = (points_x - cx) * depth_map / focal_length
    y = -(points_y - cy) * depth_map / focal_length  # Negating to match camera and real-world alignment
    z = depth_map

    # Flatten the coordinates and stack them into a point cloud
    points = np.zeros((h * w, 3), dtype=np.float32)
    points[:, 0] = x.flatten()
    points[:, 1] = y.flatten()
    points[:, 2] = z.flatten()

    # Filter out NaN and invalid points
    valid_mask = ~np.isnan(points).any(axis=1) & (points[:, 2] >= 0)
    valid_points = points[valid_mask]

    # Create the point cloud
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(valid_points)

    # Map colors to valid points
    colors = img.reshape(-1, 3) / 255.0  # Normalize to [0, 1]
    if colors.shape[0] != valid_mask.shape[0]:
        raise ValueError(f"Color array shape {colors.shape} doesn't match mask shape {valid_mask.shape}")

    point_cloud.colors = o3d.utility.Vector3dVector(colors[valid_mask])

    if debug:
        print(f"Number of valid points: {np.sum(valid_mask)}")
        print(f"Point cloud dimensions: {valid_points.shape}")

    return point_cloud

def visualize_point_cloud(point_cloud):
    """
    Visualize the point cloud using Open3D
    """
    # Create coordinate frame for reference
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)

    # Visualize
    o3d.visualization.draw_geometries([point_cloud, coordinate_frame])


def create_mesh_from_point_cloud(point_cloud, depth_threshold=0.03, iterations=100):
    """
    Create a mesh from the point cloud using Poisson surface reconstruction

    Parameters:
    -----------
    point_cloud : open3d.geometry.PointCloud
        Input point cloud
    depth_threshold : float
        Depth threshold for normal estimation
    iterations : int
        Number of iterations for normal estimation

    Returns:
    --------
    mesh : open3d.geometry.TriangleMesh
        The reconstructed mesh
    """
    # Estimate normals (required for mesh reconstruction)
    point_cloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )

    # Orient normals consistently
    point_cloud.orient_normals_towards_camera_location()

    # Perform Poisson surface reconstruction
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        point_cloud, depth=9, width=0, scale=1.1, linear_fit=False
    )

    # Optional: Remove low-density vertices
    vertices_to_remove = densities < np.quantile(densities, 0.1)
    mesh.remove_vertices_by_mask(vertices_to_remove)

    return mesh


def main():
    img_src = None  #TODO
    depth_src = None  #TODO

    point_cloud = depth_to_point_cloud(img_src=img_src, depth_src=depth_src, grayscale=True, focal_length=1100)

    print("Visualizing Point Cloud...")
    visualize_point_cloud(point_cloud=point_cloud)

    print("Creating Mesh from Point Cloud...")
    mesh = create_mesh_from_point_cloud(point_cloud=point_cloud)
    o3d.visualization.draw_geometries([mesh])

    o3d.io.write_point_cloud("point_cloud.ply", point_cloud)
    o3d.io.write_triangle_mesh("mesh.ply", mesh)
    print("Point cloud and mesh saved to disk.")


if __name__ == "__main__":
    main()

