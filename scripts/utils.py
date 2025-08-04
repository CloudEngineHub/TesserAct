import cv2
import numpy as np
from typing import Optional, Tuple


def detect_edges_and_mask_points(
    depth: np.ndarray,
    image: Optional[np.ndarray] = None,
    point_cloud: Optional[np.ndarray] = None,
    edge_threshold1: float = 30.0,
    edge_threshold2: float = 60.0,
    kernel_size: int = 3,
    blur_kernel_size: int = 5,
    edge_dilation: int = 4,
    min_depth_threshold: float = 0.1,
    max_depth_threshold: float = 10.0,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Detect edges in depth map and mask points to create sharper point clouds.

    Args:
        depth: Depth map of shape (B, H, W, 1) or (H, W, 1)
        image: Optional RGB image of shape (B, H, W, 3) or (H, W, 3) for color information
        point_cloud: Optional point cloud of shape (B, H, W, 3) or (H, W, 3)
        edge_threshold1: First threshold for Canny edge detection
        edge_threshold2: Second threshold for Canny edge detection
        kernel_size: Kernel size for edge detection
        blur_kernel_size: Kernel size for Gaussian blur
        edge_dilation: Dilation factor for edge mask
        min_depth_threshold: Minimum depth threshold to filter out invalid points
        max_depth_threshold: Maximum depth threshold to filter out invalid points

    Returns:
        masked_depth: Masked depth values of shape (N, 1)
        masked_colors: Masked color values of shape (N, 3) or zeros if no image provided
        masked_points: Masked point cloud of shape (N, 3) or None if no point cloud provided
        valid_indices: Valid indices of shape (N, 2)
    """
    if depth.ndim == 4:
        depth = depth[0]  # Shape: (H, W, 1)
        if image is not None:
            image = image[0]  # Shape: (H, W, 3)
        if point_cloud is not None:
            point_cloud = point_cloud[0]  # Shape: (H, W, 3)

    depth_2d = depth.squeeze()
    valid_depth_mask = (depth_2d > min_depth_threshold) & (depth_2d < max_depth_threshold)
    depth_normalized = np.clip((depth_2d - depth_2d.min()) / (depth_2d.max() - depth_2d.min()) * 255, 0, 255).astype(
        np.uint8
    )
    depth_blurred = cv2.GaussianBlur(depth_normalized, (blur_kernel_size, blur_kernel_size), 0)
    edges = cv2.Canny(depth_blurred, edge_threshold1, edge_threshold2, apertureSize=kernel_size)
    kernel = np.ones((edge_dilation, edge_dilation), np.uint8)

    edges_dilated = cv2.dilate(edges, kernel, iterations=1)
    edge_mask = edges_dilated > 0
    final_mask = valid_depth_mask & (~edge_mask)
    valid_indices = np.where(final_mask)
    if len(valid_indices[0]) == 0:
        if image is not None:
            return np.empty((0, 1)), np.empty((0, 3)), None if point_cloud is None else np.empty((0, 3))
        else:
            return np.empty((0, 1)), np.zeros((0, 3)), None if point_cloud is None else np.empty((0, 3))

    masked_depth = depth_2d[valid_indices].reshape(-1, 1)
    if image is not None:
        masked_colors = image[valid_indices].reshape(-1, 3)
    else:
        masked_colors = np.zeros((len(valid_indices[0]), 3))
    if point_cloud is not None:
        masked_points = point_cloud[valid_indices].reshape(-1, 3)
    else:
        masked_points = None
    return masked_depth, masked_colors, masked_points, valid_indices
