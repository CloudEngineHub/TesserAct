import os
import bpy
import sys
import trimesh
import imageio
import argparse
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import detect_edges_and_mask_points


class ArgumentParserForBlender(argparse.ArgumentParser):
    """
    Reference: https://blender.stackexchange.com/a/134596
    This class is identical to its superclass, except for the parse_args
    method (see docstring). It resolves the ambiguity generated when calling
    Blender from the CLI with a python script, and both Blender and the script
    have arguments.
    """

    def _get_argv_after_doubledash(self, args=None):
        if args is not None:
            return args
        try:
            idx = sys.argv.index("--")
            return sys.argv[idx + 1 :]
        except ValueError as e:
            return []

    def parse_known_args(self, args=None, namespace=None):
        return super().parse_known_args(args=self._get_argv_after_doubledash(args), namespace=None)


def setup_render_settings(
    engine="CYCLES",
    samples=128,
    resolution=(640, 480),
    output_path="render.png",
    use_denoising=True,
    use_transparency=True,
):
    """Set up render settings with given parameters"""
    scene = bpy.context.scene
    scene.render.engine = engine
    scene.cycles.samples = samples
    scene.cycles.use_denoising = use_denoising
    scene.render.resolution_x, scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = use_transparency
    scene.render.filepath = output_path
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[1].default_value = 20
    # bpy remove object
    if "Cube" in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects["Cube"], do_unlink=True)
    bpy.context.view_layer.update()


def load_video(video_path: str):
    """Loads a video file into a list of numpy array frames."""
    reader = imageio.get_reader(video_path)
    frames = [frame for frame in reader]
    reader.close()
    return np.array(frames)


def load_rgbd_data(video_path: str, depth_video_path: str, combined_video_path: str = None):
    """Loads a specific frame from RGB and Depth video sources."""
    if combined_video_path is None:
        color_frame = load_video(video_path)
        depth_frame = load_video(depth_video_path)
    else:
        frame = load_video(combined_video_path)
        H, W = frame.shape[1:3]
        # Assuming a side-by-side format [Color | Depth | ...]
        color_frame = frame[:, :, : W // 3]
        depth_frame = frame[:, :, W // 3 : W // 3 * 2]
    if len(depth_frame.shape) == 3:
        depth_frame = depth_frame[:, :, 0]  # Ensure depth is 2D
    return color_frame, depth_frame


def create_vertex_color_material(name="PointCloudMaterial"):
    """Creates a new material that displays vertex colors."""
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    # Get the default BSDF node
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf is None:
        bsdf = mat.node_tree.nodes.new(type="ShaderNodeBsdfPrincipled")
    # Create a node to get vertex color data
    vc_node = mat.node_tree.nodes.new(type="ShaderNodeVertexColor")
    # The default name for vertex color attributes from PLY import is 'Col'
    vc_node.layer_name = "Col"
    # Link the color output to the BSDF's base color input
    mat.node_tree.links.new(vc_node.outputs["Color"], bsdf.inputs["Base Color"])

    return mat


def add_geometry_nodes_to_ply(ply_object, material, point_radius=0.001):
    """Adds a Geometry Nodes modifier to visualize the point cloud."""
    # Add a new Geometry Nodes modifier
    modifier = ply_object.modifiers.new(name="PointCloudDisplay", type="NODES")
    node_group = modifier.node_group
    if not node_group:
        node_group = bpy.data.node_groups.new(name="PointCloudNodes", type="GeometryNodeTree")
        modifier.node_group = node_group
    node_group.nodes.clear()
    node_group.interface.clear()
    geometry_input = node_group.interface.new_socket(name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    geometry_output = node_group.interface.new_socket(
        name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry"
    )
    # Create nodes
    group_input = node_group.nodes.new(type="NodeGroupInput")
    group_input.location = (0, 0)
    mesh_to_points = node_group.nodes.new(type="GeometryNodeMeshToPoints")
    mesh_to_points.location = (250, 0)
    mesh_to_points.inputs["Radius"].default_value = point_radius
    # bpy.data.node_groups["PointCloudNodes"].nodes["Mesh to Points"].inputs[3].default_value
    set_material = node_group.nodes.new(type="GeometryNodeSetMaterial")
    set_material.location = (500, 0)
    set_material.inputs["Material"].default_value = material
    group_output = node_group.nodes.new(type="NodeGroupOutput")
    group_output.location = (750, 0)
    # Link nodes
    links = node_group.links
    links.new(group_input.outputs["Geometry"], mesh_to_points.inputs["Mesh"])
    links.new(mesh_to_points.outputs["Points"], set_material.inputs["Geometry"])
    links.new(set_material.outputs["Geometry"], group_output.inputs["Geometry"])


def create_point_cloud_from_rgbd(
    image: np.ndarray,
    depth: np.ndarray,
    intrinsics: dict = None,
    output_path: str = "./render_output.png",
    point_cloud_mat: bpy.types.Material = None,
):
    """Main function to process data, create geometry, and render."""
    setup_render_settings(output_path=output_path)

    depth_reshaped = depth.reshape(depth.shape[0], depth.shape[1], 1)
    image_reshaped = image.reshape(image.shape[0], image.shape[1], 3)
    masked_depth, masked_colors, masked_points, valid_indices = detect_edges_and_mask_points(
        depth=depth_reshaped, image=image_reshaped
    )

    # Get the original coordinates for valid indices
    H, W = image.shape[:2]
    if intrinsics is None:
        focal_length = max(W, H)
        intrinsics = {"fx": focal_length, "fy": focal_length, "cx": W / 2, "cy": H / 2}
    fx, fy, cx, cy = intrinsics["fx"], intrinsics["fy"], intrinsics["cx"], intrinsics["cy"]
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    u_valid = u[valid_indices]
    v_valid = v[valid_indices]
    Z_valid = masked_depth.flatten()
    X = (u_valid - cx) * Z_valid / fx
    Y = (v_valid - cy) * Z_valid / fy
    verts = np.column_stack((X, -Z_valid, Y))
    colors = masked_colors

    ply_path = f"{output_path}".replace(".png", ".ply")
    trimesh.PointCloud(verts, colors).export(ply_path)

    bpy.ops.wm.ply_import(filepath=ply_path)
    ply_object = bpy.context.active_object
    bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY")
    # ply_object.rotation_euler = (-np.pi / 72, 0, np.pi / 36)
    ply_object.rotation_euler = (np.pi / 36, 0, np.pi / 36)

    add_geometry_nodes_to_ply(ply_object, point_cloud_mat, point_radius=0.002)

    # Position camera to look at the object
    camera = bpy.context.scene.camera
    camera.location = (0, 0, 0)
    camera.rotation_euler = (np.pi / 2, np.pi, np.pi)
    camera.data.lens = 36

    # Render the final image
    bpy.ops.render.render(write_still=True)
    print(f"Rendering complete. Saved to {bpy.context.scene.render.filepath}")

    # Clean up the temporary PLY file
    os.remove(ply_path)
    bpy.data.objects.remove(ply_object, do_unlink=True)


if __name__ == "__main__":
    parser = ArgumentParserForBlender()
    parser.add_argument("--rgb_video", type=str)
    parser.add_argument("--depth_video", type=str)
    parser.add_argument("--combined_video", type=str)
    parser.add_argument("--render_output", default="./results/render_output/pointcloud", type=str)
    args = parser.parse_args()

    os.makedirs(args.render_output, exist_ok=True)
    image, depth = load_rgbd_data(args.rgb_video, args.depth_video, args.combined_video)
    depth = (depth - depth.min()) / (depth.max() - depth.min())
    depth = 1 - depth * 1.0  # Invert
    depth = 2 * depth + 0.4  # transform to [0.4, 2.4]
    depth = depth.mean(axis=-1)

    for frame_index in range(len(image)):
        print(f"Processing frame {frame_index}")
        point_cloud_mat = create_vertex_color_material()
        create_point_cloud_from_rgbd(
            image=image[frame_index],
            depth=depth[frame_index],
            output_path=f"{args.render_output}/{frame_index}.png",
            intrinsics=None,
            point_cloud_mat=point_cloud_mat,
        )
