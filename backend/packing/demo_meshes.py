import os
import glob
from box import Bin, MeshBox
import heuristics
from main import visualize_bin_pybullet

def get_mesh_files():
    base_dir = os.path.dirname(__file__)
    mesh_dir = os.path.join(base_dir, 'meshes')
    obj_files = glob.glob(os.path.join(mesh_dir, '*.obj'))
    off_files = glob.glob(os.path.join(mesh_dir, '*.off'))
    return obj_files + off_files

def run_demo():
    print("Loading meshes...")
    mesh_files = get_mesh_files()
    if not mesh_files:
        print("No meshes found in meshes/ directory.")
        return

    cell_size = 0.05

    b = Bin(30, 20, 50)

    for i, mesh_path in enumerate(mesh_files):
        print(f"[{i+1}/{len(mesh_files)}] Processing {os.path.basename(mesh_path)}...")
        box = MeshBox(mesh_path=mesh_path, cell_size=cell_size, name=f"mesh_{i}")

        print(f"  Shape (LxWxH cells): {box.length}x{box.width}x{box.height}")
        print(f"  Volume: {box.volume:.2f}")

        placement = heuristics.place_box_with_rule(box, b)
        if placement:
            x, y, z, rot, pose_idx = placement
            print(f"  Placed at x={x}, y={y}, z={z}, rot={rot}, pose={pose_idx}")
        else:
            print(f"  Failed to place!")

    print("\nPacking complete. Launching visualizer...")
    visualize_bin_pybullet(b, cell_size=cell_size, gui=True)

if __name__ == "__main__":
    run_demo()
