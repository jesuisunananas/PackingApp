import matplotlib.pyplot as plt
import numpy as np
import glob
from box import MeshBox

def plot_heightmaps():
    print("Loading meshes...")

    mesh_files = glob.glob('meshes/*.obj')
    mesh_files.extend(glob.glob('meshes/*.off'))
    if not mesh_files:
        print("No meshes found in meshes/ directory.")
        return

    target_mesh = mesh_files[0]
    print(f"Generating Heightmaps for: {target_mesh}")

    box = MeshBox(target_mesh, cell_size=0.05)

    num_poses = len(box.poses)
    fig, axes = plt.subplots(2, num_poses, figsize=(3 * num_poses, 6))
    fig.suptitle(f"Heightmaps for 6 Rest Poses: {target_mesh.replace('meshes/', '')}", fontsize=16)

    for i, pose in enumerate(box.poses):
        Ht = pose['Ht']
        Hb = pose['Hb']

        Hb_display = Hb.copy()
        Hb_display[Hb_display == np.inf] = np.nan

        Ht_display = Ht.copy()
        Ht_display[Ht_display == 0] = np.nan

        ax_top = axes[0, i]
        im_top = ax_top.imshow(Ht_display, cmap='viridis', origin='upper')
        ax_top.set_title(f"Top ($H_t$) Pose {i}")
        ax_top.axis('off')

        ax_bot = axes[1, i]
        im_bot = ax_bot.imshow(Hb_display, cmap='plasma', origin='upper')
        ax_bot.set_title(f"Bottom ($H_b$) Pose {i}")
        ax_bot.axis('off')

    plt.tight_layout()
    print("Close the visualization window to continue.")
    plt.show()

if __name__ == "__main__":
    plot_heightmaps()
