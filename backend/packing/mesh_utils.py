import numpy as np
import trimesh

def load_and_orient_mesh(mesh_path):
    mesh = trimesh.load(mesh_path, force='mesh')
    
    # --- DYNAMIC SCALING FIX ---
    # Find the current physical size of the mesh (Length, Width, Height)
    extents = mesh.extents
    max_extent = np.max(extents)
    
    # Avoid division by zero on corrupted flat meshes
    if max_extent > 0:
        # Force the longest side of the object to be exactly 1.0 meter. 
        # (You can change 1.0 to random.uniform(0.5, 1.5) if you want varied object sizes)
        scale_factor = 1.0 / max_extent
        mesh.apply_scale(scale_factor)
    
    # Shift to origin
    orig_bounds_0 = mesh.bounds[0].copy()
    mesh.apply_translation(-orig_bounds_0)
    return mesh, orig_bounds_0

def mesh_to_heightmaps(mesh_path, cell_size=0.05, save_mesh=False):
    mesh_orig, orig_bounds = load_and_orient_mesh(mesh_path)
    poses = []
    
    from trimesh.transformations import euler_matrix
    import math
    
    angles = [
        (0, 0, 0),
        (math.pi, 0, 0),
        (math.pi/2, 0, 0),
        (-math.pi/2, 0, 0),
        (0, math.pi/2, 0),
        (0, -math.pi/2, 0)
    ]
    
    for r in angles:
        mesh = mesh_orig.copy()
        matrix = euler_matrix(*r)
        mesh.apply_transform(matrix)
        
        offset = mesh.bounds[0].copy()
        mesh.apply_translation(-offset)
        
        bounds = mesh.bounds
        min_b, max_b = bounds[0], bounds[1]
        
        size_x = max_b[0] - min_b[0]
        size_y = max_b[1] - min_b[1]
        size_z = max_b[2] - min_b[2]
        
        cols = int(np.ceil(size_x / cell_size))
        rows = int(np.ceil(size_y / cell_size))
        
        if cols == 0 or rows == 0:
            continue

        MAX_CELLS = 1000000 
        if (cols * rows) > MAX_CELLS:
            raise MemoryError(f"Mesh requested {cols * rows} cells! Skipping to prevent RAM crash.")
            
        Ht = np.zeros((rows, cols))
        Hb = np.full((rows, cols), np.inf)
        
        xs = np.arange(cols) * cell_size + (cell_size / 2.0)
        ys = np.arange(rows) * cell_size + (cell_size / 2.0)
        XX, YY = np.meshgrid(xs, ys)
        pts_x = XX.flatten()
        pts_y = YY.flatten()
        num_pts = len(pts_x)
        
        origins_t = np.column_stack((pts_x, pts_y, np.full(num_pts, size_z + 0.1)))
        dirs_t = np.tile(np.array([0., 0., -1.]), (num_pts, 1))
        locations_t, index_ray_t, _ = mesh.ray.intersects_location(origins_t, dirs_t)
        for idx_ray, loc in zip(index_ray_t, locations_t):
            rt = idx_ray // cols
            c = idx_ray % cols
            z = loc[2]
            if z > Ht[rt, c]: Ht[rt, c] = z
                
        origins_b = np.column_stack((pts_x, pts_y, np.full(num_pts, -0.1)))
        dirs_b = np.tile(np.array([0., 0., 1.]), (num_pts, 1))
        locations_b, index_ray_b, _ = mesh.ray.intersects_location(origins_b, dirs_b)
        for idx_ray, loc in zip(index_ray_b, locations_b):
            rt = idx_ray // cols
            c = idx_ray % cols
            z = loc[2]
            if z < Hb[rt, c]: Hb[rt, c] = z
            
        if save_mesh:
            import uuid
            import os
            tmp_path = f"/tmp/packing_meshes/pose_{uuid.uuid4().hex}.obj"
            os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
            mesh.export(tmp_path)
        else:
            tmp_path = None
            
        poses.append({
            'Ht': Ht,
            'Hb': Hb,
            'mesh_path': tmp_path,
            'height': size_z
        })
            
    return poses, mesh_orig, orig_bounds
