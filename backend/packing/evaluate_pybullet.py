import pybullet as p
import pybullet_data
import time
import torch
import numpy as np
import glob
import random
import os

from model_cnn import SpatialQNetwork
from box import MeshBox # Assuming your MeshBox class can compute L, W, H, Vol

# --- CONFIGURATION ---
BIN_L, BIN_W, BIN_H = 30, 30, 50
CELL_SIZE = 0.05
PHYSICAL_L = BIN_L * CELL_SIZE # 1.5m
PHYSICAL_W = BIN_W * CELL_SIZE # 1.5m
DROP_HEIGHT = 2.5 # Drop from above the bin

def get_live_heightmap():
    """Simulates a depth camera using PyBullet Raycasting."""
    ray_starts = []
    ray_ends = []
    
    # Offset by half a cell so the ray fires through the center of the grid cell
    offset = CELL_SIZE / 2.0
    
    for x in range(BIN_L):
        for y in range(BIN_W):
            # Convert grid (x,y) to world (world_x, world_y)
            world_x = (x * CELL_SIZE) - (PHYSICAL_L / 2.0) + offset
            world_y = (y * CELL_SIZE) - (PHYSICAL_W / 2.0) + offset
            
            ray_starts.append([world_x, world_y, DROP_HEIGHT])
            ray_ends.append([world_x, world_y, 0.0]) # Fire straight down to the floor
            
    # Batch raycast is much faster than doing it one by one
    results = p.rayTestBatch(ray_starts, ray_ends)
    
    heightmap = np.zeros((BIN_L, BIN_W))
    idx = 0
    for x in range(BIN_L):
        for y in range(BIN_W):
            hit_fraction = results[idx][2] # Distance ratio from start to end
            # Calculate actual height from the floor
            hit_height = DROP_HEIGHT * (1.0 - hit_fraction)
            heightmap[x, y] = hit_height
            idx += 1
            
    return torch.tensor(heightmap, dtype=torch.float32)

def setup_pybullet_bin():
    """Builds the physical bin walls in PyBullet."""
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    
    # Load floor
    p.loadURDF("plane.urdf")
    
    # Build invisible/translucent walls to constrain the boxes
    half_l, half_w = PHYSICAL_L / 2, PHYSICAL_W / 2
    wall_thickness = 0.05
    wall_height = 1.0
    
    # --- FIX: ADDED VISUAL SHAPES SO YOU CAN SEE THE BIN ---
    wall_color = [0.2, 0.5, 0.8, 0.3] # Translucent blue
    
    # Top/Bottom Walls
    wall_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[half_l, wall_thickness, wall_height])
    wall_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[half_l, wall_thickness, wall_height], rgbaColor=wall_color)
    p.createMultiBody(0, wall_shape, wall_vis, [0, half_w, wall_height]) 
    p.createMultiBody(0, wall_shape, wall_vis, [0, -half_w, wall_height]) 
    
    # Left/Right Walls
    wall_shape_side = p.createCollisionShape(p.GEOM_BOX, halfExtents=[wall_thickness, half_w, wall_height])
    wall_vis_side = p.createVisualShape(p.GEOM_BOX, halfExtents=[wall_thickness, half_w, wall_height], rgbaColor=wall_color)
    p.createMultiBody(0, wall_shape_side, wall_vis_side, [half_l, 0, wall_height]) 
    p.createMultiBody(0, wall_shape_side, wall_vis_side, [-half_l, 0, wall_height]) 

def evaluate_agent():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Loading CQL Policy...")
    policy = SpatialQNetwork(feature_dim=8, hidden_dim=64, num_rotations=4, num_poses=6).to(device)
    policy.load_state_dict(torch.load('policy_cql_spatial.pt', map_location=device))
    policy.eval()
    
    setup_pybullet_bin()
    test_meshes = glob.glob(os.path.join("meshes", "ModelNet10", "*", "test", "*.off"))
    
    p.resetDebugVisualizerCamera(cameraDistance=2.5, cameraYaw=45, cameraPitch=-45, cameraTargetPosition=[0,0,0])

    print("\n--- Starting Physics Rollout ---")
    for step in range(15): 
        mesh_path = random.choice(test_meshes)
        box = MeshBox(mesh_path, cell_size=CELL_SIZE)
        vol = box.volume if hasattr(box, 'volume') else box._volume
        
        # --- FIX: explicitly grab grid dimensions ---
        grid_l, grid_w = box.length, box.width
        
        features = torch.tensor([
            grid_l, grid_w, box.height, vol, box.fragility,
            grid_l / BIN_L, grid_w / BIN_W, box.height / BIN_H
        ], dtype=torch.float32).unsqueeze(0).to(device) 
        
        heightmap = get_live_heightmap().unsqueeze(0).to(device) 
        
        with torch.no_grad():
            q_values = policy(heightmap, features) 

        # Apply boundary masking using GRID CELLS, not meters
        for r in range(4):
            if r % 2 == 1: 
                r_grid_l, r_grid_w = grid_w, grid_l
            else:
                r_grid_l, r_grid_w = grid_l, grid_w
                
            max_valid_x = max(0, BIN_L - int(r_grid_l))
            max_valid_y = max(0, BIN_W - int(r_grid_w))
            
            q_values[:, r, :, max_valid_x:, :] = -float('inf')
            q_values[:, r, :, :, max_valid_y:] = -float('inf')
            
        best_flat_idx = q_values.argmax().item()
        
        rot_idx = (best_flat_idx // (6 * 30 * 30)) % 4
        pose_idx = (best_flat_idx // (30 * 30)) % 6
        x_idx = (best_flat_idx // 30) % 30
        y_idx = best_flat_idx % 30
        
        print(f"Step {step+1}: Brain chose Grid(X:{x_idx}, Y:{y_idx}) | Rot:{rot_idx}, Pose:{pose_idx}")
        
        # --- FIX: Calculate physical meters for the drop translation ---
        if rot_idx % 2 == 1: 
            phys_l, phys_w = grid_w * CELL_SIZE, grid_l * CELL_SIZE
        else:
            phys_l, phys_w = grid_l * CELL_SIZE, grid_w * CELL_SIZE

        world_x = (x_idx * CELL_SIZE) - (PHYSICAL_L / 2.0) + (phys_l / 2.0)
        world_y = (y_idx * CELL_SIZE) - (PHYSICAL_W / 2.0) + (phys_w / 2.0)
        
        drop_orientation = p.getQuaternionFromEuler([0, 0, rot_idx * (np.pi/2)]) 
        
        tmp_obj = f"/tmp/pybullet_mesh_{step}.obj"
        box.mesh.export(tmp_obj)
        
        # --- FIX: Assign random bright colors to the meshes so they don't spawn invisible ---
        box_color = [random.uniform(0.3, 1), random.uniform(0.3, 1), random.uniform(0.3, 1), 1.0]
        visual_shape = p.createVisualShape(shapeType=p.GEOM_MESH, fileName=tmp_obj, rgbaColor=box_color)
        collision_shape = p.createCollisionShape(shapeType=p.GEOM_MESH, fileName=tmp_obj)
        
        body_id = p.createMultiBody(
            baseMass=vol * 10, # Scaled down mass so physics engine doesn't explode
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=[world_x, world_y, DROP_HEIGHT],
            baseOrientation=drop_orientation
        )
        
        for _ in range(240): 
            p.stepSimulation()
            time.sleep(1./240.) 
            
    print("Episode Complete! Close the window to exit.")
    while True:
        p.stepSimulation()
        time.sleep(1./240.)

if __name__ == "__main__":
    evaluate_agent()