import pybullet as p
import pybullet_data
import time
import torch
import numpy as np
import random
import os

# --- CONFIGURATION ---
BIN_L, BIN_W, BIN_H = 30, 30, 50
CELL_SIZE = 0.05
PHYSICAL_L = BIN_L * CELL_SIZE 
PHYSICAL_W = BIN_W * CELL_SIZE 
DROP_HEIGHT = 2.5 

class PyBulletPackingEnv:
    def __init__(self, render=False):
        self.render = render
        if self.render:
            self.client = p.connect(p.GUI)
            p.resetDebugVisualizerCamera(cameraDistance=2.5, cameraYaw=45, cameraPitch=-45, cameraTargetPosition=[0,0,0])
        else:
            self.client = p.connect(p.DIRECT) # Headless mode for fast training
            
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        self.current_step = 0
        self.max_steps = 15
        self.episode_boxes = []
        self.placed_body_ids = []

    def _build_bin(self):
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.loadURDF("plane.urdf")
        
        half_l, half_w = PHYSICAL_L / 2, PHYSICAL_W / 2
        wall_thickness = 0.05
        wall_height = 1.0
        
        # Transparent walls so we can see inside during render
        wall_color = [0.2, 0.5, 0.8, 0.3] if self.render else [1, 1, 1, 0]
        
        for pos, ext in [
            ([0, half_w, wall_height], [half_l, wall_thickness, wall_height]), # Top
            ([0, -half_w, wall_height], [half_l, wall_thickness, wall_height]), # Bottom
            ([half_l, 0, wall_height], [wall_thickness, half_w, wall_height]), # Right
            ([-half_l, 0, wall_height], [wall_thickness, half_w, wall_height]) # Left
        ]:
            shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=ext)
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=ext, rgbaColor=wall_color)
            p.createMultiBody(0, shape, vis, pos)

    def _get_heightmap(self):
        ray_starts, ray_ends = [], []
        offset = CELL_SIZE / 2.0
        
        for x in range(BIN_L):
            for y in range(BIN_W):
                world_x = (x * CELL_SIZE) - (PHYSICAL_L / 2.0) + offset
                world_y = (y * CELL_SIZE) - (PHYSICAL_W / 2.0) + offset
                ray_starts.append([world_x, world_y, DROP_HEIGHT])
                ray_ends.append([world_x, world_y, 0.0])
                
        results = p.rayTestBatch(ray_starts, ray_ends)
        
        heightmap = np.zeros((BIN_L, BIN_W))
        for i, res in enumerate(results):
            x, y = i // BIN_W, i % BIN_W
            heightmap[x, y] = DROP_HEIGHT * (1.0 - res[2])
            
        return torch.tensor(heightmap, dtype=torch.float32)

    def _get_box_features(self, box):
        grid_l, grid_w = box.length, box.width
        vol = box.volume if hasattr(box, 'volume') else box._volume
        features = [
            grid_l, grid_w, box.height, vol, box.fragility,
            grid_l / BIN_L, grid_w / BIN_W, box.height / BIN_H
        ]
        return torch.tensor(features, dtype=torch.float32)

    def reset(self, box_list):
        self._build_bin()
        self.current_step = 0
        self.episode_boxes = box_list
        self.placed_body_ids = []
        
        heightmap = self._get_heightmap()
        current_box = self.episode_boxes[self.current_step]
        features = self._get_box_features(current_box)
        
        return heightmap, features

    def step(self, action):
        """Action is a tuple: (x_idx, y_idx, rot_idx, pose_idx)"""
        x_idx, y_idx, rot_idx, pose_idx = action
        box = self.episode_boxes[self.current_step]
        
        # 1. Translate Action to World
        grid_l, grid_w = box.length, box.width
        if rot_idx % 2 == 1: 
            phys_l, phys_w = grid_w * CELL_SIZE, grid_l * CELL_SIZE
        else:
            phys_l, phys_w = grid_l * CELL_SIZE, grid_w * CELL_SIZE

        world_x = (x_idx * CELL_SIZE) - (PHYSICAL_L / 2.0) + (phys_l / 2.0)
        world_y = (y_idx * CELL_SIZE) - (PHYSICAL_W / 2.0) + (phys_w / 2.0)
        drop_orientation = p.getQuaternionFromEuler([0, 0, rot_idx * (np.pi/2)]) 
        
        # 2. Spawn Mesh
        tmp_obj = f"/tmp/pybullet_mesh_live_{self.current_step}.obj"
        box.mesh.export(tmp_obj)
        vol = box.volume if hasattr(box, 'volume') else box._volume
        
        vis_shape = p.createVisualShape(p.GEOM_MESH, fileName=tmp_obj, rgbaColor=[random.random(), random.random(), random.random(), 1])
        col_shape = p.createCollisionShape(p.GEOM_MESH, fileName=tmp_obj)
        
        body_id = p.createMultiBody(
            baseMass=vol * 10, 
            baseCollisionShapeIndex=col_shape,
            baseVisualShapeIndex=vis_shape,
            basePosition=[world_x, world_y, DROP_HEIGHT],
            baseOrientation=drop_orientation
        )
        self.placed_body_ids.append(body_id)
        
        # 3. Step Physics Simulation
        for _ in range(240): # 1 second of gravity
            p.stepSimulation()
            if self.render:
                time.sleep(1./240.)
                
        # 4. Check for catastrophic failure (Fell out of bin)
        pos, _ = p.getBasePositionAndOrientation(body_id)
        fell_out = pos[2] < 0.1 or abs(pos[0]) > (PHYSICAL_L / 2.0) or abs(pos[1]) > (PHYSICAL_W / 2.0)
        
        # 5. Get Next State
        next_heightmap = self._get_heightmap()
        self.current_step += 1
        done = fell_out or (self.current_step >= len(self.episode_boxes))
        
        # 6. Calculate Reward (Physics-Aware)
        if fell_out:
            reward = -20.0 # Massive penalty for tumbling out
        else:
            # Reward for keeping the overall heightmap flat and compact
            # We want to minimize the maximum height and the variance
            max_height = next_heightmap.max().item()
            compactness_reward = 1.0 - (max_height / DROP_HEIGHT) 
            reward = compactness_reward * 2.0 # Scale it up slightly
            
        # Get next features if episode isn't done
        if not done:
            next_box = self.episode_boxes[self.current_step]
            next_features = self._get_box_features(next_box)
        else:
            next_features = torch.zeros(8)
            
        return next_heightmap, next_features, reward, done