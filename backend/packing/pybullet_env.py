import pybullet as p
import pybullet_data
import time
import torch
import numpy as np
import random
import os

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
            self.client = p.connect(p.DIRECT)

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
        wall_height = 5.0

        wall_color = [0.2, 0.5, 0.8, 0.3] if self.render else [1, 1, 1, 0]

        for pos, ext in [
            ([0, half_w, wall_height], [half_l, wall_thickness, wall_height]),
            ([0, -half_w, wall_height], [half_l, wall_thickness, wall_height]),
            ([half_l, 0, wall_height], [wall_thickness, half_w + wall_thickness, wall_height]),
            ([-half_l, 0, wall_height], [wall_thickness, half_w + wall_thickness, wall_height])
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
        self.fragility_map = np.zeros((BIN_L, BIN_W), dtype=np.float32)

        heightmap = self._get_heightmap()
        current_box = self.episode_boxes[self.current_step]
        features = self._get_box_features(current_box)

        nn_heightmap = heightmap.T
        return nn_heightmap, features

    def step(self, action):
        """Action is a tuple: (x_idx, y_idx, rot_idx, pose_idx)"""
        x_idx, y_idx, rot_idx, pose_idx = action
        box = self.episode_boxes[self.current_step]

        pose = box.poses[pose_idx]
        grid_l = pose['Hb'].shape[1]
        grid_w = pose['Hb'].shape[0]
        pose_h = pose['height']

        if rot_idx % 2 == 1:
            phys_l, phys_w = grid_w * CELL_SIZE, grid_l * CELL_SIZE
            r_grid_l, r_grid_w = int(grid_w), int(grid_l)
        else:
            phys_l, phys_w = grid_l * CELL_SIZE, grid_w * CELL_SIZE
            r_grid_l, r_grid_w = int(grid_l), int(grid_w)

        x_idx = int(np.clip(x_idx, 0, max(0, BIN_L - r_grid_l)))
        y_idx = int(np.clip(y_idx, 0, max(0, BIN_W - r_grid_w)))

        world_x = (x_idx * CELL_SIZE) - (PHYSICAL_L / 2.0) + (phys_l / 2.0)
        world_y = (y_idx * CELL_SIZE) - (PHYSICAL_W / 2.0) + (phys_w / 2.0)

        import math
        from trimesh.transformations import euler_matrix
        angles = [
            (0, 0, 0),
            (math.pi, 0, 0),
            (math.pi/2, 0, 0),
            (-math.pi/2, 0, 0),
            (0, math.pi/2, 0),
            (0, -math.pi/2, 0)
        ]
        pose_matrix = euler_matrix(*angles[pose_idx])

        centered_mesh = box.mesh.copy()
        centered_mesh.apply_transform(pose_matrix)

        bounds = centered_mesh.bounds
        bbox_center = (bounds[0] + bounds[1]) / 2.0
        centered_mesh.apply_translation(-bbox_center)

        tmp_obj = f"/tmp/pybullet_mesh_live_{self.current_step}.obj"
        centered_mesh.export(tmp_obj)

        drop_orientation = p.getQuaternionFromEuler([0, 0, rot_idx * (np.pi/2)])

        current_heightmap = self._get_heightmap()

        end_x = min(BIN_L, x_idx + r_grid_l)
        end_y = min(BIN_W, y_idx + r_grid_w)

        max_height_underneath = current_heightmap[x_idx:end_x, y_idx:end_y].max().item()

        if end_x > x_idx and end_y > y_idx:
            underneath_fragility = self.fragility_map[x_idx:end_x, y_idx:end_y].max()
            self.fragility_map[x_idx:end_x, y_idx:end_y] = box.fragility
        else:
            underneath_fragility = 0.0

        phys_h = pose_h

        targeted_z = max_height_underneath + (phys_h / 2.0) + 0.02

        vol = box.volume if hasattr(box, 'volume') else box._volume

        vis_shape = p.createVisualShape(p.GEOM_MESH, fileName=tmp_obj, rgbaColor=[random.random(), random.random(), random.random(), 1])
        col_shape = p.createCollisionShape(p.GEOM_MESH, fileName=tmp_obj)

        body_id = p.createMultiBody(
            baseMass=vol * 10,
            baseCollisionShapeIndex=col_shape,
            baseVisualShapeIndex=vis_shape,
            basePosition=[world_x, world_y, targeted_z],
            baseOrientation=drop_orientation
        )
        self.placed_body_ids.append(body_id)

        for _ in range(240):
            p.stepSimulation()
            if self.render:
                time.sleep(1./240.)

        pos, _ = p.getBasePositionAndOrientation(body_id)
        fell_out = pos[2] < -0.05 or abs(pos[0]) > (PHYSICAL_L / 2.0) or abs(pos[1]) > (PHYSICAL_W / 2.0)

        next_heightmap = self._get_heightmap()
        self.current_step += 1
        done = fell_out or (self.current_step >= len(self.episode_boxes))

        if fell_out:
            reward = -20.0
        else:

            max_height = next_heightmap.max().item()
            compactness_reward = 1.0 - (max_height / DROP_HEIGHT)
            reward = compactness_reward * 2.0

            incoming_sturdiness = 1.0 - box.fragility
            fragility_violation = underneath_fragility * incoming_sturdiness

            if fragility_violation > 0.3:
                reward -= 10.0 * fragility_violation

        if not done:
            next_box = self.episode_boxes[self.current_step]
            next_features = self._get_box_features(next_box)
        else:
            next_features = torch.zeros(8)

        nn_next_heightmap = next_heightmap.T
        return nn_next_heightmap, next_features, reward, done