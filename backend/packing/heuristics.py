import numpy as np
from box import Box, Bin

'''
Bin(2,4,x) height map is initially:
[[0, 0, 0, 0],
 [0, 0, 0, 0]]

when we add a Box(2, 1, 3) (from left-top point 0,0) height map becomes:
[[3, 3, 0, 0],
 [0, 0, 0, 0]]

2 questions we ask: where do we want to put it and is it legal?
'''
def add_box(x, y, box: Box, b: Bin, rot=0, pose_idx=0):
    if not can_add_box(x, y, box, b, rot, pose_idx):
        return False

    if hasattr(box, 'poses'):
        pose = box.poses[pose_idx]
        Hb_raw = pose['Hb']
        Ht_raw = pose['Ht']
        rows, cols = Hb_raw.shape
        b_length = cols
        b_width = rows
        box_width = b_width if rot % 2 == 0 else b_length
        box_length = b_length if rot % 2 == 0 else b_width
        Hb_rot = np.rot90(Hb_raw, rot)
        Ht_rot = np.rot90(Ht_raw, rot)
    else:
        box_width = box.width if rot % 2 == 0 else box.length
        box_length = box.length if rot % 2 == 0 else box.width

    submatrix = b.height_map[y:y+box_width, x:x+box_length]

    if hasattr(box, 'poses'):
        Z = np.max(submatrix - Hb_rot)
        valid = (Hb_rot != np.inf)
        new_terrain = submatrix.copy()
        new_terrain[valid] = np.maximum(Ht_rot[valid] + Z, submatrix[valid])
        b.height_map[y:y+box_width, x:x+box_length] = new_terrain
        base_height = float(Z)
    else:
        base_height = submatrix.max()
        top_height = base_height + box.height
        b.height_map[y:y + box_width, x:x + box_length] = top_height

    b.boxes[box.name] = {
        "box": box,
        "x": x,
        "y": y,
        "z": base_height,
        "rot": rot,
        "pose_idx": pose_idx
    }
    update_access_priority(b)
    return True

def can_add_box(x, y, box: Box, b: Bin, rot=0, pose_idx=0):
    if hasattr(box, 'poses'):
        pose = box.poses[pose_idx]
        Hb_raw = pose['Hb']
        rows, cols = Hb_raw.shape
        b_length = cols
        b_width = rows
        box_width = b_width if rot % 2 == 0 else b_length
        box_length = b_length if rot % 2 == 0 else b_width
        Hb_rot = np.rot90(Hb_raw, rot)
    else:
        box_width = box.width if rot % 2 == 0 else box.length
        box_length = box.length if rot % 2 == 0 else box.width

    if x < 0: return False
    if y< 0: return False
    if x + box_length > b.height_map.shape[1]: return False
    if y + box_width > b.height_map.shape[0]: return False
    submatrix = b.height_map[y:y+box_width, x:x+box_length]

    if hasattr(box, 'poses'):
        Z = np.max(submatrix - Hb_rot)

        pose_height = pose['height']
        if Z + pose_height > b.height: return False

        valid = (Hb_rot != np.inf)
        if not np.any(valid): return False

        support_mask = (submatrix - Hb_rot) >= Z - 1e-3
        support_cells = np.count_nonzero(support_mask & valid)
        if support_cells == 0:
            return False
        return True
    else:
        if np.any((submatrix + box.height) > b.height): return False
        base_height = submatrix.max()
        support_ratio = np.count_nonzero(submatrix == base_height) / submatrix.size
        if support_ratio < 0.5:
            return False
        return True

def update_access_priority(b: Bin):
    entries = list(b.boxes.items())
    entries.sort(key=lambda item: item[1]["box"].fragility)
    b.priority_list = [i for i, _ in entries]

def all_pos_for_box(box: Box, b: Bin):
    hmap = b.height_map
    rows, cols = hmap.shape
    positions = []

    if hasattr(box, 'poses'):
        rotations = [0, 1, 2, 3]
        pose_indices = range(len(box.poses))
    else:
        rotations = [0, 1] if box.length != box.width else [0]
        pose_indices = [0]

    for p_idx in pose_indices:
        for r in rotations:
            if hasattr(box, 'poses'):
                pose = box.poses[p_idx]
                Hb_raw = pose['Hb']
                p_rows, p_cols = Hb_raw.shape
                b_length = p_cols
                b_width = p_rows

                box_width = b_width if r % 2 == 0 else b_length
                box_length = b_length if r % 2 == 0 else b_width
                Hb_rot = np.rot90(Hb_raw, r)
            else:
                box_width = box.width if r % 2 == 0 else box.length
                box_length = box.length if r % 2 == 0 else box.width

            for y in range(rows - box_width + 1):
                for x in range(cols - box_length + 1):
                    if can_add_box(x, y, box, b, r, p_idx):
                        submatrix = hmap[y:y + box_width, x:x + box_length]
                        if hasattr(box, 'poses'):
                            Z = np.max(submatrix - Hb_rot)
                            positions.append((x, y, Z, r, p_idx))
                        else:
                            Z = submatrix.max()
                            positions.append((x, y, Z, r, p_idx))

    return positions

def place_box_with_rule(box: Box, b: Bin):
    """
    Choose candidate with:
      minimal z, then minimal y, then minimal x
    and place box there.
    """
    candidates = all_pos_for_box(box, b)
    if not candidates:
        return None

    best_candidate = min(candidates, key=lambda p: (p[2], p[1], p[0]))
    best_x, best_y, best_z, best_rot, best_pose = best_candidate
    add_box(best_x, best_y, box, b, best_rot, best_pose)
    return best_candidate

def compute_compactness(b: Bin):
    max_height = np.max(b.height_map)
    if max_height == 0.0:
        return 0.0
    bounding_volume = max_height * b.length * b.width
    object_volume = 0
    for i in b.boxes.values():
        object_volume += i["box"].volume
    C_i = object_volume / bounding_volume
    return C_i

def compute_pyramid(b: Bin):
    object_volume = 0
    mask = np.zeros_like(b.height_map, dtype=bool)
    for entry in b.boxes.values():
        box = entry["box"]
        x = entry["x"]
        y = entry["y"]
        mask[y:y + box.width, x:x + box.length] = True
        object_volume += entry["box"].volume
    region_volume = float(b.height_map[mask].sum())
    if region_volume == 0:
        return 0.0
    return object_volume / region_volume

def compute_access_cost(b: Bin):
    if not b.priority_list:
        return 0.0
    N = len(b.priority_list)
    H = float(b.height)
    a_c = 0
    for i, box in enumerate(b.priority_list):
        curr_box = b.boxes[box]
        p_i = N - i
        z_i = H - (curr_box["box"].height + curr_box["z"])
        z_norm = max(0.0, min(1.0, z_i / H))
        fragility = float(curr_box["box"].fragility)
        frag_weight = 1.0
        a_c += frag_weight * z_norm * p_i
    return a_c / N

def footprint_overlap(b1: Box, b2: Box, b: Bin):
    if b1.name not in b.boxes or b2.name not in b.boxes:
        return 0.0
    x_j = b.boxes[b1.name]["x"]
    y_j = b.boxes[b1.name]["y"]
    x_k = b.boxes[b2.name]["x"]
    y_k = b.boxes[b2.name]["y"]
    x_overlap = max(0, min(x_j + b1.length, x_k + b2.length) - max(x_j, x_k))
    y_overlap = max(0, min(y_j + b1.width, y_k + b2.width) - max(y_j, y_k))
    return x_overlap * y_overlap

def vertical_stacking(b1: Box, b2: Box, b: Bin):
    z_top = b1.height + b.boxes[b1.name]["z"]
    z_base = b.boxes[b2.name]["z"]
    return z_base >= z_top

def weight_on_box(lower: Box, upper: Box, b: Bin):

    p = 1.0
    area_overlap = footprint_overlap(lower, upper, b)
    if area_overlap == 0.0:
        return 0.0
    area_upper = upper.length * upper.width
    fraction_on_lower = area_overlap / area_upper
    return fraction_on_lower * p * upper.volume

def compute_fragility_penalty(b: Bin,
                              base_scaling,
                              heavy_factor,
                              fragile_quantile,
                              alpha) -> float:
    """
    Fragility penalty:
      - For each box j, compute load_on_box from boxes stacked above.
      - Define "very fragile" as having fragility <= quantile of all fragilities.
      - For very fragile boxes, apply a much larger penalty if something sits on top.
    """

    if not b.boxes:
        return 0.0

    frag_list = np.array([entry["box"].fragility for entry in b.boxes.values()], dtype=float)

    very_fragile_thresh = np.quantile(frag_list, fragile_quantile)

    penalty = 0.0

    for _, j_entry in b.boxes.items():
        j_box = j_entry["box"]
        frag_j = float(j_box.fragility)

        load_on_box = 0.0
        for _, k_entry in b.boxes.items():
            k_box = k_entry["box"]
            if not vertical_stacking(j_box, k_box, b):
                continue
            load_on_box += weight_on_box(j_box, k_box, b)

        capacity = alpha * frag_j * float(j_box.volume)

        overload = max(0.0, load_on_box - capacity)
        if overload <= 0.0:

            continue

        if frag_j <= very_fragile_thresh:

            scale = base_scaling * heavy_factor
        else:

            scale = base_scaling

        penalty += scale * overload

    return float(penalty)

