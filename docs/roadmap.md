Plan for box.py:
    Add orientations for boxes
        Add rotation matrix support to reorient points of box
            Decision: use centroid or corners of box
Plan for heuristics.py:
    Add support for rotations and orientations


Orientation representation:
    Heightmap
        3x4x5 box -> 3x5x4 -> 4x5x3
    