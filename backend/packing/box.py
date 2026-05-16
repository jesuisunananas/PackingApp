import numpy as np
import uuid
import mesh_utils

class BigBox:
    def __init__(self, length, width, height):
        self.length = length
        self.width = width
        self.height = height

    @property
    def length(self):
        return self._length

    @length.setter
    def length(self, l):
        if l <= 0:
            raise ValueError("length cannot be zero or negative")
        self._length = l

    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, w):
        if w <= 0:
            raise ValueError("width cannot be zero or negative")
        self._width = w

    @property
    def height(self):
        return self._height

    @height.setter
    def height(self, h):
        if h <= 0:
            raise ValueError("height cannot be zero or negative")
        self._height = h

    @property
    def volume(self):
        return self.length * self.width * self.height

class Box(BigBox):
    def __init__(self, length, width, height, fragility=1.0, name=None):
        super().__init__(length, width, height)
        if name is None:
            name = f"box_{uuid.uuid4().hex[:8]}"
        self.name = name

        self.fragility = fragility

    @property
    def fragility(self):
        return self._fragility

    @fragility.setter
    def fragility(self, f):
        if f > 1 or f < 0:
            raise ValueError("fragility score must be between 0 and 1 inclusive")
        self._fragility = f

class Bin(BigBox):
    def __init__(self, length, width, height):
        super().__init__(length, width, height)
        self.height_map = np.zeros((length, width), dtype=float)
        self.priority_list = []
        self.boxes = {}

class MeshBox(Box):
    def __init__(self, mesh_path, cell_size=0.05, fragility=1.0, name=None, save_mesh=False):
        self.mesh_path = mesh_path
        self.poses, self.mesh, self.orig_bounds = mesh_utils.mesh_to_heightmaps(mesh_path, cell_size, save_mesh)

        pose0 = self.poses[0]
        self.Ht = pose0['Ht']
        self.Hb = pose0['Hb']

        rows, cols = self.Hb.shape
        length = cols
        width = rows
        height = float(pose0['height'])

        if height == 0:
            height = 1.0
        super().__init__(length, width, height, fragility, name)

    @property
    def volume(self):

        valid = (self.Hb != np.inf)
        return float(np.sum(self.Ht[valid] - self.Hb[valid]))