"""测地距离场工具（只读分析用）。
世界栅格：0.1 m 实体栅格化 → 欧氏距离变换得净空 → 每 3 格取 1（0.3 m 粗栅格）；净空 ≥ 0.3 m 的粗格为可通行；26 邻接。
field(goal) 返回从目标出发的 Dijkstra 距离与前驱；dist(p) 在粗格上取值，不可通行点取最近可通行格并加上到该格的直线距离。"""
import math
import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
RES, K, FREE = .1, 3, .3
class World:
    def __init__(self, geom, lo, hi):
        self.lo = np.asarray(lo, float); shape = tuple(int(math.ceil((hi[i] - lo[i]) / RES)) for i in range(3))
        X, Y, Z = (self.lo[i] + RES * (np.arange(shape[i]) + .5) for i in range(3)); XX, YY = np.meshgrid(X, Y, indexing='ij')
        occ = np.zeros(shape, bool)
        for o in geom:
            if o['shape'] == 'cylinder': m2 = (XX - o['x']) ** 2 + (YY - o['y']) ** 2 <= o['radius'] ** 2; h = o['height']
            else:
                c, s = math.cos(o['yaw']), math.sin(o['yaw']); dx, dy = XX - o['x'], YY - o['y']
                m2 = (abs(c * dx + s * dy) <= o['size_x'] / 2) & (abs(-s * dx + c * dy) <= o['size_y'] / 2); h = o['size_z']
            mz = (Z >= o['z'] - h / 2) & (Z <= o['z'] + h / 2)
            if mz.any() and m2.any(): occ[:, :, mz] |= m2[:, :, None]
        self.clear = (ndimage.distance_transform_edt(~occ) * RES - RES / 2)[::K, ::K, ::K]
        self.rc = RES * K; self.free = self.clear >= FREE; self.shape = self.free.shape
        self.ids = -np.ones(self.shape, int); self.ids[self.free] = np.arange(self.free.sum())
        self.cells = np.argwhere(self.free)
        r, c, w = [], [], []
        for d in [(i, j, l) for i in (-1, 0, 1) for j in (-1, 0, 1) for l in (-1, 0, 1) if (i, j, l) > (0, 0, 0)]:
            a = tuple(slice(max(0, -x), self.shape[q] - max(0, x)) for q, x in enumerate(d))
            b = tuple(slice(max(0, x), self.shape[q] - max(0, -x)) for q, x in enumerate(d))
            A, B = self.ids[a], self.ids[b]; m = (A >= 0) & (B >= 0)
            r.append(A[m]); c.append(B[m]); w.append(np.full(m.sum(), self.rc * math.sqrt(sum(x * x for x in d))))
        n = len(self.cells); self.G = coo_matrix((np.concatenate(w), (np.concatenate(r), np.concatenate(c))), shape=(n, n)).tocsr()
        _, self.near = ndimage.distance_transform_edt(~self.free, return_indices=True)
    def cell(self, p):
        return tuple(int(np.clip((p[i] - self.lo[i]) // self.rc, 0, self.shape[i] - 1)) for i in range(3))
    def center(self, c): return self.lo + self.rc * (np.asarray(c) + .5)
    def node(self, p):
        c = self.cell(p)
        if not self.free[c]: c = tuple(int(self.near[i][c]) for i in range(3))
        return self.ids[c], c
    def field(self, goal):
        n, _ = self.node(goal); d, pred = dijkstra(self.G, directed=False, indices=n, return_predecessors=True)
        return Field(self, d, pred)
class Field:
    def __init__(self, w, d, pred): self.w, self.d, self.pred = w, d, pred
    def dist(self, p):
        n, c = self.w.node(p); extra = 0. if self.w.free[self.w.cell(p)] else float(np.linalg.norm(self.w.center(c) - np.asarray(p)))
        return float(self.d[n]) + extra
    def path(self, p, length):
        n, c = self.w.node(p); pts = [self.w.center(self.w.cells[n])]; L = 0.
        while L < length and self.pred[n] >= 0:
            n = self.pred[n]; q = self.w.center(self.w.cells[n]); L += float(np.linalg.norm(q - pts[-1])); pts.append(q)
        return np.array(pts)
def line_clear(w, a, b):
    a, b = np.asarray(a, float), np.asarray(b, float); n = int(np.linalg.norm(b - a) / .1) + 2
    return min(float(w.clear[w.cell(a + t * (b - a))]) for t in np.linspace(0, 1, n))
