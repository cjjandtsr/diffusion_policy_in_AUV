"""Environment difficulty metrics per task (spec: jindu/2026-10-31_环境难度分层规范_v1_0.md).

Uses only start, goal and obstacle geometry. No policy outcome is read here.

Per task:
  rho_w, n_w      corridor density and intersecting-obstacle count, w in {1.0, 2.0}   (spec §3)
  b               max-min clearance over all free paths S->G                        (spec §4)
  kappa, theta    shortest collision-free path length / chord, total turning angle  (spec §5)
  blocked         segment_obstacle_collision(S, G, obstacles, 0.1) is not None       (spec §6)

Usage: python metrics.py <tasks.json> <repo_root> <out.json> [n_workers]
"""
import json
import math
import sys
import time
from multiprocessing import Pool

import numpy as np
from scipy import ndimage

RES_DENSITY = 0.05      # spec §3
RES_GRID = 0.1          # spec §4, §5
Z_GROUND = 0.2          # ground contact height
Z_TOP = 6.0             # spec §4 upper bound
XY_LIMIT = 30.0         # environment out-of-bounds limit
MARGIN = 3.0            # spec §4 bounding-box margin
CORRIDOR_DZ = 1.0       # spec §3 vertical margin
WIDTHS = (1.0, 2.0)     # spec §3 half widths
INFLATE_XY = 0.1        # = segment_obstacle_collision threshold
INFLATE_Z = 0.1         # = DRONE_RADIUS_Z
BISECT_TOL = 0.01

_REPO = None


def _half_height(o):
    return (o['height'] if o['shape'] == 'cylinder' else o['size_z']) / 2.0


def _xy_reach(o):
    if o['shape'] == 'cylinder':
        return o['radius']
    return math.hypot(o['size_x'], o['size_y']) / 2.0


def horizontal_distance(o, x, y):
    """Horizontal distance from points to the obstacle footprint (0 inside)."""
    dx, dy = x - o['x'], y - o['y']
    if o['shape'] == 'cylinder':
        return np.maximum(np.hypot(dx, dy) - o['radius'], 0.0)
    c, s = math.cos(o['yaw']), math.sin(o['yaw'])
    lx, ly = c * dx + s * dy, -s * dx + c * dy
    return np.hypot(np.maximum(np.abs(lx) - o['size_x'] / 2.0, 0.0),
                    np.maximum(np.abs(ly) - o['size_y'] / 2.0, 0.0))


def inside(o, x, y, z, inflate_xy=0.0, inflate_z=0.0):
    hz = _half_height(o) + inflate_z
    vertical = np.abs(z - o['z']) <= hz
    if o['shape'] == 'cylinder' and inflate_xy == 0.0:
        return vertical & (np.hypot(x - o['x'], y - o['y']) <= o['radius'])
    return vertical & (horizontal_distance(o, x, y) <= inflate_xy)


# ---------------------------------------------------------------- corridor (§3)

def corridor_density(task, w):
    S, G = np.array(task['start']), np.array(task['goal'])
    d = G[:2] - S[:2]
    L = float(np.hypot(*d))
    zlo = max(min(S[2], G[2]) - CORRIDOR_DZ, Z_GROUND)
    zhi = max(S[2], G[2]) + CORRIDOR_DZ
    zs = np.arange(zlo + RES_DENSITY / 2, zhi, RES_DENSITY)
    degenerate = L < 0.5
    if degenerate:
        centre = (S[:2] + G[:2]) / 2.0
        g = np.arange(-w + RES_DENSITY / 2, w, RES_DENSITY)
        gx, gy = np.meshgrid(g, g, indexing='ij')
        keep = np.hypot(gx, gy) <= w
        px, py = centre[0] + gx[keep], centre[1] + gy[keep]
        axis_lo, axis_hi = None, None
    else:
        u = d / L
        v = np.array([-u[1], u[0]])
        ts = np.arange(RES_DENSITY / 2, L, RES_DENSITY)
        ss = np.arange(-w + RES_DENSITY / 2, w, RES_DENSITY)
        tt, sv = np.meshgrid(ts, ss, indexing='ij')
        px = (S[0] + tt * u[0] + sv * v[0]).ravel()
        py = (S[1] + tt * u[1] + sv * v[1]).ravel()
    total = px.size * zs.size
    occupied = np.zeros((px.size, zs.size), dtype=bool)
    hits = 0
    for o in task['geometry']:
        hz = _half_height(o)
        if o['z'] + hz < zlo or o['z'] - hz > zhi:
            continue
        # cheap horizontal pre-filter on the bounding circle
        near = np.hypot(px - o['x'], py - o['y']) <= _xy_reach(o) + 1e-9
        if not near.any():
            continue
        fx, fy = px[near], py[near]
        foot = horizontal_distance(o, fx, fy) <= 0.0
        if not foot.any():
            continue
        zmask = np.abs(zs - o['z']) <= hz
        if not zmask.any():
            continue
        idx = np.nonzero(near)[0][foot]
        occupied[np.ix_(idx, np.nonzero(zmask)[0])] = True
        hits += 1
    return dict(rho=float(occupied.sum() / total), n=hits, degenerate=degenerate,
                volume_m3=float(total * RES_DENSITY ** 3))


# ---------------------------------------------------------------- grid (§4, §5)

def build_grid(task):
    S, G = task['start'], task['goal']
    xs = [S[0], G[0]]
    ys = [S[1], G[1]]
    for o in task['geometry']:
        r = _xy_reach(o)
        xs += [o['x'] - r, o['x'] + r]
        ys += [o['y'] - r, o['y'] + r]
    x0 = max(min(xs) - MARGIN, -XY_LIMIT)
    x1 = min(max(xs) + MARGIN, XY_LIMIT)
    y0 = max(min(ys) - MARGIN, -XY_LIMIT)
    y1 = min(max(ys) + MARGIN, XY_LIMIT)
    gx = np.arange(x0 + RES_GRID / 2, x1, RES_GRID)
    gy = np.arange(y0 + RES_GRID / 2, y1, RES_GRID)
    gz = np.arange(Z_GROUND + RES_GRID / 2, Z_TOP, RES_GRID)
    raw = np.zeros((gx.size, gy.size, gz.size), dtype=bool)
    infl = np.zeros_like(raw)
    for o in task['geometry']:
        r = _xy_reach(o) + INFLATE_XY
        hz = _half_height(o) + INFLATE_Z
        ix = np.nonzero((gx >= o['x'] - r) & (gx <= o['x'] + r))[0]
        iy = np.nonzero((gy >= o['y'] - r) & (gy <= o['y'] + r))[0]
        iz = np.nonzero(np.abs(gz - o['z']) <= hz)[0]
        if not (ix.size and iy.size and iz.size):
            continue
        X, Y, Z = np.meshgrid(gx[ix], gy[iy], gz[iz], indexing='ij')
        block = np.ix_(ix, iy, iz)
        raw[block] |= inside(o, X, Y, Z)
        infl[block] |= inside(o, X, Y, Z, INFLATE_XY, INFLATE_Z)
    return dict(gx=gx, gy=gy, gz=gz, raw=raw, infl=infl)


def cell_of(grid, p):
    i = int(round((p[0] - grid['gx'][0]) / RES_GRID))
    j = int(round((p[1] - grid['gy'][0]) / RES_GRID))
    k = int(round((p[2] - grid['gz'][0]) / RES_GRID))
    shape = grid['raw'].shape
    return (min(max(i, 0), shape[0] - 1), min(max(j, 0), shape[1] - 1),
            min(max(k, 0), shape[2] - 1))


def clearance_field(grid):
    """Distance (m) from each free cell centre to the nearest occupied cell centre,
    minus half a cell; ground plane z=0.2 also counts as an obstacle."""
    edt = ndimage.distance_transform_edt(~grid['raw'], sampling=RES_GRID)
    edt = np.maximum(edt - RES_GRID / 2.0, 0.0)
    edt[grid['raw']] = 0.0
    ground = (grid['gz'] - Z_GROUND)[None, None, :]
    return np.minimum(edt, ground)


ENDPOINT_BALL = 1.0     # spec v1.1 §4: clearance inside 1.0 m of S or G is not counted


def endpoint_balls(grid, S, G):
    X, Y, Z = np.meshgrid(grid['gx'], grid['gy'], grid['gz'], indexing='ij', sparse=True)
    ball = np.zeros(grid['raw'].shape, dtype=bool)
    for p in (S, G):
        ball |= (X - p[0]) ** 2 + (Y - p[1]) ** 2 + (Z - p[2]) ** 2 <= ENDPOINT_BALL ** 2
    return ball & ~grid['raw']


BAND_HALF_WIDTH = 3.0   # spec v1.1 §4: lateral half width of the route band
BAND_AXIAL_EXT = 1.0    # spec v1.1 §4: axial extension beyond S and G
BAND_DZ = 1.0           # spec v1.1 §4: band top = max(zS, zG) + 1.0


def route_band(grid, S, G):
    """Cells near the direct route: lateral |s| <= 3 m from the horizontal S-G line,
    axial t in [-1, L+1], z <= max(zS, zG) + 1 (ground z >= 0.2 by grid)."""
    X, Y, Z = np.meshgrid(grid['gx'], grid['gy'], grid['gz'], indexing='ij', sparse=True)
    d = np.array(G[:2]) - np.array(S[:2])
    L = float(np.hypot(*d))
    ztop = max(S[2], G[2]) + BAND_DZ
    if L < 0.5:
        c = (np.array(S[:2]) + np.array(G[:2])) / 2.0
        horizontal = (X - c[0]) ** 2 + (Y - c[1]) ** 2 <= BAND_HALF_WIDTH ** 2
    else:
        u = d / L
        t = (X - S[0]) * u[0] + (Y - S[1]) * u[1]
        s = -(X - S[0]) * u[1] + (Y - S[1]) * u[0]
        horizontal = (t >= -BAND_AXIAL_EXT) & (t <= L + BAND_AXIAL_EXT) & (np.abs(s) <= BAND_HALF_WIDTH)
    return horizontal & (Z <= ztop)


def max_min_clearance(clear, s_cell, g_cell, ball):
    """Largest r such that S and G are connected through free cells whose clearance
    is >= r, where cells inside the endpoint balls always count as passable."""
    structure = np.ones((3, 3, 3), dtype=bool)

    def connected(r):
        mask = (clear >= r) | ball
        labels, _ = ndimage.label(mask, structure=structure)
        return labels[s_cell] != 0 and labels[s_cell] == labels[g_cell]

    lo = 1e-6
    if not connected(lo):
        return None
    hi = float(clear.max())
    if connected(hi):
        return hi
    while hi - lo > BISECT_TOL:
        mid = (lo + hi) / 2.0
        if connected(mid):
            lo = mid
        else:
            hi = mid
    return lo


def shortest_path(task, grid, obstacles, segment_collision):
    from skimage.graph import MCP_Geometric
    S, G = list(task['start']), list(task['goal'])
    free = ~grid['infl']
    s_cell, g_cell = cell_of(grid, S), cell_of(grid, G)
    flags = []
    for name, cell in (('start', s_cell), ('goal', g_cell)):
        if not free[cell]:
            flags.append(name + '_cell_blocked_after_inflation')
    if flags:
        return None, None, None, flags
    if segment_collision(S, G, obstacles, 0.1) is None:
        return 1.0, 0.0, [S, G], flags
    costs = np.where(free, 1.0, np.inf)
    mcp = MCP_Geometric(costs, fully_connected=True, sampling=(RES_GRID,) * 3)
    cumulative, _ = mcp.find_costs([s_cell], [g_cell])
    if not np.isfinite(cumulative[g_cell]):
        return None, None, None, flags + ['unreachable']
    cells = mcp.traceback(g_cell)
    pts = [[float(grid['gx'][i]), float(grid['gy'][j]), float(grid['gz'][k])] for i, j, k in cells]
    pts[0], pts[-1] = S, G

    def visible(a, b):
        if min(a[2], b[2]) < Z_GROUND:
            return False
        return segment_collision(a, b, obstacles, 0.1) is None

    smooth = [pts[0]]
    i = 0
    while i < len(pts) - 1:
        j = i + 1
        while j + 1 < len(pts) and visible(pts[i], pts[j + 1]):
            j += 1
        smooth.append(pts[j])
        i = j
    length = sum(math.dist(a, b) for a, b in zip(smooth, smooth[1:]))
    theta = 0.0
    for a, b, c in zip(smooth, smooth[1:], smooth[2:]):
        u = np.subtract(b, a)
        v = np.subtract(c, b)
        nu, nv = np.linalg.norm(u), np.linalg.norm(v)
        if nu > 1e-9 and nv > 1e-9:
            theta += math.acos(max(-1.0, min(1.0, float(np.dot(u, v) / (nu * nv)))))
    return length / math.dist(S, G), theta, smooth, flags


def compute(task):
    sys.path.insert(0, _REPO + '/Improved-3D-Diffusion-Policy')
    from diffusion_policy_3d.env.collision_geom import ObstacleSpec, segment_obstacle_collision
    fields = ('name', 'shape', 'x', 'y', 'z', 'yaw', 'radius', 'height', 'size_x', 'size_y', 'size_z')
    obstacles = [ObstacleSpec(**{k: o[k] for k in fields}) for o in task['geometry']]
    t0 = time.time()
    out = dict(task_id=task['task_id'], split=task['split'], role=task['role'], world=task['world'],
               start=task['start'], goal=task['goal'], n_obstacles=len(obstacles),
               chord_m=math.dist(task['start'], task['goal']))
    out['blocked'] = segment_obstacle_collision(task['start'], task['goal'], obstacles, 0.1) is not None
    for w in WIDTHS:
        c = corridor_density(task, w)
        out['rho_%.1f' % w] = c['rho']
        out['n_%.1f' % w] = c['n']
        out['corridor_degenerate'] = c['degenerate']
    grid = build_grid(task)
    clear = clearance_field(grid)
    s_cell, g_cell = cell_of(grid, task['start']), cell_of(grid, task['goal'])
    ball = endpoint_balls(grid, task['start'], task['goal'])
    out['b_m'] = max_min_clearance(clear, s_cell, g_cell, ball)
    band = route_band(grid, task['start'], task['goal'])
    out['b_band_m'] = max_min_clearance(np.where(band, clear, 0.0), s_cell, g_cell, ball & band)
    out['endpoint_clearance_m'] = float(min(clear[s_cell], clear[g_cell]))
    kappa, theta, path, flags = shortest_path(task, grid, obstacles, segment_obstacle_collision)
    out['kappa'] = kappa
    out['theta_rad'] = theta
    out['path'] = path
    out['path_zmax_m'] = max(p[2] for p in path) if path else None
    out['flags'] = flags
    out['grid_shape'] = list(grid['raw'].shape)
    out['seconds'] = round(time.time() - t0, 1)
    return out


def _init(repo):
    global _REPO
    _REPO = repo


def main(tasks_path, repo, out_path, workers='6'):
    tasks = json.load(open(tasks_path))
    with Pool(int(workers), initializer=_init, initargs=(repo,)) as pool:
        rows = []
        for i, row in enumerate(pool.imap_unordered(compute, tasks)):
            rows.append(row)
            if (i + 1) % 10 == 0:
                print('done', i + 1, 'last %.1fs' % row['seconds'], flush=True)
    rows.sort(key=lambda r: (r['split'], r['task_id']))
    json.dump(rows, open(out_path, 'w'))
    print('wrote', out_path, len(rows))


if __name__ == '__main__':
    main(*sys.argv[1:])
