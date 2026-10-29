"""Procedural maps that reward heterogeneous sensing (wide/short vs narrow/long).

Each map mixes zones where a narrow long-range sensor sees more (open halls,
warehouse aisles) with zones where a wide short-range sensor sees more
(cluttered cubicles, small rooms). Output PNGs use exactly the maps_medium
format, so Env loads them unchanged via --map-directory:

    127 = obstacle, 195 = free, 208 = start marker; 2 px per 0.4 m cell.

Geometry is laid out on a block grid whose block size equals NODE_RESOLUTION
(10 cells = 4 m). Env anchors the node lattice at the start cell, and the start
cell is placed at a block centre, so every free block contains exactly one
graph node and the node count equals the free block count. Thin walls (0.8 m)
lie on block boundaries and pillars (1.6 m) on block corners, never on node
centres or on the straight lines between orthogonal neighbour nodes.

Layout written by one run:
    OUT/maps/<i>.png     training/evaluation maps (only PNGs: Env lists this dir)
    OUT/zones/<i>.png    per-cell zone labels at Env resolution (see ZONE_NAMES)
    OUT/meta.jsonl       per-map statistics and validation results
    OUT/preview.png      zone-coloured preview of the first maps

Example:
    python tools/gen_hetero_maps.py --out maps_hetero_train --num 4000 --seed 0
    python tools/gen_hetero_maps.py --out maps_hetero_test --num 100 --seed 1
    python driver_v1.py --map-directory maps_hetero_train/maps ...
"""
import argparse
from collections import deque
import json
from multiprocessing import Pool
from pathlib import Path
import sys

import numpy as np
from PIL import Image
from skimage.measure import label

BLOCK = 10           # cells per block; 10 * 0.4 m = NODE_RESOLUTION (4 m)
CELL_SIZE = 0.4      # meters per Env cell (parameter.CELL_SIZE)
PX = 2               # PNG pixels per Env cell (Env applies block_reduce(2, min))
OBSTACLE_PX, FREE_PX, START_PX = 127, 195, 208

ZONE_NAMES = {0: 'solid', 1: 'hall', 2: 'aisles', 3: 'clutter', 4: 'rooms', 5: 'connector'}
NARROW_ZONES = (1, 2)   # long sight lines: favour narrow long-range sensors
WIDE_ZONES = (3, 4)     # short sight lines: favour wide short-range sensors
FAMILIES = {'narrow': ('hall', 'aisles'), 'wide': ('clutter', 'rooms')}
ZONE_IDS = {name: i for i, name in ZONE_NAMES.items()}
ZONE_COLORS = np.array([[40, 40, 40], [86, 156, 214], [78, 201, 176], [214, 157, 86],
                        [206, 120, 160], [200, 200, 200]], dtype=np.uint8)

# Sensors used for validation and statistics: (FoV degrees, range meters).
# Equal nominal sector area (~56.5 m^2): which one sees more depends on the scene.
WIDE_SENSOR = (180.0, 6.0)
NARROW_SENSOR = (45.0, 12.0)
# Mixed maps: narrow-zone mean sight line >= 1.25x the wide-zone one, and the
# narrow/wide visible-area ratio >= 1.12x higher in narrow zones.
MIN_SIGHT_RATIO = 1.25
MIN_AREA_CONTRAST = 1.12


class MapRejected(Exception):
    pass


# ---------------------------------------------------------------------------
# Block-level layout
# ---------------------------------------------------------------------------
class Layout:
    """Free blocks, thin walls between adjacent blocks and corner pillars."""

    def __init__(self, n_blocks):
        self.n = n_blocks
        self.free = np.zeros((n_blocks, n_blocks), dtype=bool)
        self.zone = np.zeros((n_blocks, n_blocks), dtype=np.uint8)
        self.region = np.full((n_blocks, n_blocks), -1, dtype=int)
        self.vwall = np.zeros((n_blocks, n_blocks - 1), dtype=bool)   # (r,c)|(r,c+1)
        self.hwall = np.zeros((n_blocks - 1, n_blocks), dtype=bool)   # (r,c)/(r+1,c)
        self.pillar = np.zeros((n_blocks - 1, n_blocks - 1), dtype=bool)  # corner below-right of (r,c)

    def carve(self, r, c, zone, region):
        self.free[r, c] = True
        self.zone[r, c] = zone
        self.region[r, c] = region

    def wall_ref(self, a, b):
        (r0, c0), (r1, c1) = a, b
        if r0 == r1 and abs(c0 - c1) == 1:
            return self.vwall, (r0, min(c0, c1))
        if c0 == c1 and abs(r0 - r1) == 1:
            return self.hwall, (min(r0, r1), c0)
        raise ValueError('Walls exist only between orthogonally adjacent blocks')

    def set_wall(self, a, b, value=True):
        array, index = self.wall_ref(a, b)
        array[index] = value

    def has_wall(self, a, b):
        array, index = self.wall_ref(a, b)
        return bool(array[index])

    def neighbors(self, r, c):
        """Free orthogonal neighbours reachable without crossing a wall."""
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < self.n and 0 <= cc < self.n and self.free[rr, cc] \
                    and not self.has_wall((r, c), (rr, cc)):
                yield rr, cc

    def components(self, mask=None):
        mask = self.free if mask is None else (self.free & mask)
        comp = np.full(self.free.shape, -1, dtype=int)
        count = 0
        for r, c in zip(*np.nonzero(mask)):
            if comp[r, c] >= 0:
                continue
            comp[r, c] = count
            queue = deque([(r, c)])
            while queue:
                cr, cc = queue.popleft()
                for nr, nc in self.neighbors(cr, cc):
                    if mask[nr, nc] and comp[nr, nc] < 0:
                        comp[nr, nc] = count
                        queue.append((nr, nc))
            count += 1
        return comp, count

    def connected_within(self, mask):
        return self.components(mask)[1] <= 1


def split_rect(rect, rng, min_side, lo=0.4, hi=0.6):
    r0, c0, r1, c1 = rect
    h, w = r1 - r0, c1 - c0
    axis = 0 if h > w else 1 if w > h else int(rng.integers(2))
    length = h if axis == 0 else w
    if length < 2 * min_side:
        return None
    k = int(np.clip(round(length * rng.uniform(lo, hi)), min_side, length - min_side))
    if axis == 0:
        return (r0, c0, r0 + k, c1), (r0 + k, c0, r1, c1)
    return (r0, c0, r1, c0 + k), (r0, c0 + k, r1, c1)


def rect_blocks(rect):
    r0, c0, r1, c1 = rect
    return [(r, c) for r in range(r0, r1) for c in range(c0, c1)]


def jitter_inset(rect, rng, min_side=5):
    """Drop up to one block row/column per side, keeping the node budget in range."""
    r0, c0, r1, c1 = rect
    if r1 - r0 > min_side and rng.random() < 0.5:
        r0 += 1
    if r1 - r0 > min_side and rng.random() < 0.5:
        r1 -= 1
    if c1 - c0 > min_side and rng.random() < 0.5:
        c0 += 1
    if c1 - c0 > min_side and rng.random() < 0.5:
        c1 -= 1
    return r0, c0, r1, c1


# --- zone fillers ----------------------------------------------------------
def fill_hall(layout, rect, rng, region):
    """Large open area with sparse pillars: long sight lines in every direction."""
    r0, c0, r1, c1 = rect
    h, w = r1 - r0, c1 - c0
    hh = int(rng.integers(max(min(h, 5), int(np.ceil(0.5 * h))), h + 1))
    ww = int(rng.integers(max(min(w, 5), int(np.ceil(0.5 * w))), w + 1))
    rr0 = r0 + int(rng.integers(0, h - hh + 1))
    cc0 = c0 + int(rng.integers(0, w - ww + 1))
    for r, c in rect_blocks((rr0, cc0, rr0 + hh, cc0 + ww)):
        layout.carve(r, c, ZONE_IDS['hall'], region)
    spacing = int(rng.choice([3, 4]))
    off_r, off_c = int(rng.integers(spacing)), int(rng.integers(spacing))
    for r in range(rr0, rr0 + hh - 1):
        for c in range(cc0, cc0 + ww - 1):
            if (r - off_r) % spacing == 0 and (c - off_c) % spacing == 0 and rng.random() < 0.8:
                layout.pillar[r, c] = True
    # A few single-block crates add mild occlusion without closing sight lines.
    for _ in range(int(rng.integers(0, 3))):
        r = int(rng.integers(rr0 + 1, rr0 + hh - 1)) if hh > 2 else rr0
        c = int(rng.integers(cc0 + 1, cc0 + ww - 1)) if ww > 2 else cc0
        layout.free[r, c] = False
        layout.zone[r, c] = 0
        if not layout.connected_within(layout.region == region):
            layout.carve(r, c, ZONE_IDS['hall'], region)


def fill_aisles(layout, rect, rng, region):
    """Warehouse: parallel 1-block shelves and 4 m aisles along the long axis."""
    r0, c0, r1, c1 = rect
    h, w = r1 - r0, c1 - c0
    along_rows = w >= h     # aisles run along columns (horizontal) if wide
    length, cross = (w, h) if along_rows else (h, w)
    start = int(rng.integers(2))
    mid = int(rng.integers(length // 3, 2 * length // 3 + 1)) if length >= 12 else None
    for v in range(cross):
        shelf_row = (v + start) % 2 == 1 and 0 < v < cross - 1
        for u in range(length):
            cross_aisle = u in (0, length - 1) or u == mid
            if shelf_row and not cross_aisle:
                continue
            r, c = (r0 + v, c0 + u) if along_rows else (r0 + u, c0 + v)
            layout.carve(r, c, ZONE_IDS['aisles'], region)


def fill_clutter(layout, rect, rng, region):
    """Cubicles: furniture blocks, partitions and pillars; sight lines of 4-8 m."""
    rect = jitter_inset(rect, rng)
    blocks = rect_blocks(rect)
    for r, c in blocks:
        layout.carve(r, c, ZONE_IDS['clutter'], region)
    mask = layout.region == region
    for index in rng.permutation(len(blocks))[: int(0.25 * len(blocks))]:
        r, c = blocks[index]
        layout.free[r, c] = False
        layout.zone[r, c] = 0
        if not layout.connected_within(mask):
            layout.carve(r, c, ZONE_IDS['clutter'], region)
    pairs = [((r, c), (r + dr, c + dc)) for r, c in blocks for dr, dc in ((0, 1), (1, 0))
             if (r + dr, c + dc) in set(blocks)]
    for index in rng.permutation(len(pairs)):
        a, b = pairs[index]
        if layout.free[a] and layout.free[b] and rng.random() < 0.7:
            layout.set_wall(a, b)
            if not layout.connected_within(mask):
                layout.set_wall(a, b, False)
    r0, c0, r1, c1 = rect
    for r in range(r0, r1 - 1):
        for c in range(c0, c1 - 1):
            if layout.free[r:r + 2, c:c + 2].all() and rng.random() < 0.85:
                layout.pillar[r, c] = True


def fill_rooms(layout, rect, rng, region):
    """Apartment/office suite: a grid of mostly 8 m rooms joined by offset doors.

    No through-corridor: every sight line ends at a room wall within ~8 m,
    apart from the occasional view through a doorway.
    """
    r0, c0, r1, c1 = jitter_inset(rect, rng)

    def cuts(lo, hi):
        edges, x = [lo], lo
        while hi - x >= 4:
            x += 2 if rng.random() < 0.8 else 3
            edges.append(x)
        if edges[-1] != hi:
            edges[-1] = hi if hi - edges[-1] < 2 else edges[-1]
            if edges[-1] != hi:
                edges.append(hi)
        return edges

    rows, cols = cuts(r0, r1), cuts(c0, c1)
    room_of = {}
    for i, (a, b) in enumerate(zip(rows, rows[1:])):
        for j, (c, d) in enumerate(zip(cols, cols[1:])):
            for r in range(a, b):
                for cc in range(c, d):
                    layout.carve(r, cc, ZONE_IDS['rooms'], region)
                    room_of[(r, cc)] = (i, j)
    shared = {}
    for (r, c), room in room_of.items():
        for nb in ((r, c + 1), (r + 1, c)):
            other = room_of.get(nb)
            if other is not None and other != room:
                layout.set_wall((r, c), nb)
                shared.setdefault(tuple(sorted((room, other))), []).append(((r, c), nb))
    # Random spanning tree of rooms (one door each), plus a few loop doors.
    rooms = sorted(set(room_of.values()))
    parent = {room: room for room in rooms}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    keys = list(shared)
    for index in rng.permutation(len(keys)):
        a, b = keys[index]
        walls = shared[keys[index]]
        if find(a) != find(b) or rng.random() < 0.15:
            parent[find(a)] = find(b)
            door = walls[int(rng.integers(len(walls)))]
            layout.set_wall(*door, False)


FILLERS = dict(hall=fill_hall, aisles=fill_aisles, clutter=fill_clutter, rooms=fill_rooms)


def plan_regions(category, rng, n_blocks, min_side=5):
    usable = (1, 1, n_blocks - 1, n_blocks - 1)
    halves = split_rect(usable, rng, min_side)
    if category == 'mixed':
        families = ['narrow', 'wide'] if rng.random() < 0.5 else ['wide', 'narrow']
    else:
        family = 'narrow' if category == 'pure_narrow' else 'wide'
        families = [family, family]
    plan = []
    for rect, family in zip(halves, families):
        parts = split_rect(rect, rng, min_side) if rng.random() < 0.5 else None
        parts = parts if parts else (rect,)
        kinds = list(rng.permutation(FAMILIES[family]))
        for index, part in enumerate(parts):
            kind = kinds[index % 2] if rng.random() < 0.75 else str(rng.choice(FAMILIES[family]))
            plan.append((part, kind))
    if category != 'mixed' and len({kind for _, kind in plan}) == 1 and len(plan) > 1:
        # Pure maps still mix both zone types of their family.
        other = [k for k in FAMILIES[families[0]] if k != plan[-1][1]][0]
        plan[-1] = (plan[-1][0], other)
    return plan


def connect(layout, rng):
    """Wall off region borders, then open doors / carve corridors until connected."""
    n = layout.n
    for r in range(n):
        for c in range(n):
            for rr, cc in ((r, c + 1), (r + 1, c)):
                if rr < n and cc < n and layout.free[r, c] and layout.free[rr, cc] \
                        and layout.region[r, c] != layout.region[rr, cc]:
                    layout.set_wall((r, c), (rr, cc))
    region_doors = {}
    for _ in range(4 * n * n):
        comp, count = layout.components()
        if count <= 1:
            break
        main = np.argmax(np.bincount(comp[comp >= 0]))
        doors = []
        for r in range(n):
            for c in range(n):
                for rr, cc in ((r, c + 1), (r + 1, c)):
                    if rr < n and cc < n and layout.free[r, c] and layout.free[rr, cc] \
                            and layout.has_wall((r, c), (rr, cc)) and comp[r, c] != comp[rr, cc] \
                            and main in (comp[r, c], comp[rr, cc]):
                        doors.append(((r, c), (rr, cc)))
        if doors:
            a, b = doors[int(rng.integers(len(doors)))]
            layout.set_wall(a, b, False)
            key = tuple(sorted((int(layout.region[a]), int(layout.region[b]))))
            region_doors[key] = region_doors.get(key, 0) + 1
            continue
        # No shared wall: carve an L corridor from the main component.
        main_cells = np.argwhere(comp == main)
        other_cells = np.argwhere((comp >= 0) & (comp != main))
        d = np.abs(main_cells[:, None, :] - other_cells[None, :, :]).sum(-1)
        i, j = np.unravel_index(np.argmin(d), d.shape)
        (r, c), (tr, tc) = main_cells[i], other_cells[j]
        path = [(r, c)]
        order = [(0, 1), (1, 0)] if rng.random() < 0.5 else [(1, 0), (0, 1)]
        for axis, _ in order:
            while (path[-1][axis] != (tr, tc)[axis]):
                cur = list(path[-1])
                cur[axis] += 1 if (tr, tc)[axis] > cur[axis] else -1
                path.append(tuple(cur))
        for a, b in zip(path, path[1:]):
            if not layout.free[b]:
                layout.carve(*b, ZONE_IDS['connector'], -2)
            layout.set_wall(a, b, False)
    else:
        raise MapRejected('connect: did not converge')
    # Extra doors create loops between adjacent regions.
    for r in range(n):
        for c in range(n):
            for rr, cc in ((r, c + 1), (r + 1, c)):
                if rr < n and cc < n and layout.free[r, c] and layout.free[rr, cc] \
                        and layout.has_wall((r, c), (rr, cc)) \
                        and layout.region[r, c] != layout.region[rr, cc] and rng.random() < 0.12:
                    layout.set_wall((r, c), (rr, cc), False)


# ---------------------------------------------------------------------------
# Cell rendering
# ---------------------------------------------------------------------------
def render(layout):
    n, size = layout.n, layout.n * BLOCK
    occupied = ~np.kron(layout.free, np.ones((BLOCK, BLOCK), dtype=bool))
    zones = np.kron(layout.zone, np.ones((BLOCK, BLOCK), dtype=np.uint8))
    for r, c in zip(*np.nonzero(layout.vwall)):
        if layout.free[r, c] and layout.free[r, c + 1]:
            occupied[r * BLOCK:(r + 1) * BLOCK, (c + 1) * BLOCK - 1:(c + 1) * BLOCK + 1] = True
    for r, c in zip(*np.nonzero(layout.hwall)):
        if layout.free[r, c] and layout.free[r + 1, c]:
            occupied[(r + 1) * BLOCK - 1:(r + 1) * BLOCK + 1, c * BLOCK:(c + 1) * BLOCK] = True
    for r, c in zip(*np.nonzero(layout.pillar)):
        if layout.free[r:r + 2, c:c + 2].all():
            y, x = (r + 1) * BLOCK, (c + 1) * BLOCK
            occupied[y - 2:y + 2, x - 2:x + 2] = True
    zones[occupied] = 0
    assert occupied.shape == (size, size)
    return occupied, zones


def block_center(r, c):
    return r * BLOCK + BLOCK // 2, c * BLOCK + BLOCK // 2   # (row, col) cell


def start_marker_cells(r, c):
    """Cells whose 11th row-major element is the block centre (Env uses index 10)."""
    cy, cx = block_center(r, c)
    cells = [(cy - 2, x) for x in range(cx - 2, cx + 3)] + [(cy - 1, x) for x in range(cx - 2, cx + 3)]
    cells += [(cy, x) for x in range(cx, cx + 3)] + [(cy + 1, x) for x in range(cx, cx + 3)]
    return cells


# ---------------------------------------------------------------------------
# Ray casting identical to utils/ray_casting.collision_check
# ---------------------------------------------------------------------------
_RAY_CACHE = {}


def _bresenham(x1, y1):
    x, y = 0, 0
    dx, dy = abs(x1), abs(y1)
    error = dx - dy
    x_inc = 1 if x1 > 0 else -1
    y_inc = 1 if y1 > 0 else -1
    dx *= 2
    dy *= 2
    path = []
    while not (x == x1 and y == y1):    # endpoint is never written
        path.append((x, y))
        if error > 0:
            x += x_inc
            error -= dy
        else:
            y += y_inc
            error += dx
    return path


def ray_table(range_cells):
    """Offsets [720, L, 2] (dx, dy) for 0.5-degree rays and their lengths."""
    key = round(float(range_cells), 6)
    if key not in _RAY_CACHE:
        angles = np.deg2rad(np.arange(0, 360, 0.5))
        paths = [_bresenham(int(np.round(np.cos(a) * range_cells)), int(np.round(np.sin(a) * range_cells)))
                 for a in angles]
        length = max(len(p) for p in paths)
        offsets = np.zeros((len(paths), length, 2), dtype=int)
        lengths = np.array([len(p) for p in paths])
        for i, path in enumerate(paths):
            offsets[i, :len(path)] = path
        _RAY_CACHE[key] = (np.rad2deg(angles), offsets, lengths)
    return _RAY_CACHE[key]


def visible_cells(occupied, cell, range_cells, heading=None, fov=360.0):
    """Flat indices revealed by one scan from cell=(row, col) on an unknown map."""
    angles, offsets, lengths = ray_table(range_cells)
    if fov < 360:
        diff = (angles - heading + 180) % 360 - 180
        keep = np.abs(diff) <= fov / 2 + 1e-9
        offsets, lengths = offsets[keep], lengths[keep]
    h, w = occupied.shape
    xs = cell[1] + offsets[..., 0]
    ys = cell[0] + offsets[..., 1]
    step = np.arange(offsets.shape[1])[None, :]
    inside = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h) & (step < lengths[:, None])
    # Out of bounds ends a ray permanently.
    inside = np.cumprod(inside, axis=1).astype(bool)
    xs_c, ys_c = np.clip(xs, 0, w - 1), np.clip(ys, 0, h - 1)
    occ = occupied[ys_c, xs_c] & inside
    # Mark every cell up to and including the first obstacle.
    before_hit = np.cumsum(occ, axis=1) - occ == 0
    marked = inside & before_hit
    return np.unique(ys_c[marked] * w + xs_c[marked])


def sight_length(occupied, cell, max_cells):
    """Mean free distance (cells) over 360 degrees, capped at max_cells."""
    angles, offsets, lengths = ray_table(max_cells)
    h, w = occupied.shape
    xs, ys = cell[1] + offsets[..., 0], cell[0] + offsets[..., 1]
    step = np.arange(offsets.shape[1])[None, :]
    inside = np.cumprod((xs >= 0) & (xs < w) & (ys >= 0) & (ys < h) & (step < lengths[:, None]), axis=1).astype(bool)
    occ = occupied[np.clip(ys, 0, h - 1), np.clip(xs, 0, w - 1)] | ~inside
    first = np.where(occ.any(1), occ.argmax(1), lengths)
    return float(first.mean())


def best_sector_area(occupied, cell, sensor, headings=12):
    fov, rng_m = sensor
    return max(len(visible_cells(occupied, cell, rng_m / CELL_SIZE, h, fov))
               for h in np.arange(0, 360, 360 / headings))


# ---------------------------------------------------------------------------
# One map
# ---------------------------------------------------------------------------
def build_map(index, seed, category, n_blocks, min_nodes, max_nodes, init_scan_m, max_tries=200):
    rng = np.random.default_rng([seed, index])
    reasons = {}
    for attempt in range(max_tries):
        try:
            return _attempt(rng, category, n_blocks, min_nodes, max_nodes, init_scan_m, attempt, reasons)
        except MapRejected as error:
            key = str(error).split(':')[0]
            reasons[key] = reasons.get(key, 0) + 1
    raise RuntimeError(f'map {index}: no valid layout after {max_tries} tries: {reasons}')


def _attempt(rng, category, n_blocks, min_nodes, max_nodes, init_scan_m, attempt, reasons):
    layout = Layout(n_blocks)
    plan = plan_regions(category, rng, n_blocks)
    for region, (rect, kind) in enumerate(plan):
        FILLERS[kind](layout, rect, rng, region)
    connect(layout, rng)

    nodes = int(layout.free.sum())
    if not min_nodes <= nodes <= max_nodes:
        raise MapRejected(f'node_count: {nodes}')
    narrow = int(np.isin(layout.zone, NARROW_ZONES).sum())
    wide = int(np.isin(layout.zone, WIDE_ZONES).sum())
    narrow_share = narrow / max(1, narrow + wide)
    if category == 'mixed' and not 0.3 <= narrow_share <= 0.7:
        raise MapRejected(f'balance: {narrow_share:.2f}')

    occupied, zones = render(layout)

    # Start: an open block (prefer halls) whose 3x3 neighbourhood is free and wall-free.
    candidates = []
    for r, c in zip(*np.nonzero(layout.free)):
        if 1 <= r < n_blocks - 1 and 1 <= c < n_blocks - 1 and layout.free[r - 1:r + 2, c - 1:c + 2].all() \
                and not any(layout.has_wall((r, c), nb) for nb in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1))):
            candidates.append((int(layout.zone[r, c] == ZONE_IDS['hall']), r, c))
    if not candidates:
        raise MapRejected('start: no open block')
    best = max(k for k, _, _ in candidates)
    pool = [(r, c) for k, r, c in candidates if k == best]
    sr, sc = pool[int(rng.integers(len(pool)))]
    start = block_center(sr, sc)

    # Env: initial 360-degree scan, then four agents on distinct free nodes inside it.
    seen = np.zeros(occupied.size, dtype=bool)
    seen[visible_cells(occupied, start, init_scan_m / CELL_SIZE)] = True
    seen = seen.reshape(occupied.shape)
    init_nodes = sum(1 for r, c in zip(*np.nonzero(layout.free))
                     if seen[block_center(r, c)] and not occupied[block_center(r, c)])
    if init_nodes < 6:
        raise MapRejected(f'start_nodes: {init_nodes}')

    # Coverage feasibility: everything must be visible from some node with the
    # weakest range at 360 degrees; invisible slivers are sealed as obstacles.
    union = np.zeros(occupied.size, dtype=bool)
    centers = [block_center(r, c) for r, c in zip(*np.nonzero(layout.free))]
    min_range_cells = min(WIDE_SENSOR[1], NARROW_SENSOR[1]) / CELL_SIZE
    for cell in centers:
        union[visible_cells(occupied, cell, min_range_cells)] = True
    union = union.reshape(occupied.shape)
    free = ~occupied
    invisible = free & ~union
    invisible_fraction = float(invisible.sum() / free.sum())
    if invisible_fraction > 0.01:
        raise MapRejected(f'visibility: {invisible_fraction:.4f}')
    occupied = occupied | invisible

    # Single 8-connected free component (Env uses connectivity=2), containing start.
    labels = label(~occupied, connectivity=2)
    if labels.max() != 1:
        main = labels[start]
        stray = (labels > 0) & (labels != main)
        if stray.sum() > 0.005 * (~occupied).sum():
            raise MapRejected('connectivity: large stray component')
        occupied |= stray
    for cell in centers:
        if occupied[cell]:
            raise MapRejected('node_blocked')
    zones[occupied] = 0

    # Heterogeneity statistics per zone family (unknown map, best of 12 headings).
    sample = rng.permutation(len(centers))[: min(80, len(centers))]
    stats = {}
    for family, ids in (('narrow', NARROW_ZONES), ('wide', WIDE_ZONES)):
        picked = [centers[i] for i in sample if zones[centers[i]] in ids]
        if picked:
            wide_area = np.mean([best_sector_area(occupied, p, WIDE_SENSOR) for p in picked])
            narrow_area = np.mean([best_sector_area(occupied, p, NARROW_SENSOR) for p in picked])
            stats[family] = dict(nodes=len(picked),
                                 sight_m=round(np.mean([sight_length(occupied, p, 40) for p in picked]) * CELL_SIZE, 2),
                                 wide_sensor_m2=round(wide_area * CELL_SIZE ** 2, 1),
                                 narrow_sensor_m2=round(narrow_area * CELL_SIZE ** 2, 1),
                                 narrow_over_wide=round(narrow_area / wide_area, 3))
    # Each zone family must favour its own sensor, and mixed maps must show a
    # clear contrast; otherwise the map does not test capability allocation.
    if 'narrow' in stats and stats['narrow']['narrow_over_wide'] < 1.0:
        raise MapRejected('contrast: narrow zones do not favour the narrow sensor')
    if 'wide' in stats and stats['wide']['narrow_over_wide'] > 1.0:
        raise MapRejected('contrast: wide zones do not favour the wide sensor')
    if category == 'mixed':
        if 'narrow' not in stats or 'wide' not in stats:
            raise MapRejected('contrast: mixed map lacks a zone family')
        if (stats['narrow']['sight_m'] < MIN_SIGHT_RATIO * stats['wide']['sight_m'] or
                stats['narrow']['narrow_over_wide'] < MIN_AREA_CONTRAST * stats['wide']['narrow_over_wide']):
            raise MapRejected('contrast: weak mixed-map contrast')

    pixels = np.where(occupied, OBSTACLE_PX, FREE_PX).astype(np.uint8)
    for cell in start_marker_cells(sr, sc):
        if occupied[cell]:
            raise MapRejected('start_marker')
        pixels[cell] = START_PX
    meta = dict(category=category, attempts=attempt + 1, rejections=dict(reasons),
                size_m=n_blocks * BLOCK * CELL_SIZE, nodes=nodes,
                free_m2=round(float((~occupied).sum()) * CELL_SIZE ** 2, 1),
                narrow_share=round(narrow_share, 3), regions=[kind for _, kind in plan],
                start_cell_xy=[int(start[1]), int(start[0])], initial_nodes=init_nodes,
                sealed_invisible_fraction=round(invisible_fraction, 5), zone_stats=stats)
    return pixels, zones, meta


def load_like_env(path):
    """Replica of Env.import_ground_truth for validation without the full stack."""
    from skimage import io
    from skimage.measure import block_reduce
    ground_truth = io.imread(path, as_gray=True)
    if ground_truth.dtype != np.uint8:
        ground_truth = (ground_truth * 255).round()
    ground_truth = block_reduce(ground_truth.astype(int), 2, np.min)
    robot_cell = np.array(np.nonzero(ground_truth == START_PX))
    robot_cell = np.array([robot_cell[1, 10], robot_cell[0, 10]])
    ground_truth = (ground_truth > 150) | ((ground_truth <= 80) & (ground_truth >= 50))
    return ground_truth * 254 + 1, robot_cell


def _job(args):
    index, seed, category, out, n_blocks, min_nodes, max_nodes, init_scan_m = args
    pixels, zones, meta = build_map(index, seed, category, n_blocks, min_nodes, max_nodes, init_scan_m)
    Image.fromarray(np.kron(pixels, np.ones((PX, PX), dtype=np.uint8))).save(out / 'maps' / f'{index}.png')
    Image.fromarray(zones).save(out / 'zones' / f'{index}.png')
    meta.update(index=index, file=f'{index}.png')
    return meta


def write_preview(out, metas, count=12, columns=4):
    tiles = []
    for meta in metas[:count]:
        zones = np.array(Image.open(out / 'zones' / meta['file']))
        rgb = ZONE_COLORS[zones]
        x, y = meta['start_cell_xy']
        rgb[max(0, y - 3):y + 4, max(0, x - 3):x + 4] = [255, 60, 60]
        tiles.append(np.pad(rgb, ((4, 4), (4, 4), (0, 0)), constant_values=255))
    while len(tiles) % columns:
        tiles.append(np.full_like(tiles[0], 255))
    rows = [np.concatenate(tiles[i:i + columns], axis=1) for i in range(0, len(tiles), columns)]
    image = Image.fromarray(np.concatenate(rows, axis=0))
    image.resize((image.width * 2, image.height * 2), Image.NEAREST).save(out / 'preview.png')


def parse_mix(text):
    mix = {}
    for part in text.split(','):
        name, value = part.split('=')
        if name not in ('mixed', 'pure_narrow', 'pure_wide'):
            raise argparse.ArgumentTypeError(f'unknown category {name}')
        mix[name] = float(value)
    total = sum(mix.values())
    return {k: v / total for k, v in mix.items()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', required=True, help='Output root; maps go to OUT/maps')
    parser.add_argument('--num', type=int, default=100)
    parser.add_argument('--seed', type=int, default=0, help='Use different seeds for train and test sets')
    parser.add_argument('--start-index', type=int, default=0)
    parser.add_argument('--blocks', type=int, default=20, help='Canvas side in 4 m blocks (20 -> 80 m)')
    parser.add_argument('--mix', type=parse_mix, default=parse_mix('mixed=0.7,pure_narrow=0.15,pure_wide=0.15'))
    parser.add_argument('--min-nodes', type=int, default=140)
    parser.add_argument('--max-nodes', type=int, default=240, help='Keep below NODE_PADDING_SIZE (360)')
    parser.add_argument('--init-scan-range', type=float, default=6.0, help='INITIAL_SCAN_RANGE in meters')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--preview', type=int, default=12)
    args = parser.parse_args(argv)

    out = Path(args.out)
    (out / 'maps').mkdir(parents=True, exist_ok=True)
    (out / 'zones').mkdir(parents=True, exist_ok=True)
    names = list(args.mix)
    weights = np.array([args.mix[n] for n in names])
    # Deterministic category assignment with exact proportions.
    counts = np.floor(weights * args.num).astype(int)
    counts[np.argmax(weights)] += args.num - counts.sum()
    categories = np.repeat(names, counts)
    np.random.default_rng([args.seed, 10 ** 9]).shuffle(categories)
    jobs = [(args.start_index + i, args.seed, str(cat), out, args.blocks, args.min_nodes,
             args.max_nodes, args.init_scan_range) for i, cat in enumerate(categories)]
    metas = []
    with open(out / 'meta.jsonl', 'w') as log:
        runner = Pool(args.workers).imap if args.workers > 1 else map
        for done, meta in enumerate(runner(_job, jobs), 1):
            metas.append(meta)
            log.write(json.dumps(meta) + '\n')
            if done % 50 == 0 or done == len(jobs):
                print(f'{done}/{len(jobs)} maps', file=sys.stderr, flush=True)
    if args.preview:
        write_preview(out, metas, args.preview)
    nodes = np.array([m['nodes'] for m in metas])
    print(json.dumps(dict(maps=len(metas), nodes_min=int(nodes.min()), nodes_mean=float(nodes.mean()),
                          nodes_max=int(nodes.max()),
                          categories={n: int((categories == n).sum()) for n in names})))


if __name__ == '__main__':
    main()
