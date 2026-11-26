#!/usr/bin/env python3
"""HW4 path planning: shortest path (visibility graph + A*) and safest path (Voronoi graph + A*).

Usage: python3 hw4_planner.py [--obstacle-tag ID X Y FACE ...] [--obstacle-size FT] [--obstacle-center X Y]
                              [--robot-radius M] [--margin M]
--obstacle-tag (ft, facing in deg, repeatable): a measured tag on an obstacle face. The face it sits on sets that
side of the box; a tag on a +-x face also gives the box's center y, one on a +-y face its center x. A side with no
tag is placed symmetrically about the center. Only measured tags go into the robot's map.
Writes ~/HW4/path_shortest.txt and ~/HW4/path_safest.txt (x, y, theta per line, meters and radians),
~/HW4/map.txt (the tag map the robot uses), and the figure ~/HW4/results/planning.png.
"""
import argparse, heapq, math, os
import numpy as np

FT = 0.3048
HW4 = os.path.expanduser('~/HW4')

# ---- world (meters) ----

def make_world(args):
    L = 8 * FT
    cx, cy = (v * FT for v in args.obstacle_center)
    s = args.obstacle_size * FT
    box = [cx - s / 2, cy - s / 2, cx + s / 2, cy + s / 2]     # xmin, ymin, xmax, ymax
    obstacle_tags = {}
    if args.obstacle_tag:
        faces = {}
        for k, x, y, face in args.obstacle_tag:
            x, y, face = x * FT, y * FT, math.radians(face)
            obstacle_tags[int(k)] = (x, y, face)
            # which face: the closest of +x, +y, -x, -y to the direction the tag faces
            side = round(face / (math.pi / 2)) % 4
            faces[side] = (x, y)
        xs = [faces[d][0] for d in (1, 3) if d in faces]     # tags on the +-y faces sit at the center x
        ys = [faces[d][1] for d in (0, 2) if d in faces]     # tags on the +-x faces sit at the center y
        cx = np.mean(xs) if xs else cx
        cy = np.mean(ys) if ys else cy
        hx = abs(faces[0][0] - cx) if 0 in faces else abs(faces[2][0] - cx) if 2 in faces else s / 2
        hy = abs(faces[1][1] - cy) if 1 in faces else abs(faces[3][1] - cy) if 3 in faces else s / 2
        box = [faces[2][0] if 2 in faces else cx - hx, faces[3][1] if 3 in faces else cy - hy,
               faces[0][0] if 0 in faces else cx + hx, faces[1][1] if 1 in faces else cy + hy]
        if args.obstacle_dims:
            # box size known: a side with no tag is the tagged opposite side plus the size
            dx, dy = (v * 0.0254 for v in args.obstacle_dims)
            if 2 not in faces and 0 in faces: box[0] = box[2] - dx
            if 0 not in faces and 2 in faces: box[2] = box[0] + dx
            if 3 not in faces and 1 in faces: box[1] = box[3] - dy
            if 1 not in faces and 3 in faces: box[3] = box[1] + dy
    else:
        # not measured: one tag in the middle of each face, facing out
        obstacle_tags = {8: (cx + s / 2, cy, 0.0), 9: (cx, cy + s / 2, math.pi / 2),
                         10: (cx - s / 2, cy, math.pi), 11: (cx, cy - s / 2, -math.pi / 2)}
    world = {
        'L': L,
        'obstacle': tuple(box),
        'start': (7 * FT, 1 * FT, math.pi / 2),     # facing +y
        'goal': (1 * FT, 7 * FT),
        # tag id: x, y, direction the tag faces (rad). Wall tags 1 ft from each corner, counterclockwise.
        'tags': {0: (8 * FT, 1 * FT, math.pi), 1: (8 * FT, 7 * FT, math.pi),
                 2: (7 * FT, 8 * FT, -math.pi / 2), 3: (1 * FT, 8 * FT, -math.pi / 2),
                 4: (0.0, 7 * FT, 0.0), 5: (0.0, 1 * FT, 0.0),
                 6: (1 * FT, 0.0, math.pi / 2), 7: (7 * FT, 0.0, math.pi / 2), **obstacle_tags},
    }
    return world


# ---- configuration space: the robot becomes a point, the obstacle grows, the walls move in ----

def grow(box, r):
    return (box[0] - r, box[1] - r, box[2] + r, box[3] + r)


def segment_hits_box(p, q, box, eps=1e-9):
    """True if segment p-q passes through the open interior of an axis-aligned box (Liang-Barsky)."""
    x0, y0 = p; dx, dy = q[0] - p[0], q[1] - p[1]
    t0, t1 = 0.0, 1.0
    for pk, qk in ((-dx, x0 - box[0]), (dx, box[2] - x0), (-dy, y0 - box[1]), (dy, box[3] - y0)):
        if abs(pk) < eps:
            if qk <= eps:
                return False        # parallel and outside (or on the edge)
        else:
            t = qk / pk
            if pk < 0:
                t0 = max(t0, t)
            else:
                t1 = min(t1, t)
            if t0 >= t1 - eps:
                return False
    return True


def astar(nodes, neighbors, start, goal, h):
    """A* over a graph: nodes is a list of points, neighbors(i) yields (j, cost)."""
    g = {start: 0.0}
    came = {}
    frontier = [(h(start), start)]
    closed = set()
    while frontier:
        _, i = heapq.heappop(frontier)
        if i == goal:
            path = [i]
            while i in came:
                i = came[i]
                path.append(i)
            return path[::-1], g[goal]
        if i in closed:
            continue
        closed.add(i)
        for j, c in neighbors(i):
            if g[i] + c < g.get(j, math.inf):
                g[j] = g[i] + c
                came[j] = i
                heapq.heappush(frontier, (g[j] + h(j), j))
    return None, math.inf


# ---- shortest: visibility graph ----

def point_box_distance(p, box):
    return math.hypot(max(box[0] - p[0], 0, p[0] - box[2]), max(box[1] - p[1], 0, p[1] - box[3]))


def point_segment_distance(c, p, q):
    dx, dy = q[0] - p[0], q[1] - p[1]
    t = max(0.0, min(1.0, ((c[0] - p[0]) * dx + (c[1] - p[1]) * dy) / max(dx * dx + dy * dy, 1e-12)))
    return math.dist(c, (p[0] + t * dx, p[1] + t * dy))


def segment_box_distance(p, q, box):
    """smallest distance between segment p-q and an axis-aligned box (0 if they cross)"""
    if segment_hits_box(p, q, box):
        return 0.0
    corners = [(box[0], box[1]), (box[2], box[1]), (box[2], box[3]), (box[0], box[3])]
    return min([point_box_distance(p, box), point_box_distance(q, box)]
               + [point_segment_distance(c, p, q) for c in corners])


def cspace_outline(box, r, n=12):
    """the obstacle grown by r: the box's edges moved out by r, joined by quarter circles at the corners"""
    pts = []
    for (cx, cy), a0 in (((box[2], box[1]), -90), ((box[2], box[3]), 0), ((box[0], box[3]), 90), ((box[0], box[1]), 180)):
        for k in range(n + 1):
            a = math.radians(a0 + 90 * k / n)
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return pts + pts[:1]


def shortest_path(world, clearance):
    """Visibility graph around the grown obstacle; A* with straight-line distance.
    The grown obstacle (C-space) has rounded corners. Each corner arc is wrapped in two straight edges
    tangent to it (an octagon around the grown box), and their ends are the graph nodes. A path edge is
    allowed if it stays at least the clearance away from the real box, so it may touch the rounded corner."""
    box = world['obstacle']
    r = clearance
    k = r * math.tan(math.pi / 8)      # octagon vertex offset along the box edge
    corners = []
    for cx, cy, sx, sy in ((box[0], box[1], -1, -1), (box[2], box[1], 1, -1),
                           (box[2], box[3], 1, 1), (box[0], box[3], -1, 1)):
        corners += [(cx + sx * r, cy + sy * k), (cx + sx * k, cy + sy * r)]
    nodes = [world['start'][:2], world['goal']] + corners
    L = world['L']

    def free(p):
        return clearance <= p[0] <= L - clearance and clearance <= p[1] <= L - clearance

    edges = {i: [] for i in range(len(nodes))}
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            if (free(nodes[i]) and free(nodes[j])
                    and segment_box_distance(nodes[i], nodes[j], box) >= clearance - 1e-6):
                d = math.dist(nodes[i], nodes[j])
                edges[i].append((j, d)); edges[j].append((i, d))
    path, length = astar(nodes, lambda i: edges[i], 0, 1, lambda i: math.dist(nodes[i], nodes[1]))
    return [nodes[i] for i in path], length, nodes, edges, cspace_outline(box, r)


# ---- safest: Voronoi graph ----

def clearance_fields(world, res):
    """distance from each grid cell to each object: 4 walls and the obstacle"""
    L = world['L']
    n = int(round(L / res)) + 1
    xs = np.linspace(0, L, n)
    X, Y = np.meshgrid(xs, xs)          # X[row, col]: col is x, row is y
    ox0, oy0, ox1, oy1 = world['obstacle']
    ddx = np.maximum(np.maximum(ox0 - X, X - ox1), 0)
    ddy = np.maximum(np.maximum(oy0 - Y, Y - oy1), 0)
    inside = (X > ox0) & (X < ox1) & (Y > oy0) & (Y < oy1)
    d_obs = np.where(inside, 0.0, np.hypot(ddx, ddy))
    fields = np.stack([X, L - X, Y, L - Y, d_obs])  # wall x=0, x=L, y=0, y=L, obstacle
    return xs, fields


def safest_path(world, clearance, res=0.01):
    """Voronoi graph: cells about equally far from their two nearest objects. A* along it."""
    xs, fields = clearance_fields(world, res)
    order = np.sort(fields, axis=0)
    d1, d2 = order[0], order[1]
    free = d1 >= clearance
    voronoi = free & (d2 - d1 <= 1.5 * res)
    n = len(xs)

    def cell(p):
        return int(round(p[1] / res)), int(round(p[0] / res))     # row (y), col (x)

    def join(p):
        """connect a point to the Voronoi graph by climbing the clearance (the lecture's 'retraction')"""
        r, c = cell(p)
        chain = [(r, c)]
        while not voronoi[r, c]:
            best = max(((r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                        if (dr or dc) and 0 <= r + dr < n and 0 <= c + dc < n), key=lambda rc: d1[rc])
            if d1[best] <= d1[r, c]:
                break
            r, c = best
            chain.append((r, c))
        return chain

    start_chain, goal_chain = join(world['start'][:2]), join(world['goal'])
    allowed = voronoi.copy()
    for r, c in start_chain + goal_chain:
        allowed[r, c] = True
    s, g = start_chain[0], goal_chain[0]

    def neighbors(rc):
        r, c = rc
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = r + dr, c + dc
                if (dr or dc) and 0 <= rr < n and 0 <= cc < n and allowed[rr, cc]:
                    yield (rr, cc), res * math.hypot(dr, dc)

    path, length = astar(None, neighbors, s, g, lambda rc: res * math.hypot(rc[0] - g[0], rc[1] - g[1]))
    pts = [(xs[c], xs[r]) for r, c in path]
    return pts, length, xs, voronoi, d1


def simplify(pts, tol):
    """Douglas-Peucker: fewer waypoints, so the robot makes fewer stops and turns"""
    if len(pts) < 3:
        return list(pts)
    a, b = np.array(pts[0]), np.array(pts[-1])
    ab = b - a
    norm = np.linalg.norm(ab) or 1e-9
    dists = [abs(ab[0] * (p[1] - a[1]) - ab[1] * (p[0] - a[0])) / norm for p in pts[1:-1]]
    k = int(np.argmax(dists)) + 1
    if dists[k - 1] <= tol:
        return [pts[0], pts[-1]]
    return simplify(pts[:k + 1], tol)[:-1] + simplify(pts[k:], tol)


def min_clearance(world, path, step=0.005):
    """smallest distance from the robot center to any wall or the obstacle along the path"""
    L = world['L']; ox0, oy0, ox1, oy1 = world['obstacle']
    best = math.inf
    for p, q in zip(path, path[1:]):
        for t in np.linspace(0, 1, max(2, int(math.dist(p, q) / step))):
            x, y = p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])
            d_obs = math.hypot(max(ox0 - x, 0, x - ox1), max(oy0 - y, 0, y - oy1))
            best = min(best, x, L - x, y, L - y, d_obs)
    return best


def write_path(path, name, start_theta):
    """one line per waypoint: x, y, heading to hold there (the next leg's heading; the last keeps its leg)"""
    lines = [(path[0][0], path[0][1], start_theta)]
    for i in range(1, len(path)):
        nxt = path[i + 1] if i + 1 < len(path) else None
        prv = path[i - 1]
        theta = math.atan2(nxt[1] - path[i][1], nxt[0] - path[i][0]) if nxt else math.atan2(path[i][1] - prv[1], path[i][0] - prv[0])
        lines.append((path[i][0], path[i][1], theta))
    with open(os.path.join(HW4, name), 'w') as f:
        for x, y, th in lines:
            f.write(f'{x:.4f},{y:.4f},{th:.5f}\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--obstacle-tag', type=float, nargs=4, action='append', metavar=('ID', 'X', 'Y', 'FACE'),
                    help='measured obstacle tag: id, x ft, y ft, facing deg')
    ap.add_argument('--obstacle-dims', type=float, nargs=2, metavar=('X', 'Y'),
                    help='inches: box size along x and y, places the sides that have no tag')
    ap.add_argument('--obstacle-size', type=float, default=1.5, help='ft')
    ap.add_argument('--obstacle-center', type=float, nargs=2, default=(4.0, 4.0), help='ft')
    ap.add_argument('--robot-radius', type=float, default=0.15, help='m, half the robot diagonal')
    ap.add_argument('--margin', type=float, default=0.05, help='m of extra room for localization error')
    args = ap.parse_args()
    world = make_world(args)
    clearance = args.robot_radius + args.margin

    short, short_len, vnodes, vedges, grown = shortest_path(world, clearance)
    safe_raw, _, xs, voronoi, d1 = safest_path(world, clearance)
    safe = simplify(safe_raw, 0.03)
    safe_len = sum(math.dist(p, q) for p, q in zip(safe, safe[1:]))

    os.makedirs(os.path.join(HW4, 'results'), exist_ok=True)
    write_path(short, 'path_shortest.txt', world['start'][2])
    write_path(safe, 'path_safest.txt', world['start'][2])
    with open(os.path.join(HW4, 'map.txt'), 'w') as f:
        f.write('# tag id, x (m), y (m), direction the tag faces (rad)\n')
        for k, (x, y, th) in sorted(world['tags'].items()):
            f.write(f'{k},{x:.4f},{y:.4f},{th:.5f}\n')
    with open(os.path.join(HW4, 'obstacle.txt'), 'w') as f:
        f.write('# obstacle box xmin, ymin, xmax, ymax (m); robot radius + margin (m)\n')
        f.write(','.join(f'{v:.4f}' for v in world['obstacle']) + f',{clearance:.4f}\n')

    for name, p, length in (('shortest', short, short_len), ('safest', safe, safe_len)):
        turns = sum(abs(math.degrees(math.atan2(math.sin(a - b), math.cos(a - b))))
                    for a, b in zip([math.atan2(q[1] - p_[1], q[0] - p_[0]) for p_, q in zip(p[1:], p[2:])],
                                    [math.atan2(q[1] - p_[1], q[0] - p_[0]) for p_, q in zip(p, p[1:])]))
        print(f'{name}: {len(p)} waypoints, length {length:.3f} m ({length / FT:.2f} ft), '
              f'min clearance {min_clearance(world, p) * 100:.1f} cm, turning {turns:.0f} deg')
        print('   ' + '  '.join(f'({x / FT:.2f}, {y / FT:.2f})' for x, y in p) + '  ft')

    plot(world, clearance, short, vnodes, vedges, grown, safe, xs, voronoi, d1)


def plot(world, clearance, short, vnodes, vedges, grown, safe, xs, voronoi, d1):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    L = world['L'] / FT
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.9), dpi=300)
    ink, grey, accent = '#1a1a1a', '#9a9a9a', '#2a6fdb'

    def base(ax, title):
        ox0, oy0, ox1, oy1 = (v / FT for v in world['obstacle'])
        c = clearance / FT
        ax.add_patch(Rectangle((c, c), L - 2 * c, L - 2 * c, fill=False, ls=':', lw=0.7, color=grey))
        ax.add_patch(Rectangle((ox0, oy0), ox1 - ox0, oy1 - oy0, color='#cfcfcf', lw=0))
        ax.plot([0, L, L, 0, 0], [0, 0, L, L, 0], color=ink, lw=1.2)
        for k, (x, y, _) in world['tags'].items():
            ax.plot(x / FT, y / FT, 's', ms=3, color=ink)
            ax.annotate(str(k), (x / FT, y / FT), xytext=(3, 3), textcoords='offset points', fontsize=5.5, color='#444')
        sx, sy, _ = world['start']; gx, gy = world['goal']
        ax.plot(sx / FT, sy / FT, 'o', ms=5, color=ink); ax.annotate('start', (sx / FT, sy / FT), xytext=(-6, -10), textcoords='offset points', fontsize=6.5)
        ax.plot(gx / FT, gy / FT, '*', ms=7, color=ink); ax.annotate('goal', (gx / FT, gy / FT), xytext=(4, 4), textcoords='offset points', fontsize=6.5)
        ax.set_aspect('equal'); ax.set_xlim(-0.3, L + 0.3); ax.set_ylim(-0.3, L + 0.3)
        ax.set_xticks(range(0, int(L) + 1)); ax.set_yticks(range(0, int(L) + 1)); ax.tick_params(labelsize=6)
        ax.set_xlabel('x (ft)', fontsize=7); ax.set_ylabel('y (ft)', fontsize=7)
        ax.grid(lw=0.3, color='#ececec'); ax.set_title(title, fontsize=8, loc='left')
        for sp in ax.spines.values(): sp.set_visible(False)

    ax = axes[0]
    base(ax, 'Shortest: visibility graph + A*')
    ax.plot([p[0] / FT for p in grown], [p[1] / FT for p in grown], ls='--', lw=0.7, color=grey)
    for i, js in vedges.items():
        for j, _ in js:
            if i < j:
                ax.plot([vnodes[i][0] / FT, vnodes[j][0] / FT], [vnodes[i][1] / FT, vnodes[j][1] / FT], color=grey, lw=0.5)
    ax.plot([p[0] / FT for p in short], [p[1] / FT for p in short], color=accent, lw=1.8)
    ax.plot([p[0] / FT for p in short], [p[1] / FT for p in short], 'o', ms=2.5, color=accent)

    ax = axes[1]
    base(ax, 'Safest: Voronoi graph + A*')
    r, c = np.nonzero(voronoi)
    ax.scatter(xs[c] / FT, xs[r] / FT, s=0.15, color=grey, lw=0)
    ax.plot([p[0] / FT for p in safe], [p[1] / FT for p in safe], color=accent, lw=1.8)
    ax.plot([p[0] / FT for p in safe], [p[1] / FT for p in safe], 'o', ms=2.5, color=accent)
    fig.text(0.5, 0.012, 'Grey box: obstacle (from the measured tags).  Dashed: obstacle grown by the robot radius + margin (C-space).  Dotted: walls moved in by the same.\n'
             'Grey lines and dots: the graph that was searched.  Blue: planned path.  Small black squares: tags with IDs.', ha='center', fontsize=5.6, color='#555', linespacing=1.4)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    out = os.path.join(HW4, 'results', 'planning.png')
    fig.savefig(out)
    print('plot:', out)


if __name__ == '__main__':
    main()
