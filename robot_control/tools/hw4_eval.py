#!/usr/bin/env python3
"""HW4: plot a path run (estimated trajectory over the planned path) and report its numbers.

Usage: python3 hw4_eval.py shortest|safest [stamp]     (latest run if no stamp)
Reads ~/HW4/logs/{pose,waypoints}_<path>_<stamp>.csv, ~/HW4/path_<path>.txt, ~/HW4/obstacle.txt, ~/HW4/map.txt.
Writes ~/HW4/results/run_<path>_<stamp>.png.
"""
import csv, glob, math, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

FT = 0.3048
HW4 = os.path.expanduser('~/HW4')


def rows(path):
    return [[float(v) for v in l.split(',')] for l in open(path) if l.strip() and not l.startswith('#')]


def clearance_to(box, L, x, y):
    ox0, oy0, ox1, oy1 = box
    d_obs = math.hypot(max(ox0 - x, 0, x - ox1), max(oy0 - y, 0, y - oy1))
    return min(d_obs, x, L - x, y, L - y)


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else 'shortest'
    stamp = sys.argv[2] if len(sys.argv) > 2 else sorted(
        p.split(f'pose_{name}_')[1][:-4] for p in glob.glob(os.path.join(HW4, 'logs', f'pose_{name}_*.csv')))[-1]
    pose = list(csv.DictReader(open(os.path.join(HW4, 'logs', f'pose_{name}_{stamp}.csv'))))
    wps = list(csv.DictReader(open(os.path.join(HW4, 'logs', f'waypoints_{name}_{stamp}.csv'))))
    plan = rows(os.path.join(HW4, f'path_{name}.txt'))
    *box, clear = rows(os.path.join(HW4, 'obstacle.txt'))[0]
    tags = {int(r[0]): r[1:3] for r in rows(os.path.join(HW4, 'map.txt'))}
    L = 8 * FT

    xs, ys = [float(p['x']) for p in pose], [float(p['y']) for p in pose]
    driven = sum(math.dist(a, b) for a, b in zip(zip(xs, ys), zip(xs[1:], ys[1:])))
    planned = sum(math.dist(a[:2], b[:2]) for a, b in zip(plan, plan[1:]))
    t = float(pose[-1]['t']) - float(pose[0]['t'])
    closest = min(clearance_to(box, L, x, y) for x, y in zip(xs, ys))
    goal_err = math.dist(plan[-1][:2], (xs[-1], ys[-1]))
    print(f'{name} run {stamp}: time {t:.1f} s, driven {driven:.2f} m ({driven / FT:.2f} ft), '
          f'planned {planned:.2f} m ({planned / FT:.2f} ft)')
    print(f'closest estimated approach to the obstacle or a wall: {closest * 100:.1f} cm (robot center)')
    print(f'end {goal_err * 100:.1f} cm from the goal (estimate)')
    for w in wps:
        print(f"  waypoint {w['i']}: stopped {float(w['err_m']) * 100:.1f} cm from it (estimate)")

    fig, ax = plt.subplots(figsize=(4.2, 4.4), dpi=250)
    ax.add_patch(Rectangle((0, 0), L / FT, L / FT, fill=False, lw=1.2, color='#222'))
    ax.add_patch(Rectangle((box[0] / FT, box[1] / FT), (box[2] - box[0]) / FT, (box[3] - box[1]) / FT,
                           color='#d0d0d0', lw=0))
    from hw4_planner import cspace_outline
    g = cspace_outline(box, clear)
    ax.plot([p[0] / FT for p in g], [p[1] / FT for p in g], ls='--', lw=0.6, color='#999')
    for k, (x, y) in tags.items():
        ax.plot(x / FT, y / FT, 's', ms=3.5, color='#222')
        ax.annotate(str(k), (x / FT, y / FT), xytext=(3, 3), textcoords='offset points', fontsize=5.5, color='#444')
    px, py = [p[0] / FT for p in plan], [p[1] / FT for p in plan]
    ax.plot(px, py, '-o', color='#9a9a9a', lw=1.2, ms=3, label='Planned path')
    ax.plot([x / FT for x in xs], [y / FT for y in ys], '.', ms=2.2, color='#2a6fdb',
            label='Estimated robot position, every 0.5 s')
    ax.plot(px[0], py[0], 'o', ms=5, mfc='none', mec='#222'); ax.annotate('start', (px[0], py[0]), xytext=(-6, -10), textcoords='offset points', fontsize=6.5)
    ax.plot(px[-1], py[-1], '*', ms=7, color='#222'); ax.annotate('goal', (px[-1], py[-1]), xytext=(4, 4), textcoords='offset points', fontsize=6.5)
    ax.set_xlim(-0.3, 8.3); ax.set_ylim(-0.3, 8.3); ax.set_aspect('equal')
    ax.set_xticks(range(9)); ax.set_yticks(range(9)); ax.tick_params(labelsize=6)
    ax.set_xlabel('x (ft)', fontsize=7); ax.set_ylabel('y (ft)', fontsize=7)
    ax.grid(lw=0.3, color='#ececec')
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(f'{name.capitalize()} path: planned and driven ({t:.0f} s, {driven / FT:.1f} ft)', fontsize=8, loc='left')
    ax.legend(fontsize=6, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False)
    fig.tight_layout()
    out = os.path.join(HW4, 'results', f'run_{name}_{stamp}.png')
    fig.savefig(out)
    print(f'plot: {out}')


if __name__ == '__main__':
    main()
