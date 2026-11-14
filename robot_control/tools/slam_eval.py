#!/usr/bin/env python3
"""HW3: compare a SLAM run's map with the measured ground truth, and plot it.

Usage: python3 slam_eval.py square|octagon [stamp] [--start-frame]
--start-frame plots the map in the robot's start frame, without ground truth.
--feet labels the plot axes in feet (the logs stay in meters).
--lap N plots the map saved at the end of lap N, with the path driven up to then.
Reads ~/HW3/logs/{pose,map}_<shape>_<stamp>.csv (latest run if no stamp) and ~/HW3/ground_truth.txt.
The estimate lives in the robot's start frame; it is moved into the world frame with the
measured start pose. A best-fit rigid alignment is also reported, which separates the map's
shape error from the error in measuring the start pose.
"""
import csv, glob, math, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.ticker import FuncFormatter, MultipleLocator

HW3 = os.path.expanduser('~/HW3')
FOOT = 0.3048
CHI2_95 = 5.991     # 2 degrees of freedom: the ellipse holds the landmark with 95% probability


def load_truth(shape):
    tags, start = {}, None
    for line in open(os.path.join(HW3, 'ground_truth.txt')):
        f = [v.strip() for v in line.split('#')[0].split(',')]
        if f[0] == 'tag':
            tags[int(f[1])] = (float(f[2]), float(f[3]))
        elif f[0] == 'start' and f[1] == shape:
            start = 'fit' if f[2] == 'fit' else (float(f[2]), float(f[3]), math.radians(float(f[4])))
    return tags, start


def to_world(start, x, y):
    c, s = math.cos(start[2]), math.sin(start[2])
    return start[0] + c * x - s * y, start[1] + s * x + c * y


def best_fit(est, truth):
    """rigid transform (rotation + translation) that best maps est points onto truth points"""
    A, B = np.array(est), np.array(truth)
    ca, cb = A.mean(0), B.mean(0)
    U, _, Vt = np.linalg.svd((A - ca).T @ (B - cb))
    R = (U @ Vt).T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = (U @ Vt).T
    return R, cb - R @ ca


def main():
    start_frame = '--start-frame' in sys.argv
    feet = '--feet' in sys.argv
    lap_plot = None
    if '--lap' in sys.argv:
        lap_plot = int(sys.argv[sys.argv.index('--lap') + 1])
        del sys.argv[sys.argv.index('--lap'):sys.argv.index('--lap') + 2]
    argv = [a for a in sys.argv if a not in ('--start-frame', '--feet')]
    shape = argv[1] if len(argv) > 1 else 'square'
    stamp = argv[2] if len(argv) > 2 else sorted(
        p.split(f'map_{shape}_')[1][:-4] for p in glob.glob(os.path.join(HW3, 'logs', f'map_{shape}_*.csv')))[-1]
    tags, start = ({}, (0.0, 0.0, 0.0)) if start_frame else load_truth(shape)
    pose = list(csv.DictReader(open(os.path.join(HW3, 'logs', f'pose_{shape}_{stamp}.csv'))))
    maps = list(csv.DictReader(open(os.path.join(HW3, 'logs', f'map_{shape}_{stamp}.csv'))))
    final = [m for m in maps if m['lap'] == '-1']
    fitted = start == 'fit'
    if fitted:
        # start pose not measured: the rigid transform that puts the final map onto the measured tags
        common = [m for m in final if int(m['id']) in tags]
        R, t = best_fit([(float(m['x']), float(m['y'])) for m in common], [tags[int(m['id'])] for m in common])
        start = (t[0], t[1], math.atan2(R[1, 0], R[0, 0]))
    shown = final if lap_plot is None else [m for m in maps if int(m['lap']) == lap_plot]
    print(f'{shape} run {stamp}, start pose {start[0]:.3f}, {start[1]:.3f}, {math.degrees(start[2]):.1f} deg'
          + (' (estimated by fitting the final map; not measured)' if fitted else ''))
    print(f'landmarks found: {len(final)} ({", ".join("tag_" + m["id"] for m in final)})')

    # error per lap, with the measured start pose
    print('\nlap  landmarks  mean error (cm)  max (cm)  mean after best fit (cm)'
          + ('   [error uses the start pose fitted on the final map]' if fitted else ''))
    for lap in sorted({int(m['lap']) for m in maps}, key=lambda v: (v < 0, v)):
        rows = [m for m in maps if int(m['lap']) == lap and int(m['id']) in tags]
        if not rows:
            continue
        est = [to_world(start, float(m['x']), float(m['y'])) for m in rows]
        tru = [tags[int(m['id'])] for m in rows]
        err = [math.dist(e, t) for e, t in zip(est, tru)]
        fit = ''
        if len(rows) >= 3:
            R, t = best_fit(est, tru)
            fit = f'{np.mean([math.dist(R @ e + t, g) for e, g in zip(est, tru)]) * 100:.1f}'
        name = 'final' if lap == -1 else ('start' if lap == 0 else str(lap))
        print(f'{name:5}{len(rows):^11}{np.mean(err) * 100:^17.1f}{max(err) * 100:^10.1f}{fit:^26}')

    print('\nper landmark (final map, world frame)')
    for m in final:
        k = int(m['id'])
        wx, wy = to_world(start, float(m['x']), float(m['y']))
        sd = math.sqrt(max(np.linalg.eigvalsh([[float(m['sxx']), float(m['sxy'])], [float(m['sxy']), float(m['syy'])]])))
        gt = tags.get(k)
        e = f'{math.dist((wx, wy), gt) * 100:5.1f} cm' if gt else 'not in ground truth'
        print(f'  tag_{k}: ({wx:.3f}, {wy:.3f}) m, largest std {sd * 100:.1f} cm, error {e}')

    # plot in the world frame
    fig, ax = plt.subplots(figsize=(6.2, 6.2), dpi=200)
    driven = pose if lap_plot is None else [p for p in pose if int(p.get('lap') or 0) <= lap_plot]
    xs, ys = zip(*[to_world(start, float(p['x']), float(p['y'])) for p in driven])
    ax.scatter(xs, ys, s=5, color='#2a6fdb', label='Estimated robot pose, every 0.5 s', zorder=3)
    if tags:
        ax.scatter(*zip(*tags.values()), marker='s', s=40, facecolors='none', edgecolors='#6b6b6b', lw=1,
               label='Measured landmark (ground truth)', zorder=2)
    c, s = math.cos(start[2]), math.sin(start[2])
    Rw = np.array([[c, -s], [s, c]])
    for m in shown:
        wx, wy = to_world(start, float(m['x']), float(m['y']))
        P = Rw @ np.array([[float(m['sxx']), float(m['sxy'])], [float(m['sxy']), float(m['syy'])]]) @ Rw.T
        vals, vecs = np.linalg.eigh(P)
        ang = math.degrees(math.atan2(vecs[1, 1], vecs[0, 1]))
        w, h = 2 * np.sqrt(CHI2_95 * vals[::-1])
        ax.add_patch(Ellipse((wx, wy), w, h, angle=ang, fill=False, color='#1a1a1a', lw=0.8))
        ax.plot(wx, wy, '+', color='#1a1a1a', ms=6, mew=1)
        ax.annotate(f'tag_{m["id"]}', (wx, wy), xytext=(6, 4), textcoords='offset points', fontsize=7, color='#333')
    ax.plot([], [], '+', color='#1a1a1a', label='Estimated landmark, 95% ellipse')
    ax.plot(start[0], start[1], 'o', ms=6, mfc='none', mec='#2a6fdb', label='Start pose' + ('' if start_frame else (' (estimated)' if fitted else ' (measured)')))
    if tags:
        side = max(max(v) for v in tags.values())
        ax.set_xlim(-0.15, side + 0.15); ax.set_ylim(-0.15, side + 0.15)
    ax.set_aspect('equal')
    frame = 'Start frame' if start_frame else 'World'
    unit = 'ft' if feet else 'm'
    if feet:
        # same data in meters, ticks every foot and labeled in feet
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_locator(MultipleLocator(FOOT))
            axis.set_major_formatter(FuncFormatter(lambda v, _: f'{v / FOOT:.0f}'))
    ax.grid(lw=0.3, color='#ddd'); ax.set_xlabel(f'{frame} x ({unit})'); ax.set_ylabel(f'{frame} y ({unit})')
    for sp in ax.spines.values():
        sp.set_color('#999'); sp.set_linewidth(0.5)
    when = 'final map' if lap_plot is None else f'map after lap {lap_plot}'
    ax.set_title(f'EKF-SLAM, {shape} path: estimated trajectory and {when}', fontsize=10, loc='left')
    ax.legend(fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.09), ncol=2, frameon=False)
    fig.tight_layout()
    out = os.path.join(HW3, 'results', f'slam_{shape}_{stamp}{"_startframe" if start_frame else ""}{"_ft" if feet else ""}{"" if lap_plot is None else f"_lap{lap_plot}"}.png')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out)
    print(f'\nplot: {out}')


if __name__ == '__main__':
    main()
