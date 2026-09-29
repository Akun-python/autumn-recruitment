# -*- coding: utf-8 -*-
"""Regenerate every RL teaching figure from its notebook cell and check overlaps.
Run: python tools/check_rl_figs.py
Each cell is executed in a fresh subprocess with cwd = 教学 folder (so the
`out` fallback in the cell resolves to 教学/images and relative heatmap paths
resolve to the teaching folder). Before the cell closes the figure we append
code that runs the overlap checker on the live figure.
"""
import glob, os, re, subprocess, sys, io

TOOLS = os.path.abspath(os.path.join(os.path.dirname(__file__)))
ROOT = os.path.abspath(os.path.join(TOOLS, os.pardir))
TEACH = os.path.join(ROOT, '07-强化学习', '教学')

CELLS = sorted(glob.glob(os.path.join(TOOLS, 'fig_cells', '*.py')))

def build_wrapper(cell_path):
    src = open(cell_path, encoding='utf-8').read()
    name = os.path.basename(cell_path).replace('cell_', '').replace('.py', '')
    # strip trailing plt.close / plt.show so the figure stays alive
    lines = src.splitlines()
    out_lines = []
    for ln in lines:
        s = ln.strip()
        if re.match(r'^(plt\.close|plt\.show)\b', s):
            continue
        out_lines.append(ln)
    body = '\n'.join(out_lines)
    wrapper = f"""# -*- coding: utf-8 -*-
import sys, os
sys.path.insert(0, {TOOLS!r})
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
_SAVED_FIGS = []
_orig_close = plt.close
def _capture_close(*a, **k):
    try:
        f = a[0] if a else plt.gcf()
        if f is not None and not isinstance(f, (int, str)):
            _SAVED_FIGS.append(f)
    except Exception:
        pass
    try:
        _orig_close(*a, **k)
    except Exception:
        pass
plt.close = _capture_close
import builtins
_orig_show = plt.show
plt.show = lambda *a, **k: None
# ---- cell source ----
{body}
# ---- overlap check ----
try:
    fig = _SAVED_FIGS[-1] if _SAVED_FIGS else plt.gcf()
except Exception:
    fig = None
if fig is not None:
    from overlap_check import report
    report(fig, {name!r})
else:
    print('[NO_FIG]')
"""
    return wrapper

def main():
    results = io.StringIO()
    for cell in CELLS:
        wrapper = build_wrapper(cell)
        wp = os.path.join(TOOLS, '_tmp_cell_run.py')
        with open(wp, 'w', encoding='utf-8') as f:
            f.write(wrapper)
        env = dict(os.environ)
        env['PYTHONIOENCODING'] = 'utf-8'
        r = subprocess.run([sys.executable, wp], cwd=TEACH, env=env,
                           capture_output=True, text=True, encoding='utf-8',
                           errors='replace', timeout=300)
        tag = os.path.basename(cell)
        out = (r.stdout or '') + (r.stderr or '')
        if r.returncode != 0:
            results.write(f'== {tag} :: CRASH ({r.returncode})\n')
        # collect the last [name] ... line
        last = ''
        for line in out.splitlines():
            if '[' in line and ('overlap' in line or 'OK' in line or 'NO_FIG' in line):
                last = line.strip()
        if not last:
            last = 'NO_REPORT'
        results.write(f'== {tag} :: {last}\n')
        print(f'{tag}: {last}')
        os.remove(wp)
    open(os.path.join(TOOLS, 'rl_results.txt'), 'w', encoding='utf-8').write(results.getvalue())

if __name__ == '__main__':
    main()
