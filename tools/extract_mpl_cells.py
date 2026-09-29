# -*- coding: utf-8 -*-
"""Extract matplotlib figure-generating code cells from notebooks.
Usage: python tools/extract_mpl_cells.py <notebook.ipynb> [outdir]
Saves each cell that contains 'savefig' or 'plt.savefig' or a known figure
output name into outdir as cell_<NN>_<name>.py, and prints an index.
"""
import json, re, os, sys

def main():
    nb_path = sys.argv[1]
    outdir = sys.argv[2] if len(sys.argv) > 2 else 'tools/fig_cells'
    nb = json.load(open(nb_path, encoding='utf-8'))
    os.makedirs(outdir, exist_ok=True)
    index = []
    for i, c in enumerate(nb['cells']):
        if c.get('cell_type') != 'code':
            continue
        src = ''.join(c.get('source', []))
        if 'savefig' not in src and '.png' not in src:
            continue
        # guess a name from the save path
        m = re.findall(r'savefig\([^)]*?([\w\-]+\.png)', src) or \
            re.findall(r'([\w\-]+\.png)', src)
        name = m[0] if m else f'cell{i}'
        fn = os.path.join(outdir, f'cell_{i:03d}_{name.replace(".png","")}.py')
        with open(fn, 'w', encoding='utf-8') as f:
            f.write(f'# source: {os.path.basename(nb_path)} cell {i}\n')
            f.write(src)
            if not src.endswith('\n'):
                f.write('\n')
        index.append((i, name, fn))
    for i, name, fn in index:
        print(f'cell {i:3d} -> {name:40s} {fn}')

if __name__ == '__main__':
    main()