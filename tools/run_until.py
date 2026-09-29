# -*- coding: utf-8 -*-
"""逐 cell 在单一命名空间执行 notebook 到指定 cell（调试用）。"""
import sys
import traceback

import nbformat


def run_until(path, end_idx):
    nb = nbformat.read(path, as_version=4)
    ns = {'__name__': '__main__'}
    for i, cell in enumerate(nb.cells):
        if i > end_idx:
            break
        if cell.cell_type == 'markdown':
            continue
        try:
            exec(compile(''.join(cell.source), f'{path}:cell{i}', 'exec'), ns)
            print(f'cell {i} OK')
        except Exception:
            print(f'FAILED at cell {i}:')
            traceback.print_exc()
            return 1
    print('done')
    return 0


if __name__ == '__main__':
    sys.exit(run_until(sys.argv[1], int(sys.argv[2])))
