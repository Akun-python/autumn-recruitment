# -*- coding: utf-8 -*-
"""逐 cell 在单一命名空间执行 notebook，打印首个失败 cell 的完整 traceback。"""
import sys
import traceback

import nbformat


def run_sequential(path):
    nb = nbformat.read(path, as_version=4)
    ns = {'__name__': '__main__'}
    for i, cell in enumerate(nb.cells):
        src = ''.join(cell.source)
        if cell.cell_type == 'markdown':
            continue
        try:
            exec(compile(src, f'{path}:cell{i}', 'exec'), ns)
        except Exception:
            print(f'FAILED at cell {i}:')
            traceback.print_exc()
            return 1
    print('ALL CELLS OK')
    return 0


if __name__ == '__main__':
    sys.exit(run_sequential(sys.argv[1]))
