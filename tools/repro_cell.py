# -*- coding: utf-8 -*-
"""复现 notebook 指定 cell 的错误（打印完整 traceback）"""
import json
import sys

import nbformat


def main(path, cell_idx):
    nb = nbformat.read(path, as_version=4)
    src = ''.join(nb.cells[cell_idx].source)
    ns = {'__name__': '__main__'}
    try:
        exec(compile(src, f'{path}:cell{cell_idx}', 'exec'), ns)
        print('OK')
    except Exception:
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]))
