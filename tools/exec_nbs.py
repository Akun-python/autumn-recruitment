# -*- coding: utf-8 -*-
"""批量执行指定目录下的 notebook：nbclient 执行 + 回写输出 + 失败统计。"""
import glob
import io
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import nbformat
from nbclient import NotebookClient


def execute_folder(folder, pattern='*.ipynb', timeout=900):
    os.chdir(folder)
    nbs = sorted(glob.glob(pattern))
    nbs = [n for n in nbs if not n.startswith('_')]
    fails = []
    for nb in nbs:
        nbk = nbformat.read(nb, as_version=4)
        client = NotebookClient(nbk, timeout=timeout, kernel_name='python3',
                                resources={'metadata': {'path': '.'}})
        try:
            client.execute()
            nbformat.write(nbk, nb, version=4)
            print(f'## {nb} :: OK ({len(nbk.cells)} cells)')
        except Exception as e:
            tb = getattr(e, 'traceback', None)
            if tb:
                print(f'## {nb} :: EXEC FAILED')
                print('\n'.join(tb)[-3500:])
            else:
                print(f'## {nb} :: EXEC FAILED: {type(e).__name__}: {str(e)[:300]}')
            fails.append(nb)
    print('RESULT:', 'ALL PASS' if not fails else f'FAILED {len(fails)}: {fails}')
    return 0 if not fails else 1


if __name__ == '__main__':
    folder = sys.argv[1]
    sys.exit(execute_folder(folder, sys.argv[2] if len(sys.argv) > 2 else '*.ipynb'))