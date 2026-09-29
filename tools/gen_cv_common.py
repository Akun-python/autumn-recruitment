# -*- coding: utf-8 -*-
"""09-计算机视觉 教学 notebook 生成器的公共工具。

被 tools/gen_cv_yolo.py / gen_cv_unet.py / gen_cv_sam.py 复用：
md/code 建 cell，build 组装，save 写入 09-计算机视觉/教学/。
"""
import os
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   '09-计算机视觉', '教学')


def md(src):
    return new_markdown_cell(src)


def code(src):
    return new_code_cell(src)


def build(cells):
    nbk = new_notebook(cells=cells)
    nbk.metadata['kernelspec'] = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
    nbk.metadata['language_info'] = {'name': 'python', 'version': '3.10.0'}
    return nbk


def save(nbk, fname):
    path = os.path.join(OUT, fname)
    with open(path, 'w', encoding='utf-8') as f:
        nbformat.write(nbk, f)
    print('written:', path)
