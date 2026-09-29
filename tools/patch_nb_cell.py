# -*- coding: utf-8 -*-
"""Patch a notebook cell's source by literal replacement, saving UTF-8 JSON."""
import json, sys

def main():
    nb_path, cell_idx, old, new = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
    data = json.load(open(nb_path, encoding='utf-8'))
    src = ''.join(data['cells'][cell_idx].get('source', []))
    if old not in src:
        print(f'OLD NOT FOUND in cell {cell_idx}')
        sys.exit(1)
    new_src = src.replace(old, new)
    data['cells'][cell_idx]['source'] = [new_src]
    data['cells'][cell_idx]['outputs'] = []
    data['cells'][cell_idx]['execution_count'] = None
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    print('patched ok')

if __name__ == '__main__':
    main()
