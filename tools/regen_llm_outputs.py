# -*- coding: utf-8 -*-
"""Re-execute LLM notebooks and write outputs back into the .ipynb files.

For notebooks whose plot cells were patched, the embedded image/png outputs
must be regenerated so the notebook matches its source.  We execute the whole
notebook in-process (Agg backend), capturing the new figure from the patched
cell and replacing that cell's image outputs.

Usage: python tools/regen_llm_outputs.py <nb1> [nb2 ...]
"""
import json, io, sys, os, base64

TOOLS = os.path.abspath(os.path.dirname(__file__))
TEACH = os.path.abspath(os.path.join(TOOLS, os.pardir, '04-LLM大模型', '教学'))


def sanitize(src):
    out = []
    for ln in src.splitlines():
        s = ln.strip()
        if s.startswith('%') or s.startswith('%%'):
            continue
        out.append(ln)
    return '\n'.join(out)


def render_figure(fig, dpi=100):
    """Render a figure to a base64 PNG string (matplotlib version agnostic)."""
    from io import BytesIO
    import matplotlib.pyplot as plt
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
    buf.seek(0)
    data = base64.b64encode(buf.read()).decode('ascii')
    buf.close()
    return data


def exec_cell(ns, code, tag):
    try:
        exec(compile(code, tag, 'exec'), ns)
        return None
    except Exception as e:
        return f'{type(e).__name__}: {e}'


def main():
    nbs = sys.argv[1:]
    for nb in nbs:
        if not os.path.isabs(nb):
            nb = os.path.join(TEACH, nb)
        if not os.path.exists(nb):
            print(f'MISSING {nb}')
            continue
        data = json.load(open(nb, encoding='utf-8'))
        ns = {'__name__': '__main__'}
        exec('import matplotlib; matplotlib.use("Agg")', ns)
        exec('import matplotlib.pyplot as plt', ns)
        exec('import numpy as np', ns)
        plt = ns['plt']

        # find which cells produce images so we can re-map outputs by count
        tag = os.path.basename(nb)
        failures = []
        for i, c in enumerate(data['cells']):
            if c.get('cell_type') != 'code':
                continue
            src = ''.join(c.get('source', ''))
            code = sanitize(src)
            if not code.strip():
                continue
            before = set(plt.get_fignums())
            err = exec_cell(ns, code, f'{tag}:cell{i}')
            if err:
                failures.append((i, err))
                continue
            after = set(plt.get_fignums())
            newnums = sorted(after - before)
            if newnums:
                # this cell created figure(s): store PNG outputs
                outputs = []
                for fn in newnums:
                    fig = plt.figure(fn)
                    outputs.append({
                        'output_type': 'display_data',
                        'data': {'image/png': render_figure(fig)},
                        'metadata': {},
                    })
                c['outputs'] = outputs
                c['execution_count'] = i
            else:
                c['execution_count'] = i
        # drop figures to keep memory sane
        try:
            plt.close('all')
        except Exception:
            pass
        with open(nb, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
        fails = ', '.join(f'c{i}({e})' for i, e in failures[:5])
        print(f'## {tag} :: regenerated, failed: {len(failures)} {fails}')


if __name__ == '__main__':
    main()