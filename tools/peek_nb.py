"""Peek notebook cell outline: type + first line of each cell."""
import json
import glob
import os
import sys


def peek(path, max_cells=None):
    nb = json.load(open(path, encoding="utf-8"))
    print("=== ", os.path.basename(path))
    for i, c in enumerate(nb["cells"]):
        if max_cells and i >= max_cells:
            break
        src = "".join(c.get("source", []))
        first = src.strip().split("\n")[0][:90] if src.strip() else "(empty)"
        print(f"  [{i:02d}] {c['cell_type']:8s} | {first}")
    print()


if __name__ == "__main__":
    pattern = sys.argv[1]
    for f in sorted(glob.glob(pattern))[: int(sys.argv[2]) if len(sys.argv) > 2 else 3]:
        peek(f)
