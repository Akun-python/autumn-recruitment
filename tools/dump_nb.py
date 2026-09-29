"""Dump full notebook content to a text file."""
import json
import sys


def dump(path, out_path, max_len=1200):
    nb = json.load(open(path, encoding="utf-8"))
    lines = []
    for i, c in enumerate(nb["cells"]):
        src = "".join(c.get("source", []))
        lines.append(f"---- [{i:02d}] {c['cell_type']} len={len(src)} ----")
        lines.append(src[:max_len])
        if len(src) > max_len:
            lines.append(f"... (truncated, total {len(src)})")
        lines.append("")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(out_path)


if __name__ == "__main__":
    dump(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 1200)