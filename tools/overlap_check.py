# -*- coding: utf-8 -*-
"""Overlap checker for matplotlib figures.

Renders a figure and reports real collisions in display space:
  * text vs text   : intersecting glyph boxes
  * text vs patch  : glyph box pokes out of a foreign box / crosses a line
  * box vs box     : two filled patches intersect
  * arrow vs box   : the actual arrow segment crosses a box
  * arrow vs arrow : the two segments actually cross

Text intentionally inside its own hosting box is allowed.
"""
import matplotlib.pyplot as plt
from matplotlib.text import Text
from matplotlib.patches import (Rectangle, FancyBboxPatch, Circle, Ellipse,
                                FancyArrowPatch, FancyArrow, Polygon, Wedge,
                                PathPatch)

_FILLED = (Rectangle, FancyBboxPatch, Circle, Ellipse, Polygon, Wedge, PathPatch)


def _inter1(a0, a1, b0, b1):
    """1-D interval overlap."""
    return min(a1, b1) - max(a0, b0)


def _bbox_overlap(b1, b2, tol):
    return (_inter1(b1.x0, b1.x1, b2.x0, b2.x1) > tol and
            _inter1(b1.y0, b1.y1, b2.y0, b2.y1) > tol)


def _seg_seg_intersect(p1, p2, p3, p4):
    """Proper/endpoint intersection of segments p1p2 and p3p4 (display coords)."""
    def ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) - (b[1] - a[1]) * (c[0] - a[0])
    d1 = ccw(p3, p4, p1); d2 = ccw(p3, p4, p2)
    d3 = ccw(p1, p2, p3); d4 = ccw(p1, p2, p4)
    if ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0)):
        return True
    eps = 1e-9
    def on(p, a, b):
        return (min(a[0], b[0]) - eps <= p[0] <= max(a[0], b[0]) + eps and
                min(a[1], b[1]) - eps <= p[1] <= max(a[1], b[1]) + eps)
    if abs(d1) < eps and on(p1, p3, p4): return True
    if abs(d2) < eps and on(p2, p3, p4): return True
    if abs(d3) < eps and on(p3, p1, p2): return True
    if abs(d4) < eps and on(p4, p1, p2): return True
    return False


def _arrow_pair_overlaps(s1, s2):
    """Arrow shafts overlap only on a real crossing.

    Arrows that merely share an endpoint (a fan from a common origin, or a
    head-to-tail chain) are legitimate by design, so a shared endpoint alone
    does not count as overlap.
    """
    a1, a2 = s1
    b1, b2 = s2
    segs = [a1, a2, b1, b2]
    shared = 0
    for p, q in ((a1, b1), (a1, b2), (a2, b1), (a2, b2)):
        if (abs(p[0] - q[0]) < 1.0 and abs(p[1] - q[1]) < 1.0):
            shared += 1
    if shared:
        # shared endpoint: only a genuine interior crossing counts
        def ccw(x, y, z):
            return (z[1] - y[1]) * (x[0] - y[0]) - (x[1] - y[1]) * (z[0] - y[0])
        d1 = ccw(a1, b1, b2); d2 = ccw(a2, b1, b2)
        d3 = ccw(b1, a1, a2); d4 = ccw(b2, a1, a2)
        return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))
    return _seg_seg_intersect(a1, a2, b1, b2)


def _seg_bbox_intersect(p1, p2, bb, tol=2.0):
    """Segment crosses the (expanded) bbox."""
    left = (bb.x0 - tol, bb.y0), (bb.x0 - tol, bb.y1)
    right = (bb.x1 + tol, bb.y0), (bb.x1 + tol, bb.y1)
    bottom = (bb.x0, bb.y0 - tol), (bb.x1, bb.y0 - tol)
    top = (bb.x0, bb.y1 + tol), (bb.x1, bb.y1 + tol)
    for a, b in (left, right, bottom, top):
        if _seg_seg_intersect(p1, p2, a, b):
            return True
    # segment fully inside box
    if (bb.x0 <= p1[0] <= bb.x1 and bb.y0 <= p1[1] <= bb.y1 and
            bb.x0 <= p2[0] <= bb.x1 and bb.y0 <= p2[1] <= bb.y1):
        return True
    return False


def _arrow_seg(a):
    """Return the arrow's shaft as (p1, p2) in display coords, or None."""
    try:
        # FancyArrow (created by ax.arrow) is a Polygon subclass holding
        # data-coordinate offsets in _x/_y/_dx/_dy. Use its data transform.
        if isinstance(a, FancyArrow):
            ax = a.axes
            tr = ax.transData
            p1 = tr.transform((a._x, a._y))
            p2 = tr.transform((a._x + a._dx, a._y + a._dy))
            return tuple(p1), tuple(p2)
        p1 = a.get_posA(); p2 = a.get_posB()
        if a.get_transform() is not None:
            tr = a.get_transform()
            p1 = tr.transform(p1); p2 = tr.transform(p2)
        return tuple(p1), tuple(p2)
    except Exception:
        return None


def check_figure(fig, tol_px=2.0, verbose=False):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    issues = []
    inv_cache = {}

    def data_bb(ax, a):
        """Window bbox -> (x0, y0, x1, y1) in data coords of ax (approx)."""
        b = a.get_window_extent(renderer)
        if ax not in inv_cache:
            inv_cache[ax] = ax.transData.inverted()
        inv = inv_cache[ax]
        (x0, y0) = inv.transform((b.x0, b.y0))
        (x1, y1) = inv.transform((b.x1, b.y1))
        return (x0, y0, x1, y1)

    for ax in fig.axes:
        texts = [t for t in ax.texts if t.get_text().strip()]
        patches = [p for p in ax.patches
                   if p.get_visible() and p.get_alpha() != 0.0]
        arrows = [p for p in patches if isinstance(p, (FancyArrowPatch, FancyArrow))]
        filled = [p for p in patches
                  if isinstance(p, _FILLED) and
                  not isinstance(p, FancyArrow) and
                  (not hasattr(p, 'get_fill') or p.get_fill())]

        def label(a):
            if isinstance(a, Text):
                return 'T:' + (a.get_text()[:18].replace('\n', '\\n'))
            return 'P:' + a.__class__.__name__

        def bb_of(a):
            return a.get_window_extent(renderer)

        if verbose:
            inv = ax.transData.inverted()
            def dbg(a, tag):
                try:
                    b = bb_of(a)
                    (x0, y0) = inv.transform((b.x0, b.y0))
                    (x1, y1) = inv.transform((b.x1, b.y1))
                    print(f'      {tag:4s} {label(a)!r:30s} data=({x0:.2f},{y0:.2f})-({x1:.2f},{y1:.2f})')
                except Exception:
                    pass
            print(f'    -- axes {ax!r}:')
            for t in texts: dbg(t, 'T')
            for p in patches: dbg(p, 'P')

        # ---- text vs text
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                if _bbox_overlap(bb_of(texts[i]), bb_of(texts[j]), tol_px):
                    issues.append(('text-text', label(texts[i]), label(texts[j]),
                                   data_bb(ax, texts[i]), data_bb(ax, texts[j])))

        # ---- text vs filled patch (text must be inside its host or clear)
        for t in texts:
            tb = bb_of(t)
            t0 = (tb.x0, tb.y0); t1 = (tb.x1, tb.y1)
            for p in filled:
                pb = bb_of(p)
                if not _bbox_overlap(tb, pb, tol_px):
                    continue
                inside = (tb.x0 >= pb.x0 - 2 and tb.x1 <= pb.x1 + 2 and
                          tb.y0 >= pb.y0 - 2 and tb.y1 <= pb.y1 + 2)
                if not inside:
                    issues.append(('text-patch', label(t), label(p),
                                   data_bb(ax, t), data_bb(ax, p)))
            # text vs arrow: flag only if the segment crosses the glyph box
            for a in arrows:
                seg = _arrow_seg(a)
                if seg and _seg_bbox_intersect(*seg, tb, tol=2.0):
                    issues.append(('text-arrow', label(t), label(a),
                                   data_bb(ax, t), data_bb(ax, a)))

        # ---- filled vs filled (allow full containment; flag partial overlaps)
        for i in range(len(filled)):
            for j in range(i + 1, len(filled)):
                b1, b2 = bb_of(filled[i]), bb_of(filled[j])
                if not _bbox_overlap(b1, b2, tol_px):
                    continue
                # full containment is fine (e.g. token circle inside bucket box)
                if (b1.x0 >= b2.x0 - tol_px and b1.x1 <= b2.x1 + tol_px and
                        b1.y0 >= b2.y0 - tol_px and b1.y1 <= b2.y1 + tol_px):
                    continue
                if (b2.x0 >= b1.x0 - tol_px and b2.x1 <= b1.x1 + tol_px and
                        b2.y0 >= b1.y0 - tol_px and b2.y1 <= b1.y1 + tol_px):
                    continue
                issues.append(('patch-patch', label(filled[i]), label(filled[j]),
                               data_bb(ax, filled[i]), data_bb(ax, filled[j])))

        # ---- arrow vs filled (real segment crossing)
        for a in arrows:
            seg = _arrow_seg(a)
            if not seg:
                continue
            p1, p2 = seg
            for p in filled:
                if _seg_bbox_intersect(p1, p2, bb_of(p), tol=1.0):
                    issues.append(('arrow-patch', label(a), label(p),
                                   data_bb(ax, a), data_bb(ax, p)))

        # ---- arrow vs arrow (real crossing)
        for i in range(len(arrows)):
            s1 = _arrow_seg(arrows[i])
            if not s1:
                continue
            for j in range(i + 1, len(arrows)):
                s2 = _arrow_seg(arrows[j])
                if s2 and _arrow_pair_overlaps(s1, s2):
                    issues.append(('arrow-arrow', label(arrows[i]), label(arrows[j]),
                                   data_bb(ax, arrows[i]), data_bb(ax, arrows[j])))

    return issues


def report(fig, name, verbose=False):
    issues = check_figure(fig, verbose=verbose)
    if issues:
        print(f'[{name}] {len(issues)} overlap issue(s):')
        for issue in issues[:60]:
            kind = issue[0]
            a, b = issue[1], issue[2]
            line = f'    {kind:10s} {a!r}  <->  {b!r}'
            if len(issue) > 3:
                ba, bb = issue[3], issue[4]
                line += (f'   A_bbox=({ba[0]:.2f},{ba[1]:.2f})-({ba[2]:.2f},{ba[3]:.2f}) '
                         f'B_bbox=({bb[0]:.2f},{bb[1]:.2f})-({bb[2]:.2f},{bb[3]:.2f})')
            print(line)
    else:
        print(f'[{name}] OK - no overlaps')
    return issues