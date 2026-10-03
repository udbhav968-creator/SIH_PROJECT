"""Figures that depend on which model is served. build(f) returns (text_edits, table_rows)."""
import re
import html

import track

ROW_KEYS = ["Normal Road / Sound Pavement", "Crack (Longitudinal", "Severe Cavity / Pothole",
            "Waterlogging / Flooding Hazard", "Missing Zebra Crossing", "Missing Road Divider",
            "Damaged Traffic Sign"]


def edit_row(xml, row_key, values):
    """Replace the visible value in cells 1..len(values) of the table row whose
    first cell contains row_key."""
    for m in re.finditer(r'<w:tr[ >].*?</w:tr>', xml, re.S):
        row = m.group(0)
        first = track.RUN_RE.search(row)
        if not first or row_key not in html.unescape(first.group(2)):
            continue
        cells = re.findall(r'<w:tc>.*?</w:tc>', row, re.S)
        new_row = row
        for ci, val in enumerate(values, start=1):
            if val is None:
                continue
            cell = cells[ci]
            vis = list(track.RUN_RE.finditer(cell))
            if not vis:
                continue
            old = html.unescape(vis[-1].group(2))
            if old == val:
                continue
            new_cell = track.edit(cell, old, val)
            new_row = new_row.replace(cell, new_cell, 1)
        return xml[:m.start()] + new_row + xml[m.end():]
    raise ValueError(row_key)


def apply(xml, f):
    E = f["text_edits"]
    for old, new in E:
        xml = track.edit(xml, old, new)
    for key, vals in f["rows"].items():
        xml = edit_row(xml, key, vals)
    return xml
