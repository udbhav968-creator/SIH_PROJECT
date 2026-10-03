"""Tracked-change helper for the ROAD-SHIELD report.

edit(xml, old, new) finds `old` inside the visible text of ONE run (plain run,
or a run inside a previous <w:ins>) and replaces it with a tracked deletion of
`old` plus a tracked insertion of `new`, keeping the run's formatting.
"""
import re
import html

AUTHOR = "Claude (audit 2026-10-03)"
DATE = "2026-10-03T00:00:00Z"
_id = [12000]


def nid():
    _id[0] += 1
    return _id[0]


def reid(tag):
    return re.sub(r'w:id="\d+"', 'w:id="%d"' % nid(), tag)


def esc(t):
    return html.escape(t, quote=False)


RUN_RE = re.compile(r'<w:r>(<w:rPr>(?:(?!</w:rPr>).)*</w:rPr>)?<w:t(?: xml:space="preserve")?>([^<]*)</w:t></w:r>', re.S)


def _run(rpr, text, deleted=False):
    if not text:
        return ""
    tag = "w:delText" if deleted else "w:t"
    return f'<w:r>{rpr or ""}<{tag} xml:space="preserve">{esc(text)}</{tag}></w:r>'


def edit(xml, old, new, occurrence=1, must=True):
    """Replace the `occurrence`-th visible match of `old`."""
    old_e = esc(old)
    count = 0
    for m in RUN_RE.finditer(xml):
        text = m.group(2)
        if old_e not in text:
            continue
        count += 1
        if count != occurrence:
            continue
        rpr = m.group(1) or ""
        raw = html.unescape(text)
        i = raw.index(old)
        before, after = raw[:i], raw[i + len(old):]
        # is this run inside a previous insertion?
        open_ins = xml.rfind("<w:ins ", 0, m.start())
        close_ins = xml.rfind("</w:ins>", 0, m.start())
        inside_ins = open_ins > close_ins
        d = f'<w:del w:id="{nid()}" w:author="{AUTHOR}" w:date="{DATE}">{_run(rpr, old, True)}</w:del>'
        ins = (f'<w:ins w:id="{nid()}" w:author="{AUTHOR}" w:date="{DATE}">{_run(rpr, new)}</w:ins>'
               if new else "")
        if inside_ins:
            # close the earlier author's insertion around our change, then reopen it
            prev_open = re.match(r'<w:ins [^>]*>', xml[open_ins:]).group(0)
            # earlier-inserted text that we delete: <w:ins prev><w:del ours>..</w:del></w:ins>
            d = "</w:ins>" + reid(prev_open) + d + "</w:ins>" + ins + reid(prev_open)
            repl = _run(rpr, before) + d + _run(rpr, after)
        else:
            repl = _run(rpr, before) + d + ins + _run(rpr, after)
        return xml[:m.start()] + repl + xml[m.end():]
    if must:
        raise ValueError(f"not found (occurrence {occurrence}): {old[:90]!r}")
    return xml


def insert_after(xml, anchor, new):
    """Tracked insertion of `new` right after visible text `anchor` (anchor kept)."""
    return edit(xml, anchor, anchor + "\u0000" + new) if False else _insert_after(xml, anchor, new)


def _insert_after(xml, anchor, new):
    a = esc(anchor)
    for m in RUN_RE.finditer(xml):
        if a in m.group(2):
            rpr = m.group(1) or ""
            raw = html.unescape(m.group(2))
            i = raw.index(anchor) + len(anchor)
            ins = f'<w:ins w:id="{nid()}" w:author="{AUTHOR}" w:date="{DATE}">{_run(rpr, new)}</w:ins>'
            open_ins = xml.rfind("<w:ins ", 0, m.start())
            close_ins = xml.rfind("</w:ins>", 0, m.start())
            if open_ins > close_ins:
                prev_open = re.match(r'<w:ins [^>]*>', xml[open_ins:]).group(0)
                ins = "</w:ins>" + ins + reid(prev_open)
            return xml[:m.start()] + _run(rpr, raw[:i]) + ins + _run(rpr, raw[i:]) + xml[m.end():]
    raise ValueError(f"anchor not found: {anchor[:80]!r}")
