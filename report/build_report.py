"""Assemble report/drafts/*.md into report/report.docx.

Usage (from the repository root):  python report/build_report.py
Requires pandoc (2.9+) and python-docx (pip install python-docx).

The drafts stay the source of truth (they are what the git history timestamps);
this script only joins them, turns each figure + caption pair into a captioned
Word figure, sizes table columns to their content, renders $...$ maths as native
Word equations (pandoc), and applies a plain academic style.
"""
from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parent.parent
DRAFTS = ROOT / "report" / "drafts"
OUT = ROOT / "report" / "report.docx"
PARTS = ["00-abstract.md", "01-problem-and-algorithms.md", "02-theory-and-predictions.md", "03-methodology.md",
         "04-results.md", "05-surprises.md", "06-next.md", "07-references.md"]
META = """---
title: "Cache-Blocked Iterative vs Strassen Matrix Multiplication"
subtitle: "An empirical comparison. Advanced Algorithms, Programming Assignment 2"
author: "Jarrod Carmont"
date: "November 2026"
---

"""
INK, HEAD = RGBColor(0x0B, 0x0B, 0x0B), RGBColor(0x1C, 0x3F, 0x6E)

FIGURE = re.compile(r"!\[[^\]]*\]\(\.\./\.\./(figures/[^)]+)\)\s*\n\s*\n\*(Figure \d+\..*?)\*[ \t]*\n", re.S)


def figures(md: str) -> str:
    """'![alt](../../figures/x.png)' + italic caption paragraph -> one captioned figure."""
    return FIGURE.sub(lambda m: f"![{m.group(2).strip()}]({m.group(1)}){{width=16cm}}\n", md)


def size_tables(md: str) -> str:
    """Give pipe-table separator rows dash counts proportional to content width, so
    pandoc sizes columns to their content instead of equally."""
    lines, out, i = md.split("\n"), [], 0
    while i < len(lines):
        if lines[i].startswith("|") and i + 1 < len(lines) and re.fullmatch(r"\|(\s*:?-+:?\s*\|)+", lines[i + 1].strip()):
            j = i
            while j < len(lines) and lines[j].startswith("|"):
                j += 1
            rows = [lines[k] for k in range(i, j) if k != i + 1]
            cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
            ncol = len(cells[0])
            width = [max(len(r[c]) if c < len(r) else 0 for r in cells) for c in range(ncol)]
            total = sum(width) or 1
            dashes = [max(8, round(80 * w / total)) for w in width]  # >= 8 so short ID columns never wrap
            out.append(lines[i])
            out.append("|" + "|".join("-" * d for d in dashes) + "|")
            out.extend(lines[i + 2:j])
            i = j
        else:
            out.append(lines[i])
            i += 1
    return "\n".join(out)


def reference_docx(path: Path) -> None:
    """Pandoc's default reference document, restyled: A4, 2.5 cm margins, Calibri."""
    data = subprocess.run(["pandoc", "--print-default-data-file", "reference.docx"], check=True,
                          capture_output=True).stdout
    path.write_bytes(data)
    doc = Document(str(path))
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
    for side in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
        setattr(sec, side, Cm(2.5))
    # Look styles up by their stored name: python-docx's st["Heading 1"] translates the
    # name to "heading 1", which pandoc's reference document does not use.
    st = {s.name: s for s in doc.styles}
    for name in ("Normal", "Body Text", "First Paragraph", "Compact", "Abstract", "Bibliography"):
        if name in st:
            f = st[name].font
            f.name, f.size, f.color.rgb = "Calibri", Pt(11), INK
            pf = st[name].paragraph_format
            pf.space_after, pf.line_spacing = Pt(6), 1.12
    for name, size in (("Title", 22), ("Subtitle", 13), ("Heading 1", 16), ("Heading 2", 13), ("Heading 3", 11.5)):
        f = st[name].font
        f.name, f.size, f.bold, f.color.rgb = "Calibri", Pt(size), name != "Subtitle", HEAD
        st[name].paragraph_format.space_before = Pt(14 if name.startswith("Heading 1") else 10)
        st[name].paragraph_format.space_after = Pt(4)
        st[name].paragraph_format.keep_with_next = True
    for name in ("Author", "Date"):
        st[name].font.name, st[name].font.size = "Calibri", Pt(11)
    for name in ("Image Caption", "Table Caption", "Caption"):
        if name in st:
            st[name].font.name, st[name].font.size, st[name].font.italic = "Calibri", Pt(9.5), True
            st[name].font.color.rgb = RGBColor(0x52, 0x51, 0x4E)
    for name in ("Source Code", "Verbatim Char"):
        if name in st:
            st[name].font.name, st[name].font.size = "Consolas", Pt(9)
    doc.save(str(path))


def finish(path: Path) -> None:
    """Borders and a shaded header row on every table; centre figures."""
    doc = Document(str(path))
    for t in doc.tables:
        tblPr = t._tbl.tblPr
        borders = OxmlElement("w:tblBorders")
        for edge in ("top", "bottom", "insideH"):
            e = OxmlElement(f"w:{edge}")
            e.set(qn("w:val"), "single"), e.set(qn("w:sz"), "4"), e.set(qn("w:color"), "BFBFBF")
            borders.append(e)
        tblPr.append(borders)
        for cell in t.rows[0].cells:
            shd = OxmlElement("w:shd")
            shd.set(qn("w:val"), "clear"), shd.set(qn("w:color"), "auto"), shd.set(qn("w:fill"), "EEF2F7")
            cell._tc.get_or_add_tcPr().append(shd)
            for p in cell.paragraphs:
                for r in p.runs:
                    r.bold = True
        for row in t.rows:
            for cell in row.cells:
                for p in cell.paragraphs:
                    p.paragraph_format.space_after = Pt(1)
                    for r in p.runs:
                        r.font.size = Pt(9.5)
    for p in doc.paragraphs:
        if p.style.name in ("Captioned Figure", "Figure") or p._p.xpath(".//pic:pic"):
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.save(str(path))


def main() -> None:
    md = META + "\n\n".join((DRAFTS / p).read_text(encoding="utf-8").strip() for p in PARTS) + "\n"
    md = size_tables(figures(md))
    missing = re.findall(r"\]\((?!figures/)[^)]*\.png\)", md)
    if missing:
        sys.exit(f"figure paths not rewritten: {missing}")
    with tempfile.TemporaryDirectory() as tmp:
        src, ref = Path(tmp) / "report.md", Path(tmp) / "reference.docx"
        src.write_text(md, encoding="utf-8")
        reference_docx(ref)
        # --columns=20: every table row is then "longer than the line", so pandoc always
        # emits explicit relative column widths (from size_tables) instead of leaving
        # widths to the viewer, which some viewers collapse.
        subprocess.run(["pandoc", str(src), "-f", "markdown+lists_without_preceding_blankline", "-o", str(OUT), f"--reference-doc={ref}",
                        f"--resource-path={ROOT}", "--columns=20"], check=True)
    finish(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
