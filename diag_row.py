# -*- coding: utf-8 -*-
"""打印某几行的每个音（值/音区/外框/括号标记），用于查"某个字形到底去哪了"。"""
import sys
import jianpu2letter.ocr_jianpu as oj
img = sys.argv[1]; rows_want = {int(v) for v in sys.argv[2:]}
measures = oj.ocr_to_stream(img)
row_no = 0; prev = object(); out = [f"== {img}"]
for m in measures:
    if m.get("row") != prev:
        prev = m.get("row"); row_no += 1
    if row_no not in rows_want:
        continue
    for note in m.get("notes", []):
        b = note.get("box")
        out.append(f"  行{row_no} n={note.get('n')} o={note.get('o')} "
                   f"box=({int(b[0])},{int(b[1])},{int(b[2])},{int(b[3])}) "
                   f"par=({note.get('paren_before')},{note.get('paren_after')})")
open("diag_row.txt", "w", encoding="utf-8").write("\n".join(out))
print("done", len(out))
