# -*- coding: utf-8 -*-
"""诊断 5：把判成休止符 `0` 的音，再用「形状最像哪个数字」复核一遍。

真 `0` 最像的当然是 0；被误判成休止的数字（6/3/5…）最像的就是它自己 —— 用这个把
"疑似的假休止"挑出来，好在原位替换图上标红框。

用法：python diag_rest_shape.py 7.png
"""

import sys

import jianpu2letter.ocr_jianpu as oj

img = sys.argv[1] if len(sys.argv) > 1 else "7.png"
measures = oj.ocr_to_stream(img)
binary = oj._LAST_BINARY
refs = oj._all_reference_glyphs()

row_no = 0
prev = object()
out = [f"== {img}"]
n_rest = n_suspect = 0
for measure in measures:
    if measure.get("row") != prev:
        prev = measure.get("row")
        row_no += 1
    for note in measure.get("notes", []):
        if note.get("n") != 0:
            continue
        n_rest += 1
        box = note.get("box")
        if not box:
            continue
        b = {"x": int(box[0]), "y": int(box[1]), "w": int(box[2]), "h": int(box[3])}
        digit, score = oj._classify_by_reference(b, binary, refs, return_score=True)
        if digit not in (0, None) and score >= 0.45:
            n_suspect += 1
        out.append(f"   行{row_no} x={b['x']} o={note.get('o')} "
                   f"形状最像 {digit}（{score:.2f}）")
out.append(f"合计：休止 {n_rest} 个，其中疑似假休止 {n_suspect} 个")
open("diag_rest_shape.txt", "w", encoding="utf-8").write("\n".join(out))
print(out[-1])
