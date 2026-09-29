# -*- coding: utf-8 -*-
"""诊断：原位替换时哪些数字没被换掉、为什么。

打印（按行）每个音的 n / box 情况，并汇总：
  * n 在 1..7 且 box 有 → 会换成字母
  * n 是 None 或超出 1..7 → 保持原样（"没认出来"）
  * 没有 box → 保持原样（这条不该出现，出现说明有别的路径在造音）
"""

import sys

import jianpu2letter.ocr_jianpu as oj

img = sys.argv[1] if len(sys.argv) > 1 else "7.png"
measures = oj.ocr_to_stream(img)

no_box, bad_n, ok = [], [], 0
row_no = 0
prev = object()
for measure in measures:
    if measure.get("row") != prev:
        prev = measure.get("row")
        row_no += 1
    for note in measure.get("notes", []):
        n, box = note.get("n"), note.get("box")
        cx = note.get("cx")
        if box is None:
            no_box.append((row_no, cx, n))
        elif n is None or not 1 <= int(n) <= 7:
            bad_n.append((row_no, cx, n))
        else:
            ok += 1

lines = [f"== {img}: 会换字母 {ok} 个；没认出来(保持原样) {len(bad_n)} 个；"
         f"无外框 {len(no_box)} 个"]
lines.append("-- 没认出来（按 x 排序，最多 40 条）")
lines += [f"   行{r} cx={cx} n={n}" for r, cx, n in sorted(bad_n, key=lambda t: t[1])[:40]]
lines.append("-- 无外框（最多 20 条）")
lines += [f"   行{r} cx={cx} n={n}" for r, cx, n in no_box[:20]]
open("diag_overlay.txt", "w", encoding="utf-8").write("\n".join(lines))
print(lines[0])
