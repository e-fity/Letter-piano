# -*- coding: utf-8 -*-
"""诊断：某个"没认出来"的块，形状到底哪里异常。

把「没认出来的块」和「同图已确认的同类数字」并排打出来（ASCII 掩码 + 与参照字形的
最高相关 + 分类器给的结果），一眼就能看出是哪种毛病：

  * 笔画被二值化啃掉 / 断成两截   → 掩码里能看到缺口
  * 减时线、音区点糊进块里        → 掩码里多出一条横线或一个点
  * 只是"字形变体"对不上系统字体  → 掩码完整、但和参照的相关就是低

用法：python diag_shape_one.py 14.jpg 2      # 图、要对比的数字
输出：diag_shape_one.txt
"""

import sys

import jianpu2letter.ocr_jianpu as oj


def main():
    img = sys.argv[1] if len(sys.argv) > 1 else "14.jpg"
    want = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    measures = oj.ocr_to_stream(img)
    binary = oj._LAST_BINARY
    refs = oj._all_reference_glyphs()
    ref_want = refs.get(want, [])

    unknowns, samed = [], []
    row_no = 0
    prev = object()
    for measure in measures:
        if measure.get("row") != prev:
            prev = measure.get("row")
            row_no += 1
        for note in measure.get("notes", []):
            box = note.get("box")
            if not box:
                continue
            b = {"x": int(box[0]), "y": int(box[1]), "w": int(box[2]), "h": int(box[3])}
            if note.get("n") is None:
                unknowns.append((row_no, b, note.get("o")))
            elif note.get("n") == want:
                samed.append((row_no, b, note.get("o")))

    out = []

    def dump(tag, r, b, o):
        roi = binary[b["y"]:b["y"] + b["h"], b["x"]:b["x"] + b["w"]] > 0
        feat = oj._shape_feature(b, binary)
        sim = max((oj._corr(feat, x) for x in ref_want), default=-1.0)
        d, sc = oj._classify_by_reference(b, binary, refs, return_score=True)
        out.append(f"-- {tag} 行{r} box=({b['x']},{b['y']},{b['w']},{b['h']}) o={o} "
                   f"与参照{want}最高相关={sim:.3f} 分类器给={d}({sc:.2f})")
        out.extend("   " + ln for ln in oj._ascii_mask(roi, step=2))
        out.append("")

    for r, b, o in unknowns[:4]:
        dump("没认出来", r, b, o)
    for r, b, o in samed[:4]:
        dump(f"已确认的{want}", r, b, o)
    open("diag_shape_one.txt", "w", encoding="utf-8").write("\n".join(out))
    print(f"{img}: 没认出来 {len(unknowns)} 个，已确认的 {want} 有 {len(samed)} 个")


if __name__ == "__main__":
    main()
