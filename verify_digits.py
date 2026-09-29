# -*- coding: utf-8 -*-
"""对照原图，估"数字转换准确率"（可复现、可人工复核的口径）。

做法：
  1. 跑一遍识别流水线 → 每个音的位置(box) + 模型判定的数字(n)；
  2. 把**每一行**从放大后的原图里裁成一条带子（上下留白、左右各留一个字宽），
     整条单独交给 RapidOCR 再读一遍 —— 它的切法（行带子）与流水线的（整页）不同，
     且与流水线里"字形参照 / 聚类投票"是**不同族**的方法，可做交叉验证；
  3. 两边各自拼成"数字串"，逐字对齐算命中率（difflib）；行相似度 <0.98 的行会标出来，
     那几行就是**待人工核对**的位置。
  4. 模型判不出（n 为空）的音单独计"未转换"。

⚠ 口径说明：**一致率 ≠ 准确率**（两边同时错的概率存在，虽然小），所以它是**下界**；
   真正的用法是看"标了相似度的行"去人工核对。

用法：python verify_digits.py            # 跑 piano/ 全部
      python verify_digits.py 3.png      # 只跑一张
输出：verify_digits.txt
"""

import difflib
import os
import sys

import cv2

import jianpu2letter.ocr_jianpu as oj

# OCR 常见的字母↔数字混淆，认字前先还原（工具注释里写过：6→b/G、5→S）
# ⚠ **不要**把 `l`/`I` 还原成 1：行带子里那些减时线/附点常被 OCR 读成 `l`，
# 还原成 1 会凭空造出一堆"独立识别=1"的假分歧（第一版就这么把一致率压到 83%）。
CONFUSE = str.maketrans({"b": "6", "B": "6", "G": "6", "g": "6", "S": "5", "s": "5",
                         "O": "0", "o": "0", "Z": "2", "z": "2"})


def read_line(engine, crop, off_x, k=1.0):
    """整行送检：返回 [(x, 数字)]，x 换算回原图坐标。

    逐个字符按它在文本块里的相对位置插值出 x —— 和流水线里 `xc = x0 + span*(ci+0.5)/n`
    是同一套估计；`k` 是带子被放大送检的倍数，**必须除回去**，否则坐标整片偏移、
    每个音都"对不上"（第一版漏了这一步，一致率算成 38%）。
    """
    res = oj._ocr(engine, crop)
    digits = []
    for item in res or []:
        if not (isinstance(item, (list, tuple)) and len(item) >= 2):
            continue
        box, text = item[0], str(item[1]).translate(CONFUSE)
        xs = [float(p[0]) for p in box]
        x0, x1 = min(xs), max(xs)
        n = max(len(text), 1)
        for ci, ch in enumerate(text):
            if ch in "01234567":
                digits.append((off_x + (x0 + (x1 - x0) * (ci + 0.5) / n) / k, int(ch)))
    digits.sort(key=lambda t: t[0])
    return digits


def main():
    names = sys.argv[1:] or sorted(
        n for n in os.listdir("piano") if n.lower().endswith((".jpg", ".png")))
    engine = oj._make_ocr_engine()
    out, total = [], {"hit": 0, "model": 0, "unknown": 0}
    for name in names:
        path = os.path.join("piano", name)
        measures = oj.ocr_to_stream(path)
        color = oj._LAST_COLOR                 # 放大后的原图（box 就在这个坐标系里）
        rows, stat = [], {"hit": 0, "model": 0, "unknown": 0, "diff": 0}

        by_row = {}
        for measure in measures:
            by_row.setdefault(measure.get("row"), []).extend(measure.get("notes", []))

        for _key, notes in by_row.items():
            stat["unknown"] += sum(1 for n in notes
                                   if n.get("box") and n.get("n") is None)
            notes = [n for n in notes if n.get("box") and n.get("n") is not None]
            if not notes:
                continue
            notes.sort(key=lambda n: n["box"][0])
            hs = [n["box"][3] for n in notes]
            hr = int(sum(hs) / len(hs))
            x0 = max(0, int(min(n["box"][0] for n in notes) - hr))
            x1 = min(color.shape[1],
                     int(max(n["box"][0] + n["box"][2] for n in notes) + hr))
            y0 = max(0, int(min(n["box"][1] for n in notes) - 0.6 * hr))
            y1 = min(color.shape[0],
                     int(max(n["box"][1] + n["box"][3] for n in notes) + 1.6 * hr))
            crop = color[y0:y1, x0:x1]
            if crop.size == 0:
                continue
            k = max(1.0, 1400.0 / max(crop.shape[1], 1))
            if k > 1.0:
                crop = cv2.resize(crop, None, fx=k, fy=k,
                                  interpolation=cv2.INTER_CUBIC)
            got_seq = "".join(str(d) for _x, d in read_line(engine, crop, x0, k))
            model = "".join(str(n["n"]) for n in notes)
            sm = difflib.SequenceMatcher(None, model, got_seq)
            stat["hit"] += sum(b.size for b in sm.get_matching_blocks())
            stat["model"] += len(model)
            ratio = sm.ratio()
            spans = [f"{model[i1:i2]}→{got_seq[j1:j2]}"
                     for tag, i1, i2, j1, j2 in sm.get_opcodes() if tag != "equal"]
            if spans:
                stat["diff"] += 1
                rows.append(f"  行y≈{int(sum(n['box'][1] for n in notes) / len(notes))} "
                            f"相似度 {ratio:.2f} 差异段: {' | '.join(spans[:6])}")
                rows.append(f"     模型 {model}")
                rows.append(f"     独立 {got_seq}")
        rate = stat["hit"] / stat["model"] * 100 if stat["model"] else 0.0
        rows.insert(0, f"== {name}: 音 {stat['model']} 个 | 一致 {stat['hit']}"
                       f" | **不一致 {stat['diff']}**"
                       f" | 未转换(模型给不出) {stat['unknown']}"
                       f" → 一致率 {rate:.1f}%")
        out += rows
        out.append("")
        for k in total:
            total[k] += stat[k]

    rate = total["hit"] / total["model"] * 100 if total["model"] else 0.0
    head = (f"=== 全 {len(names)} 张合计：判定 {total['model']} 个 | 逐字命中 {total['hit']}"
            f" | 未转换 {total['unknown']} → **一致率 {rate:.1f}%**（下界）")
    out.insert(0, head)
    out.insert(1, "=== 口径：模型数字串 vs 把该行单独裁出来交给 RapidOCR 再读一遍的数字串，"
                  "逐字对齐命中率；标了相似度的行需要人工核对。")
    out.insert(2, "")
    open("verify_digits.txt", "w", encoding="utf-8").write("\n".join(out))
    print(head)


if __name__ == "__main__":
    main()
