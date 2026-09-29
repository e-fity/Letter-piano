# -*- coding: utf-8 -*-
"""
detect_notes.py — 简谱音符级检测 v3（纯图像结构定位 + 条带 OCR 认数字）。

v3 与 v2 的区别：
  1. 旋律行定位不依赖 OCR：靠连通域的「数字候选块」做 y 聚类，
     一行内候选块数量 >= 8 且尺寸一致 → 判定为旋律行（歌词/标题自动排除）。
  2. 数字识别：裁剪整行旋律条带 → 一次 RapidOCR → 按 x 把 token 数字对齐到数字块。
  3. 几何检测（音区点/减时线/附点/延时线）沿用。

用法：
  python detect_notes.py 1.jpg
"""

import re
import sys

import cv2
import numpy as np


def _make_ocr_engine():
    from rapidocr_onnxruntime import RapidOCR
    return RapidOCR()


def _ocr(engine, arr):
    out = engine(arr)
    return (out[0] if isinstance(out, tuple) else out) or []


def _norm_digits(text):
    """token 文本 → 数字串。i/l/I 视为 1；`|` 是小节线不计入；全角→半角。"""
    if not text:
        return ""
    t = text.translate(str.maketrans("０１２３４５６７８９", "0123456789"))
    out = []
    for ch in t:
        if ch.isdigit():
            out.append(ch)
        elif ch in "ilI":
            out.append("1")
    return "".join(out)


def _is_melody(text):
    """全图 OCR token 是否属于旋律行（含数字且不含中文）。"""
    if not text:
        return False
    if re.search(r"[一-鿿]", text):
        return False
    s = text.strip()
    return any(ch.isdigit() or ch in "ilI" for ch in s)


def _clamp_octave(o):
    return max(-3, min(3, o))


def main(path):
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    H_img, W_img = gray.shape
    bin_img = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 9)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(bin_img, connectivity=8)

    # 收集全图连通块（紧凑存储）
    comps = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 30:
            continue
        ar = w / max(h, 1)
        if not (0.25 <= ar <= 1.25):
            continue
        comps.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h), "area": int(area)})

    # 数字候选块按 y 聚类成行
    comps.sort(key=lambda c: c["y"])
    rows = []
    for c in comps:
        cy = c["y"] + c["h"] / 2
        placed = False
        for r in rows:
            if abs(cy - r["cy"]) < 18:
                r["blocks"].append(c)
                r["cy"] = sum((b["y"] + b["h"] / 2) for b in r["blocks"]) / len(r["blocks"])
                placed = True
                break
        if not placed:
            rows.append({"cy": cy, "blocks": [c]})
    rows.sort(key=lambda r: r["cy"])

    engine = _make_ocr_engine()
    # 全图 OCR：数字识别质量比条带 OCR 好（有完整上下文）
    raw = _ocr(engine, path)
    all_tokens = []
    for item in raw:
        box, text = item[0], item[1]
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        all_tokens.append({
            "x0": min(xs), "x1": max(xs), "y0": min(ys), "y1": max(ys), "text": str(text),
        })
    melody_tokens = [t for t in all_tokens if _is_melody(t["text"])]
    for t in melody_tokens:
        t["digits"] = [int(c) for c in _norm_digits(t["text"])]
    print(f"图片: {W_img} x {H_img}")

    for ri, row in enumerate(rows):
        blocks = row["blocks"]
        if len(blocks) < 8:
            continue  # 歌词行 / 标题 / 调号行
        hs = [b["h"] for b in blocks]
        H = float(np.median(hs))
        if len(hs) >= 5 and np.std(hs) / max(H, 1) > 0.35:
            continue  # 尺寸杂乱 → 不是数字行

        # 严格数字块
        digits = [b for b in blocks
                  if 0.7 * H <= b["h"] <= 1.4 * H and 0.3 <= b["w"] / max(b["h"], 1) <= 1.15]
        digits.sort(key=lambda b: b["x"])
        if len(digits) < 8:
            continue

        # 行 y 范围（扩展到符号区）
        y_top = int(min(b["y"] for b in blocks))
        y_bot = int(max(b["y"] + b["h"] for b in blocks))
        py0 = max(0, y_top - int(H * 1.1))
        py1 = min(H_img, y_bot + int(H * 1.3))

        # 行范围内重新收集所有块（含横线/竖线/点）
        row_comps = []
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if y + h < py0 or y > py1 or area < 12:
                continue
            row_comps.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h), "area": int(area)})

        vlines = [c for c in row_comps if c["h"] >= 1.5 * H and c["w"] <= 0.4 * H]
        hlines = [c for c in row_comps
                  if c["h"] <= 0.35 * H and c["w"] >= 0.4 * H and c["w"] / max(c["h"], 1) >= 3.0]
        dots = [c for c in row_comps
                if 0.012 * H * H <= c["area"] <= 0.4 * H * H and 0.4 <= c["w"] / max(c["h"], 1) <= 1.9]
        vlines.sort(key=lambda c: c["x"])

        # 该行的 melody token（全图 OCR），按 x 排序
        row_tokens = [t for t in melody_tokens if t["y0"] < py1 and t["y1"] > py0]
        row_tokens.sort(key=lambda t: t["x0"])

        # 歌词行过滤：识别出的数字太少 → 不是旋律行
        total_digits = sum(len(t["digits"]) for t in row_tokens)
        if total_digits < len(digits) * 0.4:
            continue

        # 数字块按 x 归属 token，token 内按 x 顺序赋值
        for d in digits:
            d["val"] = None
        for t in row_tokens:
            t["blocks"] = []
        for d in digits:
            dcx = d["x"] + d["w"] / 2
            for t in row_tokens:
                if t["x0"] - 4 <= dcx <= t["x1"] + 4:
                    t["blocks"].append(d)
                    break
        for t in row_tokens:
            t["blocks"].sort(key=lambda b: b["x"])
            for b, val in zip(t["blocks"], t["digits"]):
                b["val"] = val

        # 几何检测
        note_list = []
        for d in digits:
            cx = d["x"] + d["w"] / 2
            cy = d["y"] + d["h"] / 2
            up = sum(1 for p in dots
                     if abs(p["x"] + p["w"] / 2 - cx) < 0.9 * H
                     and d["y"] - 1.2 * H <= p["y"] + p["h"] / 2 <= d["y"] - 0.1 * H)
            down = sum(1 for p in dots
                       if abs(p["x"] + p["w"] / 2 - cx) < 0.9 * H
                       and d["y"] + d["h"] + 0.05 * H <= p["y"] + p["h"] / 2 <= d["y"] + d["h"] + 0.75 * H)
            beams = sum(1 for hl in hlines
                        if abs(hl["x"] + hl["w"] / 2 - cx) < 0.7 * H
                        and d["y"] + d["h"] + 0.25 * H <= hl["y"] + hl["h"] / 2 <= d["y"] + d["h"] + 1.35 * H)
            dotted = any(p for p in dots
                         if d["x"] + d["w"] + 0.05 * H <= p["x"] + p["w"] / 2 <= d["x"] + d["w"] + 0.9 * H
                         and d["y"] - 0.2 * H <= p["y"] + p["h"] / 2 <= d["y"] + d["h"] + 0.4 * H)
            ext_hl = [hl for hl in hlines
                      if d["x"] + d["w"] + 0.1 * H <= hl["x"] + hl["w"] / 2 <= d["x"] + d["w"] + 3.5 * H
                      and abs(hl["y"] + hl["h"] / 2 - cy) < 0.4 * H]
            extend = max([int(round(hl["w"] / (0.9 * H))) for hl in ext_hl] or [0])
            note_list.append({
                "num": d["val"], "o": _clamp_octave(up - down), "beams": beams,
                "dotted": dotted, "extend": extend, "cx": cx,
            })

        # 输出
        parts = []
        for nd in note_list:
            s = str(nd["num"]) if nd["num"] is not None else "?"
            if nd["o"] > 0:
                s += "^" * nd["o"]
            elif nd["o"] < 0:
                s += "v" * (-nd["o"])
            s += "_" * nd["beams"]
            if nd["dotted"]:
                s += "·"
            s += "—" * nd["extend"]
            parts.append(s)

        print(f"\n[行{ri:02d}] H≈{H:.0f}  数字块={len(digits)} 竖线={len(vlines)}")
        print("  行OCR:", " | ".join(f"'{t['text']}'→{''.join(map(str, t['digits']))}" for t in row_tokens))
        print("  音符:", " ".join(parts))

    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python detect_notes.py <图片路径>")
        raise SystemExit(1)
    raise SystemExit(main(sys.argv[1]))
