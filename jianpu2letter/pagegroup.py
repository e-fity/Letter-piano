# -*- coding: utf-8 -*-
"""把同一首曲子的多页 `_review.json` 合并成一份。

用于"一首曲子被印在两页（或多页）上"的情况 —— 在 `piano_config.json` 里用
顶层 `_groups` 声明页组与页序，例如：

    "_groups": { "演员": ["29.jpg", "30.jpg"] }

每页各自 OCR 出来的 `_review.json` 行号都从 1 开始，所以合并要做三件事：

1. **行号顺延**：第 2 页的行接在第 1 页之后（否则两组 L01 会撞车）；
2. **接缝自动判定**：谱子按"行"排版，翻页有两种断法 ——
   * 断在**两小节之间**：上一页末小节是完整的 → 直接拼，无事发生；
   * 断在**小节中间**：那一个小节被切成两半，分别落在两页的 JSON 里。
     判据：`上一页末小节拍数 + 下一页首小节拍数 == 一个完整小节`（按拍号算）。
     成立就**自动合成一个小节**；不成立就**原样保留**，并把这条接缝记下来，
     由调用方提示用户核对（宁可让用户看一眼，也别悄悄接错）。
3. **元数据取第 1 页**（曲名/调号/拍号）—— 第 2 页通常没有页眉，读不到。
"""

from . import review_io


def _beats(measure):
    """这个小节的拍数（单位＝四分音符）。"""
    return sum(float(n.get("q", 0.0)) for n in measure.get("notes", []))


def _full_beats(meter):
    """一个完整小节有多少拍（单位＝四分音符）：'4/4'→4.0、'2/4'→2.0、'6/8'→3.0。"""
    try:
        num, den = (int(v) for v in str(meter).split("/"))
        return num * 4.0 / den
    except (ValueError, ZeroDivisionError):
        return None


def merge_reviews(paths, out_path=None):
    """合并多页审核 JSON，返回 `(metadata, measures, seams)`。

    `seams[i]` 描述第 i 页与第 i+1 页之间的接缝：
        {"after_page": i, "beats_prev": x, "beats_next": y, "merged": bool}
    `merged=False` 且两半**相加不等于**一个完整小节时，多半需要人工核对。
    """
    if not paths:
        raise ValueError("没有可合并的页")

    metadata, measures = review_io.load_review(paths[0])
    seams = []
    row_offset = max((int(m.get("row", 1)) for m in measures), default=1)

    for page_index, path in enumerate(paths[1:], 1):
        _meta, page_measures = review_io.load_review(path)
        for measure in page_measures:
            measure["row"] = int(measure.get("row", 1)) + row_offset

        prev_last, next_first = measures[-1], page_measures[0]
        b_prev, b_next = _beats(prev_last), _beats(next_first)
        full = _full_beats(metadata.get("meter", "4/4"))
        merged = False
        # 两半各自都不完整、合起来正好一个整小节 → 是小节被翻页切开
        if (full and b_prev < full - 1e-6 and b_next < full - 1e-6
                and abs(b_prev + b_next - full) < 1e-6):
            prev_last["notes"] = (prev_last.get("notes") or []) + (next_first.get("notes") or [])
            prev_last["bar_x"] = next_first.get("bar_x")
            prev_last["repeat_after"] = next_first.get("repeat_after")
            page_measures = page_measures[1:]
            merged = True
        seams.append({
            "after_page": page_index,
            "beats_prev": round(b_prev, 3),
            "beats_next": round(b_next, 3),
            "merged": merged,
        })

        measures.extend(page_measures)
        row_offset = max((int(m.get("row", 1)) for m in measures), default=row_offset)

    if out_path:
        review_io.save_review(out_path, measures,
                              metadata.get("title", ""), metadata.get("key", "1=C"),
                              metadata.get("meter", "4/4"))
    return metadata, measures, seams
