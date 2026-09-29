# -*- coding: utf-8 -*-
"""另一路输出：把原谱图片里的**数字直接盖成键盘字母**（原位替换，一个笔画都不动）。

与 `letter_render.py` 的区别 —— 这是两条互补的路：

  * `letter_render` 是「识别 → **重排**渲染」：连音弧、房子横线、歌词、小节线都要按模型
    重画一遍。这些记号一旦认错，整条就错 —— 用户反复报的"弧只读出一条""房子横线没了"
    "行首歌词字丢了"全出在这里。
  * 这一路**不重排**：只把数字换成字母，减时线、音区点、附点、延时线、括线、弧、
    歌词、小节线、房子线……全部保持像素原样。凡是"重排带来的问题"在这里天然不存在。

代价（都可接受，但要清楚）：

  * 字母比数字宽（映射2 里中音 `1→S`、高音 `1→E`），必须按数字外框缩放居中，
    否则会压到旁边的附点/延时线；
  * 盖字前要先抹掉原数字：用数字四周的**背景色**（取外围一圈的亮色中位）填充，
    灰底/米色扫描件也不会留白块；
  * 字形风格与原谱的衬线数字不同（Hershey 无衬线），一眼能看出是后加的。

只替换**旋律数字**：休止 `0` 保持原样（字母谱里没有键位）；**没认出来的数字也保持原样，
但在外面画一个红框** —— 这样"没换掉"和"本来就是字母"一眼能分开。
`±2/±3` 音区没有单键，按字母谱的写法在右上角加 `^`（Shift）/`v`（右 Shift）标记。
"""

import os

import cv2
import numpy as np

from . import mapping as mp
from . import ocr_jianpu

FONT = cv2.FONT_HERSHEY_SIMPLEX


def _bg_color(img, x, y, w, h, margin):
    """取数字框**外面一圈**的亮色作为背景色（用于抹掉原数字）。

    不用写死白色：扫描件常是灰底/米色，写白会在纸上留一块白斑。
    """
    y0, y1 = max(0, y - margin), min(img.shape[0], y + h + margin)
    x0, x1 = max(0, x - margin), min(img.shape[1], x + w + margin)
    ring = img[y0:y1, x0:x1].reshape(-1, 3)
    if not len(ring):
        return np.array([255, 255, 255], dtype=np.uint8)
    return np.percentile(ring, 90, axis=0).astype(np.uint8)


def _put_letter(img, cx, cy, h_ref, w_limit, key, modifier, fit):
    """把字母画在 (cx, cy) 处：**高度按整行的字高**定，宽度不够时按可用空间收窄。

    高度用行字高（而不是数字自己的框高）是关键：原谱里 `1` 的框只有 `6/3` 的一半宽，
    若按框缩放，`1→S/E` 会被挤成小字、`6/3/5` 是满尺寸，同一行里就"大小不一"。
    """
    thickness = max(1, int(round(h_ref / 18.0)))
    # getTextSize 返回 ((宽, 高), 基线)；拿参考字号(1.0)下的**字高**来定缩放
    base_h = max(cv2.getTextSize(key, FONT, 1.0, thickness)[0][1], 1)
    scale = fit * h_ref / base_h
    tw, th = cv2.getTextSize(key, FONT, scale, thickness)[0]
    if tw > w_limit:                        # 空间不够才收（尽量不收缩，保持整行一致）
        scale *= w_limit / max(tw, 1)
        tw, th = cv2.getTextSize(key, FONT, scale, thickness)[0]
    org = (int(round(cx - tw / 2.0)), int(round(cy + th / 2.0)))
    cv2.putText(img, key, org, FONT, scale, (0, 0, 0), thickness, cv2.LINE_AA)
    if modifier:                            # ±2/±3：需要按 Shift，没有单键
        mark = "^" if modifier == mp.LEFT_SHIFT else "v"
        cv2.putText(img, mark, (org[0] + tw + 2, org[1] - th), FONT,
                    max(scale * 0.45, 0.3), (0, 0, 0), max(1, thickness - 1),
                    cv2.LINE_AA)


def _imwrite_unicode(path, img):
    """写图（cv2.imwrite 遇到中文路径会静默失败，所以走 imencode + tofile）。"""
    ext = os.path.splitext(path)[1] or ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        return False
    buf.tofile(path)
    return True


def _rest_really_zero(block, binary):
    """判成休止的音，复核它到底是不是 `0` —— 用**内孔面积比**判。

    为什么需要（用户 2026-09-21 报"第三行有 5 个数字没被转换"）：数字一旦被**误判成休止符**，
    原位替换就"故意不碰"它（休止没有键位），于是一个真数字原样留在图上 —— 既没换字母、也没标记。

    判据：`0` 是这类谱里唯一**有大封闭内孔**的数字（孔比 ≈ 0.23），`6` 只有小孔（≈ 0.11），
    其余无孔 —— `ocr_jianpu.HOLE_ZERO_MIN_RATIO`(0.16) 正是按这个**2 倍余量**定的。

    ⚠ **别再用字形相似度复核**（2026-09-29 连踩两次）：
      ① 借带门槛的 `_classify_by_reference` —— 真 `0` 的绝对相似度只有 0.46~0.55，够不着
         `min_score=0.55`，于是 42/43/45 的真 `0` 被**成片标红**（那几张小字号、数字高仅 12px）；
      ② 换成"谁最像"（argmax）**仍不够** —— 同一批真 `0` 里还有 10 个被判"最像 1 或 6"。
         根因是**相似度拿系统渲染字形当基准**，换谱、字体一对不上就不成立。
    实测（那三张图）：10 个真 `0` 的**孔比 0.211~0.235**；唯一该报警的那块（把 `6` 判成了休止）
    孔比只有 **0.137** —— 孔比把两类干净分开，而且**与字体无关**（是几何量，不是相似度）。
    """
    detail = ocr_jianpu._hole_class(block, binary, return_detail=True)[1]
    ratio = (detail or {}).get("ratio")
    if ratio is None:
        return True                     # 提不出孔（空块等）：不报警，别制造另一种误报
    return ratio >= ocr_jianpu.HOLE_ZERO_MIN_RATIO


def render(measures, out_path, color=None, scale=None, fit=0.95):
    """在原图上把数字原位盖成字母，写到 `out_path`。

    color / scale 不给时，用 `ocr_jianpu` 最近一次识别留下的缩放彩色图与倍数
    （见那边的 `_LAST_COLOR` / `_LAST_SCALE`）—— 坐标就统一在**缩放后**的图片空间里，
    所以底图必须和 `note["box"]` 同空间，别无脑传原图。

    返回 (drawn, rest, unknown, ok)：盖了字母的数字个数 / 休止符个数 /
    没认出来（原数字保留、并在外面画红框）的个数 / 是否写成功。
    """
    img = color if color is not None else ocr_jianpu._LAST_COLOR
    if img is None:
        raise RuntimeError("没有可用的底图：请先跑一次 ocr_to_stream()，或显式传 color=")
    if scale is None:
        scale = ocr_jianpu._LAST_SCALE
    img = img.copy()
    drawn = rest = unknown = 0
    unknown_boxes = []      # 要画红框的位置（缩回原尺寸后再画，线才不糊）
    # 复核"休止"用的二值图（与音符 `box` 同一坐标系）；没跑过 OCR 就是 None，那时跳过复核
    binary = ocr_jianpu._LAST_BINARY
    # 按**行**分组：字母高度取该行数字框高的中位数，一行内统一（见 _put_letter 的说明）
    rows = {}
    for measure in measures or []:
        for note in measure.get("notes", []):
            rows.setdefault(measure.get("row"), []).append(note)
    for notes in rows.values():
        heights = [n["box"][3] for n in notes if n.get("box")]
        h_ref = float(np.median(heights)) if heights else 0.0
        for i, note in enumerate(notes):
            box, n = note.get("box"), note.get("n")
            if box is None:
                continue
            x, y, w, h = (int(v) for v in box)
            if n is None or not 1 <= int(n) <= 7:
                if n == 0:
                    blob = {"x": x, "y": y, "w": w, "h": h}
                    if binary is not None and not _rest_really_zero(blob, binary):
                        # 判成休止、形状却不像 0 → 多半是把数字误判成了休止，标出来
                        unknown += 1
                        unknown_boxes.append((x, y, w, h))
                    else:
                        rest += 1   # 真休止：字母谱里没有键位，保持原样
                else:
                    # 没认出来：原数字留着，记下来等缩回原尺寸后画红框
                    #（用户 2026-09-21 选定：只标问题位置，其余不动）
                    unknown += 1
                    unknown_boxes.append((x, y, w, h))
                continue
            x, y, w, h = (int(v) for v in box)
            # 可用横向空间：不压到左右相邻的数字（各留 0.10 字高余量），
            # 上限再兜到 3 倍框宽，免得空行里画出个巨字母。
            limit = 3.0 * w
            nxt = notes[i + 1].get("box") if i + 1 < len(notes) else None
            if nxt:
                limit = min(limit, nxt[0] - x - 0.10 * h_ref)
            prv = notes[i - 1].get("box") if i > 0 else None
            if prv:
                limit = min(limit, (x + w) - (prv[0] + prv[2]) - 0.10 * h_ref)
            limit = max(limit, w)
            key, modifier = mp.key_for(int(n), int(note.get("o", 0) or 0), "mapping2")
            margin = max(2, int(round(0.10 * h)))
            img[max(0, y - margin):y + h + margin,
                max(0, x - margin):x + w + margin] = _bg_color(img, x, y, w, h, margin)
            _put_letter(img, x + w / 2.0, y + h / 2.0, h_ref or h, limit,
                        key, modifier, fit)
            drawn += 1
    if scale and abs(scale - 1.0) > 1e-6:
        # 底图是"归一化到数字高≈48px"的**放大**图；输出回到**原图尺寸**
        #（与输入同尺寸便于并排比；否则 12.jpg 那种 4.4 倍图会变成十几 MB）。
        h0 = int(round(img.shape[0] / scale))
        w0 = int(round(img.shape[1] / scale))
        img = cv2.resize(img, (w0, h0), interpolation=cv2.INTER_AREA)
        k = 1.0 / scale
    else:
        k = 1.0
    for bx, by, bw, bh in unknown_boxes:
        # 红框画在**缩回原尺寸之后**：画在放大图上再缩回来会被重采样磨成虚影
        pad = max(2, int(round(0.12 * bh * k)))
        cv2.rectangle(img, (int(bx * k) - pad, int(by * k) - pad),
                      (int((bx + bw) * k) + pad, int((by + bh) * k) + pad),
                      (0, 0, 255), 2)
    return drawn, rest, unknown, _imwrite_unicode(out_path, img)
