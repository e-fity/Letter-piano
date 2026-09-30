# -*- coding: utf-8 -*-
"""
把 note 流渲染成「中文数字谱」图片（PNG），支持自动换行。

使用 matplotlib 绘制：
  数字 1-7、休止 0、上/下音区点、减时线、附点、延时线、小节线、歌词。
  每 measures_per_line 个小节换一行（默认 4）。

中文字体：自动探测常见 CJK 字体；找不到时数字/符号仍正常，仅中文显示为方框。
"""

import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager

CJK_FONT_CANDIDATES = [
    "Noto Sans CJK SC", "Noto Sans CJK", "Source Han Sans SC", "Source Han Sans CN",
    "Microsoft YaHei", "SimHei", "PingFang SC", "WenQuanYi Zen Hei",
    "Arial Unicode MS", "Heiti SC",
]

X_PER_QUARTER = 1.6   # MusicXML：每个四分音符的横向宽度
OCR_LINE_WIDTH = 64.0 # 图片 OCR：按原图 x 坐标映射到该宽度，避免短音挤在一起
LINE_GAP = 2.9        # 基础行距（数据单位）
# 多段歌词：每多一段，该行再额外占这么高（与 `_draw_note` 里画歌词的行距是同一个数）。
# 3 段词的行因此能拿到 `LINE_GAP + 2*LYRIC_LINE_STEP` 的高度，不会压到下一行的数字上。
LYRIC_LINE_STEP = 1.05


def _verse_count(notes):
    """这一行要画几段歌词（1~3）：看有没有 `lyric2` / `lyric3`。"""
    if any(n.get("lyric3") for n in notes):
        return 3
    if any(n.get("lyric2") for n in notes):
        return 2
    return 1

# 逐音置信度：低于此值的音在输出图里**加黄底**标出（可疑，要人工看一眼）。
# **必须与 ocr_jianpu.CONF_WARN 一致** —— 那边负责算 conf，这边负责画。
# 这里单独放一份，是为了不让渲染模块去依赖 cv2 / rapidocr 那串重依赖。
CONF_WARN = 0.70
CONF_HL_COLOR = "#FFEB3B"   # 黄底

# 房子横线端点吸到小节线后，再向内缩这么多（数据单位，约 0.12 个字符格）——
# 正好贴在小节线上时，两端竖钩会和小节线连成"T"字（用户 2026-09-21 不要这样）。
HOUSE_END_GAP = 0.12


def _is_low_conf(item, conf_warn=CONF_WARN):
    """该音是否该标黄底。

    只有 OCR 路径的音带 "conf" 这个键；MusicXML / 网站 JSON / 文本谱没有它，
    一律**不**标——否则那几条路径整首谱都会被涂黄。
    """
    if "conf" not in item:
        return False
    conf = item.get("conf")
    return conf is None or float(conf) < conf_warn


def low_conf_notes(measures, conf_warn=CONF_WARN):
    """列出置信度低于阈值的音：[(行号, 行内音序, 数字, 置信度), ...]。

    行号/音序的编号方式与渲染分行一致（有 row 用 row，否则第 1 行起累积），
    所以能和 VERBOSE_DEBUG 打印的 `行X 音Y` 直接对上。
    置信度为 None 表示**拿不到分数**（未识别，或块尺寸不符合），同样算可疑。
    非 OCR 路径没有任何音带 conf，所以返回空列表。
    """
    out = []
    row_no = 0
    note_no = 0
    prev_row = object()
    for measure in measures or []:
        if measure.get("row") != prev_row:
            prev_row = measure.get("row")
            row_no += 1
            note_no = 0
        for note in measure.get("notes", []):
            note_no += 1
            if not _is_low_conf(note, conf_warn):
                continue
            conf = note.get("conf")
            out.append((row_no, note_no, note.get("n"),
                        None if conf is None else round(float(conf), 3)))
    return out


def _find_cjk_font():
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in CJK_FONT_CANDIDATES:
        if name in available:
            return name
    for f in available:
        for kw in ("CJK", "Hei", "YaHei", "Song", "WenQuanYi", "Source Han"):
            if kw in f:
                return f
    return None


def _draw_note(ax, x, y, item, cjk_font, draw_dot=True, highlight=False):
    """画一个音符（数字/休止 + 音区点）。

    draw_dot=False 时不画附点：OCR 路径把附点当成**独立字符**、由它自己那一格
    画（见 _line_cells），这里再画一次就会重叠/双点。
    highlight=True 时给数字加**黄底**（置信度偏低，见 _is_low_conf）。
    减时线不由这里画（见 _draw_note_beams）。
    """
    mk = item.get("markup") or {"beams": 0, "dotted": False, "extend": 0}
    beams = int(mk.get("beams", 0))
    dotted = bool(mk.get("dotted", False))

    if item.get("rest"):
        label, weight = "0", "normal"
    else:
        n = item.get("n")
        label, weight = (str(n) if n is not None else "?"), "bold"
    # ⚠ 黄底必须**是一个独立的衬底**，不能挂在数字自己的 bbox 上：
    # `Text.draw()` 是**在文字自己的 zorder（3）里**顺手画 bbox patch 的
    # （见 matplotlib/text.py），patch 的 `set_zorder()` 完全不起作用，
    # 于是黄底永远盖在减时线/音区点/延时线（zorder=2）上面（用户 2026-09-18、
    # 2026-09-20 两次报的"黄底把下划线遮住了"）。这里改成一个 glyph 不可见的
    # 同款文字（`color="none"` 只隐去笔画、bbox 照画）放在 zorder=1，
    # 墨迹全部压在它之上。
    if highlight:
        ax.text(x, y, label, ha="center", va="center", fontsize=15, fontweight=weight,
                color="none", zorder=1,
                bbox=dict(boxstyle="square,pad=0.22", facecolor=CONF_HL_COLOR,
                          edgecolor="none"))
    ax.text(x, y, label, ha="center", va="center",
            fontsize=15, fontweight=weight, color="black")
    if not item.get("rest"):
        o = int(item.get("o", 0))
        if o > 0:
            for i in range(o):
                ax.plot(x, y + 0.44 + i * 0.22, "ko", markersize=3.0)

    # 减时线由 _draw_note_beams 统一绘制
    beam_base = y - 0.34

    # 低音点：放在减时线之下，不与线重叠
    o_value = int(item.get("o", 0))
    if o_value < 0:
        dot_base = beam_base - max(beams, 0) * 0.18 - 0.20
        for i in range(min(-o_value, 3)):
            ax.plot(x, dot_base - i * 0.20, "ko", markersize=3.0)

    if dotted and draw_dot:
        ax.plot(x + 0.36, y - 0.08, "ko", markersize=3.0)

    # 延时线由 _draw_extends 统一绘制（需要知道下一个音符的位置，才能铺满时值区）

    # 歌词：第 1 段在音符正下方，第 2/3 段依次再往下叠（多段歌词，见 ocr_jianpu 的
    # `MAX_LYRIC_VERSES`）。行距取 `LYRIC_LINE_STEP`，与 render() 里算行高用的是同一个数。
    for vi, key in enumerate(("lyric", "lyric2", "lyric3")):
        lyric = item.get(key)
        if lyric:
            ax.text(x, y - 1.15 - vi * LYRIC_LINE_STEP, lyric, ha="center", va="top",
                    fontsize=10, color="0.2", fontfamily=cjk_font or "sans-serif")


def _draw_note_beams(ax, positioned, y):
    """每个音符各画一段独立短线（不连）。

    宽度默认固定（约字符宽的 60%）；若相邻音符过密，则收缩到相邻间距的 80%，
    保证任何情况下都不会与旁边音符的线连成一条。
    """
    total = len(positioned)
    for i, (x, item) in enumerate(positioned):
        beams = int((item.get("markup") or {}).get("beams", 0))
        if beams <= 0:
            continue
        gaps = []
        if i > 0:
            gaps.append(x - positioned[i - 1][0])
        if i + 1 < total:
            gaps.append(positioned[i + 1][0] - x)
        half = 0.22
        if gaps:
            half = min(half, 0.40 * min(gaps))
        base = y - 0.34
        for level in range(beams):
            yy = base - level * 0.18
            ax.plot([x - half, x + half], [yy, yy], "k-", lw=1.2)


def _draw_beam_groups(ax, positioned, y):
    """（保留）按连音符组画连续长线；当前默认不用，改由 _draw_note_beams 逐音画。"""
    _draw_note_beams(ax, positioned, y)


def _draw_extends(ax, x, y, item, next_x=None, obstacles=None, right_reserve=0.35,
                  mapper=None):
    """延时线：**优先用原图里每条 `-` 的实际位置**（识别阶段已记录在 ext_xs 里）。

    这样最忠实——不再靠"推算区间 + 等分居中"（那种做法在区间很宽时，会把
    单独一条 `-` 摆在正中间，导致两边各留一片空白）。
    没有 ext_xs 时（如 MusicXML 路径）才退回"按区间等分、各自居中"。
    """
    extend = int((item.get("markup") or {}).get("extend", 0))
    if extend <= 0:
        return
    ext_xs = item.get("ext_xs") or []
    if mapper is not None and ext_xs:
        for ox in ext_xs:
            cx = mapper(ox)
            ax.plot([cx - 0.21, cx + 0.21], [y, y], "k-", lw=1.4)
        return

    left = x + 0.32
    if next_x is not None and next_x - right_reserve > left + 0.2:
        right = next_x - right_reserve
    else:
        right = left + extend * 0.62
    for ox in (obstacles or []):
        if x + 0.15 < ox < right:
            right = min(right, ox - right_reserve)
    slot = max((right - left) / extend, 0.15)
    dash = min(0.42, slot * 0.7)      # 线长一致，但不超出自己的格子
    for i in range(extend):
        center = left + (i + 0.5) * slot
        ax.plot([center - dash / 2.0, center + dash / 2.0], [y, y], "k-", lw=1.4)


def _paren_after_offset(item):
    """右括号相对音符的横向偏移：若这个音带延时线，`)` 要排在**横线之后**
    （原谱是 `1 - ) 0`，不是 `1 ) - 0`）。
    """
    ext = int((item.get("markup") or {}).get("extend", 0))
    return 0.62 + (ext * 0.62 if ext else 0.0)


def _draw_parens(ax, x, y, item, right_limit=None):
    """画该音符的括号：识别阶段已把括号记在 paren_before / paren_after 上。

    right_limit = 下一条小节线的位置：`)` 随延时线后移时**不能越过小节线**
    （否则会跑到 `|`/`‖` 后面去，实测曲名3 的末两行就是这样）。
    """
    if item.get("paren_before"):
        ax.text(x - 0.62, y, "(", ha="center", va="center", fontsize=15, color="black")
    if item.get("paren_after"):
        off = _paren_after_offset(item)
        if right_limit is not None:
            room = right_limit - x - 0.30
            if room > 0.62:
                off = min(off, room)
            else:
                off = 0.62
        ax.text(x + off, y, ")", ha="center", va="center", fontsize=15, color="black")


def _draw_parens_at(ax, x, y, glyph="("):
    """在**括号自己那一格**的中心画 `(` 或 `)`。

    OCR 路径的排列是「一个字符一格」，括号已经算在字符序列里（见 _line_cells），
    所以这里只需在自己格子中间画出来——不像 _draw_parens 那样要相对音符做偏移，
    也不用担心 `)` 越过小节线（小节线也在它自己的格里，天然不会被压到）。
    """
    ax.text(x, y, glyph, ha="center", va="center", fontsize=15, color="black")


def _draw_dot_at(ax, x, y):
    """在**附点自己那一格**的中心画附点（数字右侧、基线稍下的那个小圆点）。"""
    ax.plot(x, y - 0.08, "ko", markersize=3.0)


def _next_bar(x, bar_xs):
    """返回 x 之后的第一条小节线位置（没有则 None）。"""
    for _mi, bx in bar_xs:
        if bx > x:
            return bx
    return None


def _draw_repeat_marks(ax, x, y, after=None, before_next=False, is_last=False):
    """画反复/终止记号：两条与小节线**等高同粗**的竖线 + 侧边两个点。

    :‖ → 点在线左侧；‖: → 点在线右侧；‖（双线/终止）→ 无点。
    反复记号**取代**普通小节线（否则会画出"| 再加 :‖"的重复标记）。
    """
    heavy = is_last or after in ("backward", "double")
    for dx in (0.0, 0.15):
        ax.plot([x + dx, x + dx], [y - 0.95, y + 0.95], color="0.25",
                lw=1.6 if heavy else 1.2)
    if after == "backward":
        for dy in (0.30, -0.30):
            ax.plot(x - 0.22, y + dy, "ko", markersize=3.4)
    if before_next:
        for dy in (0.30, -0.30):
            ax.plot(x + 0.38, y + dy, "ko", markersize=3.4)


def _line_cells(line_measures):
    """把一行**展开成「字符格子」序列**：识别出来的每个字符各占一格。

    顺序严格照原谱从左到右的书写顺序：

        (  1  .  -  -  )  |  0  ...
        │  │  │  │  │  │  │
        │  │  │  │  │  │  └ 小节线（含反复/终止记号）
        │  │  │  │  │  └──── `)` 右括号
        │  │  │  └──┴─────── 每条延时线 `-` 各占一格
        │  │  └───────────── 附点 `.`
        │  └──────────────── 数字（休止 0 同理）
        └─────────────────── `(` 左括号

    两条规则要分清楚：

    * **横向排列的字符**（数字、`.`、每条 `-`、`(`、`)`、`|`）→ 各占一格。
      这是"平均分配每个字符占用的空间"的对象。
    * **纵向贴在数字上/下的记号**（减时线、音区点）→ **不占横向格子**，
      仍画在数字自己那一列的上下方。它们不是横向的第 N 个字符，给它们
      分格子会变成"点跑到数字旁边去"，反而不像谱。

    返回 [{"kind":..., "item":..., "mi":...}, ...]
      kind ∈ digit / dot / ext / paren_before / paren_after / bar
      mi   = 该字符所属小节在 line_measures 里的下标（用 list.index 会因为
            两个小节内容相同而取错，所以显式带上）。
    """
    cells = []
    for mi, measure in enumerate(line_measures):
        for item in measure.get("notes", []):
            mk = item.get("markup") or {}
            if item.get("paren_before"):
                cells.append({"kind": "paren_before", "item": item, "mi": mi})
            cells.append({"kind": "digit", "item": item, "mi": mi})
            if mk.get("dotted"):
                cells.append({"kind": "dot", "item": item, "mi": mi})
            for _ in range(int(mk.get("extend", 0) or 0)):
                cells.append({"kind": "ext", "item": item, "mi": mi})
            if item.get("paren_after"):
                cells.append({"kind": "paren_after", "item": item, "mi": mi})
        cells.append({"kind": "bar", "item": measure, "mi": mi})
    return cells


def _cell_positions(n_cells, line_width):
    """n 个字符格子**平均分配**一行宽度，返回每格中心的 x（格宽 = 行宽 / 格数）。"""
    n = max(int(n_cells), 1)
    pitch = line_width / n
    return [(i + 0.5) * pitch for i in range(n)]


def draw_cell_line(ax, line_measures, y, draw_digit, n_lines, li, bar_out=None):
    """按「字符格子」把**一整行**画出来，返回只含**数字格**的 `positioned`。

    **数字谱与字母谱共用这一份实现**——两张图要一一对应，排版就只能有一处定义，
    否则两边迟早不一致。`draw_digit(ax, x, y, item)` 由调用方给：
    数字谱画数字（附点自成一格、低置信加黄底），字母谱画映射2 的字母。

    `positioned` 只含数字格，供连音弧 `_draw_slurs()` 与上方记号 `_draw_upper_marks()` 定位。
    `bar_out`（可选，传一个 list）会被填成 `[(小节号, 小节线 x), ...]`：房子横线的
    两端要**吸到小节线**上（用户 2026-09-21 要求），而小节线的位置只有这里知道。
    """
    cells = _line_cells(line_measures)
    positions = _cell_positions(len(cells), OCR_LINE_WIDTH)
    positioned = []
    for cell, ex in zip(cells, positions):
        kind, obj = cell["kind"], cell["item"]
        if kind == "digit":
            draw_digit(ax, ex, y, obj)
            positioned.append((ex, obj))
        elif kind == "dot":
            _draw_dot_at(ax, ex, y)
        elif kind == "ext":
            _draw_ext_line(ax, ex, y)
        elif kind == "paren_before":
            _draw_parens_at(ax, ex, y, "(")
        elif kind == "paren_after":
            _draw_parens_at(ax, ex, y, ")")
        elif kind == "bar":
            # 用 cell["mi"] 而不是 list.index(obj)：内容相同的两个小节会让 index 取错
            idx = cell["mi"]
            if bar_out is not None:
                bar_out.append((idx, ex))
            is_last = (li == n_lines - 1 and idx == len(line_measures) - 1)
            nxt_m = line_measures[idx + 1] if idx + 1 < len(line_measures) else None
            after = obj.get("repeat_after")
            fwd = (bool(nxt_m and nxt_m.get("repeat_before") == "forward")
                   or after == "forward")
            if after in ("backward", "double", "forward") or fwd:
                _draw_repeat_marks(ax, ex, y, after, fwd, is_last)   # 取代小节线
            else:
                ax.plot([ex, ex], [y - 0.95, y + 0.95],
                        color="0.45", lw=1.6 if is_last else 1.0)
    return positioned


def _draw_ext_line(ax, x, y):
    """在给定格子的中心画一条延时线横线（长度固定、大小一致）。"""
    ax.plot([x - 0.21, x + 0.21], [y, y], "k-", lw=1.4)


def _draw_upper_marks(ax, positioned, y, cjk_font=None):
    """画音符**正上方**的那一行记号：`>` 重音、`v` 换气、`³` 三连音、房子标号 `1./2./3.`。

    尺寸按《上海滩》实测换算（H = 数字高，数字在数据单位里约高 0.5）：
      `>`  0.85H×0.64H  → 半宽 0.21、半高 0.16，尖朝右
      `v`  0.45H×0.57H  → 半宽 0.11、半高 0.14，尖朝下
      `³`  0.45H×0.66H  → 小字号文字（实测约 0.66 倍数字高）
      房子 0.53H×0.77H  → 小字号文字 + `.`
    基线按该音的音区点数抬高，免得压住高音点；同一音上有多个记号时**依次往上叠**。
    **位置分两种**（与原谱一致）：
      `>` 重音、`v` 换气 → 画在音符**正上方**；
      `³` 三连音 → 画在音符**左上角**（原谱是 `³2` 那种写法）；
      房子标号 → 画在音符**右上角**（用户 2026-09-18 指定）。
    """
    for x, item in positioned:
        kinds = []
        if item.get("accent"):
            kinds.append("accent")
        if item.get("breath"):
            kinds.append("breath")
        if item.get("triplet"):
            kinds.append("triplet")
        if item.get("house"):
            kinds.append("house")
        if not kinds:
            continue
        max_o = max(int(item.get("o", 0) or 0), 0)
        base0 = y + 0.60 + 0.22 * max_o
        for k, kind in enumerate(kinds):
            base = base0 + 0.36 * k          # 多个记号往上叠，避免互相压
            if kind == "accent":             # 尖朝右的 V（正上方）
                ax.plot([x - 0.21, x + 0.21], [base + 0.16, base], "k-", lw=1.2)
                ax.plot([x - 0.21, x + 0.21], [base - 0.16, base], "k-", lw=1.2)
            elif kind == "breath":           # 尖朝下的 v（正上方）
                ax.plot([x - 0.11, x], [base + 0.14, base - 0.14], "k-", lw=1.2)
                ax.plot([x + 0.11, x], [base + 0.14, base - 0.14], "k-", lw=1.2)
            elif kind == "triplet":
                # 左上角：原谱写成 `³2`，小 3 贴在数字左上方
                ax.text(x - 0.32, base + 0.06, "3", ha="center", va="center",
                        fontsize=9, color="black",
                        fontfamily=cjk_font or "sans-serif")
            elif kind == "house":
                # 右上角（用户 2026-09-18 指定）：原谱的 `[2.` 就贴在房子横线左端内侧
                ax.text(x + 0.32, base + 0.12, f"{item['house']}.", ha="center",
                        va="center", fontsize=9, color="black",
                        fontfamily=cjk_font or "sans-serif")


def _draw_slur_halves(ax, positioned, y):
    """画**跨行弧的残段**：行末那截从该音伸到行右边，行首那截从行左边伸到该音。

    原谱的连音弧跨行时被行断开，每行只剩半截（用户 2026-09-18 选定"两段都画"）。
    模型里记成 `slur_left` / `slur_right`（**开放端**，没有配对的另一半）。

    形状上做了一点处理，让两段拼起来读着像一条弧：
      `slur_right`（行末那截）→ 画**上升**的半弧（峰在下一行）；
      `slur_left` （行首那截）→ 画**下降**的半弧（从左边界的高处落到该音）。
    基线与拱高算法同 `_draw_slurs()`，只是另一端延伸到行边界。
    """
    for x, item in positioned:
        max_o = max(int(item.get("o", 0) or 0), 0)
        base = y + 0.62 + 0.22 * max_o
        for side in ("slur_right", "slur_left"):
            if not item.get(side):
                continue
            x0, x1 = (x, OCR_LINE_WIDTH) if side == "slur_right" else (0.0, x)
            span = x1 - x0
            if span <= 0.05:
                continue
            rise = min(0.34, 0.16 * span)
            margin = min(0.12, 0.12 * span)
            xs = np.linspace(x0 + margin, x1 - margin, 24)
            t = (xs - (x0 + margin)) / max(span - 2 * margin, 1e-6)
            if side == "slur_right":
                ys = base + rise * np.sin(np.pi * t / 2.0)       # 上升
            else:
                ys = base + rise * np.sin(np.pi * (1.0 + t) / 2.0)  # 下降
            ax.plot(xs, ys, color="0.25", lw=1.1)


def _draw_house_brackets(ax, positioned, y, bar_xs=None):
    """画**房子横线**（`1.`/`2.`/`3.` 段落上方那条横线，两端带向下的短竖钩）。

    端点由 OCR 存在 `house_start` / `house_stop` 上 —— 与连音弧同一套思路：
    **模型里只存端点，画的时候再由这两个音的位置连线**。
    高度放在所有"上方记号"（从 `y+0.60` 起）**之上**；标号由 `_draw_upper_marks()`
    画在左端下方，正好对应原谱 `[1.` 的写法。

    ⚠ **两端吸到小节线上，但要留缝**（用户 2026-09-21）：原谱的房子横线是从**小节线**拉到
    **小节线**的（`⌐|———|⌐`），不是从数字拉到数字。模型里只存了"覆盖哪几个音"，
    所以这里拿小节线位置把两端外扩到边界；`bar_xs` 为 None 或找不到边界时，
    退回原来"数字左右各留 0.30"的画法。
    端点和竖钩再**向内缩 `HOUSE_END_GAP`** —— 正好落在小节线上时，竖钩会和那条竖线
    连成"T"字形（用户原话：不要把房子横线和小节线连在一起）。
    """
    xs = sorted({float(bx) for _mi, bx in (bar_xs or [])})
    for i, (x0, item) in enumerate(positioned):
        if not item.get("house_start"):
            continue
        end = next((j for j in range(i + 1, len(positioned))
                    if positioned[j][1].get("house_stop")), None)
        if end is None:
            continue                      # 只有起点没有终点 → 数据不全，不画
        x1 = positioned[end][0]
        bars_before = [b for b in xs if b < x0 - 1e-6]
        bars_after = [b for b in xs if b > x1 + 1e-6]
        left = max(bars_before) + HOUSE_END_GAP if bars_before else x0 - 0.30
        right = min(bars_after) - HOUSE_END_GAP if bars_after else x1 + 0.30
        span = [it for _x, it in positioned[i:end + 1]]
        max_o = max([int(it.get("o", 0) or 0) for it in span] or [0])
        top = y + 1.00 + 0.22 * max(max_o, 0)
        hook = 0.35                        # 两端向下的竖钩长度
        ax.plot([left, right], [top, top], "k-", lw=1.2)
        ax.plot([left, left], [top, top - hook], "k-", lw=1.2)
        ax.plot([right, right], [top, top - hook], "k-", lw=1.2)


def _slur_chains(positioned):
    """把「相邻两音连音」按**弧编号**聚成链，返回 [(起点下标, 终点下标), ...]。

    数据模型里连音**只存相邻对**（`slur_start` / `slur_stop` 用同一编号配对）——
    一条跨 3 个音的长弧在模型里是两对（A-B、B-C），**两对用的是同一个编号**，
    画图时在这里拼回**一条**弧，与原谱一致。

    ⚠ **必须按编号分组，不能只看"相邻性"**：两条**独立的**弧常常首尾相接
    （A-B 与 B-C，共用音 B），但**编号不同**。早先这里是"先看相邻两音有没有共同
    编号、再按相邻性把连续的 True 串成一条链"——**从不检查相邻两对是不是同一条弧**，
    于是两条独立弧被串成一条长弧。用户 2026-09-18 报的"连在一起"就是这个：
    只加编号没用，**拼链这一步必须真的去读编号**。
    """
    groups = {}
    for i in range(len(positioned) - 1):
        a_marks = set(positioned[i][1].get("slur_start") or [])
        b_marks = set(positioned[i + 1][1].get("slur_stop") or [])
        for sid in a_marks & b_marks:
            lo, hi = groups.get(sid, (i, i + 1))
            groups[sid] = (min(lo, i), max(hi, i + 1))
    return sorted(groups.values())


def _draw_slurs(ax, positioned, y):
    """画连音弧 `⌒`：一条链画成**一条**长弧，位于音符上方。

    高度要避开音区点：高音点在 `y + 0.44 + i * 0.22`，所以弧线的基线按链上
    最高的音区点数抬高，否则弧会压在点上面。

    **嵌套弧**（长弧上面再套一条短弧）还要**分层**：包含别人的那条整体再抬高一级。
    都在同一个基线上画的话，短弧的拱高会超过长弧在那一处的高度 → 两条弧交叉成"X"。
    实测《上海滩》行3 `2 6 6`（长弧音7..音9 + 嵌套短弧音8..音9）：不分层就没法看。
    原谱是同心嵌套的，外层更高。
    """
    chains = _slur_chains(positioned)
    for start, end in chains:
        x0 = positioned[start][0]
        x1 = positioned[end][0]
        span = x1 - x0
        if span <= 0.05:
            continue
        chain = [positioned[k][1] for k in range(start, end + 1)]
        max_o = max([int(it.get("o", 0) or 0) for it in chain] or [0])
        # 被几条弧**套在里面**就抬高几级，一级 = 音区点那一档（0.22）。
        # 判据是"被包含"（起止都落在自己区间内），共用一个端点也算。
        lift = 0.22 * sum(1 for a, b in chains
                          if a >= start and b <= end and (a, b) != (start, end))
        base = y + 0.62 + 0.22 * max(max_o, 0) + lift
        rise = min(0.34, 0.16 * span)
        margin = min(0.12, 0.12 * span)
        xs = np.linspace(x0 + margin, x1 - margin, 24)
        t = (xs - (x0 + margin)) / max(span - 2 * margin, 1e-6)
        ax.plot(xs, base + rise * np.sin(np.pi * t), color="0.25", lw=1.1)


def render(measures, out_path, title="", key_mark="", meter="",
           measures_per_line=4, dpi=150):
    """渲染 note 流为中文数字谱图片（自动换行）。"""
    cjk_font = _find_cjk_font()
    if cjk_font:
        plt.rcParams["font.sans-serif"] = [cjk_font]
    plt.rcParams["axes.unicode_minus"] = False

    # 分行：OCR 路径优先保留原谱 row；MusicXML 路径按指定小节数换行。
    lines = []
    if measures and all(m.get("row") is not None for m in measures):
        current_row = None
        current = []
        for measure in measures:
            if measure["row"] != current_row:
                if current:
                    lines.append(current)
                current_row = measure["row"]
                current = []
            current.append(measure)
        if current:
            lines.append(current)
    else:
        for i in range(0, len(measures), measures_per_line):
            lines.append(measures[i:i + measures_per_line])
    if not lines:
        lines = [[]]

    n_lines = len(lines)
    # 每行的 y **按行累加**，不用固定的 `li * LINE_GAP` —— 多段歌词的行要占更多高度
    # （每多一段多 `LYRIC_LINE_STEP`），否则 3 段词会压到下一行的数字上
    # （1.jpg《上海滩》三段词实测三行串在一起）。
    line_top = []
    _cur = 0.0
    for _lm in lines:
        line_top.append(_cur)
        _extra = LYRIC_LINE_STEP * (_verse_count(
            [item for m in _lm for item in m.get("notes", [])]) - 1)
        _cur += LINE_GAP + _extra
    fig_h = 1.4 + 0.397 * _cur            # 0.397 = 1.15/2.9，与原公式（每行 1.15）同比例
    fig, ax = plt.subplots(figsize=(16, fig_h))
    ax.axis("off")

    ocr_mode = bool(measures and all(m.get("row") is not None for m in measures))
    max_bar_x = 0.0     # 各行小节线映射后的最右位置（用于决定画布右边界）

    for li, line_measures in enumerate(lines):
        y = -line_top[li]
        line_notes = [item for m in line_measures for item in m.get("notes", [])]

        if ocr_mode and line_notes and all(item.get("cx") is not None for item in line_notes):
            # **字符格子排布**：把一行展开成识别出来的字符序列（`( 数字 . - - ) | …`），
            # 每个字符平均分配一格、格内居中。实现抽到了 `draw_cell_line()`，
            # **与字母谱共用同一份**——两张图要一一对应，排版就只能有一处定义。
            # 减时线与音区点仍贴在数字那一列（见 _line_cells）。
            cell_bar_xs = []            # 房子横线两端要吸到小节线上，这里收集小节线位置
            positioned = draw_cell_line(
                ax, line_measures, y,
                lambda a, x, yy, it: _draw_note(a, x, yy, it, cjk_font,
                                                draw_dot=False,   # 附点自成一格
                                                highlight=_is_low_conf(it)),
                n_lines, li, bar_out=cell_bar_xs)
            _draw_beam_groups(ax, positioned, y)
            _draw_slurs(ax, positioned, y)
            _draw_slur_halves(ax, positioned, y)
            _draw_upper_marks(ax, positioned, y, cjk_font)
            _draw_house_brackets(ax, positioned, y, cell_bar_xs)
            if line_measures and line_measures[0].get("repeat_before") == "forward":
                _draw_repeat_marks(ax, 0.35, y, None, True, False)
        else:
            # MusicXML / 网站 JSON 路径：按实际时值排版。
            # 先排出音符位置、每音所在小节的结束位置、以及小节线位置，
            # 再统一绘制（括号和延时线都要据此避让，否则 `)` 会越过小节线）。
            positioned = []
            limits = []
            bar_xs = []
            t = 0.0
            for mi, measure in enumerate(line_measures):
                m_notes = measure.get("notes", [])
                total_q = sum(max(float(it.get("q", 1.0) or 1.0), 1e-6) for it in m_notes)
                m_end = (t + total_q) * X_PER_QUARTER
                for item in m_notes:
                    q = float(item.get("q", 1.0) or 1.0)
                    if q <= 0:
                        q = 1.0
                    positioned.append(((t + q / 2.0) * X_PER_QUARTER, item))
                    limits.append(m_end)
                    t += q
                bar_xs.append((mi, m_end))

            for idx, (xc, item) in enumerate(positioned):
                _draw_note(ax, xc, y, item, cjk_font)
                _draw_parens(ax, xc, y, item, limits[idx])
            _draw_beam_groups(ax, positioned, y)
            _draw_slurs(ax, positioned, y)
            _draw_slur_halves(ax, positioned, y)
            _draw_upper_marks(ax, positioned, y, cjk_font)
            _draw_house_brackets(ax, positioned, y, bar_xs)

            obs = []
            for ox, oitem in positioned:
                if oitem.get("paren_before"):
                    obs.append(ox - 0.62)
                if oitem.get("paren_after"):
                    obs.append(ox + _paren_after_offset(oitem))
            for i, (px, item) in enumerate(positioned):
                nxt = positioned[i + 1][0] if i + 1 < len(positioned) else None
                own = []
                if item.get("paren_before"):
                    own.append(px - 0.62)
                if item.get("paren_after"):
                    own.append(px + _paren_after_offset(item))
                # 避让小節线与**其他音符**的括号；不避让自己的括号
                _draw_extends(ax, px, y, item, nxt,
                              [bx for _m, bx in bar_xs] + [o for o in obs if o not in own])

            for mi, xe in bar_xs:
                measure = line_measures[mi]
                is_last = (li == n_lines - 1 and mi == len(line_measures) - 1)
                nxt_m = line_measures[mi + 1] if mi + 1 < len(line_measures) else None
                after = measure.get("repeat_after")
                fwd = (bool(nxt_m and nxt_m.get("repeat_before") == "forward")
                       or after == "forward")
                if after in ("backward", "double", "forward") or fwd:
                    _draw_repeat_marks(ax, xe, y, after, fwd, is_last)  # 取代小节线
                else:
                    ax.plot([xe, xe], [y - 0.95, y + 0.95],
                            color="0.45", lw=1.6 if is_last else 1.0)
            if line_measures and line_measures[0].get("repeat_before") == "forward":
                _draw_repeat_marks(ax, 0.35, y, None, True, False)

    header = "  ".join([p for p in (title, key_mark, meter) if p])
    if header:
        ax.text(0.0, 1.6, header, fontsize=13, va="bottom", ha="left",
                color="0.15", fontfamily=cjk_font or "sans-serif")

    # x 范围：OCR 沿用原图位置（并把右边界扩展到能容纳行末小节线，
    # 它们来自原图坐标、可能落在最后一个音符右侧）；MusicXML 按时值取最长行。
    if ocr_mode:
        max_x = max(OCR_LINE_WIDTH, max_bar_x + 0.8)
    else:
        max_x = 0.0
        for line_measures in lines:
            t = sum(float(it.get("q", 1.0) or 1.0)
                    for m in line_measures for it in m.get("notes", []))
            max_x = max(max_x, t * X_PER_QUARTER)

    ax.set_xlim(-0.6, max_x + 0.8)
    ax.set_ylim(-line_top[-1] - 1.9, 2.2)

    plt.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path
