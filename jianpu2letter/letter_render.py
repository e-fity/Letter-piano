# -*- coding: utf-8 -*-
"""把 note 流渲染成「键盘字母简谱」图片。

字母 = 映射2 键位；音区用上/下加点表示；Shift 音区（±2/±3）另加 ↑/↓ 标记。
减时线、附点、延时线、小节线、连音与数字谱一致。
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

from . import mapping as mp
from .jianpu_render import (_draw_beam_groups, _draw_extends, _draw_house_brackets,
                            _draw_parens, _draw_repeat_marks, _draw_slur_halves,
                            _draw_slurs, _draw_upper_marks, _next_bar,
                            _paren_after_offset, draw_cell_line)

CJK_FONT_CANDIDATES = [
    "Noto Sans CJK SC", "Noto Sans CJK", "Source Han Sans SC", "Source Han Sans CN",
    "Microsoft YaHei", "SimHei", "PingFang SC", "WenQuanYi Zen Hei",
    "Arial Unicode MS", "Heiti SC",
]

X_PER_QUARTER = 1.6
OCR_LINE_WIDTH = 64.0
LINE_GAP = 2.9
# 多段歌词：每多一段，该行再额外占这么高（与 `_draw_note` 里画歌词的行距是同一个数）。
LYRIC_LINE_STEP = 1.05


def _verse_count(notes):
    """这一行要画几段歌词（1~3）：看有没有 `lyric2` / `lyric3`。"""
    if any(n.get("lyric3") for n in notes):
        return 3
    if any(n.get("lyric2") for n in notes):
        return 2
    return 1


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


def _draw_letter(ax, x, y, item, cjk_font):
    mk = item.get("markup") or {}
    beams = int(mk.get("beams", 0))
    dotted = bool(mk.get("dotted", False))
    n = item.get("n")
    o = int(item.get("o", 0))

    if item.get("rest"):
        label = "0"
    elif n is None or not (1 <= int(n) <= 7):
        label = "?"
    else:
        key, _mod = mp.key_for(int(n), o, "mapping2")
        label = key

    ax.text(x, y, label, ha="center", va="center", fontsize=15,
            fontweight="bold", color="black", family="monospace")

    # 音区：高音在上加点（间距按标准收紧）；±2/±3 另标 ⇧/⇩（需按 Shift）
    if o > 0:
        for i in range(min(o, 3)):
            ax.plot(x, y + 0.44 + i * 0.22, "ko", markersize=3.0)
    if abs(o) >= 2:
        ax.text(x + 0.30, y + 0.58, "⇧" if o > 0 else "⇩",
                ha="left", va="bottom", fontsize=8, color="0.35",
                fontfamily=cjk_font or "sans-serif")

    # 减时线由 _draw_note_beams 统一绘制
    beam_base = y - 0.34

    # 低音点：放在减时线之下，不与线重叠
    if o < 0:
        dot_base = beam_base - max(beams, 0) * 0.18 - 0.20
        for i in range(min(-o, 3)):
            ax.plot(x, dot_base - i * 0.20, "ko", markersize=3.0)

    if dotted:
        ax.plot(x + 0.36, y - 0.08, "ko", markersize=3.0)
    # 延时线由 _draw_extends 统一绘制（需知道下一个音符位置，才能铺满时值区）

    # 歌词：画在字母正下方（与数字谱一致）；多段依次往下叠，行距同 jianpu_render
    for vi, key in enumerate(("lyric", "lyric2", "lyric3")):
        lyric = item.get(key)
        if lyric:
            ax.text(x, y - 1.15 - vi * LYRIC_LINE_STEP, lyric, ha="center", va="top",
                    fontsize=10, color="0.2", fontfamily=cjk_font or "sans-serif")


def render(measures, out_path, title="", key_mark="", meter="",
           measures_per_line=4, dpi=150):
    cjk_font = _find_cjk_font()
    if cjk_font:
        plt.rcParams["font.sans-serif"] = [cjk_font]
    plt.rcParams["axes.unicode_minus"] = False

    ocr_mode = bool(measures and all(m.get("row") is not None for m in measures))
    max_bar_x = 0.0     # 各行小节线映射后的最右位置（决定画布右边界）
    lines = []
    if ocr_mode:
        current_row, current = None, []
        for measure in measures:
            if measure["row"] != current_row:
                if current:
                    lines.append(current)
                current_row, current = measure["row"], []
            current.append(measure)
        if current:
            lines.append(current)
    else:
        for i in range(0, len(measures), measures_per_line):
            lines.append(measures[i:i + measures_per_line])
    if not lines:
        lines = [[]]

    n_lines = len(lines)
    # 每行的 y **按行累加**（同 jianpu_render.render 里的说明）：多段歌词的行要占更多高度，
    # 否则 3 段词会压到下一行的数字上。
    line_top = []
    _cur = 0.0
    for _lm in lines:
        line_top.append(_cur)
        _extra = LYRIC_LINE_STEP * (_verse_count(
            [item for m in _lm for item in m.get("notes", [])]) - 1)
        _cur += LINE_GAP + _extra
    fig, ax = plt.subplots(figsize=(16, 1.4 + 0.397 * _cur))
    ax.axis("off")

    for li, line_measures in enumerate(lines):
        y = -line_top[li]
        line_notes = [item for m in line_measures for item in m.get("notes", [])]
        cells_mode = bool(ocr_mode and line_notes
                          and all(item.get("cx") is not None for item in line_notes))
        if cells_mode:
            # **字符格子排布**：与数字谱共用 `draw_cell_line()` —— 每个字符各占一格、
            # 格宽统一，两张图的字符位置因此一一对应。
            # 先前这里是"按原图 x 坐标映射"（`map_x`），那是"字母谱先只改数字谱"
            # 阶段的遗留；用户 2026-09-18 要求把排版同步过来。
            cell_bar_xs = []        # 房子横线两端要吸到小节线上（与数字谱一致）
            positioned_cells = draw_cell_line(
                ax, line_measures, y,
                lambda a, x, yy, it: _draw_letter(a, x, yy, it, cjk_font),
                n_lines, li, bar_out=cell_bar_xs)
            _draw_beam_groups(ax, positioned_cells, y)
            # 连音弧：用户 2026-09-18 要求也画在字母谱上（先前只画数字谱）。
            # 顺序与数字谱一致：减时线 → 弧 → 重音。
            _draw_slurs(ax, positioned_cells, y)
            _draw_slur_halves(ax, positioned_cells, y)
            _draw_upper_marks(ax, positioned_cells, y, cjk_font)
            _draw_house_brackets(ax, positioned_cells, y, cell_bar_xs)
            if line_measures and line_measures[0].get("repeat_before") == "forward":
                _draw_repeat_marks(ax, 0.35, y, None, True, False)
            # 下面那一段（小节线避让 / 括号偏移 / 延时线区间）是给 MusicXML 路径用的：
            # 格子排布下这些记号各占一格，不再需要偏移与避让。用空的 positioned
            # 让它自然空转，并把小节线绘制也跳过（前面已经画过，见 cells_mode 那个判断）。
            # `map_x` 仍要保留 —— 下面计算 bar_xs 时还会用到它。
            positioned = []
            limits = []
            xs = [float(item["cx"]) for item in line_notes]
            lo, hi = min(xs), max(xs)
            span = max(hi - lo, 1.0)

            def map_x(v):
                return 0.8 + (float(v) - lo) / span * (OCR_LINE_WIDTH - 1.6)
        else:
            positioned = []
            limits = []
            t = 0.0
            for measure in line_measures:
                m_notes = measure.get("notes", [])
                total_q = sum(max(float(it.get("q", 1.0) or 1.0), 1e-6) for it in m_notes)
                m_end = (t + total_q) * X_PER_QUARTER
                for item in m_notes:
                    q = float(item.get("q", 1.0) or 1.0) or 1.0
                    positioned.append(((t + q / 2.0) * X_PER_QUARTER, item))
                    limits.append(m_end)
                    t += q

        # 先算出本行所有小节线位置：延时线要避开它们（否则会有一条 `-` 压在竖线上）
        bar_xs = []
        cursor = 0.0
        for mi, measure in enumerate(line_measures):
            if ocr_mode and measure.get("notes") and measure["notes"][0].get("cx") is not None:
                if measure.get("bar_x") is not None:
                    # 优先用**原谱检测到的真实小节线位置**（推算中点会偏右）
                    xe = map_x(float(measure["bar_x"]))
                elif mi + 1 < len(line_measures) and line_measures[mi + 1].get("notes"):
                    a = max(float(x["cx"]) for x in measure["notes"])
                    b = min(float(x["cx"]) for x in line_measures[mi + 1]["notes"])
                    xe = map_x((a + b) / 2.0)
                else:
                    xe = (0.8 + (OCR_LINE_WIDTH - 1.6)) if ocr_mode else 0.0
            else:
                cursor += sum(float(x.get("q", 1.0) or 1.0)
                              for x in measure.get("notes", []))
                xe = cursor * X_PER_QUARTER
            bar_xs.append((mi, xe))
        if bar_xs:
            max_bar_x = max(max_bar_x, max(bx for _m, bx in bar_xs))
        obstacles = [bx for _m, bx in bar_xs]
        for ox, oitem in positioned:
            if oitem.get("paren_before"):
                obstacles.append(ox - 0.62)
            if oitem.get("paren_after"):
                obstacles.append(ox + _paren_after_offset(oitem))

        # OCR 路径的括号右边界 = 该音之后的第一条小节线（`)` 不能越过它）
        if limits is None:
            limits = [_next_bar(px, bar_xs) for px, _it in positioned]
        for idx, (px, item) in enumerate(positioned):
            _draw_letter(ax, px, y, item, cjk_font)
            _draw_parens(ax, px, y, item, limits[idx] if idx < len(limits) else None)
        _draw_beam_groups(ax, positioned, y)
        # 重音记号 `>` 也画在字母上（用户这次是拿**字母谱图**对比原谱提的需求）。
        # 注：先前"字母谱先只改数字谱"那句是针对**排版规则**（字符格子）说的，
        # 这里是新增记号，不涉及排版；若只想数字谱有，去掉这一行即可。
        _draw_upper_marks(ax, positioned, y, cjk_font)
        _draw_house_brackets(ax, positioned, y, bar_xs)

        for i, (px, item) in enumerate(positioned):
            nxt = positioned[i + 1][0] if i + 1 < len(positioned) else None
            own = []
            if item.get("paren_before"):
                own.append(px - 0.62)
            if item.get("paren_after"):
                own.append(px + _paren_after_offset(item))
            # 避让小節线与**其他音符**的括号；自己的括号本就排在延时线之后，不避让
            _draw_extends(ax, px, y, item, nxt,
                          [o for o in obstacles if o not in own])

        # cells_mode 时小节线已在 draw_cell_line() 里按格子画过，这里跳过，否则会画两遍
        for mi, xe in ([] if cells_mode else bar_xs):
            measure = line_measures[mi]
            is_last = (li == n_lines - 1 and mi == len(line_measures) - 1)
            nxt = line_measures[mi + 1] if mi + 1 < len(line_measures) else None
            after = measure.get("repeat_after")
            fwd = (bool(nxt and nxt.get("repeat_before") == "forward")
                   or after == "forward")
            if after in ("backward", "double", "forward") or fwd:
                _draw_repeat_marks(ax, xe, y, after, fwd, is_last)   # 取代小节线
            else:
                ax.plot([xe, xe], [y - 0.95, y + 0.95],
                        color="0.45", lw=1.6 if is_last else 1.0)
        if line_measures and line_measures[0].get("repeat_before") == "forward":
            _draw_repeat_marks(ax, 0.35, y, None, True, False)

    header = "  ".join([p for p in (title, key_mark, meter) if p])
    if header:
        ax.text(0.0, 1.6, header + "    键盘字母谱（映射2）", fontsize=13,
                va="bottom", ha="left", color="0.15",
                fontfamily=cjk_font or "sans-serif")

    # OCR 路径：把右边界扩展到能容纳行末小节线（它们来自原图坐标，
    # 可能落在最后一个音符右侧，否则会被裁掉）
    # 注：这里原来是 `max_x = (A if ocr_mode else B` —— 外层括号从未闭合，
    # 属语法错误（整个文件都 import 不进来），已拆成 if/else。
    if ocr_mode:
        max_x = max(OCR_LINE_WIDTH, max_bar_x + 0.8)
    else:
        max_x = max([sum(float(i.get("q", 1.0) or 1.0)
                         for m in lm for i in m.get("notes", [])) * X_PER_QUARTER
                     for lm in lines] or [10.0])

    ax.set_xlim(-0.6, max_x + 0.8)
    ax.set_ylim(-line_top[-1] - 1.9, 2.2)
    plt.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path
