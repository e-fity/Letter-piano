# -*- coding: utf-8 -*-
"""
note 流 → 映射2 字母谱文本行 + XML 输出。

字母谱文本约定（与网站一致）：
  大写字母 = 键盘键位（映射2）
  `_`      减时线（附在字母后，条数 = 时值减半次数）
  ` ·`     附点
  ` —`     延时线（条数 = 延长拍数）
  `0`      休止
  `⌒`      连音
  `|` `||` `||:` `:||`  小节线 / 反复
  `↑`/`↓`  Shift 扩展音域（映射2 的 ±2/±3 八度，↑=左Shift升一层，↑↑=右Shift升两层，
           ↓=左Shift降一层，↓↓=右Shift降两层）
"""

from . import mapping as mp

# 重复/终止小节线的文本后缀
_DOUBLE_BARLINES = ("double", "final", "light-heavy", "heavy-light")


def note_text(item):
    """单个 note 的字母谱文本（不含歌词）。"""
    mk = item.get("markup") or {"beams": 0, "dotted": False, "extend": 0}
    n = item.get("n")
    if item.get("rest"):
        base = "0"
    elif n is None or not (1 <= int(n) <= 7):
        base = "?"
    else:
        key, mod = mp.key_for(int(n), item.get("o", 0), "mapping2")
        base = key
        if mod == mp.LEFT_SHIFT:
            base += "↑" if item["o"] > 0 else "↓"
        elif mod == mp.RIGHT_SHIFT:
            base += "↑↑" if item["o"] > 0 else "↓↓"
    s = base + "_" * mk["beams"]
    if mk["dotted"]:
        s += " ·"
    if item.get("accent"):
        s += ">"                      # 重音记号（原谱是数字上方的小 V）
    if item.get("breath"):
        s += ","                      # v 换气记号（用 `,` 表示，避开 `v`＝低音点）
    if item.get("triplet"):
        s += "t"                      # ³ 三连音
    if item.get("house"):
        s += f"[{item['house']}]"     # 房子标号 1./2./3.
    s += " —" * mk["extend"]
    return s


def _slur_between(a, b):
    """相邻两个音符之间是否存在连音（⌒）。"""
    return bool(set(a.get("slur_start", [])) & set(b.get("slur_stop", [])))


def measure_text(measure):
    """单个小节的字母谱文本（含括号、连音，不含小节线）。"""
    notes = measure["notes"]
    parts = []
    for i, item in enumerate(notes):
        if item.get("paren_before"):
            parts.append("(")
        parts.append(note_text(item))
        if item.get("paren_after"):
            parts.append(")")
        if i < len(notes) - 1 and _slur_between(item, notes[i + 1]):
            parts.append("⌒")
    return " ".join(parts)


def _barline_sep(barline_after):
    """两个相邻小节之间的分隔（由左侧小节的 barline_after 决定）。"""
    if barline_after == "backward":
        return " :|| "
    if barline_after == "forward":
        return " ||: "
    if barline_after in _DOUBLE_BARLINES:
        return " || "
    return " | "  # single / None


def _join_segments(buf):
    parts = []
    for j, (seg, _after) in enumerate(buf):
        if j > 0:
            parts.append(_barline_sep(buf[j - 1][1]))
        parts.append(seg)

    last_after = buf[-1][1]
    if last_after == "backward":
        parts.append(" :||")
    elif last_after == "forward":
        parts.append(" ||:")
    elif last_after in _DOUBLE_BARLINES:
        parts.append(" ||")
    return "".join(parts)


def measures_to_lines(measures, measures_per_line=2):
    """note 流 → 字母谱文本行列表。

    若小节带 "row" 字段（OCR 图片路径），**按原谱行分组**：一个原谱行 = 一行字母谱，
    不按 measures_per_line 拆 —— 图片输出要跟原谱行一一对应（用户 2026-09-22 明确：
    "我的输出要按行来，而不是默认 1 行几节"）。
    否则（MusicXML / 网站 JSON，**没有行信息**）按 measures_per_line 折行。
    """
    if not measures:
        return [""]

    groups = []
    if all(m.get("row") is not None for m in measures):
        # OCR 路径：每个原谱行 = 一行字母谱（不按 measures_per_line 拆）
        cur_row, cur = None, []
        for m in measures:
            if m["row"] != cur_row:
                if cur:
                    groups.append(cur)
                cur_row, cur = m["row"], []
            cur.append(m)
        if cur:
            groups.append(cur)
    else:
        # 无行信息：按 measures_per_line 折行（与 `textscore._group_by_row` 同一口径）。
        # 原来这里是 `groups = [measures]`：docstring 与 `selftest.py` 的期望都写着
        # "按 measures_per_line 折行"，实现却整首一行 → selftest 长期是红的（用户 2026-09-22 发现）。
        # 图片路径走上面那支、且 20 张已逐字节 A/B 验证零影响，所以这个修正只动 MusicXML 输出。
        step = max(int(measures_per_line), 1)
        groups = [measures[i:i + step] for i in range(0, len(measures), step)]

    lines = []
    for grp in groups:
        # 整组拼成一行；组可能很长，但保持原谱行结构
        buf = []
        for i, m in enumerate(grp):
            seg = ("||: " if m.get("repeat_before") == "forward" else "") + measure_text(m)
            buf.append((seg, m.get("repeat_after")))
        lines.append(_join_segments(buf))
    return lines


def to_xml(title, key_mark, meter, lines, mapping="mapping2"):
    """把字母谱文本行封装为 XML 字符串（保留字母谱文本行）。"""
    import xml.etree.ElementTree as ET
    from xml.dom import minidom

    root = ET.Element("letter-score", {
        "mapping": mapping,
        "key": key_mark,
        "meter": meter,
    })
    if title:
        t = ET.SubElement(root, "title")
        t.text = title
    for i, line in enumerate(lines, 1):
        ln = ET.SubElement(root, "line", {"n": str(i)})
        ln.text = line

    raw = ET.tostring(root, encoding="unicode")
    # 美化缩进
    pretty = minidom.parseString(raw).toprettyxml(indent="  ")
    # toprettyxml 会插入 XML 声明；去掉多余空行
    return pretty
