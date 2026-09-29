# -*- coding: utf-8 -*-
"""纯文本简谱 → note 流（100% 确定的入口，不经过图像识别）。

文本语法（每行 = 原谱一行；与工具诊断输出同格式，可往返）：
    6 6·6 6 3 5 | 2 2·2 2 3 5
每个记号：
    数字 0-7      音（0=休止）
    ^ / v         高音点 / 低音点（可重复）
    _             减时线（可 1-3 条）
    · 或 .        附点
    >             重音记号（原谱画在数字上方的小 V）
    -             延时线（每条一拍）
    ( )           括号
    ⌒             与前一音连音
    | || :|| ||:  小节线 / 终止线 / 反复结束 / 反复开始
行首可写 "L03:" 之类的前缀，会被忽略。

本模块同时提供**逆运算** `to_text()`（note 流 → 文本），两者可往返：
`parse_text(to_text(measures))` 得到等价的 measures。
"""

import re

_TOKEN_RE = re.compile(r"^([0-7])([\^v]*)(_*)([·.]?)(>*)(-*)$")
_BAR_RE = re.compile(r"^(\|\|\:|:\|\||\|\||\|)$")
# 小节线记号 → measures 里的 repeat_after 取值（review_io 也用这一套映射）
_BAR_KIND = {":||": "backward", "||": "double", "||:": "forward", "|": "single"}


def parse_text(text, key_mark="1=C", meter="4/4", title=""):
    """文本 → measures（与 solfege.build_stream 输出兼容）。"""
    measures = []
    line_no = 0
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        line = re.sub(r"^L\d+[:：]\s*", "", line)  # 去掉 "L03:" 前缀
        line_no += 1

        current = []
        measure_index = 0
        pending_paren_before = False
        last_note = None

        for token in line.split():
            if _BAR_RE.match(token):
                if current:
                    measure_index += 1
                    measures.append({
                        "number": f"{line_no}-{measure_index}",
                        "row": line_no,
                        "repeat_before": None,
                        "repeat_after": _BAR_KIND[token],
                        "notes": current,
                    })
                    current = []
                continue
            if token == "(":
                pending_paren_before = True
                continue
            if token == ")":
                if current:
                    current[-1]["paren_after"] = True
                continue
            if token == "⌒":
                if current:
                    current[-1]["slur_start"] = [1]
                continue

            match = _TOKEN_RE.match(token)
            if not match:
                raise ValueError(f"第 {line_no} 行无法解析记号：{token!r}")
            digit, octave_marks, beams, dot, accent, extends = match.groups()
            n = int(digit)
            up = octave_marks.count("^")
            down = octave_marks.count("v")
            beams = len(beams)
            dotted = bool(dot)
            extend = len(extends)
            q = (1.0 / (2 ** beams)) * (1.5 if dotted else 1.0) + extend

            note = {
                "rest": n == 0,
                "n": n,
                "o": max(-3, min(3, up - down)),
                "q": round(q, 4),
                "markup": {"beams": beams, "dotted": dotted, "extend": extend},
                "lyric": "",
                "slur_start": [],
                "slur_stop": [],
                "cx": None,
                "paren_before": pending_paren_before,
                "paren_after": False,
                "accent": bool(accent),
            }
            if last_note is not None and last_note.get("slur_start"):
                note["slur_stop"] = [1]
            current.append(note)
            pending_paren_before = False
            last_note = note

        if current:
            measure_index += 1
            measures.append({
                "number": f"{line_no}-{measure_index}",
                "row": line_no,
                "repeat_before": None,
                "repeat_after": "single",
                "notes": current,
            })

    if not measures:
        raise ValueError("文本里没有可解析的音符")
    metadata = {"title": title, "key": key_mark, "meter": meter}
    return metadata, measures


def load(path, key_mark="1=C", meter="4/4", title=""):
    with open(path, "r", encoding="utf-8") as file:
        text = file.read()
    return parse_text(text, key_mark=key_mark, meter=meter, title=title)


# ---------------------------------------------------------------- 逆运算（输出）

def note_to_token(note):
    """一个 note → 一个文本记号（顺序：数字 + 音区点 + 减时线 + 附点 + 延时线）。

    例：`5v__·-` = 低音5、两条减时线、附点、一条延时线。
    未识别的音（n 为 None）写作 `?`——它是"未识别标记"，回读时会解析失败。
    """
    if note.get("rest"):
        digit = "0"
    else:
        n = note.get("n")
        digit = str(int(n)) if n is not None and 1 <= int(n) <= 7 else "?"
    o = int(note.get("o", 0) or 0)
    octave = "^" * max(o, 0) + "v" * max(-o, 0)
    mk = note.get("markup") or {}
    beams = "_" * max(int(mk.get("beams", 0) or 0), 0)
    dot = "·" if mk.get("dotted") else ""
    accent = ">" if note.get("accent") else ""
    extend = "-" * max(int(mk.get("extend", 0) or 0), 0)
    return f"{digit}{octave}{beams}{dot}{accent}{extend}"


def bar_to_token(measure):
    """小节线 → `|` / `||` / `:||` / `||:`（review_io 也复用这个映射）。"""
    after = measure.get("repeat_after")
    if after == "backward":
        return ":||"
    if after == "forward":
        return "||:"
    if after in ("double", "final", "light-heavy", "heavy-light"):
        return "||"
    return "|"


def _group_by_row(measures, measures_per_line):
    """按原谱行分组：有 row 就用 row（OCR / 文本谱），否则按小节数切（MusicXML）。"""
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
        for i in range(0, len(measures), max(measures_per_line, 1)):
            lines.append(measures[i:i + max(measures_per_line, 1)])
    return lines


def to_text(measures, measures_per_line=4, with_line_prefix=True):
    """note 流 → 文本简谱：**每行一条字符串**，方便肉眼比对原谱。

    输出形如：
        L01: ( 0 1 2 5 1 2 6· | 0 1 2 6 5 3 2 5 | 6· 5 3 6 | i 6 5 6- |

    记号顺序与 parse_text 完全一致，所以这份文本可以**直接当作 convert.py 的
    输入再读回**（唯一的例外：未识别的 `?` 回读时会报错，属预期）。
    """
    lines = _group_by_row(measures or [], measures_per_line)
    out = []
    for line_no, group in enumerate(lines, 1):
        tokens = []
        if group and group[0].get("repeat_before") == "forward":
            tokens.append("||:")
        for measure in group:
            for note in measure.get("notes", []):
                if note.get("paren_before"):
                    tokens.append("(")
                tokens.append(note_to_token(note))
                if note.get("slur_start"):
                    tokens.append("⌒")
                if note.get("paren_after"):
                    tokens.append(")")
            tokens.append(bar_to_token(measure))
        prefix = f"L{line_no:02d}: " if with_line_prefix else ""
        out.append(prefix + " ".join(tokens))
    return "\n".join(out)
