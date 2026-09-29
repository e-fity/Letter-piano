# -*- coding: utf-8 -*-
"""
首调唱名换算与简谱时值标记。

核心算法（「音名步进法」）：
1. 由调号 fifths 得到主音音名（大调 1=主音；小调取关系大调）。
2. 每个音符用「音名步进 + 八度」构成一个 diatonic index：
       index = octave * 7 + step_index(音名)
   主音的中音锚点 index 固定在第 4 个八度（可调 --tonic-octave）。
3. 差值 diff = note_index - tonic_index：
       唱名 n = diff mod 7 + 1
       八度 o = diff // 7（向下取整）
   该算法天然把临时升降号「按同音级自然音」简化（忽略 alter，只看音名）。

时值：MusicXML 的 duration/divisions = 以四分音符为 1 的时值 q，
换算成简谱记号：
   - 减时线（beam）：q<1 时为 1/2^k，k 条下划线
   - 附点（dotted）：×1.5
   - 延时线（extend）：整数 q≥1 时 q-1 条横线
"""

import math

from . import musicxml_reader as mx

# 音名 → 自然音级步进（C=0 … B=6）
STEP_INDEX = {"C": 0, "D": 1, "E": 2, "F": 3, "G": 4, "A": 5, "B": 6}

# 音名 → 音级（MIDI pitch class，C=0）
PITCH_CLASS = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def tonic_from_fifths(fifths):
    """
    由调号 fifths 得到「1」的主音信息。
    返回 dict: name(音名), step(音名首字母), pitch_class(0-11)。
    """
    name = mx.MAJOR_KEYS_BY_FIFTHS.get(int(fifths))
    if name is None:
        raise ValueError(f"不支持的调号 fifths={fifths}")
    step = name[0]
    acc = name[1:]
    pc = PITCH_CLASS[step]
    if acc == "#":
        pc += 1
    elif acc == "b":
        pc -= 1
    pc %= 12
    return {"name": name, "step": step, "pitch_class": pc}


def key_mark_from_fifths(fifths):
    """生成「1=X」字符串（ASCII 写法，如 1=Eb、1=F#）。"""
    return "1=" + mx.MAJOR_KEYS_BY_FIFTHS.get(int(fifths), "C")


def note_to_solfege(step, octave, tonic_step, tonic_octave=4):
    """
    由音符音名(step)与八度(octave)换算首调唱名 (n, o)。
    tonic_step: 主音音名首字母；tonic_octave: 主音「中音1」所在八度（默认 4）。
    """
    note_index = octave * 7 + STEP_INDEX[step]
    tonic_index = tonic_octave * 7 + STEP_INDEX[tonic_step]
    diff = note_index - tonic_index
    n = (diff % 7) + 1
    o = diff // 7  # Python // 为向下取整，负八度正确
    return n, o


def duration_markup(q):
    """
    把以四分音符为 1 的时值 q 换算成简谱记号。
    返回 dict: beams(减时线条数), dotted(是否有附点), extend(延时线条数)。
    """
    if q is None:
        q = 1.0
    q = round(float(q), 6)
    if q <= 0:
        return {"beams": 0, "dotted": False, "extend": 0}

    # 依次尝试「无附点 / 有附点」两种写法
    for dotted in (False, True):
        base = q / (1.5 if dotted else 1.0)
        if base >= 1 - 1e-6:
            b = round(base)
            if abs(base - b) < 1e-4:
                ext = b - 1
                if ext >= 0:
                    return {"beams": 0, "dotted": dotted, "extend": ext}
        else:
            k = round(-math.log2(base))
            if k >= 1 and abs(base - 2 ** (-k)) < 1e-4:
                return {"beams": k, "dotted": dotted, "extend": 0}

    # 兜底：无法标准化的时值（如三连音等）按四分音符输出
    return {"beams": 0, "dotted": False, "extend": 0}


def build_stream(score, tonic_octave=4, skip_chord=True, skip_grace=True):
    """
    把 MusicXML 解析结果转换成「note 流」。

    返回 list，每个元素为一个小节 dict：
      {
        "number": 小节号,
        "repeat_before": None / 'forward',
        "repeat_after":  None / 'forward' / 'backward' / 'single' / 'double' / 'final',
        "notes": [ note dict ... ],
      }
    note dict（非休止）：
      {"rest": False, "n": 1-7, "o": -3..3, "q": 时值(四分音符为单位),
       "markup": {beams,dotted,extend}, "lyric": 歌词, "slur_start":[..], "slur_stop":[..]}
    note dict（休止）：
      {"rest": True, "n": 0, "o": 0, "q": 时值, "markup": {...}, "lyric": ""}
    """
    tonic = tonic_from_fifths(score["key_fifths"])
    tonic_step = tonic["step"]
    divisions = score["divisions"] or 1

    measures_out = []
    open_tie = {}  # 音高键(step, octave) -> note dict，用于跨音符延音合并

    for m in score["measures"]:
        m_div = m["divisions"] or divisions
        notes_out = []

        for note in m["notes"]:
            if skip_grace and note["grace"]:
                continue
            if skip_chord and note["chord"]:
                continue
            q = note["duration"] / m_div if m_div else 0.0
            markup = duration_markup(q)

            if note["rest"]:
                notes_out.append({
                    "rest": True, "n": 0, "o": 0, "q": q, "markup": markup,
                    "lyric": "", "slur_start": [], "slur_stop": [],
                })
                continue

            step = note["step"]
            octave = note["octave"]
            if step is None or octave is None:
                # 无音高的非休止音符（罕见）跳过
                continue

            n, o = note_to_solfege(step, octave, tonic_step, tonic_octave)
            pitch_key = (step, octave)

            lyric = "".join(note["lyrics"]) or ""

            # 延音合并（tie stop 且与上一同音高音符衔接）
            if note["tie_stop"] and pitch_key in open_tie:
                prev = open_tie.pop(pitch_key)
                prev["q"] += q
                prev["markup"] = duration_markup(prev["q"])
                if lyric:
                    prev["lyric"] = (prev.get("lyric") or "") + lyric
                continue

            item = {
                "rest": False, "n": n, "o": o, "q": q, "markup": markup,
                "lyric": lyric,
                "slur_start": list(note["slur_start"]),
                "slur_stop": list(note["slur_stop"]),
            }
            if note["tie_start"]:
                open_tie[pitch_key] = item
            notes_out.append(item)

        measures_out.append({
            "number": m["number"],
            "repeat_before": m["barline_before"],
            "repeat_after": m["barline_after"],
            "notes": notes_out,
        })

    return measures_out
