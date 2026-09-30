# -*- coding: utf-8 -*-
"""
映射表定义与「首调唱名(n, o) → 键盘键位」的派生规则。

来源：https://letterpianoygx260910.z7.web.core.windows.net/
- 映射2（本工具默认）：低、中、高音依次放在 ZXCVBNM / SDFGHJK / ERTYUIO。
- 映射1（原版 26 键，仅作对照保留）。

首调唱名约定：
- n 为唱名数字 1-7（1=do … 7=si）。
- o 为相对八度：-1=低音、0=中音、1=高音；±2/±3 通过映射2的 Shift 扩展。
"""

# ---------------------------------------------------------------------------
# 映射2（默认）：低 / 中 / 高 三排键
# ---------------------------------------------------------------------------
MAPPING2 = {
    -1: {1: "Z", 2: "X", 3: "C", 4: "V", 5: "B", 6: "N", 7: "M"},  # 低音
    0: {1: "S", 2: "D", 3: "F", 4: "G", 5: "H", 6: "J", 7: "K"},   # 中音
    1: {1: "E", 2: "R", 3: "T", 4: "Y", 5: "U", 6: "I", 7: "O"},   # 高音
}

# ---------------------------------------------------------------------------
# 映射1（原版 26 键，仅对照）
# ---------------------------------------------------------------------------
MAPPING1 = {
    -1: {1: "M", 2: "N", 3: "B", 4: "Z", 5: "X", 6: "C", 7: "V"},  # 低音
    0: {1: "A", 2: "S", 3: "D", 4: "F", 5: "G", 6: "H", 7: "J"},   # 中音
    1: {1: "K", 2: "L", 3: "Y", 4: "U", 5: "I", 6: "O", 7: "P"},   # 高音
    2: {1: "T", 2: "R", 3: "E", 4: "W", 5: "Q"},                    # 双高音（部分）
}

# Shift 修饰键（用于映射2的 ±2/±3 八度扩展）
LEFT_SHIFT = "LSHIFT"    # 左 Shift
RIGHT_SHIFT = "RSHIFT"   # 右 Shift


def key_for(n, o, mapping="mapping2"):
    """
    由唱名数字 n(1-7) 与相对八度 o(-3..3) 派生键位。

    返回 (key, modifier)：
      - key      大写字母键位
      - modifier None / "LSHIFT" / "RSHIFT"
    """
    n = int(n)
    o = int(o)
    if not (1 <= n <= 7):
        raise ValueError(f"唱名数字 n 必须为 1-7，得到 {n!r}")

    if mapping == "mapping2":
        if -1 <= o <= 1:
            return MAPPING2[o][n], None
        if o == -2:
            return MAPPING2[-1][n], LEFT_SHIFT
        if o == -3:
            return MAPPING2[-1][n], RIGHT_SHIFT
        if o == 2:
            return MAPPING2[1][n], LEFT_SHIFT
        if o == 3:
            return MAPPING2[1][n], RIGHT_SHIFT
        raise ValueError(f"映射2 不支持的八度 o={o}（允许 -3..3）")

    if mapping == "mapping1":
        if o in MAPPING1 and n in MAPPING1[o]:
            return MAPPING1[o][n], None
        raise ValueError(f"映射1 不支持 n={n}, o={o}")

    raise ValueError(f"未知映射 {mapping!r}")
