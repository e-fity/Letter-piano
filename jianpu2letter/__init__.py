# -*- coding: utf-8 -*-
"""
jianpu2letter —— 中文数字谱（简谱）↔ 键盘字母谱（映射2）转换工具包。

核心（仅标准库）：
  - mapping           映射1 / 映射2 键位定义与 (n,o)→键位 派生
  - musicxml_reader   MusicXML 解析
  - solfege           音高→首调唱名(n,o) 与 时值标记
  - letterscore       note 流→字母谱文本 + XML 输出

可选（需第三方库，按需懒加载）：
  - jianpu_render     渲染中文数字谱图片（matplotlib）
  - ocr_jianpu        图片 OCR 识别（PaddleOCR + OpenCV）
"""

from . import mapping, musicxml_reader, solfege, letterscore

# 各级诊断里带了不少简谱符号（`³` 三连音、`↔` 互换、`⌒` 连音…），它们不在 GBK 里。
# Windows 控制台默认 GBK，print 这类字符会抛 UnicodeEncodeError —— 而且因为是在
# **识别过程中**打印的，一旦抛出就整张谱中断（用户 2026-09-20 的 12 张批处理里
# 7.png / 两只老虎.png 就是这样整张崩掉的）。
# 这里把 stdout/stderr 的编码错误降级为替换（`?`），任何符号都不会再让识别中断。
import sys as _sys
for _s in (_sys.stdout, _sys.stderr):
    try:
        _s.reconfigure(errors="replace")     # Python 3.7+
    except (AttributeError, ValueError):     # 非文本流 / 老版本
        pass

__version__ = "0.1.0"
__all__ = ["mapping", "musicxml_reader", "solfege", "letterscore", "__version__"]
