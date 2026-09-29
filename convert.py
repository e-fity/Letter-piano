#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一键入口：把 MusicXML 或中文数字谱图片 → 映射2键盘字母谱(XML) + 中文数字谱图片。

用法：
  python convert.py song.musicxml -o out/
  python convert.py jianpu.png -o out/ --key Eb
"""
import sys

from jianpu2letter.cli import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
