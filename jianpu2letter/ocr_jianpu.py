# -*- coding: utf-8 -*-
"""
中文数字谱图片 → note 流（v8）。

策略：
1. RapidOCR 对放大后的整图识别；只把“不含中文且至少有 4 个有效简谱数字”的
   OCR 行作为旋律行锚点，因此标题、歌词和段落编号不会混入。
2. 在锚点行附近用连通域定位每个数字块。
3. OCR token 内数字数与覆盖的图像块数一致时，建立高置信“本图字体模板”。
4. 用本图模板补全 OCR 漏掉的数字；无法可靠判断时保留 ?，不瞎猜。
5. 音区点、减时线、附点、延时线、小节线由几何位置识别。

正常运行只输出一行摘要；传入 --ocr-debug-dir 时才保存/打印详细诊断信息。
"""

import os
import re
from collections import defaultdict

import cv2
import numpy as np

SCALE = 2
TARGET_DIGIT_PX = 48      # 处理前把「数字高度」归一到这个像素数（小字号的救命设置）
# 逐行/逐音诊断（调参用）。默认**关**：全量跑 19 张时日志会到几百 KB，也很难读。
# 要排查时用环境变量打开（不用改代码）：
#     PowerShell:  $env:JIANPU_VERBOSE=1; python run_all.py ...     # 只对本次会话有效
#     bash:        JIANPU_VERBOSE=1 python run_all.py ...
# 设 0/false/no 都算关；也可以在调用方 `oj.VERBOSE_DEBUG = True` 覆盖（探针脚本就是这么干的）。
VERBOSE_DEBUG = os.environ.get("JIANPU_VERBOSE", "").strip().lower() not in (
    "", "0", "false", "no")
ENABLE_LABEL_SWAP = True    # 标签互换（2-opt）：开关（用户 2026-09-18 要求开启）
# 逐音置信度（**只用于把可疑音标出来，不参与任何判定**）：
#   conf = 该块与「它最终采用的数字」的参照字形相似度（0..1）
#   conf 为 None = 拿不到分数（未识别，或块尺寸不符合）→ 同样按可疑处理
# 已知盲区：像《采蘑菇的小姑娘》那种 5/6 **系统性**错判，字形本来就极像，
# 分数会很高、标不出来。这套只擅长抓"本来就勉强过线"的音。
CONF_WARN = 0.70         # 低于此值的音，输出图里加黄底
DISAGREE_CONF = 0.30     # 形状判据另有结论、且与最终值矛盾时，压到这个值
# 一条**真**延时线 `-` 的墨宽约 0.85H（基准来自上海滩日志：墨宽 1.7H → 2 条）。
# 原来的噪声门槛只有 0.25H，结果把 0.3H 左右的碎片（附点、数字笔画残迹等）
# 也当成了 `-` ——《南泥湾》几乎每个音都多出 1 条假 `-` 就是它。
# 另一条硬证据：有窗口本身只有 0.3~0.4H 宽，那条"线"在几何上就不可能是真 `-`。
# 设成模块常量是为了好回调（各字体/扫描件的 `-` 宽度会略有差异）。
EXT_DASH_MIN_W = 0.50    # 一条 `-` 的最小墨宽（单位：字高）
# 一条 `-` 的「逐列墨心极差」上限（单位：字高）。真 `-` 每列都落在同一高度 →
# 极差小；**阶梯状残迹**（下面减时线的尾巴等被横向膨胀连进来）各列高度差得远。
#
# 《南泥湾》实测（**这是第二轮修正后的数字**）：
#     真 `-`： 0.01 / 0.34 / 0.42        ← 三条（含第 1 行 m6 的 `2 -`、末行的 `5 -`）
#     假 `-`： 0.58 / 0.69 / 0.89 / 0.97 / 1.05
# 空档是 **0.42 → 0.58**，第二轮修正时据此取 0.50。
# ⚠ 余量只有 **1.2 倍**，比 `孔比`（2 倍）、弧宽（4 倍）薄得多，换谱容易失手。
#
# **2026-09-28 上调到 0.90**（用户报 36.jpg 的 `3 - -` 整段 `-` 不见）：0.50 偏低 ——
# 这一带混进 JPEG 噪点后，**真** `-` 的列心极差会被撑到 0.84（36.jpg 行1 音22 实测：真横线各列
# 墨心≈0，但数字中线上下 ±0.5H 处有自适应二值化产生的零星墨点），整段被误判成"阶梯残迹"丢光。
# 全库 A/B（0.50 → 0.90，40 张 / 11601 个音）：**+24 处、只涉 8 张图、其余 32 张零变化、
# 无一处由对变错**；新增的 27 条已逐条回原谱核对：**25 真 / 2 假**。那 2 条假的全在 12.jpg，
# 且**改动前后都是错的**（1 条是**附点漏检** —— 11px 小字号下 `·` 没被认出，被当成 `-`；
# 1 条是横线二值化后断成两截、0.15H 的膨胀没连上，被数成 2 条）。二者是**另外的缺陷**。
# ⚠ 但 0.90 **已落进上面"假 `-`"的区间（0.58~1.05）**：这次是 `EXT_MIN_BAND_COV`（实心度，
#   见下）独立把南泥湾那批假 `-` 拦住了，**换谱仍有失手风险**，别把它当成"修好了"。
# 根因：**真的那两条自己也不干净**——干净的 `-` 各列墨心应都在 0 附近
#（`span_lo ≈ -0.03`），实测真值却是 -0.21 / -0.29，说明它们的墨段里也混进了
# 下方的东西。于是真假只差"混进去多少"，是剂量差别而非性质差别 → 判据天生脆。
# **正确的长期修法**：不要把 ±0.60H 整条带子的墨一起投影到 x 轴找段，而应**按层**处理
#（延时线在数字中线那一层、减时线在下面一层，各层分别投影），这样两层互不污染。
# 那是重写 `_count_extends` 的核心，风险较大，暂未做。
EXT_MAX_COL_SPAN = 0.90
# 「薄带覆盖」下限（用户 2026-09-21 选的"实心度判据"，见 `_count_extends` 里的实测）。
# 段内最"实"的 0.35H 薄带里、有墨的列占段宽的比例：真 `-` 是一条横线 → 高；
# 低分辨率扫描件的 JPEG 噪点（被横向膨胀连成一段）是散点 → 低。
EXT_MIN_BAND_COV = 0.55
# 数延时线时，扫描窗口的右端最远能扫多远（单位：字高）。**只在"下一个数字"未知时兜底**
# （行末最后一个音）；只要知道下一个数字在哪，右端就是它的左缘——**不受这个上限约束**。
# 用户 2026-09-22 报"4.png / 13.png 的最后一行延时线少数"：那两行的音符稀疏
# （一行只有 5~6 个音要铺满整行宽，别行 12~13 个），音与音之间的 `-` 总长就长，
# 旧值 6.0H 把窗口截断，**截断点之后那 300~500px 根本没扫**。`[ext]` 实测：
#   4.png 行11 音4：窗口 1253..1536（恰好 5.9H）→ 只数出 1 条；下一个数字在 2038
#                   （需要 16.4H 才够）→ 该有 3 条；
#   13.png 行10 音5：窗口 557..840（5.9H）→ 1 条；下一个数字在 897（需要 6.1H）；
#   同时段普通行（4.png 行7~10 音1）窗口同样 5.9H，但音符密、横线短 → 2 条全在窗口内 ✓。
# 所以把上限抬到 20H：足够覆盖实测最稀的行（16.4H），又给"行末扫飞了"留个兜底。
EXT_SCAN_MAX_SPAN = 20.0
# 「内孔居中 → 0」要求内孔面积占块面积的比例不低于此值。
# 物理依据：0 是个圈 → 孔大；6 主体实、只有个小 counter → 孔小。
# 《南泥湾》实测：真 0 的 `孔比` = 0.23，而**全部** 6（含被误判成 0 的）都是 0.10~0.12，
# 中间空了一倍。旧值 0.10 太松，把 6 的小孔也算成了"居中孔" → 6 被成片判成 0
#（该谱第 1 行 6 个 `0` 里 5 个是这么来的，只有 1 个是 0/6 复核改的）。
HOLE_ZERO_MIN_RATIO = 0.16
# 「判 0」时孔心**必须有多居中**（`|cy − 0.5|` 的上限）。原来是 `0.16`（覆盖 cy ≤ 0.66），
# 而「判 6」要求 cy ≥ 0.60 —— **两段区间重叠**，于是 cy ∈ [0.60, 0.66] 的 6 会被"判 0"抢先返回
# （`_hole_class` 的 docstring 预警过这个重叠；此处正是它咬人的地方）。
# 实测（全 30 张、1823 个 0/6 块）：真 0 的 cy 中位 **0.487**、最大 **0.519**；
# 真 6 的 cy 中位 **0.670**；而 **[0.55, 0.60) 是一条空带**（1823 个里一个都没有）。
# 取 `0.075`（→ cy ≤ 0.575）正落在这条空带中间，两侧余量最大。
# 起因：2026-09-23 用户报 17.jpg《爱情转移》一个 6（cy=0.659）被这条抢先判成了 0。
HOLE_ZERO_MAX_DCY = 0.075
# 「完全没有封闭内孔」的上限：低于它说明这个块里一个孔都没有。
# 《南泥湾》实测三级阶梯（同一个量在不同尺度上）：
#     5 → `孔比` = 0.00（25/25，无封闭内孔）
#     6 → `孔比` ≈ 0.11（小孔、位置偏下）
#     0 → `孔比` ≈ 0.23（大孔、居中）
# 5 与 6 之间空得很宽，所以「标为 6 却毫无内孔」的块必然是 5（见下面的 5/6 复核）。
# 反过来「5 有孔 → 6」逻辑上也成立，但本谱没有反例样本可验，故意不做。
HOLE_FIVE_MAX_RATIO = 0.05
# 连音弧 `⌒` 的「拱起程度」下限（逐列墨心 y 的极差 ÷ 高，见 `_hump()`）。
# 直的横墨（延时线 `-`、减时线、小节线）≈0；真弧会明显拱起。
# 位置条件另外把 `-`/减时线挡在外面（它们不在数字上方），所以这个阈值不必太严。
ARC_MIN_HUMP = 0.35
# 连音弧的最大宽度（单位：字高）。物理依据：**连音弧只连少数几个相邻音**
#（《南泥湾》19 条真弧实测 2.0~4.4H，最长的一条跨 4 个音 = 6.1H）；
# 而**房子／反复括线是按小节、按乐句画的**，必然跨一整句 —— 实测混进来的两个
# 假块是 17.65H 和 44.13H（44H ≈ 2000px，横跨整行），差 4 倍，余量很大。
# 那两个假块其实是「房子括线 + 紧邻的 `)` 粘成一块」：比纯括线更高（0.74/0.78H
# vs 真弧 0.61~0.65H）、拱起被钩子拉到 0.61~0.79（真弧 0.79~0.88）。
# 高度也能分但只剩 0.09H 余量，换个谱型就会误伤，所以**主判据用宽度**。
ARC_MAX_W = 8.0
# 重音记号 `>`（数字上方的小 V，尖朝右）。判据全部来自《上海滩》实测的 6 个 `>`：
#   宽 0.82~0.85H、高 0.64H、填充 0.27~0.28、拱起 0.09~0.19
# 与其它「上方记号」的区分**靠三个量交叉，不是单一阈值**（实测值）：
#   音区点  0.26H / 填充 0.83        → 宽度 + 填充排掉
#   小数字（³、房子标号 `2.`）0.45~0.53H → 宽度排掉
#   `v` 换气 0.45H / 拱起 0.80       → 宽度 + 拱起**双重**排掉
#   连音弧  ≥1.6H / 拱起 0.82+       → 宽度 + 拱起排掉
# 所以**不需要**判"尖朝右还是朝下"——`v` 在尺寸与拱起上都进不来（这是实测结论，
# 本来以为必须靠方向区分，看了形状才发现不用）。
ACCENT_MIN_W = 0.60      # `>` 的最小宽度（单位：字高）
ACCENT_MAX_W = 1.20      # 最大宽度
ACCENT_MAX_FILL = 0.50   # 填充率上限（音区点 0.83 就是被它排掉的）
ACCENT_MAX_HUMP = 0.35   # 拱起上限（弧与 `v` 都是 0.80+，被它排掉）
# ⚠ 高度上限：`>` 实测 6 个样本全是 0.64H。**原来看漏了这条**——只判了宽度/填充/拱起，
# 于是"上一行掉进窗口的数字"（0.87~0.89H、填充 0.47、拱起 0.28）也全部通过，
# 造出过假 `>`（上海滩第 8 行）。加 0.80H 把数字挡在外面。
ACCENT_MAX_H = 0.80
# 上方小记号：`v` 换气 / `³` 三连音 / 房子标号 `1.` `2.` `3.`
# ⚠ **阈值目前是"暂定"**：每个记号只从《上海滩》各取到 1 个样本（见 README 7.3）。
# 先用尺寸把"小记号"圈出来（音区点 0.26H 在下面、`>` 0.82H 在上面、弧/括线更宽），
# 再用两个**有物理意义**的量交叉区分：
#   `拱起`：`v` 实测 0.80（两笔收成尖 → 逐列墨心变化大）；`³` 实测 0.21（笔画偏竖直）。
#   `底部横杠`：房子标号（小数字）≈1.0（数字有满宽基线）；`v` ≈0.2（底部是个尖）。
#   `³` 与房子标号 `3.` 是**同一个字形**，只能靠"旁边有没有房子括线"分（见 _find_wide_bracket）。
MARK_MIN_W = 0.28        # 小记号宽度下限（音区点 0.26H 被挡在外面）
MARK_MAX_W = 0.70        # 上限（`>` 是 0.82H，被挡在外面）
MARK_MIN_H = 0.35        # 高度下限（音区点 0.26H）
MARK_MAX_H = 0.82        # 高度上限。原写 0.90H，实测放进了"上一行的数字"
                         # （0.87~0.89H，且有满宽基线 → 那本来就是数字），
                         # 在第 8 行一下子造出 19 个假三连音。真记号实测 0.57~0.78H，
                         # 与 0.87 之间有 0.09H 空档 → 收到 0.82H。
MARK_MAX_FILL = 0.60     # 填充率上限（音区点 0.83 被挡在外面）
MARK_V_MIN_HUMP = 0.55   # `v` 的拱起下限（实测 0.80）
MARK_V_MAX_BAR = 0.45    # `v` 的底部横杠上限（底部是尖，实测 ≈0.2）
MARK_TRIPLET_MAX_HUMP = 0.45   # `³` 的拱起上限（实测 0.21）
HOUSE_BRACKET_MIN_W = 6.0      # 房子括线的最小宽度（实测 20.5H）
# 房子括线的另外三个特征：**两端有向下的竖钩**（所以比细弧"高"得多）、**很细**、**不太厚**。
# ⚠ 高度**下限**是关键（原来只设了上限，长连音弧就全被当成房子括线了）：
#   实测 连音弧 `高 0.47~0.65H`（就是一条细弧）；房子括线 `高 1.09~1.13H`（两端竖钩撑着）。
#   两者差近一倍，中间空着 → 取 0.85H。
#   实测填充：括线 0.08~0.09。
HOUSE_BRACKET_MIN_H = 0.85
HOUSE_BRACKET_MAX_H = 1.60
HOUSE_BRACKET_MAX_FILL = 0.25
# 房子括线**离本行数字顶多远**（单位：字高）——这是它**自己的**窗口，别再照抄
# `UPPER_MARK_MAX_LIFT`（那个 1.05H 是按"上方小记号"实测定的，见上面的教训）。
# 《孤勇者》(10.png) 实测三条括线都在 **+1.32~+1.34H**，比 1.05H 高 —— 一律被拒，
# 表现为"房子横线和房子标号整片消失"（用户 2026-09-20 报的）。
# 取 1.60H：高于实测的 1.34H，又远低于行距（该谱约 9H；上一行的括线会落在 ~2.7H 以上）。
# 括线本身还有 `宽 ≥ 6H` 这道闸，上一行的数字/小记号（宽 ≤1.2H）本来就进不来。
HOUSE_BRACKET_MAX_LIFT = 1.60
# 「上下叠着的两条弧」必须**两条都是强弧形**才留（配合 `_y_overlap` 的第二道判据）。
# 实测：真嵌套对（《孤勇者》行9）拱起 0.70~0.81；而同样"同心但不重叠"的杂块
# （三连音括线等，2026-09-20 在 6.jpg / 7.png / 上海滩 行11 各测到几条）只有 0.38~0.53。
# 0.65 取在两者之间。
ARC_PAIR_MIN_HUMP = 0.65
# 房子标号 `1` 与 `2` 的定向复核阈值（主笔画倾斜度，见 `_stroke_drift()`）。
# ⚠ **必须用实测值定，别用 ASCII 目测**：《上海滩》实测 `1` = **0.16**、`2` = **0.34**；
# 我第一次按 ASCII 目测估成 0.21 / 0.56（因为我目测时取的是"顶部 vs 最底"，
# 而 `_stroke_drift()` 取的是 30~50% 与 60~80% 两段——**窗口不同，值就不同**），
# 结果把阈值定在 0.38（在两者之上）→ 连行 4 的 `2` 也一起改判成 `1` 了。
# 取两者中间 0.25：行3 0.16 < 0.25 → 1；行4 0.34 ≥ 0.25 → 2。
# ⚠ 样本各只有 1 个（0.16 / 0.34），余量 1.56× / 1.36×，属**暂定值**。
HOUSE_LABEL_ONE_MAX_DRIFT = 0.25
# 上方记号的**窗口上限**：记号底边最多离本行数字顶多远（单位：字高）。
# 《上海滩》实测（`距顶=` 这个诊断量）：
#     真 `>`  +0.51H ；真 v/³ -0.15~+0.00H ；真 房子标号 +0.87H（3 个样本都是 0.87）
#     误检（上一行的数字等）+1.13 ~ +1.77H
# 原来开 2.2H（**照抄连音弧的窗口**）→ 把上一整行的数字都收了进来，
# 第 8 行一次造出 19 个假三连音。收到 1.05H：真值最大 0.87、误检最小 1.13，两侧都有余量。
# ⚠ 教训：**这类窗口必须按实测的"记号离数字多远"来定，不能照抄别的功能的窗口。**
UPPER_MARK_MAX_LIFT = 1.05
LABEL_SWAP_MARGIN = 0.02    # 互换需带来的相似度增益门槛（用户要求保持 0.02）
ENABLE_GRACE_ROW_FILTER = True  # 剔除以"倚音行"（整行块高 <0.8×数字字高）
REMOVE_PAREN_BLOCKS = True      # 把"括号形状的块"从数字候选里剔除
# 「细 + 填充率低」的块，到底是**括号**还是**数字 1**：
# 两者在尺寸/填充上完全重叠 —— 17.jpg 的 `1` 实测 宽≈0.31H、填充 0.38~0.44；
# 14.jpg 那个真 `(` 实测 宽 0.35H、填充 0.39。只有**直不直**分得开：
#   * 数字 `1` 是一条竖笔 → 块内「最长逐列连续墨」≈ 满高（`_max_stem_run()`）；
#   * 括号是一段弧 → 明显短（`_arc_variation()` 量的是同一件事的另一种口径，
#     但会被 `1` 起笔的那面小旗干扰，实测两者都落在 0.3~0.4，分不开，故另开这个量）。
# 实测（19 张全扫）：
#   * 分类器确认为 `1` 的细块 192 个 → **0.87~1.00**（p10 = 1.00）；
#   * 14.jpg 那个真 `(` → **0.77**；其余图的弧线碎块 0.21~0.80。
# 取 0.85：落在 0.77 与 0.87 之间（两侧余量 1.10× / 1.02×）。
PAREN_STEM_MIN_RATIO = 0.85
# 括号候选还必须在**数字行的纵向带内**（用户 2026-09-22 报 10.png「莫名多出不少 `)`」）。
# 10.png 实测：那些括号全部出自"几何兜底"那一路（该图日志里**没有**任何
# `[par-mark] OCR文本路径` 行），而命中的 45 个几何候选**绝大多数是歌词汉字的偏旁**
# （"你"的亻、他/爱/伤…）—— `row_components` 取的是"与整行页面带 `[py0,py1]` 有重叠"
# 的连通块，页面带向下留了 0.95H，歌词行正好落在里面，而汉字偏旁恰好细高。
# 只有**纵向位置**分得开（`_diag_paren_dy.py` 实测）：
#   * 真 `(` dy=+0.02H、真 `)` dy=+0.01H（且高 1.38~1.45H，比数字略高）；
#   * 45 个假货 dy = **±1.9H ~ ±4.0H**（整体错开约两个数字高）。
# 12 张全扫：真括号 |dy| 最大 0.19H（7.png），假货最小 1.90H → 取 0.55，两侧各留约 3 倍。
# 与数字候选的 `abs(cy - baseline_y) > 0.55 * H` 是同一个口径。
PAREN_MAX_DY_RATIO = 0.55
# 「谱表左边界」：行首若有**贯穿整个谱表的左括线**，则它左侧的页边内容（声部名等）不成音。
# 为什么需要（用户 2026-09-22 报 21_overlay.png：字母盖住了"男/女/低"，而部分数字又丢了）：
# 那份谱的声部名汉字尺寸与数字**几乎一样**（实测 h/H0 = 1.10~1.19、宽高比 0.96~1.00、
# 面积比 0.42~0.60），**整个落在数字候选窗口内**（高度 0.50~1.35H、宽高比 0.12~1.15），
# 于是被当成音符；"像不像数字"那一步只用来**给值**、不用来剔除块，聚类投票又给它们刷上
# 标签（实测 11 个块拿到 n=1，而它们自己的形状分只有 0.29~0.53）→ 原位替换盖成字母。
# 试过"逐块形状分闸"（阈值 0.55）：能修掉 11 个汉字，但**会误伤 14.jpg 一个真 `2`**
# （高音点与数字粘连、拆分后分数掉到 0.48），而且 0.53 的汉字离阈值只有 0.04 →
# **余量 1.04×，太脆**；全量分布显示真数字那一侧从 0.60 起才有量，阈值抬不上去。故弃用。
# 改用**结构性**判据：括线左侧是页边，一定有非音符内容，且真音符必在括线右侧。
# 实测 21.jpg：括线 宽 0.94H、高 14.9H，声部名整体在它左侧（x≤346 vs 括线 x=367），
# 而本行小节竖线只有 2.6H 高 → 3.0H 门槛把竖线挡在外面（低侧余量 1.15×，高侧 5×）。
# 找不到这种竖条（单声部谱如 19.png）时规则不触发。
LEFTMOST_BAR_MIN_H = 3.0     # 竖条最低高度（字高倍数）：高于小节竖线(实测 2.6H)、低于括线(14.9H)
LEFTMOST_BAR_MAX_W = 1.0     # 竖条最大宽度（字高倍数，实测括线 0.94H）
LEFTMOST_BAR_MAX_X = 0.25    # 竖条必须落在页面宽度左侧的这部分内（防误取页中的长竖条）
# 注意：字形聚类是全图一起做的，**增删任意候选块都会改变其他块的标签**。
# 5.png 出现 5→1（只有这一张），而"倚音行剔除"已排除，故用本开关做对照实验。
LAST_ROW_INFO = []         # 最近一次识别的「每行数字块」（供 train_digits.py 采集样本）
_LAST_BINARY = None        # 最近一次识别的二值图（同上）
_LAST_COLOR = None         # 最近一次识别的**缩放后彩色图**（供原位盖字母用）
_LAST_SCALE = 1.0          # 上面那张图相对原图的缩放倍数
_DIGIT_MODEL = None        # 本字体数字分类器（digit_model.npz），None=未加载
_LAST_KEY_MARK = None      # 最近一次从页眉读到的调号（"1=X"），None=没读到
_LAST_TITLE = None         # 最近一次从页眉读到的曲名，None=没读到

# 页眉「1=X」调号的匹配式。`♭` 常被 OCR 读成小写 `b`（甚至撇号 `'` / 反引号），
# 且中文简谱写 `1=♭B`（升降号在字母**前**），而工具用 `1=Bb`（字母在前），所以前后各捕一组。
_KEY_RE = re.compile(r"[1lI]\s*[=＝]\s*([#♯b♭'’`]?)\s*([A-Ga-g])\s*([#♯b♭'’`]?)")
# 合法调号（简谱常用的大调名）；用来挡掉 OCR 把调号读歪成的非法值（如 `Fb`）。
_VALID_KEYS = {"C", "C#", "Db", "D", "Eb", "E", "F", "F#", "Gb", "G", "Ab", "A", "Bb", "B"}
# 页眉裁剪档位：(高度比例, 宽度比例, 缩放后目标宽)。**多档取并集**——实测单档不稳：
# 裁剪太窄会漏掉被大标题推到下方的调号，太宽又会把细小的 `♭` 读丢；不同图各中一档。
_KEY_CROPS = ((0.20, 0.55, 1600), (0.34, 0.55, 1600),
              (0.34, 0.80, 1200), (0.20, 0.40, 2000))
# 曲名：页眉最上方的**居中大字**，必须**整宽**裁（窄裁剪会把"上海滩"切成"上海"，只剩半截）。
_TITLE_CROP = (0.20, 1.0, 1600)
# 曲名判定要排除的字样（署名 / 副标题 / 影视信息），否则会被"电影《…》主题曲"这种行抢走。
_TITLE_SKIP = ("作词", "作曲", "词曲", "演唱", "打谱", "记谱", "制谱", "编曲", "改编",
               "原唱", "演奏", "合唱", "唱片", "制作", "监制", "出品", "发行", "混音",
               "母带", "吉他", "钢琴", "弹唱", "扒谱", "主题曲", "片尾曲", "插曲",
               "片头曲", "电视剧", "电影", "纪录片", "宣传曲", "推广曲", "独唱", "领唱")


def _load_digit_model():
    """加载用 train_digits.py 训练出的本字体数字分类器；不存在则返回 None。"""
    global _DIGIT_MODEL
    if _DIGIT_MODEL is None:
        path = os.environ.get("JIANPU_DIGIT_MODEL") or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "digit_model.npz")
        if os.path.exists(path):
            try:
                data = np.load(path)
                feats = data["feats"].astype(np.float32)
                labels = data["labels"].astype(np.int32)
                # 预计算：去均值 + 单位化，便于用点积算相关
                centered = feats - feats.mean(axis=1, keepdims=True)
                norm = np.linalg.norm(centered, axis=1, keepdims=True) + 1e-8
                _DIGIT_MODEL = {"feats": centered / norm, "labels": labels}
                print(f"[ocr] 已加载本字体数字模型：{len(labels)} 条样本")
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] 数字模型加载失败，忽略：{exc}")
                _DIGIT_MODEL = False
        else:
            _DIGIT_MODEL = False
    return _DIGIT_MODEL or None


def _classify_by_model(block, binary, model, min_score=0.75, min_margin=0.04):
    """用本字体模型（1-最近邻）判数字；把握不足返回 None。

    判据从「像不像系统字体的 5」变成「像不像你自己的谱里的 5」——这是解决
    字体不同导致 5/6、0/6 混淆的根本办法。
    """
    if not model:
        return None
    f = _shape_feature(block, binary)
    g = f - f.mean()
    gn = g / (np.linalg.norm(g) + 1e-8)
    sims = model["feats"] @ gn
    per_label = {}
    for idx, s in enumerate(sims):
        lab = int(model["labels"][idx])
        if s > per_label.get(lab, -2.0):
            per_label[lab] = float(s)
    if not per_label:
        return None
    ranked = sorted(((s, lab) for lab, s in per_label.items()), reverse=True)
    best_s, best_lab = ranked[0]
    second_s = ranked[1][0] if len(ranked) > 1 else -2.0
    if best_s >= min_score and best_s - second_s >= min_margin:
        return best_lab
    return None
MIN_ROW_DIGITS = 3
TEMPLATE_MIN_SCORE = 0.76
TEMPLATE_MIN_MARGIN = 0.025


def _imread_unicode(path, flags=cv2.IMREAD_COLOR):
    """读取图片，兼容中文路径。

    OpenCV 的 imread 在 Windows 下无法打开含非 ASCII 字符的路径
    （中文文件名会报 can't open/read file），改为先读字节再 imdecode。
    """
    with open(path, "rb") as fp:
        data = fp.read()
    buf = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(buf, flags)


def _feature_from_array(ink_mask, canvas=(24, 32)):
    """把「墨=True 的二维数组」等比缩放并居中到固定画布，返回归一化特征向量。"""
    target_w, target_h = canvas
    ys, xs = np.nonzero(ink_mask)
    if len(xs) == 0:
        return None
    h = int(ys.max() - ys.min() + 1)
    w = int(xs.max() - xs.min() + 1)
    scale = min((target_w - 4) / max(w, 1), (target_h - 4) / max(h, 1))
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    crop = ink_mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.uint8) * 255
    resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.zeros((target_h, target_w), dtype=np.uint8)
    x0 = (target_w - nw) // 2
    y0 = (target_h - nh) // 2
    out[y0:y0 + nh, x0:x0 + nw] = resized
    return out.astype(np.float32).reshape(-1) / 255.0


def _classify_by_reference(block, binary, refs, canvas=(24, 32),
                           min_score=0.55, min_margin=0.03, return_score=False):
    """把数字块与 0-7 标准字形逐一比相似度，取最高分（需明显高于第二名）。

    return_score=True 时返回 (数字, 分数)，分数也可用来判断「这块像不像数字」
    （汉字的最高分明显低于数字）。
    """
    if not refs:
        return (None, -1.0) if return_score else None
    roi = binary[block["y"]:block["y"] + block["h"], block["x"]:block["x"] + block["w"]]
    if roi.size == 0:
        return (None, -1.0) if return_score else None
    feat = _feature_from_array(roi > 0, canvas)
    if feat is None:
        return (None, -1.0) if return_score else None
    scores = sorted(((max(_corr(feat, r) for r in refs_for_digit), digit)
                     for digit, refs_for_digit in refs.items()), reverse=True)
    if not scores:
        return (None, -1.0) if return_score else None
    best = scores[0]
    second = scores[1] if len(scores) > 1 else (-1.0, None)
    digit = None
    if best[0] >= min_score and best[0] - second[0] >= min_margin:
        digit = best[1]
    return (digit, best[0]) if return_score else digit


def _stroke_runs(block, binary):
    """块内「每行墨段数」的平均值：数字笔画少（约 1.5-2.5），汉字笔画多（≥3）。"""
    roi = binary[block["y"]:block["y"] + block["h"],
                 block["x"]:block["x"] + block["w"]] > 0
    if roi.size == 0:
        return 99.0
    total = 0
    rows = 0
    for row in roi:
        runs = 0
        prev = False
        for v in row:
            if v and not prev:
                runs += 1
            prev = bool(v)
        total += runs
        rows += 1
    return total / max(rows, 1)


_REFERENCE_CACHE = None


def _reference_glyphs_by_font(canvas=(24, 32)):
    """渲染 0-7 的标准字形，按字体分组：{字体名: {数字: [特征, ...]}}。

    分组保存是为了「按图选字体」：一个谱面的字体是统一的，用最接近的系统字体
    做参照，比把多种字体混在一起比要准得多（5/6 这类易混对尤其明显）。
    """
    global _REFERENCE_CACHE
    if _REFERENCE_CACHE is not None:
        return _REFERENCE_CACHE
    try:
        from matplotlib import font_manager
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        _REFERENCE_CACHE = {}
        return _REFERENCE_CACHE

    available = {f.name: f.fname for f in font_manager.fontManager.ttflist}
    candidate_names = [
        "SimHei", "Microsoft YaHei", "SimSun", "Noto Sans CJK SC",
        "Source Han Sans SC", "PingFang SC", "WenQuanYi Zen Hei", "Arial Unicode MS",
    ]
    paths = [(n, available[n]) for n in candidate_names if n in available]
    if not paths:
        for f in font_manager.fontManager.ttflist:
            if any(k in f.name for k in ("Hei", "YaHei", "Song", "CJK", "Han")):
                paths.append((f.name, f.fname))
                break

    by_font = {}
    for name, path in paths[:4]:
        per_digit = {}
        for size in (32, 42):
            try:
                font = ImageFont.truetype(path, size)
            except Exception:  # noqa: BLE001
                continue
            for digit in range(8):
                img = Image.new("L", (80, 92), 255)
                ImageDraw.Draw(img).text((14, 12), str(digit), fill=0, font=font)
                feat = _feature_from_array(np.array(img) < 128, canvas)
                if feat is not None:
                    per_digit.setdefault(digit, []).append(feat)
        if per_digit:
            by_font[name] = per_digit
    _REFERENCE_CACHE = by_font
    return by_font


def _all_reference_glyphs():
    """所有候选字体的字形合并（用于「这一行像不像数字」这类粗判）。"""
    merged = {}
    for per_digit in _reference_glyphs_by_font().values():
        for digit, feats in per_digit.items():
            merged.setdefault(digit, []).extend(feats)
    return merged


# 「按类定标签」的两道闸（阈值全部来自《千年等一回》5.png 的聚类实测）：
# 该图 27 个真 5 被 OCR 读成「9 个对、15 个没数、3 个读成 1」，于是 15 个空白块和
# 3 个错读块聚成**同一类（18 块）**，规则①拿这 3 票把整类钉成 1 —— 实测该类的
# 「与参照字形相似度」是 `5:0.71 6:0.71 1:0.21`（均宽 0.70H，真 1 类是 0.40H），
# 也就是说**字形完全不支持 1**，而这 18 块里 15 块根本没有投票、无从反对。
# 两道闸各自独立挡住这种「弱证据定性整类」。
VOTE_MIN_SIM = 0.55          # 闸①：投票标签至少要有这么像参照字形（与规则②同门槛）
MERGE_MIN_VOTES = 5          # 同意票 < 此数 = 证据弱，可被并入别的类
MERGE_SAME_LABEL_MIN = 0.60  # 闸②：与本标签同类的互相关低于此 = 标签已被字形否掉
MERGE_HOST_MIN = 0.90        # 闸②：与目标强类的互相关高于此 = 同一字形。
# 0.90 有实测空档支撑：该图跨数字的类间互相关最高只有 0.84（6 类↔0 类），
# 而同为 5 的那两类是 0.955 —— 取 0.90 既认得出"同一字形"，又不会跨数字误并。


def _label_blocks_by_cluster(row_records, binary, refs, cluster_thr=0.80):
    """把全图数字块按字形聚类，每类只定一个数字，再刷给该类所有块。

    为什么这样做：同一份谱面字体统一 → 同一数字的多次出现形状几乎一致，
    不同数字形状不同。按「类」定标签既能用类平均字形去噪，又能施加
    「不同类给不同数字」的约束，从结构上避免 5/6 这类易混对被压成同一个数字
    （逐块判定无法避免这种系统性塌缩）。

    阈值 0.80：放松到 0.72 会过度切分（同一数字被拆成多类），实测把 0/5/6
    都判成同一个数字（3），故保持 0.80。0 与 6 的区分另想办法（内孔位置）。
    """
    items = []
    feats = []
    for ri, record in enumerate(row_records):
        for bi, block in enumerate(record["blocks"]):
            feats.append(_shape_feature(block, binary))
            items.append((ri, bi, record["values"][bi]))

    clusters = []
    for i, f in enumerate(feats):
        v_i = items[i][2]
        for c in clusters:
            if _corr(f, c["mean"]) < cluster_thr:
                continue
            # 投票约束：本块与该类已有的 OCR 投票互相矛盾时**不合并**。
            # 0 与 6、5 与 6 都是"闭合圈状"，只看形状会被并成一类、再统一刷成
            # 同一个数字（实测屁的 6 被并进 0 的类）；加上投票约束即可分开。
            if v_i is not None and c["votes"] and v_i not in c["votes"]:
                continue
            c["idx"].append(i)
            n = len(c["idx"])
            c["mean"] = (c["mean"] * (n - 1) + f) / n
            if v_i is not None:
                c["votes"][v_i] = c["votes"].get(v_i, 0) + 1
            break
        else:
            new_c = {"idx": [i], "mean": f, "votes": {}}
            if v_i is not None:
                new_c["votes"][v_i] = 1
            clusters.append(new_c)

    used = set()
    summary = []
    for c in sorted(clusters, key=lambda c: -len(c["idx"])):
        votes = defaultdict(int)
        for i in c["idx"]:
            v = items[i][2]
            if v is not None:
                votes[v] += 1
        label = None
        vote_only = None          # ①被闸①否掉的投票标签（②也定不下来时用它兜底）
        # ① 该类至少有 2 个成员被 OCR 判为同一数字 → 采用。
        #    OCR 很少把 5/6 互混（它的错误主要是 6→b/G、5→S，都已还原），
        #    因此 OCR 投票比「参照字形相似度」可信得多。
        #    之前要求 60% 成员一致，多数情况下凑不满，只好退回相似度判定，
        #    结果 5/6 两类都被判成 5、再由「优先未用」规则错位成 6 → 整体对调。
        if votes:
            top_val, top_cnt = max(votes.items(), key=lambda kv: kv[1])
            if top_cnt >= 2:
                # 闸①：投票标签必须**站得住字形**。票少（这里只有 3 票）而字形明显
                # 不支持时，多半是 OCR 错读或串位 —— 不采纳，退回②按字形定
                # （②的"优先未用数字"会把它判成 5，正好与真 5 的那类合流）。
                sim_voted = max((_corr(c["mean"], r) for r in refs.get(top_val, [])),
                                default=-1.0)
                if sim_voted >= VOTE_MIN_SIM:
                    label = top_val
                else:
                    vote_only = top_val
                    sims = sorted((max(_corr(c["mean"], r) for r in refs[d]), d)
                                  for d in refs)
                    print(f"  [fix] {len(c['idx'])} 块的类被 {top_cnt} 票判成 {top_val}，"
                          f"但字形只像 {top_val} 的 {sim_voted:.2f}"
                          f"（最像 {sims[-1][1]} {sims[-1][0]:.2f}）→ 不采纳投票，退回字形判定")
        # ② 没有 OCR 证据的类 → 按类平均字形比参照字形；优先取未用过的数字。
        #    门槛提到 0.55：把握不足就留空（显示 ?），不做「矮子里拔将军」——
        #    之前门槛太低导致两个不同形状的类都被判成 5，再被「优先未用」错位成 6。
        if label is None:
            scores = sorted(((max(_corr(c["mean"], r) for r in refs[d]), d)
                             for d in refs), reverse=True)
            if scores and scores[0][0] >= 0.55:
                for sc, d in scores:
                    if sc < 0.50:
                        break
                    if d not in used:
                        label = d
                        break
                if label is None:
                    label = scores[0][1]
        if label is None and vote_only is not None:
            # ②也完全没把握（连 0.55 都够不着）→ 退回投票标签，别把原来有数的位置变成 "?"
            # 实测《对照谱》那张：4 块类被 2 票判成 5、字形最像 6 只有 0.52，
            # 此时留 5 比留 ? 好（5 至少是 OCR 的原话，且会照旧标黄底提示人看一眼）。
            label = vote_only
        c["label"] = label
        if label is not None:
            used.add(label)
        vote_desc = ",".join(f"{k}:{v}" for k, v in sorted(votes.items(), key=lambda kv: -kv[1])[:3])
        summary.append((len(c["idx"]), label, vote_desc))

    # 全局一致性纠错：数字 0-7 应各占一类。若某数字完全缺席、而另一数字被**多个类**
    # 占用，则其中一个类必然是被误认成该数字的——用「哪个类更像缺席数字的参照字形」
    # 来改判。实测《采蘑菇的小姑娘》：5 被判成 6，于是 6 有两个类（101 个块），5 只剩 6 个。
    by_label = defaultdict(list)
    for c in clusters:
        if c["label"] is not None:
            by_label[c["label"]].append(c)
    used_labels = set(by_label.keys())
    missing = [d for d in refs.keys() if d not in used_labels]
    for label, group in by_label.items():
        if len(group) < 2 or not missing:
            continue
        # 块数多的那个类更可能是真 label；其余按「与缺席数字的相似度」再判
        group.sort(key=lambda c: -len(c["idx"]))
        for c in group[1:]:
            scores = sorted(((max(_corr(c["mean"], r) for r in refs[d]), d)
                             for d in missing), reverse=True)
            if scores and scores[0][0] >= 0.50:
                new_label = scores[0][1]
                c["label"] = new_label
                missing.remove(new_label)
                print(f"  [fix] 一类 {label}（{len(c['idx'])} 块）改判为 {new_label}"
                      f"（相似度 {scores[0][0]:.2f}；{label} 已有多类占用）")
                if not missing:
                    break

    # 闸②「弱证据类并入强证据类」：只治"证据很少、且自己的标签被字形否掉"的类。
    # 必须放在**上面那段一致性纠错之后**——并会让两个类共用同一个标签，而那段逻辑
    # 把"同一标签有两类"当成错位信号，放在它前面会被它再改回去。
    #
    # 实测《千年等一回》5.png：18 块那类被 3 票判成 1，与真 1 类互相关只有 0.100、
    # 与真 5 类却有 0.955 → 并入 5（27 个 5 全回来）。
    # 反向不并：图里那个 2 块的 6 类同样是弱证据，但它与 71 块的真 6 类互相关 0.716
    # （≥ 闸值）→ 标签有字形支持，不动它。
    def _top_votes(c):
        return max(c["votes"].values()) if c["votes"] else 0

    strong = [c for c in clusters
              if c["label"] is not None and _top_votes(c) >= MERGE_MIN_VOTES]
    for c in clusters:
        if c["label"] is None or _top_votes(c) >= MERGE_MIN_VOTES:
            continue                       # 证据够（或本来就没标签）→ 不进这道闸
        same = [o for o in clusters if o is not c and o["label"] == c["label"]]
        if same:
            supported = max(_corr(c["mean"], o["mean"]) for o in same)
        else:
            supported = max((_corr(c["mean"], r) for r in refs.get(c["label"], [])),
                            default=-1.0)
        if supported >= MERGE_SAME_LABEL_MIN:
            continue                       # 自己这个标签站得住 → 不动
        hosts = [(_corr(c["mean"], o["mean"]), o) for o in strong
                 if o["label"] != c["label"]]
        if not hosts:
            continue
        best, host = max(hosts, key=lambda t: t[0])
        if best < MERGE_HOST_MIN:
            continue
        print(f"  [fix] 弱证据类（{len(c['idx'])} 块，投票 {c['votes']}）标签 {c['label']} "
              f"与同类仅 {supported:.2f}、与 {host['label']}（{len(host['idx'])} 块）"
              f"却 {best:.2f} → 并入 {host['label']}")
        c["label"] = host["label"]

    # 标签互换优化（2-opt）：对每一对类，若互换标签能让「类平均字形 ↔ 参照字形」
    # 的总相似度提高，就互换。**相对比较**，不依赖绝对阈值——专门治
    # 「5/6 这类易混对被大批判错」（字体差异使绝对阈值不可靠）的问题。
    def _sim(c, d):
        rs = refs.get(d)
        return max((_corr(c["mean"], r) for r in rs), default=-1.0) if rs else -1.0

    # 标签互换优化（2-opt）：由 `ENABLE_LABEL_SWAP` 开关，**当前是开的**
    # （用户 2026-09-18 要求开启；原来那句"默认停用"的注释与开关取值相反，已改）。
    # 它依据「类平均字形 ↔ 参照字形」的总相似度来成对互换标签，实测两次出问题
    # （6↔0 碎类误换、上海滩个别 5 被判成 6），收益不确定 —— 开着的代价就是这两类风险。
    # ⚠ 这行 print 曾用 `↔` 字符：GBK 控制台下会抛 UnicodeEncodeError，**整张图跑不完**
    # （用户 2026-09-20 的一次 12 张批处理里 7.png / 两只老虎.png 就是这样断掉的）。
    if ENABLE_LABEL_SWAP:
        labeled = [c for c in clusters if c["label"] is not None and len(c["idx"]) >= 4]
        rounds = 0
        improved = True
        while improved and rounds < 6:
            improved = False
            rounds += 1
            for i in range(len(labeled)):
                for j in range(i + 1, len(labeled)):
                    a, b = labeled[i], labeled[j]
                    la, lb = a["label"], b["label"]
                    if la is None or lb is None or la == lb:
                        continue
                    now = _sim(a, la) + _sim(b, lb)
                    swp = _sim(a, lb) + _sim(b, la)
                    if swp > now + LABEL_SWAP_MARGIN:
                        a["label"], b["label"] = lb, la
                        improved = True
                        print(f"  [fix] 互换标签 {la} <-> {lb}"
                              f"（两类共 {len(a['idx'])}/{len(b['idx'])} 块，"
                              f"相似度和 {now:.2f}→{swp:.2f}）")

    # 说明：曾加过一条「5/6 顺序纠错」（比较两类的"顶部横杠"特征，若标为 5 的类
    # 反而更弱就互换）。实测《采蘑菇的小姑娘》该判据的前提不成立（那种字体的 6
    # 顶部也是平的），互换后 5/6 整体对调、反而更差，故移除。
    # 5/6 在这种"5 与 6 字形极近"的字体上仍可能混淆，属已知限制。

    changed = 0
    for c in clusters:
        if c["label"] is None:
            continue
        for i in c["idx"]:
            ri, bi, _old = items[i]
            if row_records[ri]["values"][bi] != c["label"]:
                row_records[ri]["values"][bi] = c["label"]
                changed += 1
    top = sorted(summary, key=lambda t: -t[0])[:12]
    print(f"[ocr] 字形聚类 {len(clusters)} 类，按类改判 {changed} 个块；"
          f"主要类（块数→标签 [OCR投票]）："
          + " ".join(f"{n}→{l}[{v}]" for n, l, v in top))
    return changed


def _fmt_num(value, digits=2):
    """把可能是 None 的数格式化成字符串（只在 VERBOSE_DEBUG 打印里用）。"""
    return "?" if value is None else f"{value:.{digits}f}"


def _judge_zero_or_six(block, binary, H=None, return_detail=False):
    """用两条与字体无关的特征判 0 / 6；两者结论一致才返回，否则 None。

    ① 上下质量分布：0 上下对称（上半与下半墨量接近），6 下重上轻；
    ② 内孔位置：0 的封闭孔在字形正中，6 的孔在下半部。

    只用于复核当前标为 0 / 6 的块——两者不一致时保持原值不动（宁可不改）。

    return_detail=True 时返回 `(结论, 明细 dict)`，明细含：
      balance 上下质量差 / sym 对称判 / hole 内孔判 / top,bot 上下半墨密度原值 /
      h_ratio 块高 ÷ 字高。
    **为什么要打 top/bot/h_ratio**：实测《南泥湾》32 个 0/6 块的 `balance` 全都
    只有 0.00~0.09（阈值 0.12），连标着 6 的也一样 → `sym` 恒为 0，双判据退化成
    单判据，于是"内孔判"一家说了算，把 5 个真 6 改成了 0。
    最可能的解释是 bbox 把数字上/下别的墨（减时线、音区点）也框了进来，
    上下两半被摊平；`h_ratio` 明显 >1 就能证实，`top/bot` 的原值能看出哪一半被抬高。
    """
    def _result(vote, balance=None, sym=None, hole=None, top=None, bot=None,
                hole_detail=None):
        if not return_detail:
            return vote
        return vote, {
            "balance": balance, "sym": sym, "hole": hole, "top": top, "bot": bot,
            "h_ratio": (block["h"] / H) if H else None,
            "hole_detail": hole_detail,
        }

    roi = binary[block["y"]:block["y"] + block["h"],
                 block["x"]:block["x"] + block["w"]] > 0
    if roi.size == 0:
        return _result(None)
    h, w = roi.shape
    if h < 8 or w < 5:
        return _result(None)
    mh = max(1, h // 2)
    top = float(roi[:mh].mean())
    bot = float(roi[mh:].mean())
    denom = max(top + bot, 1e-6)
    balance = abs(top - bot) / denom
    if balance <= 0.12:
        sym_vote = 0        # 上下接近 → 像 0
    elif bot > top:
        sym_vote = 6        # 下重上轻 → 像 6
    else:
        sym_vote = None     # 上重下轻 → 既不像 0 也不像 6（可能是别的字）
    if return_detail:
        hole_vote, hole_detail = _hole_class(block, binary, return_detail=True)
    else:
        hole_vote, hole_detail = _hole_class(block, binary), None   # 孔居中→0，孔偏下→6
    if sym_vote is None or hole_vote is None or sym_vote != hole_vote:
        result = None
    else:
        result = sym_vote
    return _result(result, balance, sym_vote, hole_vote, top, bot, hole_detail)


def _repeat_mark(text):
    """从 OCR 文本识别反复记号，返回 (类型, 冒号在文本中的字符下标) 或 None。

    简谱里冒号只用于反复记号，因此**只要出现冒号就算**（OCR 常把反复记号与
    后面的音符并成一个文本块，如 `0 35 l: 6 - - 35`，不能要求冒号前后没内容）。
    'l:' / '|:' 形如 ‖: → 反复开始；冒号在末尾 → :‖ 反复结束。
    """
    if not text:
        return None
    t = text.replace("：", ":").replace("｜", "|").replace("‖", "|")
    if ":" not in t:
        return None
    idx = t.index(":")
    before = t[:idx].strip()
    after = t[idx + 1:].strip()
    if before and before[-1] in "lLiI|":
        return ("forward", idx)      # ‖: 反复开始
    if after == "":
        return ("backward", idx)     # :‖ 反复结束
    if before == "":
        return ("forward", idx)
    return None


def _detect_repeats_geom(row_components, H):
    """几何识别反复记号，返回 [(x, kind)]，kind ∈ {'forward','backward'}。

    依据：反复记号是**相邻两条细竖线**（间距 <0.6 字高、高度相近）+ **旁边两个点**。
    ：‖（结束）点在左侧；‖:（开始）点在右侧。
    """
    bars = [c for c in row_components
            if c["w"] <= 0.30 * H and c["h"] >= 1.10 * H]
    bars.sort(key=lambda c: c["x"])
    dots = [c for c in row_components
            if 0.008 * H * H <= c["area"] <= 0.30 * H * H
            and 0.40 <= c["w"] / max(c["h"], 1) <= 1.90
            and c["w"] <= 0.42 * H and c["h"] <= 0.42 * H]

    marks = []
    used = set()
    for i, b1 in enumerate(bars):
        if i in used:
            continue
        for j in range(i + 1, len(bars)):
            if j in used:
                continue
            b2 = bars[j]
            gap = b2["x"] - (b1["x"] + b1["w"])
            if gap > 0.6 * H:
                break
            if gap < 0 or abs(b1["h"] - b2["h"]) > 0.4 * H:
                continue
            cy = (b1["y"] + b1["h"] / 2.0 + b2["y"] + b2["h"] / 2.0) / 2.0
            left = sum(1 for p in dots
                       if abs(p["y"] + p["h"] / 2.0 - cy) <= 0.5 * H
                       and b1["x"] - 0.9 * H <= p["x"] + p["w"] / 2.0 <= b1["x"] - 0.05 * H)
            right = sum(1 for p in dots
                        if abs(p["y"] + p["h"] / 2.0 - cy) <= 0.5 * H
                        and b2["x"] + b2["w"] + 0.05 * H
                        <= p["x"] + p["w"] / 2.0 <= b2["x"] + b2["w"] + 0.9 * H)
            if max(left, right) < 1:
                # 相邻两条细竖线、但两侧都没有点 → 终止/双小节线（‖）
                marks.append(((b1["x"] + b2["x"] + b2["w"]) / 2.0, "double"))
                used.add(i)
                used.add(j)
                break
            kind = "backward" if left >= right else "forward"
            marks.append(((b1["x"] + b2["x"] + b2["w"]) / 2.0, kind))
            used.add(i)
            used.add(j)
            break
    return marks


def _row_has_end_bar(row, components, H):
    """这一行是不是**以终止双小节线 `‖` 收尾**（供"末行放宽"判断谱子是否已结束）。

    直接用现成的 `_detect_repeats_geom()`：它把"相邻两条细竖线、两侧都没有点"判成
    `double`。实测 19.png 末行返回 `[(869.5, 'double')]`、倒数第二行返回 `[]`。

    ⚠ 光有 `double` **不够**：`‖` 也可能是**行中间的记号**而不是曲终。19.png 实测
    第 14 行在 x=1923 有一对竖线、下面印着 `D.S.`（返回记号），后面还有整整一小节
    的谱 —— 那时若认作"结束"，末行的放宽就永远不会生效（第一版就是这么失效的）。
    故附加一条：`‖` 必须落在**本行最后一个数字块之后**（行尾），才是真曲终。

    行带取本行块的上下各 1.0H —— `‖` 的两条竖线比数字高（实测 2.1H），必然落在带内。
    """
    blocks = row["blocks"]
    y0 = min(b["y"] for b in blocks)
    y1 = max(b["y"] + b["h"] for b in blocks)
    right = max(b["x"] + b["w"] for b in blocks)
    band = [c for c in components
            if c["y"] < y1 + 1.0 * H and c["y"] + c["h"] > y0 - 1.0 * H]
    return any(kind == "double" and x >= right - 0.5 * H
               for x, kind in _detect_repeats_geom(band, H))


def _paren_looks_like_note(c, binary, dots, H):
    """括号候选的三条否决：**是竖笔**、**像数字**、或**上方有音区点** → 它不是括号。

    为什么需要（用户 2026-09-21 连报三次"17.jpg 的数字 `1` 被识别成 `(`/`)`"）：
    细高的 `1`（实测 w15×h49、填充 0.40~0.44）与 `(`/`)` 在尺寸/填充上**几乎一样**，
    只能靠"像不像数字"分 —— 真括号对分类器来说谁都不像（14.jpg 那个真 `(` 只有 0.48），
    而 `1` 很像 `1`（实测 0.80~0.92，多数越过 0.85 门槛）。
    另外**括号不可能带音区点**，而 `i` 形高音 1 必定带 → 这条兜住剩下那几个。

    ⚠ 必须在**所有**产生括号的地方生效：本文件原来有**两份**括号列表
    （`_detect_paren_components()` 一份 + 行内旧实现一份），后者**覆盖**前者 ——
    先前只改函数那份 → 被覆盖 → 三次修改都没生效，最后靠"每条标记来源"的探针才定位到。

    这里只放"只看**这个块自己**"的否决。还有一条要看**上下文**的（④ 纵向必须在数字行带内，
    见 `PAREN_MAX_DY_RATIO`）没法写进本函数 —— 它需要行的基线 y —— 所以挂在两个调用处。
    """
    if binary is not None:
        # ① **先看直不直**：细高的 `1` 与 `(` 在尺寸/填充上完全重叠，但 `1` 是一条
        #    竖笔、`(` 是一段弧（判据与实测见 `PAREN_STEM_MIN_RATIO`）。这一条**不看
        #    分类器**，所以那些"字形变体让分类器给不出分"的 `1`（实测 0.60~0.74，
        #    例如 19.png/20.png 那种带长斜旗的 `1`）也能兜住。
        if _max_stem_run(c, binary) >= PAREN_STEM_MIN_RATIO:
            return True
        d, sc = _classify_by_reference(c, binary, _all_reference_glyphs(),
                                       return_score=True)
        if d is not None and sc >= PAREN_MAX_DIGIT_SCORE:
            return True
    if dots:
        cx = c["x"] + c["w"] / 2
        if any(abs(p["x"] + p["w"] / 2 - cx) <= 0.35 * H
               and c["y"] - 0.9 * H <= p["y"] + p["h"] / 2 <= c["y"] + 0.1 * H
               for p in dots):
            return True
    return False


def _detect_paren_components(row_components, blocks, H, binary=None, dots=None):
    """找出「括号形状」的连通块（细、高、填充率低），且不与**别的**数字块重叠。

    注：曾把高度上限放到 3.5H、宽高比放宽到 0.10-0.55，想救《枉凝眉》行首那个
    与小节线等高的 `(`；结果在别的谱上冒出大量**假括号**（2.png 实测），而那个
    真括号因为与后面的数字粘连成一个连通块、形状已不是独立弧线，放宽也救不了。
    故回退到原判据（1.9H、0.20-0.50）。

    ⚠ 2026-09-21 修两处（用户报 14.jpg 行首 `(` 没认出来、渲染成了 `?`）：
      ① **重叠检测必须排除自己**：真 `(` 常常**本身就是数字候选块**（实测那个 `(`
         高 1.12H、宽高比 0.36，两条都落在数字候选的窗口里），原来的
         `any(... for b in blocks)` 于是拿它跟自己比 → 重叠面积 **1.00** → 被自己
         否掉。实测它与**别的**块重叠面积是 0.00。
      ② **够细的弧允许矮到 1.0H**：那个 `(` 高 1.12H，差 7% 没进 1.2H 门槛；而它的
         填充率 **0.39** 与同行数字碎片（0.54~0.62）差得很开 —— 用"细"换"可矮一点"。

    ⚠ 2026-09-22 加 ③：**纵向必须在数字行带内**（阈值与实测见 `PAREN_MAX_DY_RATIO`）。
    """
    baseline_y = None
    if blocks:
        baseline_y = (float(np.median([b["y"] for b in blocks]))
                      + float(np.median([b["y"] + b["h"] for b in blocks]))) / 2
    out = []
    for c in row_components:
        fill = c["area"] / max(c["w"] * c["h"], 1)
        min_h = 1.0 if fill <= 0.42 else 1.2
        if not (min_h * H <= c["h"] <= 2.2 * H):
            continue
        if not (0.15 * H <= c["w"] <= 0.50 * H):
            continue
        if not (0.20 <= c["w"] / max(c["h"], 1) <= 0.50):
            continue
        if fill >= 0.45:
            continue
        if (baseline_y is not None
                and abs(c["y"] + c["h"] / 2 - baseline_y) > PAREN_MAX_DY_RATIO * H):
            continue
        if any(c is not b and c["x"] < b["x"] + b["w"] and b["x"] < c["x"] + c["w"]
               and c["y"] < b["y"] + b["h"] and b["y"] < c["y"] + c["h"]
               for b in blocks):
            continue
        if _paren_looks_like_note(c, binary, dots, H):
            continue
        out.append(c)
    return out


def _overlaps_any(block, others):
    return any(block["x"] < o["x"] + o["w"] and o["x"] < block["x"] + block["w"]
               and block["y"] < o["y"] + o["h"] and o["y"] < block["y"] + block["h"]
               for o in others)


def _recognize_blocks_directly(engine, blocks, binary):
    """直接调用 OCR 的**识别器**（跳过检测器）逐块认数字。

    为什么这样可行：RapidOCR 默认先"检测文字框"再识别，孤立的单个数字会被检测器
    漏掉（这是早期"孤立数字块 OCR 全失败"的真正原因）。而识别器本身可以独立调用，
    把数字块裁出来留白放大后直接喂进去，就没有"检测"这一环可以失败了。
    """
    rec = getattr(engine, "text_recognizer", None)
    if rec is None:
        return None   # 该版本不暴露识别器 → 调用方跳过
    crops = []
    for b in blocks:
        pad = max(6, int(0.35 * b["h"]))
        x0 = max(0, b["x"] - pad)
        x1 = min(binary.shape[1], b["x"] + b["w"] + pad)
        y0 = max(0, b["y"] - pad)
        y1 = min(binary.shape[0], b["y"] + b["h"] + pad)
        roi = 255 - binary[y0:y1, x0:x1]        # 黑字白底
        scale = 48.0 / max(b["h"], 1)
        roi = cv2.resize(roi, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_CUBIC)
        roi = cv2.copyMakeBorder(roi, 12, 12, 12, 12,
                                 cv2.BORDER_CONSTANT, value=255)
        crops.append(cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR))
    try:
        outs, _elapse = rec(crops)
    except Exception:  # noqa: BLE001
        return None
    results = []
    for out in outs or []:
        text = out[0] if isinstance(out, (list, tuple)) and out else ""
        score = float(out[1]) if isinstance(out, (list, tuple)) and len(out) > 1 else 0.0
        slots = _digit_slots(str(text), allow_confusion=True)
        results.append((slots[0] if slots else None, score))
    if len(results) < len(blocks):
        results += [(None, 0.0)] * (len(blocks) - len(results))
    return results


def _arc_variation(block, binary):
    """弧线的「弯曲度」：逐行墨迹中心 x 的极差 ÷ 宽度。

    真括号是一段弧（`(` 中段右凹、上下端偏左），逐行中心会明显移动 → 约 0.3-0.6；
    竖线、数字笔画碎片是直的 → 约 0。用它把"直的碎片"从括号候选中剔除。
    """
    x, y, w, h = block["x"], block["y"], block["w"], block["h"]
    roi = binary[y:y + h, x:x + w] > 0
    if roi.size == 0 or h < 6 or w < 3:
        return 0.0
    centers = []
    for row in roi:
        nz = np.nonzero(row)[0]
        if len(nz):
            centers.append(float(nz.mean()))
    if len(centers) < 4:
        return 0.0
    return (max(centers) - min(centers)) / max(w, 1)


def _max_stem_run(block, binary):
    """块内「最长逐列连续墨」÷ 块高 = 这块里**最像竖笔的那一列**有多长。

    数字 `1` 就是一条竖笔 → ≈1；括号是一段弧 → 明显短。用于把「细 + 填充率低」
    的块分成"数字 1"与"括号"（阈值见 `PAREN_STEM_MIN_RATIO` 处的实测）。
    """
    roi = binary[block["y"]:block["y"] + block["h"],
                 block["x"]:block["x"] + block["w"]] > 0
    best = 0
    for col in roi.T:
        run = cur = 0
        for v in col:
            cur = cur + 1 if v else 0
            run = max(run, cur)
        best = max(best, run)
    return best / max(block["h"], 1)


def _hump(block, binary):
    """**横**弧的「拱起程度」：逐列墨心 y 的极差 ÷ 高。

    专治连音弧 `⌒`（横着的弧）。`_arc_variation()` 量的是逐行墨心 x，是给**竖**弧
    （括号）用的——对 `⌒` 算出来接近 0，会被当成直线，所以必须另开一个量。

    真 `⌒`：中间列墨心靠上、两端列墨心靠下 → 极差大（≈0.4~1.0）；
    延时线 `-` / 减时线 / 小节线等**直的**横墨 → 每列墨心 y 几乎一致 → ≈0。
    """
    roi = binary[block["y"]:block["y"] + block["h"],
                 block["x"]:block["x"] + block["w"]] > 0
    if roi.size == 0:
        return 0.0
    centers = []
    for col in roi.T:
        nz = np.nonzero(col)[0]
        if len(nz):
            centers.append(float(nz.mean()))
    if len(centers) < 4:
        return 0.0
    return (max(centers) - min(centers)) / max(roi.shape[0], 1)


def _y_overlap(a, b):
    """两个块的**纵向重叠比例**（相对较矮的那个，0..1）。

    只用来判断"两个弧块是不是同一条笔画的两截"：同一笔画的两截纵向几乎完全重叠，
    而上下叠着的两条独立弧只有少量重叠（《孤勇者》行9 实测 26%）。
    """
    lo = max(a["y"], b["y"])
    hi = min(a["y"] + a["h"], b["y"] + b["h"])
    if hi <= lo:
        return 0.0
    return (hi - lo) / max(min(a["h"], b["h"]), 1)


def _ascii_mask(mask, step=2):
    """把一小块二值图打成 ASCII（每 step×step 像素一格）。

    用途：像"附点被当成 `-`"这类问题，光看宽度/墨心那些**汇总数字**猜不出
    那团墨到底长什么样；把形状打出来一眼就清楚了。
    """
    h, w = mask.shape
    lines = []
    for y0 in range(0, h, step):
        row = []
        for x0 in range(0, w, step):
            blk = mask[y0:y0 + step, x0:x0 + step]
            row.append("#" if blk.any() else ".")
        lines.append("".join(row))
    return lines


def _bottom_bar(block, binary):
    """底部「基线横杠」的占比：底部 12% 的行里，**有墨的列**占多少。

    用来分开**房子标号（小数字）**与 **`v` 换气记号**：数字底部有一条**满宽基线**
    → 占比接近 1；`v` 底部**收成一个尖** → 占比很小。
    从《上海滩》的 ASCII 形状图读出来：房子标号 `2` 的底部是整行满墨、`v` 底部只有
    两三个字符宽。（这两个值还没做多样本统计，见 README 7.3 的"阈值是暂定"说明。）
    """
    roi = binary[block["y"]:block["y"] + block["h"],
                 block["x"]:block["x"] + block["w"]] > 0
    if roi.size == 0:
        return 0.0
    rows = max(1, int(round(0.12 * roi.shape[0])))
    return float(roi[-rows:].any(axis=0).mean())


def _find_wide_bracket(all_components, block, H):
    """找到这个记号旁边那条**很宽的房子括线**（否则返回 None）。

    用途有两层：① 分开 **`³` 三连音** 与 **房子标号 `3.`**——两者是**同一个字形**（小 3），
    字形上无法区分，只能靠上下文（房子标号旁边必有那条 ≥`HOUSE_BRACKET_MIN_W`H 宽的
    房子括线，《上海滩》实测 20.5H）；② 顺手拿到括线的**横向范围**，用来定"房子横线"
    画在哪几个音上（`house_start` / `house_stop`）。
    """
    y_mid = block["y"] + block["h"] / 2.0
    x0, x1 = block["x"], block["x"] + block["w"]
    for other in all_components:
        if other is block or other["w"] < HOUSE_BRACKET_MIN_W * H:
            continue
        if abs((other["y"] + other["h"] / 2.0) - y_mid) > 0.8 * H:
            continue
        if x1 >= other["x"] - 3.0 * H and x0 <= other["x"] + other["w"] + 3.0 * H:
            return other
    return None


def _stroke_drift(block, binary):
    """主笔画的**横向倾斜度**：中下段墨心与中上段墨心的横向差 ÷ 宽（取绝对值）。

    用来分开小字号字体里的 `1` 与 `2`（用户 2026-09-18 确认《上海滩》行 3 是 `1.`，
    而形状相似度分类把它认成了 `2`——这个字体把 `1` 写得带夸张左上旗 + 满宽底衬线，
    整体轮廓很像 `2`）。**但两者的主笔方向根本不同**：
      `1` 的主笔**竖直** → 上下墨心几乎对齐；`2` 是一条**斜线** → 漂移很大。
    实测：`1` ≈ **0.21**、`2` ≈ **0.56**（差 2.7 倍）。
    ⚠ 只取**中上段 / 中下段**，跳过底部的满宽基线——`1` 和 `2` 的底部都有横杠，
    把它算进来会把上下两段都拉平、把差别抹掉。
    """
    roi = binary[block["y"]:block["y"] + block["h"],
                 block["x"]:block["x"] + block["w"]] > 0
    h, w = roi.shape
    if h < 8 or w < 3:
        return 0.0

    def _cx(band):
        cols = np.nonzero(band.any(axis=0))[0]
        return float(cols.mean()) if len(cols) else None

    cu = _cx(roi[int(0.30 * h):max(int(0.30 * h) + 1, int(0.50 * h))])
    cd = _cx(roi[int(0.60 * h):max(int(0.60 * h) + 1, int(0.80 * h))])
    if cu is None or cd is None:
        return 0.0
    return abs(cd - cu) / max(w, 1)


_OCR_ENGINE = None       # OCR 引擎单例（见 `_make_ocr_engine`）


def _make_ocr_engine():
    """返回 OCR 引擎（**进程内复用同一个**）。

    以前是每次调用都 `RapidOCR()` 新建。省下的**不是那 0.5~0.8s 的构造**（构造是懒加载的），
    而是**首次推理时才付的 onnxruntime session 初始化**：实测同一张图，新建引擎 ~25s、
    复用引擎 ~10.5s，**每张图白付约 15s**（40 张 regress 全量因此从 19 分钟降到 13 分钟）。
    ⚠ 排查时别信 cProfile 的归因：那段 session 初始化会被算到"检测模型 `text_detect.__call__`"
    头上，看起来像"检测慢"，其实只发生一次。
    RapidOCR 实例无状态，同一实例喂不同图不会互相污染，所以缓存起来即可。
    """
    global _OCR_ENGINE
    if _OCR_ENGINE is None:
        from rapidocr_onnxruntime import RapidOCR
        _OCR_ENGINE = RapidOCR()
    return _OCR_ENGINE


def _ocr(engine, image):
    out = engine(image)
    return (out[0] if isinstance(out, tuple) else out) or []


def _detect_key_mark(engine, color):
    """从页眉区块读调号「1=X」。

    整页 OCR 会把整图缩到「数字高≈48px」，页眉那点小字号被缩没（实测 23.png 整页读不到
    `1=♭B`），所以**单独裁页眉小块**再 OCR。单档裁剪不稳（见 `_KEY_CROPS` 的说明），
    故**多档裁剪取并集**，再按下述规则挑一个：
      ① 只留合法调号（`_VALID_KEYS`）；
      ② 优先带升降号的（宽裁剪常把 `♭` 丢成无升降号）；
      ③ 再按出现次数多的。
    全都没读到返回 None（调用方应提示用户手填）。
    """
    if color is None:
        return None
    h, w = color.shape[:2]
    flat = ("♭", "b", "'", "’", "`")     # OCR 常把降号读成 b / 撇号 / 反引号
    sharp = ("♯", "#")
    cands = []
    for y_frac, x_frac, cap in _KEY_CROPS:
        crop = color[0:int(h * y_frac), 0:int(w * x_frac)]
        if crop.size == 0:
            continue
        fx = min(6.0, max(1.0, cap / max(crop.shape[1], 1)))
        if abs(fx - 1.0) > 1e-6:
            crop = cv2.resize(crop, None, fx=fx, fy=fx, interpolation=cv2.INTER_CUBIC)
        text = " ".join(str(it[1]) for it in _ocr(engine, crop))
        m = _KEY_RE.search(text)
        if not m:
            continue
        acc, letter, trailing = m.group(1), m.group(2).upper(), m.group(3)
        acc = acc or trailing
        if acc in flat:
            key = letter + "b"
        elif acc in sharp:
            key = letter + "#"
        else:
            key = letter
        if key in _VALID_KEYS:
            cands.append(key)
    if not cands:
        return None
    best = sorted(set(cands), key=lambda k: (-(len(k) > 1), -cands.count(k)))[0]
    return f"1={best}"


def _detect_title(engine, color):
    """从页眉读曲名。

    标题是页眉最上方的**居中大字**，所以裁**顶部 20% 整宽**再 OCR（窄裁剪会把居中标题
    切断，如"上海滩"→"上海"）。在结果里取「**居中**、最大、且不含署名/副标题字样的中文块」
    当标题。实测 22 张中 20 张正确、0 张取错、2 张留空（OCR 根本没检出标题字）。
    读不到返回 None（调用方回退到文件名，并让用户手填）。
    """
    if color is None:
        return None
    h, w = color.shape[:2]
    y_frac, x_frac, cap = _TITLE_CROP
    crop = color[0:int(h * y_frac), 0:int(w * x_frac)]
    if crop.size == 0:
        return None
    fx = min(6.0, max(1.0, cap / max(crop.shape[1], 1)))
    if abs(fx - 1.0) > 1e-6:
        crop = cv2.resize(crop, None, fx=fx, fy=fx, interpolation=cv2.INTER_CUBIC)
    best = None
    cw = max(crop.shape[1], 1)
    for it in _ocr(engine, crop):
        if not (isinstance(it, (list, tuple)) and len(it) >= 2):
            continue
        text = str(it[1])
        if not _contains_cjk(text):
            continue
        if any(k in text for k in _TITLE_SKIP) or "《" in text or "》" in text:
            continue
        # 标题是**居中**的；靠左右边的（署名、页码）一律不要——宁可留空让用户手填，
        # 也不要把署名当标题写错（实测这一条把 2 张"错标题"变成了"空"）。
        xs = [p[0] for p in it[0]]
        if not (0.25 <= (min(xs) + max(xs)) / 2 / cw <= 0.75):
            continue
        ys = [p[1] for p in it[0]]
        hgt = max(ys) - min(ys)
        if best is None or hgt > best[0]:
            best = (hgt, text)
    return best[1] if best else None


# OCR 在小号/衬线字体下常见的「数字误读」。只在**纯文本块**上启用还原
# （含中文的块不启用，否则歌词里的字会被误当数字）。
_OCR_DIGIT_MAP = {
    "l": 1, "I": 1, "i": 1, "|": 1, "!": 1, "1": 1,
    "Z": 2, "z": 2, "2": 2,
    "E": 3, "3": 3,
    "A": 4, "4": 4,
    "S": 5, "s": 5, "5": 5, "$": 5,
    "b": 6, "B": 6, "G": 6, "6": 6,
    "T": 7, "7": 7,
    "O": 0, "o": 0, "D": 0, "Q": 0, "0": 0, "8": None, "9": None,
}


_ASCII_DIGITS = "0123456789"
_FULLWIDTH_DIGITS = "０１２３４５６７８９"


def _is_digit_char(ch):
    """这个字符是不是「`int()` 能认的数字」（ASCII 0-9，或全角 ０-９）。

    ⚠ **不要用 `ch.isdigit()`**：Unicode 里 `³`(U+00B3)、`²`、`⁴` 这类**上标数字**
    的 `isdigit()` **也返回 True**，但 `int('³')` 会抛 `ValueError`。
    简谱的**三连音记号 `³`** 会被 OCR 当字符 `³` 原样输出（35.png / 39.png 实测），
    于是 `if ch.isdigit(): value = int(ch)` 在这里**直接崩掉整个文件**——
    一个字符就把整张图废掉（不是某个音错，是**零产物**）。
    判据只认"能转成 int 的那些"，`int('１')` 是可以的，`int('³')` 不行。
    """
    return ch in _ASCII_DIGITS or ch in _FULLWIDTH_DIGITS


def _digit_slots(text, allow_confusion=False):
    """把 OCR 文本转成「数字槽位」序列。

    allow_confusion=True 时启用常见误读还原（|→1、S→5、O→0…），
    只用于不含中文的纯文本块。
    """
    if not text:
        return []
    text = text.translate(str.maketrans("０１２３４５６７８９", "0123456789"))
    slots = []
    i = 0
    while i < len(text):
        ch = text[i]
        if _is_digit_char(ch):
            value = int(ch)
            slots.append(value if 0 <= value <= 7 else None)
        elif ch in "ilI":
            nxt = text[i + 1] if i + 1 < len(text) else ""
            if nxt not in ":：.．":
                slots.append(1)
            else:
                i += 1
        elif allow_confusion and ch in _OCR_DIGIT_MAP:
            slots.append(_OCR_DIGIT_MAP[ch])
        i += 1
    return slots


def _contains_cjk(text):
    return any("㐀" <= ch <= "鿿" for ch in text)


def _corr(a, b):
    aa = a - a.mean()
    bb = b - b.mean()
    denom = float(np.sqrt(np.sum(aa * aa) * np.sum(bb * bb)))
    return float(np.sum(aa * bb) / denom) if denom > 1e-8 else -1.0


def _shape_feature(block, binary):
    """数字块等比例居中到 24×32，避免宽窄数字被拉伸。"""
    roi = binary[
        block["y"]:block["y"] + block["h"],
        block["x"]:block["x"] + block["w"],
    ]
    target_w, target_h = 24, 32
    scale = min(20 / max(block["w"], 1), 28 / max(block["h"], 1))
    nw = max(1, int(round(block["w"] * scale)))
    nh = max(1, int(round(block["h"] * scale)))
    resized = cv2.resize(roi, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target_h, target_w), dtype=np.uint8)
    x0 = (target_w - nw) // 2
    y0 = (target_h - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas.astype(np.float32).reshape(-1) / 255.0


def _paren_type(binary, c):
    """判断括号连通块是 "(" 还是 ")"：弧线在哪侧，哪侧笔画像素更多。"""
    roi = binary[c["y"]:c["y"] + c["h"], c["x"]:c["x"] + c["w"]]
    if roi.size == 0:
        return "("
    mid = max(1, c["w"] // 2)
    left = int(roi[:, :mid].sum())
    right = int(roi[:, mid:].sum())
    return "(" if left >= right else ")"


def _hole_class(block, binary, return_detail=False):
    """用「内孔」位置区分 0 / 6（最可靠的形状特征）。

    0 的封闭内孔在字形正中；6 的内孔在下半部；4 的孔在上半部（不判）。
    简谱不会出现 8/9，故只需区分 0 / 6 / 其它。

    return_detail=True 时返回 `(结论, 明细 dict)`，明细含：
      ratio 内孔面积占块面积的比例 / cy 内孔中心相对高度（0=顶、1=底）/
      cx 内孔中心相对水平位置 / fill 无孔时的墨填充率。**只为 VERBOSE_DEBUG 打印。**

    ⚠ 判 0 与判 6 的**区间仍然重叠**：判 0 用 `|cy - 0.5| <= 0.16`（覆盖 cy ∈ [0.34, 0.66]），
    判 6 用 `cy >= 0.60`。实测《南泥湾》里 0 和 6 的 `cy` 都是 0.63~0.67、**完全重叠**，
    所以 `cy` 在那个字体上**没有区分力**——真正把两者分开的是 `ratio`（内孔大小）：
    真 0 的孔比 0.23，6 只有 0.10~0.12，见 `HOLE_ZERO_MIN_RATIO`。
    换字体时要留意：若某个字体的 6 也是大孔，这个重叠就会真的咬人。
    """
    def _out(vote, ratio=None, cy=None, cx=None, fill=None):
        if not return_detail:
            return vote
        return vote, {"ratio": ratio, "cy": cy, "cx": cx, "fill": fill}

    x, y, w, h = block["x"], block["y"], block["w"], block["h"]
    roi = binary[y:y + h, x:x + w]
    if roi.size == 0 or w < 6 or h < 8:
        return _out(None)
    bg = 255 - roi  # 背景=255，笔画=0
    padded = cv2.copyMakeBorder(bg, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=255)
    mask = np.zeros((padded.shape[0] + 2, padded.shape[1] + 2), np.uint8)
    cv2.floodFill(padded, mask, (0, 0), 0)
    holes = padded == 255  # 被笔画包裹、未被外部连通填充到的区域
    area = int(holes.sum())
    ratio = area / float(w * h)
    if ratio < 0.06:
        # 无内孔：部分字体把休止符印成实心椭圆，用「高填充 + 四向对称」判定。
        ink = roi > 0
        fill = float(ink.mean())
        if fill >= 0.62:
            mh, mw = max(1, h // 2), max(1, w // 2)
            top, bot = float(ink[:mh].mean()), float(ink[mh:].mean())
            left, right = float(ink[:, :mw].mean()), float(ink[:, mw:].mean())
            if abs(top - bot) / fill <= 0.25 and abs(left - right) / fill <= 0.25:
                return _out(0, ratio=ratio, fill=fill)
        return _out(None, ratio=ratio, fill=fill)
    ys, xs = np.nonzero(holes)
    cy_rel = (float(ys.mean()) - 2) / h
    cx_rel = (float(xs.mean()) - 2) / w
    if (ratio >= HOLE_ZERO_MIN_RATIO and abs(cy_rel - 0.5) <= HOLE_ZERO_MAX_DCY
            and abs(cx_rel - 0.5) <= 0.18):
        return _out(0, ratio, cy_rel, cx_rel)
    if cy_rel >= 0.60:
        return _out(6, ratio, cy_rel, cx_rel)
    return _out(None, ratio, cy_rel, cx_rel)


def _ext_window_end(block, H, next_left, next_bar, max_span):
    """延时线扫描窗口的**右端 x**：取三者最近 —— 下一个数字左缘 / 右邻小节线 / 兜底上限。

    抽出来是因为调用处要跑**两个窗口**（`max_span=EXT_SCAN_MAX_SPAN` 与旧值 6.0），
    再取条数多的那次；两边的边界算法必须完全一致，否则比较没有意义。
    """
    x_end = block["x"] + block["w"] + max_span * H
    if next_left != float("inf"):
        x_end = min(x_end, next_left - 0.05 * H)
    if next_bar != float("inf"):
        x_end = min(x_end, next_bar + 0.25 * H)
    return x_end


def _count_extends(block, binary, H, next_left, has_dot=False, row=None, note_no=None,
                   beams=None, next_bar=float("inf"), max_span=EXT_SCAN_MAX_SPAN,
                   quiet=False):
    """数延时线（"-"）的条数：在数字中线高度向右扫，数独立短横的段数。

    关键点：
      - 判定「该列是否有墨」用 any（横线只有几像素高，用比例会全判 0）；
      - 若该音有附点，扫描起点跳过附点，避免把附点当成一条 "-"；
      - 右端截在**下一个数字的左边缘**（延时线不会画进数字里）。
        曾截在「下一音符中心 - 0.5 字高」，把横线截断，导致永远只数出约 1 条。

    row / note_no **只用于 VERBOSE_DEBUG 打印**（第几行第几个音），不参与任何判定。
    加它是因为：[ext] 日志原先只有 x 坐标，而整谱里多行的行末 x 都挤在一起，
    没法把某条日志对回原谱的哪个 `-`，(2026-09-18 排查《我的中国心》时吃过这个亏)。
    """
    tag = ((f"行{row} 音{note_no} " if row is not None else "") +
           f"x={block['x']:.0f}")
    # quiet：同一次判定要跑两个窗口（见调用处"取条数多的那次"），第二次不打日志
    dbg = VERBOSE_DEBUG and not quiet
    cy = block["y"] + block["h"] / 2
    x_start = block["x"] + block["w"] + (0.78 if has_dot else 0.12) * H
    # 右端见 `_ext_window_end()`（下一个数字左缘 / 右邻小节线 / 兜底上限 三者最近）。
    x_end = _ext_window_end(block, H, next_left, next_bar, max_span)
    x0, x1 = int(x_start), int(x_end)
    # 高度带放宽到 ±0.60H：细延时线在二值化后常有纵向偏移/断裂，带太窄会整条漏掉
    #（实测只测到一条 `-` 的墨，其余几条完全没测到）。减时线在数字下方 0.75H 以外，
    # 放宽不会误收。
    y0 = int(cy - 0.60 * H)
    y1 = int(cy + 0.60 * H)
    img_h, img_w = binary.shape
    x0, x1 = max(0, x0), min(img_w, x1)
    y0, y1 = max(0, y0), min(img_h, y1)
    if x1 - x0 < 3 or y1 - y0 < 2:
        # 窗口被挤扁（数字右侧紧邻 `)`／`:||` 时会发生），数不出延时线。
        # 注意**必须返回元组**：调用方是 `extend, ext_xs = _count_extends(...)`，
        # 这里曾经只写 `return 0` → 一碰到这种音符就
        # `TypeError: cannot unpack non-iterable int object`，整个 OCR 直接崩。
        # 这里是**静默数出 0 条**的路径，日志里原本看不见，加了这行打印。
        if dbg:
            print(f"  [ext] {tag} 窗口退化（宽 {x1 - x0}px / 高 {y1 - y0}px）"
                  f" → 跳过，数 0 条")
        return 0, []

    band = binary[y0:y1, x0:x1] > 0
    # 留一份**未膨胀**的墨迹：ascii_mask 诊断要画真实形状，膨胀过的会胖一圈。
    # （`band` 下面会被 dilate 的返回值重新绑定，所以这里存引用就够，不必 copy。）
    band_ink = band
    # 横向膨胀：细延时线在二值化后常被打碎（实测出现「墨宽 0.3H 却分成 3 段」），
    # 先连起来再统计，否则每段都被当作噪点丢掉。
    gap_px = max(1, int(0.15 * H))
    kernel = np.ones((1, gap_px), np.uint8)
    band = cv2.dilate(band.astype(np.uint8), kernel, iterations=1) > 0
    # 只把「细而薄」的墨算作横线：纵向跨越超过 0.35 字高的列是竖直结构
    # （反复记号的双竖线、括号等），必须忽略——否则它们会被当成一条"-"，
    # 还会把探测窗口堵死（实测墨y 占满整个带子、墨宽仅 0.9H）。
    col_ink_h = band.sum(axis=0)  # 每列的墨迹高度（像素）
    ink = (col_ink_h > 0) & (col_ink_h <= 0.35 * H)
    # 诊断：被判为「竖向结构」（整列墨高 > 0.35H）的列数。>0 说明扫描窗口里
    # 混进了竖线/括号这类贯穿型墨——正是「延时线少数」的嫌疑来源。
    # （右端截在"下一个数字左缘"，所以夹在 `-` 和下一个数字之间的 `)`／`|`
    #  ／`:||` 全都在窗口内部。）
    tall_cols = int((col_ink_h > 0.35 * H).sum())

    # 诊断：墨在带内的纵向分布（判断是否落在带外）
    ys_ink = np.nonzero(band.any(axis=1))[0]
    ink_span = (f"墨y={(ys_ink.min() - (y1 - y0) / 2) / H:+.2f}~"
                f"{(ys_ink.max() - (y1 - y0) / 2) / H:+.2f}H"
                if len(ys_ink) else "墨y=无")

    runs = []
    start = None
    for i, v in enumerate(ink):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(ink)))

    # 合并极小断口（抗锯齿）
    merged = []
    for a, b in runs:
        if merged and a - merged[-1][1] <= 0.15 * H:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))

    total = 0
    ink_width = 0.0
    dash_xs = []          # 每条延时线在原图中的中心 x（渲染时直接用，不再推算）
    dash_ys = []          # 每条被计数的横线相对数字中心的纵向偏移（单位 H，仅诊断）
    dash_spans = []       # 每条的「逐列墨心极差」(lo, hi)，单位 H（仅诊断）
    band_mid = (y1 - y0) / 2.0
    for a, b in merged:
        width = b - a
        if width < EXT_DASH_MIN_W * H:
            # 比半字高还短 → 不是 `-`。真 `-` 约 0.85H，这里的碎片通常只有
            # 0.3H 左右（附点 / 数字笔画残迹）。旧门槛是 0.25H，太松。
            continue
        # 「薄带覆盖」：段内最"实"的 0.35H 薄带里，有墨的列占段宽的比例。
        # 为什么需要它（用户 2026-09-21 报"莫名多出不少 -"）：`band` 是**横向膨胀过**的，
        # 低分辨率扫描件（12.jpg 数字高仅 11px）的 JPEG 噪点会被膨胀连成一段，
        # 逐列墨心极差也照样过 → 整段被当成一条 `-`。但真 `-` 是**横线**、噪点是**散点**：
        # 只看段内最实的那条 0.35H 薄带，真 `-` 的列覆盖高、噪点低。
        # 实测（12.jpg 16 段假 / 4 段真；1.jpg + 5.png + 2.png 253 段全真）：
        #   假 0.26~0.52    真 0.61~0.73（12.jpg）与 0.86+（另三张）→ 取 0.55，余量 17%。
        band_h = max(2, int(round(0.35 * H)))
        cov = 0.0
        s_best = 0
        for s in range(0, max(1, band_ink.shape[0] - band_h + 1)):
            cols = np.nonzero(band_ink[s:s + band_h, a:b].any(axis=0))[0]
            if len(cols) and len(cols) / max(width, 1) > cov:
                cov = len(cols) / max(width, 1)
                s_best = s
        if cov < EXT_MIN_BAND_COV:
            continue                            # 散点，不是一条 `-`
        # **逐列**墨心（相对数字中线的偏移）的极差。宽度和 `条y` 都分不开真假：
        #   宽度：假的是 0.6~0.9H，真值 0.85H，完全重叠；
        #   条y ：阶梯残迹的平均值恰好落在 0 附近，和真 `-` 撞在一起。
        # 只有逐列看才行，但余量薄（见 EXT_MAX_COL_SPAN 的说明）。
        sub = band[:, a:b]
        col_centers = []
        for ci in range(sub.shape[1]):
            nz = np.nonzero(sub[:, ci])[0]
            if len(nz):
                col_centers.append(float(nz.mean()))
        span_lo = (min(col_centers) - band_mid) / H if col_centers else 0.0
        span_hi = (max(col_centers) - band_mid) / H if col_centers else 0.0
        if span_hi - span_lo > EXT_MAX_COL_SPAN:
            continue                            # 阶梯状残迹，不是一条 `-`
        ink_width += width
        # 这一段墨的**纵向中心**（相对数字中线，单位 H）。
        # 这是区分「真延时线」和「减时线」的关键诊断量：
        #   真 `-` 画在数字中线高度   → 条y ≈ 0
        #   减时线画在数字下方        → 条y 明显偏负（-0.3~-0.6）
        rows_any = np.nonzero(band[:, a:b].any(axis=1))[0]
        y_rel = (((rows_any.min() + rows_any.max()) / 2.0 - band_mid) / H
                 if len(rows_any) else 0.0)
        # 条数按**薄带里一条条独立墨条**数，而不是拿膨胀后的段宽折算：
        #   * 段宽把"条与条之间的空档"和"混进来的噪点"都算进去 —— 12.jpg 有 1.28~1.57H
        #     的段其实只有一条真 `-`（膨胀后被数成 2 条，渲染多一条 `-`）；
        #   * 反过来，按"墨迹总跨度"折算又会少数：7.png 的 `2 - - -` 是三条独立短横，
        #     总墨迹只有 2.1H，折算成 2 条（原谱是 3 条）。
        # 逐条数则两头都对：每条独立的横条算一条，长条（多条纹糊成一条）再按 0.95H 折算；
        # 噪声碎条因为连 0.5H 都不到，在下面那道宽度门槛就被丢掉了。
        band_rows = band_ink[s_best:s_best + band_h, a:b]
        col_ink = band_rows.any(axis=0)
        runs_band = []
        st = None
        for ci, v in enumerate(col_ink):
            if v and st is None:
                st = ci
            elif not v and st is not None:
                runs_band.append((st, ci))
                st = None
        if st is not None:
            runs_band.append((st, len(col_ink)))
        merged_band = []
        for ra, rb in runs_band:
            if merged_band and ra - merged_band[-1][1] <= 0.15 * H:
                merged_band[-1] = (merged_band[-1][0], rb)
            else:
                merged_band.append((ra, rb))
        for ra, rb in merged_band:
            run_w = rb - ra
            if run_w < EXT_DASH_MIN_W * H:
                continue                        # 太短：附点碎片 / 噪点
            k = max(1, int(round(run_w / (0.95 * H))))
            total += k
            for j in range(k):
                dash_xs.append(x0 + a + ra + (j + 0.5) * run_w / k)
                dash_ys.append(y_rel)
                dash_spans.append((span_lo, span_hi))
        if dbg:
            # 把**计数成功**那一段的**原始墨迹**打成 ASCII（只打过关的段，行数不多）。
            # 覆盖范围就是扫描带 ±0.60H：图中**正中那一行 ≈ 数字中线**。
            # "附点被当成 `-`"这类问题，看形状比看汇总数字快得多。
            print(f"     -> {tag} 计数段 x={x0 + a}..{x0 + b} 宽{(b - a) / H:.2f}H "
                  f"（正中行为数字中线）：")
            for line in _ascii_mask(band_ink[:, a:b]):
                print(f"       |{line}|")
    # 打印条件：数出了 ≥1 条，**或**窗口里既有横线墨、又有竖向墨。
    # 后者正是「少一条」的现场——光看"还剩几条"分不清是"本来就只有一条"还是
    # "被竖向墨吃掉了"，把竖墨列数一起打出来才判断得了。
    if dbg and (total >= 1 or (tall_cols > 0 and merged)):
        next_txt = "无" if next_left == float("inf") else f"{next_left:.0f}"
        beams_txt = "?" if beams is None else str(int(beams))
        # 条y = 每条被计数的横线相对数字中线的纵向偏移。真 `-` ≈ 0；
        # 若条y 明显偏负（-0.3 以下）而该音的减时线又 >0，那就是**减时线
        # 被当成了延时线**（南泥湾多出大量 `-` 的根因）。
        dash_txt = ",".join(f"{v:+.2f}" for v in dash_ys) if dash_ys else "-"
        # 列心 = 每条被计数横线的「逐列墨心极差」。真 `-` 应该很小（每列同一高度）；
        # 明显偏大（如 -0.35~+0.52）说明那团墨是**阶梯状**的残迹（数字笔画/弧线下垂），
        # 不是一条 `-` —— 这是"多出来的 `-`"最直接的判据。
        span_txt = (",".join(f"{lo:+.2f}~{hi:+.2f}" for lo, hi in dash_spans)
                    if dash_spans else "-")
        print(f"  [ext] {tag} 减时线={beams_txt} "
              f"窗口={x0}..{x1}(宽{(x1 - x0) / H:.1f}H) 后数字={next_txt} "
              f"竖列={tall_cols} 墨宽={ink_width / H:.1f}H 段数={len(merged)} "
              f"{ink_span} 条y={dash_txt} 列心={span_txt} → {total} 条")
    return min(6, total), dash_xs


def _count_beams(block, hlines, H):
    """数减时线层数：用「横线本体」判定。

    横线必须 (1) 位于数字下方一定距离内；(2) 横向覆盖该数字；(3) 按高度去重。
    低音点是圆点、不是细长横线，天然不会被误数，因此不需要为它开特例
    （曾为低音点把扫描带下移，反而把紧贴数字的减时线整条跳过）。
    """
    y_lo = block["y"] + block["h"] + 0.08 * H
    y_hi = block["y"] + block["h"] + 1.00 * H
    cx_lo = block["x"] + 0.15 * block["w"] - 0.10 * H
    cx_hi = block["x"] + 0.85 * block["w"] + 0.10 * H

    ys = []
    for line in hlines:
        ly = line["y"] + line["h"] / 2
        if not (y_lo <= ly <= y_hi):
            continue
        if line["x"] + line["w"] < cx_lo or line["x"] > cx_hi:
            continue
        if all(abs(ly - old) > 0.10 * H for old in ys):
            ys.append(ly)
    return min(3, len(ys))


def _split_merged_dot(block, binary, H):
    """把「数字 + 上方粘连的高音点」拆开。

    简谱里数字 1 的笔画细，上方的点常与数字连成一个连通块。这里在上部
    找「颈部」（墨最少的行）：颈部以上是点、以下是数字本身。
    返回 (数字块, 是否含高音点)。
    """
    y0, h, x0, w = block["y"], block["h"], block["x"], block["w"]
    if h < 1.45 * H or w > 0.75 * H:
        return block, False
    roi = binary[y0:y0 + h, x0:x0 + w] > 0
    if roi.size == 0:
        return block, False
    rows = roi.sum(axis=1)
    lo, hi = int(0.10 * h), int(0.50 * h)
    if hi <= lo:
        return block, False
    seg = rows[lo:hi]
    neck_rel = int(np.argmin(seg)) + lo
    neck_rows = float(seg.min())
    body_rows = float(np.median(rows[hi:])) if hi < h else float(rows.mean())
    if body_rows <= 0 or neck_rows > 0.60 * body_rows:
        return block, False
    if neck_rel < 3:
        return block, False
    digit = dict(block)
    digit["y"] = y0 + neck_rel
    digit["h"] = h - neck_rel
    digit["area"] = int(roi[neck_rel:].sum())
    digit["merged_dot"] = True
    return digit, True


def _divide_by_beats(notes, beats_per_measure=4.0, first_boundary_x=None):
    """按拍数划分小节；用「第一个检测到的小节线」标记弱起结束。

    比依赖每个小节竖线稳：竖线漏检/误检只影响弱起那一次，之后每满
    beats_per_measure 拍就切一小节，拍数错位会立刻体现为小节异常。
    """
    measures = []
    current = []
    acc = 0.0
    started = first_boundary_x is None  # 无竖线时直接从行首开始计量
    for note in notes:
        if not started and note["cx"] >= first_boundary_x:
            started = True
            if current:
                measures.append(current)
                current = []
                acc = 0.0
        current.append(note)
        acc += float(note.get("q", 1.0) or 1.0)
        if started and acc >= beats_per_measure - 1e-6:
            measures.append(current)
            current = []
            acc = 0.0
    if current:
        measures.append(current)
    return measures


def _zero_like(block, binary, tol=0.30):
    """校验一个数字块是否「确实像休止符 0」。

    误判出假 0 会把后面所有音符顶偏一格，所以 0 必须通过形状校验：
    圆/椭圆、上下左右基本对称、填充率合理。小节线墨迹、横线碎片都过不了。
    tol 越小越严（对称性要求越高）：0 对称、6 下重上轻，tol 小就能把 6 排除。
    """
    x, y, w, h = block["x"], block["y"], block["w"], block["h"]
    roi = binary[y:y + h, x:x + w] > 0
    if roi.size == 0 or h < 6 or w < 5:
        return False
    ar = w / max(h, 1)
    if not (0.45 <= ar <= 1.25):
        return False
    fill = float(roi.mean())
    if fill < 0.40 or fill > 0.95:
        return False
    mh, mw = max(1, h // 2), max(1, w // 2)
    if abs(float(roi[:mh].mean()) - float(roi[mh:].mean())) > tol:
        return False
    if abs(float(roi[:, :mw].mean()) - float(roi[:, mw:].mean())) > tol:
        return False
    return True


def _detect_vlines(row_components, blocks, H, digits_top, digits_bottom, relaxed):
    """检测该行的小节竖线。relaxed=True 时放宽（用于全休止符行等特殊行）。"""
    found = []
    for c in row_components:
        fill = c["area"] / max(c["w"] * c["h"], 1)
        if relaxed:
            ok = (c["w"] <= 0.45 * H and c["h"] >= 0.85 * H and fill >= 0.55)
        else:
            ok = True
            if c["w"] > 0.30 * H or c["h"] < 1.10 * H:
                ok = False
            elif fill < 0.65:
                ok = False
            elif c["y"] > digits_top + 0.45 * H:
                ok = False
            elif c["y"] + c["h"] < digits_bottom - 0.45 * H:
                ok = False
        if ok and any(b["x"] - 2 <= c["x"] + c["w"] and c["x"] <= b["x"] + b["w"] + 2
                      for b in blocks):
            ok = False
        if ok:
            found.append(c)
    return found


def _estimate_digit_height(components):
    """从连通块高度分布估计「数字字高」：取高度直方图**平滑后**的峰。

    为什么要平滑（而不是像原来那样直接 argmax 取单根最高的柱子）：
    音符一贴减时线 / 音区点，高度就会浮动，于是散在**相邻好几档**里；而低音点是
    清一色同高的小圆点，全挤在**单根**柱子上。直接 argmax 会被低音点抢走 ——
    实测 28.jpg：低音点那档 317 块 > 音符任单档 230 块 → 估成 12px，而真音符高 ≈49px；
    字高一错，行接受（判据是「块高 ≈ 估计字高」）与模板归一化全崩。
    平滑（合并相邻档）后，音符那一簇（782 块）压过低音点那一簇（413 块）→ 估回 ≈49。
    实测 29 张里只有 28.jpg 会变（12→49），其余一字不动。
    """
    hs = [c["h"] for c in components
          if 0.28 <= c["w"] / max(c["h"], 1) <= 1.20 and c["h"] >= 10]
    if not hs:
        return None
    hist, edges = np.histogram(hs, bins=24)
    kernel = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
    kernel /= kernel.sum()
    peak = int(np.argmax(np.convolve(hist, kernel, mode="same")))
    band = [h for h in hs if edges[peak] <= h <= edges[peak + 1]]
    return float(np.median(band)) if band else float(np.median(hs))


# 「表头行」判据（用户 2026-09-21 报"多出一整行假音符"，13.png/8.jpg/7.png/12.jpg 都有）。
# 那些"假行"其实是**表头自己的数字**聚成的：`1=F 4/4`、`♩=103` 里的数字本来就与乐谱数字
# 同字号、形状也是数字，所以"块高/字形分/数字个数"这些判据全放行。实测 12 张图 140 行：
#   * 真行（134 个）：横向跨度 ≥ 0.71 图宽，块间距 max/中位 ≤ 4.4；
#   * 假行（6 个）：要么是一小簇（跨度 0.03/0.07/0.08/0.38），要么块间距极不规整
#     （10.9 / 23.5 / 36.6 —— 表头"左边一簇 + 右边几个碎块"的典型形态）。
# 两条判据都是**结构量**（不依赖字体/字号/扫描质量），阈值取在两侧空档中间：
#   跨度 0.71 vs 0.38 → 取 0.55（两侧各留 ~1.4 倍）；间距比 4.4 vs 10.9 → 取 7.0（各留 1.6 倍）。
ROW_MIN_SPAN = 0.55          # 一整行乐谱必然横跨整行的**大部分**；表头小簇只占一角
ROW_MAX_GAP_RATIO = 7.0      # 音符排布是均匀的；表头是"一簇 + 远处碎块"
# 「行带里有远高于字高的块」判据（用户 2026-09-23 报 27.jpg 多出 6 行歌词）。
# 歌词汉字被二值化打碎后，**碎片**的高度/形状都混进了"数字块"，把行高中位数压进合格线
#（现有的"块高≈字高"`h_ok` 和"字形像数字"`shape_ok` 两道判据因此都拦不住它）。
# 但**汉字本身仍是完整的一大块**。实测"行带内最高的连通块 / 字高"：
#   真旋律行（含括号 `(`、连音弧、倚音组）：**≤ 1.766×**（全 40 张、435 条被接受行的上限）
#   垃圾行：39.png 行4 = **1.896×**、39.png 另一条 = **2.119×**；27.jpg 的歌词行 2.00~2.23×
# 取 1.90：真行 1.766 ↔ 阈值 ↔ 39 的 1.896，两侧各留约 3.6%。
# ⚠ **余量只有 3.6%，是薄阈值**：真行之所以能高到 1.77，是因为**括号**在谱面上就是
#   约 1.7 倍字高；若某张谱的括号画得更高（≥1.85）就会被误杀（而且是**静默**丢掉一行）。
# **2026-09-24 补**：曾把它压到 1.83 去治 39.png，后来发现下面那条 `ROW_HANZI_MIN`
#   判据强得多（0 个 vs 15 个汉字块），于是**恢复 1.90**、由那条接手。
ROW_OVERSIZE_MAX = 1.90
# 「行带里有一片汉字块」判据（2026-09-24 加，治 39.png《侧脸》行4）。
# 背景：39 那种谱**歌词字与音符一样大**，上面那串判据（text_count / shape_ok / h_ok /
# oversize）全都失效——它们都建立在"歌词字比音符大/更清楚"之上。
# 但**汉字本身仍是完整的一大块**，而且**尺寸就露在连通域里**（不需要 OCR 认字）：
#   · 高 ≥1.35×字高 —— 正好是「音符候选」的高度上限，超过它的块工具**根本看不见**；
#   · 宽高比 0.7~1.6（方正）—— 把**细窄的括号**排除在外（括号高 1.7×但宽高比只 0.2~0.5）。
# 实测（全 40 张、433 条被接受的行）：**431 行 0 个、2 行 1 个、0 行 ≥2 个**
#   ——那 2 个"1 个"的也只是个横跨图宽 **1~4%** 的碎片。而 39 的那条歌词行有 **15 个**。
# 故取 `>= 2`：**零误伤**（被接受行里一个都没有），却能拦住歌词行。
ROW_HANZI_MIN = 2
# ⚠ 这两条对**末行**要放宽：谱子没结束（还没见到 `‖`）时，后面的行必须还是旋律行，
#    哪怕它又矮又短。放宽逻辑在 `_rows_from_blocks()` 的"末行放宽"那段。
# 假休止复核：判成 0 的音，若"形状最像的那个数字"不是 0、且分数 ≥ 此值 → 改判成那个数字。
# 实测 4 个假休止（6 被误判）的形状分是 0.80~0.88，而 56 个真休止"最像的都是 0"
# （0.76~0.94）→ 取 0.75，两侧都留有余量。
FAKE_REST_MIN_SCORE = 0.75
# 「底部粘着减时线」的补救：16 分音符密集行里，减时线常与数字底部连成一个连通块，
# 块被撑高 → 归一化后形状里多一条厚底边 → 跟任何参照字形都对不上（判成 `?`）。
# 实测 14.jpg 行2 那个 `2̇`：块高 51（同图正常数字 43）、与参照 2 的相关仅 0.48；
# 剪掉底部一截再认就对上了。**只在原块判不出时**才试剪，免得把正常数字削成别的。
SUB_MERGE_TRIM_FRACS = (0.15, 0.25)
# 括号候选：若它"很像某个数字"（分类器给出数字且分 ≥ 此值）就不是括号。
# 依据：细高的 `1` 与 `(` 在尺寸/填充上几乎一样（17.jpg 实测把整张谱的 `1` 都判成了括号，
# 每行留 3~16 个假括号），但 `1` 会很像 `1`（≥0.8），真括号则谁都不像。
PAREN_MAX_DIGIT_SCORE = 0.85
# 删块用的**低**门槛：与括号候选重叠的块，只要分类器能给数字且 ≥ 此值就**留着当音符**。
# 删除是危险方向（丢音符不可逆），所以门槛比上面那个低很多：真 `(` 给不出数字（0.48）照样删，
# 而 17.jpg 那些像 `(` 的 `1`（0.75~0.92）不会因为"自己像括号"被删。
PAREN_DELETE_MIN_DIGIT_SCORE = 0.60
# 判"拉丁文本行"前先把**已知的 OCR 字母↔数字混淆**还原掉：工具的注释里写着它的常见错法是
# `6→b/G、5→S`，另外 `l/I→1、O→0、Z→2` 也常见。不还原的话，一行真旋律只要有几个数字被读成
# 字母（实测 8.jpg 有行是 `67/3`），就会被误判成"含拉丁文本的制谱信息行"而整行丢掉。
# 特别地**不把 `/` 当判据**：8.jpg 的 `1` 起笔很重，OCR 就会读成 `/`。
OCR_LETTER_TO_DIGIT = str.maketrans({
    "b": "6", "B": "6", "G": "6", "g": "6", "S": "5", "s": "5",
    "O": "0", "o": "0", "l": "1", "I": "1", "Z": "2", "z": "2",
})


def _row_has_oversize(components, cy, H0):
    """这一行的带子里有没有「远高于数字字高」的块（≈歌词汉字）。判据与实测见 ROW_OVERSIZE_MAX。"""
    tol = 0.80 * H0
    for c in components:
        if abs(c["y"] + c["h"] / 2 - cy) > tol:
            continue
        if c["area"] < 60 or not (0.15 <= c["w"] / max(c["h"], 1) <= 6.0):
            continue
        if c["h"] > ROW_OVERSIZE_MAX * H0:
            return True
    return False


def _row_hanzi_count(components, cy, H0):
    """这一行的带子里有几个「汉字块」。判据与实测见 `ROW_HANZI_MIN`。

    「汉字块」= 高 ≥1.35×字高（超出音符候选的高度上限）**且** 宽高比 0.7~1.6（方正，
    把细窄的括号排除在外）。**只看几何、不依赖 OCR 认字**——这正是它比其它判据稳的地方。
    """
    tol = 0.80 * H0
    count = 0
    for c in components:
        if abs(c["y"] + c["h"] / 2 - cy) > tol:
            continue
        if c["area"] < 60 or c["h"] < 1.35 * H0:
            continue
        ar = c["w"] / max(c["h"], 1)
        if 0.7 <= ar <= 1.6:
            count += 1
    return count


def _rows_from_blocks(components, tokens, H0, binary, text_tokens=None):
    """用「数字块按 y 聚类」定行（纯图像，不依赖 OCR），再把 OCR 文本按 y 分配给各行。

    之前用 OCR 文本聚类定行：行距密时同一行会被拆散、相邻两行会被并成一行
    （表现为行数少、整行内容错乱）。改用图像定行后行数与原谱一致得多。
    """
    cands = []
    for c in components:
        ar = c["w"] / max(c["h"], 1)
        # 高度窗口必须贴合「数字」：放宽到 1.6 倍会把歌词汉字（约 1.4-1.8 倍）
        # 也收进行里，凭空多出一堆假行（曾出现 14 行谱判成 22 行）。
        if not (0.65 * H0 <= c["h"] <= 1.35 * H0):
            continue
        if not (0.10 <= ar <= 1.25):
            continue
        if c["area"] < 0.06 * H0 * H0:
            continue
        cands.append(c)
    cands.sort(key=lambda c: c["y"])

    tol = 0.80 * H0
    rows = []
    for c in cands:
        cy = c["y"] + c["h"] / 2
        for row in rows:
            if abs(cy - row["cy"]) <= tol:
                row["blocks"].append(c)
                row["cy"] = float(np.median([b["y"] + b["h"] / 2 for b in row["blocks"]]))
                break
        else:
            rows.append({"cy": cy, "blocks": [c]})
    rows.sort(key=lambda r: r["cy"])


    if not cands:
        print("  [imgrow] 警告：没有任何候选数字块")
    else:
        if VERBOSE_DEBUG:
            print(f"  [imgrow] 候选块 {len(cands)} 个；OCR 数字文本块共 {len(tokens)} 个；"
                  f"字高估计 {H0:.0f}")
            if tokens:
                sample = ", ".join(f"{t['cy']:.0f}" for t in tokens[:8])
                print(f"  [imgrow] 文本块 cy 示例：{sample}")

    out = []
    for r in rows:
        hs = [b["h"] for b in r["blocks"]]
        H = float(np.median(hs)) if hs else 0.0
        y0 = min((b["y"] for b in r["blocks"]), default=0)
        y1 = max((b["y"] + b["h"] for b in r["blocks"]), default=0)
        r["H"] = H
        r["tokens"] = [t for t in tokens
                       if y0 - 0.6 * H <= t["cy"] <= y1 + 0.6 * H
                       and (t["y1"] - t["y0"]) <= 1.5 * H] if H > 0 else []
        digit_count = sum(1 for t in r["tokens"] for v in t["slots"] if v is not None)
        # 「纯文字行」判据（歌词行 / 制谱信息行）：旋律行的本质是「含数字」，
        # 若这一行里出现 >=2 个「纯汉字、完全不含数字」的文本块，它就是文字行。
        text_count = sum(1 for t in (text_tokens or [])
                         if y0 - 0.8 * H <= t["cy"] <= y1 + 0.8 * H)
        # 「含网址/拉丁文本的行」判据（用户 2026-09-21 报 17.jpg 末尾那行被当成音符行）：
        # 旋律行的 OCR 文本永远是数字 / 横线 / 括号 / 汉字歌词，**不会出现拉丁字母或 `/`**。
        # 实测那行的文本块是 `1/user.qzone.qq.com/601030387`，形状分仅 0.52（该图真行 0.78~0.86），
        # 而块数 46、跨度 0.70 —— 几何判据全放行，只有文本内容能一刀切开。
        # 要求"≥2 个 ASCII 字母"或含 `/`，免得把 OCR 偶尔读出的单个 `l`/`O` 当成文本行。
        latin_like = any(
            sum(1 for ch in (t.get("text") or "").translate(OCR_LETTER_TO_DIGIT)
                if ch.isascii() and ch.isalpha()) >= 2
            for t in r["tokens"])
        # 判定「是不是旋律行」：
        #   主判据——块高必须等于数字字高（±15%）。歌词汉字明显更大，用纯几何
        #   就能排除歌词行/标题行；
        #   辅助——若 OCR 在这行确实读到 >=3 个数字，也接受（容忍字号差异）。
        h_ok = 0.85 * H0 <= H <= 1.15 * H0
        # 倚音（装饰音）行：倚音按惯例**印得更小**，若整行块高不足全局数字字高的
        # 0.8 倍，判为倚音行剔除（否则会凭空多出一行，实测《枉凝眉》第 1 行上方
        # 的小字号 `321 5` 就是这样被当成了一整行）。
        grace_ok = (H >= 0.80 * H0) or not ENABLE_GRACE_ROW_FILTER
        # 再按「字形像不像数字」复核：数字块与 0-7 参照的匹配分高，
        # 汉字（如"两只老虎"）匹配分明显低。这一步专门剔除"歌词行被当成旋律行"。
        refs_chk = _all_reference_glyphs()
        shape_scores = []
        for b in sorted(r["blocks"], key=lambda b: b["x"])[:24]:
            _d, sc = _classify_by_reference(b, binary, refs_chk, return_score=True)
            if sc > -1:
                shape_scores.append(sc)
        med_shape = float(np.median(shape_scores)) if shape_scores else -1.0
        shape_ok = med_shape >= 0.50
        runs_list = [_stroke_runs(b, binary) for b in r["blocks"][:24]]
        med_runs = float(np.median(runs_list)) if runs_list else -1.0
        reason = ""
        if not grace_ok:
            reason = f"倚音行（字高{H:.0f} 明显小于数字字高 {H0:.0f}）"
        elif latin_like:
            reason = "含网址/拉丁文本（制谱信息行，不是旋律行）"
        elif len(r["blocks"]) < 3:
            reason = "块太少"
        elif text_count >= 2:
            # 注：text_tokens 现在已做**跨尺度去重**（见 ocr_jianpu 里 text_tokens
            # 的去重段）。去重前这个计数是**翻倍的**（同一块报两次），也就是
            # 这个 `>= 2` 原先是在"虚高"的计数上生效的。语义上现在才真的等于
            # 注释说的"这一行出现 >=2 个汉字块"；因为主判据是块高
            #（歌词汉字明显更大），实测影响可以忽略，但改动这一行附近时请留意。
            reason = f"纯文字行（{text_count} 个汉字块，歌词/制谱行）"
        elif H <= 0 or np.std(hs) / H > 0.45:
            reason = "高度不齐"
        elif not h_ok and digit_count < MIN_ROW_DIGITS:
            reason = f"字高{H:.0f}≠数字字高{H0:.0f}"
        elif not shape_ok:
            reason = f"字形不像数字(匹配分{med_shape:.2f})"
        elif _row_has_oversize(components, r["cy"], H0):
            reason = "含远高于字高的块（歌词汉字）"
        elif _row_hanzi_count(components, r["cy"], H0) >= ROW_HANZI_MIN:
            reason = "行带里有一片汉字块（歌词行）"
        # 表头行两条结构判据（见 ROW_MIN_SPAN / ROW_MAX_GAP_RATIO 处的实测说明）：
        # 放在最后 —— 只有"前五条都通过"的行才需要它俩兜底，正是那 6 个假行。
        xs_sorted = sorted(b["x"] for b in r["blocks"])
        if not reason:
            span = ((xs_sorted[-1] - xs_sorted[0]) / binary.shape[1]) if xs_sorted else 0.0
            if span < ROW_MIN_SPAN:
                reason = f"表头行（只横跨 {span:.0%} 图宽）"
        if not reason and len(xs_sorted) >= 3:
            gaps = [b - a for a, b in zip(xs_sorted, xs_sorted[1:])]
            med_gap = float(np.median(gaps))
            if med_gap > 0 and max(gaps) / med_gap > ROW_MAX_GAP_RATIO:
                reason = (f"表头行（块间距极不规整 max/中位="
                          f"{max(gaps) / med_gap:.1f}）")
        # 「末行放宽」（用户 2026-09-22 报"19.png 最后一行的音符被丢弃"）：
        # 谱子的末行常常又矮又短（19.png 稻香末行 `i - - - | i - 0 0 ) ‖` 只有 4 个块、
        # 只横跨 16% 图宽），会被上面两条"表头行"结构判据（跨度/块间距）当成表头丢掉。
        # 判据：**这一行以终止记号 `‖` 收尾**（`_row_has_end_bar()`）就免掉那两条结构判据
        # —— 见到 `‖` 就是曲终，末行必须收下；其余五条判据照旧生效（字形像数字、块高、
        # 块数≥3、非拉丁文本行、非纯文字行），所以末尾的制谱信息行/水印仍被挡住。
        # ⚠ 为什么要求"本行带 ‖"，而不是"只要还没见过 ‖ 就一路放宽"（用户原话的字面版）：
        #   21.jpg 实测就会翻车 —— 它第 3 行有个 4 块、只横跨 8% 图宽、形状分仅 0.59 的
        #   小簇（夹在两行真谱之间），按"还没见过 ‖"会被当旋律行收下、整页行号后移一位。
        #   要求本行带 ‖ 之后：要救的两行（19.png y≈3932、17.jpg y≈3913，行尾都有 ‖）
        #   照旧救回，而 21.jpg 那个小簇（行尾无 ‖）被挡在外面。
        #   代价：若末尾**连续两行**都很短、只有最后一行带 ‖，前一行仍会被丢
        #   （目前没有这种实测样本；真出现了就改成"尾部整段放宽"再做 A/B）。
        tail_relaxed = False
        if (reason and out and reason.startswith("表头行")
                and _row_has_end_bar(r, components, H)):
            tail_relaxed = True
            reason = ""
        if VERBOSE_DEBUG:
            verdict = ("→ 保留" + ("（末行放宽：本行以 ‖ 收尾）" if tail_relaxed else "")
                       if not reason else "→ 跳过：" + reason)
            print(f"  [imgrow] y≈{r['cy']:.0f} 块={len(r['blocks'])} H={H:.0f} "
                  f"文本块={len(r['tokens'])} 数字={digit_count} "
                  f"形状分={med_shape:.2f} 墨段={med_runs:.1f} "
                  f"最高比字高={max(hs) / H0:.2f} "
                  f"跨度={(xs_sorted[-1] - xs_sorted[0]) / binary.shape[1] if xs_sorted else 0:.2f} "
                  f"{verdict}")
        if reason:
            continue
        out.append(r)
    return out


def _dedupe_boundaries(vlines, tolerance):
    xs = sorted(v["x"] + v["w"] / 2 for v in vlines)
    groups = []
    for x in xs:
        if groups and x - groups[-1][-1] <= tolerance:
            groups[-1].append(x)
        else:
            groups.append([x])
    return [float(np.mean(g)) for g in groups]


def _make_debug_overlay(color, row_records, out_path):
    """生成原图叠加诊断图：绿框=数字块，蓝字=识别结果/音区/时值，青线=小节线。"""
    img = color.copy()
    for row_index, record in enumerate(row_records):
        y0, y1 = int(record["py0"]), int(record["py1"])
        cv2.rectangle(img, (0, y0), (img.shape[1] - 1, y1), (0, 0, 255), 2)
        notes = record.get("notes", [])
        review_line = record.get("review_line")
        for index, (block, value) in enumerate(zip(record["blocks"], record["values"])):
            x, y, w, h = block["x"], block["y"], block["w"], block["h"]
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 180, 0), 2)
            if index < len(notes):
                note = notes[index]
                mark = "?" if value is None else str(value)
                octave = int(note.get("o", 0))
                if octave > 0:
                    mark += "^" * octave
                elif octave < 0:
                    mark += "v" * (-octave)
                markup = note.get("markup", {})
                mark += "_" * int(markup.get("beams", 0))
                if markup.get("dotted"):
                    mark += "."
                mark += "-" * int(markup.get("extend", 0))
            else:
                mark = "?" if value is None else str(value)
            prefix = (f"L{review_line:02d}N{index + 1:03d}="
                      if review_line is not None else "SKIP=")
            cv2.putText(img, prefix + mark, (x, max(12, y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 0, 0), 1, cv2.LINE_AA)
        for boundary in record.get("boundaries", []):
            bx = int(round(boundary))
            cv2.line(img, (bx, y0), (bx, y1), (255, 255, 0), 2)
        # 被拒绝的竖线候选画成橙色，便于看「漏检的那条卡在哪」
        for vx, accepted in record.get("vline_candidates", []):
            if accepted:
                continue
            cv2.line(img, (int(round(vx)), y0), (int(round(vx)), y1), (0, 140, 255), 1)
        cv2.putText(img, f"row {row_index}", (8, y0 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.imwrite(out_path, img)


def ocr_to_stream(image_path, debug_dir=None, measure_mode="vline", header_ocr=True):
    """图片 → measures。

    measure_mode:
      "vline" —— 按检测到的小节竖线划分（默认，符合「看到 | 就切」的直觉）；
      "beats" —— 4/4 每满 4 拍切一小节（用第一条竖线标记弱起结束）。

    header_ocr:
      是否读页眉（调号 + 曲名）。**默认 True**（regress.py 等直接调用方行为不变）。
      调用方若已从 `piano_config.json` / `--key` / `--title` 拿到这两项，可传 False ——
      页眉那 5 次检测（`_KEY_CROPS` 四档 + `_TITLE_CROP`）**结果反正用不上**：
      读出来的值只写进 `_LAST_KEY_MARK` / `_LAST_TITLE` 两个模块全局，而 cli.py 仅在
      `not args.key` / `not args.title` 时才去读它们，且不落进 review JSON。
      实测省 **4.4~9.8s/张**（同一张图交替测 3 轮；页眉裁剪宽达 2000px，那几次检测的耗时
      波动很大）。
      传 False 时这两个全局会被**置回 None**（不能留着上一张图的值）。
    """
    color0 = _imread_unicode(image_path, cv2.IMREAD_COLOR)
    if color0 is None:
        raise ValueError(f"无法读取图片：{image_path}")

    # 自适应缩放：先把整图归一到「数字高约 TARGET_DIGIT_PX」再处理。
    # 小字号会让 OCR 与字形匹配同时退化（曾有一张 600x287 的谱，数字仅 12px 高，
    # 5/6/0 的差别在图上已糊掉，怎么调都判错）。归一化后两者都明显变准。
    gray0 = cv2.cvtColor(color0, cv2.COLOR_BGR2GRAY)
    bin0 = cv2.adaptiveThreshold(gray0, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 21, 9)
    _n0, _l0, stats0, _c0 = cv2.connectedComponentsWithStats(bin0, connectivity=8)
    comps0 = [{"x": int(s[0]), "y": int(s[1]), "w": int(s[2]), "h": int(s[3]),
               "area": int(s[4])} for s in stats0[1:] if int(s[4]) >= 8]
    h1 = _estimate_digit_height(comps0) or 12.0
    scale = float(min(6.0, max(1.0, TARGET_DIGIT_PX / max(h1, 1.0))))
    print(f"[ocr] 原图数字高≈{h1:.0f}px → 缩放 {scale:.2f}×")

    color = cv2.resize(color0, None, fx=scale, fy=scale,
                       interpolation=cv2.INTER_CUBIC)
    # 供 `overlay_letters` 用：它要在**这张缩放后的彩色图**上原位盖字母，
    # 而每个音的外框（note["box"]）就是在这个坐标系里的。
    global _LAST_COLOR, _LAST_SCALE
    _LAST_COLOR, _LAST_SCALE = color, scale
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 21, 9,
    )
    image_h, _image_w = gray.shape
    debug = debug_dir is not None
    if debug:
        os.makedirs(debug_dir, exist_ok=True)

    engine = _make_ocr_engine()
    global _LAST_KEY_MARK, _LAST_TITLE
    if header_ocr:
        _LAST_KEY_MARK = _detect_key_mark(engine, color0)
        _LAST_TITLE = _detect_title(engine, color0)
    else:
        _LAST_KEY_MARK = _LAST_TITLE = None
    # 多尺度 OCR：同时在 2 倍图与 1 倍原图上各跑一遍。
    # 小图/细字时 OCR 的文字检测器会漏掉大半（曾出现 1200x574 的谱只读出 4 个
    # 文本块），两个尺度互补能显著提高召回。1 倍结果的坐标乘 SCALE 对齐到 2 倍。
    raw_items = []
    for arr, sc in ((color, 1.0), (color0, scale)):
        try:
            res = _ocr(engine, arr)
        except Exception:  # noqa: BLE001
            res = []
        for item in res or []:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                raw_items.append((item[0], str(item[1]), sc))
    print(f"[ocr] 多尺度 OCR：检出文本块 {len(raw_items)} 个")

    # 整图 OCR → 文本块。
    # 注意：**不能**把「含中文的文本块」整条丢弃。这份谱的旋律数字常与下方歌词
    # 被 OCR 并成一个块（如 `1 2 3 1两只老虎`），整条丢弃会让整行音乐消失
    # （曾表现为「两行谱只识出一行」）。改为按字符位置截出数字那一段来用。
    tokens = []
    text_tokens = []   # 纯文字块（含汉字且完全不含数字）：用于识别"文字行"
    for box, text, sc in raw_items:
        xs = [float(p[0]) * sc for p in box]
        ys = [float(p[1]) * sc for p in box]
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
        mixed = _contains_cjk(text)
        if not _digit_slots(text, allow_confusion=not mixed):
            # 完全不含数字：若是汉字，记为"纯文字块"（歌词行/制谱行的证据）
            # ⚠ **单个汉字也算**：原谱里"每行第一个歌词字"常被单独切成一块（它与后面几个
            # 字之间留了空档），以前要求 `len >= 2` 会把它整块丢掉，表现就是"行首的歌词字
            # 不见了"。判据直接用 `_contains_cjk`（只认汉字段，标点/拉丁字母仍当噪声）。
            if _contains_cjk(text):
                text_tokens.append({"x0": x0, "x1": x1, "y0": y0, "y1": y1,
                                    "cy": (y0 + y1) / 2, "text": text})
            continue
        if mixed:
            idxs = [i for i, ch in enumerate(text) if _is_digit_char(ch) or ch in "ilI"]
            if not idxs:
                continue
            lo, hi = min(idxs), max(idxs)
            n = max(len(text), 1)
            span = x1 - x0
            sub = text[lo:hi + 1]
            slots = _digit_slots(sub)  # 混排块：只用真数字，不做误读还原
            if not slots:
                continue
            # 混排块里数字在上（旋律行在歌词之上），据此估计数字所在的行位置与字高
            h_est = 0.6 * (y1 - y0)
            tokens.append({
                "x0": x0 + span * lo / n, "x1": x0 + span * (hi + 1) / n,
                "y0": y0, "y1": y0 + h_est,
                "cy": y0 + 0.3 * (y1 - y0),
                "h_est": h_est,
                "text": sub, "slots": slots, "mixed": True,
            })
            # 混排块里的**汉字部分也要留住**：原谱常用"1."做段号、后面紧跟第一个歌词字，
            # 于是 OCR 会把它和歌词并成一块（《上海滩》行2 的 token 实测就是 `1.浪`）。
            # 上面只截了数字段，汉字整段被丢 —— 表现就是"每行的第一个歌词字不见了"
            # （用户 2026-09-20 报的；同一行的第二个"浪"在别的 token 里，所以只丢一个）。
            # 按字符位置把数字左/右两侧交回歌词对齐；y 用整块的（歌词窗口本来就宽 3.5H）。
            for _part, _px0, _px1 in ((text[:lo], x0, x0 + span * lo / n),
                                      (text[hi + 1:], x0 + span * (hi + 1) / n, x1)):
                _part = _part.strip()
                # 两端把非汉字削掉：`1.浪` 的右半截是 `.浪`，段号那个句点不属于歌词。
                while _part and not _contains_cjk(_part[0]):
                    _part = _part[1:]
                while _part and not _contains_cjk(_part[-1]):
                    _part = _part[:-1]
                if _part:
                    text_tokens.append({"x0": _px0, "x1": _px1,
                                        "y0": y0, "y1": y1,
                                        "cy": (y0 + y1) / 2, "text": _part})
        else:
            tokens.append({
                "x0": x0, "x1": x1, "y0": y0, "y1": y1,
                "cy": (y0 + y1) / 2,
                "h_est": y1 - y0,
                "text": text,
                "slots": _digit_slots(text, allow_confusion=True),
                "mixed": False,
            })

    # 两个尺度会重复报同一段文字：按位置去重（x 重叠且 y 接近的只留一个，
    # 优先保留数字更多的那个）。
    tokens.sort(key=lambda t: (-len(t["slots"]), t["cy"], t["x0"]))
    deduped = []
    for t in tokens:
        dup = False
        for k in deduped:
            if abs(t["cy"] - k["cy"]) > 0.6 * max(t["h_est"], k["h_est"]):
                continue
            overlap = min(t["x1"], k["x1"]) - max(t["x0"], k["x0"])
            if overlap > 0.5 * min(t["x1"] - t["x0"], k["x1"] - k["x0"]):
                dup = True
                break
        if not dup:
            deduped.append(t)
    tokens = deduped

    # 同一件事**也要对纯汉字块（歌词）做一遍**——上面那次漏了它们，
    # text_tokens 是在上面那个循环里、去重之前就收集好的。
    # 后果：两个尺度各报一次同一行歌词，两份都留在 text_tokens 里，
    # 逐字分配时同一个字被追加两次（`best["lyric"] += ch` 执行了两遍），
    # 成品里就出现"不不""出出""爱爱"这种**成对重复**（实测《上海滩》整首）。
    # 注意这跟"三段歌词"无关：那是 1822 行 top_cy 窗口负责的事。
    #
    # 阈值比上面紧（0.4 字高，不是 0.6）：同一块的跨尺度拷贝 cy 几乎重合，
    # 而不同段歌词之间有约 1 个字高的行距。用紧阈值是为了**保证不会把相邻的
    # 另一段歌词误合并进来**——否则字数多的那段可能把真正的第一段挤掉。
    # 同块两份文本长度不同时留**字数多的**（2 倍图常读到更完整的字，
    # 如 "看不出有未" vs "看不出有未来"），而不是先到先得。
    text_tokens.sort(key=lambda t: (-len(t["text"]), t["cy"], t["x0"]))
    deduped_text = []
    for t in text_tokens:
        dup = False
        for k in deduped_text:
            if abs(t["cy"] - k["cy"]) > 0.4 * max(t["y1"] - t["y0"], k["y1"] - k["y0"]):
                continue
            overlap = min(t["x1"], k["x1"]) - max(t["x0"], k["x0"])
            if overlap > 0.5 * min(t["x1"] - t["x0"], k["x1"] - k["x0"]):
                dup = True
                break
        if not dup:
            deduped_text.append(t)
    text_tokens = deduped_text

    # 所有连通块：较大的用于数字，小的用于点/线。
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    all_components = []
    for i in range(1, count):
        x, y, w, h, area = (int(v) for v in stats[i])
        if area >= 8:
            all_components.append({"x": x, "y": y, "w": w, "h": h, "area": area})

    # 行定位改用图像（数字块按 y 聚类），不再依赖 OCR 文本聚类。
    H0 = _estimate_digit_height(all_components)
    if H0 is None:
        raise ValueError("未能估计数字字高，请确认图片为清晰的印刷简谱。")
    # 兜底：**同一张图，在缩放图上重估偶尔会离谱**（39.png 实测估成 12、真值 48）——
    # 音符的高度正好被直方图的档位边界劈成两半（216 块在 [42.5,47.9)、173 块在
    # [47.9,53.3)），"平滑后取峰"没能把两半并起来，低音点那一档就以 123:117 险胜。
    # 而**原图上估的是对的**（h1≈23 → 23×2.09≈48），所以差得太远就退回它。
    # 实测 39 张：38 张两者只差 ≤2px（阈值 14px，7 倍余量）→ 一次都不触发；
    # 只有 39.png 差 36px 会触发。
    # ⚠ 试过"取最大簇"的改法（谁的块多听谁的）：会改动 10 张图的估计值，已否决。
    if abs(H0 - h1 * scale) > 0.30 * (h1 * scale):
        H0 = h1 * scale
    # 行定位：图像（数字块按 y 聚类）+ 双判据。
    # 为什么不用 OCR 文本聚类：OCR 常把数字读成别的字符（小字号衬线体下
    # `1`→`|`、`5`→`S`），于是"含数字的文本块"极少（曾出现整图只有 4 个），
    # 拿它定行会漏掉大半行。图像定行不依赖 OCR；接受一行的条件是
    # 「块高 ≈ 数字字高」或「OCR 在这行确实读到 ≥3 个数字」——
    # 前者排除歌词行（汉字更大），后者容忍字号差异。
    anchored_rows = _rows_from_blocks(all_components, tokens, H0, binary, text_tokens)
    if not anchored_rows:
        raise ValueError("未找到有效旋律行；请确认图片方向正确且数字清晰。")
    print(f"[ocr] 图像定行：{len(anchored_rows)} 行（字高≈{H0:.0f}）")

    row_records = []
    for row in anchored_rows:
        # 行高与基线优先取本行数字块（图像定行的结果，比 OCR 文本框稳）；
        # 没有该信息时回退到 OCR 文本框。
        if "H" in row:
            H = float(row["H"])
            baseline_y = float(row["cy"])
        else:
            tokens_all = sorted(row["tokens"], key=lambda t: t["x0"])
            heights = [t.get("h_est", t["y1"] - t["y0"]) for t in tokens_all]
            H = float(np.median(heights)) if heights else 42.0
            baseline_y = float(np.median([t["cy"] for t in tokens_all]))
        H = max(20.0, min(90.0, H))

        # 弃用「跨行 token」：OCR 常把相邻两行并成一个文本块，其高度约为
        # 2 倍字高，用它赋值会把数字错配到别的行。
        tokens_in_row = sorted(row["tokens"], key=lambda t: t["x0"])
        tokens_in_row = [t for t in tokens_in_row
                         if t.get("h_est", t["y1"] - t["y0"]) <= 1.5 * H]

        # 「谱表左边界」：本行数字带内若有贯穿整个谱表的左括线，取最左那条当左边界，
        # 它左侧的块（声部名所在的页边）不成音。判据与实测见 LEFTMOST_BAR_* 的说明。
        left_bound = None
        _bars = [c for c in all_components
                 if c["h"] >= LEFTMOST_BAR_MIN_H * H
                 and c["w"] <= LEFTMOST_BAR_MAX_W * H
                 and c["x"] + c["w"] <= LEFTMOST_BAR_MAX_X * binary.shape[1]
                 and c["y"] < baseline_y + 0.55 * H
                 and c["y"] + c["h"] > baseline_y - 0.55 * H]
        if _bars:
            left_bound = min(c["x"] for c in _bars)

        # 数字主体与 OCR 中线接近；收紧长宽比，排除括号、竖线与汉字碎块。
        candidates = []
        for c in all_components:
            cy = c["y"] + c["h"] / 2
            ar = c["w"] / max(c["h"], 1)
            # 「数字 + 上方粘连高音点」：又高又窄 → 尝试拆分
            if c["h"] >= 1.45 * H and c["w"] <= 0.75 * H:
                digit, has_dot = _split_merged_dot(c, binary, H)
                if has_dot:
                    c = digit
                    cy = c["y"] + c["h"] / 2
                    ar = c["w"] / max(c["h"], 1)
                else:
                    continue
            if abs(cy - baseline_y) > 0.55 * H:
                continue
            # 谱表左边界之外（页边里的声部名等）不是音符，见 LEFTMOST_BAR_* 的说明。
            # 用**整块都在左侧**判定（`x+w <= left_bound`），避免切掉跨界的块。
            if left_bound is not None and c["x"] + c["w"] <= left_bound:
                continue
            if not (0.50 * H <= c["h"] <= 1.35 * H):
                continue
            # 宽高比放宽到 0.12：数字 1 是细长笔画（约 0.15-0.25），
            # 原来要求 >= 0.32 会把它整体滤掉，导致显示 ? 或缺失。
            # 但细长块加高度上限，避免把小节竖线（更高）算成数字。
            if ar < 0.32 and c["h"] > 1.25 * H:
                continue
            if not (0.12 <= ar <= 1.15):
                continue
            # 极细且填充率低的块，多半是弧线（括号等）；但**先看它直不直**：
            # 17.jpg 的 `1` 就是一条细竖笔，填充率同样只有 0.38~0.44，
            # 原先一律按弧线丢掉 → 整张谱的 `1` 全没了（用户 2026-09-22 报）。
            # 判据与实测见 `PAREN_STEM_MIN_RATIO`。
            fill = c["area"] / max(c["w"] * c["h"], 1)
            is_stem = (c["w"] <= 0.70 * H and fill < 0.55
                       and _max_stem_run(c, binary) >= PAREN_STEM_MIN_RATIO)
            if c["w"] < 0.35 * H and fill < 0.45 and not is_stem:
                continue
            # 括号形状（细高、填充率低、**且是弧**）也不是数字：它由 OCR 文本单独处理，
            # 若留在这里会变成一个没有值的块，渲染成多余的 `?`。
            if c["h"] >= 1.15 * H and c["w"] <= 0.70 * H and fill < 0.55 \
                    and not is_stem:
                continue
            if c["area"] < 0.08 * H * H:
                continue
            candidates.append(c)
        candidates.sort(key=lambda c: c["x"])
        if len(candidates) < 4:
            continue

        # 每个 token 覆盖的数字块 → 赋数字；槽位数与块数一致时视为高置信种子。
        values = [None] * len(candidates)
        seed_flags = [False] * len(candidates)
        for token in tokens_in_row:
            idxs = [i for i, c in enumerate(candidates)
                    if token["x0"] - 0.30 * H <= c["x"] + c["w"] / 2 <= token["x1"] + 0.30 * H]
            slots = token["slots"]
            if not idxs or not slots:
                continue
            if len(idxs) == len(slots):
                for idx, value in zip(idxs, slots):
                    if value is not None and values[idx] is None:
                        values[idx] = value
                        seed_flags[idx] = True
            else:
                points = []
                span = token["x1"] - token["x0"]
                for j, value in enumerate(slots):
                    px = token["x0"] + span * (j + 0.5) / max(len(slots), 1)
                    points.append((px, value))
                unused = set(idxs)
                for px, value in points:
                    if value is None or not unused:
                        continue
                    best = min(unused, key=lambda i: abs(candidates[i]["x"] + candidates[i]["w"] / 2 - px))
                    distance = abs(candidates[best]["x"] + candidates[best]["w"] / 2 - px)
                    if distance <= 0.55 * H:
                        values[best] = value
                        unused.remove(best)
                        if distance <= 0.22 * H:
                            seed_flags[best] = True

        # 注意：这里**不做**「按 OCR 数字回填音符」。
        # 回填会按 OCR 结果凭空插入音符，OCR 一旦把 0000 读成 00006、
        # 或把延时线读成数字，就会多出幽灵音符（曾表现为多出的 6、
        # 以及 6--- 变成 6-6-）。宁可漏音（标 ?）也不造假音。
        pairs = list(zip(candidates, values, seed_flags))
        pairs.sort(key=lambda p: p[0]["x"])
        candidates = [p[0] for p in pairs]
        values = [p[1] for p in pairs]
        seed_flags = [p[2] for p in pairs]

        # 「内孔」初判 0 / 6，仅作候选（是否启用见下方全局理智检查）。
        hole_values = {}
        for i, c in enumerate(candidates):
            if c.get("synthetic"):
                continue
            hv = _hole_class(c, binary)
            if hv is not None:
                hole_values[i] = hv

        # 行范围：以本行数字块为基准（贴着旋律），上下预留音区点/减时线的位置。
        py0 = max(0, int(min(b["y"] for b in candidates) - 0.9 * H))
        py1 = min(image_h, int(max(b["y"] + b["h"] for b in candidates) + 0.95 * H))
        row_records.append({
            "source": row, "H": H, "baseline": baseline_y,
            "blocks": candidates, "values": values, "seed_flags": seed_flags,
            "hole_hints": hole_values,
            "py0": py0, "py1": py1,
        })

    # 说明：曾加过"行合并"（相邻两行 y 间距 <1.2 倍字高即合并，想治《枉凝眉》
    # 因倚音被拆成两行）。但 regress.py 实测它让 5.png 出现 5→1 退化（行划分变了
    # → 聚类/标注随之改变），取舍后**回退**：宁可 8.jpg 那一行被拆开，也不动
    # 其他谱面的既有正确结果。其余相关改动（倚音行剔除）保留，它只剔行、不拆行。

    if not row_records:
        raise ValueError("已找到旋律文字，但无法定位数字图形块。")

    # 「识别器直调」补值：逐块单独喂识别器（跳过检测器，故不受"孤立数字漏检"影响）。
    # 它是对整行 OCR 文本的补充——整行 OCR 常把数字与歌词并块或读成字母。
    direct_filled = 0
    direct_available = True
    for record in row_records:
        blocks = record["blocks"]
        values = record["values"]
        todo = [i for i, v in enumerate(values) if v is None]
        if not todo:
            continue
        res = _recognize_blocks_directly(engine, [blocks[i] for i in todo], binary)
        if res is None:
            direct_available = False
            break
        for idx, (digit, score) in zip(todo, res):
            if digit is not None and score >= 0.60:
                values[idx] = digit
                record["seed_flags"][idx] = True
                direct_filled += 1
    if direct_available:
        print(f"[ocr] 识别器直调补入 {direct_filled} 个数字")
    else:
        print("[ocr] 识别器直调不可用（该版本的 RapidOCR 未暴露识别器），已跳过")

    # 全局理智检查：若「内孔判 0」命中的比例过高，说明该字体可能是空心字
    # （每个数字都有内孔），此时内孔法失效，整体弃用，避免把整页判成休止符。
    total_blocks = sum(len(r["blocks"]) for r in row_records)
    hole_hits = sum(len(r["hole_hints"]) for r in row_records)
    hole_ratio = hole_hits / max(total_blocks, 1)
    use_hole = hole_ratio <= 0.40
    print(f"[ocr] 内孔命中 {hole_hits}/{total_blocks}（{hole_ratio:.0%}）"
          f"→ {'启用' if use_hole else '比例过高，弃用内孔判定'}")

    for record in row_records:
        if not use_hole:
            record["hole_hints"] = {}
            continue
        for i, hv in record["hole_hints"].items():
            if record["values"][i] is None:
                record["values"][i] = hv
                record["seed_flags"][i] = True

    # 说明：曾有一条「像 0 的形状判定」种子（圆/对称即判 0），但它在某些字体下
    # 会把 3、6 等圆一点的数字全判成休止符（《两只老虎》实测误判 18 个，而该谱
    # 真实只有 2 个 0）。现在改由下面的「标准字形相似度」统一处理 0-7，无需特例。

    # 参照字形：使用**全部候选字体合并**的字形集。
    # 曾改成「按图挑最接近的单一字体」，结果上海滩的 6 又被判成 5（退步），
    # 对小星星也无改善，故回退到合并方案。
    by_font = _reference_glyphs_by_font()
    refs = {}
    for per_digit in by_font.values():
        for digit, feats in per_digit.items():
            refs.setdefault(digit, []).extend(feats)

    # 「本字体数字模型」判定（若存在 digit_model.npz，由 train_digits.py 训练）：
    # 优先级高于系统字体参照字形——它用的是"你自己的谱里真实的 5/6 字形"。
    model = _load_digit_model()
    model_filled = 0
    if model:
        for record in row_records:
            H = record["H"]
            for i, (block, value) in enumerate(zip(record["blocks"], record["values"])):
                if value is not None:
                    continue
                if not (0.50 * H <= block["h"] <= 1.35 * H):
                    continue
                digit = _classify_by_model(block, binary, model)
                if digit is not None:
                    record["values"][i] = digit
                    record["seed_flags"][i] = True
                    model_filled += 1
        print(f"[ocr] 本字体模型补入 {model_filled} 个数字")

    # 「标准字形相似度」判定：把每个还没定值的数字块与参照字形逐一比相似度，
    # 取最像的（要求明显高于第二名）。这是不依赖 OCR 的独立路径。
    ref_filled = 0
    for record in row_records:
        H = record["H"]
        for i, (block, value) in enumerate(zip(record["blocks"], record["values"])):
            if value is not None:
                continue
            if not (0.50 * H <= block["h"] <= 1.35 * H):
                continue
            digit = _classify_by_reference(block, binary, refs,
                                           min_score=0.55, min_margin=0.03)
            if digit is not None:
                record["values"][i] = digit
                record["seed_flags"][i] = True
                ref_filled += 1
    print(f"[ocr] 标准字形相似度补入 {ref_filled} 个数字"
          f"（参照 {len(refs)} 类字形）")

    # 说明：曾有一条「5/6 复核」（看左上角有无墨判 5 / 6），在放大倍率变化后
    # 失效，把上海滩整类 6 翻成了 5（数字分布里完全没有 6、5 达 97 个）。
    # 该判据已移除；5/6 的区分交由下面的「字形聚类 + OCR 投票」处理。

    # 结构级收尾：按字形聚类给每类定一个数字，刷给该类所有块。
    # 这是对抗「逐块判定系统性塌缩」（如 6 全变 5）的关键一步。
    _label_blocks_by_cluster(row_records, binary, refs)

    # 「底部粘着减时线」的最后补救（见 SUB_MERGE_TRIM_FRACS 处的实测说明）：
    # ⚠ 必须放在**字形聚类之后**、且只兜"仍然没定值"的块。先前放在"标准字形相似度"
    # 那一趟（聚类之前），结果抢在聚类前给了错值：剪掉底部会把 `6` 的尾巴削掉、
    # 看着就像 `0`（实测 2.png / 5.png 各坏一个 6、7.png 坏一个 5）。
    # ⚠ 还要求"块高明显超过本行字高"——粘了减时线的块必然偏高（实测 51 vs 43 = 1.19×），
    # 正常数字 ≈1.0×，所以它们根本不会进来试剪。
    trim_filled = 0
    for record in row_records:
        H = record["H"]
        for i, (block, value) in enumerate(zip(record["blocks"], record["values"])):
            if value is not None or block["h"] <= 1.15 * H:
                continue
            for frac in SUB_MERGE_TRIM_FRACS:
                trimmed = dict(block)
                trimmed["h"] = max(int(block["h"] * (1.0 - frac)), 4)
                d2 = _classify_by_reference(trimmed, binary, refs,
                                            min_score=0.55, min_margin=0.03)
                if d2 is not None:
                    record["values"][i] = d2
                    trim_filled += 1
                    if VERBOSE_DEBUG:
                        print(f"  [trim] x={block['x']:.0f} 块高{block['h']:.0f}"
                              f"（行字高{H:.0f}）判不出，剪掉底部 {frac:.0%}"
                              f" → {d2}")
                    break
    if trim_filled:
        print(f"[ocr] 底部粘线的块剪掉后补入 {trim_filled} 个数字")

    # 0 / 6 定向复核（只碰这一对）：用「上下质量分布 + 内孔位置」双判据，
    # 两者一致时才改判，否则保持原值。0 与 6 都是闭合圈，需专门区分。
    zero_six_fixed = 0
    zero_six_held = 0
    refs_vote = _all_reference_glyphs()   # 第三票要用的参照字形（只建一次）
    for record in row_records:
        for i, (block, value) in enumerate(zip(record["blocks"], record["values"])):
            if value not in (0, 6):
                continue
            judged, detail = _judge_zero_or_six(
                block, binary, H=record["H"], return_detail=True)
            # 第三票（用户 2026-09-22 报 18.jpg《青花瓷》"部分 6 变成 0"）：
            # 前两票（上下对称 + 内孔）在本字体上双双退化 —— 实测这 4 个真 6 的
            # 上下差只有 0.01~0.07（阈值 0.12 → `sym` 恒判 0），而孔比 0.18~0.21
            # 越过了 `HOLE_ZERO_MIN_RATIO=0.16`（那条 0.16 是在**别的字体**上定的：
            # 那个字体的 6 孔比 0.10~0.12、真 0 ≈0.23；青花瓷的 6 是**大孔**，
            # 正好落在两者之间——`_hole_class` 的 docstring 预警过这个重叠）。
            # 于是 17 个真 6 被翻成 0；下游"假休止复核"靠形状分类器救回 13 个，
            # **剩 4 个救不回**——因为那 4 个块的 6 与 0 相似度只差 0.015~0.028，
            # 卡在分类器要求的最小间隔（0.03）里 → 返回 None → 无从施救。
            # 所以这里补第三票：**改判必须得到形状分类器支持**（它明确说出改判后的
            # 那个数字、且分数 ≥ FAKE_REST_MIN_SCORE）。实测两侧都有大空档：
            # 真 0 的 (0 相似 − 6 相似) 全 ≤ −0.082（分类器明确说 0，分 0.78~0.87）→
            # 该改的 6→0 照样成立；真 6 的间隔 18/20 为正、分类器说 6（0.75~0.84）→
            # 误翻被拦住。阈值复用兄弟机制 `FAKE_REST_MIN_SCORE`（同一条"形状证据"口径）。
            if judged is not None and judged != value:
                vote, vote_score = _classify_by_reference(
                    block, binary, refs_vote, return_score=True)
                if vote != judged or vote_score < FAKE_REST_MIN_SCORE:
                    if VERBOSE_DEBUG:
                        print(f"  [06] x={block['x']:.0f} 两票想改判 {value}→{judged}，"
                              f"但形状分类器给 {vote}({_fmt_num(vote_score)}) 不支持 → "
                              f"保持 {value}")
                    judged = None
                    zero_six_held += 1
            if VERBOSE_DEBUG:
                # 关注点：`对称判` 与 `内孔判` 不一致时这里是**保持原值**。
                # 实测《南泥湾》32 个块的 `对称判` 恒为 0（上下差都 ≤0.09），
                # 双判据退化成单判据 → 5 个真 6 被改成 0。
                # 看 `孔cy`（内孔中心的相对高度，0=顶 1=底）：
                #   `孔cy` 落在 0.60~0.66 → 被"孔居中→0"那条规则**抢先**判成 0
                #   （`|cy-0.5|<=0.16` 覆盖到 0.66，而"→6"要求 cy>=0.60，区间重叠），
                #   这就是 6 被误判的现场。`孔比` 太小则说明根本没找对孔。
                hd = detail["hole_detail"] or {}
                verdict = (f"改判 {judged}" if judged is not None and judged != value
                           else f"保持 {value}")
                print(f"  [06] x={block['x']:.0f} 现值={value} "
                      f"上下差={_fmt_num(detail['balance'])} "
                      f"top={_fmt_num(detail['top'])} "
                      f"bot={_fmt_num(detail['bot'])} "
                      f"块高/H={_fmt_num(detail['h_ratio'])} "
                      f"孔cy={_fmt_num(hd.get('cy'))} "
                      f"孔cx={_fmt_num(hd.get('cx'))} "
                      f"孔比={_fmt_num(hd.get('ratio'))} "
                      f"对称判={detail['sym']} 内孔判={detail['hole']} → {verdict}")
            if judged is not None and judged != value:
                record["values"][i] = judged
                zero_six_fixed += 1
    if zero_six_fixed:
        print(f"[ocr] 0/6 双判据改判 {zero_six_fixed} 个块")
    if zero_six_held:
        print(f"[ocr] 0/6 复核被形状分类器拦下 {zero_six_held} 个"
              f"（两票想改，第三票不支持）")

    # 「假休止」复核（用户 2026-09-21 报"6 被标记成 0 无法识别"，7.png 与 14.jpg 上都有）：
    # 判成 0 的音里混着**被误判成休止的数字** —— 原位替换图上它们既换不成字母、也留不下数字。
    # 用「形状最像哪个数字」复核：真休止最像的当然是 0（实测 56 个真 0 的形状分 0.76~0.94），
    # 而误判的那 4 个最像 6（0.80~0.88）→ 直接把值改成它最像的那个数字。
    # ⚠ 必须放在 0/6 复核**之后**：那一步正是把 14.jpg 行1 那个真 6 改成 0 的元凶
    # （它用的"上下墨量对称"在密集小字上恒为 0、退化成只看内孔），这里把形状证据补回来。
    fake_rest_fixed = 0
    refs_all = _all_reference_glyphs()
    for record in row_records:
        for i, (block, value) in enumerate(zip(record["blocks"], record["values"])):
            if value != 0:
                continue
            digit, score = _classify_by_reference(block, binary, refs_all,
                                                  return_score=True)
            if digit not in (None, 0) and score >= FAKE_REST_MIN_SCORE:
                record["values"][i] = digit
                fake_rest_fixed += 1
                if VERBOSE_DEBUG:
                    print(f"  [0?] x={block['x']:.0f} 判成休止，但形状最像 {digit}"
                          f"（{score:.2f}）→ 改判 {digit}")
    if fake_rest_fixed:
        print(f"[ocr] 假休止复核：{fake_rest_fixed} 个休止改判成数字")

    # 从可靠样本建立本图字体模板：每类只保留一个「平均字形」。
    # 用平均而不是「取最大相关」，是为了避免样本多的类（如 0）占便宜、
    # 把其它数字吸过去。
    samples_by_label = defaultdict(list)
    for record in row_records:
        for block, value, is_seed in zip(record["blocks"], record["values"], record["seed_flags"]):
            if is_seed and value is not None:
                samples_by_label[value].append(_shape_feature(block, binary))
    templates = {}
    for label, feats in samples_by_label.items():
        stack = np.stack(feats[:30])
        templates[label] = stack.mean(axis=0)

    # 剩余用「字形最近邻」补全：要求绝对分数与第一/第二名间隔都达标，避免瞎猜。
    for record in row_records:
        for i, (block, value) in enumerate(zip(record["blocks"], record["values"])):
            if value is not None:
                continue
            feature = _shape_feature(block, binary)
            scores = sorted(((_corr(feature, mean), label)
                             for label, mean in templates.items()), reverse=True)
            if not scores:
                continue
            best_score, best_label = scores[0]
            second_score = scores[1][0] if len(scores) > 1 else -1.0
            if best_score >= TEMPLATE_MIN_SCORE and best_score - second_score >= TEMPLATE_MIN_MARGIN:
                record["values"][i] = best_label

    # 标记「哪些块是真音符（有值）」：延时线的右端只看真音符，跳过竖线/括号等。
    for record in row_records:
        for block, value in zip(record["blocks"], record["values"]):
            block["_is_digit"] = value is not None

    measures = []
    total_blocks = 0
    total_labeled = 0
    kept_rows = 0

    # 0（休止符）形状校验：不像休止符的一律撤销，避免假 0 把后续音符顶偏。
    dropped_zeros = 0
    for record in row_records:
        for i, (block, value) in enumerate(zip(record["blocks"], record["values"])):
            if value != 0:
                continue
            if _zero_like(block, binary) or _hole_class(block, binary) == 0:
                continue
            record["values"][i] = None
            dropped_zeros += 1
    if dropped_zeros:
        print(f"[ocr] 撤销 {dropped_zeros} 个形状不符的休止符判定（改为 ?）")

    # 5 / 6 定向复核：本字体里 **5 没有封闭内孔、6 有** —— 所以「标为 6 却
    # 完全没有内孔」的块必然是被误判的 5。
    # 依据（《南泥湾》实测）：25 个真 5 的 `孔比` 全是 0.00；23 个真 6 是 0.10~0.12。
    # 抽查过其中 3 个标为 6 的（x=2441/1837/2568）回原谱确认都该是 5。
    # **只做这一个方向**：反向（5 有孔 → 6）逻辑上也成立，但本谱没有反例样本可验，
    # 不加，免得凭空承担风险。
    # 注意：必须放在**所有会改 values 的步骤之后**（内孔补入/模型/标准字形/聚类改判/
    # 0-6 复核/模板匹配/假 0 撤销），否则会被后面的步骤覆盖掉。
    five_six_fixed = 0
    for record in row_records:
        for i, (block, value) in enumerate(zip(record["blocks"], record["values"])):
            if value != 6:
                continue
            _hv, hole = _hole_class(block, binary, return_detail=True)
            ratio = (hole or {}).get("ratio")
            if ratio is None or ratio >= HOLE_FIVE_MAX_RATIO:
                continue
            record["values"][i] = 5
            five_six_fixed += 1
            if VERBOSE_DEBUG:
                print(f"  [56] x={block['x']:.0f} 标为 6 但 孔比={ratio:.3f}"
                      f"（毫无封闭内孔）→ 改判 5")
    if five_six_fixed:
        print(f"[ocr] 5/6 复核：{five_six_fixed} 个块由 6 改判为 5（无内孔）")

    # 逐音置信度：**必须在所有判定都做完之后算**——上面每一处
    # `record["values"][i] = ...`（内孔补入 / 本字体模型 / 标准字形 / 聚类改判 /
    # 0-6 复核 / 形状校验）都会改结论，早算就会算到中间状态。
    # 它只被记录、**不参与任何判定**。
    conf_refs = _all_reference_glyphs()
    low_conf = 0
    for record in row_records:
        H = record["H"]
        confs = []
        for block, value in zip(record["blocks"], record["values"]):
            conf = None
            if value is not None and 0.50 * H <= block["h"] <= 1.35 * H:
                best_digit, score = _classify_by_reference(
                    block, binary, conf_refs, return_score=True)
                if score is not None and score >= 0:
                    conf = round(float(score), 3)
                    # 形状判据给出的「最像的数字」与最终结论不一致 → 明确可疑
                    if best_digit is not None and best_digit != value:
                        conf = min(conf, DISAGREE_CONF)
            if conf is None or conf < CONF_WARN:
                low_conf += 1
            confs.append(conf)
        record["confs"] = confs
    if low_conf:
        print(f"[ocr] 置信度低于 {CONF_WARN:.0%} 的音 {low_conf} 个"
              f"（输出图里已加黄底，清单见 [置信度]）")

    if VERBOSE_DEBUG:
        # 5 / 6 混淆诊断。注意曾有两个函数 `_looks_like_five()` / `_refine_five_six()`
        # 用"左上角墨"判 5/6 —— 它们从未被调用（死代码，现已删除），那条判据
        # 实际上一次都没生效过。这里**只量不判**：把每个标为 5 或 6 的块的几个候选特征
        # 打出来，看哪个量真能把"真 5"和"真 6"分开（照抄 0/6 那次的打法：先找有物理
        # 意义的判据、再定阈值，不靠猜）。已知上次用"顶部横杠顺序"赌过一次、结果更差
        # 已移除，所以这次必须先看分布。
        for row_i, record in enumerate(row_records, 1):
            for note_i, (block, value) in enumerate(
                    zip(record["blocks"], record["values"]), 1):
                if value not in (5, 6):
                    continue
                roi = binary[block["y"]:block["y"] + block["h"],
                             block["x"]:block["x"] + block["w"]] > 0
                if roi.size == 0:
                    continue
                h, w = roi.shape
                if h < 8 or w < 5:
                    continue
                # 候选①：左上角 30%×30% 的墨密度（已删除的 _refine_five_six 用的就是它）
                tl_ink = float(roi[:max(1, int(0.30 * h)),
                                   :max(1, int(0.30 * w))].mean())
                # 候选②：顶部带（10%~35% 高）里"有墨的列"占比（已删除的 _looks_like_five 的一半）
                y0, y1 = int(0.10 * h), max(int(0.10 * h) + 1, int(0.35 * h))
                top_cov = float(roi[y0:y1].any(axis=0).mean())
                # 候选③：内孔面积比 / 内孔中心高度（0/6 那次靠它分的）
                _hv, hole = _hole_class(block, binary, return_detail=True)
                hole = hole or {}
                print(f"  [56] 行{row_i} 音{note_i} x={block['x']:.0f} 现值={value} "
                      f"左上角墨={tl_ink:.2f} 顶覆盖={top_cov:.2f} "
                      f"孔比={_fmt_num(hole.get('ratio'))} "
                      f"孔cy={_fmt_num(hole.get('cy'))}")

    for row_index, record in enumerate(row_records):
        H = record["H"]
        blocks = record["blocks"]
        values = record["values"]
        confs = record["confs"]
        py0, py1 = record["py0"], record["py1"]
        # 本行 OCR 文本（同样弃用跨行 token）
        tokens_in_row = [t for t in record["source"]["tokens"]
                         if (t["y1"] - t["y0"]) <= 1.5 * H]
        tokens_in_row.sort(key=lambda t: t["x0"])
        row_components = [c for c in all_components if c["y"] < py1 and c["y"] + c["h"] > py0]

        # 用中位数而非 min/max，避免个别离群块把「贯穿全行」的门槛变得苛刻
        tops = sorted(b["y"] for b in blocks)
        bots = sorted(b["y"] + b["h"] for b in blocks)
        digits_top = float(np.median(tops))
        digits_bottom = float(np.median(bots))
        # 本行数字的纵向中心。括号判据（`PAREN_MAX_DY_RATIO`）要用它；
        # 注意**不能**用 `baseline_y`（它是上一个循环的局部变量，见括号过滤处的说明）。
        digits_cy = (digits_top + digits_bottom) / 2

        # 竖线检测：先严格；若数量明显少于「应有小节数」（按总拍数估算），
        # 自动改用宽松条件重找一遍，取结果更多的一次。这样不依赖「休止符比例」
        # 这类间接信号，全休止符行、长音行都能自动适配套用。
        vlines = _detect_vlines(row_components, blocks, H, digits_top,
                                digits_bottom, relaxed=False)

        accepted_ids = {id(c) for c in vlines}
        vline_candidates = [(c["x"] + c["w"] / 2.0, id(c) in accepted_ids)
                            for c in row_components
                            if c["h"] >= 0.8 * H and c["w"] <= 0.5 * H]
        record["vline_candidates"] = vline_candidates
        hlines = [c for c in row_components
                  if c["h"] <= 0.30 * H and c["w"] >= 0.35 * H
                  and c["w"] / max(c["h"], 1) >= 3.0]
        dots = [c for c in row_components
                if 0.008 * H * H <= c["area"] <= 0.30 * H * H
                and 0.40 <= c["w"] / max(c["h"], 1) <= 1.90
                and c["area"] / max(c["w"] * c["h"], 1) >= 0.48
                and c["w"] <= 0.42 * H and c["h"] <= 0.42 * H]

        # 括号块从「数字候选」里剔除：它是标点不是音符。
        # 留着会有两个坏处：(a) 被字形分类器赋上数字值；(b) 堵住延时线的探测窗口
        # （实测 `)` 就在音符右边，导致横线被截、延时线只数出 1 条）。
        parens = _detect_paren_components(row_components, blocks, H, binary, dots)
        if parens and REMOVE_PAREN_BLOCKS:
            seed_flags = record["seed_flags"]   # 必须取本条记录的，不能用上个循环的残留
            # ⚠ 删块前先问一句"它自己像不像数字"（用户 2026-09-21 报 17.jpg 里 `1` 整批消失）：
            # 括号候选与数字块来自**同一份连通块**，所以某个数字块自己也像括号时，
            # `_overlaps_any(b, parens)` 是"自己跟自己重叠" → 会把真音符删掉 ✗。
            # 这里用**低门槛**（PAREN_DELETE_MIN_DIGIT_SCORE）判：删除是危险方向，宁可保留；
            # 14.jpg 那个真 `(` 分类器给不出数字（0.48）→ 照旧删；17.jpg 的 `1`（0.75~0.92）→ 留。
            keep = []
            for i, b in enumerate(blocks):
                if not _overlaps_any(b, parens):
                    keep.append(i)
                    continue
                _d, _sc = _classify_by_reference(b, binary, _all_reference_glyphs(),
                                                 return_score=True)
                if _d is not None and _sc >= PAREN_DELETE_MIN_DIGIT_SCORE:
                    keep.append(i)          # 像数字 → 留着当音符
            blocks = [blocks[i] for i in keep]
            values = [values[i] for i in keep]
            seed_flags = [seed_flags[i] for i in keep]
            record["blocks"] = blocks
            record["values"] = values
            record["seed_flags"] = seed_flags

        notes = []
        for block_index, (block, value) in enumerate(zip(blocks, values)):
            cx = block["x"] + block["w"] / 2
            cy = block["y"] + block["h"] / 2

            # 先数减时线：低音点的探测窗口要据此下移
            # （有减时线时低音点画在线下方，线越多点越靠下）。
            beams = _count_beams(block, hlines, H)

            up = sum(1 for p in dots
                     if abs(p["x"] + p["w"] / 2 - cx) <= 0.28 * H
                     and block["y"] - 0.68 * H <= p["y"] + p["h"] / 2 <= block["y"] - 0.14 * H)
            if block.get("merged_dot"):
                up += 1  # 高音点与数字粘连，已拆分出来
            # 低音点：窗口从「减时线之下」开始，步进按线数下移
            dot_low = 0.10 + 0.22 * beams
            down = sum(1 for p in dots
                       if abs(p["x"] + p["w"] / 2 - cx) <= 0.28 * H
                       and block["y"] + block["h"] + dot_low * H
                       <= p["y"] + p["h"] / 2
                       <= block["y"] + block["h"] + (dot_low + 0.55) * H)

            # 减时线：用横线本体判定（覆盖本数字、位于下方合适距离）
            dotted = any(
                block["x"] + block["w"] + 0.05 * H <= p["x"] + p["w"] / 2
                <= block["x"] + block["w"] + 0.65 * H
                and abs(p["y"] + p["h"] / 2 - cy) <= 0.28 * H
                for p in dots
            )

            # 延时线：数数字右侧的独立短横段数。
            # 右端取三者的最近：下一个真音符左缘 / 右邻小节线 / `EXT_SCAN_MAX_SPAN` 兜底。
            # ⚠ 必须有"小节线"这条：延时线**从不跨小节**，少了它窗口会伸进下一小节，
            #   把别人的横线/连音弧算进来（用户 2026-09-22 实测 8.jpg 行1 音2：
            #   原谱只有 `6 - -` 两条，扫到下一个数字却数出 4 条 ✗）。
            next_left = float("inf")
            for nb in blocks[block_index + 1:]:
                if nb.get("_is_digit"):
                    next_left = nb["x"]
                    break
            next_bar = float("inf")
            for _v in vlines:
                _vx = _v["x"] + _v["w"] / 2.0
                if block["x"] + block["w"] < _vx < next_bar:
                    next_bar = _vx
            # row / note_no / beams 只喂给 _count_extends 做 VERBOSE_DEBUG 打印用
            extend, ext_xs = _count_extends(block, binary, H, next_left, has_dot=dotted,
                                            next_bar=next_bar,
                                            row=row_index + 1, note_no=block_index + 1,
                                            beams=beams)
            # 再用**旧的 6.0H 兜底窗**数一遍，取条数多的那次：
            # 扩窗的目的只是"看到被 6.0H 截掉的长延时线"；若多扫反而更少，说明多收进来的
            # 墨（上方连音弧等）把那一段的判定搞坏了 —— 21.jpg 行7 音16 实测原谱
            # `i - - 0`（2 条），扩窗后只剩 1 条 ✗。取多者**永远不会比旧行为差**。
            if _ext_window_end(block, H, next_left, next_bar, 6.0) < \
                    _ext_window_end(block, H, next_left, next_bar, EXT_SCAN_MAX_SPAN) - 1:
                ext2, xs2 = _count_extends(block, binary, H, next_left, has_dot=dotted,
                                           next_bar=next_bar, row=row_index + 1,
                                           note_no=block_index + 1, beams=beams,
                                           max_span=6.0, quiet=True)
                if ext2 > extend:
                    extend, ext_xs = ext2, xs2

            q = (1.0 / (2 ** beams)) * (1.5 if dotted else 1.0) + extend
            notes.append({
                "rest": value == 0,
                "n": value,
                "o": max(-3, min(3, up - down)),
                "q": round(q, 4),
                "markup": {"beams": beams, "dotted": dotted, "extend": extend},
                "lyric": "", "slur_start": [], "slur_stop": [], "cx": cx,
                "ext_xs": ext_xs,   # 每条延时线在原图中的中心 x（渲染直接用）
                # 这个数字在**缩放后图片**里的外框 (x, y, w, h)。只在"原位替换"
                # （把原谱上的数字直接盖成字母，见 overlay_letters.py）里用得上：
                # 那条路一个笔画都不动，必须知道每个数字落在哪。
                "box": (block["x"], block["y"], block["w"], block["h"]),
                "paren_before": False, "paren_after": False,
                # 重音记号 `>`：由本行收尾的记号检测填（画在数字上方）。
                # 元素默认 False，避免下游用 .get() 拿到 None 时行为不一致。
                "accent": False,
                # 上方另外三种记号，同样由本行收尾的检测填：
                #   breath  `v` 换气    triplet  `³` 三连音    house  房子标号 "1"/"2"/"3"
                "breath": False,
                "triplet": False,
                "house": "",
                # 房子**横线**的两个端点（本节的首音/末音）：渲染时按它们的位置连线，
                # 和连音弧一样"模型里只存端点"。由检测房子标号时顺手填。
                "house_start": False,
                "house_stop": False,
                # 连音弧的**开放端**（跨行弧的残段）：
                #   slur_left  = 这个音有一条**从行首伸过来**的弧（＝下一行开头的那个音）
                #   slur_right = 这个音有一条**伸向行末**的弧
                # 原谱的连音弧跨行断开，每行只剩半截。用户 2026-09-18 选定"两段都画"。
                "slur_left": False,
                "slur_right": False,
                # 逐音置信度（0..1，None=拿不到分数）。只供"标出可疑音"用，
                # 渲染图里低于 CONF_WARN 的会加黄底；不参与任何判定。
                "conf": confs[block_index] if block_index < len(confs) else None,
            })

        # 歌词对齐：把这一行下方的汉字**逐字**分配给位置最近的音符。
        # 音符的 lyric 字段与渲染函数早已支持（会画在音符正下方），
        # 只是识别端此前没有填。歌词在音符下方约 0.5-3.5 倍字高范围。
        if text_tokens and notes:
            row_bottom = max(b["y"] + b["h"] for b in blocks)
            cand = [t for t in text_tokens
                    if row_bottom <= t["cy"] <= row_bottom + 3.5 * H]
            # 只取**最上面那一段**歌词：一份谱常有三段歌词叠排在同一竖位
            # （如 `1.浪奔 2.(是)喜 3.(又有)喜`），全取会把三段叠到同一批音符下，
            # 表现为"浪浪""爱爱"这种重复。
            if cand:
                # 只取**最上面那一段**歌词：以最上面那个块为基准，取 0.5 倍字高内
                # 的所有块。（三段歌词排得密时，这个窗口比"串链式"更稳。）
                top_cy = min(t["cy"] for t in cand)
                cand = [t for t in cand if t["cy"] - top_cy <= 0.5 * H]
            row_notes = sorted(notes, key=lambda nd: nd["cx"])
            for tk in cand:
                text = tk["text"]
                if not text:
                    continue
                span = tk["x1"] - tk["x0"]
                for ci, ch in enumerate(text):
                    if not ch.strip():
                        continue
                    xc = tk["x0"] + span * (ci + 0.5) / max(len(text), 1)
                    best = min(row_notes, key=lambda nd: abs(nd["cx"] - xc))
                    if VERBOSE_DEBUG:
                        _dh = abs(best["cx"] - xc) / H
                        if _dh > 1.0:
                            print(f"  [lyr] 行{row_index + 1} 块'{text}' ch={ch} xc={xc:.0f} "
                                  f"最近音cx={best['cx']:.0f} 距={_dh:.2f}H "
                                  f"{'附着' if _dh <= 1.3 else '丢弃'}")
                    if abs(best["cx"] - xc) <= 1.3 * H:
                        best["lyric"] = (best.get("lyric") or "") + ch

        labeled = sum(n["n"] is not None for n in notes)
        ratio = labeled / max(len(notes), 1)

        # 括号：细弧线且填充率低。必须排除与已知数字块重叠的成分，否则会把
        # 数字笔画碎片误判成 "(" / ")"。真括号很窄、呈弧形（填充率低）。
        parens = []
        for c in row_components:
            if not (0.8 * H <= c["h"] <= 1.9 * H):
                continue
            if not (0.15 * H <= c["w"] <= 0.50 * H):
                continue
            if not (0.20 <= c["w"] / max(c["h"], 1) <= 0.50):
                continue
            if c["area"] / max(c["w"] * c["h"], 1) >= 0.45:
                continue

            # ⚠ 与 `_detect_paren_components()` 同样的否决（这份旧实现在下面覆盖了函数版，
            # 先前只改函数 → 无效；见 `_paren_looks_like_note` 的说明）
            if _paren_looks_like_note(c, binary, dots, H):
                continue
            # 纵向必须在数字行带内（判据与实测见 `PAREN_MAX_DY_RATIO`）：否则歌词汉字的
            # 偏旁会被当成括号，在音符之间画出一堆假的 `(`/`)`。
            # ⚠ 行带中心**必须**用本行的 `digits_top/digits_bottom` 现算 —— `baseline_y`
            #   是上一个循环（`for row in anchored_rows`）的局部变量，泄漏到这里时是
            #   **最后一行的**基线：拿它比会把整页的真括号全判成"出带"，实测 10.png 的
            #   两行 `括号：保留` 直接消失（真括号也丢了）。
            if abs(c["y"] + c["h"] / 2 - digits_cy) > PAREN_MAX_DY_RATIO * H:
                continue
            # 与任何已识别数字块重叠则跳过（排除数字笔画碎片）
            if any(c["x"] < b["x"] + b["w"] and b["x"] < c["x"] + c["w"]
                   and c["y"] < b["y"] + b["h"] and b["y"] < c["y"] + c["h"]
                   for b in blocks):
                continue
            parens.append(c)        # 括号定位：只用 OCR 文本里的 ( ) 字符（含全角 （），RapidOCR 常输出全角）。
        paren_marks = []
        rejected_ocr_parens = 0
        fw_table = str.maketrans("（）〔〕", "()[]")
        # 几何上"像括号"的位置（供与 OCR 文本交叉确认）
        geom_xs = [c["x"] + c["w"] / 2 for c in parens]
        for tk in tokens_in_row:
            text = tk["text"].translate(fw_table)
            if "(" not in text and ")" not in text:
                continue
            n_chars = max(len(text), 1)
            span = tk["x1"] - tk["x0"]
            for idx, ch in enumerate(text):
                if ch not in "()":
                    continue
                mx = tk["x0"] + span * (idx + 0.5) / n_chars
                # **双确认**：OCR 说这里有括号，几何上必须也有一个像括号的块。
                # 否则 OCR 把数字/竖线误读成 `(` 时，图上会冒出假括号
                # （实测 2.png 多处多余 `(` 就是 OCR 文本单方面造成的）。
                if not any(abs(mx - gx) <= 1.0 * H for gx in geom_xs):
                    rejected_ocr_parens += 1
                    continue
                print(f"  [par-mark] OCR文本路径 x={mx:.0f} 字符={ch} "
                      f"几何匹配={min((abs(mx - gx) for gx in geom_xs), default=-1):.0f}px")
                paren_marks.append((mx, ch))
        # 几何兜底：弧线只要**够高**（≥1.1 倍字高）就认，不限定位置——
        # 括号常出现在行中间（如 `1 - ) 0 3 5`）。用「高」把数字碎片挡在外面，
        # 之前不加这条时产生过大量假括号。
        # 说明：括号采用**交集策略**——只有「OCR 文本读到括号」且「几何上同一位置
        # 有像括号的弧线」两者都成立时才显示。
        # 曾试过"任一条命中就显示"，结果：几何碎片会在音符之间造出假括号
        # （2.png 实测 5 处），且"紧贴音符"这类几何约束方向不可靠（真括号跟在长音
        # 后面时距离反而更远）。故改为宁缺勿假：只保留双确认的括号。
        # 几何独立证据：括号几乎与小节线等高（高度已在上游限定 1.2-2.2H），
        # 且必须"是弧"（曲率 ≥0.25）。这样 OCR 读不出括号时也能有括号，
        # 而弯曲的数字碎片因**高度不足**被挡在外面。
        for c in parens:
            if _arc_variation(c, binary) < 0.25:
                continue
            paren_marks.append((c["x"] + c["w"] / 2, _paren_type(binary, c)))
        if paren_marks or rejected_ocr_parens:
            print(f"[ocr] 括号：保留 {len(paren_marks)} 个"
                  + (f"，丢弃 {rejected_ocr_parens} 个（OCR 读到但几何上无对应弧线）"
                     if rejected_ocr_parens else ""))

        paren_marks.sort(key=lambda item: item[0])
        deduped = []
        for x_est, ch in paren_marks:
            if deduped and abs(x_est - deduped[-1][0]) <= 0.6 * H:
                continue
            deduped.append((x_est, ch))

        for px, ch in deduped:
            if ch == "(":
                right = [nd for nd in notes if nd["cx"] >= px]
                target = min(right, key=lambda nd: nd["cx"]) if right else \
                    min(notes, key=lambda nd: abs(nd["cx"] - px))
                target["paren_before"] = True
            else:
                left = [nd for nd in notes if nd["cx"] <= px]
                target = max(left, key=lambda nd: nd["cx"]) if left else \
                    min(notes, key=lambda nd: abs(nd["cx"] - px))
                target["paren_after"] = True

        # ---- 连音弧 `⌒`  ----------------------------------------------------
        # 特征：**在数字上方**、又扁又宽、且「拱起」（逐列墨心 y 的极差大，
        # 见 `_hump()`——注意不是 `_arc_variation()`，那是给竖弧用的）。
        # 位置条件（必须在数字上方）本身就排掉了延时线 `-`（在数字中线高度）
        # 和减时线（在数字下方）；宽 × 扁 × 拱 再排掉高音点（太窄）、
        # 倚音的细小音符（太高）、以及零碎墨点。
        # 注意：**不能**复用 `digits_top` 这个名字——它在上面已被定义为
        # `np.median(tops)` 并且下面的"小节线宽松重找"还要用它；这里覆盖会
        # 悄悄改掉小节线检测的结果。故另起名字。
        slur_digits_top = min((b["y"] for b in blocks), default=0)
        if VERBOSE_DEBUG:
            # **只量不判**：把"数字上方那条带子"里**所有**连通块都列出来，
            # 附上筛选用的全部量。目的是看清连音弧到底以什么形态存在——
            # 是完整的宽扁弧（那说明阈值定错），还是被打碎成若干短段
            #（细线二值化后常被打断，延时线就踩过这个坑：「墨宽 0.3H 却分成 3 段」）。
            band_c = [c for c in all_components
                      if c["y"] + c["h"] <= slur_digits_top + 0.30 * H
                      and c["y"] >= slur_digits_top - 2.2 * H]
            band_c.sort(key=lambda c: c["x"])
            for c in band_c:
                fill = c["area"] / max(c["w"] * c["h"], 1)
                # 距顶 = 这个块的**底边**离本行数字顶有多远（正数＝在上方）。
                # 这是校准"搜索窗口能开多高"的**唯一依据**：窗口开 2.2H 会把
                # **上一行的数字**也收进来（第 8 行那 19 个假三连音就是这么来的），
                # 要收紧窗口就得先知道真记号离数字顶多远。
                lift = (slur_digits_top - (c["y"] + c["h"])) / H
                print(f"  [slur?] 行{row_index + 1} x={c['x']:.0f} 顶y={c['y']:.0f} "
                      f"距顶={lift:+.2f}H 宽={c['w'] / H:.2f}H 高={c['h'] / H:.2f}H "
                      f"填充={fill:.2f} 拱起={_hump(c, binary):.2f}")
                # 小块/中等块再打一张 ASCII 形状图：数字上方的**记号**
                #（`>` 重音 / `v` 换气 / `³` 三连音 / 音区点）用宽高填充率分不干净，
                # **看形状最直接**（尖朝右 = `>`、尖朝下 = `v`、是个数字 = `³`、圆点 = 音区点）。
                # ⚠ 门槛按**宽度 ≤1.2H** 收：`>` 实测 0.85H×0.64H，早先写的"边长 ≤0.6H"
                # 把唯一要看的东西排除在外了（2026-09-18 踩过）。1.2H 同时排掉
                # 连音弧（≥1.6H）和房子括线（17~44H）。
                if c["w"] <= 1.2 * H:
                    sub = binary[c["y"]:c["y"] + c["h"], c["x"]:c["x"] + c["w"]] > 0
                    # step=2（每格 2×2 像素）：0.85H×0.64H 的 `>` 约 39×29px → 20×15 字符，
                    # 形状仍分得清，行数减半（否则一块就 30+ 行、整谱几千行）。
                    for line in _ascii_mask(sub, step=2):
                        print(f"         |{line}|")
        arcs = []
        for c in all_components:
            # 顺序很重要：**先判位置**（这个块属不属于本行的带子），再判宽度/高度。
            # 反过来的话，与这一行无关的宽块也会被扫到并打印，
            # 日志里就会出现"同一个块在每一行都报一次、行号是假的"。
            if c["y"] + c["h"] > slur_digits_top + 0.30 * H:
                continue                            # 必须落在数字上方
            if c["y"] < slur_digits_top - 2.2 * H:
                continue                            # 也别离得太远（挡标题/上一行）
            if c["w"] > ARC_MAX_W * H:
                # 太宽 → 不是连音弧，是房子/反复括线（见 ARC_MAX_W 的说明）。
                # 这类块常与紧邻的 `)` 粘成一块，光看拱起/高度分不干净。
                # 放在高度条件**之前**，是为了不被高度条件静默拒掉、看不见原因。
                if VERBOSE_DEBUG:
                    print(f"  [slur!] 行{row_index + 1} x={c['x']:.0f} "
                          f"宽={c['w'] / H:.1f}H 高={c['h'] / H:.2f}H "
                          f"拱起={_hump(c, binary):.2f} → 太宽，判为括线而非连音弧")
                continue
            if not (0.06 * H <= c["h"] <= 0.90 * H):
                # 高度上限一路放宽过来的：0.45H（按短弧估）→ 0.75H（按《南泥湾》0.63H）
                # → **0.90H**。第三次放宽的实测依据：《屁》的弧到 **0.77H**，被 0.75H
                # 挡掉了 —— 而且当时被接受的那批里最高一条是 0.74H，**只差 0.01 就过不了**，
                # 纯属侥幸（`[slur?] 行1 ` 里 x=800 宽5.87H 填充0.16 拱起0.86 的那条真弧，
                # 形状判据全过、就是卡在高度上）。
                # 上限取 0.90H 的依据：弧实测 ≤0.77H，而**房子括线 1.09~1.13H**
                #（两端竖钩撑着），两者之间取中，余量约 1.17× / 1.21×。
                continue                            # 太扁（噪点）或太高
            if c["w"] < 0.80 * H:
                continue                            # 太窄，跨不了两个音
            if c["y"] + c["h"] > slur_digits_top + 0.30 * H:
                continue                            # 必须落在数字上方
            if c["y"] < slur_digits_top - 2.2 * H:
                continue                            # 也别离得太远（挡标题/上一行）
            if c["area"] > 0.60 * c["w"] * c["h"]:
                continue                            # 细弧填不满自己的外接框
            if _hump(c, binary) < ARC_MIN_HUMP:
                continue
            arcs.append(c)
        # 同一条弧可能被连通域拆成两段：按 x 去重（先留更宽的）。
        # ⚠ **光看 x 不够**：原谱里常有"上下叠着的两条弧"（同一起点、跨度不同），
        # 它们的中心 x 也能落在 0.6H 内 —— 只看 x 就会把下面那条真弧当断片丢掉
        # （用户 2026-09-20 报的《孤勇者》"2 个弧只画出一个"）。
        # 实测该谱行9 x=660 两条候选：宽2.91H（距顶+1.09H）与宽1.85H（距顶+0.74H），
        # 纵向只重叠 26% —— 是两条独立的弧。真正的断片是同一条笔画的两截，
        # 纵向几乎完全重叠。所以再加一条"纵向重叠 > 50%"才算断片。
        arcs.sort(key=lambda c: -c["w"])
        kept_arcs = []
        for c in arcs:
            cx_c = c["x"] + c["w"] / 2.0
            near = [k for k in kept_arcs
                    if abs(cx_c - (k["x"] + k["w"] / 2.0)) <= 0.6 * H]
            if near:
                # 与已有弧同心 → 可能是**同一条弧的断片**（丢），也可能是**上下叠着的
                # 两条真弧**（留）。两条判据都要满足才留：
                #   ① 纵向不重叠（断片是同一笔画的两截，纵向几乎重合）；
                #   ② 两条都是**强弧形**（拱起 ≥ ARC_PAIR_MIN_HUMP）。
                frag = any(_y_overlap(c, k) > 0.5 for k in near)
                strong = _hump(c, binary) >= ARC_PAIR_MIN_HUMP and all(
                    _hump(k, binary) >= ARC_PAIR_MIN_HUMP for k in near)
                if frag or not strong:
                    continue
                if VERBOSE_DEBUG:
                    # 只有"本该被当碎片丢掉、靠这两条判据救回来"的才打 —— 便于核实
                    # 这条判据到底在哪几张谱、哪些位置上起了作用。
                    print(f"  [slur+] 行{row_index + 1} x={c['x']:.0f} "
                          f"宽={c['w'] / H:.2f}H 拱起={_hump(c, binary):.2f} 与已有弧同心"
                          f"（纵向最多重叠 {max(_y_overlap(c, k) for k in near):.0%}）"
                          f"→ 保留（上下叠着的两条弧）")
            kept_arcs.append(c)
        slur_arcs = 0
        slur_pairs = 0
        slur_halves = 0
        used_spans = set()     # 已经画过弧的「音区间」，用于丢弃同一条弧的碎片
        for c in kept_arcs:
            x_lo = c["x"] - 0.25 * H
            x_hi = c["x"] + c["w"] + 0.25 * H
            span = [i for i, nd in enumerate(notes) if x_lo <= nd["cx"] <= x_hi]
            if len(span) < 2:
                # 只搭上一个音 → **通常**不是连音弧；但**跨行的弧**在每行只留半截，
                # 完全可能只搭上一个音。实测《上海滩》行12 `x=150 宽=0.89H 拱起=0.84`：
                # **形状判据全过、就卡在这一步** → 整条跨行弧在行首"漏画"（用户报的）。
                # 判据：这一截必须**贴在行的首/末边缘**（块中心落在第一个音的左边，
                # 或最后一个音的右边）才算残段；其余仍按"不是弧"丢掉。
                if not (span and notes):
                    continue
                c_mid = c["x"] + c["w"] / 2.0
                hump_txt = f"{_hump(c, binary):.2f}"
                if span[0] == 0 and c_mid <= notes[0]["cx"]:
                    notes[0]["slur_left"] = True      # 从行首伸过来 → 画到行左边
                    slur_halves += 1
                    if VERBOSE_DEBUG:
                        print(f"  [slur] 行{row_index + 1} x={c['x']:.0f} "
                              f"宽={c['w'] / H:.2f}H 拱起={hump_txt} → 跨行残段（贴行首）")
                elif span[-1] == len(notes) - 1 and c_mid >= notes[-1]["cx"]:
                    notes[-1]["slur_right"] = True    # 伸向行末 → 画到行右边
                    slur_halves += 1
                    if VERBOSE_DEBUG:
                        print(f"  [slur] 行{row_index + 1} x={c['x']:.0f} "
                              f"宽={c['w'] / H:.2f}H 拱起={hump_txt} → 跨行残段（贴行末）")
                continue
            # ⚠ **同一段音上只留一条弧**：细弧在二值化后常断成几截，几个连通块会
            # 都映射到**同一段音**上 → 画出**两条重合的弧**（用户 2026-09-18 报的
            # "一个字符上面有 2 个弧"）。同一条真弧不该在同一段音上出现两次，
            # 所以按「音区间」去重，只留先到的那条。
            pair_span = (span[0], span[-1])
            if pair_span in used_spans:
                if VERBOSE_DEBUG:
                    print(f"  [slur] x={c['x']:.0f} 宽={c['w'] / H:.1f}H "
                          f"→ 与已有弧覆盖同一段音（音{span[0] + 1}..音{span[-1] + 1}）"
                          f"，判为同一条弧的碎片，丢弃")
                continue
            used_spans.add(pair_span)
            slur_arcs += 1
            # 拆成**相邻对**：模型里连音只存相邻两音（slur_start/slur_stop 同编号），
            # 这样 letterscore 的 `⌒`、审核 JSON、to_text() 全都不用改；
            # 画图时再由 jianpu_render._slur_chains() 把相邻对拼回一条长弧。
            #
            # ⚠ **每条弧必须给一个独立编号**（原来一律写死 `[1]`）：
            # 写死的话，"两条相邻弧（A-B 与 B-C）"和"一条长弧（A-B-C）"在模型里
            # **完全无法区分** —— `_slur_chains()` 只看相邻两音有没有共同编号，
            # 于是两条独立的弧会被串成一条长的（用户 2026-09-18 报的"连在一起"）。
            # 实测病例：《屁》里 `x=1494 宽2.5H（音9..音10）` 与
            # `x=1618 宽6.7H（音10..音11）` 是两条独立弧，共用音10。
            slur_id = slur_arcs
            for i in range(span[0], span[-1]):
                # ⚠ 必须 **append、不能覆盖**：**嵌套弧**（长弧 A-B-C 上面又套一条
                # 短弧 B-C）会在同一个音上留下**两个编号**。实测《上海滩》行3 的
                # `x=991 宽3.3H（音7..音9）` 与 `x=1058 宽1.9H（音8..音9）`。
                # 覆盖的话，音8 的 slur_start 只剩短弧的编号，长弧的第二对（音8,音9）
                # 就配不上号 → 画出来是"两条各跨 2 个音的弧"，而不是
                # "一条跨 3 个音 + 一条跨 2 个音（嵌套）"（用户 2026-09-20 报的）。
                notes[i]["slur_start"].append(slur_id)
                notes[i + 1]["slur_stop"].append(slur_id)
                slur_pairs += 1
            if VERBOSE_DEBUG:
                print(f"  [slur] x={c['x']:.0f} 宽={c['w'] / H:.1f}H "
                      f"高={c['h'] / H:.2f}H 拱起={_hump(c, binary):.2f} "
                      f"→ 弧#{slur_id} 连音 {len(span)} 个音"
                      f"（音{span[0] + 1}..音{span[-1] + 1}）")
        if slur_arcs or slur_halves:
            # ⚠ 必须带行号：否则两次运行的摘要只能按"出现顺序"对，
            # 而两次的行数/行分组可能不同 → 对比就是无效的（我 2026-09-18 就这么误判过）。
            print(f"[ocr] 行{row_index + 1} 连音弧：{slur_arcs} 条"
                  f"（拆成 {slur_pairs} 对相邻音）"
                  + (f" + 跨行残段 {slur_halves} 截" if slur_halves else ""))

        # ---- 重音记号 `>`（判据见 ACCENT_* 常量，全部来自实测）----
        accents = 0
        for c in all_components:
            # 位置条件**最先**判——理由同连音弧那边：否则与这一行无关的块
            # 会在每一行都被扫到并打印，日志里的行号就是假的。
            if c["y"] + c["h"] > slur_digits_top + 0.30 * H:
                continue                            # 必须落在数字上方
            if (slur_digits_top - (c["y"] + c["h"])) / H > UPPER_MARK_MAX_LIFT:
                continue                            # 太高（实测 +1.13H 以上是上一行的数字）
            if not (ACCENT_MIN_W * H <= c["w"] <= ACCENT_MAX_W * H):
                continue
            if c["h"] > ACCENT_MAX_H * H:
                continue                            # 太高的不是 `>`（是掉进窗口的数字）
            fill = c["area"] / max(c["w"] * c["h"], 1)
            if fill > ACCENT_MAX_FILL:
                continue                            # 音区点（填充 0.83）在此被排掉
            hump = _hump(c, binary)
            if hump > ACCENT_MAX_HUMP:
                continue                            # 连音弧 / `v`（0.80+）在此被排掉
            # 挂到它**下方最近的音**（重音记号画在音符正上方）
            cx_c = c["x"] + c["w"] / 2.0
            near = [nd for nd in notes if abs(nd["cx"] - cx_c) <= 0.9 * H]
            if not near:
                continue
            target = min(near, key=lambda nd: abs(nd["cx"] - cx_c))
            target["accent"] = True
            accents += 1
            if VERBOSE_DEBUG:
                lift = (slur_digits_top - (c["y"] + c["h"])) / H
                print(f"  [accent] 行{row_index + 1} x={c['x']:.0f} "
                      f"距顶={lift:+.2f}H 宽={c['w'] / H:.2f}H 高={c['h'] / H:.2f}H "
                      f"填充={fill:.2f} 拱起={hump:.2f} → 挂在最近的音上")
        if accents:
            print(f"[ocr] 重音记号（>）：{accents} 个")

        # ---- 上方小记号：`v` 换气 / `³` 三连音 / 房子标号 ----
        # 判据与常量见 MARK_*（⚠ 阈值暂定：每个记号只有 1 个实测样本）。
        # 分辨思路：先按尺寸圈出"小记号"，再用 `拱起` 与 `底部横杠` 两个**有物理意义**的
        # 量交叉；`³` 与房子标号 `3.` 同字形，只能靠"旁边有没有房子括线"分。
        mark_counts = {"breath": 0, "triplet": 0, "house": 0}
        for c in all_components:
            if c["y"] + c["h"] > slur_digits_top + 0.30 * H:
                continue
            if (slur_digits_top - (c["y"] + c["h"])) / H > UPPER_MARK_MAX_LIFT:
                continue                            # 太高（实测 +1.13H 以上是上一行的数字）
            if not (MARK_MIN_W * H <= c["w"] <= MARK_MAX_W * H):
                continue
            if not (MARK_MIN_H * H <= c["h"] <= MARK_MAX_H * H):
                continue
            fill = c["area"] / max(c["w"] * c["h"], 1)
            if fill > MARK_MAX_FILL:
                continue
            hump = _hump(c, binary)
            bar = _bottom_bar(c, binary)
            bracket = _find_wide_bracket(all_components, c, H)
            near_house = bracket is not None
            cx_c = c["x"] + c["w"] / 2.0
            near = [nd for nd in notes if abs(nd["cx"] - cx_c) <= 0.9 * H]
            if not near:
                continue
            target = min(near, key=lambda nd: abs(nd["cx"] - cx_c))
            # 分支顺序有讲究：
            #   ① `v`：拱起高 + 底部是尖。**必须最先判**，否则会被"房子标号"抢走
            #      （两者拱起都高，0.80 vs 0.90，光看拱起分不开）。
            #   ② `³`：拱起低（0.21）+ 旁边没有房子括线。
            #   ③ 房子标号：旁边有房子括线（`3.` 那种低拱起的也能落到这里）。
            if hump >= MARK_V_MIN_HUMP and bar <= MARK_V_MAX_BAR:
                target["breath"] = True
                kind = "v 换气"
                mark_counts["breath"] += 1
            elif hump <= MARK_TRIPLET_MAX_HUMP and not near_house:
                target["triplet"] = True
                kind = "³ 三连音"
                mark_counts["triplet"] += 1
            else:
                # 房子标号**不在这里处理**了 —— 已改成"先独立认括线、再用它定位标号"
                #（见下面那段）。这里只剩 `v` / `³` 两类；`near_house` 仍用于把
                # "房子标号"从 `³` 的候选里排除（两者同字形）。
                # 其余分不清的一律不标 —— **宁缺勿假**（标错了比缺了更难发现）
                if VERBOSE_DEBUG:
                    lift = (slur_digits_top - (c["y"] + c["h"])) / H
                    print(f"  [mark?] 行{row_index + 1} x={c['x']:.0f} "
                          f"距顶={lift:+.2f}H 宽={c['w'] / H:.2f}H 高={c['h'] / H:.2f}H "
                          f"填充={fill:.2f} 拱起={hump:.2f} 底杠={bar:.2f} "
                          f"房线={near_house} → 分不清，跳过")
                continue
            if VERBOSE_DEBUG:
                lift = (slur_digits_top - (c["y"] + c["h"])) / H
                print(f"  [mark] 行{row_index + 1} x={c['x']:.0f} "
                      f"距顶={lift:+.2f}H 宽={c['w'] / H:.2f}H 高={c['h'] / H:.2f}H "
                      f"填充={fill:.2f} 拱起={hump:.2f} 底杠={bar:.2f} "
                      f"房线={near_house} → {kind}")
        # ---- 房子横线 + 标号：**先独立认括线，再用它定位标号** ----
        # 为什么反过来做（用户 2026-09-18 选定）：括线又长又独特（实测 17~44H 宽、
        # 填充仅 0.09~0.17），比小字号数字好认得多；先认标号的话，标号一漏检
        # 整条横线就没了。
        house_lines = 0
        bracket_list = []       # [(括线块, 覆盖的音列表)]，标号配对时用
        for c in all_components:
            if c["w"] < HOUSE_BRACKET_MIN_W * H:
                continue
            if c["y"] + c["h"] > slur_digits_top + 0.30 * H:
                continue
            if (slur_digits_top - (c["y"] + c["h"])) / H > HOUSE_BRACKET_MAX_LIFT:
                continue                            # 太高（窗口见常量处的实测依据）
            if not (HOUSE_BRACKET_MIN_H * H <= c["h"] <= HOUSE_BRACKET_MAX_H * H):
                # 高度**下限**是关键：长连音弧只有 0.47~0.65H，房子括线是 1.09~1.13H
                #（两端有向下的竖钩撑着）。原来只设了上限 → 长弧全被当成房子括线。
                continue
            fill = c["area"] / max(c["w"] * c["h"], 1)
            if fill > HOUSE_BRACKET_MAX_FILL:
                continue                            # 太实（括线实测填充仅 0.08~0.09）
            span = [nd for nd in notes
                    if c["x"] - 0.30 * H <= nd["cx"] <= c["x"] + c["w"] + 0.30 * H]
            if len(span) < 2:
                continue                            # 只覆盖一个音 → 不是房子段
            house_lines += 1
            if VERBOSE_DEBUG:
                # **每个被当成房子的块都打**（含没找到标号的）——只打"找到标号"的，
                # 就看不出"误把长连音弧当括线"这类问题（用户 2026-09-18 正是这样发现的）。
                # `距顶` 与 `y` 也一起打：多出来的那两条高度/填充/宽度都和真括线一模一样
                #（1.09~1.13H、0.08、47H），只能靠"它到底在哪个高度"来分辨是
                # "同一行被相邻行重复认到"还是"另有其物"。
                lift = (slur_digits_top - (c["y"] + c["h"])) / H
                print(f"  [hbr] 行{row_index + 1} x={c['x']:.0f} y={c['y']} "
                      f"距顶={lift:+.2f}H 宽={c['w'] / H:.1f}H 高={c['h'] / H:.2f}H "
                      f"填充={fill:.2f} → 覆盖 {len(span)} 个音")
            # 端点存进模型，渲染时按这两个音的位置连线（同连音弧的"只存端点"思路）
            span[0]["house_start"] = True
            span[-1]["house_stop"] = True
            # 标号**不在这里找** —— 见下面那段"全局配对"。
            # 逐条横线各自去找会**抓错**：两条横线的标号挨得近时（上一段的 `1.`
            # 与本段的 `2.`），实测《上海滩》行 3 就抓成了 `2.`，而原谱写的是 `1.`
            #（用户 2026-09-18 确认）。
            bracket_list.append((c, span))

        # ---- 标号归属：**全局配对**（每个标号只归一条横线，且归给左端离它最近的那条）----
        pairs = []
        for bi, (c, _span) in enumerate(bracket_list):
            lx = c["x"] + 0.5 * H          # 横线左端：标号就贴在这里
            ly = c["y"] + c["h"] / 2.0
            for m in all_components:
                # ⚠ 宽度下限放到 **0.10H**：标号 `1` 很窄，用 `MARK_MIN_W(0.28H)` 会把它挡掉，
                # 于是"只剩一个 `2` 可选"——这正是行 3 抓错的机制之一。
                if not (0.10 * H <= m["w"] <= MARK_MAX_W * H):
                    continue
                if m["h"] > MARK_MAX_H * H:
                    continue
                m_cx = m["x"] + m["w"] / 2.0
                if abs(m_cx - lx) > 2.0 * H:
                    continue
                if abs((m["y"] + m["h"] / 2.0) - ly) > 0.9 * H:
                    continue
                pairs.append((abs(m_cx - lx), bi, m))

        pairs.sort(key=lambda t: t[0])     # 由近及远贪心：最近的先配上
        used_brackets, used_blocks = set(), set()
        for dist, bi, m in pairs:
            if bi in used_brackets or id(m) in used_blocks:
                continue                   # 每条横线只要一个标号；每个标号只归一条横线
            used_brackets.add(bi)
            used_blocks.add(id(m))
            c, span = bracket_list[bi]
            # 标号只可能是 `1.`/`2.`/`3.`：把候选集缩到这三类再比，不用绝对相似度阈值。
            all_refs = _all_reference_glyphs()
            house_refs = {d: all_refs[d] for d in (1, 2, 3) if d in all_refs}
            num, score = _classify_by_reference(m, binary, house_refs,
                                                min_score=0.0, min_margin=0.0,
                                                return_score=True)
            # **`1` / `2` 定向复核**：这个字体把小字号 `1` 写得整体轮廓像 `2`，
            # 形状相似度会认错（用户 2026-09-18 确认《上海滩》行 3 是 `1.`，
            # 却被判成 `2.`；行 4 的 `2.` 判对了）。用**主笔画的倾斜度**纠正：
            # `1` 的主笔竖直、`2` 是一条斜线（见 `_stroke_drift()`）。
            drift = _stroke_drift(m, binary)
            fixed = ""
            if num == 2 and drift < HOUSE_LABEL_ONE_MAX_DRIFT:
                num, fixed = 1, "（倾斜度偏低 → 由 2 改判 1）"
            elif num == 1 and drift >= HOUSE_LABEL_ONE_MAX_DRIFT:
                num, fixed = 2, "（倾斜度偏高 → 由 1 改判 2）"
            span[0]["house"] = str(num) if num in (1, 2, 3) else "?"
            mark_counts["house"] += 1
            if VERBOSE_DEBUG:
                print(f"  [house] 行{row_index + 1} 括线 x={c['x']:.0f} "
                      f"宽={c['w'] / H:.1f}H → 覆盖 {len(span)} 个音；标号 x={m['x']:.0f} "
                      f"宽={m['w'] / H:.2f}H 距左端={dist / H:.2f}H "
                      f"相似度={score:.2f} 倾斜={drift:.2f} → {span[0]['house']}. {fixed}")
                # 同一横线附近的**其它候选**也打出来：这样才能看出"是不是把 `1` 筛掉、
                # 只剩 `2` 可选"，或者"两个候选都在、贪心配错了"。
                rivals = [(d, mm) for d, b2, mm in pairs if b2 == bi and mm is not m]
                if rivals:
                    txt = "；".join(f"x={mm['x']:.0f} 宽={mm['w'] / H:.2f}H "
                                    f"距左端={d / H:.2f}H" for d, mm in rivals[:4])
                    print(f"         -> 同一横线还有别的候选（未采用）：{txt}")
                # 再把标号的**实际形状**打出来（每格 2×2 像素）：实测"判对的相似度 0.48、
                # 判错的 0.51"——**分数分不开对错**，只能靠形状本身判断。
                sub = binary[m["y"]:m["y"] + m["h"], m["x"]:m["x"] + m["w"]] > 0
                for line in _ascii_mask(sub, step=2):
                    print(f"         |{line}|")

        if any(mark_counts.values()) or house_lines:
            print(f"[ocr] 上方记号：换气 {mark_counts['breath']} 个 / "
                  f"三连音 {mark_counts['triplet']} 个 / "
                  f"房子横线 {house_lines} 条（标号 {mark_counts['house']} 个）")

        total_blocks += len(notes)
        total_labeled += labeled
        record["notes"] = notes
        # 「整行大部分认不出」判据（2026-09-24 加，治 39.png《侧脸》行8）：
        # 歌词字与音符**同号**时，上面那一串判据（text_count / shape_ok / h_ok /
        # oversize）**全部失效**——它们都建立在"歌词字比音符大/更清楚"之上。
        # 只有"认不出的块数"不依赖那个前提：实测全 39 张 **438 条被接受的行**里，
        # 真旋律行**最多只有 3 个**认不出；而"歌词被当成数字"的那一行是 **6 个（60%）**。
        # ⚠ 必须「数量 ≥4」**且**「占比 ≥30%」两条同时成立：只看占比会误伤短行
        #   （39.png 行4 是 1/4 = 25%，那是真旋律行）。详见 识别依据.md §2.1。
        none_cnt = sum(1 for nt in notes if nt.get("n") is None)
        if none_cnt >= 4 and none_cnt >= 0.30 * len(notes):
            record["boundaries"] = []
            if debug:
                print(f"[ocr-debug] 跳过行{row_index}: {none_cnt}/{len(notes)} 认不出"
                      f"（疑似歌词被当成数字）")
            continue
        # 锚点已经是 OCR 旋律行；只丢弃极端失配行。
        if labeled < 3 or ratio < 0.30:
            record["boundaries"] = []
            if debug:
                print(f"[ocr-debug] 跳过行{row_index}: {labeled}/{len(notes)}")
            continue
        kept_rows += 1
        record["review_line"] = kept_rows

        # 小节划分：默认按检测到的小节竖线切；可选按拍数切。
        # 竖线若明显少于「应有小节数」（按总拍数估算），自动用宽松条件重找一遍。
        total_beats = sum(float(n.get("q", 1.0) or 1.0) for n in notes)
        expected_measures = max(1, int(round(total_beats / 4.0)))
        if len(vlines) < expected_measures - 1:
            relaxed_lines = _detect_vlines(row_components, blocks, H, digits_top,
                                           digits_bottom, relaxed=True)
            if len(relaxed_lines) > len(vlines):
                vlines = relaxed_lines
                record["vline_candidates"] = [
                    (c["x"] + c["w"] / 2.0, id(c) in {id(x) for x in vlines})
                    for c in row_components
                    if c["h"] >= 0.8 * H and c["w"] <= 0.5 * H
                ]

        boundaries = _dedupe_boundaries(vlines, tolerance=0.22 * H)
        if measure_mode == "beats":
            first_boundary = min(boundaries) if boundaries else None
            segments = _divide_by_beats(notes, 4.0, first_boundary)
            cuts = list(boundaries)
        else:
            if boundaries:
                segments = [[] for _ in range(len(boundaries) + 1)]
                for note in notes:
                    segments[sum(note["cx"] >= x for x in boundaries)].append(note)
            else:
                segments = [notes]
            cuts = list(boundaries)
        record["boundaries"] = cuts

        # 反复记号（只标记、不展开）：从本行 OCR 文本里的 ':' 判断归属小节。
        seg_repeat = {}
        repeat_marks = []
        for tk in tokens_in_row:
            mark = _repeat_mark(tk["text"])
            if not mark:
                continue
            kind, ch_idx = mark
            # 按冒号在文本中的字符位置换算 x（整块中心会偏，因 OCR 常把记号
            # 与后面的音符并成一块）
            span = tk["x1"] - tk["x0"]
            n_chars = max(len(tk["text"]), 1)
            repeat_marks.append((tk["x0"] + span * (ch_idx + 0.5) / n_chars, kind))
        # 几何兜底：OCR 常整行读不出 `:`（实测整行只读出一个 `3652`），
        # 用「双细竖线 + 旁边的点」补上。
        repeat_marks.extend(_detect_repeats_geom(row_components, H))
        repeat_marks.sort(key=lambda item: item[0])
        merged_marks = []
        for mx, kind in repeat_marks:
            if merged_marks and abs(mx - merged_marks[-1][0]) <= 0.6 * H:
                continue
            merged_marks.append((mx, kind))

        for mx, kind in merged_marks:
            # 归属方向必须按语义来（否则会差一个小节）：
            #   ‖: 表示「从下一个小节开始反复」→ 归到标记**右侧**第一个小节；
            #   :‖ 表示「到上一个小节结束反复」→ 归到标记**左侧**最后一个小节。
            target = None
            if kind == "forward":
                for seg in segments:
                    if not seg:
                        continue
                    first_x = min(n["cx"] for n in seg)
                    if first_x >= mx - 0.3 * H:
                        if target is None or first_x < min(n["cx"] for n in target):
                            target = seg
            else:
                for seg in segments:
                    if not seg:
                        continue
                    last_x = max(n["cx"] for n in seg)
                    if last_x <= mx + 0.3 * H:
                        if target is None or last_x > max(n["cx"] for n in target):
                            target = seg
            if target is None:
                continue
            before, after_kind = seg_repeat.get(id(target), (False, None))
            if kind == "forward":
                before = True
            elif kind == "double":
                after_kind = "double"      # 终止/双小节线
            else:
                after_kind = "backward"
            seg_repeat[id(target)] = (before, after_kind)

        for segment_index, segment in enumerate(segments):
            if segment:
                before, after_kind = seg_repeat.get(id(segment), (False, None))
                # 该小节结束小节线的**原图 x**（来自识别到的竖线）：
                # 渲染时直接用它，就不再需要我"推算中点"（推算会让线偏右、
                # 看起来小节线后面空一大块）。
                bar_x = (boundaries[segment_index]
                         if segment_index < len(boundaries) else None)
                measures.append({
                    "number": f"{row_index + 1}-{segment_index + 1}",
                    "row": row_index,
                    "bar_x": bar_x,
                    "repeat_before": "forward" if before else None,
                    "repeat_after": after_kind or "single",
                    "notes": segment,
                })

    if not measures:
        raise ValueError("已识别到旋律行，但数字匹配率过低，未生成结果。")

    print(f"[ocr] {kept_rows} 行，数字 {total_labeled}/{total_blocks} 已识别，"
          f"其余以 ? 标记。")
    # 数字分布：便于和原谱对照（例如原谱只有 2 个 0，这里若显示十几个就是误判）
    dist = defaultdict(int)
    for m in measures:
        for note in m.get("notes", []):
            if note.get("n") is not None:
                dist[note["n"]] += 1
    print("[ocr] 数字分布：" + " ".join(f"{d}:{dist[d]}" for d in sorted(dist)))

    if debug:
        _make_debug_overlay(color, row_records, os.path.join(debug_dir, "ocr_overlay.png"))
        cv2.imwrite(os.path.join(debug_dir, "binary.png"), binary)
        with open(os.path.join(debug_dir, "ocr_tokens.txt"), "w", encoding="utf-8") as f:
            for row in anchored_rows:
                f.write(f"y={row['cy']:.1f}\t" + " | ".join(t["text"] for t in row["tokens"]) + "\n")

    # 供 train_digits.py 采集样本：暴露本次的「每行数字块」与二值图
    global LAST_ROW_INFO, _LAST_BINARY
    LAST_ROW_INFO = [{"row_index": i, "H": rec["H"], "blocks": rec["blocks"]}
                     for i, rec in enumerate(row_records)]
    _LAST_BINARY = binary

    return measures
