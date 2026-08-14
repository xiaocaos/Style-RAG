"""评估数据集：服装客服测试问题。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EvalCase:
    """单条评估用例。"""
    question: str
    ground_truth: str
    category: str


EVAL_DATASET: list[EvalCase] = [
    # ── 尺码推荐 ──────────────────────────────────────────────
    EvalCase(
        question="我身高175cm体重140斤，建议穿什么尺码？",
        ground_truth="建议尺码XL，对应身高170-178cm，体重130-150斤",
        category="尺码推荐",
    ),
    EvalCase(
        question="180cm 150斤的男生穿多大的？",
        ground_truth="建议尺码XL，对应身高170-178cm，体重130-150斤",
        category="尺码推荐",
    ),
    EvalCase(
        question="我165cm 100斤 女生 选什么码",
        ground_truth="建议尺码M，对应身高160-170cm，体重90-115斤",
        category="尺码推荐",
    ),
    EvalCase(
        question="胖一点的 185cm 180斤 能穿吗",
        ground_truth="建议尺码3XL，对应身高178-185cm，体重160-180斤",
        category="尺码推荐",
    ),
    EvalCase(
        question="小个子155cm 80斤穿啥",
        ground_truth="建议尺码S，对应身高155-165cm，体重75-95斤",
        category="尺码推荐",
    ),

    # ── 洗涤养护 ──────────────────────────────────────────────
    EvalCase(
        question="真丝连衣裙怎么洗？",
        ground_truth="建议干洗；手洗用真丝专用中性洗涤剂，水温≤25℃，浸泡≤15分钟，轻轻按压清洗；禁止搓揉、拧绞、漂白",
        category="洗涤养护",
    ),
    EvalCase(
        question="羊毛衫能机洗吗",
        ground_truth="优先干洗；手洗用羊毛专用洗涤剂，水温≤20℃，浸泡≤15分钟，轻轻按压；禁止机洗、搓揉、拧绞",
        category="洗涤养护",
    ),
    EvalCase(
        question="羽绒服怎么清洗比较好",
        ground_truth="优先干洗；可水洗款用羽绒服专用洗涤剂，水温≤30℃，浸泡≤20分钟，轻轻按压；机洗选羽绒服专用模式",
        category="洗涤养护",
    ),
    EvalCase(
        question="牛仔裤多久洗一次",
        ground_truth="薄牛仔材质避免频繁清洗，1-2周一次即可；水温≤30℃，翻面清洗减少褪色",
        category="洗涤养护",
    ),
    EvalCase(
        question="纯棉T恤怎么洗不褪色",
        ground_truth="浅色与深色分开洗，首次洗加少许盐固色；机洗用洗衣袋+轻柔模式；阴凉通风处阴干，避免暴晒褪色",
        category="洗涤养护",
    ),

    # ── 颜色选择 ──────────────────────────────────────────────
    EvalCase(
        question="黄皮肤适合什么颜色的衣服",
        ground_truth="优先选暖色调（如焦糖色、姜黄色、豆沙色），避免冷调荧光色（如荧光绿、冷粉），易显肤色暗沉；浅米色、燕麦色可柔和肤色",
        category="颜色选择",
    ),
    EvalCase(
        question="面试穿什么颜色的衣服",
        ground_truth="首选深色系（黑色、藏蓝、深灰），稳重专业；避免大面积亮色和花哨图案，保持简洁得体",
        category="颜色选择",
    ),
    EvalCase(
        question="微胖的人怎么穿显瘦",
        ground_truth="优先选深色系（黑色、深灰、藏蓝），视觉上收缩身形；避免大面积亮色和横条纹，易显臃肿",
        category="颜色选择",
    ),
    EvalCase(
        question="夏天穿什么颜色凉快",
        ground_truth="适合冷色调和浅色系（如白色、天蓝色、浅绿色），清爽降温；避免深色系，吸热且显沉闷",
        category="颜色选择",
    ),
    EvalCase(
        question="参加婚礼穿什么颜色",
        ground_truth="可选择亮色（正红、酒红、宝蓝）或金属色（金色、银色），凸显气质；避免过于朴素的颜色",
        category="颜色选择",
    ),
]
