"""服装领域知识图谱：面料、季节、洗护方式的关系网络。"""

from __future__ import annotations

from typing import Any

from src.logging_config import logger

try:
    import networkx as nx
except ImportError:
    nx = None  # type: ignore


# ── 预定义的服装知识图谱 ───────────────────────────────────────
_CLOTHING_KG_DATA: list[dict[str, Any]] = [
    # 面料 -> 季节
    {"head": "纯棉", "relation": "belongs_to_season", "tail": "春季"},
    {"head": "薄牛仔", "relation": "belongs_to_season", "tail": "春季"},
    {"head": "针织棉", "relation": "belongs_to_season", "tail": "春季"},
    {"head": "轻薄化纤", "relation": "belongs_to_season", "tail": "春季"},
    {"head": "真丝", "relation": "belongs_to_season", "tail": "夏季"},
    {"head": "棉麻", "relation": "belongs_to_season", "tail": "夏季"},
    {"head": "冰丝", "relation": "belongs_to_season", "tail": "夏季"},
    {"head": "雪纺", "relation": "belongs_to_season", "tail": "夏季"},
    {"head": "羊毛", "relation": "belongs_to_season", "tail": "秋季"},
    {"head": "羊绒", "relation": "belongs_to_season", "tail": "秋季"},
    {"head": "厚牛仔", "relation": "belongs_to_season", "tail": "秋季"},
    {"head": "灯芯绒", "relation": "belongs_to_season", "tail": "秋季"},
    {"head": "麂皮绒", "relation": "belongs_to_season", "tail": "秋季"},
    {"head": "羽绒", "relation": "belongs_to_season", "tail": "冬季"},
    {"head": "毛呢", "relation": "belongs_to_season", "tail": "冬季"},
    {"head": "加绒牛仔", "relation": "belongs_to_season", "tail": "冬季"},

    # 面料 -> 洗护方式
    {"head": "纯棉", "relation": "wash_method", "tail": "可机洗，水温≤30℃，中性洗涤剂"},
    {"head": "真丝", "relation": "wash_method", "tail": "建议干洗；手洗用真丝专用洗涤剂，水温≤25℃"},
    {"head": "羊毛", "relation": "wash_method", "tail": "优先干洗；手洗水温≤20℃，禁止搓揉"},
    {"head": "羊绒", "relation": "wash_method", "tail": "优先干洗；手洗水温≤20℃，禁止机洗"},
    {"head": "羽绒", "relation": "wash_method", "tail": "优先干洗；可水洗款用羽绒服专用洗涤剂"},
    {"head": "棉麻", "relation": "wash_method", "tail": "可机洗，水温≤30℃，中性洗涤剂"},
    {"head": "灯芯绒", "relation": "wash_method", "tail": "手洗或机洗，水温≤30℃，禁止用力搓揉"},
    {"head": "雪纺", "relation": "wash_method", "tail": "手洗优先，水温≤30℃，禁止拧绞"},
    {"head": "冰丝", "relation": "wash_method", "tail": "手洗或机洗，水温≤30℃，禁止长时间浸泡"},
    {"head": "针织棉", "relation": "wash_method", "tail": "手洗优先，水温≤25℃，禁止搓揉拧绞"},
    {"head": "薄牛仔", "relation": "wash_method", "tail": "水温≤30℃，翻面清洗减少褪色"},
    {"head": "厚牛仔", "relation": "wash_method", "tail": "水温≤30℃，翻面清洗，首次盐水浸泡固色"},
    {"head": "加绒牛仔", "relation": "wash_method", "tail": "水温≤30℃，翻面清洗，避免长时间浸泡"},
    {"head": "毛呢", "relation": "wash_method", "tail": "必须干洗，禁止水洗"},
    {"head": "麂皮绒", "relation": "wash_method", "tail": "建议干洗；人造麂皮可手洗，水温≤30℃"},
    {"head": "轻薄化纤", "relation": "wash_method", "tail": "可机洗，水温30-40℃，加柔顺剂减少静电"},

    # 面料 -> 注意事项
    {"head": "真丝", "relation": "caution", "tail": "禁止搓揉、拧绞、漂白；低温蒸汽熨烫"},
    {"head": "羊毛", "relation": "caution", "tail": "禁止机洗、搓揉、拧绞；平铺阴干防拉伸"},
    {"head": "羊绒", "relation": "caution", "tail": "禁止机洗；放防虫蛀剂；透气布袋包裹"},
    {"head": "羽绒", "relation": "caution", "tail": "通风阴干，轻轻拍打恢复蓬松；勿过度压缩"},
    {"head": "灯芯绒", "relation": "caution", "tail": "禁止用力搓揉；熨斗垫薄布，顺绒方向熨烫"},

    # 尺码 -> 身高体重范围
    {"head": "S码", "relation": "size_range", "tail": "身高155-165cm，体重75-95斤"},
    {"head": "M码", "relation": "size_range", "tail": "身高160-170cm，体重90-115斤"},
    {"head": "L码", "relation": "size_range", "tail": "身高165-175cm，体重115-135斤"},
    {"head": "XL码", "relation": "size_range", "tail": "身高170-178cm，体重130-150斤"},
    {"head": "2XL码", "relation": "size_range", "tail": "身高175-182cm，体重145-165斤"},
    {"head": "3XL码", "relation": "size_range", "tail": "身高178-185cm，体重160-180斤"},
    {"head": "4XL码", "relation": "size_range", "tail": "身高180-190cm，体重180-210斤"},
    {"head": "5XL码", "relation": "size_range", "tail": "身高190cm+，体重210斤+"},
]


class ClothingKnowledgeGraph:
    """服装领域知识图谱。"""

    def __init__(self) -> None:
        if nx is None:
            logger.warning("networkx 未安装，知识图谱功能不可用")
            self._graph = None
            return
        self._graph = nx.DiGraph()
        self._build_graph()

    def _build_graph(self) -> None:
        for item in _CLOTHING_KG_DATA:
            self._graph.add_edge(
                item["head"],
                item["tail"],
                relation=item["relation"],
            )
        logger.info("知识图谱加载完成: %d 节点, %d 边",
                     self._graph.number_of_nodes(), self._graph.number_of_edges())

    def query(self, entity: str, max_hops: int = 2) -> dict[str, list[str]]:
        """查询实体的关联信息。"""
        if self._graph is None:
            return {}
        results: dict[str, list[str]] = {}
        if entity not in self._graph:
            # 模糊匹配
            for node in self._graph.nodes():
                if entity in node or node in entity:
                    results[node] = self._get_neighbors(node, max_hops)
        else:
            results[entity] = self._get_neighbors(entity, max_hops)
        return results

    def _get_neighbors(self, node: str, max_hops: int) -> list[str]:
        """获取节点的邻居信息。"""
        info: list[str] = []
        visited: set[str] = set()

        def _dfs(n: str, depth: int) -> None:
            if depth > max_hops or n in visited:
                return
            visited.add(n)
            for _, neighbor, data in self._graph.edges(n, data=True):
                relation = data.get("relation", "related_to")
                info.append(f"{n} --{relation}--> {neighbor}")
                if depth < max_hops:
                    _dfs(neighbor, depth + 1)

        _dfs(node, 0)
        return info

    def get_context_for_query(self, question: str) -> str:
        """根据问题自动提取相关图谱知识，作为额外 context。"""
        if self._graph is None:
            return ""

        keywords = ["面料", "材质", "洗涤", "洗护", "养护", "尺码", "推荐",
                     "棉", "丝", "毛", "绒", "牛仔", "雪纺", "化纤"]
        relevant_info: list[str] = []
        for kw in keywords:
            if kw in question:
                for node in self._graph.nodes():
                    if kw in node:
                        neighbors = self._get_neighbors(node, max_hops=1)
                        relevant_info.extend(neighbors)

        if not relevant_info:
            return ""

        unique_info = list(dict.fromkeys(relevant_info))  # 去重保序
        return "【知识图谱补充信息】\n" + "\n".join(unique_info)
