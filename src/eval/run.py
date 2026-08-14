"""评估运行器：对比基础 RAG vs 高级 RAG 效果。"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from src.config import settings
from src.eval.dataset import EVAL_DATASET, EvalCase
from src.knowledge.base import KnowledgeBase
from src.logging_config import logger


def _simple_rag_answer(question: str, kb: KnowledgeBase) -> str:
    """基础 RAG：直接检索 top-k + LLM 生成（无 Rewriting/Reranking）。"""
    from langchain_community.chat_models.tongyi import ChatTongyi
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    docs = kb.get_retriever(k=3).invoke(question)
    context = "\n".join(d.page_content for d in docs)

    prompt = ChatPromptTemplate.from_template(
        "根据以下资料回答问题。\n\n资料：{context}\n\n问题：{question}"
    )
    llm = ChatTongyi(model=settings.chat_model, temperature=0.1)
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"context": context, "question": question})


def _advanced_rag_answer(question: str, kb: KnowledgeBase) -> str:
    """高级 RAG：完整流水线。"""
    from src.rag.advanced_chain import AdvancedRAGChain
    chain = AdvancedRAGChain(knowledge_base=kb)
    return chain.invoke(question)


def run_evaluation() -> dict[str, Any]:
    """运行评估，返回结果字典。"""
    kb = KnowledgeBase()
    results: list[dict] = []

    logger.info("=" * 60)
    logger.info("开始评估，共 %d 条测试用例", len(EVAL_DATASET))
    logger.info("=" * 60)

    for i, case in enumerate(EVAL_DATASET, 1):
        logger.info("[%d/%d] %s", i, len(EVAL_DATASET), case.question[:30])

        # 基础 RAG
        t0 = time.time()
        basic_answer = _simple_rag_answer(case.question, kb)
        basic_time = time.time() - t0

        # 高级 RAG
        t0 = time.time()
        adv_answer = _advanced_rag_answer(case.question, kb)
        adv_time = time.time() - t0

        results.append({
            "question": case.question,
            "category": case.category,
            "ground_truth": case.ground_truth,
            "basic_answer": basic_answer,
            "basic_time": round(basic_time, 2),
            "advanced_answer": adv_answer,
            "advanced_time": round(adv_time, 2),
        })

    # 保存结果
    output_path = Path(settings.chroma_persist_dir).parent / "eval_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info("评估完成，结果保存至: %s", output_path)
    return {"total": len(results), "results": results, "output_path": str(output_path)}


if __name__ == "__main__":
    run_evaluation()
