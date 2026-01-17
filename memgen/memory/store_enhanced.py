"""
FAISS 向量检索增强版 Experience Store

支持:
1. 简单的 Jaccard 相似度检索（无需额外依赖）
2. FAISS 向量检索（需要 faiss-cpu 和 sentence-transformers）
3. 混合检索（结合关键词和语义相似度）
"""

import json
import os
from typing import List, Dict, Optional, Tuple
import numpy as np


class ExperienceStore:
    """
    经验库存储和检索

    支持多种检索模式:
    - simple: Jaccard 相似度（默认，无需额外依赖）
    - faiss: FAISS 向量检索
    - hybrid: 混合检索
    """

    def __init__(
        self,
        store_path: str,
        index_type: str = "simple",
        topk: int = 4,
        min_score: float = 0.25,
        embedding_model: Optional[str] = None
    ):
        """
        Args:
            store_path: JSONL 文件路径
            index_type: 检索类型 (simple/faiss/hybrid)
            topk: 返回 top-k 结果
            min_score: 最小相似度阈值
            embedding_model: 嵌入模型名称（用于 faiss/hybrid）
        """
        self.store_path = store_path
        self.index_type = index_type
        self.topk = topk
        self.min_score = min_score
        self.entries: List[Dict] = []

        # Load entries if file exists
        if os.path.exists(store_path):
            self.load()

        # Initialize index based on type
        if index_type in ["faiss", "hybrid"]:
            self._init_faiss_index(embedding_model)
        else:
            self.faiss_index = None
            self.embedding_model = None

    def _init_faiss_index(self, embedding_model: Optional[str] = None):
        """初始化 FAISS 索引"""
        try:
            import faiss
            from sentence_transformers import SentenceTransformer
        except ImportError:
            print("Warning: faiss-cpu or sentence-transformers not installed.")
            print("Falling back to simple Jaccard similarity.")
            print("To use FAISS, install: pip install faiss-cpu sentence-transformers")
            self.index_type = "simple"
            self.faiss_index = None
            self.embedding_model = None
            return

        # Load embedding model
        model_name = embedding_model or "sentence-transformers/all-MiniLM-L6-v2"
        print(f"Loading embedding model: {model_name}")
        self.embedding_model = SentenceTransformer(model_name)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()

        # Create FAISS index
        self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product (cosine similarity)

        # Build index if entries exist
        if self.entries:
            self._rebuild_faiss_index()

    def _rebuild_faiss_index(self):
        """重建 FAISS 索引"""
        if self.faiss_index is None or self.embedding_model is None:
            return

        # Extract queries
        queries = [entry['query'] for entry in self.entries]

        # Encode queries
        print(f"Encoding {len(queries)} queries...")
        embeddings = self.embedding_model.encode(
            queries,
            convert_to_numpy=True,
            normalize_embeddings=True,  # Normalize for cosine similarity
            show_progress_bar=True
        )

        # Add to FAISS index
        self.faiss_index.reset()
        self.faiss_index.add(embeddings.astype('float32'))
        print(f"✓ FAISS index built: {self.faiss_index.ntotal} vectors")

    def load(self):
        """从 JSONL 文件加载经验"""
        self.entries = []
        with open(self.store_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    self.entries.append(json.loads(line))

    def save(self):
        """保存经验到 JSONL 文件"""
        os.makedirs(os.path.dirname(self.store_path) or '.', exist_ok=True)
        with open(self.store_path, 'w', encoding='utf-8') as f:
            for entry in self.entries:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    def add(self, entry: Dict):
        """添加经验条目"""
        self.entries.append(entry)

        # Update FAISS index if using FAISS
        if self.faiss_index is not None and self.embedding_model is not None:
            query_embedding = self.embedding_model.encode(
                [entry['query']],
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            self.faiss_index.add(query_embedding.astype('float32'))

    def search(self, query: str, topk: Optional[int] = None) -> List[Dict]:
        """
        检索相关经验

        Args:
            query: 查询文本
            topk: 返回 top-k 结果（默认使用初始化时的 topk）

        Returns:
            List of dicts with keys: text, score, entry
        """
        topk = topk or self.topk

        if self.index_type == "simple":
            return self._search_jaccard(query, topk)
        elif self.index_type == "faiss":
            return self._search_faiss(query, topk)
        elif self.index_type == "hybrid":
            return self._search_hybrid(query, topk)
        else:
            raise ValueError(f"Unknown index_type: {self.index_type}")

    def _search_jaccard(self, query: str, topk: int) -> List[Dict]:
        """Jaccard 相似度检索"""
        query_tokens = set(query.lower().split())

        scores = []
        for entry in self.entries:
            entry_tokens = set(entry['query'].lower().split())
            if not query_tokens or not entry_tokens:
                jaccard = 0.0
            else:
                jaccard = len(query_tokens & entry_tokens) / len(query_tokens | entry_tokens)

            scores.append({
                'text': entry.get('answer', entry.get('text', '')),
                'score': jaccard,
                'query': entry['query'],
                'entry': entry
            })

        # Sort and filter
        scores.sort(key=lambda x: x['score'], reverse=True)
        return [s for s in scores[:topk] if s['score'] >= self.min_score]

    def _search_faiss(self, query: str, topk: int) -> List[Dict]:
        """FAISS 向量检索"""
        if self.faiss_index is None or self.embedding_model is None:
            return self._search_jaccard(query, topk)

        if self.faiss_index.ntotal == 0:
            return []

        # Encode query
        query_embedding = self.embedding_model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        # Search
        scores, indices = self.faiss_index.search(query_embedding.astype('float32'), topk)

        # Build results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.entries):
                continue
            if score < self.min_score:
                continue

            entry = self.entries[idx]
            results.append({
                'text': entry.get('answer', entry.get('text', '')),
                'score': float(score),
                'query': entry['query'],
                'entry': entry
            })

        return results

    def _search_hybrid(self, query: str, topk: int, alpha: float = 0.5) -> List[Dict]:
        """
        混合检索（结合 Jaccard 和 FAISS）

        Args:
            query: 查询文本
            topk: 返回 top-k 结果
            alpha: Jaccard 权重（1-alpha 为 FAISS 权重）

        Returns:
            混合排序的结果
        """
        # Get results from both methods
        jaccard_results = self._search_jaccard(query, topk * 2)
        faiss_results = self._search_faiss(query, topk * 2)

        # Normalize scores to [0, 1]
        def normalize_scores(results):
            if not results:
                return results
            max_score = max(r['score'] for r in results)
            min_score = min(r['score'] for r in results)
            if max_score == min_score:
                return results
            for r in results:
                r['normalized_score'] = (r['score'] - min_score) / (max_score - min_score)
            return results

        jaccard_results = normalize_scores(jaccard_results)
        faiss_results = normalize_scores(faiss_results)

        # Merge results
        merged = {}
        for r in jaccard_results:
            query_key = r['query']
            merged[query_key] = {
                'text': r['text'],
                'query': r['query'],
                'entry': r['entry'],
                'jaccard_score': r.get('normalized_score', r['score']),
                'faiss_score': 0.0
            }

        for r in faiss_results:
            query_key = r['query']
            if query_key in merged:
                merged[query_key]['faiss_score'] = r.get('normalized_score', r['score'])
            else:
                merged[query_key] = {
                    'text': r['text'],
                    'query': r['query'],
                    'entry': r['entry'],
                    'jaccard_score': 0.0,
                    'faiss_score': r.get('normalized_score', r['score'])
                }

        # Compute hybrid score
        results = []
        for item in merged.values():
            hybrid_score = alpha * item['jaccard_score'] + (1 - alpha) * item['faiss_score']
            results.append({
                'text': item['text'],
                'score': hybrid_score,
                'query': item['query'],
                'entry': item['entry']
            })

        # Sort and filter
        results.sort(key=lambda x: x['score'], reverse=True)
        return [r for r in results[:topk] if r['score'] >= self.min_score]

    def __len__(self):
        return len(self.entries)


# Backward compatibility: keep old simple implementation
class SimpleExperienceStore(ExperienceStore):
    """简单版本（仅 Jaccard 检索）"""

    def __init__(self, store_path: str, topk: int = 4, min_score: float = 0.25):
        super().__init__(
            store_path=store_path,
            index_type="simple",
            topk=topk,
            min_score=min_score
        )
