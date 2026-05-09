import hashlib
import json
from typing import List, Optional

import chromadb
from chromadb.config import Settings


class ResearchMemory:
    def __init__(self, persist_dir: str = "./chroma_db"):
        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name="research_history",
            metadata={"hnsw:space": "cosine"},
        )

    def store(self, query: str, final_answer: str, claims: List[dict]) -> str:
        doc_id = hashlib.md5(query.encode()).hexdigest()
        self._collection.upsert(
            ids=[doc_id],
            documents=[final_answer],
            metadatas=[{
                "query": query,
                "claims_json": json.dumps(claims),
            }],
        )
        return doc_id

    def retrieve_similar(self, query: str, n_results: int = 3) -> List[dict]:
        if self._collection.count() == 0:
            return []
        results = self._collection.query(
            query_texts=[query],
            n_results=min(n_results, self._collection.count()),
        )
        if not results["documents"] or not results["documents"][0]:
            return []
        return [
            {
                "query": meta["query"],
                "answer": doc,
                "claims": json.loads(meta.get("claims_json", "[]")),
                "distance": dist,
            }
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ]

    def has_recent_answer(self, query: str, similarity_threshold: float = 0.15) -> Optional[dict]:
        results = self.retrieve_similar(query, n_results=1)
        if results and results[0]["distance"] <= similarity_threshold:
            return results[0]
        return None


memory = ResearchMemory()
