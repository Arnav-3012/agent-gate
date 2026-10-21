"""Dense retrieval over Qdrant: bge-small embeddings, one chunk = one source paragraph."""
import hashlib
import os
import uuid
from pathlib import Path

import yaml
from pydantic import BaseModel
from qdrant_client import QdrantClient

COLLECTION = "passages"
# bge-v1.5 retrieval convention: the instruction goes on queries only, never on passages.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Chunk(BaseModel):
    chunk_id: str
    title: str
    text: str
    source: str  # hotpotqa | squad

    @property
    def embed_text(self) -> str:
        return f"{self.title}\n{self.text}"


class Hit(BaseModel):
    chunk: Chunk
    score: float  # cosine similarity


def make_chunk(title: str, text: str, source: str) -> Chunk:
    """Deterministic id from content, so re-ingest and dedup are idempotent."""
    cid = hashlib.sha1(f"{source}\x00{title}\x00{text}".encode()).hexdigest()[:16]
    return Chunk(chunk_id=cid, title=title, text=text, source=source)


def point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_OID, chunk_id))


class Embedder:
    def __init__(self, model: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model or os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5"))
        self.dim = self.model.get_embedding_dimension()

    def passages(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        return self.model.encode(
            texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True
        ).tolist()

    def query(self, text: str) -> list[float]:
        return self.model.encode(QUERY_PREFIX + text, normalize_embeddings=True).tolist()


class Retriever:
    def __init__(
        self,
        embedder: Embedder | None = None,
        client: QdrantClient | None = None,
        top_k: int = 5,
        collection: str = COLLECTION,
        metadata_filter: bool = False,
        pool: int = 200,
    ) -> None:
        self.embedder = embedder or Embedder()
        self.client = client or QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))
        self.top_k = top_k
        self.collection = collection
        self.metadata_filter = metadata_filter  # filings corpus only (D43); see engine/rag/scope.py
        self.pool = pool
        self._companies: list[str] | None = None

    @classmethod
    def from_config(cls, finance: bool = False, path: Path = Path("config/engine.yaml"),
                    **kw: object) -> "Retriever":
        """The one retrieval config every arm shares (G4): top_k from engine.yaml; the company/year
        scope only on the filings collection, where titles carry them."""
        raw = yaml.safe_load(path.read_text())
        return cls(top_k=raw["top_k"], collection="finance" if finance else COLLECTION,
                   metadata_filter=finance and raw.get("metadata_filter", False), **kw)

    def _company_names(self) -> list[str]:
        if self._companies is None:  # one scroll over the (small) filings collection
            from engine.rag.scope import companies_in

            titles, offset = [], None
            while True:
                points, offset = self.client.scroll(self.collection, limit=500, offset=offset,
                                                    with_payload=["title"])
                titles += [p.payload["title"] for p in points]
                if offset is None:
                    break
            self._companies = companies_in(titles)
        return self._companies

    def search(self, query: str, top_k: int | None = None) -> list[Hit]:
        k = top_k or self.top_k
        res = self.client.query_points(
            self.collection,
            query=self.embedder.query(query),
            limit=max(k, self.pool) if self.metadata_filter else k,
            with_payload=True,
        )
        hits = [Hit(chunk=Chunk(**p.payload), score=p.score) for p in res.points]
        if not self.metadata_filter:
            return hits
        from engine.rag.scope import apply_scope, parse_scope

        return apply_scope(hits, parse_scope(query, self._company_names()), k)
