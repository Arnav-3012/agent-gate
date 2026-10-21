"""Dense retrieval over Qdrant: bge-small embeddings, one chunk = one source paragraph."""
import hashlib
import os
import uuid

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
    ) -> None:
        self.embedder = embedder or Embedder()
        self.client = client or QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))
        self.top_k = top_k

    def search(self, query: str, top_k: int | None = None) -> list[Hit]:
        res = self.client.query_points(
            COLLECTION,
            query=self.embedder.query(query),
            limit=top_k or self.top_k,
            with_payload=True,
        )
        return [Hit(chunk=Chunk(**p.payload), score=p.score) for p in res.points]
