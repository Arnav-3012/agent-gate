"""Export a 3D map of the corpus for the dashboard Galaxy view: dashboard/public/data/galaxy.json.

Reads vectors back from Qdrant (no re-embedding), UMAP-projects them to 3D, writes id/title/source/xyz.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient

from engine.rag.retriever import COLLECTION

OUT = Path("dashboard/public/data/galaxy.json")


def project(vectors: np.ndarray, seed: int = 0, n_neighbors: int = 15) -> np.ndarray:
    """UMAP to 3D, rescaled to [-1, 1] per axis so the view needs no camera tuning."""
    import umap

    xyz = umap.UMAP(n_components=3, n_neighbors=n_neighbors, metric="cosine", random_state=seed).fit_transform(vectors)
    lo, hi = xyz.min(0), xyz.max(0)
    return 2 * (xyz - lo) / np.maximum(hi - lo, 1e-9) - 1


def to_payload(chunk_ids: list[str], titles: list[str], sources: list[str], xyz: np.ndarray) -> dict:
    return {"points": [
        {"id": c, "title": t, "source": s, "x": round(float(p[0]), 4), "y": round(float(p[1]), 4), "z": round(float(p[2]), 4)}
        for c, t, s, p in zip(chunk_ids, titles, sources, xyz, strict=True)
    ]}


def fetch(client: QdrantClient, collection: str, limit: int) -> tuple[list[str], list[str], list[str], np.ndarray]:
    ids, titles, sources, vecs = [], [], [], []
    offset = None
    while len(ids) < limit:
        points, offset = client.scroll(collection, limit=min(256, limit - len(ids)), offset=offset, with_payload=True, with_vectors=True)
        for p in points:
            ids.append(p.payload["chunk_id"]); titles.append(p.payload["title"]); sources.append(p.payload["source"])
            vecs.append(p.vector)
        if offset is None:
            break
    return ids, titles, sources, np.asarray(vecs, dtype=np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", default=COLLECTION)
    ap.add_argument("--limit", type=int, default=5000, help="points drawn from the collection")
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    client = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))
    ids, titles, sources, vecs = fetch(client, a.collection, a.limit)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(to_payload(ids, titles, sources, project(vecs))))
    print(f"wrote {len(ids)} points -> {a.out}")


if __name__ == "__main__":
    main()
