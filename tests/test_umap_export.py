import numpy as np
import pytest

pytest.importorskip("umap")
from scripts.umap_export import project, to_payload


def test_project_is_3d_in_unit_cube_and_payload_aligned():
    rng = np.random.default_rng(0)
    vecs = np.vstack([rng.normal(0, 1, (30, 16)), rng.normal(6, 1, (30, 16))]).astype(np.float32)
    xyz = project(vecs, n_neighbors=5)
    assert xyz.shape == (60, 3) and xyz.min() >= -1 - 1e-6 and xyz.max() <= 1 + 1e-6
    out = to_payload([str(i) for i in range(60)], ["t"] * 60, ["s"] * 60, xyz)
    assert len(out["points"]) == 60 and set(out["points"][0]) == {"id", "title", "source", "x", "y", "z"}
