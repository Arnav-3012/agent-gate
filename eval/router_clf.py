"""Query-feature router test: can the QUERY ALONE (bge embedding + type + length) predict when the
strong arm beats the cheap one? Label = strong - cheap > eps (build_router_labels' tie-break margin).

Held-out end to end: per outer fold the classifier is fit on the other folds, the escalation
threshold is fitted on inner out-of-fold probabilities of those same training rows, and the test
fold is scored with neither seen. Reported against RANDOM escalation at the same rate (cascade_cv).

    python -m eval.router_clf eval/runs/baseline_general*.json --domain general \\
        --cheap default/rag --strong Qwen2.5-14B-Instruct-4bit/rag
"""
import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict

from eval.benchmark import load_files, score, with_headline_split
from eval.cascade_cv import fit_threshold, summarize


def features(rows: list[dict], questions: dict[str, str]) -> np.ndarray:
    from engine.rag.retriever import Embedder  # heavy import: only when run
    emb = Embedder()
    texts = [questions[r["id"]] for r in rows]
    types = sorted({r["type"] for r in rows})
    x_emb = np.array([emb.query(t) for t in texts])
    x_type = np.array([[r["type"] == t for t in types] for r in rows], dtype=float)
    x_len = np.array([[len(t.split()) / 30] for t in texts])
    return np.hstack([x_emb, x_type, x_len])


def run(x: np.ndarray, sc: np.ndarray, ss: np.ndarray, cc: np.ndarray, cs: np.ndarray,
        eps: float, k: int = 5, seed: int = 0, c: float = 0.1) -> dict:
    n, y = len(sc), (ss - sc) > eps
    gain = ss - sc
    folds = np.random.default_rng(seed).permutation(n) % k
    sig, esc = np.zeros(n), np.zeros(n, dtype=bool)  # sig = -P(strong helps): low -> escalate, as cascade_cv
    for f in range(k):
        tr, te = folds != f, folds == f
        clf = LogisticRegression(C=c, class_weight="balanced", max_iter=1000)
        inner = -cross_val_predict(clf, x[tr], y[tr], cv=5, method="predict_proba")[:, 1]
        t = fit_threshold(inner, gain[tr], cs[tr])
        sig[te] = -clf.fit(x[tr], y[tr]).predict_proba(x[te])[:, 1]
        esc[te] = sig[te] < t
    r = summarize(sig, sc, ss, cc, cs, esc, seed)
    r["label_rate"] = float(y.mean())
    r["clf_auc"] = float(roc_auc_score(y, -sig)) if 0 < y.sum() < n else float("nan")
    return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--domain", default="general")
    ap.add_argument("--cheap", required=True)
    ap.add_argument("--strong", required=True)
    ap.add_argument("--eps", type=float, default=yaml.safe_load(Path("config/router.yaml").read_text())["epsilon_f1"])
    a = ap.parse_args()
    arms = with_headline_split(load_files(a.files))[a.domain]
    ids = sorted(set(arms[a.cheap]) & set(arms[a.strong]))
    cheap, strong = [arms[a.cheap][i] for i in ids], [arms[a.strong][i] for i in ids]
    qpath = Path(f"eval/data/{'finance_questions' if a.domain.startswith('finance') else 'questions'}.jsonl")
    questions = {q["id"]: q["question"] for q in map(json.loads, qpath.read_text().splitlines())}
    r = run(features(cheap, questions), np.array([score(x) for x in cheap]), np.array([score(x) for x in strong]),
            np.array([x["cost"] for x in cheap]), np.array([x["cost"] for x in strong]), a.eps)
    print(f"{a.domain}: {a.cheap} -> {a.strong}, query-feature LR, n={r['n']}, eps={a.eps}, "
          f"strong-helps rate {r['label_rate']:.0%}, 5-fold held-out")
    print(f"  score  cheap {r['score']['cheap']:.3f}  strong {r['score']['strong']:.3f}  cascade {r['score']['cascade']:.3f}")
    print(f"  $/1k   cheap {r['usd_per_1k']['cheap']:.4f}  strong {r['usd_per_1k']['strong']:.4f}  cascade {r['usd_per_1k']['cascade']:.4f}")
    print(f"  escalation rate {r['escalation_rate']:.0%}  clf AUC {r['clf_auc']:.2f}")
    for name in ("cascade_minus_cheap", "cascade_minus_random_same_rate"):
        m, lo, hi = r[name]
        print(f"  {name:<32} {m:+.3f} [{lo:+.3f}, {hi:+.3f}] {'REAL' if lo > 0 else 'within noise'}")


if __name__ == "__main__":
    main()
