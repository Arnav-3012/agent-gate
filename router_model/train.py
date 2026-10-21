"""Router classifier: features -> cheapest-sufficient arm (labels from eval/build_router_labels.py).
Logistic regression with balanced class weights, standardised inputs, probability calibration when
each class has enough rows. Exports weights.json so the Lambda/engine predicts with numpy only."""
import json
from pathlib import Path

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def train(X: np.ndarray, y: list[str], calibrate_min: int = 15) -> dict:
    scaler = StandardScaler().fit(X)
    Z = scaler.transform(X)
    base = LogisticRegression(class_weight="balanced", max_iter=1000)
    counts = np.unique(y, return_counts=True)[1]
    if len(counts) > 1 and counts.min() >= calibrate_min:
        cal = CalibratedClassifierCV(base, cv=3).fit(Z, y)
        # average the per-fold linear models: a calibrated linear head that still exports as weights
        coef = np.mean([c.estimator.coef_ for c in cal.calibrated_classifiers_], axis=0)
        icpt = np.mean([c.estimator.intercept_ for c in cal.calibrated_classifiers_], axis=0)
        classes = cal.classes_
    else:
        base.fit(Z, y)
        coef, icpt, classes = base.coef_, base.intercept_, base.classes_
    return {"classes": [str(c) for c in classes], "coef": coef.tolist(), "intercept": icpt.tolist(),
            "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()}


def predict_proba(w: dict, x: np.ndarray) -> dict[str, float]:
    z = (np.asarray(x, float) - w["mean"]) / w["scale"]
    logit = np.asarray(w["coef"]) @ z + np.asarray(w["intercept"])
    if len(w["classes"]) == 2:  # sklearn stores one row for binary problems
        p1 = 1 / (1 + np.exp(-logit[0]))
        p = np.array([1 - p1, p1])
    else:
        e = np.exp(logit - logit.max())
        p = e / e.sum()
    return dict(zip(w["classes"], map(float, p), strict=True))


def route(w: dict, x: np.ndarray) -> str:
    p = predict_proba(w, x)
    return max(p, key=p.__getitem__)


def save(w: dict, path: Path = Path("router_model/weights.json")) -> None:
    path.write_text(json.dumps(w, indent=1))


def load(path: Path = Path("router_model/weights.json")) -> dict:
    return json.loads(path.read_text())
