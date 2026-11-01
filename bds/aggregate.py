"""Stage 4 - evidence aggregation: calibrated logistic regression (or rule-based fallback) + evaluation."""
import numpy as np, joblib
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

FEATS = ["nc_anomaly", "nc_asr", "nc_l1_ratio", "skew", "abs_anom", "strip_gap", "act_auc", "act_cluster"]
GROUPS = {"behavioural": ["skew", "strip_gap"], "activation": ["abs_anom", "act_auc", "act_cluster"],
          "trigger_search": ["nc_anomaly", "nc_asr", "nc_l1_ratio"]}
vec = lambda f: [float(f[c]) for c in FEATS]
_lr = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.5, class_weight="balanced", max_iter=1000))

def heuristic(f):
    c = lambda v, lo, hi: float(np.clip((v - lo) / (hi - lo), 0, 1))
    return (.35 * c(f["nc_anomaly"], 1, 4) * c(f["nc_asr"], .5, .95) + .2 * c(f["strip_gap"], .1, .5) + .15 * c(f["skew"], .3, .8)
            + .15 * c(f["abs_anom"], 2, 8) + .15 * c(f["act_auc"], .8, .99) * c(f["act_cluster"], .7, .95))

class Aggregator:
    def fit(s, rows, y): s.p = _lr().fit([vec(r) for r in rows], y); return s
    def prob(s, f): return float(s.p.predict_proba([vec(f)])[0, 1])
    def save(s, path): joblib.dump(s, path)

def _cv(X, y):
    k = min(5, int(np.bincount(y).min())); p = cross_val_predict(_lr(), X, y, cv=StratifiedKFold(k, shuffle=True, random_state=0), method="predict_proba")[:, 1]
    return roc_auc_score(y, p), p

def evaluate(rows, y):
    X, y = np.array([vec(r) for r in rows]), np.array(y); auc, p = _cv(X, y)
    out = {"ensemble_auc": auc, "FPR@0.5 (clean flagged)": float(((p >= .5) & (y == 0)).sum() / (y == 0).sum()),
           "TPR@0.5": float(((p >= .5) & (y == 1)).sum() / (y == 1).sum())}
    for g, cols in GROUPS.items():
        out[f"ablation_without_{g}_auc"] = _cv(X[:, [i for i, c in enumerate(FEATS) if c not in cols]], y)[0]
    out["baseline_neural_cleanse_only_auc"] = roc_auc_score(y, X[:, FEATS.index("nc_anomaly")])
    out["baseline_strip_only_auc"] = roc_auc_score(y, X[:, FEATS.index("strip_gap")])
    return out
