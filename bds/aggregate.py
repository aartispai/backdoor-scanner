"""Stage 4 - evidence aggregation: calibrated logistic regression (or rule-based fallback) + evaluation."""
import numpy as np, joblib
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, roc_curve

FEATS = ["nc_anomaly", "nc_asr_gap", "nc_l1_ratio", "skew", "abs_anom", "strip_gap", "act_auc", "act_cluster"]
GROUPS = {"behavioural": ["skew", "strip_gap"], "activation": ["abs_anom", "act_auc", "act_cluster"],
          "trigger_search": ["nc_anomaly", "nc_asr_gap", "nc_l1_ratio"]}
vec = lambda f: [float(f[c]) for c in FEATS]
_lr = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.5, class_weight="balanced", max_iter=1000))

def heuristic(f):
    c = lambda v, lo, hi: float(np.clip((v - lo) / (hi - lo), 0, 1))
    return (.35 * c(f["nc_anomaly"], 1, 4) * c(f["nc_asr_gap"], .1, .6) + .2 * c(f["strip_gap"], .1, .5) + .15 * c(f["skew"], .3, .8)
            + .15 * c(f["abs_anom"], 2, 8) + .15 * c(f["act_auc"], .8, .99) * c(f["act_cluster"], .7, .95))

class Aggregator:
    def fit(s, rows, y): s.p = _lr().fit([vec(r) for r in rows], y); return s
    def prob(s, f): return float(s.p.predict_proba([vec(f)])[0, 1])
    def save(s, path): joblib.dump(s, path)

def tpr_at_fpr(y, p, target=0.05):
    """Detection rate while flagging at most `target` of clean models, plus the score threshold that achieves it."""
    fpr, tpr, thr = roc_curve(y, p); i = np.where(fpr <= target)[0][-1]; t = float(thr[i])
    return float(tpr[i]), (t if np.isfinite(t) else None)

def _cv(X, y, reps=5):
    """Cross-validated AUC averaged over `reps` shuffles (small zoos are noisy) + averaged out-of-fold probabilities."""
    k = min(5, int(np.bincount(y).min())); ps = [cross_val_predict(_lr(), X, y, cv=StratifiedKFold(k, shuffle=True, random_state=r),
                                                                  method="predict_proba")[:, 1] for r in range(reps)]
    return float(np.mean([roc_auc_score(y, q) for q in ps])), np.mean(ps, 0)

def leave_attack_out(X, y, att):
    """Train on clean + all backdoors EXCEPT attack A; test on A (and held-out clean models). Tests trigger-agnostic detection."""
    clean = np.where(y == 0)[0]; folds = np.array_split(np.random.RandomState(0).permutation(clean), min(4, len(clean))); out = {}
    for A in sorted(set(att[y == 1])):
        a_idx, rest = np.where(att == A)[0], np.where((y == 1) & (att != A))[0]
        if len(rest) == 0: continue
        pc, pa = np.zeros(len(X)), np.zeros(len(a_idx))
        for f in folds:
            m = _lr().fit(X[np.r_[np.setdiff1d(clean, f), rest]], y[np.r_[np.setdiff1d(clean, f), rest]])
            pc[f] = m.predict_proba(X[f])[:, 1]; pa += m.predict_proba(X[a_idx])[:, 1] / len(folds)
        s, lab = np.r_[pc[clean], pa], np.r_[np.zeros(len(clean)), np.ones(len(a_idx))]
        out[f"held_out_{A}"] = dict(n_models=len(a_idx), auc=float(roc_auc_score(lab, s)), tpr_at_fpr0=tpr_at_fpr(lab, s, 0.0)[0],
                                    detected_at_0_5=float((pa >= .5).mean()))
    return out

def evaluate(rows, y, attacks=None):
    X, y = np.array([vec(r) for r in rows], float), np.array(y); auc, p = _cv(X, y)
    out = {"ensemble_auc_cv": auc, "FPR@0.5 (clean flagged)": float(((p >= .5) & (y == 0)).sum() / (y == 0).sum()), "TPR@0.5": float(((p >= .5) & (y == 1)).sum() / (y == 1).sum())}
    for tgt in (0.0, 0.05):
        out[f"TPR@FPR<={tgt:.0%}"], out[f"score_threshold@FPR<={tgt:.0%}"] = tpr_at_fpr(y, p, tgt)
    out["ablate_stage_group_auc"] = {g: _cv(X[:, [i for i, c in enumerate(FEATS) if c not in cols]], y)[0] for g, cols in GROUPS.items()}
    out["ablate_single_feature_auc"] = {c: _cv(np.delete(X, i, 1), y)[0] for i, c in enumerate(FEATS)}
    out["single_feature_raw_auc"] = {c: float(roc_auc_score(y, X[:, i])) for i, c in enumerate(FEATS)}
    out["baseline_neural_cleanse_only_auc"] = float(roc_auc_score(y, X[:, FEATS.index("nc_anomaly")]))
    out["baseline_strip_only_auc"] = float(roc_auc_score(y, X[:, FEATS.index("strip_gap")]))
    if attacks is not None and len(set(np.array(attacks)[y == 1])) > 1: out["leave_one_attack_out"] = leave_attack_out(X, y, np.array(attacks))
    return out
