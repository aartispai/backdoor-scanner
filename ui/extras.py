"""Data for the trigger-stamping 3D view. Read-only: it only calls the model, and never changes how a scan is run or scored."""
import torch
from bds.models import predict


def compute(model, X, res, n_img=4, seed=0):
    """A few real reference images with the model's prediction before and after the reconstructed trigger is stamped on."""
    t = res["target"]; m, p = res["trigger"]["masks"][t], res["trigger"]["pats"][t]
    trig = lambda x: (1 - m) * x + m * p
    pred = predict(model, X).argmax(1); non = (pred != t).nonzero().squeeze(1)
    if len(non) < n_img: return None
    pick = non[torch.randperm(len(non), generator=torch.Generator().manual_seed(seed))].tolist()
    chosen, seen = [], set()                                  # prefer images from different original classes
    for i in pick:
        if int(pred[i]) not in seen: chosen.append(i); seen.add(int(pred[i]))
        if len(chosen) == n_img: break
    chosen += [i for i in pick if i not in chosen][: n_img - len(chosen)]
    xs = X[chosen]; p0 = predict(model, xs).softmax(1); p1 = predict(model, trig(xs)).softmax(1)
    images = [dict(px=[int(round(float(v) * 255)) for v in xs[k].flatten()], cp=int(p0[k].argmax()), cpp=round(float(p0[k].max()), 3),
                   tp=int(p1[k].argmax()), tpp=round(float(p1[k].max()), 3)) for k in range(len(chosen))]
    flip = float((predict(model, trig(X[non])).argmax(1) == t).float().mean())
    return dict(stamp=dict(images=images, flip_rate=flip))
