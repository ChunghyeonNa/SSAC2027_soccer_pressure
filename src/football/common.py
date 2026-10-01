"""Shared geometry and evaluation helpers for the shot-decision notebooks."""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

GOAL_X, POST_Y = 52.5, 3.66
KEY = ["game_id", "pe_id"]


def in_triangle(pts, a, b, c):
    def s(p1, p2, p3):
        return (p1[:, 0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[:, 1] - p3[1])
    d1, d2, d3 = s(pts, a, b), s(pts, b, c), s(pts, c, a)
    neg = (d1 < 0) | (d2 < 0) | (d3 < 0)
    pos = (d1 > 0) | (d2 > 0) | (d3 > 0)
    return ~(neg & pos)


def geom(p, defs, gk):
    """Shot geometry at point p. defs: (n,2) outfield defenders, gk: (2,) or None."""
    post1, post2 = np.array([GOAL_X, -POST_Y]), np.array([GOAL_X, POST_Y])
    v1, v2 = post1 - p, post2 - p
    angle = abs(np.arctan2(v1[0] * v2[1] - v1[1] * v2[0], v1 @ v2))
    dist = np.hypot(GOAL_X - p[0], p[1])
    n_cone = int(in_triangle(defs, p, post1, post2).sum()) if len(defs) else 0
    near_def = np.hypot(*(defs - p).T).min() if len(defs) else 30.0
    gk_dist = np.hypot(*(gk - p)) if gk is not None else 30.0
    return dict(dist=dist, angle=angle, n_cone=n_cone, near_def=near_def, gk_dist=gk_dist)


def seg_dist(pts, a, b):
    """Distance of each point to segment a-b."""
    ab = b - a
    t = np.clip(((pts - a) @ ab) / max(ab @ ab, 1e-9), 0, 1)
    proj = a + t[:, None] * ab
    return np.hypot(*(pts - proj).T)


XG_COLS = ["dist", "angle", "n_cone", "near_def", "gk_dist", "header"]


def xg_X(df):
    """Design matrix for the xG model (geometry + header + log distance)."""
    X = df.reindex(columns=XG_COLS).astype(float).fillna(0.0)
    X["log_dist"] = np.log1p(X.dist)
    return X


def cv_predict(make_model, X, y, groups, seeds=(0, 1, 2, 3, 4), proba=True):
    """Out-of-fold predictions, game-grouped 5-fold CV, averaged over shuffled group splits."""
    pred = np.zeros(len(y))
    for seed in seeds:
        for tr, te in GroupKFold(5, shuffle=True, random_state=seed).split(X, groups=groups):
            m = make_model(seed)
            m.fit(X.iloc[tr], y.iloc[tr])
            out = m.predict_proba(X.iloc[te])[:, 1] if proba else m.predict(X.iloc[te])
            pred[te] += out / len(seeds)
    return pred


def _resampler(n_rows, groups):
    """Return a function rng -> row indices of one bootstrap resample (rows iid, or whole games)."""
    if groups is None:
        return lambda rng: rng.randint(0, n_rows, n_rows)
    codes, uniq = pd.factorize(np.asarray(groups))
    rows_by = [np.flatnonzero(codes == j) for j in range(len(uniq))]
    return lambda rng: np.concatenate([rows_by[j] for j in rng.randint(0, len(uniq), len(uniq))])


def boot_ci(y, p, fn, n=1000, seed=0, groups=None):
    """95% percentile bootstrap CI of metric fn(y, p); groups= resamples whole games (cluster bootstrap)."""
    y = np.asarray(y)
    p = np.asarray(p)
    rng = np.random.RandomState(seed)
    draw = _resampler(len(y), groups)
    v = []
    for _ in range(n):
        i = draw(rng)
        if len(np.unique(y[i])) == 2:
            v.append(fn(y[i], p[i]))
    return np.percentile(v, [2.5, 97.5])


def boot_diff(y, p1, p2, fn, n=1000, seed=0, groups=None):
    """Paired bootstrap of fn(y,p1) - fn(y,p2): mean, 95% CI, share of resamples <= 0.

    groups= resamples whole games (cluster bootstrap), appropriate when passes are clustered in games.
    """
    y, p1, p2 = map(np.asarray, (y, p1, p2))
    rng = np.random.RandomState(seed)
    draw = _resampler(len(y), groups)
    d = []
    for _ in range(n):
        i = draw(rng)
        if len(np.unique(y[i])) == 2:
            d.append(fn(y[i], p1[i]) - fn(y[i], p2[i]))
    d = np.array(d)
    return d.mean(), np.percentile(d, [2.5, 97.5]), (d <= 0).mean()


if __name__ == "__main__":
    g = geom(np.array([GOAL_X - 11, 0.0]), np.empty((0, 2)), None)
    assert abs(g["angle"] - 2 * np.arctan(3.66 / 11)) < 1e-9 and abs(g["dist"] - 11) < 1e-9
    assert abs(seg_dist(np.array([[0.0, 1.0]]), np.array([-1.0, 0]), np.array([1.0, 0]))[0] - 1) < 1e-9
    assert in_triangle(np.array([[0.2, 0.2], [2.0, 2.0]]), np.array([0, 0]), np.array([1, 0]), np.array([0, 1])).tolist() == [True, False]
    print("ok")
