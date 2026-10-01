"""Pressure-on-the-ball-carrier metrics from tracking (vectorised per event).

Coordinates: metres, team in possession attacks towards +x, goal at (GOAL_X, 0).
All functions take the carrier position/velocity and arrays of defender positions/velocities.
"""
import numpy as np

from football.common import GOAL_X

# PFF Event Data Specification v2.5, p. 6 states the pressure radii in yards (1 yd = 0.9144 m).
PFF_R_PRESSURED = 3 * 0.9144   # 2.74 m: "pressured" (P)
PFF_R_ATTEMPT = 5 * 0.9144     # 4.57 m: "attempted pressure" (A); also the common event-data radius
PFF_R_SPRINT = 4 * 0.9144      # 3.66 m: example of a sprinting defender still counted as P


def andrienko(pc, pd_, threats=None):
    """Andrienko et al. (2017) pressure per defender (Section 3.1, Eq. 1-2).

    Threat directions point from the carrier to each point in `threats` (default: the attacked goal).
    The paper uses the goal and attacking teammates as threat directions and keeps, per presser,
    the maximum over directions. Pressure zone length L grows from D_BACK (presser behind the
    carrier w.r.t. the threat direction) to D_FRONT (presser in front); pressure = (1 - d/L)^Q
    inside the zone, else 0. Note: the printed Eq. 1 has z = (1 - cos)/2, which contradicts the
    text (D_front at angle 0); z = (1 + cos)/2 follows the text.
    """
    D_FRONT, D_BACK, Q = 9.0, 3.0, 1.75
    if threats is None:
        threats = np.array([[GOAL_X, 0.0]])
    to_thr = threats - pc                                                  # (k, 2)
    to_def = pd_ - pc                                                      # (n, 2)
    d = np.hypot(*to_def.T)                                                # (n,)
    cos = (to_def @ to_thr.T) / np.maximum(d[:, None] * np.hypot(*to_thr.T)[None, :], 1e-9)   # (n, k)
    z = (1 + cos) / 2
    L = D_BACK + (D_FRONT - D_BACK) * (z ** 3 + 0.3 * z) / 1.3
    pr = np.where(d[:, None] < L, (1 - np.minimum(d[:, None] / L, 1)) ** Q, 0.0)
    return pr.max(axis=1)


def andrienko_threats(pc, mates, closer_only=True):
    """Threat points: the attacked goal plus attacking teammates (optionally only those closer to goal)."""
    goal = np.array([[GOAL_X, 0.0]])
    if mates is None or len(mates) == 0:
        return goal
    if closer_only:
        dist_goal = lambda p: np.hypot(GOAL_X - p[..., 0], p[..., 1])
        mates = mates[dist_goal(mates) < dist_goal(pc)]
    return np.vstack([goal, mates])


def pressing_intensity_total(p_int):
    """Bekkers (2025) Eq. 5: total pressure on a target, 1 - prod(1 - p_i) over pressers."""
    return 1 - np.prod(1 - np.asarray(p_int))


def time_to_intercept(pc, vc, pd_, vd, reaction=0.7, vmax=12.0):
    """Bekkers (2025) time to intercept (unravelsports implementation).

    Target extrapolated 1 s ahead (pc + vc); presser keeps its velocity for `reaction` s, then runs
    at vmax; plus a turning penalty |v_d|*theta/pi for moving away from the target.
    """
    target = pc + vc
    r = pd_ + vd * reaction
    to_t = target - pd_
    sp = np.hypot(*vd.T)
    cos = (vd * to_t).sum(1) / np.maximum(sp * np.hypot(*to_t.T), 1e-9)
    theta = np.where(sp > 1e-6, np.arccos(np.clip(cos, -1, 1)), 0.0)
    return reaction + np.hypot(*(target - r).T) / vmax + sp * theta / np.pi


def fs_arrival_time(pc, pd_, vd, alpha=1.0, vmax=10.0, t_max=10.0, dt=0.02, iters=20):
    """Narizuka et al. (2026) minimum-arrival time under the Fujimura-Sugihara motion model.

    Reachable set at time t: circle centred c(t) = p + v(1-e^{-at})/a with radius
    r(t) = vmax (t - (1-e^{-at})/a). Arrival time = smallest t with |x - c(t)| <= r(t).
    The condition is not monotone in t (a fast defender can pass over the point and leave it),
    so the first crossing is bracketed on a dt grid and then refined by bisection.
    """
    grid = np.arange(dt, t_max + dt, dt)                                   # (m,)
    gg = (1 - np.exp(-alpha * grid)) / alpha
    cc = pd_[:, None, :] + vd[:, None, :] * gg[None, :, None]              # (n, m, 2)
    reached0 = np.hypot(pc[0] - cc[..., 0], pc[1] - cc[..., 1]) <= vmax * (grid - gg)   # (n, m)
    first = np.where(reached0.any(1), reached0.argmax(1), len(grid) - 1)
    hi = grid[first]
    lo = np.where(first > 0, grid[np.maximum(first - 1, 0)], 0.0)
    for _ in range(iters):
        t = (lo + hi) / 2
        g = (1 - np.exp(-alpha * t)) / alpha
        c = pd_ + vd * g[:, None]
        reached = np.hypot(*(pc - c).T) <= vmax * (t - g)
        hi = np.where(reached, t, hi)
        lo = np.where(reached, lo, t)
    return hi


def intercept_prob(tti, threshold=1.5, sigma=0.45):
    """Logistic probability that the defender reaches the carrier within `threshold` s."""
    return 1 / (1 + np.exp(-np.pi / (np.sqrt(3) * sigma) * (threshold - tti)))


def closing_speed(pc, vc, pd_, vd):
    """Rate at which the defender-carrier distance shrinks (m/s, positive = closing)."""
    u = pc - pd_
    u = u / np.maximum(np.hypot(*u.T), 1e-9)[:, None]
    return ((vd - vc) * u).sum(1)


def defender_table(pc, vc, pd_, vd, mates=None):
    """Per-defender features used by the metrics and the learned presser model (literature defaults).

    mates: positions of the carrier's outfield teammates (for Andrienko threat directions).
    andr = goal + teammates closer to goal (main), andr_all = goal + all teammates, andr_goal = goal only.
    """
    d = np.hypot(*(pd_ - pc).T)
    to_goal = np.array([GOAL_X, 0.0]) - pc
    cos = ((pd_ - pc) @ to_goal) / np.maximum(d * np.hypot(*to_goal), 1e-9)
    tti = time_to_intercept(pc, vc, pd_, vd)
    return dict(dist=d, cos_goal=cos, closing=closing_speed(pc, vc, pd_, vd),
                speed=np.hypot(*vd.T), tti=tti, p_int=intercept_prob(tti),
                andr=andrienko(pc, pd_, andrienko_threats(pc, mates, closer_only=True)),
                andr_all=andrienko(pc, pd_, andrienko_threats(pc, mates, closer_only=False)),
                andr_goal=andrienko(pc, pd_), arrival=fs_arrival_time(pc, pd_, vd))


def pass_frames(out_dir, keys, fps=29.97, lag=6, max_gap=3.0):
    """Raw event-frame positions (players.csv.gz) + velocities from smoothed tracking (pass_frames_smoothed.csv.gz).

    keys: DataFrame with game_id, pe_id of the passes to keep. Returns one row per player per pass.
    """
    import os
    import pandas as pd
    k = ["game_id", "pe_id"]
    idx = keys.set_index(k).index
    pl = pd.read_csv(os.path.join(out_dir, "players.csv.gz"))
    raw = pl[pl.set_index(k).index.isin(idx)][k + ["player_id", "atk", "x", "y", "speed", "position"]]
    fr = pd.read_csv(os.path.join(out_dir, "pass_frames_smoothed.csv.gz"))
    fr = fr.dropna(subset=["player_id"]).drop_duplicates(k + ["offset_frames", "player_id"])
    s0 = fr[fr.offset_frames == 0][k + ["player_id", "x", "y"]].rename(columns={"x": "sx0", "y": "sy0"})
    s6 = fr[fr.offset_frames == lag][k + ["player_id", "x", "y"]].rename(columns={"x": "sx6", "y": "sy6"})
    vel = s0.merge(s6, on=k + ["player_id"])
    vel["vx"], vel["vy"] = (vel.sx0 - vel.sx6) * fps / lag, (vel.sy0 - vel.sy6) * fps / lag
    out = raw.merge(vel, on=k + ["player_id"], how="left")
    # velocity is trusted only if both smoothed frames exist and the smoothed track matches the raw position
    gap = np.hypot(out.x - out.sx0, out.y - out.sy0)
    out["vel_ok"] = out.vx.notna() & (gap <= max_gap)
    out.loc[~out.vel_ok, ["vx", "vy"]] = 0.0
    return out.drop(columns=["sx0", "sy0", "sx6", "sy6"])


if __name__ == "__main__":
    pc, vc = np.array([0.0, 0.0]), np.zeros(2)
    # defender 1 m goal-side standing, defender 1 m behind standing, defender 20 m away
    pd_ = np.array([[1.0, 0.0], [-1.0, 0.0], [20.0, 0.0]])
    t = defender_table(pc, vc, pd_, np.zeros((3, 2)))
    assert t["andr"][0] > t["andr"][1] > 0 and t["andr"][2] == 0      # goal-side pressure larger, far = none
    assert t["p_int"][0] > 0.5 > t["p_int"][2]
    # defender running at carrier closes, running away does not
    vd = np.array([[-5.0, 0.0], [-5.0, 0.0], [0.0, 0.0]])
    c = closing_speed(pc, vc, pd_, vd)
    assert c[0] > 0 > c[1]
    # FS arrival: standing defender 10 m away -> solve 10 = 10 (t - (1-e^-t)) numerically
    a = fs_arrival_time(pc, np.array([[10.0, 0.0]]), np.zeros((1, 2)))[0]
    assert abs(10 * (a - (1 - np.exp(-a))) - 10) < 1e-3
    # running towards the carrier arrives sooner than running away
    a2 = fs_arrival_time(pc, np.array([[10.0, 0.0], [10.0, 0.0]]), np.array([[-5.0, 0], [5.0, 0]]))
    assert a2[0] < a < a2[1]
    # non-monotone case from the code review: 1 m away, running at 8 m/s towards the carrier -> ~0.12 s
    fast = fs_arrival_time(pc, np.array([[1.0, 0.0]]), np.array([[-8.0, 0.0]]))[0]
    assert fast < 0.2, fast
    # Andrienko with a teammate threat: defender between carrier and a teammate gets pressure from that
    # direction even when it is "behind" w.r.t. the goal
    mate = np.array([[-10.0, 0.0]])                        # teammate behind the carrier
    def_behind = np.array([[-2.5, 0.0]])                   # defender between carrier and that teammate
    assert andrienko(pc, def_behind) < andrienko(pc, def_behind, andrienko_threats(pc, mate, closer_only=False))
    # closer_only drops a teammate further from goal than the carrier
    assert len(andrienko_threats(pc, mate, closer_only=True)) == 1
    # Bekkers Eq. 5
    assert abs(pressing_intensity_total([0.5, 0.5]) - 0.75) < 1e-12
    print("ok")
