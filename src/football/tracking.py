"""Extract player positions from the tracking files at fixed offsets before given events."""
import bz2
import json
import os
from collections import deque

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "World_cup")


def pre_event_frames(game_id, pe_ids, offsets_frames=(15, 30, 60), smoothed=False):
    """Player rows at `offset` frames before each possession event in `pe_ids`.

    Offset 0 = the event frame itself. `smoothed=True` reads the Kalman-smoothed
    coordinates (homePlayersSmoothed / awayPlayersSmoothed), better for velocities.
    Coordinates are flipped with the event's teamAttackingDirection so the team in
    possession attacks towards +x (same convention as 01_build_events).
    Returns a list of dicts: game_id, pe_id, offset_frames, player_id, atk, x, y, visible.
    """
    events = json.load(open(os.path.join(DATA_DIR, "Event Data", f"{game_id}_event.json"), encoding="utf-8"))
    meta = {}
    for e in events:
        if e["possessionEventId"] not in pe_ids:
            continue
        ge = e.get("gameEvents") or {}
        jersey = {}
        for side in ("homePlayers", "awayPlayers"):
            for p in e.get(side) or []:
                jersey[(side, str(p.get("jerseyNum")))] = p.get("playerId")
        meta[e["possessionEventId"]] = dict(
            flip=(e.get("stadiumMetadata") or {}).get("teamAttackingDirection") == "L",
            atk_home=ge.get("homeTeam") is True, jersey=jersey)

    max_off = max(offsets_frames)
    buf = deque(maxlen=max_off + 1)
    rows, done = [], set()
    path = os.path.join(DATA_DIR, "Tracking Data", f"{game_id}.jsonl_tracking.bz2")
    with bz2.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            fr = json.loads(line)
            # frames linked to several possession events are repeated as extra lines: keep one per frameNum
            if not buf or fr["frameNum"] != buf[-1]["frameNum"]:
                buf.append(fr)
            pid = fr.get("possession_event_id")
            if pid is None or int(pid) not in meta or int(pid) in done:
                continue
            pid = int(pid)
            done.add(pid)
            m = meta[pid]
            fn = fr["frameNum"]
            by_num = {b["frameNum"]: b for b in buf}
            for off in offsets_frames:
                b = by_num.get(fn - off)
                if b is None or b.get("period") != fr.get("period"):
                    continue
                for side, atk in (("homePlayers", m["atk_home"]), ("awayPlayers", not m["atk_home"])):
                    seen = set()
                    for p in b.get(side + ("Smoothed" if smoothed else "")) or []:
                        jn = str(p.get("jerseyNum"))
                        if p.get("x") is None or jn in seen:      # some frames list a jersey twice
                            continue
                        seen.add(jn)
                        x, y = (-p["x"], -p["y"]) if m["flip"] else (p["x"], p["y"])
                        rows.append(dict(game_id=game_id, pe_id=pid, offset_frames=off,
                                         player_id=m["jersey"].get((side, str(p.get("jerseyNum")))),
                                         atk=atk, x=x, y=y, visible=p.get("visibility") == "VISIBLE"))
    return rows


def possession_distances(game_id, pe_ids, max_frames=450):
    """Defender proximity over the possession that ends with each possession event in `pe_ids`.

    Window: from the game event's start_frame (the carrier receives / starts the possession) to the
    event frame (e.g. the pass), capped at `max_frames` (15 s). Raw broadcast coordinates, one record
    per frameNum, duplicate jerseys dropped. Returns one dict per event with:
    n_frames, d_start (nearest defender at possession start), d_end (at the event frame),
    d_min (minimum over the window), t_min_before (s before the event when d_min occurred),
    presser_d_start / presser_d_min (the named presser, if any).
    """
    import numpy as np
    events = json.load(open(os.path.join(DATA_DIR, "Event Data", f"{game_id}_event.json"), encoding="utf-8"))
    meta = {}
    for e in events:
        if e["possessionEventId"] not in pe_ids:
            continue
        pe = e.get("possessionEvents") or {}
        ge = e.get("gameEvents") or {}
        jersey = {}
        for side in ("homePlayers", "awayPlayers"):
            for p in e.get(side) or []:
                jersey[p.get("playerId")] = (side, str(p.get("jerseyNum")))
        atk_side = "homePlayers" if ge.get("homeTeam") is True else "awayPlayers"
        actor = pe.get("passerPlayerId") or ge.get("playerId")
        meta[e["possessionEventId"]] = dict(
            atk_side=atk_side, def_side="awayPlayers" if atk_side == "homePlayers" else "homePlayers",
            actor=jersey.get(actor), presser=jersey.get(pe.get("pressurePlayerId")))

    def xy(frame, side):
        seen, pts = set(), {}
        for p in frame.get(side) or []:
            jn = str(p.get("jerseyNum"))
            if p.get("x") is None or jn in seen:
                continue
            seen.add(jn)
            pts[jn] = (p["x"], p["y"])
        return pts

    buf = deque(maxlen=max_frames + 1)
    out, done = [], set()
    path = os.path.join(DATA_DIR, "Tracking Data", f"{game_id}.jsonl_tracking.bz2")
    with bz2.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            fr = json.loads(line)
            if not buf or fr["frameNum"] != buf[-1]["frameNum"]:
                buf.append(fr)
            pid = fr.get("possession_event_id")
            if pid is None or int(pid) not in meta or int(pid) in done:
                continue
            pid = int(pid)
            done.add(pid)
            m = meta[pid]
            if m["actor"] is None:
                continue
            start = (fr.get("game_event") or {}).get("start_frame") or fr["frameNum"]
            window = [b for b in buf if start <= b["frameNum"] <= fr["frameNum"] and b.get("period") == fr.get("period")]
            nearest, presser_d = [], []
            for b in window:
                atk, dfd = xy(b, m["atk_side"]), xy(b, m["def_side"])
                c = atk.get(m["actor"][1])
                if c is None or not dfd:
                    nearest.append(np.nan)
                    presser_d.append(np.nan)
                    continue
                d = {j: np.hypot(x - c[0], y - c[1]) for j, (x, y) in dfd.items()}
                nearest.append(min(d.values()))
                own = m["presser"] is not None and m["presser"][0] == m["def_side"]
                presser_d.append(d.get(m["presser"][1], np.nan) if own else np.nan)
            nearest, presser_d = np.array(nearest, float), np.array(presser_d, float)
            if len(nearest) == 0 or np.all(np.isnan(nearest)):
                continue
            i_min = int(np.nanargmin(nearest))
            out.append(dict(game_id=game_id, pe_id=pid, n_frames=len(window),
                            d_start=nearest[0], d_end=nearest[-1], d_min=float(np.nanmin(nearest)),
                            t_min_before=(window[-1]["frameNum"] - window[i_min]["frameNum"]) / 29.97,
                            presser_d_start=presser_d[0],
                            presser_d_min=float(np.nanmin(presser_d)) if np.any(~np.isnan(presser_d)) else np.nan))
    return out


def possession_exposure(game_id, pe_ids, max_frames=450, radii=(2.74, 4.57)):
    """Pressure exposure of the carrier over the possession ending with each event in `pe_ids`.

    Players are identified in the tracking file by position, not by jersey number or team list: at the event
    frame, all event-file players (carrier, teammates, defenders) are matched to tracking tracks by position
    (Hungarian assignment, pairs > 5 m discarded), because in a few matches the two files assign different
    jersey numbers / team sides to the same track. Those tracking identities are then followed backwards to
    the possession start (game event start_frame), capped at `max_frames` (15 s).

    Returns per event: n_frames, match_err (m, event passer vs matched track at the event frame),
    d_start, d_end, d_min, d_mean, t_under_<r> and share_under_<r> (time / share of frames with the
    nearest defender within r m), onset_<r> (s from possession start to first frame within r m, NaN if never).
    """
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    fps = 29.97
    events = json.load(open(os.path.join(DATA_DIR, "Event Data", f"{game_id}_event.json"), encoding="utf-8"))
    meta = {}
    for e in events:
        if e["possessionEventId"] not in pe_ids:
            continue
        pe = e.get("possessionEvents") or {}
        ge = e.get("gameEvents") or {}
        atk_side = "homePlayers" if ge.get("homeTeam") is True else "awayPlayers"
        def_side = "awayPlayers" if atk_side == "homePlayers" else "homePlayers"
        actor = pe.get("passerPlayerId") or ge.get("playerId")
        pts = [(p["x"], p["y"], "carrier" if p.get("playerId") == actor else "atk") for p in e.get(atk_side) or [] if p.get("x") is not None]
        pts += [(p["x"], p["y"], "def") for p in e.get(def_side) or [] if p.get("x") is not None]
        if any(r == "carrier" for _, _, r in pts):
            meta[e["possessionEventId"]] = pts

    def xy(frame, side):
        seen, pts = set(), {}
        for p in frame.get(side) or []:
            jn = str(p.get("jerseyNum"))
            if p.get("x") is None or jn in seen:
                continue
            seen.add(jn)
            pts[jn] = (p["x"], p["y"])
        return pts

    buf = deque(maxlen=max_frames + 1)
    out, done = [], set()
    path = os.path.join(DATA_DIR, "Tracking Data", f"{game_id}.jsonl_tracking.bz2")
    with bz2.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            fr = json.loads(line)
            if not buf or fr["frameNum"] != buf[-1]["frameNum"]:
                buf.append(fr)
            pid = fr.get("possession_event_id")
            if pid is None or int(pid) not in meta or int(pid) in done:
                continue
            pid = int(pid)
            done.add(pid)
            ev_pts = meta[pid]
            # carrier = tracking track closest to the event-file passer position (any team list);
            # defenders = all tracks in the other team list of the tracking file (internally consistent)
            px, py = next((x, y) for x, y, r in ev_pts if r == "carrier")
            tracks = [((side, jn), xy_) for side in ("homePlayers", "awayPlayers") for jn, xy_ in xy(fr, side).items()]
            if len(tracks) < 5:
                continue
            carrier, cxy = min(tracks, key=lambda t: np.hypot(t[1][0] - px, t[1][1] - py))
            match_err = float(np.hypot(cxy[0] - px, cxy[1] - py))
            other = "awayPlayers" if carrier[0] == "homePlayers" else "homePlayers"
            defenders = [k for k, _ in tracks if k[0] == other]
            if not defenders:
                continue
            start = (fr.get("game_event") or {}).get("start_frame") or fr["frameNum"]
            window = [b for b in buf if start <= b["frameNum"] <= fr["frameNum"] and b.get("period") == fr.get("period")]
            nearest = []
            for b in window:
                pos = {**{("homePlayers", k): v for k, v in xy(b, "homePlayers").items()},
                       **{("awayPlayers", k): v for k, v in xy(b, "awayPlayers").items()}}
                c = pos.get(carrier)
                dd = [pos[k] for k in defenders if k in pos]
                nearest.append(min(np.hypot(x - c[0], y - c[1]) for x, y in dd) if c and dd else np.nan)
            nearest = np.array(nearest, float)
            ok = ~np.isnan(nearest)
            if not ok.any():
                continue
            rec = dict(game_id=game_id, pe_id=pid, n_frames=len(window), match_err=match_err,
                       d_start=nearest[ok][0], d_end=nearest[ok][-1], d_min=float(nearest[ok].min()),
                       d_mean=float(nearest[ok].mean()))
            for r in radii:
                under = ok & (nearest < r)
                tag = f"{r:.2f}".replace(".", "")
                rec[f"t_under_{tag}"] = under.sum() / fps
                rec[f"share_under_{tag}"] = under.sum() / ok.sum()
                rec[f"onset_{tag}"] = (np.argmax(under) / fps) if under.any() else np.nan
            out.append(rec)
    return out
