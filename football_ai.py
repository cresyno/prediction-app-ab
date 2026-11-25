"""
FOOTBALL AI - script 1: multi-league Dixon-Coles goals engine
Free data from football-data.co.uk. Needs: numpy, pandas, scipy.
"""
import os
import io
import json
import time
import datetime as dt
import urllib.request

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

# ------------------------- SETTINGS -------------------------
LEAGUES = {
    "E0": "Premier League",
    "SP1": "La Liga",
    "I1": "Serie A",
    "D1": "Bundesliga",
    "F1": "Ligue 1",
    "E1": "Championship",
}
ACTIVE = ["E0", "SP1"]   # add more codes, e.g. ["E0", "SP1", "I1", "D1"]
SEASONS_BACK = 4         # seasons of history to learn from
XI = 0.0019              # time decay per day (half-life ~ 1 year)
SHRINK = 2.0             # pulls teams with few games towards average
MAX_GOALS = 10
# ------------------------------------------------------------

BASE_URL = "https://www.football-data.co.uk/mmz4281/{s}/{c}.csv"
FIXT_URL = "https://www.football-data.co.uk/fixtures.csv"
try:
    HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    HERE = os.getcwd()
DATA_DIR = os.path.join(HERE, "fb_data")
TODAY = pd.Timestamp(dt.date.today())


# ----------------------- DATA LOADING -----------------------
def get(url, name, max_age_h=None, force=False):
    """Download with a local cache. max_age_h=None -> cache forever."""
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, name)
    have = os.path.exists(path)
    fresh = have and (max_age_h is None or
                      (time.time() - os.path.getmtime(path)) < max_age_h * 3600)
    if max_age_h is not None and force:
        fresh = False
    if not fresh:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            raw = urllib.request.urlopen(req, timeout=30).read()
            if len(raw) > 200:
                with open(path, "wb") as f:
                    f.write(raw)
                have = True
        except Exception as e:
            print("  could not download " + name + ": " + str(e)[:60])
    if have:
        with open(path, "rb") as f:
            return f.read()
    return None


def parse_csv(raw):
    text = raw.decode("utf-8-sig", errors="replace")
    try:
        return pd.read_csv(io.StringIO(text), on_bad_lines="skip")
    except TypeError:           # very old pandas
        return pd.read_csv(io.StringIO(text), error_bad_lines=False)


def parse_dates(s):
    s = s.astype(str).str.strip()
    d = pd.to_datetime(s, format="%d/%m/%Y", errors="coerce")
    return d.fillna(pd.to_datetime(s, format="%d/%m/%y", errors="coerce"))


def season_codes():
    start = TODAY.year if TODAY.month >= 7 else TODAY.year - 1
    return ["%02d%02d" % (s % 100, (s + 1) % 100)
            for s in range(start, start - SEASONS_BACK, -1)]


def load_results(code, force=False):
    frames = []
    seasons = season_codes()
    need = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]
    for i, s in enumerate(seasons):
        age = 6 if i == 0 else None      # only the current season refreshes
        raw = get(BASE_URL.format(s=s, c=code), "%s_%s.csv" % (code, s), age, force)
        if raw is None:
            continue
        df = parse_csv(raw)
        if not all(c in df.columns for c in need):
            continue
        df = df[need].copy()
        df["Date"] = parse_dates(df["Date"])
        df["FTHG"] = pd.to_numeric(df["FTHG"], errors="coerce")
        df["FTAG"] = pd.to_numeric(df["FTAG"], errors="coerce")
        df = df.dropna()
        frames.append(df)
    if not frames:
        return None
    out = pd.concat(frames).sort_values("Date").reset_index(drop=True)
    out["FTHG"] = out["FTHG"].astype(int)
    out["FTAG"] = out["FTAG"].astype(int)
    return out


def load_fixtures(force=False):
    raw = get(FIXT_URL, "fixtures.csv", 6, force)
    if raw is None:
        return None
    df = parse_csv(raw)
    if not all(c in df.columns for c in ["Div", "Date", "HomeTeam", "AwayTeam"]):
        return None
    if "Time" not in df.columns:
        df["Time"] = ""
    df["Time"] = df["Time"].fillna("").astype(str)
    df = df[df["Div"].isin(ACTIVE)].copy()
    df["Date"] = parse_dates(df["Date"])
    df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam"])
    df = df[df["Date"] >= TODAY]
    return df.sort_values(["Date", "Time"]).reset_index(drop=True)


# ------------------ DIXON-COLES MODEL ------------------
def fit_league(df):
    teams = sorted(set(df.HomeTeam) | set(df.AwayTeam))
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    hi = df.HomeTeam.map(idx).to_numpy()
    ai = df.AwayTeam.map(idx).to_numpy()
    x = df.FTHG.to_numpy()
    y = df.FTAG.to_numpy()
    days = (TODAY - df.Date).dt.days.to_numpy()
    w = np.exp(-XI * np.clip(days, 0, None))
    m00 = (x == 0) & (y == 0)
    m01 = (x == 0) & (y == 1)
    m10 = (x == 1) & (y == 0)
    m11 = (x == 1) & (y == 1)

    def nll(p):
        att = p[:n] - p[:n].mean()
        dfn = p[n:2 * n] - p[n:2 * n].mean()   # defence "weakness"
        c0, gam, rho = p[2 * n], p[2 * n + 1], p[2 * n + 2]
        lam = np.exp(c0 + att[hi] + dfn[ai] + gam)
        mu = np.exp(c0 + att[ai] + dfn[hi])
        tau = np.ones_like(lam)
        tau[m00] = 1 - lam[m00] * mu[m00] * rho
        tau[m01] = 1 + lam[m01] * rho
        tau[m10] = 1 + mu[m10] * rho
        tau[m11] = 1 - rho
        ll = x * np.log(lam) - lam + y * np.log(mu) - mu + np.log(np.maximum(tau, 1e-9))
        return -(w * ll).sum() + SHRINK * ((att ** 2).sum() + (dfn ** 2).sum())

    x0 = np.zeros(2 * n + 3)
    x0[2 * n] = 0.2        # base scoring level
    x0[2 * n + 1] = 0.25   # home advantage
    x0[2 * n + 2] = -0.05  # low-score correction
    bounds = [(-3, 3)] * (2 * n) + [(-1, 1.5), (-1, 1), (-0.3, 0.3)]
    res = minimize(nll, x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 500})
    p = res.x
    s_year = TODAY.year if TODAY.month >= 7 else TODAY.year - 1
    cur = df[df.Date >= pd.Timestamp(s_year, 7, 1)]
    current = sorted(set(cur.HomeTeam) | set(cur.AwayTeam))
    if len(current) < 12:      # season barely started: use the last 400 days
        rec = df[(TODAY - df.Date).dt.days < 400]
        current = sorted(set(rec.HomeTeam) | set(rec.AwayTeam))
    return {
        "teams": teams, "idx": idx, "current": current,
        "att": p[:n] - p[:n].mean(), "dfn": p[n:2 * n] - p[n:2 * n].mean(),
        "c0": p[2 * n], "gam": p[2 * n + 1], "rho": p[2 * n + 2], "n_matches": len(df),
    }


def expected_goals(model, home, away):
    i, j = model["idx"][home], model["idx"][away]
    lam = np.exp(model["c0"] + model["att"][i] + model["dfn"][j] + model["gam"])
    mu = np.exp(model["c0"] + model["att"][j] + model["dfn"][i])
    return float(lam), float(mu)


def score_matrix(lam, mu, rho):
    g = np.arange(MAX_GOALS + 1)
    P = np.outer(poisson.pmf(g, lam), poisson.pmf(g, mu))
    P[0, 0] *= 1 - lam * mu * rho
    P[0, 1] *= 1 + lam * rho
    P[1, 0] *= 1 + mu * rho
    P[1, 1] *= 1 - rho
    return P / P.sum()          # rows = home goals, columns = away goals


# ----------------------- MARKETS -----------------------
def analyse(model, home, away):
    lam, mu = expected_goals(model, home, away)
    P = score_matrix(lam, mu, model["rho"])
    g = np.arange(P.shape[0])
    H, A = np.meshgrid(g, g, indexing="ij")
    tot, margin = H + A, H - A
    pH, pD, pA = P[margin > 0].sum(), P[margin == 0].sum(), P[margin < 0].sum()
    dh, da = P.sum(axis=1), P.sum(axis=0)

    S = {}
    S["Result (1X2)"] = {home + " win": pH, "Draw": pD, away + " win": pA}
    S["Double chance"] = {home + " or draw": pH + pD, away + " or draw": pA + pD,
                          "Either team wins": pH + pA}
    S["Draw no bet"] = {home: pH / (pH + pA), away: pA / (pH + pA)}
    tg = {}
    for line in (0.5, 1.5, 2.5, 3.5, 4.5):
        tg["Over %.1f" % line] = P[tot > line].sum()
        tg["Under %.1f" % line] = P[tot < line].sum()
    S["Total goals"] = tg
    S[home + " goals"] = {"Over %.1f" % (k + 0.5): dh[k + 1:].sum() for k in range(3)}
    S[away + " goals"] = {"Over %.1f" % (k + 0.5): da[k + 1:].sum() for k in range(3)}
    S["Both teams to score"] = {"Yes": P[(H > 0) & (A > 0)].sum(),
                                "No": P[(H == 0) | (A == 0)].sum()}
    S["Handicap " + home + " -1"] = {home + " (-1)": P[margin >= 2].sum(),
                                     "Draw": P[margin == 1].sum(),
                                     away + " (+1)": P[margin <= 0].sum()}
    S["Handicap " + away + " -1"] = {home + " (+1)": P[margin >= 0].sum(),
                                     "Draw": P[margin == -1].sum(),
                                     away + " (-1)": P[margin <= -2].sum()}
    flat = sorted(((P[i, j], "%d-%d" % (i, j)) for i in range(6) for j in range(6)),
                  reverse=True)[:5]
    S["Correct score (top 5)"] = {lab: p for p, lab in flat}
    S = {t: {l: float(v) for l, v in d.items()} for t, d in S.items()}
    return {"home": home, "away": away, "xg_h": lam, "xg_a": mu, "sections": S,
            "dist_h": dh, "dist_a": da}


def show(a, league=""):
    h, w = a["home"], a["away"]
    print("\n" + "=" * 38)
    print("%s v %s" % (h, w))
    if league:
        print(league)
    print("Expected goals: %.2f - %.2f" % (a["xg_h"], a["xg_a"]))
    for title, items in a["sections"].items():
        print("\n" + title.upper())
        for label, p in items.items():
            print("%-21s%5.1f%%  %5.2f" % (label[:21], p * 100, 1 / max(p, 1e-4)))
    for name, dist in ((h, a["dist_h"]), (w, a["dist_a"])):
        k = int(np.argmax(dist))
        print("\n%s SCORES" % name.upper())
        print("0: %.0f%%  1: %.0f%%  2: %.0f%%  3+: %.0f%%" %
              (dist[0] * 100, dist[1] * 100, dist[2] * 100, dist[3:].sum() * 100))
        if k == 0:
            print("Most likely: 0 goals (%.0f%%)" % (dist[0] * 100))
        else:
            print("Most likely: %d goal%s (%.0f%%)" % (k, "" if k == 1 else "s", dist[k] * 100))
            print("Safer: Over %.1f (%.0f%%)" % (k - 0.5, dist[k:].sum() * 100))
    print("\nColumns: probability, fair odds (1/prob).")
    print("If SportyBet pays MORE than fair odds it may be value.")
    print("NOT BACKTESTED YET - treat as an estimate.")


def short_line(a):
    s = a["sections"]
    r = s["Result (1X2)"]
    one = list(r.values())
    return "  1/X/2 %.0f/%.0f/%.0f  O2.5 %.0f  BTTS %.0f" % (
        one[0] * 100, one[1] * 100, one[2] * 100,
        s["Total goals"]["Over 2.5"] * 100, s["Both teams to score"]["Yes"] * 100)


# ----------------------- INTERFACE -----------------------
def build_all(force=False):
    models = {}
    for code in ACTIVE:
        name = LEAGUES.get(code, code)
        print("\n%s: loading data..." % name)
        df = load_results(code, force)
        if df is None or len(df) < 100:
            print("  not enough data, skipping")
            continue
        t0 = time.time()
        m = fit_league(df)
        models[code] = m
        print("  fitted %d matches, %d teams, home advantage x%.2f (%.0fs)" %
              (m["n_matches"], len(m["teams"]), np.exp(m["gam"]), time.time() - t0))
    return models


def pick_number(prompt, lo, hi):
    try:
        v = int(input(prompt).strip())
        if lo <= v <= hi:
            return v
    except ValueError:
        pass
    return None


def fixtures_menu(models, force=False):
    fx = load_fixtures(force)
    if fx is None or len(fx) == 0:
        print("\nNo upcoming fixtures found (or download failed).")
        print("Use option 2 to pick a match yourself.")
        return
    items, skipped = [], 0
    for _, r in fx.iterrows():
        m = models.get(r["Div"])
        if m is None or r["HomeTeam"] not in m["idx"] or r["AwayTeam"] not in m["idx"]:
            skipped += 1
            continue
        a = analyse(m, r["HomeTeam"], r["AwayTeam"])
        a["league"] = LEAGUES.get(r["Div"], r["Div"])
        a["date"] = r["Date"].strftime("%Y-%m-%d")
        a["time"] = r["Time"]
        items.append(a)
    if not items:
        print("\nNo fixtures could be predicted.")
        return
    print("\nUPCOMING (%d matches)" % len(items))
    for n, a in enumerate(items, 1):
        print("%2d. %s %s  %s v %s" % (n, a["date"][5:], a["time"], a["home"][:12], a["away"][:12]))
        print(short_line(a))
    if skipped:
        print("(%d fixtures skipped: team not in data)" % skipped)
    out = {"generated": dt.datetime.now().isoformat(timespec="seconds"), "matches": []}
    for a in items:
        out["matches"].append({
            "league": a["league"], "date": a["date"], "time": a["time"],
            "home": a["home"], "away": a["away"],
            "xg_home": round(a["xg_h"], 3), "xg_away": round(a["xg_a"], 3),
            "markets": {t: {l: round(v, 4) for l, v in d.items()}
                        for t, d in a["sections"].items()},
        })
    with open(os.path.join(HERE, "predictions.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("\nSaved predictions.json (for the Vercel page later).")
    while True:
        k = pick_number("\nMatch number for full detail (Enter = back): ", 1, len(items))
        if k is None:
            return
        show(items[k - 1], items[k - 1]["league"])


def pick_match_menu(models):
    codes = list(models)
    for n, c in enumerate(codes, 1):
        print("%d) %s" % (n, LEAGUES.get(c, c)))
    k = pick_number("League: ", 1, len(codes))
    if k is None:
        return
    code = codes[k - 1]
    m = models[code]
    teams = m["current"]
    for n, t in enumerate(teams, 1):
        print("%2d. %s" % (n, t))
    h = pick_number("HOME team number: ", 1, len(teams))
    a = pick_number("AWAY team number: ", 1, len(teams))
    if h is None or a is None or h == a:
        print("Invalid choice.")
        return
    show(analyse(m, teams[h - 1], teams[a - 1]), LEAGUES.get(code, code))


def ratings_menu(models):
    codes = list(models)
    for n, c in enumerate(codes, 1):
        print("%d) %s" % (n, LEAGUES.get(c, c)))
    k = pick_number("League: ", 1, len(codes))
    if k is None:
        return
    m = models[codes[k - 1]]
    rows = []
    for t in m["current"]:
        i = m["idx"][t]
        rows.append((m["att"][i] - m["dfn"][i], t, np.exp(m["att"][i]), np.exp(m["dfn"][i])))
    rows.sort(reverse=True)
    print("\n%-15s%6s%6s" % ("Team", "Att", "Def"))
    for _, t, a, d in rows:
        print("%-15s%6.2f%6.2f" % (t[:15], a, d))
    print("Att > 1 = scores more than average.")
    print("Def < 1 = concedes less than average.")


def main():
    print("FOOTBALL AI - script 1")
    print("First run downloads data and fits the model. Please wait...")
    models = build_all()
    if not models:
        print("\nNo league could be loaded. Check your internet and run again.")
        return
    refresh = False
    while True:
        print("\n==== MENU ====")
        print("1) Upcoming fixtures")
        print("2) Pick a match yourself")
        print("3) Team ratings")
        print("4) Refresh data and refit")
        print("0) Exit")
        try:
            c = input("> ").strip()
        except EOFError:
            return
        if c == "1":
            fixtures_menu(models, refresh)
            refresh = False
        elif c == "2":
            pick_match_menu(models)
        elif c == "3":
            ratings_menu(models)
        elif c == "4":
            models = build_all(force=True) or models
            refresh = True
        elif c == "0":
            return


if __name__ == "__main__":
    main()