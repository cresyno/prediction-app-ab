import os, io, json, time, datetime as dt, urllib.request
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

LEAGUES = {"E0": "Premier League", "E1": "Championship", "SP1": "La Liga", "SP2": "Segunda Division", "I1": "Serie A", "D1": "Bundesliga", "F1": "Ligue 1", "N1": "Eredivisie", "P1": "Primeira Liga"}
ACTIVE = ["E0", "E1", "SP1", "SP2", "I1", "D1", "F1", "N1", "P1"]
SEASONS_BACK = 4; XI = 0.0019; SHRINK = 2.0; MAX_GOALS = 10
XG_WEIGHT = 0.5; CONVERSION_RATE = 0.33
FOOTBALL_DATA_API_KEY = "2a67295f23384b56829c9b228e10b369"

BASE_URL = "https://www.football-data.co.uk/mmz4281/{s}/{c}.csv"
try: HERE = os.path.dirname(os.path.abspath(__file__))
except NameError: HERE = os.getcwd()
DATA_DIR = os.path.join(HERE, "fb_data")
TODAY = pd.Timestamp(dt.date.today())

def get(url, name, max_age_h=None, force=False):
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, name)
    have = os.path.exists(path)
    fresh = have and (max_age_h is None or (time.time() - os.path.getmtime(path)) < max_age_h * 3600)
    if max_age_h is not None and force: fresh = False
    if not fresh:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            raw = urllib.request.urlopen(req, timeout=30).read()
            if len(raw) > 200:
                with open(path, "wb") as f: f.write(raw)
                have = True
        except: pass
    if have:
        with open(path, "rb") as f: return f.read()
    return None

def parse_csv(raw):
    text = raw.decode("utf-8-sig", errors="replace")
    try: return pd.read_csv(io.StringIO(text), on_bad_lines="skip")
    except: return pd.read_csv(io.StringIO(text), error_bad_lines=False)

def parse_dates(s):
    s = s.astype(str).str.strip()
    d = pd.to_datetime(s, format="%d/%m/%Y", errors="coerce")
    return d.fillna(pd.to_datetime(s, format="%d/%m/%y", errors="coerce"))

def season_codes():
    start = TODAY.year if TODAY.month >= 7 else TODAY.year - 1
    return ["%02d%02d" % (s % 100, (s + 1) % 100) for s in range(start, start - SEASONS_BACK, -1)]

def load_results(code, force=False):
    frames = []
    seasons = season_codes()
    need = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "HST", "AST"]
    for i, s in enumerate(seasons):
        age = 6 if i == 0 else None
        raw = get(BASE_URL.format(s=s, c=code), "%s_%s.csv" % (code, s), age, force)
        if raw is None: continue
        df = parse_csv(raw)
        if not all(c in df.columns for c in need):
            if all(c in df.columns for c in ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]):
                df["HST"], df["AST"] = df["FTHG"], df["FTAG"]
            else: continue
        df = df[need].copy()
        df["Date"] = parse_dates(df["Date"])
        for c in need[3:]: df[c] = pd.to_numeric(df[c], errors="coerce")
        df["HST"] = df["HST"].fillna(df["FTHG"])
        df["AST"] = df["AST"].fillna(df["FTAG"])
        frames.append(df.dropna())
    if not frames: return None
    out = pd.concat(frames).sort_values("Date").reset_index(drop=True)
    out["FTHG"], out["FTAG"] = out["FTHG"].astype(int), out["FTAG"].astype(int)
    return out

def clean_name(name):
    # Strips common suffixes so "FC Barcelona" matches "Barcelona"
    for suffix in [" FC", " AFC", " CF", " AC", " SC", " SV", " 04"]:
        name = name.replace(suffix, "")
    return name.strip()

def load_fixtures(force=False):
    api_codes = {"E0": "PL", "E1": "ELC", "SP1": "PD", "SP2": "SD", "I1": "SA", "D1": "BL1", "F1": "FL1", "N1": "DED", "P1": "PPL"}
    today_str = TODAY.strftime('%Y-%m-%d')
    end_date_str = (TODAY + pd.Timedelta(days=7)).strftime('%Y-%m-%d')
    all_matches = []
    
    for league_code in ACTIVE:
        comp_code = api_codes.get(league_code)
        if not comp_code: continue
        url = f"https://api.football-data.org/v4/competitions/{comp_code}/matches?dateFrom={today_str}&dateTo={end_date_str}&status=SCHEDULED"
        req = urllib.request.Request(url, headers={'X-Auth-Token': FOOTBALL_DATA_API_KEY})
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                data = json.loads(response.read().decode())
            for m in data.get('matches', []):
                match_date = pd.to_datetime(m['utcDate']).tz_convert('UTC').tz_localize(None)
                match_time = match_date.strftime('%H:%M')
                h_name = clean_name(m['homeTeam']['name'])
                a_name = clean_name(m['awayTeam']['name'])
                all_matches.append({"Div": league_code, "Date": match_date, "Time": match_time, "HomeTeam": h_name, "AwayTeam": a_name})
        except Exception as e: print(f"API Error {comp_code}: {e}")

    if not all_matches: return None
    df = pd.DataFrame(all_matches)
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values(["Date", "Time"]).reset_index(drop=True)

def fit_league(df):
    teams = sorted(set(df.HomeTeam) | set(df.AwayTeam))
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    hi, ai = df.HomeTeam.map(idx).to_numpy(), df.AwayTeam.map(idx).to_numpy()
    x, y = df.FTHG.to_numpy(), df.FTAG.to_numpy()
    y_h = (x * (1 - XG_WEIGHT)) + (df.HST.to_numpy() * CONVERSION_RATE * XG_WEIGHT)
    y_a = (y * (1 - XG_WEIGHT)) + (df.AST.to_numpy() * CONVERSION_RATE * XG_WEIGHT)
    w = np.exp(-XI * np.clip((TODAY - df.Date).dt.days.to_numpy(), 0, None))
    m00, m01, m10, m11 = (x == 0) & (y == 0), (x == 0) & (y == 1), (x == 1) & (y == 0), (x == 1) & (y == 1)

    def nll(p):
        att, dfn = p[:n] - p[:n].mean(), p[n:2 * n] - p[n:2 * n].mean()
        c0, gam, rho = p[2 * n], p[2 * n + 1], p[2 * n + 2]
        lam = np.exp(c0 + att[hi] + dfn[ai] + gam)
        mu = np.exp(c0 + att[ai] + dfn[hi])
        tau = np.ones_like(lam)
        tau[m00], tau[m01], tau[m10], tau[m11] = 1 - lam[m00] * mu[m00] * rho, 1 + lam[m01] * rho, 1 + mu[m10] * rho, 1 - rho
        ll = y_h * np.log(lam) - lam + y_a * np.log(mu) - mu + np.log(np.maximum(tau, 1e-9))
        return -(w * ll).sum() + SHRINK * ((att ** 2).sum() + (dfn ** 2).sum())

    x0 = np.zeros(2 * n + 3)
    x0[2 * n], x0[2 * n + 1], x0[2 * n + 2] = 0.2, 0.25, -0.05
    bounds = [(-3, 3)] * (2 * n) + [(-1, 1.5), (-1, 1), (-0.3, 0.3)]
    res = minimize(nll, x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 500})
    p = res.x
    return {"teams": teams, "idx": idx, "att": p[:n] - p[:n].mean(), "dfn": p[n:2 * n] - p[n:2 * n].mean(), "c0": p[2 * n], "gam": p[2 * n + 1], "rho": p[2 * n + 2], "n_matches": len(df)}

def analyse(model, home, away):
    i, j = model["idx"][home], model["idx"][away]
    lam = np.exp(model["c0"] + model["att"][i] + model["dfn"][j] + model["gam"])
    mu = np.exp(model["c0"] + model["att"][j] + model["dfn"][i])
    g = np.arange(MAX_GOALS + 1)
    P = np.outer(poisson.pmf(g, lam), poisson.pmf(g, mu))
    P[0, 0] *= 1 - lam * mu * model["rho"]; P[0, 1] *= 1 + lam * model["rho"]
    P[1, 0] *= 1 + mu * model["rho"]; P[1, 1] *= 1 - model["rho"]
    P = P / P.sum()
    H, A = np.meshgrid(g, g, indexing="ij")
    tot, margin = H + A, H - A
    dh, da = P.sum(axis=1), P.sum(axis=0)
    
    S = {}
    S["Result"] = {"Home Win": P[margin > 0].sum(), "Draw": P[margin == 0].sum(), "Away Win": P[margin < 0].sum()}
    S["Total Goals"] = {f"Over {l}": P[tot > l].sum() for l in [0.5, 1.5, 2.5, 3.5]}
    S["Team Goals"] = {f"{home} Over 0.5": dh[1:].sum(), f"{home} Over 1.5": dh[2:].sum(), f"{away} Over 0.5": da[1:].sum(), f"{away} Over 1.5": da[2:].sum()}
    S["BTTS"] = {"Yes": P[(H > 0) & (A > 0)].sum(), "No": P[(H == 0) | (A == 0)].sum()}
    flat = sorted(((P[i, j], f"{i}-{j}") for i in range(4) for j in range(4)), reverse=True)[:4]
    S["Correct Score"] = {lab: p for p, lab in flat}
    return {"home": home, "away": away, "xg_h": float(lam), "xg_a": float(mu), "sections": {t: {l: float(v) for l, v in d.items()} for t, d in S.items()}}

def build_all(force=False):
    models = {}
    for code in ACTIVE:
        df = load_results(code, force)
        if df is None or len(df) < 100: continue
        models[code] = fit_league(df)
    return models
