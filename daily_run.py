"""
daily_run.py  (safer version)

Drop-in replacement for the old daily_run.py. Keep football_ai.py as it is.
What changed, in plain words:
  1. Team names are matched strictly. If a team cannot be matched with confidence it is
     SKIPPED and listed, instead of being guessed (the old code turned "Paris FC" into
     "Paris SG").
  2. Fixtures come only from leagues the free football-data.org plan really covers.
     (The old code asked for Brazil's Serie A as "Belgian league", and for leagues the free
     plan does not include, so those fixtures were empty or wrong.)
  3. Upcoming games are taken with status SCHEDULED *and* TIMED (the old code only asked
     for SCHEDULED, which can drop games whose kick-off time is already confirmed).
  4. Kick-off times are shown in Nigerian time (WAT, UTC+1).
  5. Nothing fails silently: every skipped game, failed download and stale data file is
     printed and saved in predictions.json under "skipped" / "warnings".
  6. Every first prediction of a game is appended to predictions_log.csv so the model can
     later be graded on real games it predicted BEFORE they were played.
  7. The API key is read from api_key.txt (next to this file) or the environment variable
     FOOTBALL_DATA_API_KEY, so it does not have to live in the code.
"""
import os, re, csv, json, time, difflib, unicodedata, urllib.request, urllib.error
import datetime as dt
import pandas as pd
import football_ai as fa

# Leagues to run. Each needs (a) a results CSV on football-data.co.uk and
# (b) a competition on the football-data.org FREE plan for the fixtures.
RUN = ["E0", "E1", "SP1", "D1", "N1", "I1", "F1"]
FREE_API = {"E0": "PL", "E1": "ELC", "SP1": "PD", "D1": "BL1", "N1": "DED", "I1": "SA", "F1": "FL1"}
NAMES = {"E0": "Premier League", "E1": "Championship", "SP1": "La Liga", "D1": "Bundesliga",
         "N1": "Eredivisie", "I1": "Serie A", "F1": "Ligue 1"}
DAYS_AHEAD = 7
WAT = pd.Timedelta(hours=1)        # Nigeria = UTC+1, no daylight saving
HERE = fa.HERE

# --------------------------------------------------------------------------- names
def norm(s):
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()

DROP = {"fc", "afc", "cf", "ac", "sc", "sv", "ssc", "as", "us", "rc", "rcd", "cd", "ud", "calcio"}

def key(s):
    toks = [t for t in norm(s).split() if t not in DROP and not (t.isdigit() and len(t) <= 4)]
    return " ".join(toks)

# API name -> football-data.co.uk name. (Taken from the old file; the wrong
# "Paris" -> "Paris SG" line is removed.)
ALIAS_RAW = {
    "Manchester City": "Man City", "Manchester United": "Man United", "Nottingham Forest": "Nott'm Forest",
    "Wolverhampton Wanderers": "Wolves", "Tottenham Hotspur": "Tottenham", "Paris Saint-Germain": "Paris SG",
    "FC Bayern München": "Bayern Munich", "Borussia Mönchengladbach": "M'gladbach",
    "Club Atlético de Madrid": "Ath Madrid", "Athletic Club": "Ath Bilbao", "Real Sociedad de Fútbol": "Sociedad",
    "Real Betis Balompié": "Betis", "RC Celta de Vigo": "Celta", "Rayo Vallecano de Madrid": "Vallecano",
    "Deportivo Alavés": "Alaves", "FC Internazionale Milano": "Inter", "1. FC Köln": "FC Koln",
    "1. FC Union Berlin": "Union Berlin", "Eintracht Frankfurt": "Ein Frankfurt", "Borussia Dortmund": "Dortmund",
    "Bayer 04 Leverkusen": "Leverkusen", "Hamburger SV": "Hamburg", "Stade Rennais FC 1901": "Rennes",
    "Stade Brestois 29": "Brest", "RC Strasbourg Alsace": "Strasbourg", "Olympique de Marseille": "Marseille",
    "Olympique Lyonnais": "Lyon", "AS Monaco FC": "Monaco", "Lille OSC": "Lille", "ACF Fiorentina": "Fiorentina",
    "AC Milan": "Milan", "Parma Calcio 1913": "Parma", "US Sassuolo Calcio": "Sassuolo", "Como 1907": "Como",
    "Bologna FC 1909": "Bologna", "Hellas Verona FC": "Verona", "Genoa CFC": "Genoa", "OGC Nice": "Nice",
    "Angers SCO": "Angers", "FC Lorient": "Lorient", "Toulouse FC": "Toulouse", "Le Havre AC": "Le Havre",
    "AJ Auxerre": "Auxerre", "FC Metz": "Metz", "FC Nantes": "Nantes", "Paris FC": "Paris FC",
    "Queens Park Rangers": "QPR", "West Bromwich Albion": "West Brom", "Blackburn Rovers": "Blackburn",
    "Birmingham City": "Birmingham", "Stoke City": "Stoke", "Derby County": "Derby", "Leeds United": "Leeds",
    "Coventry City": "Coventry", "Preston North End": "Preston", "Norwich City": "Norwich",
    "Charlton Athletic": "Charlton", "Swansea City": "Swansea", "Cardiff City": "Cardiff", "Hull City": "Hull",
    "Ipswich Town": "Ipswich", "Sheffield United": "Sheffield United", "Sheffield Wednesday": "Sheffield Weds",
    "Oxford United FC": "Oxford", "Leicester City": "Leicester", "Newcastle United": "Newcastle",
    "West Ham United": "West Ham", "Brighton & Hove Albion": "Brighton", "AFC Bournemouth": "Bournemouth",
    "Real Madrid CF": "Real Madrid", "FC Barcelona": "Barcelona", "Real Oviedo": "Oviedo", "Levante UD": "Levante",
    "RCD Espanyol de Barcelona": "Espanyol", "RCD Mallorca": "Mallorca", "Elche CF": "Elche", "Girona FC": "Girona",
    "CA Osasuna": "Osasuna", "Valencia CF": "Valencia", "Villarreal CF": "Villarreal", "Getafe CF": "Getafe",
    "Sevilla FC": "Sevilla", "FC Augsburg": "Augsburg", "SV Werder Bremen": "Werder Bremen",
    "VfB Stuttgart": "Stuttgart", "VfL Wolfsburg": "Wolfsburg", "TSG 1899 Hoffenheim": "Hoffenheim",
    "1. FSV Mainz 05": "Mainz", "SC Freiburg": "Freiburg", "FC St. Pauli 1910": "St Pauli",
    "1. FC Heidenheim 1846": "Heidenheim", "RB Leipzig": "RB Leipzig", "AFC Ajax": "Ajax",
    "Feyenoord Rotterdam": "Feyenoord", "FC Twente '65": "Twente", "FC Utrecht": "Utrecht",
    "SC Heerenveen": "Heerenveen", "NEC Nijmegen": "Nijmegen", "PEC Zwolle": "Zwolle", "FC Groningen": "Groningen",
    "Sparta Rotterdam": "Sparta Rotterdam", "Go Ahead Eagles": "Go Ahead Eagles", "Fortuna Sittard": "For Sittard",
    "NAC Breda": "NAC Breda", "Heracles Almelo": "Heracles", "FC Volendam": "Volendam", "Excelsior Rotterdam": "Excelsior",
}
ALIAS = {key(k): v for k, v in ALIAS_RAW.items()}


def match_team(api_names, model_teams):
    """Return (team, how) or (None, reason). Never guesses loosely."""
    teams = set(model_teams)
    by_key = {}
    for t in model_teams:
        by_key.setdefault(key(t), []).append(t)
    names = [n for n in api_names if n]
    for n in names:                                   # 1 exact
        if n in teams:
            return n, "exact"
    for n in names:                                   # 2 alias table (accents, FC/AC and years ignored)
        t = ALIAS.get(key(n))
        if t in teams:
            return t, "alias"
    for n in names:                                   # 3 same name after removing FC/AC/years
        c = by_key.get(key(n), [])
        if len(c) == 1:
            return c[0], "clean"
    best = []                                         # 4 very close spelling only
    for n in names:
        kn = key(n)
        for k, c in by_key.items():
            if len(c) == 1:
                best.append((difflib.SequenceMatcher(None, kn, k).ratio(), c[0]))
    best.sort(reverse=True)
    if best and best[0][0] >= 0.88 and (len(best) == 1 or best[0][0] - best[1][0] >= 0.05):
        return best[0][1], "close"
    return None, "no safe match"


# --------------------------------------------------------------------------- API
def get_key():
    k = os.environ.get("FOOTBALL_DATA_API_KEY", "").strip()
    if k:
        return k, "env"
    p = os.path.join(HERE, "api_key.txt")
    if os.path.exists(p):
        k = open(p).read().strip()
        if k:
            return k, "file"
    return getattr(fa, "FOOTBALL_DATA_API_KEY", "").strip(), "code"


def fetch_fixtures(codes, warnings):
    key_, src = get_key()
    if not key_:
        warnings.append("No API key found. Put it in api_key.txt next to this file.")
        return []
    if src == "code":
        warnings.append("API key is still written inside football_ai.py. Move it to api_key.txt "
                        "and make sure that file is NOT uploaded to GitHub.")
    today = pd.Timestamp(dt.datetime.utcnow().date())
    d1, d2 = today.strftime("%Y-%m-%d"), (today + pd.Timedelta(days=DAYS_AHEAD)).strftime("%Y-%m-%d")
    rows = []
    for i, code in enumerate(codes):
        comp = FREE_API.get(code)
        if comp is None:
            warnings.append("%s: not covered by the free football-data.org plan, skipped" % code)
            continue
        if i:
            time.sleep(6.5)                          # free plan: 10 calls per minute
        url = "https://api.football-data.org/v4/competitions/%s/matches?dateFrom=%s&dateTo=%s" % (comp, d1, d2)
        data = None
        for attempt in range(2):
            try:
                req = urllib.request.Request(url, headers={"X-Auth-Token": key_})
                with urllib.request.urlopen(req, timeout=30) as r:
                    data = json.loads(r.read().decode())
                break
            except urllib.error.HTTPError as e:
                if e.code == 429 and attempt == 0:
                    print("  rate limit hit, waiting 60 seconds...")
                    time.sleep(60)
                    continue
                warnings.append("%s (%s): API error %s" % (code, comp, e.code))
                break
            except Exception as e:
                warnings.append("%s (%s): %s" % (code, comp, type(e).__name__))
                break
        if not data:
            continue
        n = 0
        for m in data.get("matches", []):
            if m.get("status") not in ("SCHEDULED", "TIMED"):
                continue                              # already played / live / postponed
            ko = pd.to_datetime(m["utcDate"]).tz_convert("UTC").tz_localize(None) + WAT
            h, a = m["homeTeam"], m["awayTeam"]
            rows.append({"Div": code, "Date": ko, "Time": ko.strftime("%H:%M"),
                         "H": [h.get("name"), h.get("shortName")], "A": [a.get("name"), a.get("shortName")]})
            n += 1
        print("  %-15s %d upcoming games" % (NAMES.get(code, code), n))
    return rows


# --------------------------------------------------------------------------- freshness + log
def stale_warnings(codes):
    out = []
    season = fa.season_codes()[0]
    for c in codes:
        p = os.path.join(fa.DATA_DIR, "%s_%s.csv" % (c, season))
        if not os.path.exists(p):
            out.append("%s: no results file for the current season (%s) - ratings use old data" % (c, season))
        else:
            age_h = (time.time() - os.path.getmtime(p)) / 3600
            if age_h > 30:
                out.append("%s: results data is %.0f hours old - download probably failed" % (c, age_h))
    return out


LOG = os.path.join(HERE, "predictions_log.csv")
LOG_COLS = ["logged_at", "date", "league", "home", "away", "xg_home", "xg_away", "p_home", "p_draw", "p_away",
            "p_over05", "p_over15", "p_over25", "p_btts_yes", "p_home_o05", "p_away_o05"]


def log_predictions(matches):
    seen = set()
    if os.path.exists(LOG):
        with open(LOG, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                seen.add((r["date"], r["home"], r["away"]))
    new = [m for m in matches if (m["date"], m["home"], m["away"]) not in seen]
    if not new:
        return 0
    write_head = not os.path.exists(LOG)
    with open(LOG, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_head:
            w.writerow(LOG_COLS)
        for m in new:
            S = m["markets"]
            tg = S["Team Goals"]
            w.writerow([dt.datetime.now().isoformat(timespec="seconds"), m["date"], m["league"], m["home"], m["away"],
                        m["xg_home"], m["xg_away"], round(S["Result"]["Home Win"], 4), round(S["Result"]["Draw"], 4),
                        round(S["Result"]["Away Win"], 4), round(S["Total Goals"]["Over 0.5"], 4),
                        round(S["Total Goals"]["Over 1.5"], 4), round(S["Total Goals"]["Over 2.5"], 4),
                        round(S["BTTS"]["Yes"], 4), round(tg["%s Over 0.5" % m["home"]], 4),
                        round(tg["%s Over 0.5" % m["away"]], 4)])
    return len(new)


# --------------------------------------------------------------------------- main
def run_automated():
    warnings, skipped = [], []
    fa.ACTIVE = list(RUN)
    fa.LEAGUES.update(NAMES)
    print("Building models...")
    models = fa.build_all(force=True)
    if not models:
        print("!! No models could be built (results download failed?). Nothing written.")
        return
    warnings += stale_warnings(list(models))
    print("Fetching fixtures...")
    fx = fetch_fixtures([c for c in RUN if c in models], warnings)
    if not fx:
        print("!! No fixtures received.")
        for w in warnings:
            print("  WARNING:", w)
        return

    out_matches = []
    for r in sorted(fx, key=lambda r: (r["Date"], r["Div"])):
        m = models[r["Div"]]
        teams = list(m["idx"].keys())
        h, hw = match_team(r["H"], teams)
        a, aw = match_team(r["A"], teams)
        if not h or not a:
            miss = [("%s" % r["H"][0]) if not h else None, ("%s" % r["A"][0]) if not a else None]
            skipped.append("%s: %s vs %s - team not in model: %s" % (NAMES[r["Div"]], r["H"][0], r["A"][0],
                           ", ".join(x for x in miss if x)))
            continue
        try:
            res = fa.analyse(m, h, a)
        except Exception as e:
            skipped.append("%s: %s vs %s - %s" % (NAMES[r["Div"]], h, a, e))
            continue
        out_matches.append({
            "league": NAMES[r["Div"]], "date": r["Date"].strftime("%Y-%m-%d"), "time": r["Time"],
            "home": res["home"], "away": res["away"],
            "xg_home": round(res["xg_h"], 2), "xg_away": round(res["xg_a"], 2),
            "markets": res["sections"], "form_home": res["form_h"], "form_away": res["form_a"],
        })

    out = {"generated": dt.datetime.now().isoformat(timespec="seconds"), "times": "Nigerian time (WAT)",
           "matches": out_matches, "skipped": skipped, "warnings": warnings}
    with open(os.path.join(HERE, "predictions.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    added = log_predictions(out_matches)
    print("\n=== Saved %d matches (%d new rows in predictions_log.csv) ===" % (len(out_matches), added))
    if skipped:
        print("\nSKIPPED (%d) - shown so nothing is hidden:" % len(skipped))
        for s in skipped:
            print("  -", s)
    for w in warnings:
        print("WARNING:", w)


if __name__ == "__main__":
    run_automated()
        
