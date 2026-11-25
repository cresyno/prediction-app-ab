import football_ai as fa
import json, datetime as dt, os

def run_automated():
    print("Starting automated daily predictions...")
    models = fa.build_all(force=True) 
    if not models: return

    fx = fa.load_fixtures(force=True)
    if fx is None or len(fx) == 0: return

    items = []
    for _, r in fx.iterrows():
        m = models.get(r["Div"])
        if m is None or r["HomeTeam"] not in m["idx"] or r["AwayTeam"] not in m["idx"]: continue
        a = fa.analyse(m, r["HomeTeam"], r["AwayTeam"])
        a["league"] = fa.LEAGUES.get(r["Div"], r["Div"])
        a["date"], a["time"] = r["Date"].strftime("%Y-%m-%d"), r["Time"]
        items.append(a)

    if not items: return

    out = {"generated": dt.datetime.now().isoformat(timespec="seconds"), "matches": []}
    for a in items:
        out["matches"].append({
            "league": a["league"], "date": a["date"], "time": a["time"],
            "home": a["home"], "away": a["away"],
            "xg_home": round(a["xg_h"], 2), "xg_away": round(a["xg_a"], 2),
            "markets": a["sections"] # Saves ALL the detailed markets now
        })

    with open(os.path.join(fa.HERE, "predictions.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"Success: Saved {len(items)} matches to predictions.json")

if __name__ == "__main__":
    run_automated()
