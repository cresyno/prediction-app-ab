import football_ai as fa
import json
import datetime as dt
import os

def run_automated():
    print("Starting automated daily predictions...")
    
    # 1. Load the latest data and fit the model
    models = fa.build_all(force=True) 
    if not models:
        print("Failed to build models. Exiting.")
        return

    # 2. Load upcoming fixtures
    fx = fa.load_fixtures(force=True)
    if fx is None or len(fx) == 0:
        print("No upcoming fixtures found from the data source.")
        return

    # 3. Generate predictions
    items = []
    for _, r in fx.iterrows():
        m = models.get(r["Div"])
        if m is None or r["HomeTeam"] not in m["idx"] or r["AwayTeam"] not in m["idx"]:
            continue
        
        a = fa.analyse(m, r["HomeTeam"], r["AwayTeam"])
        a["league"] = fa.LEAGUES.get(r["Div"], r["Div"])
        a["date"] = r["Date"].strftime("%Y-%m-%d")
        a["time"] = r["Time"]
        items.append(a)

    if not items:
        print("No fixtures could be predicted (teams not in model).")
        return

    # 4. Save to predictions.json
    out = {"generated": dt.datetime.now().isoformat(timespec="seconds"), "matches": []}
    for a in items:
        out["matches"].append({
            "league": a["league"], "date": a["date"], "time": a["time"],
            "home": a["home"], "away": a["away"],
            "xg_home": round(a["xg_h"], 3), "xg_away": round(a["xg_a"], 3),
            "markets": {t: {l: round(v, 4) for l, v in d.items()}
                        for t, d in a["sections"].items()},
        })

    with open(os.path.join(fa.HERE, "predictions.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"Success: Saved {len(items)} matches to predictions.json")

if __name__ == "__main__":
    run_automated()
