import football_ai as fa
import json, datetime as dt, os

def run_automated():
    print("=== STARTING ROBOT ===")
    print("Building models (downloading historical data)...")
    models = fa.build_all(force=True)
    print(f"Models built for: {list(models.keys())}")
    
    if not models:
        print("FATAL ERROR: No models built. Data download failed.")
        return

    print("Loading fixtures from API...")
    fx = fa.load_fixtures(force=True)
    if fx is None or len(fx) == 0:
        print("FATAL ERROR: No fixtures found from API.")
        return
    print(f"Found {len(fx)} raw fixtures from API.")
    
    items = []
    for _, r in fx.iterrows():
        m = models.get(r["Div"])
        if m is None:
            continue
        if r["HomeTeam"] not in m["idx"] or r["AwayTeam"] not in m["idx"]:
            print(f"Skipping {r['HomeTeam']} vs {r['AwayTeam']}: Team name mismatch.")
            continue
        
        a = fa.analyse(m, r["HomeTeam"], r["AwayTeam"])
        a["league"] = fa.LEAGUES.get(r["Div"], r["Div"])
        a["date"] = r["Date"].strftime("%Y-%m-%d")
        a["time"] = r["Time"]
        items.append(a)

    print(f"Successfully generated predictions for {len(items)} matches.")
    
    if not items:
        print("FATAL ERROR: 0 matches predicted. Not saving JSON.")
        return

    out = {"generated": dt.datetime.now().isoformat(timespec="seconds"), "matches": []}
    for a in items:
        out["matches"].append({
            "league": a["league"], "date": a["date"], "time": a["time"],
            "home": a["home"], "away": a["away"],
            "xg_home": round(a["xg_h"], 2), "xg_away": round(a["xg_a"], 2),
            "markets": a["sections"] 
        })

    with open(os.path.join(fa.HERE, "predictions.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"=== SUCCESS: Saved {len(items)} matches to predictions.json ===")

if __name__ == "__main__":
    run_automated()
