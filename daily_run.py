import football_ai as fa
import json, datetime as dt, os, difflib

def find_best_match(api_name, model_teams):
    # 1. Exact match
    if api_name in model_teams: return api_name
    
    # 2. Clean and check
    clean_name = api_name
    for suffix in [" FC", " AFC", " CF", " AC", " SC", " SV", " 04", "1900", "1901", "1907", "1909", "1913", " 07", " 29", " 65"]:
        clean_name = clean_name.replace(suffix, "")
    clean_name = clean_name.strip()
    if clean_name in model_teams: return clean_name
    
    # 3. Manual overrides for tricky cases
    overrides = {
        "Manchester City": "Man City", "Manchester United": "Man United",
        "Nottingham Forest": "Nott'm Forest", "Wolverhampton Wanderers": "Wolves",
        "Tottenham Hotspur": "Spurs", "Paris Saint-Germain": "Paris SG",
        "Paris": "Paris SG", "FC Bayern München": "Bayern Munich",
        "Borussia Mönchengladbach": "M'gladbach", "Club Atlético de Madrid": "Ath Madrid",
        "Athletic Club": "Ath Bilbao", "Real Sociedad de Fútbol": "Sociedad",
        "Real Betis Balompié": "Betis", "RC Celta de Vigo": "Celta",
        "Rayo Vallecano de Madrid": "Vallecano", "Deportivo Alavés": "Alaves",
        "FC Internazionale Milano": "Inter", "Sport Lisboa e Benfica": "Benfica",
        "Sporting Clube de Portugal": "Sporting CP", "Sporting Clube de Braga": "Sp Braga",
        "1. FC Köln": "FC Koln", "1. Köln": "FC Koln", "1. Union Berlin": "Union Berlin",
        "Eintracht Frankfurt": "Ein Frankfurt", "Borussia Dortmund": "Dortmund",
        "Bayer 04 Leverkusen": "Leverkusen", "Hamburger": "Hamburg",
        "Stade Rennais 1901": "Rennes", "Stade Brestois 29": "Brest",
        "RC Strasbourg Alsace": "Strasbourg", "Olympique de Marseille": "Marseille",
        "Olympique Lyonnais": "Lyon", "AS Monaco": "Monaco", "Lille OSC": "Lille",
        "ACF Fiorentina": "Fiorentina", "AS Roma": "Roma", "SS Lazio": "Lazio",
        "SSC Napoli": "Napoli", "AC Milan": "Milan", "US Lecce": "Lecce",
        "Cagliari Calcio": "Cagliari", "Torino": "Torino", "Udinese Calcio": "Udinese",
        "GenoaC": "Genoa", "Parma Calcio 1913": "Parma", "Frosinone Calcio": "Frosinone",
        "US Sassuolo Calcio": "Sassuolo", "Atalanta BC": "Atalanta", "AC Monza": "Monza",
        "Como 1907": "Como", "Venezia": "Venezia", "Bologna 1909": "Bologna",
        "Queens Park Rangers": "QPR", "West Bromwich Albion": "West Brom",
        "Blackburn Rovers": "Blackburn", "Bolton Wanderers": "Bolton",
        "Birmingham City": "Birmingham", "Stoke City": "Stoke", "Derby County": "Derby",
        "Leeds United": "Leeds", "Coventry City": "Coventry", "Preston North End": "Preston",
        "Norwich City": "Norwich", "Charlton Athletic": "Charlton", "Lincoln City": "Lincoln",
        "Swansea City": "Swansea", "Cardiff City": "Cardiff", "Hull City": "Hull",
        "Ipswich Town": "Ipswich", "Sunderland": "Sunderland", "Levante UD": "Levante",
        "Real Racing Club de Santander": "Santander", "Málaga": "Malaga",
        "RC Deportivo La Coruña": "La Coruna", "RCD Espanyol de Barcelona": "Espanyol",
        "Getafe CF": "Getafe", "PSV": "PSV", "AFC Ajax": "Ajax",
        "Feyenoord Rotterdam": "Feyenoord", "AZ": "AZ Alkmaar",
        "FC Twente '65": "Twente", "FC Utrecht": "Utrecht", "SC Heerenveen": "Heerenveen",
        "Sparta Rotterdam": "Sparta Rotterdam", "Go Ahead Eagles": "Go Ahead Eagles",
        "NEC": "Nijmegen", "PEC Zwolle": "Zwolle", "Fortuna Sittard": "Fortuna Sittard",
        "FC Groningen": "Groningen", "Willem II Tilburg": "Willem II",
        "SBV Excelsior": "Excelsior", "Telstar 1963": "Telstar",
        "SC Cambuur-Leeuwarden": "Cambuur", "ADO Den Haag": "Den Haag",
        "Moreirense": "Moreirense", "FC Arouca": "Arouca", "Rio Ave": "Rio Ave",
        "Gil Vicente": "Gil Vicente", "GD Estoril Praia": "Estoril",
        "CD Santa Clara": "Santa Clara", "CF Estrela da Amadora": "Amadora",
        "CS Marítimo": "Maritimo", "CD Nacional": "Nacional",
        "Académico de Viseu": "Academico Viseu", "FC Alverca": "Alverca",
        "FC Famalicão": "Famalicao"
    }
    
    if api_name in overrides: return overrides[api_name]
    if clean_name in overrides: return overrides[clean_name]
    
    # 4. Fuzzy fallback
    matches = difflib.get_close_matches(clean_name, model_teams, n=1, cutoff=0.6)
    if matches: return matches[0]
    
    return None

def run_automated():
    print("=== STARTING ROBOT ===")
    models = fa.build_all(force=True)
    if not models: return

    fx = fa.load_fixtures(force=True)
    if fx is None or len(fx) == 0: return
    
    items = []
    for _, r in fx.iterrows():
        m = models.get(r["Div"])
        if m is None: continue
        
        model_teams = list(m["idx"].keys())
        h_team = find_best_match(r["HomeTeam"], model_teams)
        a_team = find_best_match(r["AwayTeam"], model_teams)
        
        if not h_team or not a_team:
            print(f"Skipping {r['HomeTeam']} vs {r['AwayTeam']}: No match found.")
            continue
            
        a = fa.analyse(m, h_team, a_team)
        a["league"] = fa.LEAGUES.get(r["Div"], r["Div"])
        a["date"] = r["Date"].strftime("%Y-%m-%d")
        a["time"] = r["Time"]
        items.append(a)

    print(f"Successfully generated predictions for {len(items)} matches.")
    if not items: return

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
