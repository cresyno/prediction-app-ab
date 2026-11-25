"""
FOOTBALL AI - script 2: walk-forward BACKTEST
Save in the SAME folder as football_ai.py (script 1), then run.
It replays past matches: for each one, the model only sees EARLIER games.
"""
import os
import sys
import time

try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
except NameError:
    pass
import numpy as np
import pandas as pd

try:
    import football_ai as fa
except ImportError:
    print("Save this file in the SAME folder as football_ai.py, then run again.")
    raise SystemExit

# ------------------------- SETTINGS -------------------------
TEST_SEASONS = 3        # newest seasons to test on (current partial season counts)
HISTORY_SEASONS = 6     # seasons loaded in total (the rest = training history)
STEP_DAYS = 14          # refit the model every N days

# --- FIXED: TUNING CONFIGS ---
# Testing different XI (time decay) and SHRINK (regression) values
CONFIGS = [(0.0019, 0.5), (0.0019, 1.0), (0.0019, 4.0), (0.0010, 2.0), (0.0035, 2.0)]

EDGE = 0.05             # value bet when model prob x odds >= 1 + EDGE
MIN_TRAIN = 300         # matches needed before a fit is allowed
# ------------------------------------------------------------

fa.SEASONS_BACK = HISTORY_SEASONS
REAL_TODAY = fa.TODAY
KEEP = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]
BET_1X2 = [("B365H", "B365D", "B365A"), ("AvgH", "AvgD", "AvgA"), ("PSH", "PSD", "PSA")]
MK_1X2 = [("PSCH", "PSCD", "PSCA"), ("AvgCH", "AvgCD", "AvgCA"),
          ("B365CH", "B365CD", "B365CA")] + BET_1X2
BET_OU = [("B365>2.5", "B365<2.5"), ("Avg>2.5", "Avg<2.5"), ("P>2.5", "P<2.5")]
MK_OU = [("PC>2.5", "PC<2.5"), ("AvgC>2.5", "AvgC<2.5"),
         ("B365C>2.5", "B365C<2.5")] + BET_OU


def load_full(code):
    frames = []
    for i, s in enumerate(fa.season_codes()):
        raw = fa.get(fa.BASE_URL.format(s=s, c=code), "%s_%s.csv" % (code, s),
                     6 if i == 0 else None)
        if raw is None:
            continue
        df = fa.parse_csv(raw)
        if not all(c in df.columns for c in KEEP):
            continue
        df = df.copy()
        df["Date"] = fa.parse_dates(df["Date"])
        for c in ("FTHG", "FTAG"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=KEEP)
        df["season"] = s
        frames.append(df)
    if not frames:
        return None, []
    full = pd.concat(frames, ignore_index=True, sort=False)
    full = full.sort_values("Date").reset_index(drop=True)
    return full, fa.season_codes()[:TEST_SEASONS]


def pick_cols(df, options):
    for cols in options:
        if all(c in df.columns for c in cols):
            sub = df[list(cols)].apply(pd.to_numeric, errors="coerce")
            if sub.notna().all(axis=1).mean() >= 0.7:
                return cols
    return None


def num(r, cols, k):
    return pd.to_numeric(r[cols[k]], errors="coerce") if cols else np.nan


def record(a, r, code, b1, m1, bo, mo):
    s, h, w = a["sections"], a["home"], a["away"]
    p1, px, p2 = list(s["Result (1X2)"].values())
    tg, hg, ag = s["Total goals"], s[h + " goals"], s[w + " goals"]
    return {
        "league": code, "date": r["Date"], "home": h, "away": w,
        "gh": int(r["FTHG"]), "ga": int(r["FTAG"]),
        "p1": p1, "px": px, "p2": p2,
        "o15": tg["Over 1.5"], "o25": tg["Over 2.5"], "u25": tg["Under 2.5"],
        "o35": tg["Over 3.5"], "btts": s["Both teams to score"]["Yes"],
        "h05": hg["Over 0.5"], "a05": ag["Over 0.5"],
        "h15": hg["Over 1.5"], "a15": ag["Over 1.5"],
        "bk_h": num(r, b1, 0), "bk_d": num(r, b1, 1), "bk_a": num(r, b1, 2),
        "mk_h": num(r, m1, 0), "mk_d": num(r, m1, 1), "mk_a": num(r, m1, 2),
        "bk_o": num(r, bo, 0), "bk_u": num(r, bo, 1),
        "mk_o": num(r, mo, 0), "mk_u": num(r, mo, 1),
    }


def walk_forward(code, full, test_seasons, xi, sh):
    fa.XI = xi
    fa.SHRINK = sh
    test = full[full.season.isin(test_seasons)]
    b1, m1 = pick_cols(test, BET_1X2), pick_cols(test, MK_1X2)
    bo, mo = pick_cols(test, BET_OU), pick_cols(test, MK_OU)
    print("  odds: bet=%s market=%s | O/U bet=%s market=%s" %
          (b1 and b1[0], m1 and m1[0], bo and bo[0], mo and mo[0]))
    rows, skipped, step = [], 0, 0
    d, end = test.Date.min(), test.Date.max()
    n_steps = int((end - d).days / STEP_DAYS) + 1
    while d <= end:
        step += 1
        block = test[(test.Date >= d) & (test.Date < d + pd.Timedelta(days=STEP_DAYS))]
        train = full[full.Date < d]
        if len(block) and len(train) >= MIN_TRAIN:
            fa.TODAY = d                      # the model's "now" = start of window
            m = fa.fit_league(train[KEEP].astype({"FTHG": int, "FTAG": int}))
            for _, r in block.iterrows():
                h, w = r["HomeTeam"], r["AwayTeam"]
                if h not in m["idx"] or w not in m["idx"]:
                    skipped += 1
                    continue
                rows.append(record(fa.analyse(m, h, w), r, code, b1, m1, bo, mo))
        if step % 5 == 0 or step == n_steps:
            print("  window %d/%d" % (step, n_steps))
        d += pd.Timedelta(days=STEP_DAYS)
    fa.TODAY = REAL_TODAY
    return rows, skipped


# ------------------------- SCORING -------------------------
def ll(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def outcome(gh, ga):
    return np.where(gh > ga, 0, np.where(gh == ga, 1, 2))


def core_loss(R):
    gh, ga = R.gh.to_numpy(), R.ga.to_numpy()
    P = R[["p1", "px", "p2"]].to_numpy(float)
    l1 = -np.mean(np.log(np.clip(P[np.arange(len(R)), outcome(gh, ga)], 1e-6, 1)))
    return l1, ll(R.o25.to_numpy(), ((gh + ga) > 2).astype(float))


def calib(name, p, y):
    print("\n%s: predicted vs actual" % name)
    edges = [0, .5, .6, .7, .8, .9, 1.0001]
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi)
        if m.sum() >= 20:
            print("  %3.0f-%3.0f%%  pred %4.1f%%  actual %4.1f%%  n=%d" %
                  (lo * 100, min(hi, 1) * 100, p[m].mean() * 100, y[m].mean() * 100, m.sum()))


def roi_line(label, profit):
    n = len(profit)
    if n < 10:
        print("  %-20s only %d bets" % (label, n))
        return
    print("  %-20s %5d bets  win %2.0f%%  ROI %+5.1f%% (+/-%.1f)" %
          (label, n, (profit > 0).mean() * 100, profit.mean() * 100,
           profit.std() / np.sqrt(n) * 100))


def report(R):
    gh, ga = R.gh.to_numpy(), R.ga.to_numpy()
    tot = gh + ga
    ev = [
        ("Home win", R.p1.to_numpy(), gh > ga),
        ("Draw", R.px.to_numpy(), gh == ga),
        ("Away win", R.p2.to_numpy(), ga > gh),
        ("Total >1.5", R.o15.to_numpy(), tot > 1),
        ("Total >2.5", R.o25.to_numpy(), tot > 2),
        ("Total >3.5", R.o35.to_numpy(), tot > 3),
        ("BTTS yes", R.btts.to_numpy(), (gh > 0) & (ga > 0)),
        ("Team scores 1+", np.concatenate([R.h05.to_numpy(), R.a05.to_numpy()]),
         np.concatenate([gh > 0, ga > 0])),
        ("Team scores 2+", np.concatenate([R.h15.to_numpy(), R.a15.to_numpy()]),
         np.concatenate([gh > 1, ga > 1])),
    ]
    print("\n%d matches tested" % len(R))
    print("\nSCORECARD (log-loss, lower = better)")
    print("%-15s %6s %6s  %5s %5s" % ("Market", "Model", "Const", "Pred", "Real"))
    for name, p, y in ev:
        y = y.astype(float)
        print("%-15s %6.3f %6.3f  %4.0f%% %4.0f%%" %
              (name, ll(p, y), ll(np.full_like(p, y.mean()), y), p.mean() * 100, y.mean() * 100))
    print("Const = always guessing the average rate.")
    print("Model must beat Const to be useful.")
    for name, p, y in ev:
        if name in ("Team scores 1+", "Total >1.5", "Total >2.5", "BTTS yes"):
            calib(name, p, y.astype(float))

    print("\nMODEL vs BOOKMAKER MARKET (margin removed)")
    ok = R[["mk_h", "mk_d", "mk_a"]].notna().all(axis=1).to_numpy()
    if ok.sum() >= 100:
        X = R[ok]
        out = outcome(X.gh.to_numpy(), X.ga.to_numpy())
        imp = 1 / X[["mk_h", "mk_d", "mk_a"]].to_numpy(float)
        pm = imp / imp.sum(axis=1, keepdims=True)
        pmod = X[["p1", "px", "p2"]].to_numpy(float)
        ix = np.arange(len(X))
        fr = np.bincount(out, minlength=3) / len(out)
        print("1X2 (%d): model %.4f  market %.4f  const %.4f" % (
            len(X), -np.mean(np.log(np.clip(pmod[ix, out], 1e-6, 1))),
            -np.mean(np.log(pm[ix, out])), -np.mean(np.log(fr[out]))))
    else:
        print("1X2: no usable odds columns")
    ok = R[["mk_o", "mk_u"]].notna().all(axis=1).to_numpy()
    if ok.sum() >= 100:
        X = R[ok]
        y = ((X.gh + X.ga) > 2).to_numpy(float)
        io, iu = 1 / X.mk_o.to_numpy(float), 1 / X.mk_u.to_numpy(float)
        print("O/U 2.5 (%d): model %.4f  market %.4f  const %.4f" % (
            len(X), ll(X.o25.to_numpy(), y), ll(io / (io + iu), y),
            ll(np.full(len(y), y.mean()), y)))
    else:
        print("O/U 2.5: no usable odds columns")
    print("Market below Model = bookmaker is sharper.")

    print("\nVALUE-BET TEST (flat 1 unit, pre-match odds)")
    if (R[["bk_h", "bk_d", "bk_a"]].notna().all(axis=1).sum() < 100 and
            R[["bk_o", "bk_u"]].notna().all(axis=1).sum() < 100):
        print("  no usable odds columns in the data")
    ok = R[["bk_h", "bk_d", "bk_a"]].notna().all(axis=1).to_numpy()
    if ok.sum() >= 100:
        X = R[ok]
        out = outcome(X.gh.to_numpy(), X.ga.to_numpy())
        O = X[["bk_h", "bk_d", "bk_a"]].to_numpy(float)
        P = X[["p1", "px", "p2"]].to_numpy(float)
        W = np.zeros(O.shape, bool)
        W[np.arange(len(X)), out] = True
        pay = np.where(W, O - 1, -1.0)
        print("1X2:")
        roi_line("bet everything", pay.ravel())
        for e in (EDGE, 2 * EDGE):
            roi_line("model edge >= %d%%" % round(e * 100), pay[P * O >= 1 + e])
    ok = R[["bk_o", "bk_u"]].notna().all(axis=1).to_numpy()
    if ok.sum() >= 100:
        X = R[ok]
        over = ((X.gh + X.ga) > 2).to_numpy()
        O = X[["bk_o", "bk_u"]].to_numpy(float)
        P = np.column_stack([X.o25.to_numpy(float), X.u25.to_numpy(float)])
        W = np.column_stack([over, ~over])
        pay = np.where(W, O - 1, -1.0)
        print("Over/Under 2.5:")
        roi_line("bet everything", pay.ravel())
        for e in (EDGE, 2 * EDGE):
            roi_line("model edge >= %d%%" % round(e * 100), pay[P * O >= 1 + e])
    print("\n(+/-) is one standard error. If ROI is within")
    print("+/-2x that, it is indistinguishable from luck.")


def main():
    print("FOOTBALL AI - script 2: BACKTEST")
    data = {}
    for code in fa.ACTIVE:
        print("\nLoading %s..." % fa.LEAGUES.get(code, code))
        full, tests = load_full(code)
        if full is None or len(full) < MIN_TRAIN:
            print("  not enough data, skipping")
            continue
        data[code] = (full, tests)
    if not data:
        print("No data. Run script 1 first so files are saved.")
        return
    runs = []
    for xi, sh in CONFIGS:
        print("\n#### XI = %s  SHRINK = %s" % (xi, sh))
        t0, rows = time.time(), []
        for code, (full, tests) in data.items():
            print("%s:" % fa.LEAGUES.get(code, code))
            r, sk = walk_forward(code, full, tests, xi, sh)
            rows += r
            print("  %d matches predicted (%d skipped: no history)" % (len(r), sk))
        if not rows:
            print("Nothing to score.")
            return
        R = pd.DataFrame(rows)
        l1, lo = core_loss(R)
        print("XI %s SHRINK %s: 1X2 loss %.4f | O/U2.5 loss %.4f | %.0fs" %
              (xi, sh, l1, lo, time.time() - t0))
        runs.append((l1 + lo, xi, sh, l1, lo, R))
    runs.sort(key=lambda t: t[0])
    if len(runs) > 1:
        print("\n==== SETTINGS RANKING (lower loss = better) ====")
        for _, xi, sh, l1, lo, _R in runs:
            print("XI %-7s SHRINK %-4s  1X2 %.4f  O/U %.4f" % (xi, sh, l1, lo))
        print("Best: XI = %s, SHRINK = %s" % (runs[0][1], runs[0][2]))
        print("Put these in football_ai.py (top settings).")
        print("Tiny gaps (<0.002) are noise: keep defaults.")
    best = runs[0][5]
    best.to_csv(os.path.join(fa.HERE, "backtest_predictions.csv"), index=False)
    report(best)
    print("\nSaved backtest_predictions.csv")


if __name__ == "__main__":
    main()