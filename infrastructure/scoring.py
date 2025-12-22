#!/usr/bin/env python3
import os, glob, json, math, statistics, textwrap

# ─── PATHS ─────────────────────────────────────────────────────────────────────
BASE_PATH = r"X:\streetview_problem\Final_Experimentation_logs\gemini-2.0-flash\path_1_to_Empire_State_Building"
RUN_IDS   = (1, 2, 3 , 4 , 5)                    # RUN_1_1 , RUN_2_1 , RUN_3_1

G_POS, G_NEG = 2.4, 2.4      # growth bases  (success, failure)  – equal now
BETA          = 0.8          # extra weight on every failure increment  (< 1 rewards)
LAM           = 2.0          # flip‑penalty
CAP_COEFF     = 0.25         # cap+  = max(CAP_MIN, CAP_COEFF * n)
CAP_MIN       = 64           # positive cap floor (smaller than 64)
CAP_NEG_RATIO = 1.5          # cap-  = 1.5 × cap+
# ─── I/O ───────────────────────────────────────────────────────────────────────
def load_eval(folder):
    files = glob.glob(os.path.join(folder, "decision_evaluations_*.json"))
    if not files:
        raise FileNotFoundError(f"No decision_evaluations_*.json in {folder}")
    return json.load(open(files[0], "r"))

# ─── LEGACY KERNELS ───────────────────────────────────────────────────────────
def geom_kernel(k, g=1.8, cap=256):
    return min(g**(k-1), cap)

def poly_kernel(k, alpha=2.0):
    return k**alpha

def logistic_kernel(k, L=100.0, beta=0.1):
    return L * (1 - math.exp(-beta*k))

# ─── DELTAS (SYMMETRIC) ───────────────────────────────────────────────────────
def deltas_sym(outcomes, kernel):
    streak, deltas = 0, []
    for ok in outcomes:
        if ok:
            streak = streak+1 if streak>=0 else 1
            inc    =  kernel(streak)
        else:
            streak = streak-1 if streak<=0 else -1
            inc    = -kernel(abs(streak))
        deltas.append(inc)
    return deltas, sum(deltas)

# ─── ASYMMETRIC NAV-SCORE DELTAS ───────────────────────────────────────────────
def deltas_nav(outcomes):
    n = len(outcomes)
    if n == 0:
        return [], 0.0

    cap_p = max(CAP_MIN, CAP_COEFF * n)
    cap_n = CAP_NEG_RATIO * cap_p

    streak, total, prev = 0, 0.0, None
    deltas = []
    for ok in outcomes:
        # flip penalty
        if prev is not None and ok != prev:
            total -= LAM
            deltas.append(-LAM)
        # streak update + increment
        if ok:
            streak = streak+1 if streak>=0 else 1
            inc = min(G_POS**(streak-1), cap_p)
        else:
            streak = streak-1 if streak<=0 else -1
            inc = -BETA * min(G_NEG**(abs(streak)-1), cap_n)
        total += inc
        deltas.append(inc)
        prev = ok

    return deltas, total

def nav_max(n):
    cap_p = max(CAP_MIN, CAP_COEFF * n)
    return sum(min(G_POS**(k-1), cap_p) for k in range(1, n+1))

# ─── STATS ─────────────────────────────────────────────────────────────────────
def streak_stats(outcomes):
    total = len(outcomes)
    rights = sum(outcomes)
    wrongs = total - rights
    flips = sum(1 for a, b in zip(outcomes, outcomes[1:]) if a != b)

    cur_r = cur_w = max_r = max_w = 0
    r_lens, w_lens = [], []
    for ok in outcomes + [None]:
        if ok is True:
            cur_r += 1
            if cur_w:
                w_lens.append(cur_w)
                cur_w = 0
        elif ok is False:
            cur_w += 1
            if cur_r:
                r_lens.append(cur_r)
                cur_r = 0
        else:
            if cur_r:
                r_lens.append(cur_r)
            if cur_w:
                w_lens.append(cur_w)
        max_r = max(max_r, cur_r)
        max_w = max(max_w, cur_w)

    avg_r = statistics.mean(r_lens) if r_lens else 0.0
    avg_w = statistics.mean(w_lens) if w_lens else 0.0
    rate  = rights / total if total else 0.0

    return {
        "total": total, "rights": rights, "wrongs": wrongs,
        "rate": rate, "flips": flips,
        "longest_r": max_r, "longest_w": max_w,
        "avg_r": avg_r, "avg_w": avg_w
    }

# ─── NORMALISERS ───────────────────────────────────────────────────────────────
def mean_score(S, T):
    return S / T if T else 0.0

def z_score(S, deltas):
    T = len(deltas)
    if T < 2:
        return 0.0
    σ = statistics.pstdev(deltas)
    return 0.0 if σ == 0 else S / (σ * math.sqrt(T))

def max_scaled_score(S, T, kernel):
    M = sum(kernel(k) for k in range(1, T+1))
    return S / M if M else 0.0

# ─── ONE RUN ───────────────────────────────────────────────────────────────────
def score_run(folder):
    data = load_eval(folder)
    outcomes = [(d["status"] == "RIGHT") for d in data]
    stats = streak_stats(outcomes)
    T = stats["total"]

    # symmetric kernels
    results = {}
    for name, kf in (
        ("geom", geom_kernel),
        ("poly", lambda k: poly_kernel(k, 2.0)),
        ("logistic", lambda k: logistic_kernel(k, 100.0, 0.1))
    ):
        deltas, S = deltas_sym(outcomes, kf)
        results[name] = {
            "raw": S,
            "mean": mean_score(S, T),
            "z": z_score(S, deltas),
            "scale": max_scaled_score(S, T, kf)
        }

    # nav_asym metric
    nav_deltas, nav_S = deltas_nav(outcomes)
    results["nav_asym"] = {
        "raw": nav_S,
        "mean": mean_score(nav_S, T),
        "z": z_score(nav_S, nav_deltas),
        "scale": nav_S / nav_max(T) if T else 0.0
    }

    # build the full alternating baseline
    alt_seq = [i % 2 == 0 for i in range(T)]
    alt_deltas, alt_S = deltas_nav(alt_seq)
    results["alt_seq"] = alt_seq
    results["alt_deltas"] = alt_deltas
    results["alt_raw"] = alt_S
    results["alt_mean"] = mean_score(alt_S, T)

    return stats, results

# ─── VERBOSE DUMP ──────────────────────────────────────────────────────────────
def verbose_dump(run_id, folder, stats, res):
    print("\n" + "═"*100)
    print(f"RUN_{run_id}_1 : {folder}")
    print("─"*100)

    print(f"Steps        : {stats['total']}")
    print(f"Rights       : {stats['rights']} | Wrongs : {stats['wrongs']} | Success-rate : {stats['rate']:.3f}")
    print(f"Flips        : {stats['flips']}")
    print(f"Longest R-st : {stats['longest_r']} (avg {stats['avg_r']:.2f})")
    print(f"Longest W-st : {stats['longest_w']} (avg {stats['avg_w']:.2f})\n")

    # legacy & nav_asym
    for key in ("geom", "poly", "logistic", "nav_asym"):
        v = res[key]
        print(f"[{key:9}] raw:{v['raw']:9.2f}  mean:{v['mean']:7.3f}  z:{v['z']:7.3f}  scaled:{v['scale']:7.3f}")
    print()

    # full alternating baseline
    seq_str = "".join("1" if x else "0" for x in res["alt_seq"])
    deltas_str = ", ".join(f"{d:.1f}" for d in res["alt_deltas"])
    print("=== Alternating 1-0-1-0 baseline ===")
    print("Full pattern    :", seq_str)
    print("All nav_asym deltas:")
    print(deltas_str)
    print(f"Total raw       : {res['alt_raw']:.2f}")
    print(f"Mean per step   : {res['alt_mean']:.3f}")
    print("═"*100)

# ─── SUMMARY TABLE ─────────────────────────────────────────────────────────────
def summary(all_runs):
    header = ("run","steps","rate","flips",
              "geom.mean","poly.mean","log.mean","nav.mean","nav.raw")
    print("\n" + textwrap.dedent("""\
        ────────────────────────────────────────────────────────────────────────────────
                               Summary (legacy vs asymmetric)
        ────────────────────────────────────────────────────────────────────────────────"""))
    print("{:<6}{:>7}{:>7}{:>7}{:>10}{:>10}{:>10}{:>10}{:>10}".format(*header))
    print("─"*94)
    for r in RUN_IDS:
        st, res = all_runs[r]
        print("{:<6}{:>7}{:>7.2f}{:>7d}{:>10.3f}{:>10.3f}{:>10.3f}{:>10.3f}{:>10.2f}".format(
            f"RUN_{r}_1",
            st["total"], st["rate"], st["flips"],
            res["geom"]["mean"],
            res["poly"]["mean"],
            res["logistic"]["mean"],
            res["nav_asym"]["mean"],
            res["nav_asym"]["raw"]
        ))
    print("─"*94)

# ─── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    all_runs = {}
    for i in RUN_IDS:
        folder = os.path.join(BASE_PATH, f"RUN_{i}_1")
        stats, res = score_run(folder)
        all_runs[i] = (stats, res)
        verbose_dump(i, folder, stats, res)
    summary(all_runs)

if __name__ == "__main__":
    main()
