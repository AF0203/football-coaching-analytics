# scripts/matching.py
"""
Trainer-Matching System
→ Drei Rankings: Fortführer | Erneuerer | Balance
→ Pool: Meta-Cluster + top 10 stilähnlichste aus anderen Clustern

Aufruf:
    python3 scripts/matching.py --team bl_44 --liga bundesliga --saison 2025_26
    python3 scripts/matching.py --team bl_44 --liga bundesliga --saison 2025_26 --top 5
"""

import sys, os, argparse, warnings, json
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
))
from config import PROCESSED_DIR, TEAM_PROFILES_DIR, TRAINER_DIR

TOP_PROFILES_DIR     = PROCESSED_DIR / "team_top_profiles"
TRAINER_PROFILES_DIR = PROCESSED_DIR / "trainer_profiles"

# ── Konfiguration ─────────────────────────────────────────────────

META_CLUSTER_POOL = {
    3: [3, 0],
    0: [0, 3],
    1: [1, 0],
    2: [2, 1],
}

META_CLUSTER_NAMES = {
    0: "Internationale Klasse",
    1: "Organisierte Mittelklasse",
    2: "Tief & Kontrolliert",
    3: "Elite",
}

# Anzahl Wildcard-Trainer aus anderen Clustern
WILDCARD_N = 10

# Drei Ranking-Profile
RANKINGS = {
    "fortführer": {
        "label":       "FORTFÜHRER",
        "description": "Trainer die Dortmunds besten Fußball fortführen können",
        "perf":        0.35,
        "style":       0.60,
        "structure":   0.05,
    },
    "erneuerer": {
        "label":       "ERNEUERER",
        "description": "Trainer die maximale Performance-Verbesserung bringen",
        "perf":        0.80,
        "style":       0.15,
        "structure":   0.05,
    },
    "balance": {
        "label":       "BALANCE",
        "description": "Bester Kompromiss aus Stil und Performance",
        "perf":        0.55,
        "style":       0.40,
        "structure":   0.05,
    },
}

# Performance Bonus/Malus
WEAKNESS_WEIGHT = 2.0
STRENGTH_MALUS  = 1.0

SEQUENCE_PROFILES = [
    "abstoß","aufbau","mf","prog","shot",
    "transition","gegenpressing","high_block",
    "mid_low_block","prog_against","shots_against",
    "aufbau_abbruch"
]

SEQUENCE_LABELS = {
    "abstoß":        "Abstoß / Torausstoss",
    "aufbau":        "Spielaufbau",
    "mf":            "Mittelfeld-Kombinationen",
    "prog":          "Progression ins letzte Drittel",
    "shot":          "Aktionen vor dem Schuss",
    "transition":    "Umschaltspiel",
    "gegenpressing": "Gegenpressing",
    "high_block":    "Hoher Pressing-Block",
    "mid_low_block": "Mittlerer/Tiefer Block",
    "prog_against":  "Gegner-Progression (zugelassen)",
    "shots_against": "Gegner-Abschlüsse (zugelassen)",
    "aufbau_abbruch":"Aufbau-Fehler",
}

PERFORMANCE_METRICS = [
    "deep_buildup_rate",
    "buildup_success_rate",
    "chance_threat_rate",
    "chance_quality",
    "chance_quantity_pg",
    "total_xg_pg",
    "chance_counter_rate",
    "avg_time_to_shot",
    "avg_ppda",
    "avg_time_to_recovery",
    "high_block_quality",
    "mid_block_quality",
    "low_block_quality",
    "xga_quality",
    "xga_total_pg",
    "shots_against_pg",
]

PERFORMANCE_LABELS = {
    "deep_buildup_rate":    "Tiefer Aufbau",
    "buildup_success_rate": "Aufbau-Erfolgsrate",
    "chance_threat_rate":   "Chancen-Bedrohung",
    "chance_quality":       "Chancenqualität (xG/Schuss)",
    "chance_quantity_pg":   "Schüsse pro Spiel",
    "total_xg_pg":          "xG pro Spiel",
    "chance_counter_rate":  "Konter-Effizienz",
    "avg_time_to_shot":     "Zeit zum Abschluss",
    "avg_ppda":             "Pressing-Intensität (PPDA)",
    "avg_time_to_recovery": "Ballrückeroberung",
    "high_block_quality":   "Hoher Block Qualität",
    "mid_block_quality":    "Mittlerer Block Qualität",
    "low_block_quality":    "Tiefer Block Qualität",
    "xga_quality":          "Gegner-Chancenqualität",
    "xga_total_pg":         "xGA pro Spiel",
    "shots_against_pg":     "Gegner-Schüsse pro Spiel",
}

INVERT_METRICS = {
    "avg_ppda",
    "avg_time_to_shot",
    "avg_time_to_recovery",
    "xga_quality",
    "xga_total_pg",
    "shots_against_pg",
}


# ══════════════════════════════════════════════════════════════════
#  DATEN LADEN
# ══════════════════════════════════════════════════════════════════

def load_team_season_profiles(team_id, liga):
    liga_dir = TEAM_PROFILES_DIR / liga
    profiles = {}
    for name in SEQUENCE_PROFILES + ["formation","network","performance","meta_cluster"]:
        path = liga_dir / f"profiles_{name}.csv"
        if path.exists():
            df  = pd.read_csv(path)
            row = df[df["teamId"]==team_id]
            if not row.empty:
                profiles[name] = row
    return profiles


def load_team_top_profiles(team_id):
    top_dir = TOP_PROFILES_DIR / team_id
    if not top_dir.exists():
        raise FileNotFoundError(f"Keine Top-Profile für {team_id}")
    profiles = {}
    for name in SEQUENCE_PROFILES + ["formation","network","performance","meta_cluster"]:
        path = top_dir / f"profiles_{name}.csv"
        if path.exists():
            profiles[name] = pd.read_csv(path)
    return profiles


def load_trainer_profiles():
    profiles = {}
    for name in SEQUENCE_PROFILES + ["formation","network","performance","meta_cluster"]:
        path = TRAINER_PROFILES_DIR / f"profiles_{name}.csv"
        if path.exists():
            profiles[name] = pd.read_csv(path)
    return profiles


def load_trainer_index():
    path = TRAINER_DIR / "trainer_index.csv"
    df   = pd.read_csv(path).drop_duplicates(subset=["trainer_id"])
    return df.set_index("trainer_id")


# ══════════════════════════════════════════════════════════════════
#  HILFSFUNKTIONEN
# ══════════════════════════════════════════════════════════════════

def get_row(profiles, entity_id, name):
    if name not in profiles:
        return None
    obj = profiles[name]
    if isinstance(obj, pd.DataFrame):
        if "teamId" in obj.columns:
            row = obj[obj["teamId"]==entity_id]
            return row if not row.empty else None
        return obj
    return obj


def get_val(profiles, entity_id, name, col):
    row = get_row(profiles, entity_id, name)
    if row is None or col not in row.columns:
        return np.nan
    v = row[col].iloc[0]
    return float(v) if pd.notna(v) else np.nan


def get_meta_cluster(profiles, entity_id):
    row = get_row(profiles, entity_id, "meta_cluster")
    if row is None:
        return None
    return int(row["meta_cluster"].iloc[0])


def get_all_meta_clusters(trainer_profiles):
    if "meta_cluster" not in trainer_profiles:
        return {}
    df = trainer_profiles["meta_cluster"]
    return dict(zip(df["teamId"], df["meta_cluster"].astype(int)))


def get_cluster(profiles, entity_id, name):
    row = get_row(profiles, entity_id, name)
    if row is None or "cluster" not in row.columns:
        return None
    return row["cluster"].iloc[0]


def get_zscore_vector(profiles, entity_id, profile_names):
    vector = []
    for name in profile_names:
        row = get_row(profiles, entity_id, name)
        if row is None:
            continue
        for col in sorted([c for c in row.columns if c.endswith("_zscore")]):
            v = row[col].iloc[0]
            vector.append(float(v) if pd.notna(v) else 0.0)
    return np.array(vector)


def euclidean_dist(v1, v2):
    n  = max(len(v1), len(v2))
    v1 = np.pad(v1, (0, n-len(v1)))
    v2 = np.pad(v2, (0, n-len(v2)))
    return np.sqrt(np.sum((v1-v2)**2))


def compute_similarity(v1, v2, global_max):
    dist = euclidean_dist(v1, v2)
    return round(max(0.0, 1.0 - dist/global_max) * 100, 1)


def get_perf_zscore(profiles, entity_id, metric):
    z_col = f"{metric}_zscore"
    val   = get_val(profiles, entity_id, "performance", z_col)
    if pd.isna(val):
        return np.nan
    return -val if metric in INVERT_METRICS else val


# ══════════════════════════════════════════════════════════════════
#  PERFORMANCE SCORE
# ══════════════════════════════════════════════════════════════════

def compute_raw_performance_score(team_top, team_id,
                                   trainer_profiles, tr_id):
    score   = 0.0
    details = []

    for metric in PERFORMANCE_METRICS:
        top_z = get_perf_zscore(team_top, team_id, metric)
        tr_z  = get_perf_zscore(trainer_profiles, tr_id, metric)

        if pd.isna(top_z) or pd.isna(tr_z):
            continue

        label = PERFORMANCE_LABELS.get(metric, metric)

        if top_z < -0.3:
            if tr_z > 0.3:
                b = WEAKNESS_WEIGHT * tr_z * abs(top_z)
                score += b
                details.append(("bonus", label, top_z, tr_z, b))
            elif tr_z > top_z:
                b = WEAKNESS_WEIGHT * 0.3 * (tr_z - top_z)
                score += b
                details.append(("small_bonus", label, top_z, tr_z, b))
        elif top_z > 0.5:
            if tr_z < top_z - 1.0:
                m = STRENGTH_MALUS * (top_z - tr_z) * 0.5
                score -= m
                details.append(("malus", label, top_z, tr_z, -m))

    return score, details


# ══════════════════════════════════════════════════════════════════
#  POOL-AUFBAU
# ══════════════════════════════════════════════════════════════════

def build_pool(team_mc, trainer_mcs, team_id,
               team_top, trainer_profiles, trainer_index,
               wildcard_n=10):
    """
    Pool = alle aus primärem Meta-Cluster
         + top N stilähnlichste aus anderen Clustern
    """
    rules   = META_CLUSTER_POOL.get(team_mc, [team_mc])
    primary = rules[0]

    # Team Top-Vektor
    tv = get_zscore_vector(team_top, team_id, SEQUENCE_PROFILES)

    # Alle Trainer mit Distanz berechnen
    all_trainers = list(trainer_mcs.keys())
    trainer_dists = []
    for tr in all_trainers:
        trv  = get_zscore_vector(trainer_profiles, tr, SEQUENCE_PROFILES)
        dist = euclidean_dist(tv, trv)
        trainer_dists.append((tr, dist, trainer_mcs[tr]))
    trainer_dists.sort(key=lambda x: x[1])

    # Primärer Pool — alle aus richtigem Meta-Cluster
    primary_pool = [t for t,d,mc in trainer_dists if mc==primary]
    print(f"  Primärer Cluster {primary} "
          f"({META_CLUSTER_NAMES.get(primary,'?')}): "
          f"{len(primary_pool)} Trainer")

    pool = set(primary_pool)

    # Wildcard — top N ähnlichste aus anderen Clustern
    wildcards = [
        t for t,d,mc in trainer_dists
        if mc != primary and t not in pool
    ][:wildcard_n]

    for tr in wildcards:
        pool.add(tr)

    print(f"  Wildcards (andere Cluster, top {wildcard_n} ähnlichste): "
          f"{len(wildcards)} Trainer")
    print(f"  Pool gesamt: {len(pool)} Trainer")

    return list(pool), tv, trainer_dists


# ══════════════════════════════════════════════════════════════════
#  SCORING
# ══════════════════════════════════════════════════════════════════

def score_pool(pool, team_id, tv, team_top, team_season,
               trainer_profiles, trainer_mcs, trainer_index,
               trainer_dists):
    """Berechnet alle Scores für alle Trainer im Pool"""

    # Distanz-Dict für schnellen Zugriff
    dist_dict = {t: d for t,d,mc in trainer_dists}

    # Global max Distanz im Pool für Normierung
    pool_dists = [dist_dict[t] for t in pool if t in dist_dict]
    global_max = max(pool_dists) if pool_dists else 1.0
    global_min = min(pool_dists) if pool_dists else 0.0

    # Performance Scores berechnen für Normierung
    raw_perf = {}
    perf_det = {}
    for tr in pool:
        raw, det = compute_raw_performance_score(
            team_top, team_id, trainer_profiles, tr
        )
        raw_perf[tr] = raw
        perf_det[tr] = det

    perf_min   = min(raw_perf.values())
    perf_max   = max(raw_perf.values())
    perf_range = perf_max - perf_min if perf_max > perf_min else 1.0

    team_fc = get_cluster(team_top, team_id, "formation")
    team_nc = get_cluster(team_top, team_id, "network")

    results = []
    for tr in pool:
        dist = dist_dict.get(tr, global_max)

        # Stil-Score 0-100 (normiert im Pool)
        if global_max > global_min:
            sim = round(
                max(0.0, 1.0 - (dist - global_min)/(global_max - global_min)) * 100,
                1
            )
        else:
            sim = 100.0

        # Performance-Score 0-100 (normiert im Pool)
        perf_norm = round(
            (raw_perf[tr] - perf_min) / perf_range * 100, 1
        )

        # Formation / Netzwerk
        tr_fc      = get_cluster(trainer_profiles, tr, "formation")
        tr_nc      = get_cluster(trainer_profiles, tr, "network")
        form_match = (team_fc is not None and tr_fc is not None
                      and tr_fc == team_fc)
        net_match  = (team_nc is not None and tr_nc is not None
                      and tr_nc == team_nc)
        struct_score = (
            (100.0 if form_match else 0.0) * 0.5 +
            (100.0 if net_match  else 0.0) * 0.5
        )

        # Trainer Info
        if tr in trainer_index.index:
            info = trainer_index.loc[tr]
            name = info["trainer_name"]
            club = info["verein"]
            liga = info["liga_id"]
        else:
            name = tr
            club = ""
            liga = ""

        # Scores für alle drei Rankings berechnen
        scores = {}
        for key, cfg in RANKINGS.items():
            scores[key] = round(
                perf_norm   * cfg["perf"]      +
                sim         * cfg["style"]     +
                struct_score* cfg["structure"],
                2
            )

        results.append({
            "trainer_id":   tr,
            "name":         name,
            "club":         club,
            "liga":         liga,
            "meta_cluster": trainer_mcs.get(tr, -1),
            "is_wildcard":  trainer_mcs.get(tr, -1) not in
                            META_CLUSTER_POOL.get(
                                get_meta_cluster(
                                    {"meta_cluster": pd.DataFrame([{
                                        "teamId": team_id,
                                        "meta_cluster": list(META_CLUSTER_POOL.keys())[0]
                                    }])},
                                    team_id
                                ) or 0, []
                            ),
            "similarity":   sim,
            "perf_raw":     round(raw_perf[tr], 2),
            "perf_norm":    perf_norm,
            "form_match":   form_match,
            "net_match":    net_match,
            "form_cluster": tr_fc,
            "net_cluster":  tr_nc,
            "struct_score": struct_score,
            "scores":       scores,
            "perf_details": perf_det[tr],
        })

    return results


# ══════════════════════════════════════════════════════════════════
#  OUTPUT
# ══════════════════════════════════════════════════════════════════

def print_trainer_report(rank, r, team_id, team_season,
                          team_top, trainer_profiles,
                          global_max, team_mc, ranking_key):
    tr_id  = r["trainer_id"]
    mc     = META_CLUSTER_NAMES.get(r["meta_cluster"],"?")
    mc_t   = META_CLUSTER_NAMES.get(team_mc,"?")
    mc_ok  = "✅" if r["meta_cluster"]==team_mc else "⚠️ "
    cfg    = RANKINGS[ranking_key]

    print(f"\n{'━'*65}")
    print(f"  #{rank:2d}  {r['name'].upper()}")
    print(f"       {r['club']} | {r['liga']}")
    print(f"{'━'*65}")
    print(f"  GESAMT-SCORE: {r['scores'][ranking_key]:.1f}/100")
    print(f"    Performance: {r['perf_norm']:.1f}/100 "
          f"(×{cfg['perf']}) | "
          f"Stil: {r['similarity']:.1f}/100 (×{cfg['style']})")
    f_str = "✅ Match" if r["form_match"] else "❌ kein Match"
    n_str = "✅ Match" if r["net_match"]  else "❌ kein Match"
    print(f"    Formation: {f_str} | Netzwerk: {n_str}")

    # Meta-Cluster
    print(f"\n  META-CLUSTER:")
    print(f"    Verein (Saison): {mc_t}")
    print(f"    Verein (Top):    {mc_t}")
    print(f"    Trainer:         {mc}  {mc_ok}")

    # Struktur
    team_fc_s = get_cluster(team_season, team_id, "formation")
    team_nc_s = get_cluster(team_season, team_id, "network")
    team_fc   = get_cluster(team_top,    team_id, "formation")
    team_nc   = get_cluster(team_top,    team_id, "network")
    fc_ok     = "✅" if r["form_match"] else "❌"
    nc_ok     = "✅" if r["net_match"]  else "❌"

    print(f"\n  STRUKTUR:")
    print(f"    Formation: "
          f"Saison={team_fc_s} | Top={team_fc} | "
          f"Trainer={r['form_cluster']}  {fc_ok}")
    print(f"    Netzwerk:  "
          f"Saison={team_nc_s} | Top={team_nc} | "
          f"Trainer={r['net_cluster']}  {nc_ok}")

    # Taktische Ähnlichkeit
    print(f"\n  TAKTISCHE ÄHNLICHKEIT (Trainer vs Saison | vs Top):")
    for prof in SEQUENCE_PROFILES:
        label = SEQUENCE_LABELS.get(prof, prof)
        sv    = get_zscore_vector(team_season, team_id, [prof])
        tv    = get_zscore_vector(team_top,    team_id, [prof])
        trv   = get_zscore_vector(trainer_profiles, tr_id, [prof])

        if len(sv)==0 or len(trv)==0:
            continue

        sim_s = compute_similarity(sv, trv, global_max)
        sim_t = compute_similarity(tv, trv, global_max) \
                if len(tv)>0 else None

        flag = ""
        if sim_t is not None:
            if sim_t >= 85:
                flag = " ← sehr ähnlich"
            elif sim_t <= 60:
                flag = " ← abweichend"

        sim_t_str = f"{sim_t}%" if sim_t is not None else "n/a"
        print(f"    {label:<38} {sim_s:>5}%  |  {sim_t_str:>5}{flag}")

    # Performance
    print(f"\n  PERFORMANCE (Z-Score: Saison | Top | Trainer | Bewertung):")
    for metric in PERFORMANCE_METRICS:
        label    = PERFORMANCE_LABELS.get(metric, metric)
        season_z = get_perf_zscore(team_season, team_id, metric)
        top_z    = get_perf_zscore(team_top,    team_id, metric)
        tr_z     = get_perf_zscore(trainer_profiles, tr_id, metric)

        if pd.isna(season_z) and pd.isna(top_z) and pd.isna(tr_z):
            continue

        s_str  = f"{season_z:+.2f}" if pd.notna(season_z) else "  n/a"
        t_str  = f"{top_z:+.2f}"   if pd.notna(top_z)    else "  n/a"
        tr_str = f"{tr_z:+.2f}"    if pd.notna(tr_z)     else "  n/a"

        if pd.notna(season_z) and pd.notna(tr_z):
            diff = tr_z - season_z
            if season_z < -0.3 and tr_z > 0.3:
                bew = "✅ verbessert"
            elif abs(diff) < 0.5:
                bew = "➡️  ähnlich"
            elif diff > 0.5:
                bew = "📈 besser"
            else:
                bew = "⚠️  schwächer"
        else:
            bew = ""

        print(f"    {label:<32} {s_str}  {t_str}  {tr_str}   {bew}")


def print_section(ranking_key, top_results, team_id, team_season,
                   team_top, trainer_profiles, global_max, team_mc,
                   top_n):
    cfg = RANKINGS[ranking_key]

    print(f"\n\n{'═'*65}")
    print(f"  {cfg['label']}")
    print(f"  {cfg['description']}")
    print(f"  Gewichtung: Performance {int(cfg['perf']*100)}% | "
          f"Stil {int(cfg['style']*100)}% | "
          f"Struktur {int(cfg['structure']*100)}%")
    print(f"{'═'*65}")

    # Übersichts-Tabelle
    print(f"\n  {'#':>2}  {'Name':<25} {'Score':>6}  "
          f"{'Perf':>6}  {'Stil':>5}  F  N  MC")
    print(f"  {'─'*62}")
    for i, r in enumerate(top_results, 1):
        f   = "✅" if r["form_match"] else "  "
        n   = "✅" if r["net_match"]  else "  "
        mc  = "✅" if r["meta_cluster"]==team_mc else "⚠️ "
        print(f"  {i:>2}  {r['name']:<25} "
              f"{r['scores'][ranking_key]:>6.1f}  "
              f"{r['perf_norm']:>6.1f}  "
              f"{r['similarity']:>5.1f}  {f}  {n}  {mc}")

    # Detailberichte
    print(f"\n{'─'*65}")
    print(f"  DETAILLIERTE BERICHTE")
    print(f"{'─'*65}")

    for i, r in enumerate(top_results, 1):
        print_trainer_report(
            i, r, team_id, team_season, team_top,
            trainer_profiles, global_max, team_mc, ranking_key
        )


# ══════════════════════════════════════════════════════════════════
#  JSON EXPORT
# ══════════════════════════════════════════════════════════════════

def build_trainer_json(r, team_id, team_season, team_top,
                        trainer_profiles, global_max, team_mc,
                        ranking_key, rank):
    tr_id = r["trainer_id"]
    cfg   = RANKINGS[ranking_key]

    # Taktische Ähnlichkeit
    taktisch = {}
    for prof in SEQUENCE_PROFILES:
        label = SEQUENCE_LABELS.get(prof, prof)
        sv    = get_zscore_vector(team_season, team_id, [prof])
        tv    = get_zscore_vector(team_top,    team_id, [prof])
        trv   = get_zscore_vector(trainer_profiles, tr_id, [prof])
        if len(sv) == 0 or len(trv) == 0:
            continue
        sim_s = compute_similarity(sv, trv, global_max)
        sim_t = compute_similarity(tv, trv, global_max) if len(tv) > 0 else None
        flag  = ""
        if sim_t is not None:
            if sim_t >= 85:
                flag = "sehr ähnlich"
            elif sim_t <= 60:
                flag = "abweichend"
        taktisch[prof] = {
            "label":     label,
            "vs_saison": sim_s,
            "vs_top":    sim_t,
            "flag":      flag,
        }

    # Performance
    performance = {}
    for metric in PERFORMANCE_METRICS:
        label    = PERFORMANCE_LABELS.get(metric, metric)
        season_z = get_perf_zscore(team_season, team_id, metric)
        top_z    = get_perf_zscore(team_top,    team_id, metric)
        tr_z     = get_perf_zscore(trainer_profiles, tr_id, metric)
        if pd.isna(season_z) and pd.isna(top_z) and pd.isna(tr_z):
            continue
        if pd.notna(season_z) and pd.notna(tr_z):
            diff = tr_z - season_z
            if season_z < -0.3 and tr_z > 0.3:
                bew = "verbessert"
            elif abs(diff) < 0.5:
                bew = "ähnlich"
            elif diff > 0.5:
                bew = "besser"
            else:
                bew = "schwächer"
        else:
            bew = ""
        performance[metric] = {
            "label":     label,
            "season":    round(float(season_z), 3) if pd.notna(season_z) else None,
            "top":       round(float(top_z),    3) if pd.notna(top_z)    else None,
            "trainer":   round(float(tr_z),     3) if pd.notna(tr_z)     else None,
            "bewertung": bew,
        }

    # Struktur
    team_fc_s = get_cluster(team_season, team_id, "formation")
    team_nc_s = get_cluster(team_season, team_id, "network")
    team_fc   = get_cluster(team_top,    team_id, "formation")
    team_nc   = get_cluster(team_top,    team_id, "network")

    return {
        "rank":         rank,
        "ranking_type": ranking_key,
        "trainer_id":   tr_id,
        "name":         r["name"],
        "club":         r["club"],
        "liga":         r["liga"],
        "scores": {
            "gesamt":      r["scores"][ranking_key],
            "performance": r["perf_norm"],
            "stil":        r["similarity"],
            "gewichtung": {
                "performance": cfg["perf"],
                "stil":        cfg["style"],
                "struktur":    cfg["structure"],
            }
        },
        "meta_cluster": {
            "verein_name":  META_CLUSTER_NAMES.get(team_mc, "?"),
            "trainer_name": META_CLUSTER_NAMES.get(r["meta_cluster"], "?"),
            "match":        r["meta_cluster"] == team_mc,
        },
        "struktur": {
            "formation": {
                "saison":  int(team_fc_s) if team_fc_s is not None else None,
                "top":     int(team_fc)   if team_fc   is not None else None,
                "trainer": int(r["form_cluster"]) if r["form_cluster"] is not None else None,
                "match":   r["form_match"],
            },
            "netzwerk": {
                "saison":  int(team_nc_s) if team_nc_s is not None else None,
                "top":     int(team_nc)   if team_nc   is not None else None,
                "trainer": int(r["net_cluster"]) if r["net_cluster"] is not None else None,
                "match":   r["net_match"],
            },
        },
        "taktische_ähnlichkeit": taktisch,
        "performance":           performance,
    }


def save_matching_results(results, team_id, liga, saison,
                           team_season, team_top, trainer_profiles,
                           global_max, team_mc):
    out_dir = PROCESSED_DIR / "matching_results" /               f"{team_id}_{liga}_{saison}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Speichere Ergebnisse...")
    for ranking_key in ["fortführer", "erneuerer", "balance"]:
        top_results = sorted(
            results,
            key=lambda x: x["scores"][ranking_key],
            reverse=True
        )
        output = []
        for rank, r in enumerate(top_results, 1):
            entry = build_trainer_json(
                r, team_id, team_season, team_top,
                trainer_profiles, global_max, team_mc,
                ranking_key, rank
            )
            output.append(entry)

        class NpEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.integer,)):
                    return int(obj)
                if isinstance(obj, (np.floating,)):
                    return float(obj)
                if isinstance(obj, (np.bool_,)):
                    return bool(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super().default(obj)

        out_file = out_dir / f"{ranking_key}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2, cls=NpEncoder)
        print(f"    ✅ {out_file.name} ({len(output)} Trainer)")

    print(f"  Gespeichert in: {out_dir}")


# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--team",       type=str, required=True)
    parser.add_argument("--liga",       type=str, required=True)
    parser.add_argument("--saison",     type=str, required=True)
    parser.add_argument("--top",        type=int, default=5)
    parser.add_argument("--wildcards",  type=int, default=10)
    args = parser.parse_args()

    print(f"\n{'═'*65}")
    print(f"  TRAINER-MATCHING: {args.team} | {args.liga} | {args.saison}")
    print(f"{'═'*65}")

    # Daten laden
    print("\n  Lade Profile...")
    team_season   = load_team_season_profiles(args.team, args.liga)
    team_top      = load_team_top_profiles(args.team)
    trainer_profs = load_trainer_profiles()
    trainer_index = load_trainer_index()

    team_mc = get_meta_cluster(team_top, args.team)
    tr_mcs  = get_all_meta_clusters(trainer_profs)

    if team_mc is None:
        print("❌ Kein Meta-Cluster!")
        return

    print(f"  Meta-Cluster: {team_mc} "
          f"({META_CLUSTER_NAMES.get(team_mc,'?')})")

    # Pool aufbauen
    print("\n  Baue Pool...")
    pool, tv, trainer_dists = build_pool(
        team_mc, tr_mcs, args.team,
        team_top, trainer_profs, trainer_index,
        wildcard_n=args.wildcards
    )

    # Global max für Ähnlichkeits-Darstellung im Output
    pool_dists = [d for t,d,mc in trainer_dists if t in pool]
    global_max = max(pool_dists) if pool_dists else 1.0

    # Scoring
    print("\n  Berechne Scores...")
    results = score_pool(
        pool, args.team, tv, team_top, team_season,
        trainer_profs, tr_mcs, trainer_index, trainer_dists
    )

    # Header
    print(f"\n{'═'*65}")
    print(f"  TRAINER-MATCHING ERGEBNIS")
    print(f"  Verein:      {args.team} ({META_CLUSTER_NAMES.get(team_mc,'?')})")
    print(f"  Pool:        {len(results)} Trainer")
    print(f"  Top N:       {args.top} pro Kategorie")
    print(f"{'═'*65}")

    # Drei Rankings ausgeben
    for key in ["fortführer", "erneuerer", "balance"]:
        cfg         = RANKINGS[key]
        top_results = sorted(
            results,
            key=lambda x: x["scores"][key],
            reverse=True
        )[:args.top]

        print_section(
            key, top_results, args.team,
            team_season, team_top, trainer_profs,
            global_max, team_mc, args.top
        )

    # Ergebnisse speichern
    save_matching_results(
        results, args.team, args.liga, args.saison,
        team_season, team_top, trainer_profs,
        global_max, team_mc
    )

    print(f"\n{'═'*65}")
    print(f"  FERTIG")
    print(f"{'═'*65}\n")


if __name__ == "__main__":
    main()