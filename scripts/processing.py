# scripts/processing.py
"""
Data Processing Pipeline
Rohdaten → saubere Master DB pro Liga

Änderungen v2:
- Carries nur wenn selber Spieler
- Sequenzen strikt unterbrochen
- Standards behalten als eigene Events
"""

import os, sys
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
))
from config import LIGEN_DIR, PROCESSED_DIR, SAISON_CLEAN_DIR

# ── Konstanten ────────────────────────────────────────────────────

RELEVANT_TYPES = [
    "Pass", "Shot", "TakeOn",
    "BallRecovery", "Tackle", "Interception",
    "Clearance", "BlockedPass", "BallTouch",
    "Challenge", "Aerial", "Foul",
    "KeeperPickup", "KeeperSweeper",
    # Standards
    "GoalKick", "ThrowIn", "CornerAwarded",
    "FreekickTaken",
    # Sonstiges
    "OffsideProwl", "Card",
    "SubstitutionOn", "SubstitutionOff",
    "Start", "End",
]

SHOT_TYPES = [
    "SavedShot", "MissedShots",
    "ShotOnPost", "Goal", "BlockedShot"
]

# Events die eine Sequenz HART beenden
HARD_SEQUENCE_BREAK = [
    "Foul",
    "KeeperPickup",
    "ThrowIn",
    "CornerAwarded",
    "FreekickTaken",
    "OffsideProwl",
    "Card",
    "Start",
    "End",
    "SubstitutionOn",
    "SubstitutionOff",
]

# Events die Ballbesitz-Wechsel bedeuten
BALL_WIN_TYPES = [
    "Tackle",
    "Interception",
    "BallRecovery",
    "KeeperPickup",
    "KeeperSweeper",
]

COLS_KEEP = [
    "matchId", "teamId", "playerId", "playerName",
    "type", "outcomeType",
    "x", "y", "endX", "endY",
    "period", "event_seconds",
    "defensiveThird", "sequence_id",
    "is_standard", "sequence_type",
    "liga", "saison",
]

LIGA_PREFIXES = {
    "bundesliga":     "bl",
    "bundesliga2":    "bl2",
    "premier_league": "pl",
    "la_liga":        "ll",
    "serie_a":        "sa",
    "ligue_1":        "l1",
    "eredivisie":     "ed",
}


# ══════════════════════════════════════════════════════════════════
#  HAUPT-FUNKTION
# ══════════════════════════════════════════════════════════════════

def prepare_db(df, liga, saison):
    """
    Komplette Processing Pipeline v2:
    1. Types filtern + vereinheitlichen
    2. Koordinaten normalisieren
    3. event_seconds berechnen
    4. Carries hinzufügen (nur selber Spieler)
    5. Sequenzen bauen (strikt)
    6. matchId/teamId mit Liga-Prefix
    7. Spalten bereinigen
    """
    print(f"\n  Verarbeite {liga} {saison}...")
    print(f"  Input: {len(df):,} Events")

    df = df.copy()

    # ── 1. Spalten sicherstellen ──────────────────────────────────
    for col in [
        "x","y","endX","endY","minute","second",
        "period","defensiveThird","matchId","teamId",
        "playerId","playerName","type","outcomeType"
    ]:
        if col not in df.columns:
            df[col] = np.nan

    df["liga"]   = liga
    df["saison"] = saison

    # ── 2. Shot Types vereinheitlichen ───────────────────────────
    mask_goal = df["type"] == "Goal"
    mask_shot = df["type"].isin([
        "SavedShot","MissedShots",
        "ShotOnPost","BlockedShot"
    ])
    df.loc[mask_goal, "type"]        = "Shot"
    df.loc[mask_goal, "outcomeType"] = "Successful"
    df.loc[mask_shot, "type"]        = "Shot"
    df.loc[mask_shot, "outcomeType"] = "Unsuccessful"

    # ── 3. Standards markieren ────────────────────────────────────
    standard_types = [
        "KeeperPickup","ThrowIn","CornerAwarded","FreekickTaken"
    ]
    df["is_standard"] = df["type"].isin(standard_types).astype(int)

    # ── 4. Irrelevante Types filtern ──────────────────────────────
    df = df[
        df["type"].isin(RELEVANT_TYPES + ["Shot"])
    ].copy()

    # ── 5. event_seconds berechnen ───────────────────────────────
    df["minute"] = pd.to_numeric(
        df["minute"], errors="coerce"
    ).fillna(0)
    df["second"] = pd.to_numeric(
        df["second"], errors="coerce"
    ).fillna(0)
    df["event_seconds"] = df["minute"] * 60 + df["second"]

    # ── 6. Koordinaten normalisieren ─────────────────────────────
    df = _normalize_coordinates(df)

    # ── 7. Defensiv-Third Flag ───────────────────────────────────
    df["defensiveThird"] = (df["x"] <= 33).astype(int)

    # ── 8. Liga-Prefix ────────────────────────────────────────────
    prefix = LIGA_PREFIXES.get(liga, liga[:3])
    df["matchId"] = prefix + "_" + df["matchId"].astype(str)
    df["teamId"]  = prefix + "_" + df["teamId"].astype(str)

    # ── 9. Sortieren ─────────────────────────────────────────────
    df = df.sort_values([
        "matchId", "period", "event_seconds"
    ]).reset_index(drop=True)

    # ── 10. Carries hinzufügen (v2) ───────────────────────────────
    df = _add_carries_v2(df)

    # ── 11. Sequenzen bauen (v2) ──────────────────────────────────
    df = _build_sequences_v2(df)

    # ── 12. Spalten bereinigen ────────────────────────────────────
    df = df[
        [c for c in COLS_KEEP if c in df.columns]
    ].copy()

    for col in ["x","y","endX","endY"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    print(f"  Output:    {len(df):,} Events")
    print(f"  Spiele:    {df['matchId'].nunique():,}")
    print(f"  Teams:     {df['teamId'].nunique():,}")
    print(f"  Sequenzen: {df['sequence_id'].nunique():,}")

    return df


# ══════════════════════════════════════════════════════════════════
#  KOORDINATEN NORMALISIERUNG
# ══════════════════════════════════════════════════════════════════

def _normalize_coordinates(df):
    """
    Alle Teams spielen von links nach rechts
    Prüft über defensiveThird Flag
    """
    df = df.copy()
    for col in ["x","y","endX","endY"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for (match_id, team_id), group in df.groupby(
        ["matchId","teamId"]
    ):
        de = group[group["defensiveThird"] == 1]
        if len(de) > 0 and de["x"].mean() > 50:
            mask = (
                (df["matchId"] == match_id) &
                (df["teamId"]  == team_id)
            )
            for col in ["x","endX"]:
                df.loc[mask, col] = 100 - df.loc[mask, col]
            for col in ["y","endY"]:
                df.loc[mask, col] = 100 - df.loc[mask, col]

    return df


# ══════════════════════════════════════════════════════════════════
#  CARRIES V2 — nur selber Spieler
# ══════════════════════════════════════════════════════════════════

def _add_carries_v2(df):
    """
    Carry nur bauen wenn:
    → nächstes Event = selber Spieler
    → vorheriges Event NICHT Unsuccessful
    → Distanz zwischen 3m und 30m
    → Zeitlücke < 8 Sekunden
    → kein Hard Break Event dazwischen
    """
    df = df.sort_values([
        "matchId","period","event_seconds"
    ]).reset_index(drop=True)

    carries   = []
    n_skipped = 0

    for (match_id, period), group in df.groupby(
        ["matchId","period"]
    ):
        group = group.sort_values(
            "event_seconds"
        ).reset_index(drop=True)

        for i in range(len(group) - 1):
            curr  = group.iloc[i]
            next_ = group.iloc[i+1]

            # ── Checks ───────────────────────────────────────────

            # Selbes Team
            if curr["teamId"] != next_["teamId"]:
                n_skipped += 1
                continue

            # Selber Spieler
            if pd.isna(curr.get("playerId")) or \
               pd.isna(next_.get("playerId")):
                continue

            # Koordinaten vorhanden
            if pd.isna(curr["endX"]) or pd.isna(curr["endY"]):
                continue
            if pd.isna(next_["x"]) or pd.isna(next_["y"]):
                continue

            # Vorheriges Event nicht Unsuccessful
            if curr.get("outcomeType") == "Unsuccessful":
                n_skipped += 1
                continue

            # Hard Break Event
            if curr["type"] in HARD_SEQUENCE_BREAK:

                n_skipped += 1
                continue
            if next_["type"] in HARD_SEQUENCE_BREAK:
                n_skipped += 1
                continue

            # Distanz
            dist = np.sqrt(
                (next_["x"]    - curr["endX"])**2 +
                (next_["y"]    - curr["endY"])**2
            )
            if dist < 2 or dist > 60:
                continue

            # Zeitlücke
            time_gap = next_["event_seconds"] - curr["event_seconds"]
            if time_gap > 10 or time_gap < 0:
                continue

            # ── Carry bauen ───────────────────────────────────────
            carries.append({
                "matchId":       match_id,
                "teamId":        curr["teamId"],
                "playerId":      next_["playerId"],
                "playerName":    next_["playerName"],
                "type":          "Carry",
                "outcomeType":   "Successful",
                "x":             float(curr["endX"]),
                "y":             float(curr["endY"]),
                "endX":          float(next_["x"]),
                "endY":          float(next_["y"]),
                "period":        curr["period"],
                "event_seconds": float(curr["event_seconds"]) + 0.5,
                "defensiveThird":int(curr["endX"] <= 33),
                "is_standard":   0,
                "liga":          curr["liga"],
                "saison":        curr["saison"],
            })

    if carries:
        carries_df = pd.DataFrame(carries)
        df = pd.concat([df, carries_df], ignore_index=True)
        df = df.sort_values([
            "matchId","period","event_seconds"
        ]).reset_index(drop=True)

    print(f"  Carries hinzugefügt: {len(carries):,} "
          f"(übersprungen: {n_skipped:,})")

    return df


# ══════════════════════════════════════════════════════════════════
#  SEQUENZEN V2 — strikt
# ══════════════════════════════════════════════════════════════════

def _build_sequences_v2(df):
    """
    Sequenz endet wenn:

    HARD BREAK:
    → bestimmte Event-Typen (Foul, Einwurf etc.)
    → Team-Wechsel (Gegner hat Ball)
    → Halbzeit-Wechsel

    SOFT BREAK:
    → Zeitlücke > 10 Sekunden

    Jede Sequenz bekommt auch:
    → sequence_type: "open_play" oder "standard"
    """
    df = df.sort_values([
        "matchId","period","event_seconds"
    ]).reset_index(drop=True)

    sequence_ids   = []
    sequence_types = []
    seq_id         = 0
    current_type   = "open_play"

    prev_match  = None
    prev_team   = None
    prev_period = None
    prev_time   = None
    prev_type   = None

    for idx, row in df.iterrows():

        new_sequence = False

        # ── Hard Break Checks ─────────────────────────────────────

        # Neues Spiel
        if row["matchId"] != prev_match:
            new_sequence = True

        # Neue Halbzeit
        elif row["period"] != prev_period:
            new_sequence = True

        # Team-Wechsel = Gegner hat Ball
        elif row["teamId"] != prev_team:
            new_sequence = True

        # Hard Break Event-Typ
        elif prev_type in HARD_SEQUENCE_BREAK:
            new_sequence = True

        elif row["type"] in HARD_SEQUENCE_BREAK:
            new_sequence = True

        # Erfolgreicher Tackle/Interception = Ballgewinn
        elif (prev_type in BALL_WIN_TYPES and
              str(df.at[idx-1, "outcomeType"]) == "Successful"
              if idx > 0 else False):
            new_sequence = True

        # ── Soft Break ────────────────────────────────────────────
        elif (prev_time is not None and
              (row["event_seconds"] - prev_time) > 10):
            new_sequence = True

        # ── Neue Sequenz starten ──────────────────────────────────
        if new_sequence:
            seq_id += 1
            # Sequenz-Typ bestimmen
            if row["type"] in [
                "GoalKick","ThrowIn",
                "CornerAwarded","FreekickTaken"
            ]:
                current_type = "standard"
            else:
                current_type = "open_play"

        sequence_ids.append(seq_id)
        sequence_types.append(current_type)

        # State updaten
        prev_match  = row["matchId"]
        prev_team   = row["teamId"]
        prev_period = row["period"]
        prev_time   = row["event_seconds"]
        prev_type   = row["type"]

    df["sequence_id"]   = sequence_ids
    df["sequence_type"] = sequence_types

    # Stats
    n_seq      = df["sequence_id"].nunique()
    n_open     = df[
        df["sequence_type"] == "open_play"
    ]["sequence_id"].nunique()
    n_standard = df[
        df["sequence_type"] == "standard"
    ]["sequence_id"].nunique()

    print(f"  Sequenzen gesamt:   {n_seq:,}")
    print(f"  → Open Play:        {n_open:,}")
    print(f"  → Standards:        {n_standard:,}")

    return df


# ══════════════════════════════════════════════════════════════════
#  RUNNER — mit Saison-Auswahl
# ══════════════════════════════════════════════════════════════════

def process_all_ligen(saison_filter=None):
    """
    saison_filter: z.B. "2025_26" → nur diese Saison verarbeiten
                   None → alle Saisons verarbeiten
    """
    SAISON_CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    processed = []

    for liga in LIGA_PREFIXES.keys():
        liga_dir = LIGEN_DIR / liga
        if not liga_dir.exists():
            continue

        csv_files = [
            f for f in liga_dir.glob("*.csv")
            if not f.name.startswith("._")
            and not f.name.startswith(".")
        ]

        # ── Saison-Filter anwenden ────────────────────────────────
        if saison_filter:
            csv_files = [
                f for f in csv_files
                if saison_filter in f.stem
            ]

        if not csv_files:
            continue

        for csv_file in csv_files:
            saison = csv_file.stem.replace(f"{liga}_","")

            print(f"\n{'='*50}")
            print(f" {liga} | {saison}")
            print(f"{'='*50}")

            try:
                df = pd.read_csv(
                    csv_file,
                    low_memory  = False,
                    encoding    = "utf-8-sig"
                )
                df = prepare_db(df, liga, saison)

                out = SAISON_CLEAN_DIR / \
                    f"{liga}_{saison}_clean.csv"
                df.to_csv(out, index=False)
                print(f"  ✅ Gespeichert → {out.name}")

                processed.append({
                    "liga":    liga,
                    "saison":  saison,
                    "events":  len(df),
                    "spiele":  df["matchId"].nunique(),
                    "teams":   df["teamId"].nunique(),
                    "datei":   out.name
                })

            except Exception as e:
                print(f"  ❌ Fehler: {e}")
                import traceback
                traceback.print_exc()

    if processed:
        summary = pd.DataFrame(processed)
        summary.to_csv(
            SAISON_CLEAN_DIR / "processing_summary.csv",
            index=False
        )
        print(f"\n{'='*50}")
        print(" ZUSAMMENFASSUNG")
        print(f"{'='*50}")
        print(summary.to_string(index=False))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Processing Pipeline"
    )
    parser.add_argument(
        "--saison",
        type    = str,
        default = None,
        help    = "Saison filtern z.B. '2025_26'"
    )
    args = parser.parse_args()

    process_all_ligen(saison_filter=args.saison)