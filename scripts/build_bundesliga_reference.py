# scripts/build_liga_references.py
"""
Referenzen für alle Ligen berechnen
→ Premier League, La Liga, Bundesliga 2
→ gleiche Logik wie build_bl_reference.py

Aufruf:
    python scripts/build_liga_references.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys, os

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
))
from config import PROCESSED_DIR, TEAM_PROFILES_DIR

PROFILE_NAMES = [
    "abstoß", "aufbau", "mf", "prog", "shot",
    "transition", "gegenpressing", "high_block",
    "mid_low_block", "prog_against", "shots_against",
    "aufbau_abbruch", "pass_score", "formation",
    "network", "performance"
]

SKIP_COLS = {
    "teamId", "liga", "cluster", "n_seqs", "n_games",
    "n_passes", "matchId", "teamName", "seq_length"
}

# Liga → (Ordner, Output-Dateiname)
LIGAS = {
    "premier_league": ("premier_league", "pl_reference.csv"),
    "la_liga":        ("la_liga",        "laliga_reference.csv"),
    "bundesliga":   ("bundesliga",   "bl2_reference.csv"),
}


def build_reference(liga_folder, out_file):
    profiles_dir = TEAM_PROFILES_DIR / liga_folder
    if not profiles_dir.exists():
        print(f"  ⚠️  Ordner nicht gefunden: {profiles_dir}")
        return False

    reference_rows = []

    for name in PROFILE_NAMES:
        path = profiles_dir / f"profiles_{name}.csv"
        if not path.exists():
            continue

        df = pd.read_csv(path)
        numeric_cols = [
            c for c in df.columns
            if df[c].dtype in [np.float64, np.float32,
                               np.int64, np.int32]
            and c not in SKIP_COLS
        ]

        for col in numeric_cols:
            values = df[col].dropna()
            if len(values) < 2:
                continue
            reference_rows.append({
                "profile": name,
                "metric":  col,
                "bl_mean": values.mean(),
                "bl_std":  values.std() + 1e-6,
                "bl_min":  values.min(),
                "bl_max":  values.max(),
                "bl_n":    len(values)
            })

    if not reference_rows:
        print(f"  ⚠️  Keine Daten gefunden")
        return False

    ref_df   = pd.DataFrame(reference_rows)
    out_path = PROCESSED_DIR / out_file
    ref_df.to_csv(out_path, index=False)

    print(f"  ✅ {out_file}")
    print(f"     Profile:  {ref_df['profile'].nunique()}")
    print(f"     Metriken: {len(ref_df)}")
    return True


print("\n" + "="*55)
print("  LIGA-REFERENZEN BERECHNEN")
print("="*55)

for liga, (folder, outfile) in LIGAS.items():
    print(f"\n  → {liga}...")
    build_reference(folder, outfile)

print(f"\n{'='*55}")
print(f"  FERTIG")
print(f"{'='*55}\n")