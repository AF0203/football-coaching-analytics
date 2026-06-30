# scripts/build_bl_reference.py
"""
Bundesliga-Referenz berechnen + speichern
→ einmalig ausführen nach build_profiles.py
→ wird für Trainer Z-Scores genutzt

Aufruf:
    python scripts/build_bl_reference.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

import sys, os
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
))
from config import PROCESSED_DIR, TEAM_PROFILES_DIR

# ── Alle Bundesliga-Profile laden ────────────────────────────────

BL_PROFILES_DIR = TEAM_PROFILES_DIR / "bundesliga"

profile_names = [
    "abstoß", "aufbau", "mf", "prog", "shot",
    "transition", "gegenpressing", "high_block",
    "mid_low_block", "prog_against", "shots_against",
    "aufbau_abbruch", "pass_score", "formation",
    "network", "performance"
]

# Spalten die keine Metriken sind
SKIP_COLS = {
    "teamId", "liga", "cluster", "n_seqs", "n_games",
    "n_passes", "matchId", "teamName", "seq_length"
}

reference_rows = []

for name in profile_names:
    path = BL_PROFILES_DIR / f"profiles_{name}.csv"
    if not path.exists():
        print(f"⚠️  {name} nicht gefunden — überspringe")
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

reference_df = pd.DataFrame(reference_rows)

# ── Speichern ─────────────────────────────────────────────────────
out_file = PROCESSED_DIR / "bl_reference.csv"
reference_df.to_csv(out_file, index=False)

print(f"\n✅ bl_reference.csv gespeichert")
print(f"   Profile:         {reference_df['profile'].nunique()}")
print(f"   Metriken gesamt: {len(reference_df)}")
print(f"\nÜbersicht:")
print(reference_df.groupby("profile")["metric"].count().to_string())