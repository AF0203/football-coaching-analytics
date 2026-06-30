# config.py
from pathlib import Path
import sys

# ── Festplatte ────────────────────────────────────────────────────
BASE_DIR = Path("/Volumes/Crucial X9/football-coaching")

if not BASE_DIR.exists():
    sys.exit("❌ Festplatte nicht verbunden! Bitte Crucial X9 anschließen.")

# ── Daten ─────────────────────────────────────────────────────────
DATA_DIR      = BASE_DIR / "data"
RAW_DIR       = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR    = BASE_DIR / "models"
OUTPUT_DIR    = BASE_DIR / "output"
SAISON_CLEAN_DIR  = PROCESSED_DIR / "saison_clean"
TEAM_PROFILES_DIR = PROCESSED_DIR / "team_profiles"

# ── Ligen ─────────────────────────────────────────────────────────
LIGEN_DIR = RAW_DIR / "ligen"
LIGEN = [
    "bundesliga",
    "bundesliga2",
    "premier_league",
    "la_liga",
]

# ── Trainer ───────────────────────────────────────────────────────
TRAINER_DIR            = RAW_DIR / "trainer"
TRAINER_INDEX          = TRAINER_DIR / "trainer_index.csv"
MIN_SPIELE_PRO_STATION = 15
STATION_WEIGHTS        = [0.7, 0.3]

# ── Scraper ───────────────────────────────────────────────────────
SCRAPER_REPO = BASE_DIR / "scraping" / "whoscored-scraper"

# ── Liga Faktoren ─────────────────────────────────────────────────
LIGA_FAKTOREN = {
    "bundesliga":      0.93,
    "bundesliga2":     0.80,
    "premier_league":  1.00,
    "la_liga":         0.95,
    "serie_a":         0.91,
    "ligue_1":         0.88,
    "eredivisie":      0.82,
}

# ── Ordner erstellen ──────────────────────────────────────────────
for d in [
    DATA_DIR, RAW_DIR, PROCESSED_DIR,
    MODELS_DIR, OUTPUT_DIR,
    TRAINER_DIR,
    SAISON_CLEAN_DIR,
    TEAM_PROFILES_DIR,
    *[LIGEN_DIR / liga for liga in LIGEN],
    *[TEAM_PROFILES_DIR / liga for liga in LIGEN], 
]:
    d.mkdir(parents=True, exist_ok=True)

print("✅ Config geladen — Festplatte verbunden!")