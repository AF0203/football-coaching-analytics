# scraping/scrape_wechsel.py
"""
Wechsel-Scraper für ML-Datenbank
→ scrapped drei Teile pro Trainerwechsel:
   1. Verein: letzte 10 Spiele VOR Wechsel
   2. Trainer: letzte 10 Spiele VOR Wechsel
   3. Verein: erste 10 Spiele NACH Wechsel

→ speichert drei separate Raw-CSVs pro Wechsel:
   {wechsel_id}_verein_vorher_raw.csv
   {wechsel_id}_trainer_vorher_raw.csv
   {wechsel_id}_verein_nachher_raw.csv

Aufruf: python3 scraping/scrape_wechsel.py
"""

import sys, os, time, random
import pandas as pd
from pathlib import Path

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
))
from config import SCRAPER_REPO, PROCESSED_DIR

sys.path.insert(0, str(SCRAPER_REPO))
import main
from selenium import webdriver
from selenium.webdriver.firefox.options import Options

# ══════════════════════════════════════════════════════════════════
#  KONFIGURATION — hier anpassen pro Wechsel
# ══════════════════════════════════════════════════════════════════

WECHSEL_ID = "w_002"

# Liga des Vereins (bundesliga / bundesliga2 / laliga / premier_league)
LIGA_VEREIN = "bundesliga"

# Liga des Trainers (woher kommt er)
LIGA_TRAINER = "bundesliga2"

# team_id aus unserer DB (z.B. bl_219 für Mainz)
TEAM_ID = "bl_42"

# Whoscored interne team_id des Vereins
TEAM_ID_RAW_VEREIN = "42"

# Whoscored interne team_id des Trainers (sein vorheriger Verein)
TEAM_ID_RAW_TRAINER = "1150"

URLS_VEREIN_VORHER = [
    "https://www.whoscored.com/matches/1910744/live/germany-bundesliga-2025-2026-werder-bremen-wolfsburg",
    "https://www.whoscored.com/matches/1910758/live/germany-bundesliga-2025-2026-rb-leipzig-werder-bremen",
    "https://www.whoscored.com/matches/1910778/live/germany-bundesliga-2025-2026-werder-bremen-fc-koln",
    "https://www.whoscored.com/matches/1910786/live/germany-bundesliga-2025-2026-hamburger-sv-werder-bremen",
    "https://www.whoscored.com/matches/1910656/live/germany-bundesliga-2025-2026-werder-bremen-vfb-stuttgart",
    "https://www.whoscored.com/matches/1910658/live/germany-bundesliga-2025-2026-augsburg-werder-bremen",
    "https://www.whoscored.com/matches/1910797/live/germany-bundesliga-2025-2026-borussia-dortmund-werder-bremen",
    "https://www.whoscored.com/matches/1910812/live/germany-bundesliga-2025-2026-werder-bremen-eintracht-frankfurt",
    "https://www.whoscored.com/matches/1910814/live/germany-bundesliga-2025-2026-bayer-leverkusen-werder-bremen",
    "https://www.whoscored.com/matches/1910688/live/germany-bundesliga-2025-2026-werder-bremen-hoffenheim",
]

URLS_TRAINER_VORHER = [
    "https://www.whoscored.com/matches/1834827/live/germany-2-bundesliga-2024-2025-fortuna-duesseldorf-schalke-04",
    "https://www.whoscored.com/matches/1834841/live/germany-2-bundesliga-2024-2025-magdeburg-fortuna-duesseldorf",
    "https://www.whoscored.com/matches/1910904/live/germany-2-bundesliga-2025-2026-arminia-bielefeld-fortuna-duesseldorf",
    "https://www.whoscored.com/matches/1910917/live/germany-2-bundesliga-2025-2026-fortuna-duesseldorf-hannover-96",
    "https://www.whoscored.com/matches/1910927/live/germany-2-bundesliga-2025-2026-paderborn-fortuna-duesseldorf",
    "https://www.whoscored.com/matches/1910942/live/germany-2-bundesliga-2025-2026-fortuna-duesseldorf-karlsruher-sc",
    "https://www.whoscored.com/matches/1910972/live/germany-2-bundesliga-2025-2026-preussen-muenster-fortuna-duesseldorf",
    "https://www.whoscored.com/matches/1910979/live/germany-2-bundesliga-2025-2026-fortuna-duesseldorf-darmstadt",
    "https://www.whoscored.com/matches/1910913/live/germany-2-bundesliga-2025-2026-bochum-fortuna-duesseldorf",
    "https://www.whoscored.com/matches/1910945/live/germany-2-bundesliga-2025-2026-fortuna-duesseldorf-nuernberg",
]

URLS_VEREIN_NACHHER = [
    "https://www.whoscored.com/matches/1910835/live/germany-bundesliga-2025-2026-freiburg-werder-bremen",
    "https://www.whoscored.com/matches/1910715/live/germany-bundesliga-2025-2026-werder-bremen-bayern-munich",
    "https://www.whoscored.com/matches/1910729/live/germany-bundesliga-2025-2026-st-pauli-werder-bremen",
    "https://www.whoscored.com/matches/1910751/live/germany-bundesliga-2025-2026-werder-bremen-fc-heidenheim",
    "https://www.whoscored.com/matches/1910768/live/germany-bundesliga-2025-2026-union-berlin-werder-bremen",
    "https://www.whoscored.com/matches/1910789/live/germany-bundesliga-2025-2026-werder-bremen-mainz-05",
    "https://www.whoscored.com/matches/1910839/live/germany-bundesliga-2025-2026-wolfsburg-werder-bremen",
    "https://www.whoscored.com/matches/1910848/live/germany-bundesliga-2025-2026-werder-bremen-rb-leipzig",
    "https://www.whoscored.com/matches/1910851/live/germany-bundesliga-2025-2026-fc-koln-werder-bremen",
    "https://www.whoscored.com/matches/1910866/live/germany-bundesliga-2025-2026-werder-bremen-hamburger-sv",
]

# ══════════════════════════════════════════════════════════════════

# Output-Ordner
from config import RAW_DIR
ML_RAW_DIR = RAW_DIR / "wechsel"
ML_RAW_DIR.mkdir(parents=True, exist_ok=True)

WECHSEL_DIR = ML_RAW_DIR / WECHSEL_ID
WECHSEL_DIR.mkdir(parents=True, exist_ok=True)


def accept_cookies(driver):
    driver.execute_script("""
        const buttons = Array.from(document.querySelectorAll('button'));
        for (const b of buttons) {
            const txt = (b.innerText || '').toLowerCase().trim();
            if (txt.includes('accept') || txt.includes('akzeptieren') ||
                txt.includes('consent') || txt.includes('agree')) {
                b.click(); break;
            }
        }
    """)
    time.sleep(random.uniform(3, 5))


def scrape_spiele(driver, urls, team_id_raw, team_label, opp_label, part_name):
    """
    Scrapped eine Liste von Spielen.
    team_label:  wie wir das Haupt-Team nennen (z.B. "bl_219_vorher")
    opp_label:   wie wir den Gegner nennen    (z.B. "opp_bl_219_vorher")
    """
    all_events = []

    for i, url in enumerate(urls, 1):
        url = url.strip()
        if not url or "..." in url:
            print(f"  ⚠️  URL {i} nicht gesetzt — übersprungen")
            continue

        print(f"  Spiel {i}/{len(urls)}: {url.split('/')[-1]}")

        try:
            match_data = main.getMatchData(driver, url, close_window=False)
            df         = main.createEventsDF(match_data)

            # Listen-Spalten zu Strings
            for col in df.columns:
                if df[col].apply(lambda x: isinstance(x, list)).any():
                    df[col] = df[col].astype(str)

            # teamId ersetzen
            df["teamId"] = df["teamId"].apply(
                lambda x: team_label
                if str(x) == str(team_id_raw)
                else opp_label
            )

            df["liga_id"]    = LIGA_VEREIN if "verein" in part_name else LIGA_TRAINER
            df["wechsel_id"] = WECHSEL_ID
            df["part"]       = part_name
            df["source_url"] = url

            all_events.append(df)

            n_team = (df["teamId"] == team_label).sum()
            n_opp  = (df["teamId"] == opp_label).sum()
            print(f"    ✅ {len(df)} Events "
                  f"(Team: {n_team} | Gegner: {n_opp})")

            time.sleep(random.uniform(8, 12))

        except Exception as e:
            print(f"    ❌ Fehler: {e}")
            time.sleep(random.uniform(5, 10))

    return pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame()


def save_part(df, filename):
    out = WECHSEL_DIR / filename
    df.to_csv(out, index=False)
    print(f"\n  ✅ Gespeichert: {out.name}")
    print(f"     Events:  {len(df):,}")
    print(f"     Spiele:  {df['matchId'].nunique()}")


# ══════════════════════════════════════════════════════════════════
#  BROWSER STARTEN
# ══════════════════════════════════════════════════════════════════

print(f"\n{'='*60}")
print(f"  WECHSEL-SCRAPER: {WECHSEL_ID}")
print(f"  Verein: {TEAM_ID} ({LIGA_VEREIN})")
print(f"  Trainer kommt aus: {LIGA_TRAINER}")
print(f"{'='*60}")

options = Options()
driver  = webdriver.Firefox(options=options)
driver.get("https://www.whoscored.com/")
time.sleep(random.uniform(10, 15))
accept_cookies(driver)


# ══════════════════════════════════════════════════════════════════
#  TEIL 1 — VEREIN VORHER
# ══════════════════════════════════════════════════════════════════

print(f"\n{'─'*60}")
print(f"  TEIL 1 — Verein vorher ({TEAM_ID})")
print(f"{'─'*60}")

df_verein_vorher = scrape_spiele(
    driver      = driver,
    urls        = URLS_VEREIN_VORHER,
    team_id_raw = TEAM_ID_RAW_VEREIN,
    team_label  = f"{TEAM_ID}_vorher",
    opp_label   = f"opp_{TEAM_ID}_vorher",
    part_name   = "verein_vorher",
)

if not df_verein_vorher.empty:
    save_part(df_verein_vorher, "verein_vorher_raw.csv")

time.sleep(random.uniform(10, 15))


# ══════════════════════════════════════════════════════════════════
#  TEIL 2 — TRAINER VORHER
# ══════════════════════════════════════════════════════════════════

print(f"\n{'─'*60}")
print(f"  TEIL 2 — Trainer vorher ({LIGA_TRAINER})")
print(f"{'─'*60}")

df_trainer_vorher = scrape_spiele(
    driver      = driver,
    urls        = URLS_TRAINER_VORHER,
    team_id_raw = TEAM_ID_RAW_TRAINER,
    team_label  = f"trainer_{WECHSEL_ID}",
    opp_label   = f"opp_trainer_{WECHSEL_ID}",
    part_name   = "trainer_vorher",
)

if not df_trainer_vorher.empty:
    save_part(df_trainer_vorher, "trainer_vorher_raw.csv")

time.sleep(random.uniform(10, 15))


# ══════════════════════════════════════════════════════════════════
#  TEIL 3 — VEREIN NACHHER
# ══════════════════════════════════════════════════════════════════

print(f"\n{'─'*60}")
print(f"  TEIL 3 — Verein nachher ({TEAM_ID})")
print(f"{'─'*60}")

df_verein_nachher = scrape_spiele(
    driver      = driver,
    urls        = URLS_VEREIN_NACHHER,
    team_id_raw = TEAM_ID_RAW_VEREIN,
    team_label  = f"{TEAM_ID}_nachher",
    opp_label   = f"opp_{TEAM_ID}_nachher",
    part_name   = "verein_nachher",
)

if not df_verein_nachher.empty:
    save_part(df_verein_nachher, "verein_nachher_raw.csv")


# ══════════════════════════════════════════════════════════════════
#  ZUSAMMENFASSUNG
# ══════════════════════════════════════════════════════════════════

driver.quit()

print(f"\n{'='*60}")
print(f"  FERTIG — {WECHSEL_ID}")
print(f"  Gespeichert in: {WECHSEL_DIR}")
print(f"  Dateien:")
for f in sorted(WECHSEL_DIR.glob("*.csv")):
    df_tmp = pd.read_csv(f)
    print(f"    {f.name}: {len(df_tmp):,} Events, "
          f"{df_tmp['matchId'].nunique()} Spiele")
print(f"{'='*60}\n")