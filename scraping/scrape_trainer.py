# scraping/scrape_trainer.py
"""
Trainer-Scraper
→ eine Station (letzte 15-18 Spiele)
→ Trainer-Team UND Gegner werden gespeichert
→ Trainer-Team: teamId = trainer_id
→ Gegner-Team:  teamId = opp_{trainer_id}
→ liga_id wird manuell gesetzt

Aufruf: python3 scraping/scrape_trainer.py
"""

import sys, os, time
import pandas as pd
import random
from pathlib import Path

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
))
from config import SCRAPER_REPO, TRAINER_DIR

sys.path.insert(0, str(SCRAPER_REPO))
import main
from selenium import webdriver
from selenium.webdriver.firefox.options import Options

# ══════════════════════════════════════════════════════════════════
#  KONFIGURATION — hier anpassen pro Trainer
# ══════════════════════════════════════════════════════════════════

TRAINER_ID = "tr_006"

STATION = {
    "liga_id": "premier_league",
    "team_id": "167",
    "urls": [
        "https://www.whoscored.com/matches/1903222/live/england-premier-league-2025-2026-manchester-city-liverpool",
        "https://www.whoscored.com/matches/1903242/live/england-premier-league-2025-2026-manchester-city-leeds",
        "https://www.whoscored.com/matches/1903251/live/england-premier-league-2025-2026-fulham-manchester-city",
        "https://www.whoscored.com/matches/1903480/live/england-premier-league-2025-2026-manchester-city-sunderland",
        "https://www.whoscored.com/matches/1903488/live/england-premier-league-2025-2026-crystal-palace-manchester-city",
        "https://www.whoscored.com/matches/1903290/live/england-premier-league-2025-2026-manchester-city-west-ham",
        "https://www.whoscored.com/matches/1903323/live/england-premier-league-2025-2026-nottingham-forest-manchester-city",
        "https://www.whoscored.com/matches/1903340/live/england-premier-league-2025-2026-manchester-city-wolves",
        "https://www.whoscored.com/matches/1903277/live/england-premier-league-2025-2026-tottenham-manchester-city",
        "https://www.whoscored.com/matches/1903298/live/england-premier-league-2025-2026-liverpool-manchester-city",
        "https://www.whoscored.com/matches/1903327/live/england-premier-league-2025-2026-manchester-city-fulham",
        "https://www.whoscored.com/matches/1903385/live/england-premier-league-2025-2026-manchester-city-newcastle",
        "https://www.whoscored.com/matches/1903473/live/england-premier-league-2025-2026-chelsea-manchester-city",
        "https://www.whoscored.com/matches/1903378/live/england-premier-league-2025-2026-manchester-city-arsenal",
        "https://www.whoscored.com/matches/1903389/live/england-premier-league-2025-2026-burnley-manchester-city",
        "https://www.whoscored.com/matches/1903413/live/england-premier-league-2025-2026-everton-manchester-city",
        "https://www.whoscored.com/matches/1903432/live/england-premier-league-2025-2026-manchester-city-brentford",
        "https://www.whoscored.com/matches/1903466/live/england-premier-league-2025-2026-manchester-city-crystal-palace",
    ]
}
# ══════════════════════════════════════════════════════════════════

OUTPUT_RAW = TRAINER_DIR / f"{TRAINER_ID}_raw.csv"
TRAINER_DIR.mkdir(parents=True, exist_ok=True)

# ── Browser starten ───────────────────────────────────────────────
options = Options()
driver  = webdriver.Firefox(options=options)
driver.get("https://www.whoscored.com/")
time.sleep(random.uniform(10, 15))

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

# ── Scrapen ───────────────────────────────────────────────────────
all_events  = []
team_id_raw = str(STATION["team_id"])
urls        = STATION["urls"]

for i, url in enumerate(urls, 1):
    if not url.strip():
        continue

    print(f"Spiel {i}/{len(urls)}: {url}")

    try:
        match_data = main.getMatchData(
            driver, url, close_window=False
        )
        df = main.createEventsDF(match_data)

        # Listen-Spalten zu Strings
        for col in df.columns:
            if df[col].apply(
                lambda x: isinstance(x, list)
            ).any():
                df[col] = df[col].astype(str)

        # ── teamId ersetzen ───────────────────────────────────────
        # Trainer-Team → TRAINER_ID
        # Gegner-Team  → opp_{TRAINER_ID}
        df["teamId"] = df["teamId"].apply(
            lambda x: TRAINER_ID
            if str(x) == team_id_raw
            else f"opp_{TRAINER_ID}"
        )

        df["liga_id"]    = STATION["liga_id"]
        df["source_url"] = url

        all_events.append(df)

        # Stats ausgeben
        n_trainer = (df["teamId"] == TRAINER_ID).sum()
        n_opp     = (df["teamId"] == f"opp_{TRAINER_ID}").sum()
        print(f"  ✅ {len(df)} Events "
              f"(Trainer: {n_trainer}, Gegner: {n_opp})")

        time.sleep(random.uniform(8, 12))

    except Exception as e:
        print(f"  ❌ Fehler: {e}")
        time.sleep(random.uniform(5, 10))

driver.quit()

# ── Speichern ─────────────────────────────────────────────────────
if all_events:
    new_df = pd.concat(all_events, ignore_index=True)

    if OUTPUT_RAW.exists():
        existing = pd.read_csv(OUTPUT_RAW)
        combined = pd.concat(
            [existing, new_df], ignore_index=True
        ).drop_duplicates()
    else:
        combined = new_df

    combined.to_csv(OUTPUT_RAW, index=False)

    n_trainer = (combined["teamId"] == TRAINER_ID).sum()
    n_opp     = (combined["teamId"] == f"opp_{TRAINER_ID}").sum()

    print(f"\n{'='*50}")
    print(f"✅ Gespeichert: {OUTPUT_RAW.name}")
    print(f"   Events gesamt:  {len(combined):,}")
    print(f"   Trainer-Events: {n_trainer:,}")
    print(f"   Gegner-Events:  {n_opp:,}")
    print(f"   Spiele: {combined['matchId'].nunique()}")
else:
    print("Keine Events gescrapt!")