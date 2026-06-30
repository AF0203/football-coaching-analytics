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

TRAINER_ID = "tr_065"

STATION = {
    "liga_id": "bundesliga",
    "team_id": "796",
    "urls": [
        "https://www.whoscored.com/matches/1643259/live/germany-bundesliga-2022-2023-union-berlin-bochum",
        "https://www.whoscored.com/matches/1643283/live/germany-bundesliga-2022-2023-borussia-m-gladbach-union-berlin",
        "https://www.whoscored.com/matches/1643295/live/germany-bundesliga-2022-2023-union-berlin-bayer-leverkusen",
        "https://www.whoscored.com/matches/1643321/live/germany-bundesliga-2022-2023-augsburg-union-berlin",
        "https://www.whoscored.com/matches/1643333/live/germany-bundesliga-2022-2023-union-berlin-freiburg",
        "https://www.whoscored.com/matches/1643217/live/germany-bundesliga-2022-2023-hoffenheim-union-berlin",
        "https://www.whoscored.com/matches/1643232/live/germany-bundesliga-2022-2023-union-berlin-werder-bremen",
        "https://www.whoscored.com/matches/1743391/live/germany-bundesliga-2023-2024-union-berlin-mainz-05",
        "https://www.whoscored.com/matches/1743408/live/germany-bundesliga-2023-2024-darmstadt-union-berlin",
        "https://www.whoscored.com/matches/1743410/live/germany-bundesliga-2023-2024-union-berlin-rb-leipzig",
        "https://www.whoscored.com/matches/1743421/live/germany-bundesliga-2023-2024-wolfsburg-union-berlin",
        "https://www.whoscored.com/matches/1743429/live/germany-bundesliga-2023-2024-union-berlin-hoffenheim",
        "https://www.whoscored.com/matches/1743443/live/germany-bundesliga-2023-2024-fc-heidenheim-union-berlin",
        "https://www.whoscored.com/matches/1743446/live/germany-bundesliga-2023-2024-borussia-dortmund-union-berlin",
        "https://www.whoscored.com/matches/1743455/live/germany-bundesliga-2023-2024-union-berlin-vfb-stuttgart",
        "https://www.whoscored.com/matches/1743468/live/germany-bundesliga-2023-2024-werder-bremen-union-berlin",
        "https://www.whoscored.com/matches/1743473/live/germany-bundesliga-2023-2024-union-berlin-eintracht-frankfurt",
        "https://www.whoscored.com/matches/1743528/live/germany-bundesliga-2023-2024-bayer-leverkusen-union-berlin",
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