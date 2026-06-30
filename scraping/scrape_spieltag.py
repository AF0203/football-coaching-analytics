# scraping/scrape_spieltag.py
"""
Spieltag-Scraper — nach jedem Spieltag ausführen
Konfiguration: LIGA, SAISON, URLS anpassen
Führt automatisch Deduplizierung durch
"""

import sys, os, time, logging
import pandas as pd
import random
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
))
from config import SCRAPER_REPO, LIGEN_DIR

sys.path.insert(0, str(SCRAPER_REPO))
import main
from selenium import webdriver
from selenium.webdriver.firefox.options import Options

# ══════════════════════════════════════════════════════════════════
#  KONFIGURATION — hier anpassen vor jedem Spieltag
# ══════════════════════════════════════════════════════════════════

LIGA     = "bundesliga"   # bundesliga | bundesliga2 | premier_league | la_liga
SAISON   = "2025_26"
SPIELTAG = 28             # nur für Logging/Tracking

URLS = [
    # Hier die WhoScored-Match-URLs des Spieltags einfügen
    # z.B. "https://www.whoscored.com/Matches/123456/Live/..."
]

# ══════════════════════════════════════════════════════════════════

# ── Logging Setup ─────────────────────────────────────────────────
log_dir = LIGEN_DIR / "logs"
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level    = logging.INFO,
    format   = "%(asctime)s [%(levelname)s] %(message)s",
    handlers = [
        logging.FileHandler(
            log_dir / f"{LIGA}_{SAISON}_scrape.log",
            encoding="utf-8"
        ),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# ── Pfade ─────────────────────────────────────────────────────────
OUTPUT_RAW  = LIGEN_DIR / LIGA / f"{LIGA}_{SAISON}_raw.csv"
OUTPUT_LOG  = LIGEN_DIR / LIGA / f"{LIGA}_{SAISON}_scrape_log.csv"
OUTPUT_RAW.parent.mkdir(parents=True, exist_ok=True)

# ── Bereits gescrapte URLs prüfen ────────────────────────────────
scraped_urls = set()
if OUTPUT_LOG.exists():
    scrape_log = pd.read_csv(OUTPUT_LOG)
    scraped_urls = set(
        scrape_log[scrape_log["status"] == "success"]["url"].tolist()
    )
    log.info(f"Bereits gescrapt: {len(scraped_urls)} Spiele")

# Nur neue URLs scrapen
new_urls = [u for u in URLS if u not in scraped_urls]
log.info(f"Neue Spiele zum Scrapen: {len(new_urls)}")

if not new_urls:
    log.info("Alle Spiele bereits gescrapt — nichts zu tun.")
    sys.exit()

# ── Browser starten ───────────────────────────────────────────────
options = Options()
driver  = webdriver.Firefox(options=options)
driver.get("https://www.whoscored.com/")
time.sleep(random.uniform(10, 15))

# Cookie-Banner wegklicken
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
log_entries = []

for i, url in enumerate(new_urls, 1):
    log.info(f"Spiel {i}/{len(new_urls)}: {url}")

    try:
        match_data = main.getMatchData(driver, url, close_window=False)
        df         = main.createEventsDF(match_data)

        df["liga"]     = LIGA
        df["saison"]   = SAISON
        df["spieltag"] = SPIELTAG
        df["source_url"] = url

        all_events.append(df)

        log_entries.append({
            "url":       url,
            "status":    "success",
            "n_events":  len(df),
            "timestamp": datetime.now().isoformat(),
            "spieltag":  SPIELTAG
        })

        log.info(f"  ✅ {len(df)} Events gescrapt")
        time.sleep(random.uniform(10, 18))

    except Exception as e:
        log.error(f"  ❌ Fehler: {e}")
        log_entries.append({
            "url":       url,
            "status":    "error",
            "n_events":  0,
            "timestamp": datetime.now().isoformat(),
            "spieltag":  SPIELTAG
        })
        time.sleep(random.uniform(5, 10))

driver.quit()

# ── Speichern ─────────────────────────────────────────────────────
if all_events:
    new_df = pd.concat(all_events, ignore_index=True)

    # Raw-Datei: append + deduplizieren
    if OUTPUT_RAW.exists():
        existing = pd.read_csv(OUTPUT_RAW)
        combined = pd.concat(
            [existing, new_df], ignore_index=True
        ).drop_duplicates()
    else:
        combined = new_df

    combined.to_csv(OUTPUT_RAW, index=False)
    log.info(f"✅ Raw-Datei: {len(combined):,} Events → {OUTPUT_RAW.name}")
else:
    log.warning("Keine neuen Events gescrapt!")

# Scrape-Log aktualisieren
new_log = pd.DataFrame(log_entries)
if OUTPUT_LOG.exists():
    existing_log = pd.read_csv(OUTPUT_LOG)
    new_log = pd.concat(
        [existing_log, new_log], ignore_index=True
    )
new_log.to_csv(OUTPUT_LOG, index=False)

log.info(f"✅ Scrape-Log: {len(new_log)} Einträge → {OUTPUT_LOG.name}")
log.info("─" * 50)
log.info(f"Erfolgreich: {sum(e['status']=='success' for e in log_entries)}")
log.info(f"Fehler:      {sum(e['status']=='error'   for e in log_entries)}")