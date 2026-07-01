# scripts/vis_matching.py
"""
Trainer-Matching Visualisierung
→ lädt gespeicherte JSON Ergebnisse aus matching.py
→ erstellt eine Infografik-Karte pro Trainer

Aufruf:
    python3 scripts/vis_matching.py \
        --team bl_44 \
        --liga bundesliga \
        --saison 2025_26 \
        --trainer tr_006 \
        --ranking fortführer
"""

import sys, os, argparse, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from pathlib import Path

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
))
from config import PROCESSED_DIR

# ── Design ────────────────────────────────────────────────────────
BG          = "#0D0D0D"
PANEL       = "#111827"
PANEL2      = "#1A2332"
TEXT        = "#E8E8E8"
TEXT_DIM    = "#6B7280"
ACCENT      = "#4CC9F0"   # Cyan
ACCENT2     = "#F72585"   # Magenta
ACCENT3     = "#06D6A0"   # Grün
LINE        = "#1F2937"
LINE2       = "#2D3748"

BEWERTUNG_COLOR = {
    "verbessert": ACCENT3,
    "besser":     ACCENT,
    "ähnlich":    TEXT_DIM,
    "schwächer":  ACCENT2,
}

BEWERTUNG_ICON = {
    "verbessert": "++",
    "besser":     "+",
    "ähnlich":    "~",
    "schwächer":  "!",
}


# ══════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════

def load_trainer_data(team_id, liga, saison, trainer_id, ranking):
    results_dir = PROCESSED_DIR / "matching_results" / \
                  f"{team_id}_{liga}_{saison}"
    json_file   = results_dir / f"{ranking}.json"

    if not json_file.exists():
        raise FileNotFoundError(
            f"Keine Ergebnisse gefunden: {json_file}\n"
            f"Bitte zuerst matching.py laufen lassen."
        )

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    for entry in data:
        if entry["trainer_id"] == trainer_id:
            return entry

    raise ValueError(
        f"Trainer {trainer_id} nicht in {ranking}.json gefunden."
    )


def add_top_bar(fig):
    ax = fig.add_axes([0, 0.988, 1, 0.012])
    ax.set_facecolor(ACCENT)
    ax.set_xticks([])
    ax.set_yticks([])


def add_watermark(fig, team_id, saison):
    fig.text(0.99, 0.005, f"{team_id.upper()} · {saison}",
             color=TEXT_DIM, fontsize=7, alpha=0.35,
             va="bottom", ha="right")


def draw_panel(ax, facecolor=None):
    ax.set_facecolor(facecolor or PANEL)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[:].set_visible(False)


def score_bar(ax, x, y, w, h, value, color, bg=LINE2, label=None):
    """Zeichnet einen Fortschrittsbalken"""
    ax.add_patch(plt.Rectangle(
        (x, y), w, h, facecolor=bg,
        transform=ax.transAxes, zorder=2, clip_on=True
    ))
    ax.add_patch(plt.Rectangle(
        (x, y), w * (value / 100), h, facecolor=color, alpha=0.85,
        transform=ax.transAxes, zorder=3, clip_on=True
    ))
    if label:
        ax.text(x + w + 0.01, y + h / 2, f"{value:.1f}",
                color=color, fontsize=8, fontweight="bold",
                va="center", transform=ax.transAxes)


# ══════════════════════════════════════════════════════════════════
#  HAUPTGRAFIK
# ══════════════════════════════════════════════════════════════════

def create_matching_card(data, output_dir, team_id, saison, ranking):
    name    = data["name"]
    club    = data["club"]
    liga    = data["liga"]
    scores  = data["scores"]
    mc      = data["meta_cluster"]
    struktur= data["struktur"]
    taktisch= data["taktische_ähnlichkeit"]
    perf    = data["performance"]
    rank    = data["rank"]

    ranking_labels = {
        "fortführer": "FORTFÜHRER",
        "erneuerer":  "ERNEUERER",
        "balance":    "BALANCE",
    }

    # Figure Setup — hoch und schmal wie eine Infokarte
    fig = plt.figure(figsize=(20, 14), facecolor=BG)
    fig.patch.set_facecolor(BG)
    add_top_bar(fig)

    # ── HEADER ────────────────────────────────────────────────────
    # Ranking Badge
    fig.text(0.04, 0.980,
             f"#{rank}  {ranking_labels.get(ranking, ranking.upper())}",
             color=ACCENT, fontsize=10, fontweight="bold",
             va="top", alpha=0.8)

    # Name
    fig.text(0.04, 0.968,
             name.upper(),
             color=TEXT, fontsize=26, fontweight="bold", va="top")

    # Club + Liga
    fig.text(0.04, 0.948,
             f"{club}  ·  {liga}",
             color=TEXT_DIM, fontsize=11, va="top")

    # Trennlinie
    ax_line = fig.add_axes([0.04, 0.938, 0.92, 0.001])
    ax_line.set_facecolor(LINE2)
    ax_line.set_xticks([])
    ax_line.set_yticks([])

    # ── SCORES ────────────────────────────────────────────────────
    ax_scores = fig.add_axes([0.04, 0.858, 0.920, 0.068])
    draw_panel(ax_scores, PANEL2)

    # Gesamt Score — groß
    ax_scores.text(0.04, 0.80, "GESAMT-SCORE",
                   color=TEXT_DIM, fontsize=7, fontweight="bold",
                   va="top", transform=ax_scores.transAxes)
    ax_scores.text(0.04, 0.45,
                   f"{scores['gesamt']:.1f}",
                   color=ACCENT, fontsize=28, fontweight="bold",
                   va="center", transform=ax_scores.transAxes)
    ax_scores.text(0.13, 0.45, "/ 100",
                   color=TEXT_DIM, fontsize=12,
                   va="center", transform=ax_scores.transAxes)

    # Performance Score
    ax_scores.text(0.28, 0.80, "PERFORMANCE",
                   color=TEXT_DIM, fontsize=7, fontweight="bold",
                   va="top", transform=ax_scores.transAxes)
    ax_scores.text(0.28, 0.48,
                   f"{scores['performance']:.1f}",
                   color=ACCENT3, fontsize=20, fontweight="bold",
                   va="center", transform=ax_scores.transAxes)
    ax_scores.text(0.28, 0.15,
                   f"× {int(scores['gewichtung']['performance']*100)}%",
                   color=TEXT_DIM, fontsize=8,
                   va="bottom", transform=ax_scores.transAxes)

    # Stil Score
    ax_scores.text(0.50, 0.80, "STIL",
                   color=TEXT_DIM, fontsize=7, fontweight="bold",
                   va="top", transform=ax_scores.transAxes)
    ax_scores.text(0.50, 0.48,
                   f"{scores['stil']:.1f}",
                   color=ACCENT, fontsize=20, fontweight="bold",
                   va="center", transform=ax_scores.transAxes)
    ax_scores.text(0.50, 0.15,
                   f"× {int(scores['gewichtung']['stil']*100)}%",
                   color=TEXT_DIM, fontsize=8,
                   va="bottom", transform=ax_scores.transAxes)

    # Score Bars
    score_bar(ax_scores, 0.68, 0.60, 0.28, 0.12,
              scores["performance"], ACCENT3, label=True)
    score_bar(ax_scores, 0.68, 0.35, 0.28, 0.12,
              scores["stil"], ACCENT, label=True)
    ax_scores.text(0.68, 0.82, "Perf.",
                   color=TEXT_DIM, fontsize=7,
                   va="top", transform=ax_scores.transAxes)
    ax_scores.text(0.68, 0.55, "Stil",
                   color=TEXT_DIM, fontsize=7,
                   va="top", transform=ax_scores.transAxes)

    # ── META + STRUKTUR ───────────────────────────────────────────
    ax_meta = fig.add_axes([0.04, 0.778, 0.455, 0.068])
    draw_panel(ax_meta, PANEL)

    ax_meta.text(0.04, 0.92, "META-CLUSTER",
                 color=TEXT_DIM, fontsize=7, fontweight="bold",
                 va="top", transform=ax_meta.transAxes)

    mc_icon  = "[OK]" if mc["match"] else "[!]"
    mc_color = ACCENT3 if mc["match"] else ACCENT2
    ax_meta.text(0.04, 0.55,
                 f"{mc['verein_name']}",
                 color=TEXT_DIM, fontsize=8,
                 va="center", transform=ax_meta.transAxes)
    ax_meta.text(0.04, 0.20,
                 f"Trainer: {mc['trainer_name']}  {mc_icon}",
                 color=mc_color, fontsize=8, fontweight="bold",
                 va="center", transform=ax_meta.transAxes)

    ax_struk = fig.add_axes([0.505, 0.778, 0.455, 0.068])
    draw_panel(ax_struk, PANEL)

    ax_struk.text(0.04, 0.92, "STRUKTUR",
                  color=TEXT_DIM, fontsize=7, fontweight="bold",
                  va="top", transform=ax_struk.transAxes)

    f_data   = struktur["formation"]
    n_data   = struktur["netzwerk"]
    f_icon   = "[OK]" if f_data["match"] else "[x]"
    n_icon   = "[OK]" if n_data["match"] else "[x]"
    f_color  = ACCENT3 if f_data["match"] else TEXT_DIM
    n_color  = ACCENT3 if n_data["match"] else TEXT_DIM

    ax_struk.text(0.04, 0.55,
                  f"Formation: Saison={f_data['saison']} | "
                  f"Top={f_data['top']} | "
                  f"Trainer={f_data['trainer']}  {f_icon}",
                  color=f_color, fontsize=8,
                  va="center", transform=ax_struk.transAxes)
    ax_struk.text(0.04, 0.20,
                  f"Netzwerk:  Saison={n_data['saison']} | "
                  f"Top={n_data['top']} | "
                  f"Trainer={n_data['trainer']}  {n_icon}",
                  color=n_color, fontsize=8,
                  va="center", transform=ax_struk.transAxes)

    # ── TAKTISCHE ÄHNLICHKEIT ─────────────────────────────────────
    ax_takt = fig.add_axes([0.04, 0.030, 0.455, 0.735])
    draw_panel(ax_takt, PANEL)

    ax_takt.text(0.04, 0.97, "TAKTISCHE ÄHNLICHKEIT",
                 color=TEXT_DIM, fontsize=7, fontweight="bold",
                 va="top", transform=ax_takt.transAxes)
    ax_takt.text(0.55, 0.97, "vs Saison",
                 color=TEXT_DIM, fontsize=7,
                 va="top", ha="center", transform=ax_takt.transAxes)
    ax_takt.text(0.80, 0.97, "vs Top",
                 color=TEXT_DIM, fontsize=7,
                 va="top", ha="center", transform=ax_takt.transAxes)

    profiles = list(taktisch.items())
    n_prof   = len(profiles)
    y_step   = 0.88 / max(n_prof, 1)

    for i, (prof, vals) in enumerate(profiles):
        y = 0.90 - i * y_step

        label   = vals["label"]
        vs_s    = vals["vs_saison"]
        vs_t    = vals["vs_top"]
        flag    = vals.get("flag", "")

        # Label
        ax_takt.text(0.02, y, label,
                     color=TEXT, fontsize=8,
                     va="center", transform=ax_takt.transAxes)

        # vs Saison Bar
        bar_color_s = ACCENT if vs_s >= 75 else \
                      TEXT_DIM if vs_s >= 55 else ACCENT2
        ax_takt.add_patch(plt.Rectangle(
            (0.42, y - 0.012), 0.26 * (vs_s / 100), 0.024,
            facecolor=bar_color_s, alpha=0.7,
            transform=ax_takt.transAxes, zorder=2, clip_on=True
        ))
        ax_takt.text(0.69, y, f"{vs_s:.0f}%",
                     color=bar_color_s, fontsize=8, fontweight="bold",
                     va="center", ha="right",
                     transform=ax_takt.transAxes)

        # vs Top Bar
        if vs_t is not None:
            bar_color_t = ACCENT if vs_t >= 75 else \
                          TEXT_DIM if vs_t >= 55 else ACCENT2
            ax_takt.add_patch(plt.Rectangle(
                (0.72, y - 0.012), 0.20 * (vs_t / 100), 0.024,
                facecolor=bar_color_t, alpha=0.7,
                transform=ax_takt.transAxes, zorder=2, clip_on=True
            ))
            ax_takt.text(0.93, y, f"{vs_t:.0f}%",
                         color=bar_color_t, fontsize=8, fontweight="bold",
                         va="center", ha="right",
                         transform=ax_takt.transAxes)

        # Flag
        if flag == "sehr ähnlich":
            ax_takt.text(0.94, y, "←",
                         color=ACCENT3, fontsize=7,
                         va="center", transform=ax_takt.transAxes)
        elif flag == "abweichend":
            ax_takt.text(0.94, y, "←",
                         color=ACCENT2, fontsize=7,
                         va="center", transform=ax_takt.transAxes)

        # Trennlinie
        if i < n_prof - 1:
            ax_takt.add_patch(plt.Rectangle(
                (0.02, y - y_step * 0.5), 0.96, 0.001,
                facecolor=LINE, alpha=0.5,
                transform=ax_takt.transAxes, zorder=1, clip_on=True
            ))

    # ── PERFORMANCE ───────────────────────────────────────────────
    ax_perf = fig.add_axes([0.505, 0.030, 0.455, 0.735])
    draw_panel(ax_perf, PANEL)

    ax_perf.text(0.02, 0.982, "PERFORMANCE",
                 color=TEXT_DIM, fontsize=7, fontweight="bold",
                 va="top", transform=ax_perf.transAxes)
    ax_perf.text(0.50, 0.982, "Saison",
                 color=TEXT_DIM, fontsize=7,
                 va="top", ha="center", transform=ax_perf.transAxes)
    ax_perf.text(0.65, 0.982, "Top",
                 color=TEXT_DIM, fontsize=7,
                 va="top", ha="center", transform=ax_perf.transAxes)
    ax_perf.text(0.80, 0.982, "Trainer",
                 color=TEXT_DIM, fontsize=7,
                 va="top", ha="center", transform=ax_perf.transAxes)
    ax_perf.text(0.95, 0.982, "",
                 color=TEXT_DIM, fontsize=7,
                 va="top", ha="center", transform=ax_perf.transAxes)

    metrics = list(perf.items())
    n_met   = len(metrics)
    y_step  = 0.94 / max(n_met, 1)

    for i, (metric, vals) in enumerate(metrics):
        y = 0.955 - i * y_step

        label   = vals["label"]
        season  = vals["season"]
        top     = vals["top"]
        trainer = vals["trainer"]
        bew     = vals["bewertung"]

        bew_color = BEWERTUNG_COLOR.get(bew, TEXT_DIM)
        bew_icon  = BEWERTUNG_ICON.get(bew, "")

        # Label
        ax_perf.text(0.02, y, label,
                     color=TEXT, fontsize=8,
                     va="center", transform=ax_perf.transAxes)

        # Saison
        s_str = f"{season:+.2f}" if season is not None else "n/a"
        s_col = ACCENT if season and season >= 0 else \
                ACCENT2 if season and season < 0 else TEXT_DIM
        ax_perf.text(0.50, y, s_str,
                     color=s_col, fontsize=8, fontweight="bold",
                     va="center", ha="center",
                     transform=ax_perf.transAxes)

        # Top
        t_str = f"{top:+.2f}" if top is not None else "n/a"
        t_col = ACCENT if top and top >= 0 else \
                ACCENT2 if top and top < 0 else TEXT_DIM
        ax_perf.text(0.65, y, t_str,
                     color=t_col, fontsize=8, fontweight="bold",
                     va="center", ha="center",
                     transform=ax_perf.transAxes)

        # Trainer
        tr_str = f"{trainer:+.2f}" if trainer is not None else "n/a"
        ax_perf.text(0.80, y, tr_str,
                     color=bew_color, fontsize=8, fontweight="bold",
                     va="center", ha="center",
                     transform=ax_perf.transAxes)

        # Bewertung Icon
        ax_perf.text(0.95, y, bew_icon,
                     color=bew_color, fontsize=9,
                     va="center", ha="center",
                     transform=ax_perf.transAxes)

        # Trennlinie
        if i < n_met - 1:
            ax_perf.add_patch(plt.Rectangle(
                (0.02, y - y_step * 0.5), 0.96, 0.001,
                facecolor=LINE, alpha=0.5,
                transform=ax_perf.transAxes, zorder=1, clip_on=True
            ))

    add_watermark(fig, team_id, saison)

    # Speichern
    out_file = output_dir / \
        f"{team_id}_{data['trainer_id']}_{ranking}.png"
    plt.savefig(out_file, dpi=150,
                facecolor=BG, edgecolor="none")
    plt.close(fig)
    print(f"  ✅ {out_file.name}")
    return out_file


# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--team",    type=str, required=True)
    parser.add_argument("--liga",    type=str, required=True)
    parser.add_argument("--saison",  type=str, required=True)
    parser.add_argument("--trainer", type=str, required=True)
    parser.add_argument("--ranking", type=str, default="balance",
                        choices=["fortführer","erneuerer","balance"])
    parser.add_argument("--output",  type=str, default=None)
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  VIS_MATCHING: {args.team} | {args.trainer} | {args.ranking}")
    print(f"{'='*60}")

    output_dir = Path(args.output) if args.output else \
        PROCESSED_DIR / "visualizations" / args.team
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Lade Ergebnisse...")
    data = load_trainer_data(
        args.team, args.liga, args.saison,
        args.trainer, args.ranking
    )

    print(f"  Trainer: {data['name']} ({data['club']})")
    print(f"  Ranking: #{data['rank']} im {args.ranking} Ranking")
    print(f"  Score:   {data['scores']['gesamt']:.1f}/100")
    print(f"\n  Erstelle Grafik...")

    create_matching_card(
        data, output_dir,
        args.team, args.saison, args.ranking
    )

    print(f"\n{'='*60}")
    print(f"  FERTIG")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()