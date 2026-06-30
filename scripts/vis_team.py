# scripts/vis_team.py
"""
Taktische Team-Analyse — Visualisierungen
→ Performance Bar Chart (alle 16 Metriken)
→ Sequenz-Profile: alle Sequenzen eingefärbt nach Cluster
  + Legende mit Z-Score pro Cluster
→ Ballgewinn-Profile: Punkte eingefärbt nach Cluster

Aufruf:
    python3 scripts/vis_team.py --team bl_44 --liga bundesliga --saison 2025_26
    python3 scripts/vis_team.py \
    --trainer tr_010
"""

import sys, os, argparse, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mplsoccer import Pitch
from pathlib import Path

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
))
from config import PROCESSED_DIR, MODELS_DIR, SAISON_CLEAN_DIR, TEAM_PROFILES_DIR, TRAINER_DIR

TRAINER_PROFILES_DIR = PROCESSED_DIR / "trainer_profiles"

# ── Design ────────────────────────────────────────────────────────
BG          = "#0D0D0D"
PANEL       = "#111827"
TEXT        = "#E8E8E8"
TEXT_DIM    = "#6B7280"
ACCENT      = "#4CC9F0"
ACCENT2     = "#F72585"
LINE        = "#1F2937"
PITCH_GREEN = "#0B3D0B"
PITCH_LINE  = "#1A6B1A"

CLUSTER_COLORS = [
    "#4CC9F0",  # Cyan
    "#F72585",  # Magenta
    "#06D6A0",  # Grün
    "#F4A261",  # Orange
    "#7B2FBE",  # Lila
    "#FFD93D",  # Gelb
    "#FF6B6B",  # Rot
    "#A8DADC",  # Hellblau
    "#C77DFF",  # Violett
    "#80B918",  # Hellgrün
    "#FF9F1C",  # Dunkelorange
    "#E9C46A",  # Gold
]

INVERT_METRICS = {
    "avg_ppda", "avg_time_to_shot", "avg_time_to_recovery",
    "xga_quality", "xga_total_pg", "shots_against_pg",
}

PERFORMANCE_LABELS = {
    "deep_buildup_rate":    "Tiefer Aufbau",
    "buildup_success_rate": "Aufbau-Erfolg",
    "chance_threat_rate":   "Chancen-Bedrohung",
    "chance_quality":       "Chancenqualität",
    "chance_quantity_pg":   "Schüsse/Spiel",
    "total_xg_pg":          "xG/Spiel",
    "chance_counter_rate":  "Konter-Effizienz",
    "avg_time_to_shot":     "Zeit zum Abschluss",
    "avg_ppda":             "PPDA (Pressing)",
    "avg_time_to_recovery": "Ballrückeroberung",
    "high_block_quality":   "Hoher Block",
    "mid_block_quality":    "Mittlerer Block",
    "low_block_quality":    "Tiefer Block",
    "xga_quality":          "Gegner xG/Schuss",
    "xga_total_pg":         "xGA/Spiel",
    "shots_against_pg":     "Gegner Schüsse/Spiel",
}

FEAT_BASE = [
    "x","y","endX","endY","delta_x","delta_y",
    "distance","angle","is_carry","zone_x"
]


# ══════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════

def load_model(name):
    with open(MODELS_DIR / f"{name}_km.pkl", "rb") as f:
        km = pickle.load(f)
    with open(MODELS_DIR / f"{name}_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return km, scaler


def add_features(df):
    df = df.copy()
    if "delta_x" not in df.columns:
        df["delta_x"]  = df["endX"] - df["x"]
        df["delta_y"]  = df["endY"] - df["y"]
        df["distance"] = np.sqrt(df["delta_x"]**2 + df["delta_y"]**2)
        df["angle"]    = np.arctan2(df["delta_y"], df["delta_x"])
        df["is_carry"] = (df["type"] == "Carry").astype(int)
        df["zone_x"]   = (df["x"] / 100 * 4).astype(int).clip(0, 3)
    return df


def build_seq_matrix(seq_df, valid_ids, n_actions=3):
    df = add_features(seq_df[seq_df["sequence_id"].isin(valid_ids)].copy())
    if df.empty:
        return np.array([]), [], []

    frames = []
    for rank in range(1, n_actions + 1):
        r = df[df["action_rank"] == rank]
        if r.empty:
            continue
        r = r[["sequence_id","teamId"] + FEAT_BASE].set_index("sequence_id")
        r.columns = ["teamId"] + [f"a{rank}_{c}" for c in FEAT_BASE]
        frames.append(r)

    if not frames:
        return np.array([]), [], []

    combined = frames[0]
    for f in frames[1:]:
        combined = combined.join(
            f.drop(columns=["teamId"], errors="ignore"), how="inner"
        )

    dist_cols = [c for c in combined.columns if c.endswith("_distance")]
    if dist_cols:
        combined["total_distance"] = combined[dist_cols].sum(axis=1)

    feat_cols = [c for c in combined.columns if c != "teamId"]
    X         = np.nan_to_num(combined[feat_cols].values.astype(np.float64))
    return X, combined.index.tolist(), combined["teamId"].tolist()


def load_zscores(profile_name, entity_id, liga, is_trainer=False):
    """Lädt cluster_X_zscore aus gespeichertem Profil"""
    if is_trainer:
        path = TRAINER_PROFILES_DIR / f"profiles_{profile_name}.csv"
    else:
        path = TEAM_PROFILES_DIR / liga / f"profiles_{profile_name}.csv"
    if not path.exists():
        return {}
    df  = pd.read_csv(path)
    row = df[df["teamId"] == entity_id]
    if row.empty:
        return {}
    zscores = {}
    for col in row.columns:
        if col.endswith("_zscore") and col.startswith("cluster_"):
            parts = col.replace("_zscore", "").split("_")
            if len(parts) >= 2 and parts[1].isdigit():
                cluster_id = int(parts[1])
                zscores[cluster_id] = float(row[col].iloc[0])
    return zscores


def load_cluster_names(profile_name):
    """Lädt Cluster-Namen aus cluster_descriptions.csv"""
    path = PROCESSED_DIR / "cluster_descriptions.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    # profile_name z.B. "abstoß" → cluster_type "abstoß"
    sub = df[df["cluster_type"] == profile_name]
    if sub.empty:
        return {}
    return dict(zip(sub["cluster_id"], sub["cluster_name"]))


def make_pitch():
    return Pitch(
        pitch_type="opta",
        pitch_color=PITCH_GREEN,
        line_color=PITCH_LINE,
        linewidth=1.5,
    )


def save_fig(fig, path):
    plt.savefig(path, dpi=150,
                facecolor=BG, edgecolor="none")
    plt.close(fig)
    print(f"    ✅ {Path(path).name}")


def add_top_bar(fig):
    ax = fig.add_axes([0, 0.985, 1, 0.015])
    ax.set_facecolor(ACCENT)
    ax.set_xticks([])
    ax.set_yticks([])


def add_watermark(fig, team_id, saison):
    fig.text(0.98, 0.01, f"{team_id.upper()} · {saison}",
             color=TEXT_DIM, fontsize=8, alpha=0.4,
             va="bottom", ha="right")


# ══════════════════════════════════════════════════════════════════
#  GRAFIK 1 — PERFORMANCE BAR CHART
# ══════════════════════════════════════════════════════════════════

def plot_performance(team_id, liga, saison, output_dir, is_trainer=False):
    print("  → Performance Chart...")

    if is_trainer:
        path = TRAINER_PROFILES_DIR / "profiles_performance.csv"
    else:
        path = TEAM_PROFILES_DIR / liga / "profiles_performance.csv"
    if not path.exists():
        print("    ⚠️  Profil nicht gefunden")
        return

    df  = pd.read_csv(path)
    row = df[df["teamId"] == team_id]
    if row.empty:
        print("    ⚠️  Team nicht gefunden")
        return

    metrics = list(PERFORMANCE_LABELS.keys())
    labels  = [PERFORMANCE_LABELS[m] for m in metrics]
    zscores = []
    for m in metrics:
        z_col = f"{m}_zscore"
        if z_col in row.columns and pd.notna(row[z_col].iloc[0]):
            val = float(row[z_col].iloc[0])
            zscores.append(-val if m in INVERT_METRICS else val)
        else:
            zscores.append(0.0)

    order   = sorted(range(len(zscores)), key=lambda i: zscores[i])
    labels  = [labels[i]  for i in order]
    zscores = [zscores[i] for i in order]

    fig = plt.figure(figsize=(14, 10), facecolor=BG)
    fig.patch.set_facecolor(BG)
    add_top_bar(fig)

    fig.text(0.04, 0.958, "PERFORMANCE ANALYSE",
             color=TEXT, fontsize=20, fontweight="bold", va="top")
    fig.text(0.04, 0.928,
             "Z-Score relativ zum Bundesliga-Durchschnitt · positiv = besser als Durchschnitt",
             color=TEXT_DIM, fontsize=10, va="top")

    ax = fig.add_axes([0.28, 0.06, 0.68, 0.85])
    ax.set_facecolor(PANEL)

    for i, (label, z) in enumerate(zip(labels, zscores)):
        color = ACCENT if z >= 0 else ACCENT2
        alpha = min(1.0, 0.35 + abs(z) * 0.22)
        ax.barh(i, z, color=color, alpha=alpha, height=0.62, zorder=3)
        x_pos = z + (0.07 if z >= 0 else -0.07)
        ha    = "left" if z >= 0 else "right"
        ax.text(x_pos, i, f"{z:+.2f}", color=TEXT,
                fontsize=9, va="center", ha=ha, fontweight="bold")

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, color=TEXT, fontsize=10)
    ax.tick_params(axis="x", colors=TEXT_DIM, labelsize=9)
    ax.set_facecolor(PANEL)
    ax.spines[:].set_visible(False)
    ax.axvline(0, color=LINE, linewidth=1.5, zorder=2)
    ax.set_xlim(-3.5, 3.5)
    for x in [-3,-2,-1,1,2,3]:
        ax.axvline(x, color=LINE, linewidth=0.4, alpha=0.4, zorder=1)

    fig.text(0.28, 0.02, "■  Stärke", color=ACCENT, fontsize=9)
    fig.text(0.42, 0.02, "■  Schwäche", color=ACCENT2, fontsize=9)
    add_watermark(fig, team_id, saison)
    save_fig(fig, output_dir / f"{team_id}_performance.png")


# ══════════════════════════════════════════════════════════════════
#  SEQUENZ PROFILE — alle Pfeile, Farbe = Cluster, Legende mit Z-Score
# ══════════════════════════════════════════════════════════════════

def plot_seq_profile(seq_df, valid_ids, model_name, profile_name,
                     team_id, liga, title, subtitle,
                     saison, output_dir, filename, n_actions=3,
                     is_trainer=False):

    km, scaler = load_model(model_name)
    X, seq_ids, _ = build_seq_matrix(seq_df, valid_ids, n_actions)
    if len(X) == 0:
        print(f"    ⚠️  Keine Vektoren")
        return

    labels  = km.predict(scaler.transform(X))
    K       = km.n_clusters
    seq_res = pd.DataFrame({"sequence_id": seq_ids, "cluster": labels})

    # Z-Scores + Cluster-Namen aus gespeicherten Profilen
    zscores       = load_zscores(profile_name, team_id, liga, is_trainer)
    cluster_names = load_cluster_names(profile_name)

    # Koordinaten mergen
    coords = add_features(
        seq_df[seq_df["sequence_id"].isin(valid_ids)].copy()
    )
    coords = coords.merge(seq_res, on="sequence_id", how="inner")

    cluster_counts = seq_res["cluster"].value_counts().sort_index()
    total          = len(seq_res)

    # Figure: Spielfeld links, Legende rechts
    fig = plt.figure(figsize=(16, 9), facecolor=BG)
    fig.patch.set_facecolor(BG)
    add_top_bar(fig)

    fig.text(0.04, 0.958, title,
             color=TEXT, fontsize=20, fontweight="bold", va="top")
    fig.text(0.04, 0.928, subtitle,
             color=TEXT_DIM, fontsize=10, va="top")

    # Spielfeld
    ax_pitch = fig.add_axes([0.02, 0.05, 0.72, 0.85])
    ax_pitch.set_facecolor(PITCH_GREEN)
    pitch = make_pitch()
    pitch.draw(ax=ax_pitch)

    # Pfeile pro Cluster
    for cluster_id in range(K):
        if cluster_id not in cluster_counts.index:
            continue
        color     = CLUSTER_COLORS[cluster_id % len(CLUSTER_COLORS)]
        clust_seq = coords[coords["cluster"] == cluster_id]

        for sid in clust_seq["sequence_id"].unique():
            s = clust_seq[
                clust_seq["sequence_id"] == sid
            ].sort_values("action_rank")
            for i in range(len(s) - 1):
                r1, r2 = s.iloc[i], s.iloc[i+1]
                if pd.isna(r1["x"]) or pd.isna(r2["x"]):
                    continue
                ax_pitch.annotate("",
                    xy=(r2["x"], r2["y"]),
                    xytext=(r1["x"], r1["y"]),
                    arrowprops=dict(
                        arrowstyle="-|>",
                        color=color,
                        lw=0.5,
                        alpha=0.18,
                        mutation_scale=7,
                    ),
                    zorder=2
                )

    # Legende rechts
    ax_leg = fig.add_axes([0.76, 0.05, 0.22, 0.85])
    ax_leg.set_facecolor(PANEL)
    ax_leg.set_xticks([])
    ax_leg.set_yticks([])
    ax_leg.spines[:].set_visible(False)

    ax_leg.text(0.08, 0.94, "CLUSTER ANALYSE",
                color=TEXT_DIM, fontsize=8, fontweight="bold",
                va="top", transform=ax_leg.transAxes)


    # Cluster nach Häufigkeit sortieren
    sorted_clusters = cluster_counts.sort_values(ascending=False).index

    n_clusters = len(sorted_clusters)
    y_start    = 0.86
    y_step     = min(0.12, 0.78 / max(n_clusters, 1))

    for i, cluster_id in enumerate(sorted_clusters):
        color = CLUSTER_COLORS[cluster_id % len(CLUSTER_COLORS)]
        count = cluster_counts[cluster_id]
        pct   = count / total * 100
        z     = zscores.get(cluster_id, np.nan)
        z_str = f"{z:+.2f}σ" if pd.notna(z) else "n/a"
        y     = y_start - i * y_step

        # Farb-Box
        ax_leg.add_patch(plt.Rectangle(
            (0.06, y - 0.025), 0.10, 0.048,
            facecolor=color, alpha=0.9,
            transform=ax_leg.transAxes, zorder=3,
            clip_on=True
        ))


        name = cluster_names.get(cluster_id, f"Cluster {cluster_id}")
        ax_leg.text(0.21, y + 0.01, name,
                    color=TEXT, fontsize=8.5, fontweight="bold",
                    va="center", transform=ax_leg.transAxes)
        # Z-Score — groß und prominent
        z_color = ACCENT if (pd.notna(z) and z >= 0) else ACCENT2
        ax_leg.text(0.21, y - 0.022,
                    z_str,
                    color=z_color, fontsize=11, fontweight="bold",
                    va="center", transform=ax_leg.transAxes)

        # Sequenz-Info
        ax_leg.text(0.62, y - 0.022,
                    f"{count} · {pct:.0f}%",
                    color=TEXT_DIM, fontsize=8,
                    va="center", transform=ax_leg.transAxes)

        # Trennlinie
        if i < n_clusters - 1:
            line_y = y - y_step * 0.5
            ax_leg.plot([0.06, 0.94], [line_y, line_y],
                        color=LINE, linewidth=0.5,
                        transform=ax_leg.transAxes, zorder=1)

    # Legende Footer
    ax_leg.text(0.08, 0.04,
                "σ = Z-Score vs Bundesliga\npositiv = überdurchschnittlich häufig",
                color=TEXT_DIM, fontsize=7.5, va="bottom",
                transform=ax_leg.transAxes, linespacing=1.6)

    add_watermark(fig, team_id, saison)
    save_fig(fig, output_dir / filename)


# ══════════════════════════════════════════════════════════════════
#  BALLGEWINN PROFILE — Punkte/Pfeile, alle Cluster, Legende mit Z-Score
# ══════════════════════════════════════════════════════════════════

def plot_ballgewinn(df_points, model_name, profile_name,
                    team_id, liga, title, subtitle,
                    saison, output_dir, filename,
                    feat_cols=None,
                    x_col="win_x", y_col="win_y",
                    arrow=False,
                    loss_x=None, loss_y=None,
                    rec_x=None,  rec_y=None,
                    is_trainer=False):

    km, scaler = load_model(model_name)
    df = df_points.copy()

    if feat_cols is None:
        if arrow:
            if "delta_x" not in df.columns:
                df["delta_x"] = df[rec_x] - df[loss_x]
                df["delta_y"] = df[rec_y] - df[loss_y]
            feat_cols = [loss_x, loss_y, rec_x, rec_y, "delta_x","delta_y"]
        else:
            feat_cols = [x_col, y_col]

    X = np.nan_to_num(df[feat_cols].values.astype(np.float64))
    df["cluster"] = km.predict(scaler.transform(X))

    zscores        = load_zscores(profile_name, team_id, liga, is_trainer)
    cluster_names  = load_cluster_names(profile_name)
    cluster_counts = df["cluster"].value_counts().sort_index()
    total          = len(df)

    fig = plt.figure(figsize=(16, 9), facecolor=BG)
    fig.patch.set_facecolor(BG)
    add_top_bar(fig)

    fig.text(0.04, 0.958, title,
             color=TEXT, fontsize=20, fontweight="bold", va="top")
    fig.text(0.04, 0.928, subtitle,
             color=TEXT_DIM, fontsize=10, va="top")

    ax_pitch = fig.add_axes([0.02, 0.05, 0.72, 0.85])
    ax_pitch.set_facecolor(PITCH_GREEN)
    pitch = make_pitch()
    pitch.draw(ax=ax_pitch)

    for cluster_id in cluster_counts.index:
        color    = CLUSTER_COLORS[cluster_id % len(CLUSTER_COLORS)]
        clust_df = df[df["cluster"] == cluster_id]

        if arrow:
            for _, row in clust_df.iterrows():
                ax_pitch.annotate("",
                    xy=(row[rec_x],  row[rec_y]),
                    xytext=(row[loss_x], row[loss_y]),
                    arrowprops=dict(
                        arrowstyle="-|>", color=color,
                        lw=0.5, alpha=0.15, mutation_scale=6,
                    ),
                    zorder=2
                )
        else:
            ax_pitch.scatter(
                clust_df[x_col], clust_df[y_col],
                color=color, alpha=0.25, s=18, zorder=2
            )

    # Legende
    ax_leg = fig.add_axes([0.76, 0.05, 0.22, 0.85])
    ax_leg.set_facecolor(PANEL)
    ax_leg.set_xticks([])
    ax_leg.set_yticks([])
    ax_leg.spines[:].set_visible(False)

    ax_leg.text(0.08, 0.94, "CLUSTER ANALYSE",
                color=TEXT_DIM, fontsize=8, fontweight="bold",
                va="top", transform=ax_leg.transAxes)

    sorted_clusters = cluster_counts.sort_values(ascending=False).index
    n_clusters      = len(sorted_clusters)
    y_start         = 0.86
    y_step          = min(0.12, 0.78 / max(n_clusters, 1))

    for i, cluster_id in enumerate(sorted_clusters):
        color = CLUSTER_COLORS[cluster_id % len(CLUSTER_COLORS)]
        count = cluster_counts[cluster_id]
        pct   = count / total * 100
        z     = zscores.get(cluster_id, np.nan)
        z_str = f"{z:+.2f}σ" if pd.notna(z) else "n/a"
        y     = y_start - i * y_step

        ax_leg.add_patch(plt.Rectangle(
            (0.06, y - 0.025), 0.10, 0.048,
            facecolor=color, alpha=0.9,
            transform=ax_leg.transAxes, zorder=3,
            clip_on=True
        ))
        name = cluster_names.get(cluster_id, f"Cluster {cluster_id}")
        ax_leg.text(0.21, y + 0.01, name,
                    color=TEXT, fontsize=9.0, fontweight="bold",
                    va="center", transform=ax_leg.transAxes)
        z_color = ACCENT if (pd.notna(z) and z >= 0) else ACCENT2
        ax_leg.text(0.21, y - 0.022, z_str,
                    color=z_color, fontsize=11, fontweight="bold",
                    va="center", transform=ax_leg.transAxes)
        ax_leg.text(0.62, y - 0.022, f"{count} · {pct:.0f}%",
                    color=TEXT_DIM, fontsize=8,
                    va="center", transform=ax_leg.transAxes)
        if i < n_clusters - 1:
            line_y = y - y_step * 0.5
            ax_leg.plot([0.06, 0.94], [line_y, line_y],
                        color=LINE, linewidth=0.5,
                        transform=ax_leg.transAxes, zorder=1)

    ax_leg.text(0.08, 0.04,
                "σ = Z-Score vs Bundesliga\npositiv = überdurchschnittlich häufig",
                color=TEXT_DIM, fontsize=7.5, va="bottom",
                transform=ax_leg.transAxes, linespacing=1.6)

    add_watermark(fig, team_id, saison)
    save_fig(fig, output_dir / filename)


# ══════════════════════════════════════════════════════════════════
#  SEQUENZ EXTRAKTION
# ══════════════════════════════════════════════════════════════════

def extract_abstoß(df_team):
    actions = df_team[df_team["type"].isin(["Pass","Carry"])].copy()
    rows, ids = [], []
    for seq_id, seq in actions.groupby("sequence_id"):
        seq = seq.sort_values("event_seconds")
        if seq.iloc[0]["x"] >= 7:
            continue
        pc = seq[seq["type"].isin(["Pass","Carry"])]
        if len(pc) < 3:
            continue
        f3 = pc.iloc[:3].copy()
        f3["action_rank"] = range(1, 4)
        rows.append(f3)
        ids.append(seq_id)
    return pd.concat(rows) if rows else pd.DataFrame(), ids


def extract_aufbau(df_team):
    actions = df_team[df_team["type"].isin(["Pass","Carry"])].copy()
    rows, ids = [], []
    for seq_id, seq in actions.groupby("sequence_id"):
        seq = seq.sort_values("event_seconds")
        if seq.iloc[0]["x"] >= 33:
            continue
        pc = seq[seq["type"].isin(["Pass","Carry"])]
        if len(pc) < 3:
            continue
        f3 = pc.iloc[:3].copy()
        f3["action_rank"] = range(1, 4)
        rows.append(f3)
        ids.append(seq_id)
    return pd.concat(rows) if rows else pd.DataFrame(), ids


def extract_mf(df_team):
    actions = df_team[df_team["type"].isin(["Pass","Carry"])].copy()
    rows, ids = [], []
    for seq_id, seq in actions.groupby("sequence_id"):
        seq = seq.sort_values("event_seconds")
        crossing = seq[
            (seq["outcomeType"] == "Successful") &
            (seq["x"] < 50) & (seq["endX"] >= 50) &
            (seq["type"].isin(["Pass","Carry"]))
        ]
        if crossing.empty:
            continue
        ci        = seq.index.get_loc(crossing.index[0])
        remaining = seq.iloc[ci+1:]
        pc_after  = remaining[remaining["type"].isin(["Pass","Carry"])]
        if len(pc_after) < 3:
            continue
        f3 = pc_after.iloc[:3].copy()
        i1 = remaining.index.get_loc(f3.index[0])
        i2 = remaining.index.get_loc(f3.index[1])
        i3 = remaining.index.get_loc(f3.index[2])
        if not (i2-i1 == 1 and i3-i2 == 1):
            continue
        f3["action_rank"] = range(1, 4)
        rows.append(f3)
        ids.append(seq_id)
    return pd.concat(rows) if rows else pd.DataFrame(), ids


def extract_prog(df_team):
    actions = df_team[df_team["type"].isin(["Pass","Carry"])].copy()
    rows, ids = [], []
    for seq_id, seq in actions.groupby("sequence_id"):
        seq = seq.sort_values("event_seconds")
        if seq.iloc[0]["x"] >= 33:
            continue
        crossing = seq[
            (seq["outcomeType"] == "Successful") &
            (seq["x"] < 66) & (seq["endX"] > 66) &
            (seq["type"].isin(["Pass","Carry"]))
        ]
        if crossing.empty:
            continue
        ci = seq.index.get_loc(crossing.index[0])
        if ci < 1 or ci >= len(seq) - 1:
            continue
        three = seq.iloc[[ci-1, ci, ci+1]].copy()
        if not all(r["type"] in ["Pass","Carry"]
                   for _, r in three.iterrows()):
            continue
        three["action_rank"] = range(1, 4)
        rows.append(three)
        ids.append(seq_id)
    return pd.concat(rows) if rows else pd.DataFrame(), ids


def extract_shot(df_team, df_all):
    shots_all    = df_all[df_all["type"] == "Shot"].copy()
    shot_seq_ids = set(shots_all["sequence_id"].unique())
    actions      = df_team[df_team["type"].isin(["Pass","Carry"])].copy()
    relevant     = actions[
        actions["sequence_id"].isin(shot_seq_ids)
    ].sort_values(["sequence_id","event_seconds"])
    rows, ids = [], []
    for seq_id, seq in relevant.groupby("sequence_id"):
        shot_time = shots_all[
            shots_all["sequence_id"] == seq_id
        ]["event_seconds"].min()
        pc_before = seq[seq["event_seconds"] < shot_time]
        if len(pc_before) < 3:
            continue
        last_3 = pc_before.iloc[-3:].copy()
        times  = last_3["event_seconds"].values
        if not (times[1] > times[0] and times[2] > times[1]):
            continue
        last_3["action_rank"] = range(1, 4)
        rows.append(last_3)
        ids.append(seq_id)
    return pd.concat(rows) if rows else pd.DataFrame(), ids


def extract_transition(df_team, df_all, team_id):
    BALL_WIN  = ["Tackle","Interception","BallRecovery"]
    ball_wins = df_all[
        df_all["type"].isin(BALL_WIN) &
        (df_all["outcomeType"] == "Successful") &
        (df_all["x"] > 50) & df_all["x"].notna() &
        (df_all["teamId"] == team_id)
    ].copy()
    rows, ids = [], []
    for (match_id, t_id), wins in ball_wins.groupby(["matchId","teamId"]):
        team_act = df_team[
            df_team["matchId"] == match_id
        ].sort_values("event_seconds")
        for _, win_row in wins.iterrows():
            win_time = win_row["event_seconds"]
            after    = team_act[team_act["event_seconds"] > win_time]
            pc_after = after[after["type"].isin(["Pass","Carry"])]
            if len(pc_after) < 2:
                continue
            first_2 = pc_after.iloc[:2]
            if first_2["sequence_id"].nunique() > 1:
                continue
            if first_2.iloc[0]["x"] < 50:
                continue
            two = first_2.copy()
            two["action_rank"] = range(1, 3)
            rows.append(two)
            ids.append(first_2["sequence_id"].iloc[0])
    return pd.concat(rows) if rows else pd.DataFrame(), ids


def extract_aufbau_abbruch(df_team):
    actions = df_team[
        df_team["type"].isin(["Pass","Carry","BallTouch","TakeOn"])
    ].copy()
    rows, ids = [], []
    for seq_id, seq in actions.groupby("sequence_id"):
        seq  = seq.sort_values("event_seconds")
        last = seq.iloc[-1]
        is_err = (
            (last["type"] == "Pass" and
             last["outcomeType"] == "Unsuccessful") or
            last["type"] == "BallTouch" or
            (last["type"] == "TakeOn" and
             last["outcomeType"] == "Unsuccessful")
        )
        if not is_err or pd.isna(last["x"]) or last["x"] >= 50:
            continue
        if len(seq) < 2:
            continue
        before = seq.iloc[-2]
        if before["type"] not in ["Pass","Carry"]:
            continue
        two = seq.iloc[-2:].copy()
        two["action_rank"] = range(1, 3)
        rows.append(two)
        ids.append(seq_id)
    return pd.concat(rows) if rows else pd.DataFrame(), ids


def extract_prog_against(df_all, team_id):
    opp_map = {}
    for m_id, m_df in df_all.groupby("matchId"):
        teams = m_df["teamId"].unique()
        if len(teams) == 2:
            opp_map[m_id] = {teams[0]: teams[1], teams[1]: teams[0]}

    # Nur Spiele in denen unser Team gespielt hat
    our_matches = set(df_all[df_all["teamId"] == team_id]["matchId"].unique())
    actions_opp = df_all[
        df_all["type"].isin(["Pass","Carry"]) &
        df_all["matchId"].isin(our_matches)
    ].copy()
    rows, ids = [], []
    for seq_id, seq in actions_opp.groupby("sequence_id"):
        seq      = seq.sort_values("event_seconds")
        t_id     = seq.iloc[0]["teamId"]
        match_id = seq.iloc[0]["matchId"]
        if opp_map.get(match_id, {}).get(t_id) != team_id:
            continue
        if seq.iloc[0]["x"] >= 33:
            continue
        crossing = seq[
            (seq["outcomeType"] == "Successful") &
            (seq["x"] < 66) & (seq["endX"] > 66) &
            (seq["type"].isin(["Pass","Carry"]))
        ]
        if crossing.empty:
            continue
        ci = seq.index.get_loc(crossing.index[0])
        if ci < 1 or ci >= len(seq) - 1:
            continue
        three = seq.iloc[[ci-1, ci, ci+1]].copy()
        if not all(r["type"] in ["Pass","Carry"]
                   for _, r in three.iterrows()):
            continue
        three["action_rank"] = range(1, 4)
        three["teamId"]      = team_id
        rows.append(three)
        ids.append(seq_id)
    return pd.concat(rows) if rows else pd.DataFrame(), ids


def extract_shots_against(df_all, team_id):
    opp_map = {}
    for m_id, m_df in df_all.groupby("matchId"):
        teams = m_df["teamId"].unique()
        if len(teams) == 2:
            opp_map[m_id] = {teams[0]: teams[1], teams[1]: teams[0]}

    # Nur Spiele in denen unser Team gespielt hat
    our_matches = set(df_all[df_all["teamId"] == team_id]["matchId"].unique())

    shots_opp    = df_all[
        (df_all["type"] == "Shot") &
        (df_all["teamId"] != team_id) &
        (df_all["matchId"].isin(our_matches))
    ].copy()
    shot_seq_ids = set(shots_opp["sequence_id"].unique())
    actions_opp  = df_all[
        (df_all["type"].isin(["Pass","Carry"])) &
        (df_all["teamId"] != team_id) &
        (df_all["matchId"].isin(our_matches))
    ].copy()
    relevant = actions_opp[
        actions_opp["sequence_id"].isin(shot_seq_ids)
    ].sort_values(["sequence_id","event_seconds"])
    rows, ids = [], []
    for seq_id, seq in relevant.groupby("sequence_id"):
        shot_time = shots_opp[
            shots_opp["sequence_id"] == seq_id
        ]["event_seconds"].min()
        pc_before = seq[seq["event_seconds"] < shot_time]
        if len(pc_before) < 3:
            continue
        last_3 = pc_before.iloc[-3:].copy()
        times  = last_3["event_seconds"].values
        if not (times[1] > times[0] and times[2] > times[1]):
            continue
        last_3["action_rank"] = range(1, 4)
        last_3["teamId"]      = team_id
        rows.append(last_3)
        ids.append(seq_id)
    return pd.concat(rows) if rows else pd.DataFrame(), ids


def extract_gegenpressing(df_all, team_id):
    DEFENSIVE = ["Tackle","Interception","BallRecovery"]
    HARD      = ["Shot","Foul","CornerAwarded","ThrowIn",
                 "FreekickTaken","OffsideProwl"]
    ALLOWED   = ["Pass","Carry","TakeOn","BallTouch","Aerial"]
    rows = []
    for m_id, m_df in df_all.groupby("matchId"):
        m_df = m_df.sort_values("event_seconds").reset_index(drop=True)
        if m_df["teamId"].nunique() < 2:
            continue
        is_loss = (
            ((m_df["type"] == "Pass") &
             (m_df["outcomeType"] == "Unsuccessful") &
             (m_df["teamId"] == team_id))
            | ((m_df["type"] == "BallTouch") &
               (m_df["teamId"] == team_id))
        )
        for idx in m_df.index[is_loss].tolist():
            lr = m_df.loc[idx]
            lx = lr.get("endX", lr["x"])
            ly = lr.get("endY", lr["y"])
            if pd.isna(lx) or pd.isna(ly):
                continue
            win = m_df.loc[idx+1:idx+15]
            if win.empty:
                continue
            for _, r in win.iterrows():
                if r["teamId"] == team_id:
                    if (r["type"] in DEFENSIVE and
                            r["outcomeType"] == "Successful"):
                        rows.append({
                            "teamId":     team_id,
                            "loss_x":     float(lx),
                            "loss_y":     float(ly),
                            "recovery_x": float(r["x"]),
                            "recovery_y": float(r["y"]),
                        })
                    break
                else:
                    if r["type"] in HARD:
                        break
                    if r["type"] not in ALLOWED:
                        break
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def extract_high_block(df_all, team_id):
    DEFENSIVE = ["Tackle","Interception","BallRecovery"]
    rows = []
    for m_id, m_df in df_all.groupby("matchId"):
        m_df    = m_df.sort_values("event_seconds").reset_index(drop=True)
        teams   = m_df["teamId"].unique()
        opp_ids = [t for t in teams if t != team_id]
        if not opp_ids:
            continue
        opp         = opp_ids[0]
        seq_order   = m_df.drop_duplicates("sequence_id", keep="first")["sequence_id"].reset_index(drop=True)
        seq_pos_map = {s: i for i, s in enumerate(seq_order)}
        m_df["seq_pos"] = m_df["sequence_id"].map(seq_pos_map)
        seq_groups  = {n: g for n, g in m_df.groupby("sequence_id")}
        seq_meta    = m_df.groupby("sequence_id").agg(
            team_id = ("teamId",  "first"),
            start_x = ("x",       "first"),
            seq_pos = ("seq_pos", "first"),
            n_teams = ("teamId",  "nunique"),
        ).reset_index().sort_values("seq_pos")
        cands = seq_meta[
            (seq_meta["team_id"] == opp) &
            (seq_meta["n_teams"] == 1) &
            (seq_meta["start_x"] < 33) &
            seq_meta["start_x"].notna()
        ]
        for _, cand in cands.iterrows():
            nxt = seq_meta[seq_meta["seq_pos"] == cand["seq_pos"]+1]
            if nxt.empty or nxt["team_id"].iloc[0] != team_id:
                continue
            ns   = seq_groups[nxt["sequence_id"].iloc[0]]
            wins = ns[ns["type"].isin(DEFENSIVE) &
                      (ns["outcomeType"] == "Successful")]
            if wins.empty:
                continue
            wr = wins.iloc[0]
            rows.append({
                "teamId": team_id,
                "win_x":  float(wr["x"]),
                "win_y":  float(wr["y"]),
            })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def extract_mid_low_block(df_all, team_id):
    DEFENSIVE = ["Tackle","Interception","BallRecovery"]
    rows = []
    for m_id, m_df in df_all.groupby("matchId"):
        m_df    = m_df.sort_values("event_seconds").reset_index(drop=True)
        teams   = m_df["teamId"].unique()
        opp_ids = [t for t in teams if t != team_id]
        if not opp_ids:
            continue
        opp         = opp_ids[0]
        seq_order   = m_df.drop_duplicates("sequence_id", keep="first")["sequence_id"].reset_index(drop=True)
        seq_pos_map = {s: i for i, s in enumerate(seq_order)}
        m_df["seq_pos"] = m_df["sequence_id"].map(seq_pos_map)
        seq_groups  = {n: g for n, g in m_df.groupby("sequence_id")}
        seq_meta    = m_df.groupby("sequence_id").agg(
            team_id    = ("teamId",  "first"),
            n_above_50 = ("x", lambda x: (x > 50).sum()),
            seq_pos    = ("seq_pos", "first"),
            n_teams    = ("teamId",  "nunique"),
        ).reset_index().sort_values("seq_pos")
        cands = seq_meta[
            (seq_meta["team_id"] == opp) &
            (seq_meta["n_teams"] == 1) &
            (seq_meta["n_above_50"] >= 2)
        ]
        for _, cand in cands.iterrows():
            nxt = seq_meta[seq_meta["seq_pos"] == cand["seq_pos"]+1]
            if nxt.empty or nxt["team_id"].iloc[0] != team_id:
                continue
            ns   = seq_groups[nxt["sequence_id"].iloc[0]]
            wins = ns[ns["type"].isin(DEFENSIVE) &
                      (ns["outcomeType"] == "Successful")]
            if wins.empty:
                continue
            wr = wins.iloc[0]
            rows.append({
                "teamId": team_id,
                "win_x":  float(wr["x"]),
                "win_y":  float(wr["y"]),
            })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--team",    type=str, default=None)
    parser.add_argument("--trainer", type=str, default=None)
    parser.add_argument("--liga",    type=str, default="bundesliga")
    parser.add_argument("--saison",  type=str, default="")
    parser.add_argument("--output",  type=str, default=None)
    args = parser.parse_args()

    if not args.team and not args.trainer:
        print("❌ Bitte --team oder --trainer angeben!")
        return

    is_trainer = args.trainer is not None
    entity_id  = args.trainer if is_trainer else args.team
    mode_label = f"TRAINER: {entity_id}" if is_trainer                  else f"TEAM: {entity_id} | {args.liga} | {args.saison}"

    print(f"\n{'='*60}")
    print(f"  VIS_TEAM — {mode_label}")
    print(f"{'='*60}")

    output_dir = Path(args.output) if args.output else         PROCESSED_DIR / "visualizations" / entity_id
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Output: {output_dir}")

    if is_trainer:
        # Trainer Raw-CSV laden
        from scripts.processing import prepare_db
        raw_file = TRAINER_DIR / f"{entity_id}_raw.csv"
        if not raw_file.exists():
            print(f"❌ Trainer-Rohdaten nicht gefunden: {raw_file}")
            return
        print(f"\n  Lade Trainer-Daten...")
        df_raw = pd.read_csv(raw_file, low_memory=False)
        for col in df_raw.columns:
            if df_raw[col].apply(lambda x: isinstance(x, list)).any():
                df_raw[col] = df_raw[col].astype(str)
        liga_id = df_raw["liga_id"].iloc[0]             if "liga_id" in df_raw.columns else args.liga
        df_all = prepare_db(df_raw, liga_id, "trainer")

        def fix_team_id(x):
            x = str(x)
            if x.endswith(f"opp_{entity_id}"):
                return f"opp_{entity_id}"
            elif x.endswith(entity_id):
                return entity_id
            return x

        df_all["teamId"]     = df_all["teamId"].apply(fix_team_id)
        df_all["liga_id"]    = liga_id
        df_all["sequence_id"] = (
            entity_id + "_" + df_all["sequence_id"].astype(str)
        )
        df_team = df_all[df_all["teamId"] == entity_id].copy()
        print(f"  Events: {len(df_team):,} | Spiele: {df_team['matchId'].nunique()}")
    else:
        # Team Clean-DB laden
        clean_file = SAISON_CLEAN_DIR /             f"{args.liga}_{args.saison}_clean.csv"
        if not clean_file.exists():
            print(f"❌ Clean-DB nicht gefunden: {clean_file}")
            return
        print(f"\n  Lade Clean-DB...")
        df_all = pd.read_csv(clean_file, low_memory=False)
        df_all["sequence_id"] = (
            df_all["liga"] + "_" + df_all["sequence_id"].astype(str)
        )
        if args.team not in df_all["teamId"].unique():
            print(f"❌ Team {args.team} nicht gefunden!")
            return
        df_team = df_all[df_all["teamId"] == args.team].copy()
        print(f"  Events: {len(df_team):,} | Spiele: {df_team['matchId'].nunique()}")

    print(f"\n  Erstelle Grafiken...")

    t  = entity_id
    l  = args.liga
    s  = args.saison
    od = output_dir

    # 1. Performance
    plot_performance(t, l, s, od, is_trainer=is_trainer)

    # 2. Abstoß
    print("  → Abstoß...")
    seq_df, ids = extract_abstoß(df_team)
    if not seq_df.empty:
        plot_seq_profile(seq_df, ids, "seq_cluster_abstoß", "abstoß",
            t, l, "ABSTOSS",
            "Sequenzmuster beim Torausstoss · Farbe = Cluster",
            s, od, f"{t}_abstoß.png", n_actions=3, is_trainer=is_trainer)

    # 3. Aufbau
    print("  → Aufbau...")
    seq_df, ids = extract_aufbau(df_team)
    if not seq_df.empty:
        plot_seq_profile(seq_df, ids, "seq_cluster_aufbau", "aufbau",
            t, l, "SPIELAUFBAU",
            "Sequenzmuster im Aufbau aus dem eigenen Drittel · Farbe = Cluster",
            s, od, f"{t}_aufbau.png", n_actions=3, is_trainer=is_trainer)

    # 4. Mittelfeld
    print("  → Mittelfeld...")
    seq_df, ids = extract_mf(df_team)
    if not seq_df.empty:
        plot_seq_profile(seq_df, ids, "seq_cluster_mf", "mf",
            t, l, "MITTELFELD",
            "Kombinationsmuster nach Überschreiten der Mittellinie · Farbe = Cluster",
            s, od, f"{t}_mf.png", n_actions=3, is_trainer=is_trainer)

    # 5. Progression
    print("  → Progression...")
    seq_df, ids = extract_prog(df_team)
    if not seq_df.empty:
        plot_seq_profile(seq_df, ids, "seq_cluster_prog", "prog",
            t, l, "PROGRESSION",
            "Sequenzmuster beim Eindringen ins letzte Drittel · Farbe = Cluster",
            s, od, f"{t}_prog.png", n_actions=3, is_trainer=is_trainer)

    # 6. Vor Schuss
    print("  → Vor Schuss...")
    seq_df, ids = extract_shot(df_team, df_all)
    if not seq_df.empty:
        plot_seq_profile(seq_df, ids, "seq_cluster_shot", "shot",
            t, l, "VOR DEM ABSCHLUSS",
            "Sequenzmuster in den letzten Aktionen vor dem Schuss · Farbe = Cluster",
            s, od, f"{t}_shot.png", n_actions=3, is_trainer=is_trainer)

    # 7. Transition
    print("  → Transition...")
    seq_df, ids = extract_transition(df_team, df_all, t)
    if not seq_df.empty:
        plot_seq_profile(seq_df, ids, "seq_cluster_tr", "transition",
            t, l, "UMSCHALTSPIEL",
            "Sequenzmuster nach Ballgewinn im gegnerischen Drittel · Farbe = Cluster",
            s, od, f"{t}_transition.png", n_actions=2, is_trainer=is_trainer)

    # 8. Aufbau-Abbruch
    print("  → Aufbau-Abbruch...")
    seq_df, ids = extract_aufbau_abbruch(df_team)
    if not seq_df.empty:
        plot_seq_profile(seq_df, ids, "seq_cluster_ab", "aufbau_abbruch",
            t, l, "AUFBAU-ABBRUCH",
            "Sequenzmuster bei Ballverlusten im eigenen Aufbau · Farbe = Cluster",
            s, od, f"{t}_aufbau_abbruch.png", n_actions=2, is_trainer=is_trainer)

    # 9. Progression gegen sich
    print("  → Progression gegen sich...")
    seq_df, ids = extract_prog_against(df_all, t)
    if not seq_df.empty:
        plot_seq_profile(seq_df, ids, "seq_cluster_prog_against", "prog_against",
            t, l, "GEGNERISCHE PROGRESSION",
            "Wie der Gegner ins letzte Drittel eindringt · Farbe = Cluster",
            s, od, f"{t}_prog_against.png", n_actions=3, is_trainer=is_trainer)

    # 10. Schüsse gegen sich
    print("  → Schüsse gegen sich...")
    seq_df, ids = extract_shots_against(df_all, t)
    if not seq_df.empty:
        plot_seq_profile(seq_df, ids, "seq_cluster_shots_against", "shots_against",
            t, l, "GEGNERISCHE ABSCHLÜSSE",
            "Wie der Gegner zu Abschlüssen kommt · Farbe = Cluster",
            s, od, f"{t}_shots_against.png", n_actions=3, is_trainer=is_trainer)

    # 11. Gegenpressing
    print("  → Gegenpressing...")
    press_df = extract_gegenpressing(df_all, t)
    if not press_df.empty:
        plot_ballgewinn(
            press_df, "seq_cluster_press", "gegenpressing",
            t, l, "GEGENPRESSING",
            "Ballrückeroberung nach eigenem Ballverlust · Farbe = Cluster",
            s, od, f"{t}_gegenpressing.png",
            arrow=True,
            loss_x="loss_x", loss_y="loss_y",
            rec_x="recovery_x", rec_y="recovery_y"
        ,
            is_trainer=is_trainer
        )

    # 12. Hoher Block
    print("  → Hoher Block...")
    hb_df = extract_high_block(df_all, t)
    if not hb_df.empty:
        plot_ballgewinn(
            hb_df, "high_block", "high_block",
            t, l, "HOHER BLOCK",
            "Ballgewinne gegen tief aufbauenden Gegner · Farbe = Cluster",
            s, od, f"{t}_high_block.png",
            x_col="win_x", y_col="win_y"
        ,
            is_trainer=is_trainer
        )

    # 13. Mid/Low Block
    print("  → Mid/Low Block...")
    ml_df = extract_mid_low_block(df_all, t)
    if not ml_df.empty:
        plot_ballgewinn(
            ml_df, "mid_low_block", "mid_low_block",
            t, l, "MITTLERER / TIEFER BLOCK",
            "Ballgewinne gegen Gegner im eigenen Drittel · Farbe = Cluster",
            s, od, f"{t}_mid_low_block.png",
            x_col="win_x", y_col="win_y"
        ,
            is_trainer=is_trainer
        )

    print(f"\n{'='*60}")
    print(f"  FERTIG — {t}")
    print(f"  Grafiken: {od}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()