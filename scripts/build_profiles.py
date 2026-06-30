# scripts/build_profiles.py
"""
Profile-Builder für Saison-Teams
→ liest Clean-DB aus processed/saison_clean/
→ berechnet alle 15 Profile pro Liga
→ speichert in processed/team_profiles/{liga}/
→ Z-Scores auf "pro Spiel" Basis (pg)

Aufruf:
    python3 scripts/build_profiles.py --saison 2025_26
    python3 scripts/build_profiles.py --saison 2025_26 --liga bundesliga
"""

import sys, os, argparse, pickle, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
))
from config import (
    PROCESSED_DIR, MODELS_DIR,
    SAISON_CLEAN_DIR, TEAM_PROFILES_DIR, LIGEN
)

# ══════════════════════════════════════════════════════════════════
#  HELPER FUNKTIONEN
# ══════════════════════════════════════════════════════════════════

def load_model(name):
    with open(MODELS_DIR / f"{name}_km.pkl", "rb") as f:
        km = pickle.load(f)
    with open(MODELS_DIR / f"{name}_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return km, scaler


def save_profile(df, liga, name):
    out_dir = TEAM_PROFILES_DIR / liga
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"profiles_{name}.csv"
    df.to_csv(out, index=False)
    print(f"  ✅ profiles_{name}.csv ({len(df)} Teams)")


def add_liga_zscores(df, pg_cols):
    """Z-Scores pro Liga auf pg-Werten (pro Spiel)"""
    for col in pg_cols:
        df[col.replace("pg","zscore")] = np.nan

    for liga in df["liga"].unique():
        mask      = df["liga"] == liga
        liga_data = df[mask]
        for col in pg_cols:
            mean = liga_data[col].mean()
            std  = liga_data[col].std() + 1e-6
            df.loc[mask, col.replace("pg","zscore")] = (
                liga_data[col] - mean
            ) / std
    return df


def build_cluster_profile(seq_df, km, K, liga_map, games_per_team):
    """
    Cluster-Profil: abs + pg (pro Spiel) + zscore
    Z-Scores auf pg-Basis statt freq-Basis
    """
    counts = seq_df.groupby(
        ["teamId","cluster"]
    ).size().unstack(fill_value=0)

    counts.columns = [
        f"cluster_{c}_abs" if isinstance(c, int) else c
        for c in counts.columns
    ]

    for c in range(K):
        col = f"cluster_{c}_abs"
        if col not in counts.columns:
            counts[col] = 0

    abs_cols = [f"cluster_{c}_abs" for c in range(K)]
    pg_cols  = [f"cluster_{c}_pg"  for c in range(K)]

    counts["liga"]    = counts.index.map(liga_map)
    counts["n_seqs"]  = counts[abs_cols].sum(axis=1)
    counts["n_games"] = counts.index.map(games_per_team)

    # pg = abs / n_games
    for col in abs_cols:
        counts[col.replace("abs","pg")] = (
            counts[col] / counts["n_games"]
        )

    counts = add_liga_zscores(counts, pg_cols)
    return counts.reset_index()


# ── xG Modell laden ───────────────────────────────────────────────
def load_xg_model():
    import torch
    import torch.nn as nn

    class xGNet(nn.Module):
        def __init__(self, input_dim):
            super(xGNet, self).__init__()
            self.network = nn.Sequential(
                nn.Linear(input_dim, 64), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(32, 16), nn.ReLU(),
                nn.Linear(16, 1), nn.Sigmoid()
            )
        def forward(self, x):
            return self.network(x).squeeze(1)

    with open(MODELS_DIR / "xg_model_config.pkl", "rb") as f:
        xg_config = pickle.load(f)

    xg_mean      = np.array(xg_config["X_mean"])
    xg_std       = np.array(xg_config["X_std"])
    FEATURE_COLS = xg_config["feature_cols"]
    input_dim    = xg_config["input_dim"]

    model = xGNet(input_dim=input_dim)
    model.load_state_dict(
        torch.load(
            MODELS_DIR / "xg_model_weights.pt",
            map_location="cpu"
        )
    )
    model.eval()
    return model, xg_mean, xg_std, FEATURE_COLS

try:
    XG_MODEL, XG_MEAN, XG_STD, XG_FEATURE_COLS = load_xg_model()
    print(f"✅ xG-Modell geladen ({len(XG_FEATURE_COLS)} Features)")
except Exception as e:
    XG_MODEL = None
    XG_MEAN  = None
    XG_STD   = None
    XG_FEATURE_COLS = []
    print(f"⚠️  xG-Modell nicht geladen: {e}")


def compute_xg(shots_df, df_all):
    import torch
    if XG_MODEL is None or len(shots_df) == 0:
        shots_out = shots_df.copy()
        shots_out["xg"] = 0.08
        return shots_out

    shots = shots_df.copy()
    shots["distance_to_goal"]   = np.sqrt(
        (100 - shots["x"])**2 + (50 - shots["y"])**2
    )
    shots["angle_to_goal"]      = np.arctan2(
        abs(shots["y"] - 50), 100 - shots["x"]
    )
    shots["is_central"]         = (
        (shots["y"] >= 30) & (shots["y"] <= 70)
    ).astype(int)
    shots["in_box"]             = (
        (shots["x"] >= 83) &
        (shots["y"] >= 21) &
        (shots["y"] <= 79)
    ).astype(int)
    shots["distance_to_center"] = abs(shots["y"] - 50)

    def get_shot_zone(x, y):
        SHOT_ZONES = {
            1: (83,100, 0,30), 2: (83,100,30,50),
            3: (83,100,50,70), 4: (83,100,70,100),
            5: (75, 83,25,50), 6: (75, 83,50,75),
            7: (66, 83,30,70), 8: (66,100, 0,25),
            9: (66,100,75,100),
        }
        for zone_id, (x1,x2,y1,y2) in SHOT_ZONES.items():
            if x1<=x<=x2 and y1<=y<=y2:
                return zone_id
        return 0

    shots["shot_zone"] = shots.apply(
        lambda r: get_shot_zone(r["x"], r["y"]), axis=1
    )
    seq_stats = df_all.groupby("sequence_id").agg(
        seq_length=("type","count")
    ).reset_index()
    shots = shots.merge(seq_stats, on="sequence_id", how="left")
    shots["sequence_length"]        = shots["seq_length"].fillna(1)
    shots["time_since_last_action"] = 0.5
    shots["prev_pass"]              = 0
    shots["prev_carry"]             = 0
    shots["prev_takeOn"]            = 0
    shots["pressure_score"]         = 0
    shots["pressure_nearby"]        = 0
    shots = shots.fillna(0)

    for col in XG_FEATURE_COLS:
        if col not in shots.columns:
            shots[col] = 0

    X      = shots[XG_FEATURE_COLS].values.astype(np.float32)
    X_norm = (X - XG_MEAN) / (XG_STD + 1e-8)

    with torch.no_grad():
        xg_preds = XG_MODEL(torch.FloatTensor(X_norm)).numpy()

    shots["xg"] = xg_preds
    return shots


# ══════════════════════════════════════════════════════════════════
#  FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════

def build_seq_vectors_fast(seq_df, complete_ids):
    df = seq_df[seq_df["sequence_id"].isin(complete_ids)].copy()

    feature_cols_base = [
        "x","y","endX","endY",
        "delta_x","delta_y",
        "distance","angle",
        "is_carry","zone_x"
    ]

    if "delta_x" not in df.columns:
        df["delta_x"]  = df["endX"] - df["x"]
        df["delta_y"]  = df["endY"] - df["y"]
        df["distance"] = np.sqrt(df["delta_x"]**2 + df["delta_y"]**2)
        df["angle"]    = np.arctan2(df["delta_y"], df["delta_x"])
        df["is_carry"] = (df["type"] == "Carry").astype(int)
        df["zone_x"]   = (df["x"] / 100 * 4).astype(int).clip(0, 3)

    rank1 = df[df["action_rank"]==1][
        ["sequence_id","teamId"] + feature_cols_base
    ].set_index("sequence_id")
    rank2 = df[df["action_rank"]==2][
        ["sequence_id"] + feature_cols_base
    ].set_index("sequence_id")
    rank3 = df[df["action_rank"]==3][
        ["sequence_id"] + feature_cols_base
    ].set_index("sequence_id")

    rank1.columns = ["teamId"] + [f"a1_{c}" for c in feature_cols_base]
    rank2.columns = [f"a2_{c}" for c in feature_cols_base]
    rank3.columns = [f"a3_{c}" for c in feature_cols_base]

    combined = rank1.join(rank2, how="inner").join(rank3, how="inner")
    combined["total_distance"] = (
        combined["a1_distance"] +
        combined["a2_distance"] +
        combined["a3_distance"]
    )

    feature_cols_all = (
        [f"a1_{c}" for c in feature_cols_base] +
        [f"a2_{c}" for c in feature_cols_base] +
        [f"a3_{c}" for c in feature_cols_base] +
        ["total_distance"]
    )

    X        = np.nan_to_num(combined[feature_cols_all].values.astype(np.float64))
    seq_ids  = combined.index.tolist()
    team_ids = combined["teamId"].tolist()
    return X, seq_ids, team_ids


def build_seq_vectors_fast_2(seq_df, complete_ids):
    df = seq_df[seq_df["sequence_id"].isin(complete_ids)].copy()

    feature_cols_base = [
        "x","y","endX","endY",
        "delta_x","delta_y",
        "distance","angle",
        "is_carry","zone_x"
    ]

    if "delta_x" not in df.columns:
        df["delta_x"]  = df["endX"] - df["x"]
        df["delta_y"]  = df["endY"] - df["y"]
        df["distance"] = np.sqrt(df["delta_x"]**2 + df["delta_y"]**2)
        df["angle"]    = np.arctan2(df["delta_y"], df["delta_x"])
        df["is_carry"] = (df["type"] == "Carry").astype(int)
        df["zone_x"]   = (df["x"] / 100 * 4).astype(int).clip(0, 3)

    rank1 = df[df["action_rank"]==1][
        ["sequence_id","teamId"] + feature_cols_base
    ].set_index("sequence_id")
    rank2 = df[df["action_rank"]==2][
        ["sequence_id"] + feature_cols_base
    ].set_index("sequence_id")

    rank1.columns = ["teamId"] + [f"a1_{c}" for c in feature_cols_base]
    rank2.columns = [f"a2_{c}" for c in feature_cols_base]

    combined = rank1.join(rank2, how="inner")
    combined["total_distance"] = (
        combined["a1_distance"] + combined["a2_distance"]
    )

    feature_cols_all = (
        [f"a1_{c}" for c in feature_cols_base] +
        [f"a2_{c}" for c in feature_cols_base] +
        ["total_distance"]
    )

    X        = np.nan_to_num(combined[feature_cols_all].values.astype(np.float64))
    seq_ids  = combined.index.tolist()
    team_ids = combined["teamId"].tolist()
    return X, seq_ids, team_ids


def get_shot_seq_ids(df_all):
    return set(df_all[df_all["type"]=="Shot"]["sequence_id"].unique())


def get_opp_map(df_all):
    match_teams = df_all.groupby("matchId")["teamId"].apply(list).to_dict()
    opp_map = {}
    for m_id, teams in match_teams.items():
        if len(teams) >= 2:
            unique_teams = list(set(teams))
            if len(unique_teams) == 2:
                opp_map[m_id] = {
                    unique_teams[0]: unique_teams[1],
                    unique_teams[1]: unique_teams[0]
                }
    return opp_map


# ══════════════════════════════════════════════════════════════════
#  PROFILE-BUILDER FUNKTIONEN
# ══════════════════════════════════════════════════════════════════

def build_abstoß_profile(actions, df_all, liga_map, liga, games_per_team):
    print("  → Abstoß...")
    km, scaler = load_model("seq_cluster_abstoß")
    K = km.n_clusters
    rows, valid_ids = [], []
    for seq_id, seq_df in actions.groupby("sequence_id"):
        seq = seq_df.sort_values("event_seconds")
        if seq.iloc[0]["x"] >= 7:
            continue
        pc = seq[seq["type"].isin(["Pass","Carry"])]
        if len(pc) < 3:
            continue
        first_3 = pc.iloc[:3].copy()
        first_3["action_rank"] = range(1, 4)
        first_3["sequence_id"] = seq_id
        rows.append(first_3)
        valid_ids.append(seq_id)
    if not rows:
        return
    seq_out = pd.concat(rows)
    X, seq_ids, team_ids = build_seq_vectors_fast(seq_out, valid_ids)
    labels = km.predict(scaler.transform(X))
    seq_res = pd.DataFrame({"sequence_id":seq_ids,"teamId":team_ids,"cluster":labels})
    profile = build_cluster_profile(seq_res, km, K, liga_map, games_per_team)
    save_profile(profile, liga, "abstoß")


def build_aufbau_profile(actions, df_all, liga_map, liga, games_per_team):
    print("  → Aufbau...")
    km, scaler = load_model("seq_cluster_aufbau")
    K = km.n_clusters
    rows, valid_ids = [], []
    for seq_id, seq_df in actions.groupby("sequence_id"):
        seq = seq_df.sort_values("event_seconds")
        if seq.iloc[0]["x"] >= 33:
            continue
        pc = seq[seq["type"].isin(["Pass","Carry"])]
        if len(pc) < 3:
            continue
        first_3 = pc.iloc[:3].copy()
        first_3["action_rank"] = range(1, 4)
        first_3["sequence_id"] = seq_id
        rows.append(first_3)
        valid_ids.append(seq_id)
    if not rows:
        return
    seq_out = pd.concat(rows)
    X, seq_ids, team_ids = build_seq_vectors_fast(seq_out, valid_ids)
    labels = km.predict(scaler.transform(X))
    seq_res = pd.DataFrame({"sequence_id":seq_ids,"teamId":team_ids,"cluster":labels})
    profile = build_cluster_profile(seq_res, km, K, liga_map, games_per_team)
    save_profile(profile, liga, "aufbau")


def build_mf_profile(actions, df_all, liga_map, liga, games_per_team):
    print("  → Mittelfeld...")
    km, scaler = load_model("seq_cluster_mf")
    K = km.n_clusters
    rows, valid_ids = [], []
    for seq_id, seq_df in actions.groupby("sequence_id"):
        seq = seq_df.sort_values("event_seconds")
        crossing = seq[
            (seq["outcomeType"] == "Successful") &
            (seq["x"] < 50) & (seq["endX"] >= 50) &
            (seq["type"].isin(["Pass","Carry"]))
        ]
        if len(crossing) == 0:
            continue
        cross_idx = seq.index.get_loc(crossing.index[0])
        remaining = seq.iloc[cross_idx+1:]
        pc_after  = remaining[remaining["type"].isin(["Pass","Carry"])]
        if len(pc_after) < 3:
            continue
        first_3 = pc_after.iloc[:3].copy()
        idx_1 = remaining.index.get_loc(first_3.index[0])
        idx_2 = remaining.index.get_loc(first_3.index[1])
        idx_3 = remaining.index.get_loc(first_3.index[2])
        if not (idx_2-idx_1==1 and idx_3-idx_2==1):
            continue
        first_3["action_rank"] = range(1, 4)
        first_3["sequence_id"] = seq_id
        rows.append(first_3)
        valid_ids.append(seq_id)
    if not rows:
        return
    seq_out = pd.concat(rows)
    X, seq_ids, team_ids = build_seq_vectors_fast(seq_out, valid_ids)
    labels = km.predict(scaler.transform(X))
    seq_res = pd.DataFrame({"sequence_id":seq_ids,"teamId":team_ids,"cluster":labels})
    profile = build_cluster_profile(seq_res, km, K, liga_map, games_per_team)
    save_profile(profile, liga, "mf")


def build_prog_profile(actions, df_all, liga_map, liga, games_per_team):
    print("  → Progression...")
    km, scaler = load_model("seq_cluster_prog")
    K = km.n_clusters
    rows, valid_ids = [], []
    for seq_id, seq_df in actions.groupby("sequence_id"):
        seq = seq_df.sort_values("event_seconds")
        if seq.iloc[0]["x"] >= 33:
            continue
        crossing = seq[
            (seq["outcomeType"] == "Successful") &
            (seq["x"] < 66) & (seq["endX"] > 66) &
            (seq["type"].isin(["Pass","Carry"]))
        ]
        if len(crossing) == 0:
            continue
        cross_idx = seq.index.get_loc(crossing.index[0])
        if cross_idx < 1 or cross_idx >= len(seq)-1:
            continue
        before = seq.iloc[cross_idx-1]
        cross  = seq.iloc[cross_idx]
        after  = seq.iloc[cross_idx+1]
        if not all(r["type"] in ["Pass","Carry"] for r in [before, cross, after]):
            continue
        three = pd.DataFrame([before, cross, after])
        three["action_rank"] = range(1, 4)
        three["sequence_id"] = seq_id
        rows.append(three)
        valid_ids.append(seq_id)
    if not rows:
        return
    seq_out = pd.concat(rows)
    X, seq_ids, team_ids = build_seq_vectors_fast(seq_out, valid_ids)
    labels = km.predict(scaler.transform(X))
    seq_res = pd.DataFrame({"sequence_id":seq_ids,"teamId":team_ids,"cluster":labels})
    profile = build_cluster_profile(seq_res, km, K, liga_map, games_per_team)
    save_profile(profile, liga, "prog")


def build_shot_profile(actions, df_all, liga_map, liga, games_per_team):
    print("  → Vor Schuss...")
    km, scaler = load_model("seq_cluster_shot")
    K = km.n_clusters
    shot_seq_ids = get_shot_seq_ids(df_all)
    shots_all    = df_all[df_all["type"]=="Shot"].copy()
    relevant     = actions[
        actions["sequence_id"].isin(shot_seq_ids)
    ].sort_values(["sequence_id","event_seconds"])
    rows, valid_ids = [], []
    for seq_id, seq_df in relevant.groupby("sequence_id"):
        shot_time = shots_all[
            shots_all["sequence_id"]==seq_id
        ]["event_seconds"].min()
        pc_before = seq_df[seq_df["event_seconds"] < shot_time]
        if len(pc_before) < 3:
            continue
        last_3 = pc_before.iloc[-3:]
        times  = last_3["event_seconds"].values
        if not (times[1]>times[0] and times[2]>times[1]):
            continue
        last_3 = last_3.copy()
        last_3["action_rank"] = range(1, 4)
        last_3["sequence_id"] = seq_id
        rows.append(last_3)
        valid_ids.append(seq_id)
    if not rows:
        return
    seq_out = pd.concat(rows)
    X, seq_ids, team_ids = build_seq_vectors_fast(seq_out, valid_ids)
    labels = km.predict(scaler.transform(X))
    seq_res = pd.DataFrame({"sequence_id":seq_ids,"teamId":team_ids,"cluster":labels})
    profile = build_cluster_profile(seq_res, km, K, liga_map, games_per_team)
    save_profile(profile, liga, "shot")


def build_transition_profile(actions, df_all, liga_map, liga, games_per_team):
    print("  → Transition...")
    km, scaler = load_model("seq_cluster_tr")
    K = km.n_clusters
    BALL_WIN_TYPES = ["Tackle","Interception","BallRecovery"]
    ball_wins = df_all[
        df_all["type"].isin(BALL_WIN_TYPES) &
        (df_all["outcomeType"]=="Successful") &
        (df_all["x"] > 50) & df_all["x"].notna()
    ].copy()
    rows, valid_ids = [], []
    for (match_id, team_id), wins in ball_wins.groupby(["matchId","teamId"]):
        team_actions = actions[
            (actions["matchId"]==match_id) &
            (actions["teamId"]==team_id)
        ].sort_values("event_seconds")
        for _, win_row in wins.iterrows():
            win_time = win_row["event_seconds"]
            if win_row["x"] <= 50:
                continue
            after    = team_actions[team_actions["event_seconds"]>win_time]
            pc_after = after[after["type"].isin(["Pass","Carry"])]
            if len(pc_after) < 2:
                continue
            first_2 = pc_after.iloc[:2]
            if first_2["sequence_id"].nunique() > 1:
                continue
            if first_2.iloc[0]["x"] < 50:
                continue
            seq_id = first_2["sequence_id"].iloc[0]
            two = first_2.copy()
            two["action_rank"] = range(1, 3)
            two["sequence_id"] = seq_id
            rows.append(two)
            valid_ids.append(seq_id)
    if not rows:
        return
    seq_out = pd.concat(rows)
    X, seq_ids, team_ids = build_seq_vectors_fast_2(seq_out, valid_ids)
    labels = km.predict(scaler.transform(X))
    seq_res = pd.DataFrame({"sequence_id":seq_ids,"teamId":team_ids,"cluster":labels})
    profile = build_cluster_profile(seq_res, km, K, liga_map, games_per_team)
    save_profile(profile, liga, "transition")


def build_gegenpressing_profile(actions, df_all, liga_map, liga, games_per_team):
    print("  → Gegenpressing...")
    km, scaler = load_model("seq_cluster_press")
    K = km.n_clusters
    DEFENSIVE_SUCCESS = ["Tackle","Interception","BallRecovery"]
    HARD_BREAK_OPP    = ["Shot","Foul","CornerAwarded",
                          "ThrowIn","FreekickTaken","OffsideProwl"]
    ALLOWED_OPP       = ["Pass","Carry","TakeOn","BallTouch","Aerial"]
    press_rows = []
    for m_id, m_df in df_all.groupby("matchId"):
        m_df = m_df.sort_values("event_seconds").reset_index(drop=True)
        if m_df["teamId"].nunique() < 2:
            continue
        is_loss = (
            ((m_df["type"]=="Pass") & (m_df["outcomeType"]=="Unsuccessful"))
            | (m_df["type"]=="BallTouch")
        )
        for idx in m_df.index[is_loss].tolist():
            loss_row  = m_df.loc[idx]
            team_id   = loss_row["teamId"]
            loss_time = loss_row["event_seconds"]
            loss_x    = loss_row["endX"] if (
                loss_row["type"]=="Pass" and
                pd.notna(loss_row.get("endX"))
            ) else loss_row["x"]
            loss_y    = loss_row["endY"] if (
                loss_row["type"]=="Pass" and
                pd.notna(loss_row.get("endY"))
            ) else loss_row["y"]
            window = m_df.loc[idx+1:idx+15]
            if window.empty:
                continue
            recovered_row = None
            for _, r in window.iterrows():
                if r["teamId"] == team_id:
                    if (r["type"] in DEFENSIVE_SUCCESS and
                        r["outcomeType"]=="Successful"):
                        recovered_row = r
                    break
                else:
                    if r["type"] in HARD_BREAK_OPP:
                        break
                    if r["type"] not in ALLOWED_OPP:
                        break
            if recovered_row is None:
                continue
            press_rows.append({
                "teamId":           team_id,
                "loss_x":           loss_x,
                "loss_y":           loss_y,
                "recovery_x":       recovered_row["x"],
                "recovery_y":       recovered_row["y"],
                "delta_x":          recovered_row["x"] - loss_x,
                "delta_y":          recovered_row["y"] - loss_y,
                "time_to_recovery": recovered_row["event_seconds"] - loss_time,
            })
    if not press_rows:
        return
    press_df = pd.DataFrame(press_rows)
    feature_cols = [
        "loss_x","loss_y","recovery_x","recovery_y","delta_x","delta_y"
    ]
    X        = np.nan_to_num(press_df[feature_cols].values.astype(np.float64))
    X_scaled = scaler.transform(X)
    press_df["cluster"] = km.predict(X_scaled)

    # Recovery Speed — bleibt liga-relativ (Mittelwert, kein Zählwert)
    team_speed = press_df.groupby("teamId").agg(
        avg_time_to_recovery=("time_to_recovery","mean")
    ).reset_index()
    team_speed["liga"] = team_speed["teamId"].map(liga_map)
    team_speed["recovery_speed_zscore"] = np.nan
    for l in team_speed["liga"].unique():
        mask = team_speed["liga"] == l
        data = team_speed.loc[mask,"avg_time_to_recovery"]
        mean = data.mean()
        std  = data.std() + 1e-6
        team_speed.loc[mask,"recovery_speed_zscore"] = -(data-mean)/std

    profile = build_cluster_profile(
        press_df[["teamId","cluster"]], km, K, liga_map, games_per_team
    )
    profile = profile.merge(
        team_speed[["teamId","avg_time_to_recovery","recovery_speed_zscore"]],
        on="teamId", how="left"
    )
    save_profile(profile, liga, "gegenpressing")


def build_high_block_profile(actions, df_all, liga_map, liga, games_per_team):
    print("  → High Block...")
    km, scaler = load_model("high_block")
    K = km.n_clusters
    DEFENSIVE_SUCCESS = ["Tackle","Interception","BallRecovery"]
    opp_map = get_opp_map(df_all)
    rows = []
    for m_id, m_df in df_all.groupby("matchId"):
        m_df = m_df.sort_values("event_seconds").reset_index(drop=True)
        seq_order   = m_df.drop_duplicates("sequence_id",keep="first")["sequence_id"].reset_index(drop=True)
        seq_pos_map = {sid: i for i, sid in enumerate(seq_order)}
        m_df["seq_pos"] = m_df["sequence_id"].map(seq_pos_map)
        seq_groups  = {n: g for n, g in m_df.groupby("sequence_id")}
        seq_meta    = m_df.groupby("sequence_id").agg(
            team_id=("teamId","first"),
            start_x=("x","first"),
            seq_pos=("seq_pos","first"),
            n_teams=("teamId","nunique"),
        ).reset_index().sort_values("seq_pos")
        teams = m_df["teamId"].unique()
        for team_id in teams:
            opp_id = opp_map.get(m_id, {}).get(team_id)
            if not opp_id:
                continue
            candidates = seq_meta[
                (seq_meta["team_id"]==opp_id) &
                (seq_meta["n_teams"]==1) &
                (seq_meta["start_x"]<33) &
                seq_meta["start_x"].notna()
            ]
            for _, cand in candidates.iterrows():
                next_row = seq_meta[seq_meta["seq_pos"]==cand["seq_pos"]+1]
                if next_row.empty:
                    continue
                if next_row["team_id"].iloc[0] != team_id:
                    continue
                next_seq = seq_groups[next_row["sequence_id"].iloc[0]]
                our_wins = next_seq[
                    next_seq["type"].isin(DEFENSIVE_SUCCESS) &
                    (next_seq["outcomeType"]=="Successful")
                ]
                if our_wins.empty:
                    continue
                win_row = our_wins.iloc[0]
                rows.append({
                    "teamId": team_id,
                    "win_x":  win_row["x"],
                    "win_y":  win_row["y"],
                })
    if not rows:
        return
    hb_df    = pd.DataFrame(rows)
    X        = np.nan_to_num(hb_df[["win_x","win_y"]].values.astype(np.float64))
    hb_df["cluster"] = km.predict(scaler.transform(X))
    profile = build_cluster_profile(
        hb_df[["teamId","cluster"]], km, K, liga_map, games_per_team
    )
    save_profile(profile, liga, "high_block")


def build_mid_low_block_profile(actions, df_all, liga_map, liga, games_per_team):
    print("  → Mid/Low Block...")
    km, scaler = load_model("mid_low_block")
    K = km.n_clusters
    DEFENSIVE_SUCCESS = ["Tackle","Interception","BallRecovery"]
    opp_map = get_opp_map(df_all)
    rows = []
    for m_id, m_df in df_all.groupby("matchId"):
        m_df = m_df.sort_values("event_seconds").reset_index(drop=True)
        seq_order   = m_df.drop_duplicates("sequence_id",keep="first")["sequence_id"].reset_index(drop=True)
        seq_pos_map = {sid: i for i, sid in enumerate(seq_order)}
        m_df["seq_pos"] = m_df["sequence_id"].map(seq_pos_map)
        seq_groups  = {n: g for n, g in m_df.groupby("sequence_id")}
        seq_meta    = m_df.groupby("sequence_id").agg(
            team_id    =("teamId","first"),
            n_above_50 =("x", lambda x: (x>50).sum()),
            seq_pos    =("seq_pos","first"),
            n_teams    =("teamId","nunique"),
        ).reset_index().sort_values("seq_pos")
        teams = m_df["teamId"].unique()
        for team_id in teams:
            opp_id = opp_map.get(m_id, {}).get(team_id)
            if not opp_id:
                continue
            candidates = seq_meta[
                (seq_meta["team_id"]==opp_id) &
                (seq_meta["n_teams"]==1) &
                (seq_meta["n_above_50"]>=2)
            ]
            for _, cand in candidates.iterrows():
                next_row = seq_meta[seq_meta["seq_pos"]==cand["seq_pos"]+1]
                if next_row.empty:
                    continue
                if next_row["team_id"].iloc[0] != team_id:
                    continue
                next_seq = seq_groups[next_row["sequence_id"].iloc[0]]
                our_wins = next_seq[
                    next_seq["type"].isin(DEFENSIVE_SUCCESS) &
                    (next_seq["outcomeType"]=="Successful")
                ]
                if our_wins.empty:
                    continue
                win_row = our_wins.iloc[0]
                rows.append({
                    "teamId": team_id,
                    "win_x":  win_row["x"],
                    "win_y":  win_row["y"],
                })
    if not rows:
        return
    ml_df    = pd.DataFrame(rows)
    X        = np.nan_to_num(ml_df[["win_x","win_y"]].values.astype(np.float64))
    ml_df["cluster"] = km.predict(scaler.transform(X))
    profile = build_cluster_profile(
        ml_df[["teamId","cluster"]], km, K, liga_map, games_per_team
    )
    save_profile(profile, liga, "mid_low_block")


def build_prog_against_profile(actions, df_all, liga_map, liga, games_per_team):
    print("  → Progression gegen sich...")
    km, scaler = load_model("seq_cluster_prog_against")
    K = km.n_clusters
    opp_map = get_opp_map(df_all)
    rows, valid_ids = [], []
    for seq_id, seq_df in actions.groupby("sequence_id"):
        seq      = seq_df.sort_values("event_seconds")
        team_id  = seq.iloc[0]["teamId"]
        match_id = seq.iloc[0]["matchId"]
        opp_id   = opp_map.get(match_id, {}).get(team_id)
        if not opp_id:
            continue
        if seq.iloc[0]["x"] >= 33:
            continue
        crossing = seq[
            (seq["outcomeType"]=="Successful") &
            (seq["x"]<66) & (seq["endX"]>66) &
            (seq["type"].isin(["Pass","Carry"]))
        ]
        if len(crossing)==0:
            continue
        cross_idx = seq.index.get_loc(crossing.index[0])
        if cross_idx<1 or cross_idx>=len(seq)-1:
            continue
        before = seq.iloc[cross_idx-1]
        cross  = seq.iloc[cross_idx]
        after  = seq.iloc[cross_idx+1]
        if not all(r["type"] in ["Pass","Carry"] for r in [before,cross,after]):
            continue
        three = pd.DataFrame([before,cross,after])
        three["action_rank"] = range(1,4)
        three["sequence_id"] = seq_id
        three["teamId"]      = opp_id
        rows.append(three)
        valid_ids.append(seq_id)
    if not rows:
        return
    seq_out = pd.concat(rows)
    X, seq_ids, team_ids = build_seq_vectors_fast(seq_out, valid_ids)
    labels = km.predict(scaler.transform(X))
    seq_res = pd.DataFrame({"sequence_id":seq_ids,"teamId":team_ids,"cluster":labels})
    profile = build_cluster_profile(seq_res, km, K, liga_map, games_per_team)
    save_profile(profile, liga, "prog_against")


def build_shots_against_profile(actions, df_all, liga_map, liga, games_per_team):
    print("  → Schüsse gegen sich...")
    km, scaler = load_model("seq_cluster_shots_against")
    K = km.n_clusters
    opp_map      = get_opp_map(df_all)
    shot_seq_ids = get_shot_seq_ids(df_all)
    shots_all    = df_all[df_all["type"]=="Shot"].copy()
    shots_all["opponent_teamId"] = shots_all.apply(
        lambda r: opp_map.get(r["matchId"],{}).get(r["teamId"]), axis=1
    )
    relevant = actions[
        actions["sequence_id"].isin(shot_seq_ids)
    ].sort_values(["sequence_id","event_seconds"])
    rows, valid_ids = [], []
    for seq_id, seq_df in relevant.groupby("sequence_id"):
        shot_row = shots_all[shots_all["sequence_id"]==seq_id]
        if shot_row.empty:
            continue
        shot_time = shot_row["event_seconds"].min()
        opp_id    = shot_row["opponent_teamId"].iloc[0]
        if pd.isna(opp_id):
            continue
        pc_before = seq_df[seq_df["event_seconds"]<shot_time]
        if len(pc_before)<3:
            continue
        last_3 = pc_before.iloc[-3:].copy()
        times  = last_3["event_seconds"].values
        if not (times[1]>times[0] and times[2]>times[1]):
            continue
        last_3["action_rank"] = range(1,4)
        last_3["sequence_id"] = seq_id
        last_3["teamId"]      = opp_id
        rows.append(last_3)
        valid_ids.append(seq_id)
    if not rows:
        return
    seq_out = pd.concat(rows)
    X, seq_ids, team_ids = build_seq_vectors_fast(seq_out, valid_ids)
    labels = km.predict(scaler.transform(X))
    seq_res = pd.DataFrame({"sequence_id":seq_ids,"teamId":team_ids,"cluster":labels})
    profile = build_cluster_profile(seq_res, km, K, liga_map, games_per_team)
    save_profile(profile, liga, "shots_against")


def build_aufbau_abbruch_profile(actions, df_all, liga_map, liga, games_per_team):
    print("  → Aufbau-Abbrüche...")
    km, scaler = load_model("seq_cluster_ab")
    K = km.n_clusters
    rows, valid_ids = [], []
    for seq_id, seq_df in actions.groupby("sequence_id"):
        seq  = seq_df.sort_values("event_seconds")
        last = seq.iloc[-1]
        is_error = (
            (last["type"]=="Pass" and last["outcomeType"]=="Unsuccessful") or
            last["type"]=="BallTouch" or
            (last["type"]=="TakeOn" and last["outcomeType"]=="Unsuccessful")
        )
        if not is_error:
            continue
        if pd.isna(last["x"]) or last["x"]>=50:
            continue
        if len(seq)<2:
            continue
        before = seq.iloc[-2]
        if before["type"] not in ["Pass","Carry"]:
            continue
        two = pd.DataFrame([before,last])
        two["action_rank"] = range(1,3)
        two["sequence_id"] = seq_id
        rows.append(two)
        valid_ids.append(seq_id)
    if not rows:
        return
    seq_out = pd.concat(rows)
    X, seq_ids, team_ids = build_seq_vectors_fast_2(seq_out, valid_ids)
    labels = km.predict(scaler.transform(X))
    seq_res = pd.DataFrame({"sequence_id":seq_ids,"teamId":team_ids,"cluster":labels})
    profile = build_cluster_profile(seq_res, km, K, liga_map, games_per_team)
    save_profile(profile, liga, "aufbau_abbruch")


def build_formation_profile(df_all, liga_map, liga):
    print("  → Formation...")
    km, scaler = load_model("formation_cluster")
    passes_h1 = df_all[
        (df_all["type"]=="Pass") &
        (df_all["period"]=="FirstHalf") &
        df_all["x"].notna() & df_all["y"].notna() &
        df_all["playerId"].notna()
    ].copy()
    player_avg = passes_h1.groupby(
        ["matchId","teamId","playerId"]
    ).agg(avg_x=("x","mean"),avg_y=("y","mean"),n_passes=("x","count")).reset_index()
    player_avg = player_avg[player_avg["n_passes"]>=5]
    N_PLAYERS = 11
    form_rows = []
    for (match_id,team_id), group in player_avg.groupby(["matchId","teamId"]):
        if len(group)<9:
            continue
        sorted_p = group.sort_values("avg_x")
        if len(sorted_p)>N_PLAYERS:
            sorted_p = sorted_p.nlargest(N_PLAYERS,"n_passes").sort_values("avg_x")
        xs = sorted_p["avg_x"].values
        ys = sorted_p["avg_y"].values
        if len(xs)<N_PLAYERS:
            pad = N_PLAYERS-len(xs)
            xs  = np.concatenate([xs,[xs[-1]]*pad])
            ys  = np.concatenate([ys,[ys[-1]]*pad])
        outfield_x = xs[1:]
        outfield_y = ys[1:]
        third      = max(1,len(outfield_x)//3)
        form_rows.append({
            "matchId":match_id,"teamId":team_id,
            "n_def":((xs>=0)&(xs<33)).sum(),
            "n_mid":((xs>=33)&(xs<66)).sum(),
            "n_att":((xs>=66)&(xs<=100)).sum(),
            "def_line_x":outfield_x[:third].mean(),
            "mid_line_x":outfield_x[third:2*third].mean(),
            "att_line_x":outfield_x[2*third:].mean(),
            "def_line_y_std":outfield_y[:third].std() if third>1 else 0,
            "mid_line_y_std":outfield_y[third:2*third].std() if third>1 else 0,
            "att_line_y_std":outfield_y[2*third:].std() if (len(outfield_x)-2*third)>1 else 0,
            "compactness_x":outfield_x.std(),
            "compactness_y":outfield_y.std(),
            "depth_def_mid":outfield_x[third:2*third].mean()-outfield_x[:third].mean(),
            "depth_mid_att":outfield_x[2*third:].mean()-outfield_x[third:2*third].mean(),
            "total_depth":outfield_x[2*third:].mean()-outfield_x[:third].mean(),
        })
    if not form_rows:
        return
    form_df = pd.DataFrame(form_rows)
    feature_cols = [
        "n_def","n_mid","n_att",
        "def_line_x","mid_line_x","att_line_x",
        "def_line_y_std","mid_line_y_std","att_line_y_std",
        "compactness_x","compactness_y",
        "depth_def_mid","depth_mid_att","total_depth"
    ]
    team_avg = form_df.groupby("teamId")[feature_cols].mean().reset_index()
    X        = np.nan_to_num(team_avg[feature_cols].values.astype(np.float64))
    team_avg["cluster"] = km.predict(scaler.transform(X))
    team_avg["liga"]    = team_avg["teamId"].map(liga_map)
    save_profile(team_avg, liga, "formation")


def build_network_profile(df_all, liga_map, liga):
    print("  → Netzwerk...")
    km, scaler = load_model("network_cluster")
    N_ZONES_X = 4
    N_ZONES_Y = 3
    x_bins    = np.linspace(0,100,N_ZONES_X+1)
    y_bins    = np.linspace(0,100,N_ZONES_Y+1)
    N_ZONES   = N_ZONES_X * N_ZONES_Y
    zone_coords = {
        z: (
            (x_bins[z%N_ZONES_X]+x_bins[z%N_ZONES_X+1])/2,
            (y_bins[z//N_ZONES_X]+y_bins[z//N_ZONES_X+1])/2
        )
        for z in range(N_ZONES)
    }
    passes = df_all[
        (df_all["type"]=="Pass") &
        (df_all["outcomeType"]=="Successful") &
        df_all["x"].notna() & df_all["y"].notna() &
        df_all["endX"].notna() & df_all["endY"].notna()
    ].copy()

    def get_zone(x,y):
        zx = np.clip(np.digitize(x,x_bins)-1,0,N_ZONES_X-1)
        zy = np.clip(np.digitize(y,y_bins)-1,0,N_ZONES_Y-1)
        return zy*N_ZONES_X+zx

    passes["zone_start"] = get_zone(passes["x"].values,passes["y"].values)
    passes["zone_end"]   = get_zone(passes["endX"].values,passes["endY"].values)

    net_rows = []
    for team_id, team_passes in passes.groupby("teamId"):
        z_start = team_passes["zone_start"].values
        z_end   = team_passes["zone_end"].values
        x_start = z_start % N_ZONES_X
        x_end   = z_end   % N_ZONES_X
        forward_ratio  = (x_end>x_start).mean()
        lateral_ratio  = (x_end==x_start).mean()
        backward_ratio = (x_end<x_start).mean()
        avg_zone_jump  = np.abs(x_end-x_start).mean()
        pair_counts    = pd.Series(list(zip(z_start,z_end))).value_counts()
        probs          = pair_counts.values/pair_counts.values.sum()
        entropy        = -np.sum(probs*np.log(probs+1e-10))
        max_entropy    = np.log(len(pair_counts))
        norm_entropy   = entropy/max_entropy if max_entropy>0 else 0
        zone_activity  = np.bincount(np.concatenate([z_start,z_end]),minlength=N_ZONES)
        hub_zone       = zone_activity.argmax()
        hub_x,hub_y    = zone_coords[hub_zone]
        hub_conc       = zone_activity.max()/zone_activity.sum()
        net_rows.append({
            "teamId":           team_id,
            "forward_ratio":    forward_ratio,
            "lateral_ratio":    lateral_ratio,
            "backward_ratio":   backward_ratio,
            "avg_zone_jump":    avg_zone_jump,
            "network_entropy":  norm_entropy,
            "hub_x":            hub_x,
            "hub_y":            hub_y,
            "hub_concentration":hub_conc,
        })
    if not net_rows:
        return
    net_df = pd.DataFrame(net_rows)
    feature_cols = [
        "forward_ratio","lateral_ratio","backward_ratio",
        "avg_zone_jump","network_entropy",
        "hub_x","hub_y","hub_concentration"
    ]
    X = np.nan_to_num(net_df[feature_cols].values.astype(np.float64))
    net_df["cluster"] = km.predict(scaler.transform(X))
    net_df["liga"]    = net_df["teamId"].map(liga_map)
    save_profile(net_df, liga, "network")


def build_pass_score_profile(actions, df_all, liga_map, liga):
    print("  → Pass-Score...")
    try:
        import torch
        import torch.nn as nn

        class PassNet(nn.Module):
            def __init__(self, input_dim):
                super(PassNet, self).__init__()
                self.network = nn.Sequential(
                    nn.Linear(input_dim,64),nn.ReLU(),nn.Dropout(0.3),
                    nn.Linear(64,32),nn.ReLU(),nn.Dropout(0.2),
                    nn.Linear(32,16),nn.ReLU(),
                    nn.Linear(16,1),nn.Sigmoid()
                )
            def forward(self, x):
                return self.network(x).squeeze(1)

        with open(MODELS_DIR/"pass_model_config.pkl","rb") as f:
            ps_config = pickle.load(f)
        ps_mean      = np.array(ps_config["X_mean"])
        ps_std       = np.array(ps_config["X_std"])
        FEATURE_COLS = ps_config["feature_cols"]
        input_dim    = ps_config["input_dim"]

        ps_model = PassNet(input_dim=input_dim)
        ps_model.load_state_dict(
            torch.load(MODELS_DIR/"pass_model_weights.pt", map_location="cpu")
        )
        ps_model.eval()

        passes = df_all[
            (df_all["type"]=="Pass") &
            df_all["x"].notna() & df_all["y"].notna() &
            df_all["endX"].notna() & df_all["endY"].notna()
        ].copy()

        passes["target"]            = (passes["outcomeType"]=="Successful").astype(int)
        passes["delta_x"]           = passes["endX"]-passes["x"]
        passes["delta_y"]           = passes["endY"]-passes["y"]
        passes["distance"]          = np.sqrt(passes["delta_x"]**2+passes["delta_y"]**2)
        passes["angle"]             = np.arctan2(passes["delta_y"],passes["delta_x"])
        passes["forward"]           = (passes["delta_x"]>0).astype(int)
        passes["progressive"]       = (passes["delta_x"]>=10).astype(int)
        passes["switch"]            = (passes["distance"]>32).astype(int)
        passes["into_final_third"]  = (passes["endX"]>=66).astype(int)
        passes["into_box"]          = (
            (passes["endX"]>=83)&(passes["endY"]>=21)&(passes["endY"]<=79)
        ).astype(int)
        passes["centrality"]        = abs(passes["y"]-50)
        passes["buildup_height"]    = passes["x"]
        passes["sequence_position"] = passes.groupby("sequence_id").cumcount()+1

        DEFENSIVE_TYPES = ["Tackle","Interception","Challenge","BallRecovery"]
        press_rows = []
        for match_id in df_all["matchId"].unique():
            match = df_all[df_all["matchId"]==match_id]
            teams = match["teamId"].unique()
            if len(teams)<2:
                continue
            for team_id in teams:
                opp_id  = [t for t in teams if t!=team_id][0]
                opp_def = match[
                    (match["teamId"]==opp_id) &
                    (match["type"].isin(DEFENSIVE_TYPES))
                ]
                press_rows.append({
                    "matchId":      match_id,
                    "teamId":       team_id,
                    "gegner_avg_x": float(opp_def["x"].mean() if not opp_def.empty else 50.0)
                })
        press_lines = pd.DataFrame(press_rows)
        passes = passes.merge(press_lines, on=["matchId","teamId"], how="left")
        passes["gegner_avg_x"]   = passes["gegner_avg_x"].fillna(50.0)
        passes["under_pressure"] = 0
        passes = passes.fillna(0)

        for col in FEATURE_COLS:
            if col not in passes.columns:
                passes[col] = 0
        if len(passes)==0:
            return

        X      = passes[FEATURE_COLS].values.astype(np.float32)
        X_norm = (X-ps_mean)/(ps_std+1e-8)
        with torch.no_grad():
            predicted = ps_model(torch.FloatTensor(X_norm)).numpy()

        passes["predicted_success"] = predicted
        passes["actual_success"]    = passes["target"].astype(float)
        passes["pass_score"]        = passes["actual_success"]-passes["predicted_success"]

        team_ps = passes.groupby("teamId").agg(
            avg_pass_score=("pass_score","mean"),
            std_pass_score=("pass_score","std"),
            n_passes=("pass_score","count"),
            success_rate=("actual_success","mean"),
            avg_predicted=("predicted_success","mean"),
        ).reset_index()
        team_ps["liga"] = team_ps["teamId"].map(liga_map)

        team_ps["pass_score_zscore"] = np.nan
        for l in team_ps["liga"].dropna().unique():
            mask = team_ps["liga"]==l
            data = team_ps.loc[mask,"avg_pass_score"]
            mean = data.mean()
            std  = data.std()+1e-6
            team_ps.loc[mask,"pass_score_zscore"] = (data-mean)/std

        save_profile(team_ps, liga, "pass_score")

    except ImportError:
        print("    ⚠️  torch nicht installiert")
    except Exception as e:
        print(f"    ⚠️  Pass-Score Fehler: {e}")


def build_performance_profile(actions, df_all, liga_map, liga):
    print("  → Performance-Metriken...")

    shot_seq_ids = get_shot_seq_ids(df_all)
    opp_map      = get_opp_map(df_all)

    # Deep Buildup
    db_rows = []
    for seq_id, seq_df in actions.groupby("sequence_id"):
        seq     = seq_df.sort_values("event_seconds")
        team_id = seq.iloc[0]["teamId"]
        if seq.iloc[0]["x"]>=7:
            continue
        crossing = seq[
            (seq["outcomeType"]=="Successful") &
            (seq["x"]<50) & (seq["endX"]>=50) &
            (seq["type"].isin(["Pass","Carry"]))
        ]
        success = False
        if len(crossing)>0:
            cross_idx = seq.index.get_loc(crossing.index[0])
            after     = seq.iloc[cross_idx+1:]
            pc_after  = after[after["type"].isin(["Pass","Carry"])]
            if len(pc_after)>=1 and pc_after.iloc[0]["outcomeType"]=="Successful":
                success = True
        db_rows.append({"teamId":team_id,"success":success})

    team_db = pd.DataFrame(db_rows).groupby("teamId").agg(
        deep_buildup_rate=("success","mean")
    ).reset_index()

    # Buildup Success
    bu_rows = []
    for seq_id, seq_df in actions.groupby("sequence_id"):
        seq     = seq_df.sort_values("event_seconds")
        team_id = seq.iloc[0]["teamId"]
        if seq.iloc[0]["x"]>=33:
            continue
        crossing = seq[
            (seq["outcomeType"]=="Successful") &
            (seq["x"]<66) & (seq["endX"]>=66) &
            (seq["type"].isin(["Pass","Carry"]))
        ]
        success = False
        if len(crossing)>0:
            cross_idx = seq.index.get_loc(crossing.index[0])
            after     = seq.iloc[cross_idx+1:]
            pc_after  = after[after["type"].isin(["Pass","Carry"])]
            if len(pc_after)>=1 and pc_after.iloc[0]["outcomeType"]=="Successful":
                success = True
        bu_rows.append({"teamId":team_id,"success":success})

    team_bu = pd.DataFrame(bu_rows).groupby("teamId").agg(
        buildup_success_rate=("success","mean")
    ).reset_index()

    # Chance Threat
    ct_rows = []
    for seq_id, seq_df in actions.groupby("sequence_id"):
        seq     = seq_df.sort_values("event_seconds")
        team_id = seq.iloc[0]["teamId"]
        crossing = seq[
            (seq["outcomeType"]=="Successful") &
            (seq["x"]<50) & (seq["endX"]>=50) &
            (seq["type"].isin(["Pass","Carry"]))
        ]
        if len(crossing)==0:
            continue
        cross_idx = seq.index.get_loc(crossing.index[0])
        after     = seq.iloc[cross_idx+1:]
        pc_after  = after[after["type"].isin(["Pass","Carry"])]
        if len(pc_after)<1:
            continue
        if pc_after.iloc[0]["outcomeType"]!="Successful":
            continue
        ct_rows.append({"teamId":team_id,"ends_in_shot":seq_id in shot_seq_ids})

    team_ct = pd.DataFrame(ct_rows).groupby("teamId").agg(
        chance_threat_rate=("ends_in_shot","mean")
    ).reset_index() if ct_rows else pd.DataFrame(columns=["teamId","chance_threat_rate"])

    # xG mit echtem Modell
    shots_own = df_all[df_all["type"]=="Shot"].copy()
    shots_own = compute_xg(shots_own, df_all)

    team_shots = shots_own.groupby("teamId").agg(
        chance_quantity=("xg","count"),
        chance_quality =("xg","mean"),
        total_xg       =("xg","sum"),
    ).reset_index()

    # xGa
    shots_against = shots_own.copy()
    shots_against["opp_teamId"] = shots_against.apply(
        lambda r: opp_map.get(r["matchId"],{}).get(r["teamId"]), axis=1
    )
    shots_against = shots_against.dropna(subset=["opp_teamId"])
    team_xga = shots_against.groupby("opp_teamId").agg(
        xga_quality  =("xg","mean"),
        xga_total    =("xg","sum"),
        shots_against=("xg","count")
    ).reset_index().rename(columns={"opp_teamId":"teamId"})

    # Umschaltspiel
    BALL_WIN_TYPES = ["Tackle","Interception","BallRecovery"]
    ball_wins_opp  = df_all[
        df_all["type"].isin(BALL_WIN_TYPES) &
        (df_all["outcomeType"]=="Successful") &
        (df_all["x"]>50) & df_all["x"].notna()
    ].copy()
    counter_rows, quality_rows = [], []
    for (match_id,team_id), wins in ball_wins_opp.groupby(["matchId","teamId"]):
        team_actions = df_all[
            (df_all["matchId"]==match_id) & (df_all["teamId"]==team_id)
        ].sort_values("event_seconds")
        for _, win_row in wins.iterrows():
            win_time = win_row["event_seconds"]
            after    = team_actions[team_actions["event_seconds"]>win_time].head(15)
            if after.empty:
                continue
            shots_after = after[
                (after["type"]=="Shot") & (after["event_seconds"]<=win_time+10)
            ]
            has_shot = len(shots_after)>0
            counter_rows.append({"teamId":team_id,"ends_in_shot":has_shot})
            if has_shot:
                quality_rows.append({
                    "teamId":team_id,
                    "time_to_shot":shots_after.iloc[0]["event_seconds"]-win_time
                })

    team_counter = pd.DataFrame(counter_rows).groupby("teamId").agg(
        chance_counter_rate=("ends_in_shot","mean")
    ).reset_index() if counter_rows else pd.DataFrame(columns=["teamId","chance_counter_rate"])

    team_counter_q = pd.DataFrame(quality_rows).groupby("teamId").agg(
        avg_time_to_shot=("time_to_shot","mean")
    ).reset_index() if quality_rows else pd.DataFrame(columns=["teamId","avg_time_to_shot"])

    # PPDA
    DEFENSIVE_ACTIONS = ["Tackle","Interception","BallRecovery","Challenge"]
    ppda_rows = []
    for match_id, m_df in df_all.groupby("matchId"):
        teams = m_df["teamId"].unique()
        if len(teams)<2:
            continue
        for team_id in teams:
            opp_id  = opp_map.get(match_id,{}).get(team_id)
            if not opp_id:
                continue
            opp_pass = m_df[(m_df["teamId"]==opp_id)&(m_df["type"]=="Pass")&(m_df["x"]>50)]
            own_def  = m_df[(m_df["teamId"]==team_id)&(m_df["type"].isin(DEFENSIVE_ACTIONS))&(m_df["x"]>50)]
            if len(own_def)==0:
                continue
            ppda_rows.append({"teamId":team_id,"ppda":len(opp_pass)/len(own_def)})

    team_ppda = pd.DataFrame(ppda_rows).groupby("teamId").agg(
        avg_ppda=("ppda","mean")
    ).reset_index() if ppda_rows else pd.DataFrame(columns=["teamId","avg_ppda"])

    # Press Quality
    press_q_rows = []
    DEFENSIVE_SUCCESS = ["Tackle","Interception","BallRecovery"]
    HARD_BREAK = ["Shot","Foul","CornerAwarded","ThrowIn","FreekickTaken","OffsideProwl"]
    ALLOWED    = ["Pass","Carry","TakeOn","BallTouch","Aerial"]
    for m_id, m_df in df_all.groupby("matchId"):
        m_df = m_df.sort_values("event_seconds").reset_index(drop=True)
        is_loss = (
            ((m_df["type"]=="Pass")&(m_df["outcomeType"]=="Unsuccessful"))
            | (m_df["type"]=="BallTouch")
        )
        for idx in m_df.index[is_loss].tolist():
            loss_row  = m_df.loc[idx]
            team_id   = loss_row["teamId"]
            loss_time = loss_row["event_seconds"]
            window    = m_df.loc[idx+1:idx+15]
            if window.empty:
                continue
            for _, r in window.iterrows():
                if r["teamId"]==team_id:
                    if r["type"] in DEFENSIVE_SUCCESS and r["outcomeType"]=="Successful":
                        press_q_rows.append({
                            "teamId":team_id,
                            "time_to_recovery":r["event_seconds"]-loss_time
                        })
                    break
                else:
                    if r["type"] in HARD_BREAK:
                        break
                    if r["type"] not in ALLOWED:
                        break

    team_press_q = pd.DataFrame(press_q_rows).groupby("teamId").agg(
        avg_time_to_recovery=("time_to_recovery","mean")
    ).reset_index() if press_q_rows else pd.DataFrame(columns=["teamId","avg_time_to_recovery"])

    # Block Quality
    seq_first = actions.sort_values(
        ["sequence_id","event_seconds"]
    ).groupby("sequence_id").first()[["matchId","teamId","x"]].reset_index()
    seq_first = seq_first[seq_first["x"]<33].copy()
    seq_first["opp_teamId"] = seq_first.apply(
        lambda r: opp_map.get(r["matchId"],{}).get(r["teamId"]), axis=1
    )
    seq_first = seq_first.dropna(subset=["opp_teamId"])
    crossing_66 = set(actions[
        (actions["outcomeType"]=="Successful") &
        (actions["x"]<66) & (actions["endX"]>=66) &
        (actions["type"].isin(["Pass","Carry"]))
    ]["sequence_id"].unique())
    seq_first["opp_reached_66"] = seq_first["sequence_id"].isin(crossing_66)
    team_hbq = seq_first.groupby("opp_teamId").agg(
        high_block_quality=("opp_reached_66",lambda x: 1-x.mean())
    ).reset_index().rename(columns={"opp_teamId":"teamId"})

    cross_50_ids = set(actions[
        (actions["outcomeType"]=="Successful") &
        (actions["x"]<50) & (actions["endX"]>=50) &
        (actions["type"].isin(["Pass","Carry"]))
    ]["sequence_id"].unique())
    cross_66_ids = crossing_66

    def get_block_quality(cross_ids, shot_ids, opp_map, actions):
        seq_team  = actions.groupby("sequence_id")["teamId"].first().to_dict()
        seq_match = actions.groupby("sequence_id")["matchId"].first().to_dict()
        rows = []
        for seq_id in cross_ids:
            team_id  = seq_team.get(seq_id)
            match_id = seq_match.get(seq_id)
            opp_id   = opp_map.get(match_id,{}).get(team_id) if match_id else None
            if not opp_id:
                continue
            rows.append({"opp_teamId":opp_id,"ends_in_shot":seq_id in shot_ids})
        if not rows:
            return pd.DataFrame(columns=["teamId","quality"])
        df_r = pd.DataFrame(rows)
        return df_r.groupby("opp_teamId").agg(
            quality=("ends_in_shot",lambda x: 1-x.mean())
        ).reset_index().rename(columns={"opp_teamId":"teamId"})

    team_mbq = get_block_quality(cross_50_ids,shot_seq_ids,opp_map,actions).rename(columns={"quality":"mid_block_quality"})
    team_lbq = get_block_quality(cross_66_ids,shot_seq_ids,opp_map,actions).rename(columns={"quality":"low_block_quality"})

    # Zusammenführen
    games_per_team = df_all.groupby("teamId")["matchId"].nunique().reset_index()
    games_per_team.columns = ["teamId","n_games"]

    perf = team_db.merge(team_bu,      on="teamId",how="outer")
    perf = perf.merge(team_ct,         on="teamId",how="outer")
    perf = perf.merge(team_shots,      on="teamId",how="outer")
    perf = perf.merge(team_counter,    on="teamId",how="outer")
    perf = perf.merge(team_counter_q,  on="teamId",how="outer")
    perf = perf.merge(team_ppda,       on="teamId",how="outer")
    perf = perf.merge(team_press_q,    on="teamId",how="outer")
    perf = perf.merge(team_hbq,        on="teamId",how="outer")
    perf = perf.merge(team_mbq,        on="teamId",how="outer")
    perf = perf.merge(team_lbq,        on="teamId",how="outer")
    perf = perf.merge(team_xga,        on="teamId",how="outer")
    perf = perf.merge(games_per_team,  on="teamId",how="left")

    perf["liga"] = perf["teamId"].map(liga_map)

    for col in ["chance_quantity","total_xg","xga_total","shots_against"]:
        if col in perf.columns:
            perf[f"{col}_pg"] = (perf[col]/perf["n_games"]).round(3)

    zscore_cols = [
        "deep_buildup_rate","buildup_success_rate",
        "chance_threat_rate","chance_quality","chance_quantity_pg",
        "total_xg_pg","chance_counter_rate","avg_time_to_shot",
        "avg_ppda","avg_time_to_recovery","high_block_quality",
        "mid_block_quality","low_block_quality",
        "xga_quality","xga_total_pg","shots_against_pg"
    ]
    for col in zscore_cols:
        if col not in perf.columns:
            perf[col] = np.nan
        perf[f"{col}_zscore"] = np.nan

    for l in perf["liga"].dropna().unique():
        mask = perf["liga"]==l
        for col in zscore_cols:
            if col not in perf.columns:
                continue
            data = perf.loc[mask,col]
            mean = data.mean()
            std  = data.std()+1e-6
            perf.loc[mask,f"{col}_zscore"] = (data-mean)/std

    km_perf, scaler_perf = load_model("performance_cluster")
    perf_feat_cols = [f"{c}_zscore" for c in zscore_cols]
    X_perf = np.nan_to_num(perf[perf_feat_cols].values.astype(np.float64))
    perf["cluster"] = km_perf.predict(scaler_perf.transform(X_perf))

    save_profile(perf, liga, "performance")


def build_meta_cluster(liga):
    print("  → Z-Score Meta-Cluster...")
    km_meta, scaler_meta = load_model("meta_cluster_zscore")
    profile_names = [
        "abstoß","aufbau","mf","prog","shot",
        "transition","gegenpressing","high_block",
        "mid_low_block","prog_against","shots_against",
        "aufbau_abbruch"
    ]
    meta_dfs = []
    for name in profile_names:
        path = TEAM_PROFILES_DIR / liga / f"profiles_{name}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        zscore_cols = [c for c in df.columns if c.endswith("_zscore")]
        if not zscore_cols:
            continue
        renamed = df[["teamId"]+zscore_cols].rename(
            columns={c: f"{name}_{c}" for c in zscore_cols}
        )
        meta_dfs.append(renamed.set_index("teamId"))
    if not meta_dfs:
        return
    meta_combined = pd.concat(meta_dfs, axis=1).fillna(0)
    X_meta        = meta_combined.values.astype(np.float64)
    X_meta_scaled = scaler_meta.transform(X_meta)
    labels        = km_meta.predict(X_meta_scaled)
    meta_df       = meta_combined.reset_index()[["teamId"]].copy()
    meta_df["meta_cluster"] = labels
    save_profile(meta_df, liga, "meta_cluster")


# ══════════════════════════════════════════════════════════════════
#  HAUPT-PIPELINE
# ══════════════════════════════════════════════════════════════════

def build_profiles_for_liga(liga, saison):
    print(f"\n{'='*60}")
    print(f" PROFILE BUILDER: {liga} | {saison}")
    print(f"{'='*60}")

    clean_file = SAISON_CLEAN_DIR / f"{liga}_{saison}_clean.csv"
    if not clean_file.exists():
        print(f"  ⚠️  Clean-DB nicht gefunden: {clean_file.name}")
        return

    print(f"  Lade {clean_file.name}...")
    df_all = pd.read_csv(clean_file, low_memory=False)
    df_all["sequence_id"] = (
        df_all["liga"] + "_" + df_all["sequence_id"].astype(str)
    )

    print(f"  Events:    {len(df_all):,}")
    print(f"  Teams:     {df_all['teamId'].nunique()}")
    print(f"  Sequenzen: {df_all['sequence_id'].nunique():,}")

    liga_map      = df_all.groupby("teamId")["liga"].first().to_dict()
    games_per_team = df_all.groupby("teamId")["matchId"].nunique().to_dict()

    actions = df_all[df_all["type"].isin(["Pass","Carry"])].copy()
    actions["delta_x"]  = actions["endX"] - actions["x"]
    actions["delta_y"]  = actions["endY"] - actions["y"]
    actions["distance"] = np.sqrt(actions["delta_x"]**2+actions["delta_y"]**2)
    actions["angle"]    = np.arctan2(actions["delta_y"],actions["delta_x"])
    actions["is_carry"] = (actions["type"]=="Carry").astype(int)
    actions["zone_x"]   = (actions["x"]/100*4).astype(int).clip(0,3)

    print(f"\n  Berechne Profile...")

    build_abstoß_profile(actions, df_all, liga_map, liga, games_per_team)
    build_aufbau_profile(actions, df_all, liga_map, liga, games_per_team)
    build_mf_profile(actions, df_all, liga_map, liga, games_per_team)
    build_prog_profile(actions, df_all, liga_map, liga, games_per_team)
    build_shot_profile(actions, df_all, liga_map, liga, games_per_team)
    build_transition_profile(actions, df_all, liga_map, liga, games_per_team)
    build_gegenpressing_profile(actions, df_all, liga_map, liga, games_per_team)
    build_high_block_profile(actions, df_all, liga_map, liga, games_per_team)
    build_mid_low_block_profile(actions, df_all, liga_map, liga, games_per_team)
    build_prog_against_profile(actions, df_all, liga_map, liga, games_per_team)
    build_shots_against_profile(actions, df_all, liga_map, liga, games_per_team)
    build_aufbau_abbruch_profile(actions, df_all, liga_map, liga, games_per_team)
    build_formation_profile(df_all, liga_map, liga)
    build_pass_score_profile(actions, df_all, liga_map, liga)
    build_network_profile(df_all, liga_map, liga)
    build_performance_profile(actions, df_all, liga_map, liga)
    build_meta_cluster(liga)

    print(f"\n  ✅ Alle Profile für {liga} gespeichert")
    print(f"     → {TEAM_PROFILES_DIR / liga}/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--saison", type=str, required=True)
    parser.add_argument("--liga",   type=str, default=None)
    args = parser.parse_args()

    ligen = [args.liga] if args.liga else LIGEN
    for liga in ligen:
        build_profiles_for_liga(liga, args.saison)

    print(f"\n{'='*60}")
    print(f" ✅ FERTIG — alle Profile aktuell")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()