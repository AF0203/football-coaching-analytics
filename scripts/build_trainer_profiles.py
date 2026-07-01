"""
Trainer-Profile Builder
→ identische Struktur wie build_profiles.py
→ pro Trainer ein Ordner mit 17 Profile-CSVs
→ Z-Scores gegen liga-spezifische Referenz
→ cluster_X_pg statt cluster_X_freq (pro Spiel Normierung)

Aufruf:
    python3 scripts/build_trainer_profiles.py
    python3 scripts/build_trainer_profiles.py --trainer tr_006
"""

import sys, os, argparse, pickle, warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
))
from config import (
    PROCESSED_DIR, MODELS_DIR,
    TRAINER_DIR, TEAM_PROFILES_DIR
)

# ── Pfade ─────────────────────────────────────────────────────────
TRAINER_PROFILES_DIR = PROCESSED_DIR / "trainer_profiles"
TRAINER_PROFILES_DIR.mkdir(parents=True, exist_ok=True)
TRAINER_INDEX = TRAINER_DIR / "trainer_index.csv"

# ── Globale Liga-Variable (wird pro Trainer gesetzt) ──────────────
_current_liga = "bundesliga"

# ── Liga-Referenzen laden ─────────────────────────────────────────
def load_ref_dict(liga):
    ref_paths = {
        "bundesliga":     PROCESSED_DIR / "bl_reference.csv",
        "bundesliga_2":   PROCESSED_DIR / "bl2_reference.csv",
        "premier_league": PROCESSED_DIR / "pl_reference.csv",
        "la_liga":        PROCESSED_DIR / "laliga_reference.csv",
    }
    p = ref_paths.get(liga, ref_paths["bundesliga"])
    if not p.exists():
        p = ref_paths["bundesliga"]
    ref = pd.read_csv(p)
    return {
        (row["profile"], row["metric"]): {
            "mean": row["bl_mean"],
            "std":  row["bl_std"]
        }
        for _, row in ref.iterrows()
    }

_ref_cache = {}

def get_zscore(profile, metric, value, liga=None):
    """Liga-aware Z-Score Berechnung"""
    if liga is None:
        liga = _current_liga
    if liga not in _ref_cache:
        _ref_cache[liga] = load_ref_dict(liga)
    ref_dict = _ref_cache[liga]
    key = (profile, metric)
    if key not in ref_dict:
        # Fallback auf Bundesliga
        if "bundesliga" not in _ref_cache:
            _ref_cache["bundesliga"] = load_ref_dict("bundesliga")
        ref_dict = _ref_cache["bundesliga"]
        if key not in ref_dict:
            return np.nan
    ref = ref_dict[key]
    if ref["std"] == 0:
        return np.nan
    return (value - ref["mean"]) / ref["std"]


# ── Meta-Cluster Spalten laden ────────────────────────────────────
def get_meta_cluster_cols():
    profile_names = [
        "abstoß","aufbau","mf","prog","shot",
        "transition","gegenpressing","high_block",
        "mid_low_block","prog_against","shots_against",
        "aufbau_abbruch"
    ]
    cols = []
    for name in profile_names:
        path = TEAM_PROFILES_DIR / "bundesliga" / f"profiles_{name}.csv"
        if not path.exists():
            continue
        df_tmp = pd.read_csv(path)
        z_cols = [c for c in df_tmp.columns if c.endswith("_zscore")]
        for c in z_cols:
            cols.append(f"{name}_{c}")
    return cols

META_CLUSTER_COLS = get_meta_cluster_cols()
print(f"✅ Meta-Cluster Spalten: {len(META_CLUSTER_COLS)}")


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
    XG_MEAN = XG_STD = XG_FEATURE_COLS = None
    print(f"⚠️  xG-Modell nicht geladen: {e}")


def compute_xg(shots_df, df_all):
    import torch
    if XG_MODEL is None or len(shots_df) == 0:
        shots_df = shots_df.copy()
        shots_df["xg"] = 0.08
        return shots_df

    shots = shots_df.copy()
    shots["distance_to_goal"]   = np.sqrt((100 - shots["x"])**2 + (50 - shots["y"])**2)
    shots["angle_to_goal"]      = np.arctan2(abs(shots["y"] - 50), 100 - shots["x"])
    shots["is_central"]         = ((shots["y"] >= 30) & (shots["y"] <= 70)).astype(int)
    shots["in_box"]             = ((shots["x"] >= 83) & (shots["y"] >= 21) & (shots["y"] <= 79)).astype(int)
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

    shots["shot_zone"] = shots.apply(lambda r: get_shot_zone(r["x"], r["y"]), axis=1)
    seq_stats = df_all.groupby("sequence_id").agg(seq_length=("type","count")).reset_index()
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
#  PROCESSING
# ══════════════════════════════════════════════════════════════════

def process_trainer_raw(raw_file, trainer_id):
    sys.path.insert(0, str(Path(__file__).parent))
    from processing import prepare_db

    df = pd.read_csv(raw_file, low_memory=False)

    for col in df.columns:
        if df[col].apply(lambda x: isinstance(x, list)).any():
            df[col] = df[col].astype(str)

    liga_id = df["liga_id"].iloc[0] if "liga_id" in df.columns else "bundesliga"
    df_clean = prepare_db(df, liga_id, "trainer")

    def fix_team_id(x):
        x = str(x)
        if x.endswith(f"opp_{trainer_id}"):
            return f"opp_{trainer_id}"
        elif x.endswith(trainer_id):
            return trainer_id
        return x

    df_clean["teamId"]  = df_clean["teamId"].apply(fix_team_id)
    df_clean["liga_id"] = liga_id

    n_trainer = (df_clean["teamId"]==trainer_id).sum()
    n_opp     = (df_clean["teamId"]==f"opp_{trainer_id}").sum()
    print(f"  Trainer-Events: {n_trainer:,}")
    print(f"  Gegner-Events:  {n_opp:,}")

    if n_trainer == 0:
        raise ValueError(f"Keine Trainer-Events!")

    return df_clean


# ══════════════════════════════════════════════════════════════════
#  HELPER
# ══════════════════════════════════════════════════════════════════

def load_model(name):
    with open(MODELS_DIR / f"{name}_km.pkl", "rb") as f:
        km = pickle.load(f)
    with open(MODELS_DIR / f"{name}_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return km, scaler


def save_trainer_profile(df, trainer_id, name):
    out_file = TRAINER_PROFILES_DIR / f"profiles_{name}.csv"
    if out_file.exists():
        existing = pd.read_csv(out_file)
        existing = existing[existing["teamId"] != trainer_id]
        combined = pd.concat([existing, df], ignore_index=True)
    else:
        combined = df
    combined.to_csv(out_file, index=False)
    print(f"    ✅ profiles_{name}.csv ({len(combined)} Trainer)")


def build_cluster_profile_trainer(seq_df, km, K,
                                   profile_name, trainer_id,
                                   n_games):
    cluster_counts = seq_df[
        seq_df["teamId"]==trainer_id
    ].groupby("cluster").size()

    n_total = cluster_counts.sum() if len(cluster_counts)>0 else 0

    row = {
        "teamId":  trainer_id,
        "liga":    _current_liga,
        "n_seqs":  n_total,
        "n_games": n_games,
    }

    for c in range(K):
        n  = int(cluster_counts.get(c, 0))
        pg = n / n_games if n_games > 0 else 0
        row[f"cluster_{c}_abs"]    = n
        row[f"cluster_{c}_pg"]     = round(pg, 6)
        row[f"cluster_{c}_zscore"] = get_zscore(
            profile_name, f"cluster_{c}_pg", pg
        )
    return pd.DataFrame([row])


def build_seq_vectors_fast(seq_df, complete_ids):
    df = seq_df[seq_df["sequence_id"].isin(complete_ids)].copy()
    feature_cols_base = ["x","y","endX","endY","delta_x","delta_y","distance","angle","is_carry","zone_x"]
    if "delta_x" not in df.columns:
        df["delta_x"]  = df["endX"] - df["x"]
        df["delta_y"]  = df["endY"] - df["y"]
        df["distance"] = np.sqrt(df["delta_x"]**2+df["delta_y"]**2)
        df["angle"]    = np.arctan2(df["delta_y"],df["delta_x"])
        df["is_carry"] = (df["type"]=="Carry").astype(int)
        df["zone_x"]   = (df["x"]/100*4).astype(int).clip(0,3)

    rank1 = df[df["action_rank"]==1][["sequence_id","teamId"]+feature_cols_base].set_index("sequence_id")
    rank2 = df[df["action_rank"]==2][["sequence_id"]+feature_cols_base].set_index("sequence_id")
    rank3 = df[df["action_rank"]==3][["sequence_id"]+feature_cols_base].set_index("sequence_id")

    rank1.columns = ["teamId"]+[f"a1_{c}" for c in feature_cols_base]
    rank2.columns = [f"a2_{c}" for c in feature_cols_base]
    rank3.columns = [f"a3_{c}" for c in feature_cols_base]

    combined = rank1.join(rank2,how="inner").join(rank3,how="inner")
    combined["total_distance"] = combined["a1_distance"]+combined["a2_distance"]+combined["a3_distance"]
    feature_cols_all = ([f"a1_{c}" for c in feature_cols_base]+[f"a2_{c}" for c in feature_cols_base]+[f"a3_{c}" for c in feature_cols_base]+["total_distance"])
    X        = np.nan_to_num(combined[feature_cols_all].values.astype(np.float64))
    seq_ids  = combined.index.tolist()
    team_ids = combined["teamId"].tolist()
    return X, seq_ids, team_ids


def build_seq_vectors_fast_2(seq_df, complete_ids):
    df = seq_df[seq_df["sequence_id"].isin(complete_ids)].copy()
    feature_cols_base = ["x","y","endX","endY","delta_x","delta_y","distance","angle","is_carry","zone_x"]
    if "delta_x" not in df.columns:
        df["delta_x"]  = df["endX"] - df["x"]
        df["delta_y"]  = df["endY"] - df["y"]
        df["distance"] = np.sqrt(df["delta_x"]**2+df["delta_y"]**2)
        df["angle"]    = np.arctan2(df["delta_y"],df["delta_x"])
        df["is_carry"] = (df["type"]=="Carry").astype(int)
        df["zone_x"]   = (df["x"]/100*4).astype(int).clip(0,3)

    rank1 = df[df["action_rank"]==1][["sequence_id","teamId"]+feature_cols_base].set_index("sequence_id")
    rank2 = df[df["action_rank"]==2][["sequence_id"]+feature_cols_base].set_index("sequence_id")

    rank1.columns = ["teamId"]+[f"a1_{c}" for c in feature_cols_base]
    rank2.columns = [f"a2_{c}" for c in feature_cols_base]

    combined = rank1.join(rank2,how="inner")
    combined["total_distance"] = combined["a1_distance"]+combined["a2_distance"]
    feature_cols_all = ([f"a1_{c}" for c in feature_cols_base]+[f"a2_{c}" for c in feature_cols_base]+["total_distance"])
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
        unique = list(set(teams))
        if len(unique)==2:
            opp_map[m_id] = {unique[0]:unique[1], unique[1]:unique[0]}
    return opp_map


# ══════════════════════════════════════════════════════════════════
#  PROFIL-FUNKTIONEN
# ══════════════════════════════════════════════════════════════════

def build_abstoß_profile(actions, df_all, trainer_id, n_games):
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
        first_3["action_rank"] = range(1,4)
        first_3["sequence_id"] = seq_id
        rows.append(first_3)
        valid_ids.append(seq_id)
    if not rows:
        return
    seq_out = pd.concat(rows)
    X, sids, tids = build_seq_vectors_fast(seq_out, valid_ids)
    labels  = km.predict(scaler.transform(X))
    seq_res = pd.DataFrame({"sequence_id":sids,"teamId":tids,"cluster":labels})
    profile = build_cluster_profile_trainer(seq_res, km, K, "abstoß", trainer_id, n_games)
    save_trainer_profile(profile, trainer_id, "abstoß")


def build_aufbau_profile(actions, df_all, trainer_id, n_games):
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
        first_3["action_rank"] = range(1,4)
        first_3["sequence_id"] = seq_id
        rows.append(first_3)
        valid_ids.append(seq_id)
    if not rows:
        return
    seq_out = pd.concat(rows)
    X, sids, tids = build_seq_vectors_fast(seq_out, valid_ids)
    labels  = km.predict(scaler.transform(X))
    seq_res = pd.DataFrame({"sequence_id":sids,"teamId":tids,"cluster":labels})
    profile = build_cluster_profile_trainer(seq_res, km, K, "aufbau", trainer_id, n_games)
    save_trainer_profile(profile, trainer_id, "aufbau")


def build_mf_profile(actions, df_all, trainer_id, n_games):
    km, scaler = load_model("seq_cluster_mf")
    K = km.n_clusters
    rows, valid_ids = [], []
    for seq_id, seq_df in actions.groupby("sequence_id"):
        seq = seq_df.sort_values("event_seconds")
        crossing = seq[
            (seq["outcomeType"]=="Successful") &
            (seq["x"]<50) & (seq["endX"]>=50) &
            (seq["type"].isin(["Pass","Carry"]))
        ]
        if len(crossing)==0:
            continue
        cross_idx = seq.index.get_loc(crossing.index[0])
        remaining = seq.iloc[cross_idx+1:]
        pc_after  = remaining[remaining["type"].isin(["Pass","Carry"])]
        if len(pc_after)<3:
            continue
        first_3 = pc_after.iloc[:3].copy()
        idx_1 = remaining.index.get_loc(first_3.index[0])
        idx_2 = remaining.index.get_loc(first_3.index[1])
        idx_3 = remaining.index.get_loc(first_3.index[2])
        if not (idx_2-idx_1==1 and idx_3-idx_2==1):
            continue
        first_3["action_rank"] = range(1,4)
        first_3["sequence_id"] = seq_id
        rows.append(first_3)
        valid_ids.append(seq_id)
    if not rows:
        return
    seq_out = pd.concat(rows)
    X, sids, tids = build_seq_vectors_fast(seq_out, valid_ids)
    labels  = km.predict(scaler.transform(X))
    seq_res = pd.DataFrame({"sequence_id":sids,"teamId":tids,"cluster":labels})
    profile = build_cluster_profile_trainer(seq_res, km, K, "mf", trainer_id, n_games)
    save_trainer_profile(profile, trainer_id, "mf")


def build_prog_profile(actions, df_all, trainer_id, n_games):
    km, scaler = load_model("seq_cluster_prog")
    K = km.n_clusters
    rows, valid_ids = [], []
    for seq_id, seq_df in actions.groupby("sequence_id"):
        seq = seq_df.sort_values("event_seconds")
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
        rows.append(three)
        valid_ids.append(seq_id)
    if not rows:
        return
    seq_out = pd.concat(rows)
    X, sids, tids = build_seq_vectors_fast(seq_out, valid_ids)
    labels  = km.predict(scaler.transform(X))
    seq_res = pd.DataFrame({"sequence_id":sids,"teamId":tids,"cluster":labels})
    profile = build_cluster_profile_trainer(seq_res, km, K, "prog", trainer_id, n_games)
    save_trainer_profile(profile, trainer_id, "prog")


def build_shot_profile(actions, df_all, trainer_id, n_games):
    km, scaler = load_model("seq_cluster_shot")
    K = km.n_clusters
    shot_seq_ids = get_shot_seq_ids(df_all)
    shots_all    = df_all[df_all["type"]=="Shot"].copy()
    relevant     = actions[actions["sequence_id"].isin(shot_seq_ids)].sort_values(["sequence_id","event_seconds"])
    rows, valid_ids = [], []
    for seq_id, seq_df in relevant.groupby("sequence_id"):
        shot_time = shots_all[shots_all["sequence_id"]==seq_id]["event_seconds"].min()
        pc_before = seq_df[seq_df["event_seconds"]<shot_time]
        if len(pc_before)<3:
            continue
        last_3 = pc_before.iloc[-3:]
        times  = last_3["event_seconds"].values
        if not (times[1]>times[0] and times[2]>times[1]):
            continue
        last_3 = last_3.copy()
        last_3["action_rank"] = range(1,4)
        last_3["sequence_id"] = seq_id
        rows.append(last_3)
        valid_ids.append(seq_id)
    if not rows:
        return
    seq_out = pd.concat(rows)
    X, sids, tids = build_seq_vectors_fast(seq_out, valid_ids)
    labels  = km.predict(scaler.transform(X))
    seq_res = pd.DataFrame({"sequence_id":sids,"teamId":tids,"cluster":labels})
    profile = build_cluster_profile_trainer(seq_res, km, K, "shot", trainer_id, n_games)
    save_trainer_profile(profile, trainer_id, "shot")


def build_transition_profile(actions, df_all, trainer_id, n_games):
    km, scaler = load_model("seq_cluster_tr")
    K = km.n_clusters
    BALL_WIN_TYPES = ["Tackle","Interception","BallRecovery"]
    ball_wins = df_all[
        df_all["type"].isin(BALL_WIN_TYPES) &
        (df_all["outcomeType"]=="Successful") &
        (df_all["x"]>50) & df_all["x"].notna() &
        (df_all["teamId"]==trainer_id)
    ].copy()
    rows, valid_ids = [], []
    for (match_id,team_id), wins in ball_wins.groupby(["matchId","teamId"]):
        team_act = actions[(actions["matchId"]==match_id)&(actions["teamId"]==team_id)].sort_values("event_seconds")
        for _, win_row in wins.iterrows():
            win_time = win_row["event_seconds"]
            if win_row["x"] <= 50:
                continue
            after    = team_act[team_act["event_seconds"]>win_time]
            pc_after = after[after["type"].isin(["Pass","Carry"])]
            if len(pc_after)<2:
                continue
            first_2 = pc_after.iloc[:2]
            if first_2["sequence_id"].nunique()>1:
                continue
            if first_2.iloc[0]["x"]<50:
                continue
            seq_id = first_2["sequence_id"].iloc[0]
            two = first_2.copy()
            two["action_rank"] = range(1,3)
            two["sequence_id"] = seq_id
            rows.append(two)
            valid_ids.append(seq_id)
    if not rows:
        return
    seq_out = pd.concat(rows)
    X, sids, tids = build_seq_vectors_fast_2(seq_out, valid_ids)
    labels  = km.predict(scaler.transform(X))
    seq_res = pd.DataFrame({"sequence_id":sids,"teamId":tids,"cluster":labels})
    profile = build_cluster_profile_trainer(seq_res, km, K, "transition", trainer_id, n_games)
    save_trainer_profile(profile, trainer_id, "transition")


def build_gegenpressing_profile(actions, df_all, trainer_id, n_games):
    km, scaler = load_model("seq_cluster_press")
    K = km.n_clusters
    DEFENSIVE_SUCCESS = ["Tackle","Interception","BallRecovery"]
    HARD_BREAK = ["Shot","Foul","CornerAwarded","ThrowIn","FreekickTaken","OffsideProwl"]
    ALLOWED    = ["Pass","Carry","TakeOn","BallTouch","Aerial"]
    press_rows = []
    for m_id, m_df in df_all.groupby("matchId"):
        m_df = m_df.sort_values("event_seconds").reset_index(drop=True)
        if m_df["teamId"].nunique()<2:
            continue
        is_loss = (
            ((m_df["type"]=="Pass")&(m_df["outcomeType"]=="Unsuccessful")&(m_df["teamId"]==trainer_id))
            |((m_df["type"]=="BallTouch")&(m_df["teamId"]==trainer_id))
        )
        for idx in m_df.index[is_loss].tolist():
            loss_row  = m_df.loc[idx]
            loss_time = loss_row["event_seconds"]
            loss_x    = loss_row["endX"] if (loss_row["type"]=="Pass" and pd.notna(loss_row.get("endX"))) else loss_row["x"]
            loss_y    = loss_row["endY"] if (loss_row["type"]=="Pass" and pd.notna(loss_row.get("endY"))) else loss_row["y"]
            if pd.isna(loss_x) or pd.isna(loss_y):
                continue
            window = m_df.loc[idx+1:idx+15]
            if window.empty:
                continue
            recovered_row = None
            for _, r in window.iterrows():
                if r["teamId"]==trainer_id:
                    if r["type"] in DEFENSIVE_SUCCESS and r["outcomeType"]=="Successful":
                        recovered_row = r
                    break
                else:
                    if r["type"] in HARD_BREAK:
                        break
                    if r["type"] not in ALLOWED:
                        break
            if recovered_row is None:
                continue
            if pd.isna(recovered_row["x"]) or pd.isna(recovered_row["y"]):
                continue
            press_rows.append({
                "teamId":           trainer_id,
                "loss_x":           loss_x,
                "loss_y":           loss_y,
                "recovery_x":       recovered_row["x"],
                "recovery_y":       recovered_row["y"],
                "delta_x":          recovered_row["x"]-loss_x,
                "delta_y":          recovered_row["y"]-loss_y,
                "time_to_recovery": recovered_row["event_seconds"]-loss_time,
            })
    if not press_rows:
        return
    press_df  = pd.DataFrame(press_rows)
    feat_cols = ["loss_x","loss_y","recovery_x","recovery_y","delta_x","delta_y"]
    press_df[feat_cols] = press_df[feat_cols].fillna(0)
    X_press = np.nan_to_num(press_df[feat_cols].values.astype(np.float64))
    press_df["cluster"] = km.predict(scaler.transform(X_press))
    profile = build_cluster_profile_trainer(press_df[["teamId","cluster"]], km, K, "gegenpressing", trainer_id, n_games)
    avg_recovery = press_df[press_df["teamId"]==trainer_id]["time_to_recovery"].mean()
    profile["avg_time_to_recovery"] = avg_recovery
    profile["recovery_speed_zscore"] = get_zscore("gegenpressing","avg_time_to_recovery",avg_recovery)
    save_trainer_profile(profile, trainer_id, "gegenpressing")


def build_high_block_profile(actions, df_all, trainer_id, n_games):
    km, scaler = load_model("high_block")
    K = km.n_clusters
    DEFENSIVE_SUCCESS = ["Tackle","Interception","BallRecovery"]
    opp_id = f"opp_{trainer_id}"
    rows   = []
    for m_id, m_df in df_all.groupby("matchId"):
        m_df = m_df.sort_values("event_seconds").reset_index(drop=True)
        seq_order   = m_df.drop_duplicates("sequence_id",keep="first")["sequence_id"].reset_index(drop=True)
        seq_pos_map = {sid:i for i,sid in enumerate(seq_order)}
        m_df["seq_pos"] = m_df["sequence_id"].map(seq_pos_map)
        seq_groups  = {n:g for n,g in m_df.groupby("sequence_id")}
        seq_meta    = m_df.groupby("sequence_id").agg(
            team_id=("teamId","first"),start_x=("x","first"),
            seq_pos=("seq_pos","first"),n_teams=("teamId","nunique"),
        ).reset_index().sort_values("seq_pos")
        candidates = seq_meta[(seq_meta["team_id"]==opp_id)&(seq_meta["n_teams"]==1)&(seq_meta["start_x"]<33)&seq_meta["start_x"].notna()]
        for _, cand in candidates.iterrows():
            next_row = seq_meta[seq_meta["seq_pos"]==cand["seq_pos"]+1]
            if next_row.empty or next_row["team_id"].iloc[0]!=trainer_id:
                continue
            next_seq = seq_groups[next_row["sequence_id"].iloc[0]]
            our_wins = next_seq[next_seq["type"].isin(DEFENSIVE_SUCCESS)&(next_seq["outcomeType"]=="Successful")]
            if our_wins.empty:
                continue
            win_row = our_wins.iloc[0]
            rows.append({"teamId":trainer_id,"win_x":win_row["x"],"win_y":win_row["y"]})
    if not rows:
        return
    hb_df = pd.DataFrame(rows)
    X_hb  = np.nan_to_num(hb_df[["win_x","win_y"]].values.astype(np.float64))
    hb_df["cluster"] = km.predict(scaler.transform(X_hb))
    profile = build_cluster_profile_trainer(hb_df[["teamId","cluster"]], km, K, "high_block", trainer_id, n_games)
    save_trainer_profile(profile, trainer_id, "high_block")


def build_mid_low_block_profile(actions, df_all, trainer_id, n_games):
    km, scaler = load_model("mid_low_block")
    K = km.n_clusters
    DEFENSIVE_SUCCESS = ["Tackle","Interception","BallRecovery"]
    opp_id = f"opp_{trainer_id}"
    rows   = []
    for m_id, m_df in df_all.groupby("matchId"):
        m_df = m_df.sort_values("event_seconds").reset_index(drop=True)
        seq_order   = m_df.drop_duplicates("sequence_id",keep="first")["sequence_id"].reset_index(drop=True)
        seq_pos_map = {sid:i for i,sid in enumerate(seq_order)}
        m_df["seq_pos"] = m_df["sequence_id"].map(seq_pos_map)
        seq_groups  = {n:g for n,g in m_df.groupby("sequence_id")}
        seq_meta    = m_df.groupby("sequence_id").agg(
            team_id=("teamId","first"),n_above_50=("x",lambda x:(x>50).sum()),
            seq_pos=("seq_pos","first"),n_teams=("teamId","nunique"),
        ).reset_index().sort_values("seq_pos")
        candidates = seq_meta[(seq_meta["team_id"]==opp_id)&(seq_meta["n_teams"]==1)&(seq_meta["n_above_50"]>=2)]
        for _, cand in candidates.iterrows():
            next_row = seq_meta[seq_meta["seq_pos"]==cand["seq_pos"]+1]
            if next_row.empty or next_row["team_id"].iloc[0]!=trainer_id:
                continue
            next_seq = seq_groups[next_row["sequence_id"].iloc[0]]
            our_wins = next_seq[next_seq["type"].isin(DEFENSIVE_SUCCESS)&(next_seq["outcomeType"]=="Successful")]
            if our_wins.empty:
                continue
            win_row = our_wins.iloc[0]
            rows.append({"teamId":trainer_id,"win_x":win_row["x"],"win_y":win_row["y"]})
    if not rows:
        return
    ml_df = pd.DataFrame(rows)
    X_ml  = np.nan_to_num(ml_df[["win_x","win_y"]].values.astype(np.float64))
    ml_df["cluster"] = km.predict(scaler.transform(X_ml))
    profile = build_cluster_profile_trainer(ml_df[["teamId","cluster"]], km, K, "mid_low_block", trainer_id, n_games)
    save_trainer_profile(profile, trainer_id, "mid_low_block")


def build_prog_against_profile(actions, df_all, trainer_id, n_games):
    km, scaler = load_model("seq_cluster_prog_against")
    K = km.n_clusters
    opp_id      = f"opp_{trainer_id}"
    actions_opp = df_all[(df_all["type"].isin(["Pass","Carry"]))&(df_all["teamId"]==opp_id)].copy()
    if "delta_x" not in actions_opp.columns:
        actions_opp["delta_x"]  = actions_opp["endX"]-actions_opp["x"]
        actions_opp["delta_y"]  = actions_opp["endY"]-actions_opp["y"]
        actions_opp["distance"] = np.sqrt(actions_opp["delta_x"]**2+actions_opp["delta_y"]**2)
        actions_opp["angle"]    = np.arctan2(actions_opp["delta_y"],actions_opp["delta_x"])
        actions_opp["is_carry"] = (actions_opp["type"]=="Carry").astype(int)
        actions_opp["zone_x"]   = (actions_opp["x"]/100*4).astype(int).clip(0,3)
    rows, valid_ids = [], []
    for seq_id, seq_df in actions_opp.groupby("sequence_id"):
        seq = seq_df.sort_values("event_seconds")
        if seq.iloc[0]["x"] >= 33:
            continue
        crossing = seq[(seq["outcomeType"]=="Successful")&(seq["x"]<66)&(seq["endX"]>66)&(seq["type"].isin(["Pass","Carry"]))]
        if len(crossing)==0:
            continue
        cross_idx = seq.index.get_loc(crossing.index[0])
        if cross_idx<1 or cross_idx>=len(seq)-1:
            continue
        before = seq.iloc[cross_idx-1]; cross = seq.iloc[cross_idx]; after = seq.iloc[cross_idx+1]
        if not all(r["type"] in ["Pass","Carry"] for r in [before,cross,after]):
            continue
        three = pd.DataFrame([before,cross,after])
        three["action_rank"] = range(1,4)
        three["sequence_id"] = seq_id
        three["teamId"]      = trainer_id
        rows.append(three)
        valid_ids.append(seq_id)
    if not rows:
        return
    seq_out = pd.concat(rows)
    X, sids, tids = build_seq_vectors_fast(seq_out, valid_ids)
    labels  = km.predict(scaler.transform(X))
    seq_res = pd.DataFrame({"sequence_id":sids,"teamId":[trainer_id]*len(sids),"cluster":labels})
    profile = build_cluster_profile_trainer(seq_res, km, K, "prog_against", trainer_id, n_games)
    save_trainer_profile(profile, trainer_id, "prog_against")


def build_shots_against_profile(actions, df_all, trainer_id, n_games):
    km, scaler = load_model("seq_cluster_shots_against")
    K = km.n_clusters
    opp_id    = f"opp_{trainer_id}"
    shots_opp = df_all[(df_all["type"]=="Shot")&(df_all["teamId"]==opp_id)].copy()
    shot_seq_ids_opp = set(shots_opp["sequence_id"].unique())
    actions_opp = df_all[(df_all["type"].isin(["Pass","Carry"]))&(df_all["teamId"]==opp_id)].copy()
    if "delta_x" not in actions_opp.columns:
        actions_opp["delta_x"]  = actions_opp["endX"]-actions_opp["x"]
        actions_opp["delta_y"]  = actions_opp["endY"]-actions_opp["y"]
        actions_opp["distance"] = np.sqrt(actions_opp["delta_x"]**2+actions_opp["delta_y"]**2)
        actions_opp["angle"]    = np.arctan2(actions_opp["delta_y"],actions_opp["delta_x"])
        actions_opp["is_carry"] = (actions_opp["type"]=="Carry").astype(int)
        actions_opp["zone_x"]   = (actions_opp["x"]/100*4).astype(int).clip(0,3)
    relevant = actions_opp[actions_opp["sequence_id"].isin(shot_seq_ids_opp)].sort_values(["sequence_id","event_seconds"])
    rows, valid_ids = [], []
    for seq_id, seq_df in relevant.groupby("sequence_id"):
        shot_time = shots_opp[shots_opp["sequence_id"]==seq_id]["event_seconds"].min()
        pc_before = seq_df[seq_df["event_seconds"]<shot_time]
        if len(pc_before)<3:
            continue
        last_3 = pc_before.iloc[-3:].copy()
        times  = last_3["event_seconds"].values
        if not (times[1]>times[0] and times[2]>times[1]):
            continue
        last_3["action_rank"] = range(1,4)
        last_3["sequence_id"] = seq_id
        last_3["teamId"]      = trainer_id
        rows.append(last_3)
        valid_ids.append(seq_id)
    if not rows:
        return
    seq_out = pd.concat(rows)
    X, sids, tids = build_seq_vectors_fast(seq_out, valid_ids)
    labels  = km.predict(scaler.transform(X))
    seq_res = pd.DataFrame({"sequence_id":sids,"teamId":[trainer_id]*len(sids),"cluster":labels})
    profile = build_cluster_profile_trainer(seq_res, km, K, "shots_against", trainer_id, n_games)
    save_trainer_profile(profile, trainer_id, "shots_against")


def build_aufbau_abbruch_profile(actions, df_all, trainer_id, n_games):
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
    X, sids, tids = build_seq_vectors_fast_2(seq_out, valid_ids)
    labels  = km.predict(scaler.transform(X))
    seq_res = pd.DataFrame({"sequence_id":sids,"teamId":tids,"cluster":labels})
    profile = build_cluster_profile_trainer(seq_res, km, K, "aufbau_abbruch", trainer_id, n_games)
    save_trainer_profile(profile, trainer_id, "aufbau_abbruch")


def build_pass_score_profile(actions, df_all, trainer_id):
    try:
        import torch
        import torch.nn as nn

        class PassNet(nn.Module):
            def __init__(self, input_dim):
                super(PassNet,self).__init__()
                self.network = nn.Sequential(
                    nn.Linear(input_dim,64),nn.ReLU(),nn.Dropout(0.3),
                    nn.Linear(64,32),nn.ReLU(),nn.Dropout(0.2),
                    nn.Linear(32,16),nn.ReLU(),
                    nn.Linear(16,1),nn.Sigmoid()
                )
            def forward(self,x):
                return self.network(x).squeeze(1)

        with open(MODELS_DIR/"pass_model_config.pkl","rb") as f:
            ps_config = pickle.load(f)
        ps_mean      = np.array(ps_config["X_mean"])
        ps_std       = np.array(ps_config["X_std"])
        FEATURE_COLS = ps_config["feature_cols"]
        input_dim    = ps_config["input_dim"]

        ps_model = PassNet(input_dim=input_dim)
        ps_model.load_state_dict(torch.load(MODELS_DIR/"pass_model_weights.pt", map_location="cpu"))
        ps_model.eval()

        passes = df_all[(df_all["type"]=="Pass")&(df_all["teamId"]==trainer_id)&df_all["x"].notna()&df_all["endX"].notna()].copy()
        passes["target"]           = (passes["outcomeType"]=="Successful").astype(int)
        passes["delta_x"]          = passes["endX"]-passes["x"]
        passes["delta_y"]          = passes["endY"]-passes["y"]
        passes["distance"]         = np.sqrt(passes["delta_x"]**2+passes["delta_y"]**2)
        passes["angle"]            = np.arctan2(passes["delta_y"],passes["delta_x"])
        passes["forward"]          = (passes["delta_x"]>0).astype(int)
        passes["progressive"]      = (passes["delta_x"]>=10).astype(int)
        passes["switch"]           = (passes["distance"]>32).astype(int)
        passes["into_final_third"] = (passes["endX"]>=66).astype(int)
        passes["into_box"]         = ((passes["endX"]>=83)&(passes["endY"]>=21)&(passes["endY"]<=79)).astype(int)
        passes["centrality"]       = abs(passes["y"]-50)
        passes["buildup_height"]   = passes["x"]
        passes["sequence_position"]= passes.groupby("sequence_id").cumcount()+1
        passes["gegner_avg_x"]     = 50.0
        passes["under_pressure"]   = 0

        for col in FEATURE_COLS:
            if col not in passes.columns:
                passes[col] = 0
        passes = passes.dropna(subset=FEATURE_COLS).fillna(0)
        if len(passes)==0:
            return

        X_ps   = passes[FEATURE_COLS].values.astype(np.float32)
        X_norm = (X_ps-ps_mean)/(ps_std+1e-8)
        with torch.no_grad():
            predicted = ps_model(torch.FloatTensor(X_norm)).numpy()

        passes["pass_score"] = passes["target"].astype(float) - predicted
        avg_ps = passes["pass_score"].mean()

        ps_df = pd.DataFrame([{
            "teamId":            trainer_id,
            "liga":              _current_liga,
            "avg_pass_score":    avg_ps,
            "pass_score_zscore": get_zscore("pass_score","avg_pass_score",avg_ps),
            "n_passes":          len(passes),
        }])
        save_trainer_profile(ps_df, trainer_id, "pass_score")

    except ImportError:
        print("      ⚠️  torch nicht installiert")
    except Exception as e:
        print(f"      ⚠️  Pass-Score: {e}")


def build_formation_profile(df_all, trainer_id):
    km, scaler = load_model("formation_cluster")
    passes_h1 = df_all[
        (df_all["type"]=="Pass")&(df_all["period"]=="FirstHalf")&
        (df_all["teamId"]==trainer_id)&df_all["x"].notna()&df_all["y"].notna()&df_all["playerId"].notna()
    ].copy()
    player_avg = passes_h1.groupby(["matchId","teamId","playerId"]).agg(avg_x=("x","mean"),avg_y=("y","mean"),n_passes=("x","count")).reset_index()
    player_avg = player_avg[player_avg["n_passes"]>=5]
    N_PLAYERS  = 11
    form_rows  = []
    for (match_id,team_id), group in player_avg.groupby(["matchId","teamId"]):
        if len(group)<9:
            continue
        sorted_p = group.sort_values("avg_x")
        if len(sorted_p)>N_PLAYERS:
            sorted_p = sorted_p.nlargest(N_PLAYERS,"n_passes").sort_values("avg_x")
        xs = sorted_p["avg_x"].values; ys = sorted_p["avg_y"].values
        if len(xs)<N_PLAYERS:
            pad = N_PLAYERS-len(xs)
            xs  = np.concatenate([xs,[xs[-1]]*pad])
            ys  = np.concatenate([ys,[ys[-1]]*pad])
        outfield_x = xs[1:]; outfield_y = ys[1:]
        third = max(1,len(outfield_x)//3)
        form_rows.append({
            "matchId":match_id,"teamId":team_id,
            "n_def":((xs>=0)&(xs<33)).sum(),"n_mid":((xs>=33)&(xs<66)).sum(),"n_att":((xs>=66)&(xs<=100)).sum(),
            "def_line_x":outfield_x[:third].mean(),"mid_line_x":outfield_x[third:2*third].mean(),"att_line_x":outfield_x[2*third:].mean(),
            "def_line_y_std":outfield_y[:third].std() if third>1 else 0,
            "mid_line_y_std":outfield_y[third:2*third].std() if third>1 else 0,
            "att_line_y_std":outfield_y[2*third:].std() if (len(outfield_x)-2*third)>1 else 0,
            "compactness_x":outfield_x.std(),"compactness_y":outfield_y.std(),
            "depth_def_mid":outfield_x[third:2*third].mean()-outfield_x[:third].mean(),
            "depth_mid_att":outfield_x[2*third:].mean()-outfield_x[third:2*third].mean(),
            "total_depth":outfield_x[2*third:].mean()-outfield_x[:third].mean(),
        })
    if not form_rows:
        return
    form_df   = pd.DataFrame(form_rows)
    feat_cols = ["n_def","n_mid","n_att","def_line_x","mid_line_x","att_line_x","def_line_y_std","mid_line_y_std","att_line_y_std","compactness_x","compactness_y","depth_def_mid","depth_mid_att","total_depth"]
    team_avg  = form_df.groupby("teamId")[feat_cols].mean().reset_index()
    X_form    = np.nan_to_num(team_avg[feat_cols].values.astype(np.float64))
    team_avg["cluster"] = km.predict(scaler.transform(X_form))[0]
    team_avg["liga"]    = _current_liga
    for col in feat_cols:
        val = team_avg[col].iloc[0]
        team_avg[f"{col}_zscore"] = get_zscore("formation", col, val)
    save_trainer_profile(team_avg, trainer_id, "formation")


def build_network_profile(df_all, trainer_id):
    km, scaler = load_model("network_cluster")
    N_ZONES_X = 4; N_ZONES_Y = 3
    x_bins    = np.linspace(0,100,N_ZONES_X+1); y_bins = np.linspace(0,100,N_ZONES_Y+1)
    N_ZONES   = N_ZONES_X * N_ZONES_Y
    zone_coords = {z:((x_bins[z%N_ZONES_X]+x_bins[z%N_ZONES_X+1])/2,(y_bins[z//N_ZONES_X]+y_bins[z//N_ZONES_X+1])/2) for z in range(N_ZONES)}
    passes_net = df_all[(df_all["type"]=="Pass")&(df_all["outcomeType"]=="Successful")&(df_all["teamId"]==trainer_id)&df_all["x"].notna()&df_all["endX"].notna()].copy()

    def get_zone(x,y):
        zx = np.clip(np.digitize(x,x_bins)-1,0,N_ZONES_X-1)
        zy = np.clip(np.digitize(y,y_bins)-1,0,N_ZONES_Y-1)
        return zy*N_ZONES_X+zx

    if len(passes_net)==0:
        return

    passes_net["zone_start"] = get_zone(passes_net["x"].values,passes_net["y"].values)
    passes_net["zone_end"]   = get_zone(passes_net["endX"].values,passes_net["endY"].values)
    z_start = passes_net["zone_start"].values; z_end = passes_net["zone_end"].values
    x_s = z_start%N_ZONES_X; x_e = z_end%N_ZONES_X
    forward_ratio  = (x_e>x_s).mean(); lateral_ratio = (x_e==x_s).mean(); backward_ratio = (x_e<x_s).mean()
    avg_zone_jump  = np.abs(x_e-x_s).mean()
    pair_counts    = pd.Series(list(zip(z_start,z_end))).value_counts()
    probs          = pair_counts.values/pair_counts.values.sum()
    entropy        = -np.sum(probs*np.log(probs+1e-10))
    max_entropy    = np.log(len(pair_counts))
    norm_entropy   = entropy/max_entropy if max_entropy>0 else 0
    zone_activity  = np.bincount(np.concatenate([z_start,z_end]),minlength=N_ZONES)
    hub_zone       = zone_activity.argmax()
    hub_x,hub_y   = zone_coords[hub_zone]
    hub_conc       = zone_activity.max()/zone_activity.sum()

    net_features = {"forward_ratio":forward_ratio,"lateral_ratio":lateral_ratio,"backward_ratio":backward_ratio,"avg_zone_jump":avg_zone_jump,"network_entropy":norm_entropy,"hub_x":hub_x,"hub_y":hub_y,"hub_concentration":hub_conc}
    feat_cols = list(net_features.keys())
    X_net = np.array([[net_features[c] for c in feat_cols]])
    net_df = pd.DataFrame([{"teamId":trainer_id,"liga":_current_liga,**net_features}])
    net_df["cluster"] = km.predict(scaler.transform(X_net))[0]
    for col in feat_cols:
        net_df[f"{col}_zscore"] = get_zscore("network", col, net_features[col])
    save_trainer_profile(net_df, trainer_id, "network")


def build_performance_profile(actions, df_all, trainer_id):
    opp_id       = f"opp_{trainer_id}"
    shot_seq_ids = get_shot_seq_ids(df_all)
    opp_map      = get_opp_map(df_all)

    db_rows = []
    for seq_id, seq_df in actions.groupby("sequence_id"):
        seq = seq_df.sort_values("event_seconds")
        if seq.iloc[0]["x"]>=7:
            continue
        crossing = seq[(seq["outcomeType"]=="Successful")&(seq["x"]<50)&(seq["endX"]>=50)&(seq["type"].isin(["Pass","Carry"]))]
        success = False
        if len(crossing)>0:
            cross_idx = seq.index.get_loc(crossing.index[0])
            after     = seq.iloc[cross_idx+1:]
            pc_after  = after[after["type"].isin(["Pass","Carry"])]
            if len(pc_after)>=1 and pc_after.iloc[0]["outcomeType"]=="Successful":
                success = True
        db_rows.append({"success":success})

    bu_rows = []
    for seq_id, seq_df in actions.groupby("sequence_id"):
        seq = seq_df.sort_values("event_seconds")
        if seq.iloc[0]["x"]>=33:
            continue
        crossing = seq[(seq["outcomeType"]=="Successful")&(seq["x"]<66)&(seq["endX"]>=66)&(seq["type"].isin(["Pass","Carry"]))]
        success = False
        if len(crossing)>0:
            cross_idx = seq.index.get_loc(crossing.index[0])
            after     = seq.iloc[cross_idx+1:]
            pc_after  = after[after["type"].isin(["Pass","Carry"])]
            if len(pc_after)>=1 and pc_after.iloc[0]["outcomeType"]=="Successful":
                success = True
        bu_rows.append({"success":success})

    ct_rows = []
    for seq_id, seq_df in actions.groupby("sequence_id"):
        seq = seq_df.sort_values("event_seconds")
        crossing = seq[(seq["outcomeType"]=="Successful")&(seq["x"]<50)&(seq["endX"]>=50)&(seq["type"].isin(["Pass","Carry"]))]
        if len(crossing)==0:
            continue
        cross_idx = seq.index.get_loc(crossing.index[0])
        after     = seq.iloc[cross_idx+1:]
        pc_after  = after[after["type"].isin(["Pass","Carry"])]
        if len(pc_after)<1 or pc_after.iloc[0]["outcomeType"]!="Successful":
            continue
        ct_rows.append({"ends_in_shot":seq_id in shot_seq_ids})

    shots_tr  = df_all[(df_all["type"]=="Shot")&(df_all["teamId"]==trainer_id)].copy()
    shots_opp = df_all[(df_all["type"]=="Shot")&(df_all["teamId"]==opp_id)].copy()
    shots_tr  = compute_xg(shots_tr, df_all)
    shots_opp = compute_xg(shots_opp, df_all)

    n_games = df_all[df_all["teamId"]==trainer_id]["matchId"].nunique()

    BALL_WIN_TYPES = ["Tackle","Interception","BallRecovery"]
    ball_wins_tr   = df_all[df_all["type"].isin(BALL_WIN_TYPES)&(df_all["outcomeType"]=="Successful")&(df_all["x"]>50)&df_all["x"].notna()&(df_all["teamId"]==trainer_id)].copy()
    counter_rows, quality_rows = [], []
    for (match_id,team_id), wins in ball_wins_tr.groupby(["matchId","teamId"]):
        team_ev = df_all[(df_all["matchId"]==match_id)&(df_all["teamId"]==team_id)].sort_values("event_seconds")
        for _, win_row in wins.iterrows():
            win_time    = win_row["event_seconds"]
            after       = team_ev[team_ev["event_seconds"]>win_time].head(15)
            if after.empty:
                continue
            shots_after = after[(after["type"]=="Shot")&(after["event_seconds"]<=win_time+10)]
            has_shot    = len(shots_after)>0
            counter_rows.append({"ends_in_shot":has_shot})
            if has_shot:
                quality_rows.append({"time_to_shot":shots_after.iloc[0]["event_seconds"]-win_time})

    DEFENSIVE_ACTIONS = ["Tackle","Interception","BallRecovery","Challenge"]
    ppda_rows = []
    for match_id, m_df in df_all.groupby("matchId"):
        opp_pass = m_df[(m_df["teamId"]==opp_id)&(m_df["type"]=="Pass")&(m_df["x"]>50)]
        own_def  = m_df[(m_df["teamId"]==trainer_id)&(m_df["type"].isin(DEFENSIVE_ACTIONS))&(m_df["x"]>50)]
        if len(own_def)==0:
            continue
        ppda_rows.append(len(opp_pass)/len(own_def))

    press_q_rows = []
    DEFENSIVE_SUCCESS = ["Tackle","Interception","BallRecovery"]
    HARD_BREAK = ["Shot","Foul","CornerAwarded","ThrowIn","FreekickTaken","OffsideProwl"]
    ALLOWED    = ["Pass","Carry","TakeOn","BallTouch","Aerial"]
    for m_id, m_df in df_all.groupby("matchId"):
        m_df = m_df.sort_values("event_seconds").reset_index(drop=True)
        is_loss = (((m_df["type"]=="Pass")&(m_df["outcomeType"]=="Unsuccessful")&(m_df["teamId"]==trainer_id))|((m_df["type"]=="BallTouch")&(m_df["teamId"]==trainer_id)))
        for idx in m_df.index[is_loss].tolist():
            loss_time = m_df.loc[idx,"event_seconds"]
            window    = m_df.loc[idx+1:idx+15]
            if window.empty:
                continue
            for _, r in window.iterrows():
                if r["teamId"]==trainer_id:
                    if r["type"] in DEFENSIVE_SUCCESS and r["outcomeType"]=="Successful":
                        press_q_rows.append({"time_to_recovery":r["event_seconds"]-loss_time})
                    break
                else:
                    if r["type"] in HARD_BREAK:
                        break
                    if r["type"] not in ALLOWED:
                        break

    seq_opp_deep = df_all[(df_all["type"].isin(["Pass","Carry"]))&(df_all["teamId"]==opp_id)].sort_values(["sequence_id","event_seconds"]).groupby("sequence_id").first()[["matchId","teamId","x"]].reset_index()
    seq_opp_deep = seq_opp_deep[seq_opp_deep["x"]<33]
    crossing_66_seqs = set(df_all[(df_all["type"].isin(["Pass","Carry"]))&(df_all["teamId"]==opp_id)&(df_all["outcomeType"]=="Successful")&(df_all["x"]<66)&(df_all["endX"]>=66)]["sequence_id"].unique())
    opp_cross_50 = set(df_all[(df_all["type"].isin(["Pass","Carry"]))&(df_all["teamId"]==opp_id)&(df_all["outcomeType"]=="Successful")&(df_all["x"]<50)&(df_all["endX"]>=50)]["sequence_id"].unique())
    shots_opp_seqs = set(shots_opp["sequence_id"].unique())

    perf = pd.DataFrame([{"teamId":trainer_id}])
    if db_rows:
        perf["deep_buildup_rate"] = pd.DataFrame(db_rows)["success"].mean()
    if bu_rows:
        perf["buildup_success_rate"] = pd.DataFrame(bu_rows)["success"].mean()
    if ct_rows:
        perf["chance_threat_rate"] = pd.DataFrame(ct_rows)["ends_in_shot"].mean()
    if len(shots_tr)>0 and n_games>0:
        perf["chance_quantity"]  = len(shots_tr)
        perf["chance_quality"]   = shots_tr["xg"].mean()
        perf["total_xg"]         = shots_tr["xg"].sum()
    if counter_rows:
        perf["chance_counter_rate"] = pd.DataFrame(counter_rows)["ends_in_shot"].mean()
    if quality_rows:
        perf["avg_time_to_shot"] = pd.DataFrame(quality_rows)["time_to_shot"].mean()
    if ppda_rows:
        perf["avg_ppda"] = np.mean(ppda_rows)
    if press_q_rows:
        perf["avg_time_to_recovery"] = pd.DataFrame(press_q_rows)["time_to_recovery"].mean()
    if len(shots_opp)>0 and n_games>0:
        perf["shots_against"] = len(shots_opp)
        perf["xga_quality"]   = shots_opp["xg"].mean()
        perf["xga_total"]     = shots_opp["xg"].sum()
    if len(seq_opp_deep)>0:
        seq_opp_deep["reached_66"] = seq_opp_deep["sequence_id"].isin(crossing_66_seqs)
        perf["high_block_quality"] = 1-seq_opp_deep["reached_66"].mean()
    if opp_cross_50:
        in_shot = sum(1 for s in opp_cross_50 if s in shots_opp_seqs)
        perf["mid_block_quality"] = 1-in_shot/len(opp_cross_50)
    if crossing_66_seqs:
        in_shot = sum(1 for s in crossing_66_seqs if s in shots_opp_seqs)
        perf["low_block_quality"] = 1-in_shot/len(crossing_66_seqs)

    perf["liga"]    = _current_liga
    perf["n_games"] = n_games

    for col in ["chance_quantity","total_xg","xga_total","shots_against"]:
        if col in perf.columns:
            perf[f"{col}_pg"] = (perf[col]/n_games).round(3)

    zscore_cols = ["deep_buildup_rate","buildup_success_rate","chance_threat_rate","chance_quality","chance_quantity_pg","total_xg_pg","chance_counter_rate","avg_time_to_shot","avg_ppda","avg_time_to_recovery","high_block_quality","mid_block_quality","low_block_quality","xga_quality","xga_total_pg","shots_against_pg"]
    for col in zscore_cols:
        if col in perf.columns:
            perf[f"{col}_zscore"] = get_zscore("performance", col, perf[col].iloc[0])
        else:
            perf[f"{col}_zscore"] = np.nan

    km_perf, scaler_perf = load_model("performance_cluster")
    perf_vector = np.nan_to_num(np.array([[perf[f"{c}_zscore"].iloc[0] if f"{c}_zscore" in perf.columns else 0 for c in zscore_cols]]))
    perf["cluster"] = km_perf.predict(scaler_perf.transform(perf_vector))[0]
    save_trainer_profile(perf, trainer_id, "performance")


def build_meta_cluster_trainer(trainer_id):
    km_meta, scaler_meta = load_model("meta_cluster_zscore")
    profile_names = ["abstoß","aufbau","mf","prog","shot","transition","gegenpressing","high_block","mid_low_block","prog_against","shots_against","aufbau_abbruch"]
    meta_dfs = []
    for name in profile_names:
        path = TRAINER_PROFILES_DIR / f"profiles_{name}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df = df[df["teamId"]==trainer_id]
        if df.empty:
            continue
        zscore_cols = [c for c in df.columns if c.endswith("_zscore")]
        if not zscore_cols:
            continue
        renamed = df[["teamId"]+zscore_cols].rename(columns={c: f"{name}_{c}" for c in zscore_cols})
        meta_dfs.append(renamed.set_index("teamId"))
    if not meta_dfs:
        print(f"    ⚠️  Keine Profile für Meta-Cluster")
        return
    meta_combined = pd.concat(meta_dfs, axis=1).fillna(0)
    for col in META_CLUSTER_COLS:
        if col not in meta_combined.columns:
            meta_combined[col] = 0
    meta_combined = meta_combined[META_CLUSTER_COLS]
    X_meta = meta_combined.values.astype(np.float64)
    labels = km_meta.predict(scaler_meta.transform(X_meta))
    meta_df = pd.DataFrame([{"teamId":trainer_id,"meta_cluster":labels[0]}])
    save_trainer_profile(meta_df, trainer_id, "meta_cluster")


# ══════════════════════════════════════════════════════════════════
#  HAUPT-PIPELINE
# ══════════════════════════════════════════════════════════════════

def build_profiles_for_trainer(trainer_id, trainer_name,
                                verein, liga, raw_file):
    global _current_liga
    _current_liga = liga if liga else "bundesliga"

    print(f"\n{'─'*60}")
    print(f"  {trainer_name} ({trainer_id}) — Liga: {_current_liga}")
    print(f"{'─'*60}")

    try:
        df_clean = process_trainer_raw(raw_file, trainer_id)
    except Exception as e:
        print(f"  ❌ Processing Fehler: {e}")
        return

    df_clean["sequence_id"] = trainer_id + "_" + df_clean["sequence_id"].astype(str)
    n_games = df_clean[df_clean["teamId"]==trainer_id]["matchId"].nunique()

    print(f"  Events:    {len(df_clean):,}")
    print(f"  Spiele:    {n_games}")

    actions = df_clean[(df_clean["type"].isin(["Pass","Carry"]))&(df_clean["teamId"]==trainer_id)].copy()
    if "delta_x" not in actions.columns:
        actions["delta_x"]  = actions["endX"] - actions["x"]
        actions["delta_y"]  = actions["endY"] - actions["y"]
        actions["distance"] = np.sqrt(actions["delta_x"]**2+actions["delta_y"]**2)
        actions["angle"]    = np.arctan2(actions["delta_y"],actions["delta_x"])
        actions["is_carry"] = (actions["type"]=="Carry").astype(int)
        actions["zone_x"]   = (actions["x"]/100*4).astype(int).clip(0,3)

    print(f"\n  Berechne Profile...")

    for fname, func, args in [
        ("Abstoß",              build_abstoß_profile,         (actions, df_clean, trainer_id, n_games)),
        ("Aufbau",              build_aufbau_profile,          (actions, df_clean, trainer_id, n_games)),
        ("Mittelfeld",          build_mf_profile,              (actions, df_clean, trainer_id, n_games)),
        ("Progression",         build_prog_profile,            (actions, df_clean, trainer_id, n_games)),
        ("Vor Schuss",          build_shot_profile,            (actions, df_clean, trainer_id, n_games)),
        ("Transition",          build_transition_profile,      (actions, df_clean, trainer_id, n_games)),
        ("Gegenpressing",       build_gegenpressing_profile,   (actions, df_clean, trainer_id, n_games)),
        ("High Block",          build_high_block_profile,      (actions, df_clean, trainer_id, n_games)),
        ("Mid/Low Block",       build_mid_low_block_profile,   (actions, df_clean, trainer_id, n_games)),
        ("Prog. gegen sich",    build_prog_against_profile,    (actions, df_clean, trainer_id, n_games)),
        ("Schüsse gegen sich",  build_shots_against_profile,   (actions, df_clean, trainer_id, n_games)),
        ("Aufbau-Abbrüche",     build_aufbau_abbruch_profile,  (actions, df_clean, trainer_id, n_games)),
        ("Pass-Score",          build_pass_score_profile,      (actions, df_clean, trainer_id)),
        ("Formation",           build_formation_profile,       (df_clean, trainer_id)),
        ("Netzwerk",            build_network_profile,         (df_clean, trainer_id)),
        ("Performance",         build_performance_profile,     (actions, df_clean, trainer_id)),
        ("Meta-Cluster",        build_meta_cluster_trainer,    (trainer_id,)),
    ]:
        try:
            print(f"    → {fname}...")
            func(*args)
        except Exception as e:
            print(f"      ⚠️  {e}")

    print(f"  ✅ {trainer_name} — alle Profile gespeichert")


def build_all_trainer_profiles(trainer_filter=None):
    trainer_index = pd.read_csv(TRAINER_INDEX)
    if trainer_filter:
        trainer_index = trainer_index[trainer_index["trainer_id"]==trainer_filter]
    trainer_index = trainer_index.drop_duplicates(subset=["trainer_id"])

    print(f"\n{'='*60}")
    print(f" TRAINER PROFILE BUILDER")
    print(f" {len(trainer_index)} Trainer zu verarbeiten")
    print(f"{'='*60}")

    for _, trainer in trainer_index.iterrows():
        trainer_id   = trainer["trainer_id"]
        trainer_name = trainer["trainer_name"]
        raw_file     = TRAINER_DIR / f"{trainer_id}_raw.csv"

        if not raw_file.exists():
            print(f"\n  ⚠️  {trainer_name}: Rohdaten fehlen")
            continue

        try:
            build_profiles_for_trainer(
                trainer_id   = trainer_id,
                trainer_name = trainer_name,
                verein       = trainer.get("verein", ""),
                liga         = trainer.get("liga_id", "bundesliga"),
                raw_file     = raw_file
            )
        except Exception as e:
            print(f"\n  ❌ Fehler bei {trainer_name}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"✅ FERTIG")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainer", type=str, default=None)
    args = parser.parse_args()
    build_all_trainer_profiles(trainer_filter=args.trainer)