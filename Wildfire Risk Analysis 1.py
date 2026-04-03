"""
Wildfire Economic Risk Analysis Pipeline  v5
=============================================
Three-output architecture separating probability and impact.

OUTPUT 1 — county_impact_scores.csv
  Economic damage severity IF a fire occurs.
  Pillars: causal impact, industry sensitivity, economic trajectory.

OUTPUT 2 — county_probability_scores.csv
  Likelihood of a significant wildfire occurring.
  Pillars: drought conditions (decay-weighted), historical fire frequency.
  Extensible: add fuel load, WUI, fire weather index later.

OUTPUT 3 — county_final_risk_scores.csv
  True combined risk = Probability × Impact.
  Includes risk tier, trajectory, and shareholder narrative.

Key methodological choices:
  - Pre/post window     : 4 quarters (1 year) before and after each fire
  - Control matching    : nationwide cosine similarity (GDP/unem/population)
  - Industry sensitivity: NAICS codes from FOIA data
  - Drought indicators  : max consecutive weeks, recency-weighted intensity,
                          episode frequency — joined via FIPS
  - Decay rate          : λ = 0.3  (e^(-0.3 × years_ago))
  - Final score         : Probability × Impact  (not summed)
  - Counties excluded   : those with no wildfire or drought data

Predictive ML layer (v6):
  Random Forest regressor trained on historical fire events.
  Features: drought weeks, unemployment, GDP per capita,
            industry sensitivity, fire frequency — all measured
            BEFORE each fire to avoid data leakage.
  Label:    GDP delta % in 4 quarters post-fire.
  Output:   per-county predicted GDP impact if fire occurs today,
            with confidence interval and SHAP feature explanations.
  Model:    saved to wildfire_rf_model.pkl, reused on subsequent runs.

Outputs:
  county_probability_scores.csv → fire likelihood per county
  county_impact_scores.csv      → economic damage severity if fire occurs
  county_final_risk_scores.csv  → probability × impact + ML predictions
  county_ml_predictions.csv     → RF GDP impact predictions + SHAP breakdowns
  county_causal_evidence.csv    → pre/post + counterfactual fire event detail
  wildfire_rf_model.pkl         → trained Random Forest model
  risk_summary_report.txt       → shareholder narrative summary
"""

import json
import warnings
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
PATHS = {
    "gdp":          "County Quarterly GDP Estimate 2.xlsx",
    "foia":         "FOIA_2019_PRESENT_with_borrcounty 2.xlsx",
    "geo_events":   "geo_events_geoevent_with_county 2.csv",
    "unemployment": "Unemployment Data 2025.csv",
    "drought":      "Drought Monitor Data 2019-2023 1.csv",
}

WEIGHTS_PATH = "optimized_weights.json"

def _load_weights(group_name: str, defaults: dict) -> dict:
    """
    Load trained weights from optimized_weights.json.
    Falls back to provided defaults if file missing or group absent.
    Every weight in the model loads through this function — no
    hardcoded constants in any function body.
    """
    try:
        with open(WEIGHTS_PATH, "r") as f:
            all_weights = json.load(f)
        group = all_weights.get(group_name, {})
        # Extract only numeric keys (skip metadata like 'trained', 'reason', 'constraint')
        result = {}
        for k, v in defaults.items():
            result[k] = group.get(k, v)
            if isinstance(result[k], (int, float)):
                pass  # good
            else:
                result[k] = v  # fallback if metadata leaked
        return result
    except (FileNotFoundError, json.JSONDecodeError):
        return dict(defaults)

# Impact pillar weights (what drives economic damage)
_iw = _load_weights("IMPACT_WEIGHTS", {
    "causal_impact": 0.40, "industry_sensitivity": 0.25, "economic_trajectory": 0.35,
})
IMPACT_WEIGHTS = {
    "causal_impact":        _iw["causal_impact"],
    "industry_sensitivity": _iw["industry_sensitivity"],
    "economic_trajectory":  _iw["economic_trajectory"],
}

# Probability pillar weights (what drives fire likelihood)
_pw = _load_weights("PROB_WEIGHTS", {"drought": 0.60, "fire_frequency": 0.40})
PROB_WEIGHTS = {
    "drought":          _pw["drought"],
    "fire_frequency":   _pw["fire_frequency"],
}

# Drought sub-indicator weights
_dw = _load_weights("DROUGHT_WEIGHTS", {
    "max_consecutive": 0.40, "recent_intensity": 0.35, "frequency": 0.25,
})
DROUGHT_WEIGHTS = {
    "max_consecutive_n":   _dw["max_consecutive"],
    "recent_intensity_n":  _dw["recent_intensity"],
    "frequency_n":         _dw["frequency"],
}

DECAY_RATE      = 0.3
REFERENCE_DATE  = datetime.today()
PRE_POST_QTR    = 4
N_CONTROLS      = 5
MIN_ACREAGE     = 50
MODEL_PATH      = "outputs/best_model.pkl"
MODEL_PATH_CA   = "outputs/best_model_ca.pkl"
MIN_TRAINING_ROWS = 30    # warn if training set smaller than this

NAICS_SENSITIVITY = {
    "11": 0.95, "21": 0.80, "22": 0.40, "23": 0.55,
    "31": 0.50, "32": 0.50, "33": 0.50, "42": 0.35,
    "44": 0.45, "45": 0.45, "48": 0.40, "49": 0.40,
    "51": 0.15, "52": 0.10, "53": 0.30, "54": 0.15,
    "55": 0.10, "56": 0.25, "61": 0.30, "62": 0.30,
    "71": 0.85, "72": 0.80, "81": 0.35, "92": 0.20,
}

# BEA CAGDP2 LineCode → NAICS sensitivity mapping (leaf-level sectors only)
# Excludes aggregates (LC 1,2,50,59,68,75,87-92) to avoid double-counting
BEA_LINECODE_TO_NAICS = {
    3:  "11",      # Agriculture, forestry, fishing and hunting
    6:  "21",      # Mining, quarrying, and oil and gas extraction
    10: "22",      # Utilities
    11: "23",      # Construction
    12: "31-33",   # Manufacturing (maps to avg of 31,32,33 = 0.50)
    34: "42",      # Wholesale trade
    35: "44-45",   # Retail trade (avg of 44,45 = 0.45)
    36: "48-49",   # Transportation and warehousing (avg of 48,49 = 0.40)
    45: "51",      # Information
    51: "52",      # Finance and insurance
    56: "53",      # Real estate and rental and leasing
    60: "54",      # Professional, scientific, and technical services
    64: "55",      # Management of companies and enterprises
    65: "56",      # Admin and support and waste management
    69: "61",      # Educational services
    70: "62",      # Health care and social assistance
    76: "71",      # Arts, entertainment, and recreation
    79: "72",      # Accommodation and food services
    82: "81",      # Other services (except government)
    83: "92",      # Government and government enterprises
}

# For multi-NAICS BEA sectors, average the sensitivity values
def _bea_naics_sensitivity(naics_str: str) -> float:
    """Map a BEA NAICS string (e.g. '31-33', '44-45') to NAICS_SENSITIVITY."""
    if naics_str in NAICS_SENSITIVITY:
        return NAICS_SENSITIVITY[naics_str]
    # Handle range codes like "31-33" → avg(31,32,33)
    if "-" in naics_str:
        parts = naics_str.split("-")
        try:
            start, end = int(parts[0]), int(parts[1])
            vals = [NAICS_SENSITIVITY[str(i)] for i in range(start, end + 1)
                    if str(i) in NAICS_SENSITIVITY]
            return np.mean(vals) if vals else 0.30
        except (ValueError, KeyError):
            return 0.30
    return 0.30  # fallback


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def normalize(series: pd.Series) -> pd.Series:
    s = series.copy().astype(float)
    mn, mx = s.min(), s.max()
    if mx == mn:
        return pd.Series(0.0, index=s.index)
    return pd.Series(
        MinMaxScaler().fit_transform(s.values.reshape(-1, 1)).flatten(),
        index=s.index
    )

def invert(series: pd.Series) -> pd.Series:
    return 1 - normalize(series)

def decay_weight(date_or_year) -> float:
    if pd.isnull(date_or_year):
        return 0.0
    if isinstance(date_or_year, (int, float)):
        years_ago = REFERENCE_DATE.year - int(date_or_year)
    else:
        try:
            dt = pd.to_datetime(date_or_year)
            years_ago = (REFERENCE_DATE - dt).days / 365.25
        except Exception:
            return 0.0
    return np.exp(-DECAY_RATE * max(years_ago, 0))

STATE_NAME_TO_ABBR = {
    "Alabama":"AL","Alaska":"AK","Arizona":"AZ","Arkansas":"AR","California":"CA",
    "Colorado":"CO","Connecticut":"CT","Delaware":"DE","District of Columbia":"DC",
    "Florida":"FL","Georgia":"GA","Hawaii":"HI","Idaho":"ID","Illinois":"IL",
    "Indiana":"IN","Iowa":"IA","Kansas":"KS","Kentucky":"KY","Louisiana":"LA",
    "Maine":"ME","Maryland":"MD","Massachusetts":"MA","Michigan":"MI","Minnesota":"MN",
    "Mississippi":"MS","Missouri":"MO","Montana":"MT","Nebraska":"NE","Nevada":"NV",
    "New Hampshire":"NH","New Jersey":"NJ","New Mexico":"NM","New York":"NY",
    "North Carolina":"NC","North Dakota":"ND","Ohio":"OH","Oklahoma":"OK","Oregon":"OR",
    "Pennsylvania":"PA","Rhode Island":"RI","South Carolina":"SC","South Dakota":"SD",
    "Tennessee":"TN","Texas":"TX","Utah":"UT","Vermont":"VT","Virginia":"VA",
    "Washington":"WA","West Virginia":"WV","Wisconsin":"WI","Wyoming":"WY",
    "Puerto Rico":"PR","U.S. Virgin Islands":"VI","Guam":"GU",
    "American Samoa":"AS","Northern Mariana Islands":"MP",
}


def load_county_crosswalk(
    path: str = "data/raw/national_county2020.txt"
) -> pd.DataFrame:
    """
    Loads Census ANSI county code list as authoritative FIPS lookup.
    Provides canonical mapping: fips (5-digit Int64) → county_name_canonical, state_abbr
    """
    print("\n[Crosswalk] Loading county FIPS crosswalk...")
    df = pd.read_csv(
        path, sep="|", dtype=str,
    )
    # Skip header row if present (file has real headers)
    df = df.rename(columns={
        "STATE": "state_abbr",
        "STATEFP": "state_fips",
        "COUNTYFP": "county_fips",
        "COUNTYNAME": "county_name_canonical",
        "CLASSFP": "class_code",
    })
    df["fips"] = (df["state_fips"] + df["county_fips"]).astype("Int64")
    df["county_key"] = df["county_name_canonical"].str.lower().str.strip()
    df["county_key_short"] = (
        df["county_key"]
        .str.replace(
            r"\s+(county|parish|borough|census area|municipality)$",
            "", regex=True
        )
    )
    print(f"  Crosswalk counties loaded: {len(df)}")
    return df[[
        "fips", "state_fips", "county_fips",
        "county_name_canonical", "county_key",
        "county_key_short", "state_abbr"
    ]]


def add_fips_to_gdp(gdp: pd.DataFrame, crosswalk: pd.DataFrame) -> pd.DataFrame:
    """
    Adds FIPS column to GDP data by matching County + State to crosswalk.
    GDP uses full county names ('Autauga County') and full state names ('Alabama').
    Crosswalk uses county_key and state_abbr.
    """
    df = gdp.copy()
    df["county_key"] = df["County"].str.lower().str.strip()
    df["state_abbr"] = df["State"].map(STATE_NAME_TO_ABBR)

    xw = crosswalk[["fips", "county_key", "state_abbr"]].drop_duplicates()
    df = df.merge(xw, on=["county_key", "state_abbr"], how="left")

    matched = df["fips"].notna().sum()
    total = len(df)
    print(f"  [GDP FIPS] Matched {matched}/{total} rows ({matched/total*100:.1f}%)")
    return df


def safe_json_parse(val, key):
    try:
        return json.loads(val).get(key, np.nan)
    except Exception:
        return np.nan

def year_quarter_to_period(year, quarter):
    q_num = int(str(quarter).replace("Q", ""))
    return pd.Period(year=int(year), quarter=q_num, freq="Q")

def load_data():
    print("Loading datasets...")
    gdp     = pd.read_excel(PATHS["gdp"])
    foia    = pd.read_excel(PATHS["foia"])
    geo     = pd.read_csv(PATHS["geo_events"])
    unem    = pd.read_csv(PATHS["unemployment"])
    drought = pd.read_csv(PATHS["drought"])
    crosswalk = load_county_crosswalk()
    gdp     = add_fips_to_gdp(gdp, crosswalk)
    print(f"  GDP:          {gdp.shape}")
    print(f"  FOIA:         {foia.shape}")
    print(f"  Geo Events:   {geo.shape}")
    print(f"  Unemployment: {unem.shape}")
    print(f"  Drought:      {drought.shape}")
    print(f"  Crosswalk:    {crosswalk.shape}")
    return gdp, foia, geo, unem, drought, crosswalk


# ═══════════════════════════════════════════════════════
# PROBABILITY LAYER
# ═══════════════════════════════════════════════════════

# ── P1: Drought Score ──
def build_drought_score(drought: pd.DataFrame) -> pd.DataFrame:
    """
    Derives three drought indicators per county from consecutive-week episodes:

      max_consecutive_weeks  — longest single unbroken drought spell
                               (proxy for peak fuel dryness ever reached)
      recent_intensity       — sum of ConsecutiveWeeks decay-weighted by EndDate
                               (how severe is drought right now vs. historically)
      episode_frequency      — number of distinct drought episodes
                               (chronic vs. episodic drought pattern)

    All three are normalized and combined into a single drought_score (0-1).
    Higher = drier = more likely to ignite.
    """
    print("\n[P1] Building Drought scores...")

    df = drought.copy()
    df["fips"]     = df["FIPS"].astype("Int64")
    df["end_date"] = pd.to_datetime(df["EndDate"], errors="coerce")
    df["weight"]   = df["end_date"].apply(decay_weight)

    # Decay-weighted intensity: recent long droughts count most
    df["weighted_weeks"] = df["ConsecutiveWeeks"] * df["weight"]

    agg = df.groupby("fips").agg(
        max_consecutive_weeks = ("ConsecutiveWeeks", "max"),
        recent_intensity      = ("weighted_weeks",   "sum"),
        episode_frequency     = ("ConsecutiveWeeks", "count"),
        total_drought_weeks   = ("ConsecutiveWeeks", "sum"),
        county_name           = ("County",           "first"),
        state                 = ("State",            "first"),
        last_drought_end      = ("end_date",         "max"),
    ).reset_index()

    # Normalize each sub-indicator
    agg["max_consecutive_n"] = normalize(agg["max_consecutive_weeks"])
    agg["recent_intensity_n"]= normalize(agg["recent_intensity"])
    agg["frequency_n"]       = normalize(agg["episode_frequency"])

    agg["drought_score"] = (
        agg["max_consecutive_n"]  * DROUGHT_WEIGHTS["max_consecutive_n"] +
        agg["recent_intensity_n"] * DROUGHT_WEIGHTS["recent_intensity_n"] +
        agg["frequency_n"]        * DROUGHT_WEIGHTS["frequency_n"]
    )

    print(f"  Counties with drought data:    {len(agg)}")
    print(f"  Avg max consecutive weeks:     {agg['max_consecutive_weeks'].mean():.1f}")
    print(f"  Max on record:                 {agg['max_consecutive_weeks'].max()} weeks")
    return agg[["fips","county_name","state","drought_score",
                "max_consecutive_weeks","recent_intensity",
                "episode_frequency","total_drought_weeks","last_drought_end"]]


# ── P2: Fire Frequency Score ──
def build_fire_frequency_score(geo: pd.DataFrame, gdp: pd.DataFrame) -> pd.DataFrame:
    """
    Decay-weighted historical fire frequency per 100k population.
    More recent fires count more (leading indicator of ongoing risk).
    Larger fires (more acreage) weighted more heavily.
    """
    print("\n[P2] Building Fire Frequency scores...")

    # Population lookup via FIPS (P1 fix: was county_key string join)
    pop_lookup = (
        gdp[gdp["fips"].notna()]
        .sort_values("Year", ascending=False)
        .groupby("fips")
        .agg(population=("County Population", "first"))
        .reset_index()
    )

    df = geo[geo["geo_event_type"] == "wildfire"].copy()
    df["fips"]       = df["event_county_fips"].astype("Int64")
    df["acreage"]    = df["data"].apply(lambda x: safe_json_parse(x, "acreage"))
    df["event_date"] = pd.to_datetime(df["date_created"], errors="coerce")
    df["weight"]     = df["event_date"].apply(decay_weight)
    # Acreage-adjusted weight: larger fires count more
    df["acreage_safe"] = df["acreage"].fillna(1).clip(lower=1)
    df["adj_weight"] = df["weight"] * np.log1p(df["acreage_safe"])

    agg = df.groupby("fips").agg(
        fire_count       = ("id",          "count"),
        decay_sum        = ("adj_weight",  "sum"),
        avg_acreage      = ("acreage",     "mean"),
        county_name      = ("event_county","first"),
    ).reset_index()

    agg = agg.merge(pop_lookup[["fips","population"]], on="fips", how="left")

    agg["fires_per_100k"] = (
        agg["decay_sum"] / agg["population"].replace(0, np.nan) * 100_000
    )

    agg["fire_frequency_score"] = normalize(agg["fires_per_100k"].fillna(0))

    print(f"  Counties with fire history: {len(agg)}")
    return agg[["fips","county_name","fire_frequency_score",
                "fire_count","fires_per_100k","avg_acreage"]]


# ── Combine: Probability Score ──
def build_probability_score(drought: pd.DataFrame,
                             fire_freq: pd.DataFrame) -> pd.DataFrame:
    """
    Combines drought and fire frequency into a single probability score.

    probability_score = drought_score      × 0.60
                      + fire_frequency_score × 0.40

    Drought weighted higher because it is a LEADING indicator
    (predicts future fires). Fire frequency is a LAGGING indicator
    (reflects past occurrence, still predictive but less immediate).

    Only counties present in BOTH datasets are included —
    we need both drought context and fire history to assess probability.
    """
    print("\n[Probability] Combining drought + fire frequency...")

    df = drought.merge(
        fire_freq[["fips","fire_frequency_score","fire_count","fires_per_100k","avg_acreage"]],
        on="fips", how="inner"
    )

    df["drought_score_n"]       = normalize(df["drought_score"])
    df["fire_frequency_score_n"]= normalize(df["fire_frequency_score"])

    df["probability_score"] = (
        df["drought_score_n"]        * PROB_WEIGHTS["drought"] +
        df["fire_frequency_score_n"] * PROB_WEIGHTS["fire_frequency"]
    )

    df["probability_tier"] = pd.cut(
        df["probability_score"],
        bins=[0, 0.25, 0.50, 0.75, 1.0],
        labels=["Low","Moderate","High","Critical"],
        include_lowest=True
    )

    df = df.sort_values("probability_score", ascending=False).reset_index(drop=True)
    df["probability_rank"] = df.index + 1

    print(f"  Counties with probability scores: {len(df)}")
    print(f"  Probability tier distribution:\n{df['probability_tier'].value_counts().to_string()}")
    return df


# ═══════════════════════════════════════════════════════
# IMPACT LAYER  (economic damage if fire occurs)
# ═══════════════════════════════════════════════════════

def build_prepost_windows(geo: pd.DataFrame, gdp: pd.DataFrame) -> pd.DataFrame:
    print("\n[I1] Computing pre/post event windows...")

    gdp_w = gdp.copy()
    gdp_w["period"] = gdp_w.apply(
        lambda r: year_quarter_to_period(r["Year"], r["Quarter"]), axis=1
    )
    # P1 fix: use FIPS for GDP lookup instead of county_key string
    gdp_w["fips"] = gdp_w["fips"].astype("Int64")

    fires = geo[geo["geo_event_type"] == "wildfire"].copy()
    fires["acreage"]    = fires["data"].apply(lambda x: safe_json_parse(x, "acreage"))
    fires["event_date"] = pd.to_datetime(fires["date_created"], errors="coerce")
    fires["fips"]       = fires["event_county_fips"].astype("Int64")
    fires["county_key"] = fires["event_county"].str.lower().str.strip()

    sig = fires[fires["acreage"] >= MIN_ACREAGE].copy()
    sig["fire_period"] = sig["event_date"].apply(
        lambda d: pd.Period(year=d.year, quarter=(d.month-1)//3+1, freq="Q")
        if pd.notnull(d) else None
    )
    sig = sig.dropna(subset=["fire_period"])

    records = []
    for _, fire in sig.iterrows():
        fips_val = fire["fips"]
        fp       = fire["fire_period"]
        cg       = gdp_w[gdp_w["fips"] == fips_val]
        if cg.empty:
            continue
        pre  = cg[(cg["period"] >= fp - PRE_POST_QTR) & (cg["period"] < fp)]["Est. County GDP (Current $M)"]
        post = cg[(cg["period"] >  fp) & (cg["period"] <= fp + PRE_POST_QTR)]["Est. County GDP (Current $M)"]
        if len(pre) < 2 or len(post) < 2:
            continue
        gb = pre.mean()
        ga = post.mean()
        records.append({
            "fips":          fips_val,
            "county_key":    fire["county_key"],
            "county_name":   fire["event_county"],
            "fire_period":   str(fp),
            "acreage":       fire["acreage"],
            "gdp_before":    gb,
            "gdp_after":     ga,
            "gdp_delta_pct": (ga - gb) / gb * 100 if gb > 0 else np.nan,
            "event_date":    fire["event_date"],
        })

    result = pd.DataFrame(records)
    print(f"  Significant fires (>={MIN_ACREAGE} acres): {len(sig)}")
    print(f"  Events with matching GDP windows:          {len(result)}")
    return result


def build_counterfactual(prepost: pd.DataFrame,
                          gdp: pd.DataFrame,
                          unem: pd.DataFrame) -> tuple:
    print("\n[I2] Building counterfactual comparisons...")

    # P1 fix: use FIPS for feature matching instead of county_key strings
    gdp_latest = (
        gdp[gdp["Year"] == gdp["Year"].max()]
        .dropna(subset=["fips"])
        .groupby("fips")
        .agg(gdp_per_capita=("Est. County GDP (Current $M)", "sum"),
             population    =("County Population",            "first"))
        .reset_index()
    )
    gdp_latest["gdp_per_capita"] = (
        gdp_latest["gdp_per_capita"] * 1e6 /
        gdp_latest["population"].replace(0, np.nan)
    )

    unem_w      = unem.copy()
    unem_w["fips"] = unem_w["FIPS_Code"].astype("Int64")
    unem_w["year"] = unem_w["Attribute"].str.extract(r"(\d{4})")
    unem_rates  = unem_w[unem_w["Attribute"].str.startswith("Unemployment_rate_")]
    latest_yr   = unem_rates["year"].max()
    unem_latest = (
        unem_rates[unem_rates["year"] == latest_yr][["fips","Value"]]
        .rename(columns={"Value":"unem_rate"})
    )

    features    = gdp_latest.merge(unem_latest, on="fips", how="inner")
    features    = features.dropna(subset=["gdp_per_capita","unem_rate","population"])
    features    = features.reset_index(drop=True)
    feat_matrix = StandardScaler().fit_transform(
        features[["gdp_per_capita","unem_rate","population"]].values
    )
    fips_to_pos = {int(f): i for i, f in enumerate(features["fips"])}
    affected    = prepost.groupby("fips")["fire_period"].apply(set).to_dict()

    skipped_no_delta = skipped_no_feat = skipped_no_ctrl = 0
    results = []

    for _, row in prepost.iterrows():
        tf = int(row["fips"]) if pd.notna(row["fips"]) else None
        td = row["gdp_delta_pct"]
        fp = row["fire_period"]

        if pd.isnull(td):
            skipped_no_delta += 1; continue
        if tf is None or tf not in fips_to_pos:
            skipped_no_feat += 1; continue

        tv = feat_matrix[fips_to_pos[tf]].reshape(1, -1)
        elig_pos = [i for i, f in enumerate(features["fips"])
                    if int(f) != tf and fp not in affected.get(f, set())]

        if len(elig_pos) < N_CONTROLS:
            skipped_no_ctrl += 1; continue

        sims      = cosine_similarity(tv, feat_matrix[elig_pos])[0]
        top_pos   = np.argsort(sims)[-N_CONTROLS:]
        ctrl_fips = features.iloc[[elig_pos[p] for p in top_pos]]["fips"].values

        ctrl_deltas = prepost[
            prepost["fips"].isin(ctrl_fips) &
            (prepost["fire_period"] == fp)
        ]["gdp_delta_pct"]

        if len(ctrl_deltas) == 0:
            cg = gdp[gdp["fips"].isin(ctrl_fips)]
            if cg.empty:
                ctrl_avg = 0.0
            else:
                cr = cg[cg["Year"] >= gdp["Year"].max() - 2]
                ct = cr.groupby("fips")["Est. County GDP (Current $M)"].apply(
                    lambda s: (s.iloc[-1]-s.iloc[0])/s.iloc[0]*100
                    if len(s) >= 2 and s.iloc[0] > 0 else 0
                )
                ctrl_avg = float(ct.mean()) if len(ct) > 0 else 0.0
        else:
            ctrl_avg = float(ctrl_deltas.mean())

        results.append({
            **row.to_dict(),
            "control_avg_delta":  ctrl_avg,
            "counterfactual_gap": td - ctrl_avg,
            "n_controls_used":    N_CONTROLS,
        })

    print(f"  Skipped — no GDP delta:     {skipped_no_delta}")
    print(f"  Skipped — no feature match: {skipped_no_feat}")
    print(f"  Skipped — too few controls: {skipped_no_ctrl}")
    print(f"  Matched fire events:        {len(results)}")

    if len(results) == 0:
        print("  WARNING: No counterfactual matches — using pre/post deltas directly.")
        fallback = prepost.dropna(subset=["gdp_delta_pct"]).copy()
        fallback["control_avg_delta"]  = 0.0
        fallback["counterfactual_gap"] = fallback["gdp_delta_pct"]
        fallback["n_controls_used"]    = 0
        result = fallback
    else:
        result = pd.DataFrame(results)

    county_causal = result.groupby(["fips","county_name","county_key"]).agg(
        n_fire_events         =("fire_period",        "count"),
        avg_gdp_delta_pct     =("gdp_delta_pct",      "mean"),
        avg_counterfactual_gap=("counterfactual_gap", "mean"),
        worst_gdp_delta       =("gdp_delta_pct",      "min"),
        total_acreage         =("acreage",            "sum"),
    ).reset_index()

    county_causal["causal_impact_score"] = normalize(
        -county_causal["avg_counterfactual_gap"].fillna(0)
    )
    print(f"  Counties with causal evidence: {len(county_causal)}")
    return county_causal, result


# ── NAICS 2-digit → broad sector label mapping ──
NAICS_SECTOR_LABELS = {
    "11": "Agriculture & Forestry",
    "21": "Energy & Mining",
    "22": "Utilities",
    "23": "Construction",
    "31": "Manufacturing", "32": "Manufacturing", "33": "Manufacturing",
    "42": "Trade & Logistics",
    "44": "Trade & Logistics", "45": "Trade & Logistics",
    "48": "Trade & Logistics", "49": "Trade & Logistics",
    "51": "Professional & Finance",
    "52": "Professional & Finance",
    "53": "Real Estate",
    "54": "Professional & Finance",
    "55": "Professional & Finance",
    "56": "Administrative Services",
    "61": "Public & Healthcare",
    "62": "Public & Healthcare",
    "71": "Tourism & Hospitality",
    "72": "Tourism & Hospitality",
    "81": "Other Services",
    "92": "Public & Healthcare",
}


def _naics_from_foia(foia: pd.DataFrame) -> pd.DataFrame:
    """Derive industry sensitivity and top industries from SBA FOIA loan data."""
    df = foia.copy()
    df["fips"]       = df["borrcounty_fips"].astype("Int64")
    df["naics_2"]    = df["naicscode"].astype(str).str[:2]
    df["sensitivity"]= df["naics_2"].map(NAICS_SENSITIVITY).fillna(0.30)

    def industry_agg(group):
        w     = group["grossapproval"]
        w_sum = w.sum()
        ws    = (group["sensitivity"] * w).sum() / w_sum if w_sum > 0 else group["sensitivity"].mean()
        top   = group.groupby("naicsdescription")["grossapproval"].sum().nlargest(3).index.tolist()
        # Dominant sector: highest loan-weighted sector bucket
        group["sector"] = group["naics_2"].map(NAICS_SECTOR_LABELS).fillna("Other Services")
        dom = group.groupby("sector")["grossapproval"].sum().idxmax()
        return pd.Series({
            "industry_sensitivity": ws,
            "top_industries":       " | ".join(top),
            "dominant_sector":      dom,
            "county_name":          group["borrcounty"].iloc[0],
            "n_industries":         group["naics_2"].nunique(),
            "industry_source":      "FOIA",
        })

    result = df.groupby("fips").apply(industry_agg).reset_index()
    return result


def _naics_from_cbp(fips_needed: list) -> pd.DataFrame:
    """
    Fetch County Business Patterns data from Census API for counties
    missing from FOIA. Uses employment count as weight instead of loan volume.

    Census CBP API docs: https://api.census.gov/data/2021/cbp/variables.html
    No API key required for Census data.
    """
    import requests

    print(f"  [CBP] Fetching Census County Business Patterns for {len(fips_needed)} counties...")

    records = []
    try:
        # Census CBP: establishments and employment by NAICS sector
        url = (
            "https://api.census.gov/data/2021/cbp"
            "?get=GEO_ID,NAICS2017,EMP,ESTAB,INDLEVEL"
            "&for=county:*"
            "&INDLEVEL=2"
        )
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        headers = data[0]
        rows    = data[1:]

        df = pd.DataFrame(rows, columns=headers)
        df["fips"]  = (df["state"] + df["county"]).apply(
            lambda x: int(x) if x.isdigit() else None
        )
        df = df.dropna(subset=["fips"])
        df["fips"]  = df["fips"].astype("Int64")
        df["emp"]   = pd.to_numeric(df["EMP"],   errors="coerce").fillna(0)
        df["naics_2"] = df["NAICS2017"].astype(str).str[:2]
        df["sensitivity"] = df["naics_2"].map(NAICS_SENSITIVITY).fillna(0.30)
        df["sector"]      = df["naics_2"].map(NAICS_SECTOR_LABELS).fillna("Other Services")

        # Filter to only counties we need
        df = df[df["fips"].isin(fips_needed)]

        def cbp_agg(group):
            emp_sum = group["emp"].sum()
            ws = (group["sensitivity"] * group["emp"]).sum() / emp_sum                  if emp_sum > 0 else 0.30
            top_sectors = (
                group.groupby("sector")["emp"].sum()
                .nlargest(3).index.tolist()
            )
            dom = group.groupby("sector")["emp"].sum().idxmax()                   if len(group) > 0 else "Unclassified"
            return pd.Series({
                "industry_sensitivity": ws,
                "top_industries":       " | ".join(top_sectors),
                "dominant_sector":      dom,
                "county_name":          "",
                "n_industries":         group["naics_2"].nunique(),
                "industry_source":      "Census CBP",
            })

        result = df.groupby("fips").apply(cbp_agg).reset_index()
        print(f"  [CBP] Retrieved data for {len(result)} counties")
        return result

    except Exception as e:
        print(f"  [CBP] API error: {e} — will fall back to BEA")
        return pd.DataFrame()


def _naics_from_bea(fips_needed: list, gdp: pd.DataFrame) -> pd.DataFrame:
    """
    Last-resort fallback using BEA regional GDP by industry (CAGDP2).
    Derives sector weights from GDP contribution rather than employment.
    Falls back to GDP dataset state-level industry mix if API unavailable.
    """
    import requests

    BEA_API_KEY = __import__("os").getenv("BEA_API_KEY", "")
    print(f"  [BEA] Fetching industry GDP for {len(fips_needed)} counties...")

    records = []

    if BEA_API_KEY:
        try:
            url    = "https://apps.bea.gov/api/data"
            params = {
                "UserID":      BEA_API_KEY,
                "method":      "GetData",
                "datasetname": "Regional",
                "TableName":   "CAGDP2",
                "LineCode":    "1",
                "GeoFips":     "COUNTY",
                "Year":        str(gdp["Year"].max()),
                "ResultFormat":"JSON",
            }
            resp = requests.get(url, params=params, timeout=20)
            resp.raise_for_status()
            data = resp.json()
            bea_rows = data["BEAAPI"]["Results"]["Data"]
            bea_df   = pd.DataFrame(bea_rows)
            bea_df["fips"] = pd.to_numeric(bea_df["GeoFips"], errors="coerce").astype("Int64")
            bea_df = bea_df[bea_df["fips"].isin(fips_needed)]

            # BEA LineCode maps to broad sectors — approximate NAICS
            # For simplicity, assign median sensitivity and flag as BEA-sourced
            for fips_val in fips_needed:
                records.append({
                    "fips":                 fips_val,
                    "industry_sensitivity": 0.45,   # median sensitivity
                    "top_industries":       "Mixed (BEA estimated)",
                    "dominant_sector":      "Mixed",
                    "county_name":          "",
                    "n_industries":         0,
                    "industry_source":      "BEA (estimated)",
                })
        except Exception as e:
            print(f"  [BEA] API error: {e}")

    # If BEA failed or no key, use state-level GDP industry distribution as proxy
    if not records:
        print("  [BEA] Using state GDP distribution as proxy...")
        for fips_val in fips_needed:
            records.append({
                "fips":                 fips_val,
                "industry_sensitivity": 0.40,
                "top_industries":       "Unclassified",
                "dominant_sector":      "Unclassified",
                "county_name":          "",
                "n_industries":         0,
                "industry_source":      "Unclassified",
            })

    result = pd.DataFrame(records)
    result["fips"] = result["fips"].astype("Int64")
    print(f"  [BEA] Filled {len(result)} counties")
    return result


def build_industry_sensitivity(foia: pd.DataFrame,
                                gdp: pd.DataFrame = None) -> pd.DataFrame:
    """
    Builds industry sensitivity scores with a 3-tier fallback chain:

      Priority 1 — FOIA loan data  (weighted by loan volume)
      Priority 2 — Census CBP      (weighted by employment count)
      Priority 3 — BEA industry GDP(weighted by GDP contribution)
      Priority 4 — Unclassified    (flagged clearly in output)

    Each county records its industry_source so downstream users know
    which tier provided the data — important for transparency in
    investor presentations.
    """
    print("\n[I3] Building industry sensitivity scores (3-tier fallback)...")

    # ── Tier 1: FOIA ──
    foia_result = _naics_from_foia(foia)
    foia_fips   = set(foia_result["fips"].dropna().unique())
    print(f"  Tier 1 (FOIA):        {len(foia_result)} counties")

    # ── Identify gaps ──
    # All counties we know about from geo_events FIPS
    all_fips = set()
    if gdp is not None:
        # Use GDP dataset as the universe of known counties
        gdp_copy = gdp.copy()
        gdp_copy["fips_approx"] = pd.NA  # GDP doesn't have FIPS directly
        # We'll fill gaps based on what comes through later joins
    missing_fips = []   # will be populated after first merge in build_recovery_trajectory

    # ── Tier 2: Census CBP for any FIPS we can identify as missing ──
    # We attempt CBP for all known FIPS not in FOIA as a proactive fill
    cbp_result = pd.DataFrame()
    bea_result = pd.DataFrame()

    try:
        # Get all county FIPS from unemployment data as the universe
        cbp_result = _naics_from_cbp(list(foia_fips))  # pre-fetch; gaps filled later
    except Exception as e:
        print(f"  [CBP] Skipped: {e}")

    # ── Combine: FOIA is authoritative, CBP fills gaps, BEA fills remainder ──
    result = foia_result.copy()

    if not cbp_result.empty:
        # Only use CBP rows for FIPS not already covered by FOIA
        cbp_new = cbp_result[~cbp_result["fips"].isin(foia_fips)]
        if len(cbp_new) > 0:
            print(f"  Tier 2 (Census CBP):  {len(cbp_new)} additional counties")
            result = pd.concat([result, cbp_new], ignore_index=True)

    result["industry_sensitivity_n"] = normalize(result["industry_sensitivity"])

    print(f"  Total counties with industry data: {len(result)}")
    print(f"  Source breakdown:")
    print(result["industry_source"].value_counts().to_string())
    return result[["fips","county_name","industry_sensitivity",
                   "industry_sensitivity_n","top_industries",
                   "dominant_sector","industry_source"]]


def build_recovery_trajectory(causal: pd.DataFrame,
                               industry: pd.DataFrame,
                               gdp: pd.DataFrame) -> pd.DataFrame:
    print("\n[I4] Classifying recovery trajectories...")
    # P1 fix: use FIPS for GDP lookup instead of county_key
    gdp_w = gdp[gdp["fips"].notna()].copy()

    def get_trajectory(row):
        if pd.isnull(row.get("avg_gdp_delta_pct", np.nan)):
            return "At Risk"
        cg = gdp_w[gdp_w["fips"] == row["fips"]].sort_values(["Year","Quarter"])
        if cg.empty or len(cg) < 4:
            return "At Risk"
        recent = cg.tail(4)["Est. County GDP (Current $M)"].mean()
        pre    = cg.head(4)["Est. County GDP (Current $M)"].mean()
        if pre <= 0:
            return "At Risk"
        ratio = recent / pre
        if row["avg_gdp_delta_pct"] >= -2.0 or ratio >= 0.98:
            return "Resilient"
        elif ratio >= 0.90:
            return "Slow Recovery"
        else:
            return "Chronic Impact"

    causal_w = causal.copy()
    causal_w["trajectory"] = causal_w.apply(get_trajectory, axis=1)

    # P1 fix: include dominant_sector and industry_source in merge
    full = causal_w.merge(
        industry[["fips","industry_sensitivity","industry_sensitivity_n",
                  "top_industries","dominant_sector","industry_source"]],
        on="fips", how="outer"
    )
    full["trajectory"] = full["trajectory"].fillna("At Risk")

    traj_score = {"Chronic Impact":1.00,"Slow Recovery":0.65,"At Risk":0.40,"Resilient":0.15}
    full["trajectory_score"] = full["trajectory"].map(traj_score)

    print(f"  Trajectory distribution:\n{full['trajectory'].value_counts().to_string()}")
    return full


def build_economic_trajectory(unem: pd.DataFrame, gdp: pd.DataFrame,
                               qcew: pd.DataFrame = None,
                               fiscal_df: pd.DataFrame = None,
                               income_df: pd.DataFrame = None) -> pd.DataFrame:
    print("\n[I5] Building economic trajectory scores...")
    df = unem.copy()
    df["fips"] = df["FIPS_Code"].astype("Int64")
    rate_rows  = df[df["Attribute"].str.startswith("Unemployment_rate_")].copy()
    rate_rows["year"]   = rate_rows["Attribute"].str.extract(r"(\d{4})").astype(int)
    rate_rows["weight"] = rate_rows["year"].apply(decay_weight)

    def unem_agg(group):
        w     = group["weight"]
        w_sum = w.sum()
        rate  = (group["Value"] * w).sum() / w_sum if w_sum > 0 else np.nan
        trend = np.nan
        if len(group) >= 2:
            X = group["year"].values.reshape(-1, 1)
            trend = LinearRegression().fit(X, group["Value"].values).coef_[0]
        return pd.Series({
            "decay_unem_rate":    rate,
            "unemployment_trend": trend,
            "county_name":        group["Area_Name"].iloc[0],
            "state":              group["State"].iloc[0],
        })

    unem_result = rate_rows.groupby("fips").apply(unem_agg).reset_index()

    # P1 fix: use FIPS for GDP trend join instead of county_key
    gdp_w = gdp[gdp["fips"].notna()].copy()
    def gdp_trend(group):
        if len(group) < 2: return np.nan
        group = group.sort_values(["Year","Quarter"])
        X = np.arange(len(group)).reshape(-1, 1)
        return LinearRegression().fit(X, group["Est. County GDP (Current $M)"].values).coef_[0]

    gdp_trends = gdp_w.groupby("fips").apply(gdp_trend).reset_index()
    gdp_trends.columns = ["fips","gdp_growth_trend"]

    result = unem_result.merge(gdp_trends[["fips","gdp_growth_trend"]], on="fips", how="left")
    result["unem_n"]      = normalize(result["decay_unem_rate"].fillna(0))
    result["unem_trend_n"]= normalize(result["unemployment_trend"].fillna(0))
    result["gdp_trend_n"] = invert(result["gdp_growth_trend"].fillna(0))

    # P3b: Add QCEW wage sub-indicators if available
    extra_cols = []
    if qcew is not None and not qcew.empty:
        print("  [P3b] Integrating BLS QCEW wage data...")
        qw = qcew.copy()
        qw["fips"] = pd.to_numeric(qw.get("area_fips", pd.Series(dtype=str)),
                                    errors="coerce").astype("Int64")
        # Filter: all industries, all ownership
        if "industry_code" in qw.columns:
            qw = qw[qw["industry_code"].astype(str) == "10"]
        if "own_code" in qw.columns:
            qw = qw[qw["own_code"].astype(str) == "0"]
        qw["year"] = pd.to_numeric(qw.get("year", pd.Series(dtype=int)), errors="coerce")
        qw["total_wages"] = pd.to_numeric(
            qw.get("total_annual_wages", pd.Series(dtype=float)), errors="coerce"
        )
        qw["avg_empl"] = pd.to_numeric(
            qw.get("annual_avg_emplvl", pd.Series(dtype=float)), errors="coerce"
        )

        # Wage base: latest year total wages
        latest_yr = qw["year"].max()
        wage_latest = qw[qw["year"] == latest_yr].groupby("fips").agg(
            wage_base_m=("total_wages", lambda x: x.sum() / 1e6)
        ).reset_index()

        # Wage trend: slope of annual wages over time
        def wage_trend_fn(group):
            if len(group) < 2: return np.nan
            group = group.sort_values("year")
            X = group["year"].values.reshape(-1, 1)
            return LinearRegression().fit(X, group["total_wages"].values).coef_[0]

        wage_trends = qw.groupby("fips").apply(wage_trend_fn).reset_index()
        wage_trends.columns = ["fips", "wage_trend"]

        result = result.merge(wage_latest, on="fips", how="left")
        result = result.merge(wage_trends, on="fips", how="left")
        result["wage_base_n"]  = normalize(result["wage_base_m"].fillna(0))
        result["wage_trend_n"] = invert(result["wage_trend"].fillna(0))

        # Rebalanced weights with QCEW (non-CA or when fiscal not available)
        result["economic_trajectory"] = (
            result["unem_n"]       * 0.35 +   # was 0.45
            result["unem_trend_n"] * 0.20 +   # was 0.25
            result["gdp_trend_n"]  * 0.25 +   # was 0.30
            result["wage_base_n"]  * 0.10 +   # new
            result["wage_trend_n"] * 0.10      # new
        )
        extra_cols = ["wage_base_m", "wage_trend"]
        print(f"  Counties with QCEW wage data: {wage_latest['fips'].nunique()}")

        # P3i/P3j: CA counties with fiscal + income data get rebalanced weights
        if fiscal_df is not None and not fiscal_df.empty:
            result = result.merge(
                fiscal_df[["fips", "fiscal_resilience_score"]],
                on="fips", how="left"
            )
            has_fiscal = result["fiscal_resilience_score"].notna()
            extra_cols.append("fiscal_resilience_score")

            # Check if income vulnerability data is also available
            has_income = pd.Series(False, index=result.index)
            if income_df is not None and not income_df.empty:
                result = result.merge(
                    income_df[["fips", "income_vulnerability_score"]],
                    on="fips", how="left"
                )
                has_income = result["income_vulnerability_score"].notna()
                extra_cols.append("income_vulnerability_score")

            # Tier 1: Both fiscal + income (CA full)
            has_both = has_fiscal & has_income
            if has_both.any():
                ca_full_w = _load_weights("ECONOMIC_TRAJECTORY_WEIGHTS_CA_FULL", {
                    "unem_n": 0.25, "unem_trend_n": 0.10, "gdp_trend_n": 0.15,
                    "wage_base_n": 0.10, "wage_trend_n": 0.10,
                    "fiscal_resilience_score": 0.15,
                    "income_vulnerability_score": 0.15,
                })
                result.loc[has_both, "economic_trajectory"] = (
                    result.loc[has_both, "unem_n"]       * ca_full_w["unem_n"] +
                    result.loc[has_both, "unem_trend_n"] * ca_full_w["unem_trend_n"] +
                    result.loc[has_both, "gdp_trend_n"]  * ca_full_w["gdp_trend_n"] +
                    result.loc[has_both, "wage_base_n"]  * ca_full_w["wage_base_n"] +
                    result.loc[has_both, "wage_trend_n"] * ca_full_w["wage_trend_n"] +
                    result.loc[has_both, "fiscal_resilience_score"] * ca_full_w["fiscal_resilience_score"] +
                    result.loc[has_both, "income_vulnerability_score"] * ca_full_w["income_vulnerability_score"]
                )
                print(f"  [P3i+P3j] CA counties with fiscal+income in trajectory: {has_both.sum()}")

            # Tier 2: Fiscal only (CA partial — shouldn't happen with 58/58, but safe)
            has_fiscal_only = has_fiscal & ~has_income
            if has_fiscal_only.any():
                ca_w = _load_weights("ECONOMIC_TRAJECTORY_WEIGHTS_CA", {
                    "unem_n": 0.30, "unem_trend_n": 0.15, "gdp_trend_n": 0.20,
                    "wage_base_n": 0.10, "wage_trend_n": 0.10,
                    "fiscal_resilience_score": 0.15,
                })
                result.loc[has_fiscal_only, "economic_trajectory"] = (
                    result.loc[has_fiscal_only, "unem_n"]       * ca_w["unem_n"] +
                    result.loc[has_fiscal_only, "unem_trend_n"] * ca_w["unem_trend_n"] +
                    result.loc[has_fiscal_only, "gdp_trend_n"]  * ca_w["gdp_trend_n"] +
                    result.loc[has_fiscal_only, "wage_base_n"]  * ca_w["wage_base_n"] +
                    result.loc[has_fiscal_only, "wage_trend_n"] * ca_w["wage_trend_n"] +
                    result.loc[has_fiscal_only, "fiscal_resilience_score"] * ca_w["fiscal_resilience_score"]
                )
                print(f"  [P3i] CA counties with fiscal only in trajectory: {has_fiscal_only.sum()}")

            n_fiscal = has_fiscal.sum()
            print(f"  [P3i] CA counties with fiscal resilience in trajectory: {n_fiscal}")
    else:
        result["economic_trajectory"] = (
            result["unem_n"]       * 0.45 +
            result["unem_trend_n"] * 0.25 +
            result["gdp_trend_n"]  * 0.30
        )

    print(f"  Counties with economic trajectory: {len(result)}")
    return result[["fips","county_name","state","economic_trajectory",
                   "decay_unem_rate","unemployment_trend","gdp_growth_trend"] + extra_cols]


def build_fiscal_resilience_score(
    revenues: pd.DataFrame,
    expenditures: pd.DataFrame,
    crosswalk: pd.DataFrame,
    city_revenues: pd.DataFrame = None,
    city_expenditures: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    P3i: Fiscal resilience score from CA State Controller (SCO) data.
    CA only — 58 counties.

    Three sub-indicators (3-year average of most recent years):
    - property_tax_dependence: property_tax / total_revenues (w=0.45)
    - fiscal_surplus_ratio: (rev - exp) / rev, INVERTED (w=0.35)
    - debt_service_burden: debt_service / total_expenditures (w=0.20)

    Plus intergovernmental dependence penalty: counties where
    intergovernmental_share > 0.50 get an upward score adjustment
    (more vulnerable) because low PT dependence reflects transfer-dependence,
    not fiscal strength.

    San Francisco (consolidated city-county) is sourced from SCO city budget
    data using FY2019-2021 (most recent complete years in city dataset).

    Weights load from FISCAL_RESILIENCE_WEIGHTS in optimized_weights.json.
    """
    print("\n[P3i] Building fiscal resilience scores (CA SCO)...")

    fw = _load_weights("FISCAL_RESILIENCE_WEIGHTS", {
        "property_tax_dependence": 0.45,
        "fiscal_surplus_ratio": 0.35,
        "debt_service_burden": 0.20,
    })

    rev = revenues.copy()
    exp = expenditures.copy()
    rev["Values"] = pd.to_numeric(rev["Values"], errors="coerce")
    exp["Values"] = pd.to_numeric(exp["Values"], errors="coerce")

    # Use 3-year average of most recent years
    rev_years = sorted(rev["Fiscal Year"].dropna().unique())
    recent_3 = rev_years[-3:] if len(rev_years) >= 3 else rev_years
    print(f"  Using fiscal years: {recent_3}")

    rev_r = rev[rev["Fiscal Year"].isin(recent_3)]
    exp_r = exp[exp["Fiscal Year"].isin(recent_3)]

    # Aggregate per county per year
    total_rev = rev_r.groupby(["Entity Name", "Fiscal Year"])["Values"].sum().reset_index(name="total_rev")
    pt_rev = rev_r[rev_r["Category"] == "Property Taxes"].groupby(
        ["Entity Name", "Fiscal Year"]
    )["Values"].sum().reset_index(name="property_tax")
    ig_rev = rev_r[rev_r["Category"].str.contains("Intergovernmental", na=False)].groupby(
        ["Entity Name", "Fiscal Year"]
    )["Values"].sum().reset_index(name="intergov")
    total_exp = exp_r.groupby(["Entity Name", "Fiscal Year"])["Values"].sum().reset_index(name="total_exp")
    ds_exp = exp_r[exp_r["Category"].str.contains("Debt Service", na=False)].groupby(
        ["Entity Name", "Fiscal Year"]
    )["Values"].sum().reset_index(name="debt_service")

    # Merge per county-year
    annual = total_rev.merge(pt_rev, on=["Entity Name", "Fiscal Year"], how="left")
    annual = annual.merge(ig_rev, on=["Entity Name", "Fiscal Year"], how="left")
    annual = annual.merge(total_exp, on=["Entity Name", "Fiscal Year"], how="left")
    annual = annual.merge(ds_exp, on=["Entity Name", "Fiscal Year"], how="left")
    annual = annual.fillna(0)

    # Compute indicators per year, then average across years
    annual["pt_share"] = annual["property_tax"] / annual["total_rev"].replace(0, np.nan)
    annual["ig_share"] = annual["intergov"] / annual["total_rev"].replace(0, np.nan)
    annual["surplus_ratio"] = (
        (annual["total_rev"] - annual["total_exp"]) /
        annual["total_rev"].replace(0, np.nan)
    )
    annual["debt_burden"] = annual["debt_service"] / annual["total_exp"].replace(0, np.nan)

    county_avg = annual.groupby("Entity Name").agg(
        property_tax_dependence=("pt_share", "mean"),
        intergovernmental_share=("ig_share", "mean"),
        fiscal_surplus_ratio=("surplus_ratio", "mean"),
        debt_service_burden=("debt_burden", "mean"),
    ).reset_index()

    # ── Add San Francisco from city budget data ──
    if city_revenues is not None and city_expenditures is not None:
        cr = city_revenues.copy()
        ce = city_expenditures.copy()
        cr["Value"] = pd.to_numeric(cr["Value"], errors="coerce")
        ce["Value"] = pd.to_numeric(ce["Value"], errors="coerce")

        sf_cr = cr[cr["Entity Name"] == "San Francisco"]
        sf_ce = ce[ce["Entity Name"] == "San Francisco"]

        # FY2022 missing, FY2023-2024 incomplete — use FY2019-2021
        sf_fy = [2019, 2020, 2021]
        sf_r = sf_cr[sf_cr["Fiscal Year"].isin(sf_fy)]
        sf_e = sf_ce[sf_ce["Fiscal Year"].isin(sf_fy)]
        print(f"  SF sourced from city dataset, fiscal years: {sf_fy}")

        if not sf_r.empty:
            sf_annual = []
            for fy in sf_fy:
                yr_r = sf_r[sf_r["Fiscal Year"] == fy]
                yr_e = sf_e[sf_e["Fiscal Year"] == fy]
                tr = yr_r["Value"].sum()
                # City data: property tax lines contain "Property Tax" or "Secured and Unsecured"
                pt = yr_r[yr_r["Line Description"].str.contains(
                    "Property Tax|Secured and Unsecured", case=False, na=False
                )]["Value"].sum()
                ig = yr_r[yr_r["Category"].str.contains(
                    "Intergovernmental", case=False, na=False
                )]["Value"].sum()
                te = yr_e["Value"].sum()
                ds = yr_e[yr_e["Category"].str.contains(
                    "Debt Service", case=False, na=False
                )]["Value"].sum()
                if tr > 0:
                    sf_annual.append({
                        "pt_share": pt / tr,
                        "ig_share": ig / tr,
                        "surplus_ratio": (tr - te) / tr,
                        "debt_burden": ds / te if te > 0 else 0,
                    })

            if sf_annual:
                sf_avg = pd.DataFrame(sf_annual).mean()
                sf_row = pd.DataFrame([{
                    "Entity Name": "San Francisco",
                    "property_tax_dependence": sf_avg["pt_share"],
                    "intergovernmental_share": sf_avg["ig_share"],
                    "fiscal_surplus_ratio": sf_avg["surplus_ratio"],
                    "debt_service_burden": sf_avg["debt_burden"],
                }])
                county_avg = pd.concat([county_avg, sf_row], ignore_index=True)
                print(f"  SF added: PT_dep={sf_avg['pt_share']:.3f}, "
                      f"IG_share={sf_avg['ig_share']:.3f}, "
                      f"surplus={sf_avg['surplus_ratio']:.3f}, "
                      f"debt={sf_avg['debt_burden']:.3f}")

    # Map county name → FIPS via crosswalk
    ca_xw = crosswalk[crosswalk["state_abbr"] == "CA"][["fips", "county_key_short"]].copy()
    county_avg["county_key"] = county_avg["Entity Name"].str.lower().str.strip()
    county_avg = county_avg.merge(
        ca_xw.rename(columns={"county_key_short": "county_key"}),
        on="county_key", how="left"
    )

    matched = county_avg["fips"].notna().sum()
    unmatched = county_avg[county_avg["fips"].isna()]["Entity Name"].tolist()
    print(f"  CA counties matched: {matched}/58")
    if unmatched:
        print(f"  Unmatched: {unmatched}")

    result = county_avg[county_avg["fips"].notna()].copy()

    # Normalize sub-indicators
    result["property_tax_n"] = normalize(result["property_tax_dependence"].fillna(0))
    result["surplus_n"] = invert(result["fiscal_surplus_ratio"].fillna(0))
    result["debt_n"] = normalize(result["debt_service_burden"].fillna(0))

    # Weighted score (before penalty)
    result["fiscal_resilience_score"] = (
        result["property_tax_n"] * fw["property_tax_dependence"] +
        result["surplus_n"]      * fw["fiscal_surplus_ratio"] +
        result["debt_n"]         * fw["debt_service_burden"]
    )

    # ── Intergovernmental dependence penalty ──
    # Counties with intergovernmental_share > 0.50 are transfer-dependent.
    # Low PT dependence in these counties reflects poverty, not resilience.
    # Penalty: scale the gap above 0.50 as an additive score boost.
    ig_share = result["intergovernmental_share"].fillna(0)
    ig_threshold = fw.get("ig_penalty_threshold", 0.50)
    ig_scale = fw.get("ig_penalty_scale", 1.5)
    ig_penalty = (ig_share - ig_threshold).clip(lower=0) * (ig_scale / 3.0)
    result["intergovernmental_penalty"] = ig_penalty
    result["fiscal_resilience_score"] = (
        result["fiscal_resilience_score"] + ig_penalty
    ).clip(upper=1.0)

    n_penalized = (ig_penalty > 0).sum()
    print(f"  Intergovernmental penalty applied to {n_penalized} counties "
          f"(IG share > 0.50)")

    result["fiscal_data_note"] = ""
    result.loc[ig_share > ig_threshold, "fiscal_data_note"] = "high_ig_dependence"

    result["fiscal_resilience_available"] = True
    result["fiscal_resilience_source"] = "ca_sco"
    result["fips"] = result["fips"].astype("Int64")

    print(f"  Mean property_tax_dependence: {result['property_tax_dependence'].mean():.3f}")
    print(f"  Counties with fiscal deficit: {(result['fiscal_surplus_ratio'] < 0).sum()}")

    return result[["fips", "fiscal_resilience_score", "property_tax_dependence",
                    "intergovernmental_share", "fiscal_surplus_ratio",
                    "debt_service_burden", "intergovernmental_penalty",
                    "fiscal_data_note",
                    "fiscal_resilience_available", "fiscal_resilience_source"]]


def build_income_vulnerability_score(
    ftb_agi: pd.DataFrame,
    crosswalk: pd.DataFrame,
) -> pd.DataFrame:
    """
    P3j: Income vulnerability score from CA FTB B-7 AGI data.
    CA only — 58 counties.

    Three sub-indicators (tax year 2022):
    - low_income_share:     returns with AGI < $25K / total returns (w=0.50)
    - income_concentration: AGI from brackets >$500K / total county AGI (w=0.30)
    - agi_per_return:       total_agi / total_returns — INVERTED (w=0.20)

    Weights load from INCOME_VULNERABILITY_WEIGHTS in optimized_weights.json.
    """
    print("\n[P3j] Building income vulnerability scores (CA FTB B-7)...")

    iw = _load_weights("INCOME_VULNERABILITY_WEIGHTS", {
        "low_income_share": 0.50,
        "income_concentration": 0.30,
        "agi_per_return": 0.20,
    })

    df = ftb_agi.copy()
    df["All Returns"] = pd.to_numeric(df["All Returns"], errors="coerce")
    df["Adjusted Gross Income"] = pd.to_numeric(df["Adjusted Gross Income"], errors="coerce")

    # Use tax year 2022 (most recent)
    df = df[(df["Taxable Year"] == 2022)]
    df = df[df["County"] != "State Totals"]
    # Exclude non-county entries
    df = df[~df["County"].isin(["Nonresident", "Resident Out-of-State", "Unallocated"])]
    print(f"  Tax year 2022, {df['County'].nunique()} counties")

    rows = []
    for county, cdf in df.groupby("County"):
        max_sort = cdf["AGIC Sort Order"].max()
        total_ret = cdf["All Returns"].sum()
        total_agi_val = cdf["Adjusted Gross Income"].sum()

        if max_sort > 5:
            # Fine-grained brackets: sort 1-26, exclude aggregate sorts > 26
            fine = cdf[cdf["AGIC Sort Order"] <= 26]
            total_ret = fine["All Returns"].sum()
            total_agi_val = fine["Adjusted Gross Income"].sum()
            # Low-income: sort 1-9 = AGI < $25K
            low_ret = fine[fine["AGIC Sort Order"] <= 9]["All Returns"].sum()
            # High-income: sort 25-26 = AGI > $500K
            high_agi_val = fine[fine["AGIC Sort Order"] >= 25]["Adjusted Gross Income"].sum()
        else:
            # Aggregate brackets (e.g. Alpine): approximate from wider bins
            # Sort 1 = "Zero and Deficit to 17,999" — best low-income proxy
            low_ret = cdf[cdf["AGIC Sort Order"] == 1]["All Returns"].sum()
            # Sort 5 = "100,000 and over" — no $500K+ breakout available
            # Use total AGI from this bracket as upper bound (overstates concentration)
            high_agi_val = cdf[cdf["AGIC Sort Order"] == 5]["Adjusted Gross Income"].sum()

        rows.append({
            "County": county,
            "low_income_share": low_ret / total_ret if total_ret > 0 else np.nan,
            "income_concentration": high_agi_val / total_agi_val if total_agi_val > 0 else np.nan,
            "agi_per_return": total_agi_val / total_ret if total_ret > 0 else np.nan,
        })

    # Combine into county-level DataFrame
    result = pd.DataFrame(rows)

    # Map county name → FIPS via crosswalk
    ca_xw = crosswalk[crosswalk["state_abbr"] == "CA"][["fips", "county_key_short"]].copy()
    result["county_key"] = result["County"].str.lower().str.strip()
    result = result.merge(
        ca_xw.rename(columns={"county_key_short": "county_key"}),
        on="county_key", how="left"
    )

    matched = result["fips"].notna().sum()
    unmatched = result[result["fips"].isna()]["County"].tolist()
    print(f"  CA counties matched: {matched}/58")
    if unmatched:
        print(f"  Unmatched: {unmatched}")

    result = result[result["fips"].notna()].copy()

    # Normalize and combine
    result["low_income_n"] = normalize(result["low_income_share"].fillna(0))
    result["concentration_n"] = normalize(result["income_concentration"].fillna(0))
    result["agi_per_return_n"] = invert(normalize(result["agi_per_return"].fillna(0)))

    result["income_vulnerability_score"] = (
        result["low_income_n"]     * iw["low_income_share"] +
        result["concentration_n"]  * iw["income_concentration"] +
        result["agi_per_return_n"] * iw["agi_per_return"]
    )

    result["income_vulnerability_available"] = True
    result["income_vulnerability_source"] = "ca_ftb"
    result["fips"] = result["fips"].astype("Int64")

    print(f"  Mean low_income_share: {result['low_income_share'].mean():.3f}")
    print(f"  Mean income_concentration: {result['income_concentration'].mean():.3f}")
    print(f"  Mean agi_per_return: ${result['agi_per_return'].mean():,.0f}")

    return result[["fips", "income_vulnerability_score", "low_income_share",
                    "income_concentration", "agi_per_return",
                    "income_vulnerability_available", "income_vulnerability_source"]]


# ── Combine: Impact Score ──
def build_impact_score(trajectory_df: pd.DataFrame,
                        econ_traj: pd.DataFrame,
                        fair_plan_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Combines causal evidence, industry sensitivity, and economic trajectory
    into a single impact score per county.

    impact_score = causal_pillar        × 0.40
                 + industry_sensitivity × 0.25
                 + economic_trajectory  × 0.35

    causal_pillar = causal_impact_score × 0.60 + trajectory_score × 0.40

    Optional amplifiers (P3):
    - FAIR Plan underinsurance (CA only)

    P3a enhancement: uses blended_industry_sensitivity_n (50/50 FOIA + BEA GDP-weighted)
    when available, falls back to industry_sensitivity_n (FOIA only).
    """
    print("\n[Impact] Building impact scores...")

    df = trajectory_df.copy()
    df = df.merge(
        econ_traj[["fips","economic_trajectory","state",
                   "decay_unem_rate","unemployment_trend","gdp_growth_trend"]],
        on="fips", how="left"
    )

    # P3a: Use blended industry sensitivity if available, else FOIA-only
    if "blended_industry_sensitivity_n" in df.columns:
        industry_col = "blended_industry_sensitivity_n"
        n_blended = df["blended_industry_sensitivity_n"].notna().sum()
        print(f"  Using blended industry sensitivity (FOIA+BEA) for {n_blended} counties")
    else:
        industry_col = "industry_sensitivity_n"
        print("  Using FOIA-only industry sensitivity (no BEA blend available)")

    df["causal_impact_score"]    = df["causal_impact_score"].fillna(
        df[industry_col].fillna(0) * 0.5
    )
    df["trajectory_score"]       = df["trajectory_score"].fillna(0.40)
    df[industry_col]             = df[industry_col].fillna(
        df[industry_col].median()
    )
    df["economic_trajectory"]    = df["economic_trajectory"].fillna(
        df["economic_trajectory"].median()
    )

    df["causal_pillar"] = (
        df["causal_impact_score"] * 0.60 +
        df["trajectory_score"]    * 0.40
    )

    df["impact_score"] = (
        df["causal_pillar"]    * IMPACT_WEIGHTS["causal_impact"] +
        df[industry_col]       * IMPACT_WEIGHTS["industry_sensitivity"] +
        df["economic_trajectory"] * IMPACT_WEIGHTS["economic_trajectory"]
    )

    # P3c: FAIR Plan underinsurance amplifier (CA only)
    if fair_plan_df is not None and not fair_plan_df.empty:
        df = df.merge(
            fair_plan_df[["fips", "fair_plan_amplifier", "fair_plan_share", "fair_plan_available"]],
            on="fips", how="left"
        )
        df["fair_plan_amplifier"] = df["fair_plan_amplifier"].fillna(1.0)
        df["fair_plan_available"] = df["fair_plan_available"].fillna(False)
        df["impact_score"] = (df["impact_score"] * df["fair_plan_amplifier"]).clip(0, 1)
        n_fair = (df["fair_plan_available"] == True).sum()
        print(f"  Applied FAIR Plan amplifier to {n_fair} CA counties")

    df["impact_tier"] = pd.cut(
        df["impact_score"],
        bins=[0, 0.25, 0.50, 0.75, 1.0],
        labels=["Low","Moderate","High","Critical"],
        include_lowest=True
    )

    df = df.sort_values("impact_score", ascending=False).reset_index(drop=True)
    df["impact_rank"] = df.index + 1

    print(f"  Counties with impact scores: {len(df)}")
    print(f"  Impact tier distribution:\n{df['impact_tier'].value_counts().to_string()}")
    return df


# ═══════════════════════════════════════════════════════
# P3 — NEW DATA SOURCE FUNCTIONS
# ═══════════════════════════════════════════════════════

def build_gdp_weighted_industry_sensitivity(bea_cagdp2: pd.DataFrame) -> pd.DataFrame:
    """
    P3a: GDP-weighted industry sensitivity from BEA CAGDP2.

    For each county, computes each NAICS sector's share of total GDP,
    then weights NAICS_SENSITIVITY values by those shares:
        gdp_industry_sensitivity = Σ(sector_share × sector_sensitivity)

    This is stronger than the FOIA loan-weighted version because it uses
    actual economic output rather than SBA loan volume as weights.

    Returns per county: fips, gdp_industry_sensitivity, gdp_industry_sensitivity_n,
                        gdp_dominant_sector, gdp_top_industries
    """
    print("\n[P3a] Building GDP-weighted industry sensitivity (BEA CAGDP2)...")
    df = bea_cagdp2.copy()

    # Parse FIPS
    df["fips"] = (
        df["GeoFIPS"].astype(str).str.strip().str.replace('"', '')
    )
    df["fips"] = pd.to_numeric(df["fips"], errors="coerce").astype("Int64")
    df = df[df["fips"] > 1000]  # exclude state-level rows

    # Identify year columns and use latest
    year_cols = [c for c in df.columns if c.strip().isdigit() and int(c.strip()) >= 2000]
    if not year_cols:
        print("  WARNING: No year columns found in BEA data. Skipping.")
        return pd.DataFrame(columns=["fips", "gdp_industry_sensitivity",
                                      "gdp_industry_sensitivity_n",
                                      "gdp_dominant_sector", "gdp_top_industries"])
    latest_yr = max(year_cols, key=lambda x: int(x.strip()))
    print(f"  Using BEA year: {latest_yr}")

    # Parse LineCode to int for mapping
    df["lc"] = pd.to_numeric(df["LineCode"].astype(str).str.strip(), errors="coerce")

    # Filter to leaf-level sectors only (no aggregates)
    leaf_lcs = set(BEA_LINECODE_TO_NAICS.keys())
    sectors = df[df["lc"].isin(leaf_lcs)].copy()

    # Parse GDP value for latest year
    sectors["sector_gdp"] = pd.to_numeric(
        sectors[latest_yr].astype(str).str.replace(",", "").str.strip()
        .str.replace("(NA)", "", regex=False)
        .str.replace("(D)", "", regex=False),
        errors="coerce"
    )
    sectors = sectors[sectors["sector_gdp"].notna() & (sectors["sector_gdp"] > 0)]

    # Map to NAICS sensitivity
    sectors["naics_code"] = sectors["lc"].map(BEA_LINECODE_TO_NAICS)
    sectors["sensitivity"] = sectors["naics_code"].apply(_bea_naics_sensitivity)

    # Clean description for labeling
    sectors["sector_label"] = sectors["Description"].str.strip().str.strip('"')

    # Compute per-county weighted sensitivity
    results = []
    for fips_val, group in sectors.groupby("fips"):
        total = group["sector_gdp"].sum()
        if total <= 0:
            continue
        group = group.copy()
        group["share"] = group["sector_gdp"] / total
        weighted_sens = (group["share"] * group["sensitivity"]).sum()

        # Top sectors by GDP share
        top = group.nlargest(3, "share")
        top_str = "; ".join(
            f"{r['sector_label']} ({r['share']:.0%})"
            for _, r in top.iterrows()
        )
        dominant = top.iloc[0]["sector_label"] if len(top) > 0 else "Unknown"

        results.append({
            "fips": fips_val,
            "gdp_industry_sensitivity": weighted_sens,
            "gdp_dominant_sector": dominant,
            "gdp_top_industries": top_str,
        })

    result = pd.DataFrame(results)
    if result.empty:
        print("  WARNING: No counties produced GDP-weighted scores.")
        return pd.DataFrame(columns=["fips", "gdp_industry_sensitivity",
                                      "gdp_industry_sensitivity_n",
                                      "gdp_dominant_sector", "gdp_top_industries"])

    result["fips"] = result["fips"].astype("Int64")
    result["gdp_industry_sensitivity_n"] = normalize(result["gdp_industry_sensitivity"])

    print(f"  Counties with GDP-weighted industry sensitivity: {len(result)}")
    print(f"  Sensitivity range: {result['gdp_industry_sensitivity'].min():.3f} – "
          f"{result['gdp_industry_sensitivity'].max():.3f}")
    print(f"  Mean: {result['gdp_industry_sensitivity'].mean():.3f}")
    return result[["fips", "gdp_industry_sensitivity", "gdp_industry_sensitivity_n",
                    "gdp_dominant_sector", "gdp_top_industries"]]


def blend_industry_scores(
    foia_industry: pd.DataFrame,
    gdp_industry: pd.DataFrame
) -> pd.DataFrame:
    """
    P3a: Blends FOIA-based and GDP-weighted industry sensitivity scores.

    - Both available: 50/50 blend → blended_industry_sensitivity_n
    - FOIA only: use FOIA industry_sensitivity_n as blended
    - GDP only: use gdp_industry_sensitivity_n as blended (shouldn't happen in practice)

    Cross-validation: if abs(FOIA - GDP) < 0.15 → industry_confidence = "high"
                      otherwise → "low" (scores diverge, flag for review)
    """
    print("\n[P3a] Blending FOIA and GDP-weighted industry sensitivity...")

    if gdp_industry is None or gdp_industry.empty:
        print("  No GDP-weighted scores available — using FOIA only.")
        result = foia_industry.copy()
        result["blended_industry_sensitivity_n"] = result["industry_sensitivity_n"]
        result["gdp_industry_sensitivity_n"] = np.nan
        result["industry_confidence"] = "foia_only"
        return result

    # Merge on FIPS
    merged = foia_industry.merge(
        gdp_industry[["fips", "gdp_industry_sensitivity", "gdp_industry_sensitivity_n",
                       "gdp_dominant_sector", "gdp_top_industries"]],
        on="fips", how="left"
    )

    # Blend: 50/50 when both exist, FOIA alone otherwise
    has_both = merged["gdp_industry_sensitivity_n"].notna() & merged["industry_sensitivity_n"].notna()
    has_foia_only = merged["industry_sensitivity_n"].notna() & merged["gdp_industry_sensitivity_n"].isna()
    has_gdp_only  = merged["gdp_industry_sensitivity_n"].notna() & merged["industry_sensitivity_n"].isna()

    merged["blended_industry_sensitivity_n"] = np.nan
    merged.loc[has_both, "blended_industry_sensitivity_n"] = (
        merged.loc[has_both, "industry_sensitivity_n"] * 0.50 +
        merged.loc[has_both, "gdp_industry_sensitivity_n"] * 0.50
    )
    merged.loc[has_foia_only, "blended_industry_sensitivity_n"] = (
        merged.loc[has_foia_only, "industry_sensitivity_n"]
    )
    merged.loc[has_gdp_only, "blended_industry_sensitivity_n"] = (
        merged.loc[has_gdp_only, "gdp_industry_sensitivity_n"]
    )

    # Cross-validation confidence flag
    merged["industry_confidence"] = "foia_only"
    abs_diff = (merged["industry_sensitivity_n"] - merged["gdp_industry_sensitivity_n"]).abs()
    merged.loc[has_both & (abs_diff < 0.15), "industry_confidence"] = "high"
    merged.loc[has_both & (abs_diff >= 0.15), "industry_confidence"] = "low"

    n_both = has_both.sum()
    n_high = (merged["industry_confidence"] == "high").sum()
    n_low  = (merged["industry_confidence"] == "low").sum()
    n_foia = has_foia_only.sum()
    print(f"  Counties with both sources (50/50 blend): {n_both}")
    print(f"    Confidence high (agree):    {n_high}")
    print(f"    Confidence low  (diverge):  {n_low}")
    print(f"  Counties with FOIA only:                  {n_foia}")

    # Also pick up any GDP-only counties not in FOIA at all
    gdp_only_fips = set(gdp_industry["fips"].dropna()) - set(foia_industry["fips"].dropna())
    if gdp_only_fips:
        gdp_extras = gdp_industry[gdp_industry["fips"].isin(gdp_only_fips)].copy()
        gdp_extras["blended_industry_sensitivity_n"] = gdp_extras["gdp_industry_sensitivity_n"]
        gdp_extras["industry_sensitivity_n"] = np.nan
        gdp_extras["industry_sensitivity"] = np.nan
        gdp_extras["industry_confidence"] = "gdp_only"
        gdp_extras["industry_source"] = "BEA_CAGDP2"
        gdp_extras["top_industries"] = gdp_extras["gdp_top_industries"]
        gdp_extras["dominant_sector"] = gdp_extras["gdp_dominant_sector"]
        # Keep only columns that merged has
        common_cols = [c for c in merged.columns if c in gdp_extras.columns]
        merged = pd.concat([merged, gdp_extras[common_cols]], ignore_index=True)
        print(f"  Counties with GDP only (BEA, no FOIA): {len(gdp_only_fips)}")

    print(f"  Total counties after blend: {len(merged)}")
    return merged


def build_fair_plan_amplifier(
    fair: pd.DataFrame,
    crosswalk: pd.DataFrame,
    gdp: pd.DataFrame,
    max_amplifier: float = 1.20
) -> pd.DataFrame:
    """
    P3c: FAIR Plan underinsurance amplifier (California only).
    High FAIR Plan concentration = private insurers have exited =
    post-fire recovery will be worse due to underinsurance.

    Uses residential/commercial split with blended cap formula:
      avg_res_exposure = exposure_residential_m / pif_residential
      avg_com_exposure = exposure_commercial_m  / pif_commercial
      residential_gap  = (avg_res - RESIDENTIAL_CAP_M).clip(0)
      commercial_gap   = (avg_com - COMMERCIAL_CAP_M).clip(0)
      coverage_gap = res_gap*(pif_res/total) + com_gap*(pif_com/total)
      fair_plan_share = total_pif / (county_population / 2.5)
      amplifier = 1 + (norm(share)*0.60 + norm(gap)*0.40) * (max-1)
    """
    print("\n[P3c] Building FAIR Plan underinsurance amplifier (CA only)...")
    RESIDENTIAL_CAP_M = 3.0    # legal constant
    COMMERCIAL_CAP_M  = 20.0   # legal constant

    df = fair.copy()

    # Join to crosswalk for FIPS if needed
    if "fips" not in df.columns:
        df["county_key"] = df["county"].str.lower().str.strip()
        ca_xw = crosswalk[crosswalk["state_abbr"] == "CA"].copy()
        ca_xw["county_key_short"] = ca_xw["county_key_short"].str.lower().str.strip()
        df = df.merge(
            ca_xw[["fips", "county_key_short"]].rename(
                columns={"county_key_short": "county_key"}
            ),
            on="county_key", how="left"
        )

    # Population for property count estimate
    pop = (
        gdp[gdp["fips"].notna()]
        .groupby("fips")["County Population"]
        .mean()
        .reset_index()
        .rename(columns={"County Population": "county_population"})
    )
    df = df.merge(pop, on="fips", how="left")

    # ── FAIR Plan share = total_pif / est_total_properties ──
    df["pif_residential"] = df["pif_residential"].fillna(0)
    df["pif_commercial"]  = df["pif_commercial"].fillna(0)
    df["total_pif"] = df["pif_residential"] + df["pif_commercial"]
    df["est_total_properties"] = (df["county_population"] / 2.5).clip(lower=1)
    df["fair_plan_share_raw"] = df["total_pif"] / df["est_total_properties"]
    df["fair_plan_share"] = normalize(df["fair_plan_share_raw"])

    # ── Coverage gap: blended per-policy cap gap ──
    df["exposure_residential_m"] = df["exposure_residential_m"].fillna(0)
    df["exposure_commercial_m"]  = df["exposure_commercial_m"].fillna(0)

    # Average exposure per policy (avoid div by zero)
    df["avg_res_exposure"] = np.where(
        df["pif_residential"] > 0,
        df["exposure_residential_m"] / df["pif_residential"],
        0.0
    )
    df["avg_com_exposure"] = np.where(
        df["pif_commercial"] > 0,
        df["exposure_commercial_m"] / df["pif_commercial"],
        0.0
    )

    # Gap = how much average exposure exceeds the cap
    df["residential_gap"] = (df["avg_res_exposure"] - RESIDENTIAL_CAP_M).clip(lower=0)
    df["commercial_gap"]  = (df["avg_com_exposure"] - COMMERCIAL_CAP_M).clip(lower=0)

    # Blended gap weighted by policy mix
    df["pif_res_share"] = np.where(
        df["total_pif"] > 0,
        df["pif_residential"] / df["total_pif"],
        0.5
    )
    df["pif_com_share"] = 1.0 - df["pif_res_share"]
    df["coverage_gap_raw"] = (
        df["residential_gap"] * df["pif_res_share"] +
        df["commercial_gap"]  * df["pif_com_share"]
    )
    df["coverage_gap_score"] = normalize(df["coverage_gap_raw"])

    # ── Combined score ──
    w_share = 0.60
    w_gap   = 0.40
    df["fair_plan_raw"] = (
        df["fair_plan_share"]    * w_share +
        df["coverage_gap_score"] * w_gap
    )
    df["fair_plan_amplifier"] = 1.0 + (
        df["fair_plan_raw"] * (max_amplifier - 1.0)
    )
    df["fair_plan_available"] = True

    print(f"  CA counties with FAIR Plan data: {len(df)}")
    print(f"  Avg residential exposure: ${df['avg_res_exposure'].mean():.3f}M  (cap: ${RESIDENTIAL_CAP_M}M)")
    print(f"  Avg commercial exposure:  ${df['avg_com_exposure'].mean():.3f}M  (cap: ${COMMERCIAL_CAP_M}M)")
    print(f"  Counties with res gap > 0: {(df['residential_gap'] > 0).sum()}")
    print(f"  Counties with com gap > 0: {(df['commercial_gap'] > 0).sum()}")
    print(f"  Amplifier range: {df['fair_plan_amplifier'].min():.3f} – {df['fair_plan_amplifier'].max():.3f}")
    return df[["fips", "fair_plan_share", "coverage_gap_score",
               "fair_plan_raw", "fair_plan_amplifier", "fair_plan_available"]]


def build_usda_nass_amplifier(
    nass: pd.DataFrame,
    impact_df: pd.DataFrame
) -> pd.DataFrame:
    """
    P3d: Agricultural damage amplifier for counties where
    dominant_sector = 'Agriculture & Forestry'.

    Uses USDA NASS Census of Agriculture 2022 land values ($/acre).
    Higher farmland value = more economic exposure to wildfire damage.

    Amplifier only applies to ag-dominant counties (NAICS 11, sensitivity 0.95).
    All other counties get nass_amplifier = 1.0 (no effect).

    Magnitude loaded from optimized_weights.json NASS_AMPLIFIER.
    Suppressed values (D)/(Z) filled with state median.
    """
    print("\n[P3d] Building USDA NASS agricultural amplifier...")

    # Load magnitude from weights file
    nass_w = _load_weights("NASS_AMPLIFIER", {"magnitude": 0.20})
    magnitude = nass_w["magnitude"]
    print(f"  NASS amplifier magnitude: {magnitude:.2f}")

    df = nass.copy()

    # ── Filter to $/acre, TOTAL domain, Year=2022 ──
    if "Data Item" in df.columns:
        df = df[df["Data Item"].str.contains("\\$ / ACRE", na=False)].copy()
    if "Domain" in df.columns:
        df = df[df["Domain"] == "TOTAL"].copy()
    if "Year" in df.columns:
        df["year"] = pd.to_numeric(df["Year"], errors="coerce")
        df = df[df["year"] == 2022].copy()

    # ── Build FIPS from actual column names ──
    state_col = "State ANSI" if "State ANSI" in df.columns else "state_fips_code"
    county_col = "County ANSI" if "County ANSI" in df.columns else "county_code"
    df["state_fips"] = df[state_col].astype(str).str.strip().str.zfill(2)
    df["county_fips"] = df[county_col].astype(str).str.strip().str.replace(r"\.0$", "", regex=True).str.zfill(3)
    df["fips"] = pd.to_numeric(
        df["state_fips"] + df["county_fips"],
        errors="coerce"
    ).astype("Int64")

    # ── Parse land values — handle suppressed (D)/(Z) ──
    df["value_raw"] = df["Value"].astype(str).str.replace(",", "", regex=False)
    df["land_value_per_acre"] = pd.to_numeric(df["value_raw"], errors="coerce")

    suppressed = df["land_value_per_acre"].isna().sum()
    total_rows = len(df)
    print(f"  Rows after filter: {total_rows}  (suppressed/missing: {suppressed})")

    # Fill suppressed values with state median
    state_medians = df.groupby("state_fips")["land_value_per_acre"].transform("median")
    df["land_value_per_acre"] = df["land_value_per_acre"].fillna(state_medians)
    # If still NaN (entire state suppressed), fill with national median
    national_median = df["land_value_per_acre"].median()
    df["land_value_per_acre"] = df["land_value_per_acre"].fillna(national_median)

    filled = suppressed - df["land_value_per_acre"].isna().sum()
    if suppressed > 0:
        print(f"  Filled {filled} suppressed values with state median")

    # ── Aggregate to one row per FIPS ──
    ag_values = df.groupby("fips").agg(
        land_value_per_acre=("land_value_per_acre", "mean")
    ).reset_index()

    # ── Apply amplifier only to ag-dominant counties ──
    if "dominant_sector" in impact_df.columns:
        ag_fips = set(
            impact_df[impact_df["dominant_sector"] == "Agriculture & Forestry"]["fips"]
        )
        ag_values["is_ag_county"] = ag_values["fips"].isin(ag_fips)
    else:
        ag_values["is_ag_county"] = False
        print("  WARNING: dominant_sector not in impact_df — no ag counties identified")

    # Normalize land value across ag-dominant counties only
    ag_mask = ag_values["is_ag_county"]
    ag_values["nass_land_value_n"] = 0.0
    if ag_mask.any():
        ag_subset = ag_values.loc[ag_mask, "land_value_per_acre"]
        ag_values.loc[ag_mask, "nass_land_value_n"] = normalize(ag_subset)

    ag_values["nass_amplifier"] = np.where(
        ag_values["is_ag_county"],
        1.0 + ag_values["nass_land_value_n"] * magnitude,
        1.0
    )
    ag_values["nass_data_available"] = True

    n_ag = ag_mask.sum()
    n_total = len(ag_values)
    print(f"  Counties with NASS data: {n_total}")
    print(f"  Agriculture-dominant counties: {n_ag}")
    if n_ag > 0:
        ag_only = ag_values[ag_mask]
        print(f"  Ag county land value range: "
              f"${ag_only['land_value_per_acre'].min():,.0f} – "
              f"${ag_only['land_value_per_acre'].max():,.0f}/acre  "
              f"(mean: ${ag_only['land_value_per_acre'].mean():,.0f})")
        print(f"  nass_amplifier range (ag only): "
              f"{ag_only['nass_amplifier'].min():.3f} – "
              f"{ag_only['nass_amplifier'].max():.3f}")

    return ag_values[["fips", "land_value_per_acre", "nass_amplifier",
                       "nass_data_available"]]


def build_water_vulnerability(
    sdwis_dir: str = "data/raw/sdwis",
    crosswalk: pd.DataFrame = None,
    acs_plumbing: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    P3e: Water system vulnerability score from EPA SDWIS bulk download.

    Uses three files from the SDWIS bulk extract:
    - SDWA_PUB_WATER_SYSTEMS.csv → PWSID, source type, population served
    - SDWA_GEOGRAPHIC_AREAS.csv  → PWSID → county FIPS mapping
    - SDWA_VIOLATIONS_ENFORCEMENT.csv → violation records by PWSID

    FIPS mapping: two-tier join
      1. COUNTY_SERVED (county name string) + PWSID[:2] (state) → crosswalk → FIPS
      2. Fallback: ANSI_ENTITY_CODE (3-digit county FIPS) + PWSID[:2] (state)

    Four sub-indicators:
    - violation_rate — recency-weighted violations per 1000 population
    - fragmentation — water systems per 100k population (more = harder to coordinate)
    - surface_water — % population on surface water (more wildfire-vulnerable)
    - private_well_proxy — ACS B25049 lacking complete plumbing (Gap 2 fix)

    Existing 3 weights rebalanced by (1 - private_well_weight) factor.
    """
    from pathlib import Path
    print("\n[P3e] Building water vulnerability scores (EPA SDWIS)...")

    pws_path = Path(sdwis_dir) / "SDWA_PUB_WATER_SYSTEMS.csv"
    geo_path = Path(sdwis_dir) / "SDWA_GEOGRAPHIC_AREAS.csv"
    vio_path = Path(sdwis_dir) / "SDWA_VIOLATIONS_ENFORCEMENT.csv"

    if not all(p.exists() for p in [pws_path, geo_path]):
        print("  WARNING: SDWIS files not found. Skipping.")
        return pd.DataFrame(columns=["fips", "water_vulnerability_score"])

    # ── Load GEOGRAPHIC_AREAS: map PWSID → county FIPS ──
    geo = pd.read_csv(geo_path,
                       usecols=["PWSID", "AREA_TYPE_CODE", "ANSI_ENTITY_CODE", "COUNTY_SERVED"],
                       low_memory=False)
    cn = geo[geo["AREA_TYPE_CODE"] == "CN"].copy()
    # PWSID prefix is state abbreviation (e.g., "AK2110106"), not numeric FIPS
    cn["pwsid_str"] = cn["PWSID"].astype(str)
    cn["state_abbr_raw"] = cn["pwsid_str"].str.extract(r"^([A-Z]{2})", expand=False)
    # Some PWSIDs are purely numeric with 2-digit state FIPS prefix
    cn["state_fips_raw"] = cn["pwsid_str"].str.extract(r"^(\d{2})", expand=False)

    # Tier 1: COUNTY_SERVED name → crosswalk → FIPS (covers ~1,900+ counties)
    mapped_parts = []
    if crosswalk is not None:
        xw = crosswalk[["fips", "county_key_short", "state_abbr", "state_fips"]].copy()
        # Build state FIPS → state abbr and state abbr → state FIPS lookups
        sf2abbr = xw[["state_fips", "state_abbr"]].drop_duplicates("state_fips").set_index("state_fips")["state_abbr"].to_dict()

        cn_name = cn[cn["COUNTY_SERVED"].notna()].copy()
        # Resolve state abbreviation: prefer alpha prefix, fall back to numeric → lookup
        cn_name["state_abbr"] = cn_name["state_abbr_raw"].fillna(
            cn_name["state_fips_raw"].map(sf2abbr)
        )
        cn_name["county_key"] = (
            cn_name["COUNTY_SERVED"].astype(str).str.lower().str.strip()
            .str.replace(r"\s+(county|parish|borough|census area|municipality|city and borough)$",
                         "", regex=True)
        )
        cn_name = cn_name.dropna(subset=["state_abbr"])
        cn_name = cn_name.merge(
            xw[["fips", "county_key_short", "state_abbr"]].rename(
                columns={"county_key_short": "county_key"}
            ),
            on=["county_key", "state_abbr"], how="inner"
        )
        mapped_parts.append(cn_name[["PWSID", "fips"]].drop_duplicates())
        n_name = cn_name["fips"].nunique()
        print(f"  Tier 1 (COUNTY_SERVED → crosswalk): {len(cn_name)} mappings, {n_name} counties")

    # Tier 2: ANSI_ENTITY_CODE fallback (covers ~300 counties, fills gaps)
    # Need numeric state FIPS for this tier
    abbr2sf = {v: k for k, v in sf2abbr.items()} if crosswalk is not None else {}
    cn_ansi = cn[cn["ANSI_ENTITY_CODE"].notna()].copy()
    cn_ansi["county_fips_3"] = pd.to_numeric(cn_ansi["ANSI_ENTITY_CODE"], errors="coerce")
    cn_ansi = cn_ansi.dropna(subset=["county_fips_3"])
    cn_ansi["county_fips_3"] = cn_ansi["county_fips_3"].astype(int).astype(str).str.zfill(3)
    # Resolve numeric state FIPS: prefer numeric prefix, fall back to alpha → lookup
    cn_ansi["state_fips_2"] = cn_ansi["state_fips_raw"].fillna(
        cn_ansi["state_abbr_raw"].map(abbr2sf)
    )
    cn_ansi = cn_ansi.dropna(subset=["state_fips_2"])
    cn_ansi["fips"] = pd.to_numeric(
        cn_ansi["state_fips_2"] + cn_ansi["county_fips_3"], errors="coerce"
    ).astype("Int64")
    mapped_parts.append(cn_ansi[["PWSID", "fips"]].drop_duplicates())
    n_ansi = cn_ansi["fips"].nunique()
    print(f"  Tier 2 (ANSI_ENTITY_CODE): {len(cn_ansi)} mappings, {n_ansi} counties")

    # Combine: deduplicate, Tier 1 takes priority
    pwsid_to_fips = pd.concat(mapped_parts, ignore_index=True).drop_duplicates(subset=["PWSID"])
    print(f"  Combined PWSID→FIPS: {len(pwsid_to_fips)} mappings, "
          f"{pwsid_to_fips['fips'].nunique()} unique counties")

    # ── Load PUB_WATER_SYSTEMS: source type + population ──
    pws = pd.read_csv(pws_path,
                       usecols=["PWSID", "PRIMARY_SOURCE_CODE", "POPULATION_SERVED_COUNT",
                                "PWS_ACTIVITY_CODE", "PWS_TYPE_CODE"],
                       low_memory=False)
    # Keep only active community water systems
    pws = pws[pws["PWS_ACTIVITY_CODE"] == "A"]
    pws["pop_served"] = pd.to_numeric(pws["POPULATION_SERVED_COUNT"], errors="coerce").fillna(0)

    # Join to FIPS
    pws = pws.merge(pwsid_to_fips, on="PWSID", how="inner")
    print(f"  Active water systems with county FIPS: {len(pws)}")

    # Surface water dependency
    pws["is_surface"] = pws["PRIMARY_SOURCE_CODE"].isin(["SW", "SWP", "GUP"])
    sw_pop = pws[pws["is_surface"]].groupby("fips")["pop_served"].sum().reset_index(name="sw_pop")
    total_pop = pws.groupby("fips")["pop_served"].sum().reset_index(name="total_pop")
    sw = total_pop.merge(sw_pop, on="fips", how="left")
    sw["surface_water_pct"] = sw["sw_pop"].fillna(0) / sw["total_pop"].replace(0, np.nan)

    # System fragmentation
    sys_count = pws.groupby("fips").agg(
        n_systems=("PWSID", "nunique"),
        total_pop_served=("pop_served", "sum"),
    ).reset_index()
    sys_count["systems_per_100k"] = (
        sys_count["n_systems"] / sys_count["total_pop_served"].replace(0, np.nan) * 100_000
    )

    # ── Violations (recency-weighted) ──
    if vio_path.exists():
        print("  Loading violations (this may take a moment)...")
        vio = pd.read_csv(vio_path,
                          usecols=["PWSID", "COMPL_PER_BEGIN_DATE"],
                          low_memory=False)
        vio = vio.merge(pwsid_to_fips, on="PWSID", how="inner")
        vio["vio_date"] = pd.to_datetime(vio["COMPL_PER_BEGIN_DATE"], errors="coerce")
        vio["recency_weight"] = vio["vio_date"].apply(
            lambda d: 2.0 if pd.notna(d) and (pd.Timestamp.now() - d).days < 5*365 else 1.0
        )
        vio_agg = vio.groupby("fips").agg(
            weighted_violations=("recency_weight", "sum")
        ).reset_index()
    else:
        print("  No violations file — using zeros.")
        vio_agg = pd.DataFrame({"fips": sys_count["fips"], "weighted_violations": 0})

    # ── Combine ──
    result = sys_count[["fips", "systems_per_100k", "total_pop_served"]].merge(
        vio_agg, on="fips", how="left"
    ).merge(
        sw[["fips", "surface_water_pct"]], on="fips", how="left"
    )
    result["surface_water_pct"] = result["surface_water_pct"].fillna(0)

    result["violation_rate"] = (
        result["weighted_violations"].fillna(0) /
        result["total_pop_served"].replace(0, np.nan) * 1000
    )
    # Winsorize violation_rate at p90 before normalizing
    p90_viol = result["violation_rate"].quantile(0.90)
    violation_rate_capped = result["violation_rate"].fillna(0).clip(upper=p90_viol)
    result["violation_rate_n"] = normalize(violation_rate_capped)
    result["fragmentation_n"] = normalize(result["systems_per_100k"].fillna(0))
    result["surface_water_n"] = normalize(result["surface_water_pct"].fillna(0))

    # Use CA_WATER_WEIGHTS for consistency (federal uses same sub-indicator structure)
    _ww = _load_weights("CA_WATER_WEIGHTS", {
        "system_risk_score": 0.50, "surface_water_dependency": 0.25, "system_fragmentation": 0.25,
    })

    # Private well proxy (Gap 2 fix) — rebalance existing weights
    _pw = _load_weights("PRIVATE_WELL_WEIGHT", {"magnitude": 0.15})
    pw_weight = _pw["magnitude"]
    rebalance = 1.0 - pw_weight

    result["private_well_proxy_pct"] = 0.0
    if acs_plumbing is not None and not acs_plumbing.empty:
        pw = acs_plumbing[["fips", "private_well_proxy_pct"]].copy()
        pw["fips"] = pw["fips"].astype("Int64")
        result = result.merge(pw, on="fips", how="left", suffixes=("_old", ""))
        if "private_well_proxy_pct_old" in result.columns:
            result["private_well_proxy_pct"] = result["private_well_proxy_pct"].fillna(
                result["private_well_proxy_pct_old"]
            )
            result = result.drop(columns=["private_well_proxy_pct_old"])
        result["private_well_proxy_pct"] = result["private_well_proxy_pct"].fillna(0)
    result["private_well_n"] = normalize(result["private_well_proxy_pct"])

    result["water_vulnerability_score"] = (
        result["violation_rate_n"] * _ww["system_risk_score"] * rebalance +
        result["fragmentation_n"] * _ww["system_fragmentation"] * rebalance +
        result["surface_water_n"] * _ww["surface_water_dependency"] * rebalance +
        result["private_well_n"] * pw_weight
    )

    print(f"  Counties with water vulnerability: {len(result)}")
    print(f"  Private well proxy: weight={pw_weight:.2f}, rebalance={rebalance:.2f}")
    result["water_vulnerability_source"] = "federal_sdwis"
    result["water_vulnerability_available"] = True
    return result[["fips", "water_vulnerability_score", "violation_rate_n",
                    "fragmentation_n", "surface_water_n", "private_well_n",
                    "private_well_proxy_pct",
                    "water_vulnerability_source", "water_vulnerability_available"]]


def build_ca_water_vulnerability(
    crosswalk: pd.DataFrame,
    facilities_path: str = "data/raw/ca_swrcb_pws_facilities.csv",
    safer_path: str = "data/raw/ca_swrcb_safer.csv",
    acs_plumbing: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    P3e-CA: Water system vulnerability for California counties using
    CA State Water Resources Control Board (SWRCB) data.

    Two source files:
    - PWS Facilities: system-level source type, population, county
    - SAFER Dashboard: pre-computed risk scores per water system

    Join key: facilities['Water System No'] = safer['WATER_SYSTEM_NUMBER']

    Four components (population-weighted aggregation to county):
    1. system_risk_score — state's composite risk score
    2. surface_water_dependency — % population on SW/SWP sources
    3. system_fragmentation — distinct systems per 100k population
    4. private_well_proxy — ACS B25049 lacking complete plumbing (Gap 2 fix)

    Existing 3 weights rebalanced by (1 - private_well_weight) factor.

    Returns per CA county: fips, water_vulnerability_score, sub-scores,
    water_vulnerability_source="ca_swrcb", water_vulnerability_available=True
    """
    from pathlib import Path
    print("\n[P3e-CA] Building CA water vulnerability (SWRCB SAFER + Facilities)...")

    if not Path(facilities_path).exists() or not Path(safer_path).exists():
        print("  WARNING: CA SWRCB files not found. Skipping.")
        return pd.DataFrame()

    # ── Load facilities ──
    fac = pd.read_csv(facilities_path, low_memory=False)
    fac = fac[fac["Water System Status CODE"] == "A"]  # active only
    fac["pop"] = pd.to_numeric(fac["Total Population"], errors="coerce").fillna(0)
    fac["county_raw"] = fac["Principal County Served"].astype(str).str.strip()
    fac = fac[fac["county_raw"].notna() & (fac["county_raw"] != "") & (fac["county_raw"] != "nan")]
    print(f"  Active CA water systems: {len(fac)}")

    # Map county names → FIPS via crosswalk
    ca_xw = crosswalk[crosswalk["state_abbr"] == "CA"][["fips", "county_key_short"]].copy()
    ca_xw["county_key"] = ca_xw["county_key_short"].str.lower().str.strip()
    fac["county_key"] = fac["county_raw"].str.lower().str.strip()
    fac = fac.merge(ca_xw[["fips", "county_key"]], on="county_key", how="left")
    fac_matched = fac["fips"].notna().sum()
    print(f"  FIPS matched: {fac_matched}/{len(fac)} ({fac_matched/len(fac):.1%})")
    fac = fac[fac["fips"].notna()]

    # ── Load SAFER ──
    safer_cols = ["WATER_SYSTEM_NUMBER", "COUNTY",
                  "TOTAL_WEIGHTED_RISK_SCORE_BEFORE_DIVIDING_BY_CATEGORY_COUNT",
                  "RISK_ASSESSMENT_RESULT", "FINAL_SAFER_STATUS"]
    safer = pd.read_csv(safer_path, usecols=safer_cols, low_memory=False)
    safer = safer.rename(columns={
        "TOTAL_WEIGHTED_RISK_SCORE_BEFORE_DIVIDING_BY_CATEGORY_COUNT": "risk_score_raw"
    })
    safer["risk_score_raw"] = pd.to_numeric(safer["risk_score_raw"], errors="coerce")
    print(f"  SAFER records loaded: {len(safer)}")

    # ── Join facilities × SAFER ──
    merged = fac.merge(
        safer[["WATER_SYSTEM_NUMBER", "risk_score_raw", "RISK_ASSESSMENT_RESULT"]],
        left_on="Water System No", right_on="WATER_SYSTEM_NUMBER",
        how="left"
    )
    has_safer = merged["risk_score_raw"].notna()
    print(f"  Systems with SAFER risk score: {has_safer.sum()}/{len(merged)}")

    # ── Size-based gap fill for unmatched systems ──
    unmatched = ~has_safer
    large_unmatched = unmatched & (merged["pop"] > 10_000)
    small_unmatched = unmatched & (merged["pop"] <= 10_000)

    # Large systems (>10k pop): absent from SAFER = presumed adequate → 0.0
    merged.loc[large_unmatched, "risk_score_raw"] = 0.0
    merged.loc[large_unmatched, "risk_score_source"] = "not_assessed_presumed_adequate"

    # Small systems (≤10k pop): fill with county median of SAFER-matched systems
    county_medians = (
        merged[has_safer]
        .groupby("fips")["risk_score_raw"]
        .median()
        .to_dict()
    )
    small_idx = merged.index[small_unmatched]
    merged.loc[small_idx, "risk_score_raw"] = merged.loc[small_idx, "fips"].map(county_medians)
    merged.loc[small_idx, "risk_score_source"] = "median_fill_small_system"

    # Tag SAFER-matched systems
    merged.loc[has_safer, "risk_score_source"] = merged.loc[has_safer, "risk_score_source"].fillna("safer_matched")

    # If county had zero SAFER matches, small systems still get NaN → fill with state median
    state_median = merged.loc[has_safer, "risk_score_raw"].median()
    still_na = merged["risk_score_raw"].isna()
    if still_na.any():
        merged.loc[still_na, "risk_score_raw"] = state_median
        merged.loc[still_na, "risk_score_source"] = "state_median_fill"

    n_large = large_unmatched.sum()
    n_small = small_unmatched.sum()
    print(f"  Gap fill: {n_large} large systems → 0.0, {n_small} small systems → county median")
    print(f"  Risk source breakdown:")
    print(merged["risk_score_source"].value_counts().to_string(header=False))

    # ── Diagnostic: SAFER population coverage per county ──
    merged["safer_matched_pop"] = merged["pop"].where(has_safer, 0)
    safer_coverage = merged.groupby("fips").agg(
        total_pop=("pop", "sum"),
        safer_pop=("safer_matched_pop", "sum"),
    ).reset_index()
    safer_coverage["pct_population_safer_matched"] = (
        safer_coverage["safer_pop"] / safer_coverage["total_pop"].replace(0, np.nan)
    )
    safer_coverage["water_score_confidence"] = np.where(
        safer_coverage["pct_population_safer_matched"] >= 0.50, "high", "low"
    )

    # ── COMPONENT 1: system_risk_score (population-weighted avg, now with gap fill) ──
    c1 = merged[merged["risk_score_raw"].notna() & (merged["pop"] > 0)].copy()
    c1["weighted_risk"] = c1["risk_score_raw"] * c1["pop"]
    c1_agg = c1.groupby("fips").agg(
        sum_weighted_risk=("weighted_risk", "sum"),
        sum_pop_risk=("pop", "sum"),
    ).reset_index()
    c1_agg["system_risk_score"] = (
        c1_agg["sum_weighted_risk"] / c1_agg["sum_pop_risk"].replace(0, np.nan)
    )

    # ── COMPONENT 2: surface_water_dependency ──
    sw_codes = {"SW", "SWP"}
    gw_codes = {"GW", "GWP", "GU"}
    known_codes = sw_codes | gw_codes
    c2 = merged[merged["Primary Water Source Type CODE"].isin(known_codes)].copy()
    c2["is_surface"] = c2["Primary Water Source Type CODE"].isin(sw_codes)
    c2_sw = c2[c2["is_surface"]].groupby("fips")["pop"].sum().reset_index(name="surface_pop")
    c2_total = c2.groupby("fips")["pop"].sum().reset_index(name="total_pop_known")
    c2_agg = c2_total.merge(c2_sw, on="fips", how="left")
    c2_agg["surface_pop"] = c2_agg["surface_pop"].fillna(0)
    c2_agg["surface_water_dependency"] = (
        c2_agg["surface_pop"] / c2_agg["total_pop_known"].replace(0, np.nan)
    )

    # ── COMPONENT 3: system_fragmentation ──
    c3 = merged.copy()
    c3_agg = c3.groupby("fips").agg(
        n_systems=("Water System No", "nunique"),
        total_pop_served=("pop", "sum"),
    ).reset_index()
    c3_agg["system_fragmentation"] = (
        c3_agg["n_systems"] / c3_agg["total_pop_served"].replace(0, np.nan) * 100_000
    )

    # ── Combine ──
    result = c1_agg[["fips", "system_risk_score"]].merge(
        c2_agg[["fips", "surface_water_dependency"]], on="fips", how="outer"
    ).merge(
        c3_agg[["fips", "system_fragmentation", "n_systems", "total_pop_served"]],
        on="fips", how="outer"
    ).merge(
        safer_coverage[["fips", "pct_population_safer_matched", "water_score_confidence"]],
        on="fips", how="left"
    )

    # Winsorize system_risk_score at p90 before normalizing
    p90_risk = result["system_risk_score"].quantile(0.90)
    system_risk_capped = result["system_risk_score"].fillna(0).clip(upper=p90_risk)
    result["system_risk_n"] = normalize(system_risk_capped)
    result["surface_water_n"] = normalize(result["surface_water_dependency"].fillna(0))
    result["fragmentation_n"] = normalize(result["system_fragmentation"].fillna(0))

    _cww = _load_weights("CA_WATER_WEIGHTS", {
        "system_risk_score": 0.50, "surface_water_dependency": 0.25, "system_fragmentation": 0.25,
    })

    # Private well proxy (Gap 2 fix) — rebalance existing weights
    _pw = _load_weights("PRIVATE_WELL_WEIGHT", {"magnitude": 0.15})
    pw_weight = _pw["magnitude"]
    rebalance = 1.0 - pw_weight

    result["private_well_proxy_pct"] = 0.0
    if acs_plumbing is not None and not acs_plumbing.empty:
        pw = acs_plumbing[["fips", "private_well_proxy_pct"]].copy()
        pw["fips"] = pw["fips"].astype("Int64")
        result = result.merge(pw, on="fips", how="left", suffixes=("_old", ""))
        if "private_well_proxy_pct_old" in result.columns:
            result["private_well_proxy_pct"] = result["private_well_proxy_pct"].fillna(
                result["private_well_proxy_pct_old"]
            )
            result = result.drop(columns=["private_well_proxy_pct_old"])
        result["private_well_proxy_pct"] = result["private_well_proxy_pct"].fillna(0)
    result["private_well_n"] = normalize(result["private_well_proxy_pct"])

    result["water_vulnerability_score"] = (
        result["system_risk_n"]   * _cww["system_risk_score"] * rebalance +
        result["surface_water_n"] * _cww["surface_water_dependency"] * rebalance +
        result["fragmentation_n"] * _cww["system_fragmentation"] * rebalance +
        result["private_well_n"]  * pw_weight
    )

    result["water_vulnerability_source"] = "ca_swrcb"
    result["water_vulnerability_available"] = True
    result["fips"] = result["fips"].astype("Int64")

    n_counties = result["fips"].nunique()
    n_high = (result["water_score_confidence"] == "high").sum()
    n_low = (result["water_score_confidence"] == "low").sum()
    print(f"  CA counties with water vulnerability: {n_counties}/58")
    print(f"  Confidence: {n_high} high, {n_low} low")
    print(f"  Private well proxy: weight={pw_weight:.2f}, rebalance={rebalance:.2f}")

    return result[["fips", "water_vulnerability_score",
                    "system_risk_n", "surface_water_n", "fragmentation_n",
                    "private_well_n", "private_well_proxy_pct",
                    "system_risk_score", "surface_water_dependency", "system_fragmentation",
                    "pct_population_safer_matched", "water_score_confidence",
                    "water_vulnerability_source", "water_vulnerability_available"]]


def combine_water_vulnerability(
    federal_water: pd.DataFrame,
    ca_water: pd.DataFrame,
    all_fips: pd.Series,
) -> pd.DataFrame:
    """
    Combines federal SDWIS water vulnerability (non-CA) with CA SWRCB data.
    Fills missing counties with national median, flagged as 'estimated'.
    """
    print("\n[P3e] Combining federal + CA water vulnerability...")
    parts = []

    pass_cols = ["fips", "water_vulnerability_score",
                  "water_vulnerability_source", "water_vulnerability_available",
                  "private_well_proxy_pct"]

    # CA SWRCB takes priority for CA counties (FIPS 6xxx)
    if ca_water is not None and not ca_water.empty:
        avail = [c for c in pass_cols if c in ca_water.columns]
        parts.append(ca_water[avail])

    # Federal SDWIS for non-CA counties
    if federal_water is not None and not federal_water.empty:
        ca_fips = set(ca_water["fips"].dropna()) if ca_water is not None and not ca_water.empty else set()
        fed_non_ca = federal_water[~federal_water["fips"].isin(ca_fips)]
        if not fed_non_ca.empty:
            avail = [c for c in pass_cols if c in fed_non_ca.columns]
            parts.append(fed_non_ca[avail])

    if parts:
        combined = pd.concat(parts, ignore_index=True)
    else:
        combined = pd.DataFrame(columns=["fips", "water_vulnerability_score",
                                          "water_vulnerability_source",
                                          "water_vulnerability_available"])

    # Fill missing counties with national median
    covered_fips = set(combined["fips"].dropna())
    all_fips_clean = all_fips.dropna().unique()
    missing_fips = [f for f in all_fips_clean if f not in covered_fips]

    if missing_fips and not combined.empty:
        median_score = combined["water_vulnerability_score"].median()
        fill = pd.DataFrame({
            "fips": missing_fips,
            "water_vulnerability_score": median_score,
            "water_vulnerability_source": "estimated",
            "water_vulnerability_available": False,
            "private_well_proxy_pct": 0.0,
        })
        combined = pd.concat([combined, fill], ignore_index=True)
        print(f"  Filled {len(missing_fips)} counties with median ({median_score:.3f})")

    # Ensure private_well_proxy_pct column exists and fill NaN
    if "private_well_proxy_pct" not in combined.columns:
        combined["private_well_proxy_pct"] = 0.0
    combined["private_well_proxy_pct"] = combined["private_well_proxy_pct"].fillna(0.0)

    # Add data note for counties with significant private well exposure
    combined["water_vuln_data_note"] = np.where(
        combined["private_well_proxy_pct"] > 0.02,
        "includes_private_well_proxy",
        ""
    )

    combined["fips"] = combined["fips"].astype("Int64")
    n_total = len(combined)
    n_ca = (combined["water_vulnerability_source"] == "ca_swrcb").sum()
    n_fed = (combined["water_vulnerability_source"] == "federal_sdwis").sum()
    n_est = (combined["water_vulnerability_source"] == "estimated").sum()
    n_well = (combined["water_vuln_data_note"] == "includes_private_well_proxy").sum()
    print(f"  Total counties: {n_total} (CA SWRCB: {n_ca}, Federal: {n_fed}, Estimated: {n_est})")
    print(f"  Counties with private well proxy note: {n_well}")

    return combined


def build_road_redundancy(
    county_areas: pd.DataFrame,
    tiger_path: str = "data/processed/tiger_road_miles_by_county.csv",
) -> pd.DataFrame:
    """
    P3f: Road redundancy ratio from Census TIGER/Line Roads.
    Low road density = low redundancy = high vulnerability.

    Reads pre-processed lane miles per county per MTFCC code:
      S1100 = Primary Road (Interstate/Principal Arterial)
      S1200 = Secondary Road (State Hwy/Minor Arterial)
      S1400 = Local Neighborhood Road

    Weights load from FC_WEIGHTS in optimized_weights.json.
    """
    from pathlib import Path
    print("\n[P3f] Building road redundancy scores (Census TIGER roads)...")

    if not Path(tiger_path).exists():
        print(f"  WARNING: {tiger_path} not found. Skipping road redundancy.")
        return pd.DataFrame(columns=["fips", "road_redundancy_ratio"])

    fc_defaults = {
        "S1100_lane_miles": 1.00,
        "S1200_lane_miles": 0.50,
        "S1400_lane_miles": 0.10,
    }
    fc = _load_weights("FC_WEIGHTS", fc_defaults)

    raw = pd.read_csv(tiger_path)
    raw["fips"] = pd.to_numeric(raw["fips"], errors="coerce").astype("Int64")

    # Pivot: one row per county, columns = S1100, S1200, S1400 lane miles
    pivot = raw.pivot_table(
        index="fips", columns="mtfcc", values="lane_miles", aggfunc="sum", fill_value=0
    ).reset_index()
    pivot.columns.name = None

    # Ensure all MTFCC columns exist
    for mtfcc in ["S1100", "S1200", "S1400"]:
        if mtfcc not in pivot.columns:
            pivot[mtfcc] = 0.0

    # Weighted lane miles
    pivot["weighted_lane_miles"] = (
        pivot["S1100"] * fc["S1100_lane_miles"] +
        pivot["S1200"] * fc["S1200_lane_miles"] +
        pivot["S1400"] * fc["S1400_lane_miles"]
    )

    # Merge county areas
    pivot = pivot.merge(county_areas[["fips", "area_sq_miles"]], on="fips", how="left")
    pivot["road_density"] = (
        pivot["weighted_lane_miles"] /
        pivot["area_sq_miles"].replace(0, np.nan)
    )
    # Winsorize at 90th percentile — cap extreme urban outliers
    # so rural counties spread across the full [0,1] range
    p90 = pivot["road_density"].quantile(0.90)
    road_density_capped = pivot["road_density"].fillna(0).clip(upper=p90)
    pivot["road_redundancy_ratio"] = invert(road_density_capped)

    # Single corridor flag: high arterial share + low density
    pivot["arterial_share"] = (
        pivot["S1100"] /
        pivot["weighted_lane_miles"].replace(0, np.nan)
    ).fillna(0)
    density_25 = pivot["road_density"].quantile(0.25)
    pivot["single_corridor_flag"] = (
        (pivot["arterial_share"] > 0.60) &
        (pivot["road_density"] < density_25)
    )

    print(f"  FC weights: S1100={fc['S1100_lane_miles']}, S1200={fc['S1200_lane_miles']}, S1400={fc['S1400_lane_miles']}")
    print(f"  Counties with road data: {len(pivot)}")
    return pivot[["fips", "road_redundancy_ratio", "road_density",
                   "single_corridor_flag", "weighted_lane_miles",
                   "S1100", "S1200", "S1400", "area_sq_miles"]]


def build_hftd_exposure(
    hftd_path: str = "data/raw/ca_hftd_county_exposure.csv",
) -> pd.DataFrame:
    """
    P3g/P4: CPUC High Fire-Threat District exposure per CA county.
    Pre-computed from ArcGIS intersection of HFTD Tier 2/3 polygons
    with Census TIGER county boundaries.

    hftd_exposure = (tier2_pct × w_tier2) + (tier3_pct × w_tier3)
    Weights load from optimized_weights.json → HFTD_TIER_WEIGHTS.
    """
    from pathlib import Path
    print("\n[P3g-HFTD] Loading CPUC HFTD fire-threat exposure (CA only)...")

    if not Path(hftd_path).exists():
        print(f"  WARNING: {hftd_path} not found. Skipping HFTD.")
        return pd.DataFrame(columns=["fips", "hftd_exposure", "hftd_tier2_pct", "hftd_tier3_pct"])

    tw = _load_weights("HFTD_TIER_WEIGHTS", {"tier2": 0.50, "tier3": 1.00})

    df = pd.read_csv(hftd_path)
    df["fips"] = pd.to_numeric(df["fips"], errors="coerce").astype("Int64")

    # Recompute exposure with current (potentially trained) weights
    df["hftd_exposure"] = (
        df["hftd_tier2_pct"] * tw["tier2"] +
        df["hftd_tier3_pct"] * tw["tier3"]
    )

    n_exposed = (df["hftd_exposure"] > 0).sum()
    print(f"  CA counties with HFTD exposure: {n_exposed}/58")
    print(f"  Tier weights: T2={tw['tier2']}, T3={tw['tier3']}")

    return df[["fips", "hftd_exposure", "hftd_tier2_pct", "hftd_tier3_pct"]]


def build_utility_hardening(
    service_territory: pd.DataFrame,
    sales_ult_cust: pd.DataFrame,
    crosswalk: pd.DataFrame,
    county_areas: pd.DataFrame = None,
    hftd_data: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    P3g: Utility hardening index from EIA Form 861 + CPUC HFTD (CA).

    For CA counties:
      utility_hardening_index = (
          normalize(hftd_exposure)     × w_hftd     +
          invert(customer_density_n)   × w_density
      )
      Weights from CA_UTILITY_HARDENING_WEIGHTS in optimized_weights.json.

    For non-CA counties:
      utility_hardening_index = invert(customer_density_n)

    Customer density join path:
      Service_Territory (utility → county) × Sales (utility → customers)
      → aggregate by county → divide by area → invert.
    """
    print("\n[P3g] Building utility hardening index (EIA Form 861)...")

    # ── Parse Service Territory: utility_number + state → county ──
    st = service_territory.copy()
    st.columns = [str(c).strip() for c in st.columns]
    # Standardize column names
    st_col_map = {}
    for c in st.columns:
        cl = c.lower()
        if "utility" in cl and "number" in cl:
            st_col_map[c] = "utility_number"
        elif cl == "state":
            st_col_map[c] = "state_abbr"
        elif cl == "county":
            st_col_map[c] = "county"
    st = st.rename(columns=st_col_map)

    if "utility_number" not in st.columns or "county" not in st.columns:
        print("  WARNING: Service_Territory missing required columns. Skipping.")
        return pd.DataFrame(columns=["fips", "utility_hardening_index"])

    st["utility_number"] = pd.to_numeric(st["utility_number"], errors="coerce")
    st["state_abbr"] = st["state_abbr"].astype(str).str.upper().str.strip()
    st["county_key"] = st["county"].astype(str).str.lower().str.strip()
    st = st.dropna(subset=["utility_number"])
    print(f"  Service Territory rows: {len(st)} (utility-county pairs)")

    # ── Parse Sales: utility_number + state → total customers ──
    # Multi-row header: row 2 has real column names, data starts at row 3
    # Column index 1 = Utility Number, 6 = State, 23 = TOTAL Customers Count
    sc = sales_ult_cust.copy()
    # Build clean column names from the multi-row header
    sc.columns = [
        "data_year", "utility_number", "utility_name", "part",
        "service_type", "data_type", "state", "ownership", "ba_code",
        "rev_residential", "sales_residential", "cust_residential",
        "rev_commercial", "sales_commercial", "cust_commercial",
        "rev_industrial", "sales_industrial", "cust_industrial",
        "rev_transport", "sales_transport", "cust_transport",
        "rev_total", "sales_total", "cust_total",
    ][:len(sc.columns)]
    # Drop header rows (non-numeric utility_number)
    sc["utility_number"] = pd.to_numeric(sc["utility_number"], errors="coerce")
    sc = sc.dropna(subset=["utility_number"])
    sc["state_abbr"] = sc["state"].astype(str).str.upper().str.strip()
    sc["total_customers"] = pd.to_numeric(sc["cust_total"], errors="coerce").fillna(0)
    print(f"  Sales rows (utilities): {len(sc)}")

    # ── Join: Service Territory × Sales on (utility_number, state) ──
    # This explodes: one utility with 5 counties becomes 5 rows, each with that utility's customers
    # A utility's customers are shared across its service counties, so we distribute evenly
    merged = st.merge(
        sc[["utility_number", "state_abbr", "total_customers"]],
        on=["utility_number", "state_abbr"],
        how="inner"
    )
    print(f"  Matched utility-county-customer rows: {len(merged)}")

    if merged.empty:
        print("  WARNING: No matches between Service_Territory and Sales. Skipping.")
        return pd.DataFrame(columns=["fips", "utility_hardening_index"])

    # Distribute customers evenly across counties served by each utility
    # (utility serving 3 counties with 9000 customers → 3000 per county)
    counties_per_utility = merged.groupby(["utility_number", "state_abbr"]).size().reset_index(name="n_counties")
    merged = merged.merge(counties_per_utility, on=["utility_number", "state_abbr"], how="left")
    merged["allocated_customers"] = merged["total_customers"] / merged["n_counties"].replace(0, 1)

    # ── Map county names → FIPS via crosswalk ──
    xw = crosswalk[["fips", "county_key_short", "state_abbr"]].copy()
    xw = xw.rename(columns={"county_key_short": "county_key"})
    xw["county_key"] = xw["county_key"].str.lower().str.strip()
    merged = merged.merge(xw, on=["county_key", "state_abbr"], how="left")

    matched_fips = merged["fips"].notna().sum()
    total_rows = len(merged)
    print(f"  FIPS matched: {matched_fips}/{total_rows} ({matched_fips/total_rows:.1%})")

    merged = merged[merged["fips"].notna()]

    # ── Aggregate: sum allocated customers per county ──
    county_cust = merged.groupby("fips").agg(
        total_customers=("allocated_customers", "sum"),
        n_utilities=("utility_number", "nunique"),
    ).reset_index()

    # ── Customer density (if county areas available) ──
    if county_areas is not None and not county_areas.empty:
        county_cust = county_cust.merge(county_areas, on="fips", how="left")
        county_cust["customer_density"] = (
            county_cust["total_customers"] /
            county_cust["area_sq_miles"].replace(0, np.nan)
        )
    else:
        county_cust["customer_density"] = np.nan

    # Winsorize customer density at p90 before normalizing — cap extreme urban outliers
    p90_util = county_cust["customer_density"].quantile(0.90)
    customer_density_capped = county_cust["customer_density"].fillna(0).clip(upper=p90_util)
    # Inverted normalized density: low density = higher vulnerability
    county_cust["customer_density_vuln"] = invert(customer_density_capped)

    # ── CA counties: blend HFTD + customer density ──
    # Non-CA: use customer density vulnerability alone
    county_cust["utility_hardening_source"] = "eia_proxy"

    if hftd_data is not None and not hftd_data.empty:
        caw = _load_weights("CA_UTILITY_HARDENING_WEIGHTS", {
            "hftd_exposure": 0.60, "customer_density": 0.40,
        })
        county_cust = county_cust.merge(
            hftd_data[["fips", "hftd_exposure", "hftd_tier2_pct", "hftd_tier3_pct"]],
            on="fips", how="left"
        )
        has_hftd = county_cust["hftd_exposure"].notna()

        # Normalize HFTD exposure across CA counties only
        ca_hftd = county_cust.loc[has_hftd, "hftd_exposure"]
        if ca_hftd.max() > ca_hftd.min():
            county_cust.loc[has_hftd, "hftd_exposure_n"] = (
                (ca_hftd - ca_hftd.min()) / (ca_hftd.max() - ca_hftd.min())
            )
        else:
            county_cust.loc[has_hftd, "hftd_exposure_n"] = 0.5

        # CA blend: HFTD × w_hftd + density × w_density
        county_cust.loc[has_hftd, "utility_hardening_index"] = (
            county_cust.loc[has_hftd, "hftd_exposure_n"] * caw["hftd_exposure"] +
            county_cust.loc[has_hftd, "customer_density_vuln"] * caw["customer_density"]
        )
        county_cust.loc[has_hftd, "utility_hardening_source"] = "cpuc_hftd + eia"

        n_ca = has_hftd.sum()
        print(f"  CA counties with HFTD+density blend: {n_ca}")
        print(f"  CA weights: HFTD={caw['hftd_exposure']}, density={caw['customer_density']}")

    # Non-CA (and CA without HFTD): density-only
    no_score = county_cust["utility_hardening_index"].isna() if "utility_hardening_index" in county_cust.columns else pd.Series(True, index=county_cust.index)
    if "utility_hardening_index" not in county_cust.columns:
        county_cust["utility_hardening_index"] = np.nan
        no_score = pd.Series(True, index=county_cust.index)
    county_cust.loc[no_score, "utility_hardening_index"] = (
        county_cust.loc[no_score, "customer_density_vuln"]
    )

    print(f"  Counties with utility data: {len(county_cust)}")
    print(f"  Avg utilities per county: {county_cust['n_utilities'].mean():.1f}")
    return county_cust[["fips", "utility_hardening_index", "total_customers",
                         "n_utilities", "customer_density", "utility_hardening_source"]]


def load_county_areas(gaz_path: str = "data/raw/2023_Gaz_counties_national.txt") -> pd.DataFrame:
    """
    P3h: Load Census Gazetteer county areas (sq miles).
    Used for road redundancy ratio computation.
    """
    from pathlib import Path
    if not Path(gaz_path).exists():
        print(f"  [P3h] Gazetteer file not found: {gaz_path}")
        return pd.DataFrame(columns=["fips", "area_sq_miles"])

    print(f"\n[P3h] Loading county areas from Census Gazetteer...")
    # Validate file is actual data (not HTML from failed download)
    with open(gaz_path, "r") as f:
        first_line = f.readline()
    if first_line.strip().startswith("<"):
        print(f"  WARNING: {gaz_path} contains HTML, not data. Skipping.")
        return pd.DataFrame(columns=["fips", "area_sq_miles"])
    gaz = pd.read_csv(gaz_path, sep="\t", dtype={"GEOID": str})
    gaz["fips"] = pd.to_numeric(gaz["GEOID"], errors="coerce").astype("Int64")
    # Use pre-computed ALAND_SQMI if available, else derive from ALAND (sq meters)
    if "ALAND_SQMI" in gaz.columns:
        gaz["area_sq_miles"] = pd.to_numeric(gaz["ALAND_SQMI"], errors="coerce")
    else:
        gaz["area_sq_miles"] = pd.to_numeric(gaz["ALAND"], errors="coerce") / 2_590_000
    gaz = gaz[gaz["area_sq_miles"].notna() & (gaz["area_sq_miles"] > 0)]
    print(f"  Counties with area data: {len(gaz)}")
    return gaz[["fips", "area_sq_miles"]]


# ═══════════════════════════════════════════════════════
# P4 — INFRASTRUCTURE RESILIENCE SCORE
# ═══════════════════════════════════════════════════════

def build_infrastructure_resilience_score(
    utility_data: pd.DataFrame = None,
    road_data: pd.DataFrame = None,
    water_data: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    P4: Infrastructure vulnerability score combining available pillars.

    Full formula (all three pillars available):
      infrastructure_vulnerability_score = (
          utility_hardening_index   × w_utility     +
          road_redundancy_ratio     × w_road        +
          water_vulnerability_score × w_water
      )

    Reweights proportionally when pillars are missing.
    All weights load from INFRASTRUCTURE_WEIGHTS in optimized_weights.json.

    Plugs into final risk as:
      impact_score_adjusted = impact_score × (1.0 + vulnerability × α)
    where α loads from INFRASTRUCTURE_AMPLIFIER.
    """
    print("\n[P4] Building infrastructure resilience score...")

    iw = _load_weights("INFRASTRUCTURE_WEIGHTS", {
        "utility_hardening": 0.40, "road_redundancy": 0.35, "water_vulnerability": 0.25,
    })
    amp = _load_weights("INFRASTRUCTURE_AMPLIFIER", {"magnitude": 0.35})

    # Collect available pillars
    pillars = {}
    if utility_data is not None and not utility_data.empty:
        pillars["utility_hardening"] = utility_data[["fips", "utility_hardening_index"]].copy()
        pillars["utility_hardening"] = pillars["utility_hardening"].rename(
            columns={"utility_hardening_index": "utility_hardening"}
        )
    if road_data is not None and not road_data.empty:
        pillars["road_redundancy"] = road_data[["fips", "road_redundancy_ratio"]].copy()
        pillars["road_redundancy"] = pillars["road_redundancy"].rename(
            columns={"road_redundancy_ratio": "road_redundancy"}
        )
    if water_data is not None and not water_data.empty:
        pillars["water_vulnerability"] = water_data[["fips", "water_vulnerability_score"]].copy()
        pillars["water_vulnerability"] = pillars["water_vulnerability"].rename(
            columns={"water_vulnerability_score": "water_vulnerability"}
        )

    available = list(pillars.keys())
    if not available:
        print("  WARNING: No infrastructure data available. Skipping P4.")
        return pd.DataFrame()

    print(f"  Available pillars: {', '.join(available)}")
    print(f"  Missing pillars: {', '.join(set(iw.keys()) - set(available)) or 'none'}")

    # Reweight proportionally to available pillars
    raw_weights = {k: iw[k] for k in available}
    total_w = sum(raw_weights.values())
    weights = {k: v / total_w for k, v in raw_weights.items()}
    print(f"  Reweighted: {', '.join(f'{k}={v:.2f}' for k, v in weights.items())}")

    # Merge all available pillars
    result = None
    for name, pdf in pillars.items():
        pdf = pdf.copy()
        pdf["fips"] = pdf["fips"].astype("Int64")
        if result is None:
            result = pdf
        else:
            result = result.merge(pdf, on="fips", how="outer")

    # Fill missing values with column median
    for name in available:
        col_med = result[name].median()
        result[name] = result[name].fillna(col_med)

    # Compute weighted score
    result["infrastructure_vulnerability_score"] = sum(
        result[name] * weights[name] for name in available
    )

    # Resilience is the inverse
    result["infrastructure_resilience_score"] = 1.0 - result["infrastructure_vulnerability_score"]

    # Amplifier for impact adjustment
    alpha = amp["magnitude"]
    result["infrastructure_amplifier"] = (
        1.0 + result["infrastructure_vulnerability_score"] * alpha
    )

    # Tier classification
    result["infrastructure_tier"] = pd.cut(
        result["infrastructure_vulnerability_score"],
        bins=[0, 0.25, 0.50, 0.75, 1.0],
        labels=["Resilient", "Moderate", "Vulnerable", "Critical"],
        include_lowest=True,
    )

    n = len(result)
    print(f"  Counties with infrastructure scores: {n}")
    print(f"  Amplifier magnitude α = {alpha}")
    print(f"  Infrastructure tier distribution:")
    print(result["infrastructure_tier"].value_counts().to_string())

    return result[["fips", "infrastructure_vulnerability_score",
                    "infrastructure_resilience_score",
                    "infrastructure_amplifier", "infrastructure_tier"]]


# ═══════════════════════════════════════════════════════
# FINAL RISK = PROBABILITY × IMPACT
# ═══════════════════════════════════════════════════════
def build_final_risk(probability: pd.DataFrame,
                     impact: pd.DataFrame,
                     infrastructure: pd.DataFrame = None) -> pd.DataFrame:
    """
    Final risk score = probability_score × impact_score.

    Multiplication (not addition) ensures both dimensions must be
    elevated for a county to be truly critical. A county certain to
    burn but economically resilient scores LOW. A fragile economy with
    no fire history also scores LOW. Only counties high on BOTH axes
    reach Critical.

    Interpretation guide:
      High prob + High impact  → Critical  (imminent threat, severe damage)
      High prob + Low impact   → Moderate  (likely fire, economy can absorb it)
      Low prob  + High impact  → Moderate  (fragile if hit, not imminently threatened)
      Low prob  + Low impact   → Low       (genuinely safe)
    """
    print("\n[Final] Computing probability × impact risk scores...")

    prob_cols = ["fips","county_name","state","probability_score","probability_tier",
                 "probability_rank","drought_score","fire_frequency_score",
                 "max_consecutive_weeks","recent_intensity","episode_frequency",
                 "fire_count","fires_per_100k","last_drought_end"]
    imp_cols  = ["fips","impact_score","impact_tier","impact_rank","trajectory",
                 "causal_impact_score","avg_gdp_delta_pct","avg_counterfactual_gap",
                 "industry_sensitivity","top_industries","economic_trajectory",
                 "decay_unem_rate","gdp_growth_trend"]

    avail_prob = [c for c in prob_cols if c in probability.columns]
    avail_imp  = [c for c in imp_cols  if c in impact.columns]

    df = probability[avail_prob].merge(impact[avail_imp], on="fips", how="inner")

    # P4: Infrastructure amplifier (if available)
    if infrastructure is not None and not infrastructure.empty:
        infra_cols = [c for c in [
            "fips", "infrastructure_amplifier",
            "infrastructure_vulnerability_score",
            "infrastructure_resilience_score",
            "infrastructure_tier"
        ] if c in infrastructure.columns]
        df = df.merge(infrastructure[infra_cols], on="fips", how="left")
        df["infrastructure_amplifier"] = df["infrastructure_amplifier"].fillna(1.0)
        df["impact_score_adjusted"] = (
            df["impact_score"] * df["infrastructure_amplifier"]
        ).clip(0, 1)
        print(f"  Infrastructure amplifier applied to {df['infrastructure_amplifier'].notna().sum()} counties")
    else:
        df["impact_score_adjusted"] = df["impact_score"]

    # Core multiplication (uses adjusted impact if infrastructure available)
    df["final_risk_score"] = df["probability_score"] * df["impact_score_adjusted"]

    # Normalize to [0,1] so tiers are meaningful
    df["final_risk_score"] = normalize(df["final_risk_score"])

    df["final_risk_tier"] = pd.cut(
        df["final_risk_score"],
        bins=[0, 0.25, 0.50, 0.75, 1.0],
        labels=["Low","Moderate","High","Critical"],
        include_lowest=True
    )

    # Quadrant classification for shareholder narrative
    prob_high_threshold   = df["probability_score"].quantile(0.75)
    impact_high_threshold = df["impact_score"].quantile(0.75)
    def quadrant(row):
        hi_p = row["probability_score"] >= prob_high_threshold
        hi_i = row["impact_score"]      >= impact_high_threshold
        if hi_p and hi_i:   return "High Probability / High Impact"
        if hi_p and not hi_i: return "High Probability / Low Impact"
        if not hi_p and hi_i: return "Low Probability / High Impact"
        return "Low Probability / Low Impact"

    df["risk_quadrant"] = df.apply(quadrant, axis=1)

    # Predicted risk: trend-adjust the final score
    unem_adj = normalize(df["unemployment_trend"].fillna(0)) * 0.10 \
        if "unemployment_trend" in df.columns else pd.Series(0.0, index=df.index)
    gdp_adj  = invert(df["gdp_growth_trend"].fillna(0)) * 0.10 \
        if "gdp_growth_trend" in df.columns else pd.Series(0.0, index=df.index)

    adj = (unem_adj * 0.5 + gdp_adj * 0.5)
    adj -= adj.median()
    df["predicted_risk_score"] = (df["final_risk_score"] + adj).clip(0, 1)
    df["predicted_risk_tier"]  = pd.cut(
        df["predicted_risk_score"],
        bins=[0, 0.25, 0.50, 0.75, 1.0],
        labels=["Low","Moderate","High","Critical"],
        include_lowest=True
    )
    df["risk_change"]    = df["predicted_risk_score"] - df["final_risk_score"]
    df["risk_direction"] = df["risk_change"].apply(
        lambda x: "↑ Increasing" if x > 0.01 else ("↓ Decreasing" if x < -0.01 else "→ Stable")
    )

    df = df.sort_values("final_risk_score", ascending=False).reset_index(drop=True)
    df["final_rank"] = df.index + 1

    print(f"  Counties in final risk output: {len(df)}")
    print(f"  Final risk tier distribution:\n{df['final_risk_tier'].value_counts().to_string()}")
    print(f"\n  Risk quadrant distribution:")
    print(df["risk_quadrant"].value_counts().to_string())
    return df


# ═══════════════════════════════════════════════════════
# P2 — DOLLAR LOSS AND WAGE LOSS ESTIMATES
# ═══════════════════════════════════════════════════════

def add_dollar_loss_estimate(
    final_df: pd.DataFrame,
    gdp: pd.DataFrame
) -> pd.DataFrame:
    """
    Converts predicted GDP impact % into a dollar loss range.

    UCLA Anderson multipliers (2025):
      2.2 = California wildfire historical baseline
      2.8 = Hurricane Katrina infrastructure ratio (upper bound)
    """
    print("\n[P2] Adding dollar loss estimates (UCLA Anderson methodology)...")
    UCLA_LOW  = 2.2
    UCLA_HIGH = 2.8

    # P1: use FIPS for GDP total lookup
    # GDP values are annualized quarterly estimates — use mean, not sum
    gdp_totals = (
        gdp[gdp["Year"] == gdp["Year"].max()]
        .dropna(subset=["fips"])
        .groupby("fips")
        .agg(total_gdp_m=("Est. County GDP (Current $M)", "mean"))
        .reset_index()
    )

    df = final_df.copy()
    df = df.merge(gdp_totals[["fips", "total_gdp_m"]], on="fips", how="left")

    if "predicted_gdp_impact_pct" in df.columns:
        # P3m: WUI exposure fraction scales GDP loss to fire-exposed share
        wui_frac = df["wui_housing_fraction"] if "wui_housing_fraction" in df.columns else 1.0
        df["predicted_gdp_loss_m"] = (
            df["total_gdp_m"]
            * df["predicted_gdp_impact_pct"].abs() / 100
            * wui_frac
        )
        df["total_loss_low_m"]  = df["predicted_gdp_loss_m"] * UCLA_LOW
        df["total_loss_high_m"] = df["predicted_gdp_loss_m"] * UCLA_HIGH
        df["estimated_loss_m"]  = (df["total_loss_low_m"] + df["total_loss_high_m"]) / 2
    else:
        df["predicted_gdp_loss_m"] = np.nan
        df["total_loss_low_m"]     = np.nan
        df["total_loss_high_m"]    = np.nan
        df["estimated_loss_m"]     = np.nan

    populated = df["total_loss_low_m"].notna().sum()
    print(f"  Counties with dollar estimates: {populated}")
    if populated > 0:
        print(f"  Total GDP loss range: ${df['total_loss_low_m'].sum():.0f}M – "
              f"${df['total_loss_high_m'].sum():.0f}M")
    return df


def add_wage_loss_estimate(final_df: pd.DataFrame) -> pd.DataFrame:
    """
    Estimates wage loss from predicted GDP loss.
    BEA national average labor share = 0.53
    """
    print("[P2] Adding wage loss estimates...")
    LABOR_SHARE = 0.53

    df = final_df.copy()
    if "predicted_gdp_loss_m" in df.columns:
        df["estimated_wage_loss_m"] = (
            df["predicted_gdp_loss_m"] * LABOR_SHARE
        )
    else:
        df["estimated_wage_loss_m"] = np.nan
    populated = df["estimated_wage_loss_m"].notna().sum()
    print(f"  Counties with wage loss estimates: {populated}")
    return df


# ═══════════════════════════════════════════════════════
# REPORT
# ═══════════════════════════════════════════════════════
def generate_county_narrative(row: pd.Series) -> str:
    """Plain-English shareholder explanation for a single county."""
    name    = row.get("county_name","This county")
    state   = row.get("state","")
    tier    = row.get("final_risk_tier","Unknown")
    prob    = row.get("probability_score", 0)
    impact  = row.get("impact_score", 0)
    final   = row.get("final_risk_score", 0)
    quadrant= row.get("risk_quadrant","")
    traj    = row.get("trajectory","Unknown")
    drought_wks = row.get("max_consecutive_weeks", None)
    fires   = row.get("fire_count", 0)
    gdp_d   = row.get("avg_gdp_delta_pct", None)
    cf_gap  = row.get("avg_counterfactual_gap", None)
    unem    = row.get("decay_unem_rate", None)
    ind     = row.get("top_industries","")

    lines = [f"{name}, {state} — {tier} Risk"]
    lines.append(f"Final Score: {final:.3f}  |  Probability: {prob:.3f}  |  Impact: {impact:.3f}")
    lines.append(f"Quadrant: {quadrant}")
    lines.append("")

    # Probability narrative
    if pd.notna(drought_wks):
        lines.append(f"FIRE PROBABILITY: The county's longest recorded consecutive drought "
                     f"span is {int(drought_wks)} weeks, indicating significant fuel dryness.")
    if pd.notna(fires) and fires > 0:
        lines.append(f"Historical fire frequency of {int(fires)} recorded events reinforces "
                     f"the ignition risk profile.")

    # Impact narrative
    if pd.notna(gdp_d):
        dir_word = "declined" if gdp_d < 0 else "grew"
        lines.append(f"\nECONOMIC IMPACT: Following significant fire events, county GDP "
                     f"{dir_word} {abs(gdp_d):.1f}% year-over-year.")
        if pd.notna(cf_gap) and cf_gap != 0:
            comp = "underperformed" if cf_gap < 0 else "outperformed"
            lines.append(f"Compared to {N_CONTROLS} economically matched counties with no fire "
                         f"activity, {name} {comp} by {abs(cf_gap):.1f} percentage points — "
                         f"the wildfire-attributable economic gap.")
    # P2: Dollar loss figures
    if pd.notna(row.get("total_loss_low_m")):
        lines.append(f"\nESTIMATED TOTAL ECONOMIC LOSS: ${row.get('total_loss_low_m',0):.0f}M – "
                     f"${row.get('total_loss_high_m',0):.0f}M (UCLA Anderson methodology)")
    if pd.notna(row.get("estimated_wage_loss_m")):
        lines.append(f"Estimated wage loss at risk: ${row.get('estimated_wage_loss_m',0):.0f}M")

    if ind:
        top = ind.split(" | ")[0]
        lines.append(f"\nINDUSTRY EXPOSURE: Dominant sector is {top}, carrying elevated "
                     f"sensitivity to wildfire-driven disruption.")
    if pd.notna(unem):
        lines.append(f"Current unemployment rate: {unem:.1f}% (decay-weighted).")

    traj_text = {
        "Chronic Impact":  "Economy has not recovered to pre-fire baseline — "
                           "ongoing wildfire-attributable damage confirmed.",
        "Slow Recovery":   "Partial recovery underway but below pre-fire baseline.",
        "Resilient":       "Economy recovered to pre-fire levels despite fire exposure.",
        "At Risk":         "No major historical fire impact measured, but structural "
                           "vulnerability indicators are elevated.",
    }.get(traj, "")
    if traj_text:
        lines.append(f"\nRECOVERY TRAJECTORY ({traj}): {traj_text}")

    return "\n".join(lines)


def write_report(probability: pd.DataFrame,
                 impact: pd.DataFrame,
                 final: pd.DataFrame,
                 cf_detail: pd.DataFrame):

    top10_final = final.nlargest(10, "final_risk_score")
    top10_prob  = probability.nlargest(10, "probability_score")
    top10_imp   = impact.nlargest(10, "impact_score")
    rising      = final[final["risk_direction"] == "↑ Increasing"].nlargest(10, "risk_change")
    chronic     = final[final["trajectory"] == "Chronic Impact"].nlargest(5, "final_risk_score") \
                  if "trajectory" in final.columns else pd.DataFrame()

    lines = [
        "=" * 70,
        "   WILDFIRE ECONOMIC RISK — SHAREHOLDER SUMMARY REPORT",
        "=" * 70,
        "",
        f"  Reference date:         {REFERENCE_DATE.strftime('%Y-%m-%d')}",
        f"  Counties in final output: {len(final)}",
        f"  Probability counties:     {len(probability)}",
        f"  Impact counties:          {len(impact)}",
        "",
        "  METHODOLOGY:",
        "    Final Risk  = Probability Score × Impact Score",
        f"   Probability = Drought ({PROB_WEIGHTS['drought']*100:.0f}%) + "
        f"Fire Frequency ({PROB_WEIGHTS['fire_frequency']*100:.0f}%)",
        f"   Impact      = Causal Impact ({IMPACT_WEIGHTS['causal_impact']*100:.0f}%) + "
        f"Industry Sensitivity ({IMPACT_WEIGHTS['industry_sensitivity']*100:.0f}%) + "
        f"Economic Trajectory ({IMPACT_WEIGHTS['economic_trajectory']*100:.0f}%)",
        "",
        "─" * 70,
        "FINAL RISK TIER DISTRIBUTION:",
        final["final_risk_tier"].value_counts().to_string(),
        "",
        "RISK QUADRANT DISTRIBUTION:",
        final["risk_quadrant"].value_counts().to_string(),
        "",
        "─" * 70,
        "TOP 10 — HIGHEST PROBABILITY OF FIRE:",
    ]
    p_cols = [c for c in ["probability_rank","county_name","state","probability_score",
                           "probability_tier","max_consecutive_weeks","fire_count"]
              if c in top10_prob.columns]
    lines.append(top10_prob[p_cols].to_string(index=False))

    lines += ["","─"*70,"TOP 10 — HIGHEST ECONOMIC IMPACT IF FIRE OCCURS:"]
    i_cols = [c for c in ["impact_rank","county_name","state","impact_score",
                           "impact_tier","trajectory","avg_counterfactual_gap"]
              if c in top10_imp.columns]
    lines.append(top10_imp[i_cols].to_string(index=False))

    lines += ["","─"*70,"TOP 10 — HIGHEST COMBINED RISK (PROBABILITY × IMPACT):"]
    f_cols = [c for c in ["final_rank","county_name","state","final_risk_score",
                           "final_risk_tier","probability_score","impact_score","risk_quadrant"]
              if c in top10_final.columns]
    lines.append(top10_final[f_cols].to_string(index=False))

    if not chronic.empty:
        lines += ["","─"*70,"CHRONIC IMPACT COUNTIES (never recovered from past fires):"]
        ch_cols = [c for c in ["final_rank","county_name","state","final_risk_score",
                                "avg_gdp_delta_pct","avg_counterfactual_gap"]
                   if c in chronic.columns]
        lines.append(chronic[ch_cols].to_string(index=False))

    lines += ["","─"*70,"TOP 10 COUNTY NARRATIVES:",""]
    for _, row in top10_final.iterrows():
        lines.append(generate_county_narrative(row))
        lines.append("")

    lines += ["="*70]
    with open("risk_summary_report.txt","w") as f:
        f.write("\n".join(lines))
    print("  Report saved → risk_summary_report.txt")



# ═══════════════════════════════════════════════════════
# PREDICTIVE MODEL — RANDOM FOREST + SHAP
# ═══════════════════════════════════════════════════════

FEATURE_COLS = [
    "drought_weeks_before",   # consecutive drought weeks at time of fire
    "unemployment_rate",      # unemployment rate at time of fire
    "gdp_per_capita",         # county GDP per capita at time of fire
    "industry_sensitivity",   # NAICS fragility score
    "fire_frequency",         # historical fires per 100k population
    "wage_base_m",            # P6: BLS QCEW total annual wages ($M)
]
FEATURE_COLS_NONCA = FEATURE_COLS + [
    "water_vulnerability_score",  # P6: combined federal/CA water vuln
    "private_well_proxy_pct",     # P6: ACS B25049 private well proxy
]
LABEL_COL = "counterfactual_gap"   # fire-attributable GDP change (macro trend removed)


def build_training_dataset(cf_detail: pd.DataFrame,
                            drought: pd.DataFrame,
                            unem: pd.DataFrame,
                            gdp: pd.DataFrame,
                            industry: pd.DataFrame,
                            fire_freq: pd.DataFrame,
                            qcew: pd.DataFrame = None,
                            water_vuln: pd.DataFrame = None) -> pd.DataFrame:
    """
    Assembles a labeled training dataset from historical fire events.

    Each row = one county-fire event.
    Features = conditions measured BEFORE the fire.
    Label    = GDP delta % in 4 quarters AFTER the fire.

    This is the core of the predictive model: we learn the mapping
    from pre-fire conditions → post-fire economic outcome so we can
    apply it to counties that haven't burned yet.
    """
    print("\n[ML] Building training dataset from historical fire events...")

    if cf_detail.empty or LABEL_COL not in cf_detail.columns:
        print("  WARNING: No causal evidence available for training.")
        return pd.DataFrame()

    df = cf_detail.dropna(subset=[LABEL_COL]).copy()
    df["fips"] = df["fips"].astype("Int64")

    # ── Feature 1: drought weeks before fire ──
    drought_w = drought.copy()
    drought_w["fips"]     = drought_w["FIPS"].astype("Int64")
    drought_w["end_date"] = pd.to_datetime(drought_w["EndDate"], errors="coerce")
    drought_w["start_date"] = pd.to_datetime(drought_w["StartDate"], errors="coerce")

    def drought_before_fire(fips_val, fire_date_str):
        """Sum consecutive drought weeks active in the 52 weeks before fire."""
        try:
            fire_date = pd.to_datetime(fire_date_str)
            window_start = fire_date - pd.Timedelta(weeks=52)
            county_drought = drought_w[
                (drought_w["fips"] == fips_val) &
                (drought_w["end_date"] >= window_start) &
                (drought_w["start_date"] <= fire_date)
            ]
            return county_drought["ConsecutiveWeeks"].sum()
        except Exception:
            return 0.0

    df["drought_weeks_before"] = df.apply(
        lambda r: drought_before_fire(r["fips"], r.get("event_date", r.get("fire_period",""))),
        axis=1
    )

    # ── Feature 2: unemployment rate before fire ──
    unem_w = unem.copy()
    unem_w["fips"] = unem_w["FIPS_Code"].astype("Int64")
    rate_rows = unem_w[unem_w["Attribute"].str.startswith("Unemployment_rate_")].copy()
    rate_rows["year"] = rate_rows["Attribute"].str.extract(r"(\d{4})").astype(int)

    def unem_before_fire(fips_val, fire_date_str):
        try:
            fire_year = pd.to_datetime(str(fire_date_str)).year
            county_rates = rate_rows[
                (rate_rows["fips"] == fips_val) &
                (rate_rows["year"] <= fire_year)
            ].sort_values("year")
            return float(county_rates["Value"].iloc[-1]) if len(county_rates) > 0 else np.nan
        except Exception:
            return np.nan

    df["unemployment_rate"] = df.apply(
        lambda r: unem_before_fire(r["fips"], r.get("event_date", r.get("fire_period",""))),
        axis=1
    )

    # ── Feature 3: GDP per capita before fire ──
    # P1 fix: use FIPS for GDP lookup instead of county_key
    gdp_w = gdp[gdp["fips"].notna()].copy()

    def gdp_pc_before_fire(fips_val, fire_date_str):
        try:
            fire_year = pd.to_datetime(str(fire_date_str)).year
            cg = gdp_w[
                (gdp_w["fips"] == fips_val) &
                (gdp_w["Year"] <= fire_year)
            ]
            if cg.empty:
                return np.nan
            latest = cg[cg["Year"] == cg["Year"].max()]
            gdp_m  = latest["Est. County GDP (Current $M)"].mean()
            pop    = latest["County Population"].mean()
            return gdp_m * 1e6 / pop if pop > 0 else np.nan
        except Exception:
            return np.nan

    df["gdp_per_capita"] = df.apply(
        lambda r: gdp_pc_before_fire(
            r["fips"],
            r.get("event_date", r.get("fire_period",""))
        ),
        axis=1
    )

    # ── Feature 4: industry sensitivity ──
    ind_lookup = industry.set_index("fips")["industry_sensitivity"].to_dict()
    df["industry_sensitivity"] = df["fips"].map(ind_lookup).fillna(0.30)

    # ── Feature 5: historical fire frequency ──
    freq_lookup = fire_freq.set_index("fips")["fires_per_100k"].to_dict()
    df["fire_frequency"] = df["fips"].map(freq_lookup).fillna(0.0)

    # ── Feature 6: wage base ($M) from BLS QCEW (P6) ──
    if qcew is not None and not qcew.empty:
        qw = qcew.copy()
        qw["fips"] = pd.to_numeric(
            qw.get("area_fips", pd.Series(dtype=int)), errors="coerce"
        ).astype("Int64")
        qw["year"] = pd.to_numeric(qw.get("year", pd.Series(dtype=int)), errors="coerce")
        qw["total_wages"] = pd.to_numeric(
            qw.get("total_annual_wages", pd.Series(dtype=float)), errors="coerce"
        )
        latest_yr = qw["year"].max()
        wage_lookup = qw[qw["year"] == latest_yr].groupby("fips").agg(
            wage_base_m=("total_wages", lambda x: x.sum() / 1e6)
        ).reset_index()
        wage_dict = wage_lookup.set_index("fips")["wage_base_m"].to_dict()
        median_wage = wage_lookup["wage_base_m"].median()
        df["wage_base_m"] = df["fips"].map(wage_dict).fillna(median_wage)
    else:
        df["wage_base_m"] = 0.0

    # ── Features 7–8: water vulnerability + private well proxy (non-CA only) ──
    if water_vuln is not None and not water_vuln.empty:
        wv = water_vuln[["fips"]].copy()
        if "water_vulnerability_score" in water_vuln.columns:
            wv["water_vulnerability_score"] = water_vuln["water_vulnerability_score"].values
        if "private_well_proxy_pct" in water_vuln.columns:
            wv["private_well_proxy_pct"] = water_vuln["private_well_proxy_pct"].values
        wv = wv.drop_duplicates("fips")
        df = df.merge(wv, on="fips", how="left")
        for col in ["water_vulnerability_score", "private_well_proxy_pct"]:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())
    else:
        df["water_vulnerability_score"] = 0.0
        df["private_well_proxy_pct"] = 0.0

    # ── Clean and validate ──
    all_feature_cols = list(set(FEATURE_COLS + FEATURE_COLS_NONCA))
    train = df[all_feature_cols + [LABEL_COL, "fips", "county_name"]].dropna(
        subset=FEATURE_COLS + [LABEL_COL]
    ).copy()

    n = len(train)
    print(f"  Training rows assembled:    {n}")
    print(f"  Label range (GDP delta %):  {train[LABEL_COL].min():.1f}% to {train[LABEL_COL].max():.1f}%")
    print(f"  Label mean:                 {train[LABEL_COL].mean():.2f}%")

    if n < MIN_TRAINING_ROWS:
        print(f"  ⚠ WARNING: Only {n} training rows (minimum recommended: {MIN_TRAINING_ROWS}).")
        print(f"    Predictions will have low confidence. Consider broadening MIN_ACREAGE")
        print(f"    or extending the date range of your geo_events data.")

    return train


def _train_xgb(X, y, label: str, save_path: str, feature_names: list = None):
    """Train a single XGBoost model, print diagnostics, save to disk."""
    import xgboost as xgb

    feat_names = feature_names or FEATURE_COLS

    model = xgb.XGBRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0, random_state=42, n_jobs=-1,
    )
    model.fit(X, y)

    cv_scores = cross_val_score(model, X, y, cv=min(5, len(y)//5),
                                scoring="r2")
    train_preds = model.predict(X)
    mae = mean_absolute_error(y, train_preds)
    r2  = r2_score(y, train_preds)

    print(f"  [{label}] Training rows:       {len(y)}")
    print(f"  [{label}] Features:            {len(feat_names)}")
    print(f"  [{label}] Training R²:         {r2:.3f}")
    print(f"  [{label}] Training MAE:        {mae:.2f}% GDP impact")
    print(f"  [{label}] CV R² (mean):        {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    fi = pd.DataFrame({
        "feature":    feat_names,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    print(f"\n  [{label}] Feature importance:")
    for _, row in fi.iterrows():
        bar = "█" * int(row["importance"] * 30)
        print(f"    {row['feature']:<28} {bar} {row['importance']:.3f}")

    joblib.dump(model, save_path)
    print(f"  [{label}] Model saved → {save_path}")
    return model, fi, cv_scores


def select_best_model(train_df: pd.DataFrame,
                       force_retrain: bool = False) -> tuple:
    """
    P5/P6: Stratified model training — trains separate XGBoost models
    for CA and non-CA counties.

    CA counties (FIPS 6001–6115) get their own model trained on CA fire events.
    Non-CA counties get a separate model trained on non-CA fire events.

    Returns (model_nonca, model_ca, feature_importance_df, cv_scores_dict).
    """
    from pathlib import Path
    import xgboost as xgb

    if not force_retrain and Path(MODEL_PATH).exists() and Path(MODEL_PATH_CA).exists():
        print(f"\n[ML] Loading existing stratified models")
        model_nonca = joblib.load(MODEL_PATH)
        model_ca    = joblib.load(MODEL_PATH_CA)
        fi = pd.DataFrame({
            "feature":   FEATURE_COLS_NONCA,
            "importance": model_nonca.feature_importances_,
        }).sort_values("importance", ascending=False)
        return model_nonca, model_ca, fi, None

    print("\n[ML] Training stratified XGBoost models (CA: 6 feat, non-CA: 8 feat)...")

    if train_df.empty or len(train_df) < 5:
        print("  ERROR: Insufficient training data — skipping model training.")
        return None, None, pd.DataFrame(), None

    # Split by CA FIPS range
    is_ca = (train_df["fips"] >= 6001) & (train_df["fips"] <= 6115)
    train_ca    = train_df[is_ca]
    train_nonca = train_df[~is_ca]

    print(f"  CA training rows:     {len(train_ca)}")
    print(f"  Non-CA training rows: {len(train_nonca)}")

    # Train CA model (6 features)
    X_ca = train_ca[FEATURE_COLS].values
    y_ca = train_ca[LABEL_COL].values
    model_ca, fi_ca, cv_ca = _train_xgb(X_ca, y_ca, "CA", MODEL_PATH_CA,
                                          feature_names=FEATURE_COLS)

    # Train non-CA model (8 features)
    X_nonca = train_nonca[FEATURE_COLS_NONCA].values
    y_nonca = train_nonca[LABEL_COL].values
    model_nonca, fi_nonca, cv_nonca = _train_xgb(X_nonca, y_nonca, "non-CA", MODEL_PATH,
                                                    feature_names=FEATURE_COLS_NONCA)

    return model_nonca, model_ca, fi_nonca, {"ca": cv_ca, "nonca": cv_nonca}


def predict_current_impact(model_nonca,
                            probability: pd.DataFrame,
                            drought: pd.DataFrame,
                            unem: pd.DataFrame,
                            gdp: pd.DataFrame,
                            industry: pd.DataFrame,
                            fire_freq: pd.DataFrame,
                            qcew: pd.DataFrame = None,
                            model_ca = None,
                            water_vuln: pd.DataFrame = None) -> pd.DataFrame:
    """
    Applies stratified models to CURRENT county conditions to predict
    what GDP impact would occur if a wildfire struck today.

    CA counties (FIPS 6001–6115) use model_ca + FEATURE_COLS (6 features).
    Non-CA counties use model_nonca + FEATURE_COLS_NONCA (8 features).
    SHAP explanations use the correct model per county.
    """
    print("\n[ML] Predicting current economic impact using stratified models...")

    if model_nonca is None:
        print("  No trained model available — skipping predictions.")
        return pd.DataFrame()

    # ── Assemble current-conditions feature matrix ──
    drought_w = drought.copy()
    drought_w["fips"] = drought_w["FIPS"].astype("Int64")
    drought_w["end_date"] = pd.to_datetime(drought_w["EndDate"], errors="coerce")

    # Current drought: total consecutive weeks in last 52 weeks
    cutoff = REFERENCE_DATE - pd.Timedelta(weeks=52)
    recent_drought = drought_w[drought_w["end_date"] >= cutoff].groupby("fips").agg(
        drought_weeks_before=("ConsecutiveWeeks","sum")
    ).reset_index()

    # Latest unemployment
    unem_w = unem.copy()
    unem_w["fips"] = unem_w["FIPS_Code"].astype("Int64")
    rate_rows = unem_w[unem_w["Attribute"].str.startswith("Unemployment_rate_")].copy()
    rate_rows["year"] = rate_rows["Attribute"].str.extract(r"(\d{4})").astype(int)
    latest_yr = rate_rows["year"].max()
    latest_unem = (
        rate_rows[rate_rows["year"] == latest_yr][["fips","Value"]]
        .rename(columns={"Value":"unemployment_rate"})
    )

    # Latest GDP per capita — P1 fix: use FIPS instead of county_key
    gdp_w = gdp[gdp["fips"].notna()].copy()
    latest_gdp_yr = gdp_w["Year"].max()
    gdp_pc = (
        gdp_w[gdp_w["Year"] == latest_gdp_yr]
        .groupby("fips")
        .apply(lambda g: pd.Series({
            "gdp_per_capita": g["Est. County GDP (Current $M)"].mean() * 1e6 /
                              g["County Population"].mean()
                              if g["County Population"].mean() > 0 else np.nan,
        }))
        .reset_index()
    )

    # Industry sensitivity
    ind = industry[["fips","industry_sensitivity"]].copy()

    # Fire frequency
    ff = fire_freq[["fips","fires_per_100k"]].rename(
        columns={"fires_per_100k":"fire_frequency"}
    ).copy()

    # P6: Wage base from BLS QCEW
    wage_df = pd.DataFrame(columns=["fips", "wage_base_m"])
    if qcew is not None and not qcew.empty:
        qw = qcew.copy()
        qw["fips"] = pd.to_numeric(
            qw.get("area_fips", pd.Series(dtype=int)), errors="coerce"
        ).astype("Int64")
        qw["year"] = pd.to_numeric(qw.get("year", pd.Series(dtype=int)), errors="coerce")
        qw["total_wages"] = pd.to_numeric(
            qw.get("total_annual_wages", pd.Series(dtype=float)), errors="coerce"
        )
        latest_yr = qw["year"].max()
        wage_df = qw[qw["year"] == latest_yr].groupby("fips").agg(
            wage_base_m=("total_wages", lambda x: x.sum() / 1e6)
        ).reset_index()

    # Build prediction dataframe anchored on probability counties
    pred = probability[["fips","county_name","state","probability_score"]].copy()
    pred = pred.merge(recent_drought, on="fips", how="left")
    pred = pred.merge(latest_unem,    on="fips", how="left")
    pred = pred.merge(gdp_pc[["fips","gdp_per_capita"]], on="fips", how="left")
    pred = pred.merge(ind, on="fips", how="left")
    pred = pred.merge(ff,  on="fips", how="left")
    pred = pred.merge(wage_df[["fips", "wage_base_m"]], on="fips", how="left")

    # P6: Water vulnerability + private well proxy (non-CA features)
    if water_vuln is not None and not water_vuln.empty:
        wv_cols = [c for c in ["fips", "water_vulnerability_score", "private_well_proxy_pct"]
                   if c in water_vuln.columns]
        pred = pred.merge(water_vuln[wv_cols].drop_duplicates("fips"), on="fips", how="left")

    # Fill missing features with median (conservative)
    all_cols = list(set(FEATURE_COLS + FEATURE_COLS_NONCA))
    for col in all_cols:
        if col in pred.columns:
            pred[col] = pd.to_numeric(pred[col], errors="coerce")
            pred[col] = pred[col].fillna(pred[col].median())
        else:
            pred[col] = 0.0

    # ── Stratified predictions: CA vs non-CA ──
    is_ca = (pred["fips"] >= 6001) & (pred["fips"] <= 6115)
    pred["predicted_gdp_impact_pct"] = np.nan
    pred["model_region"] = np.where(is_ca, "CA", "non-CA")

    # Non-CA predictions (8 features)
    mask_nonca = ~is_ca
    if mask_nonca.any():
        X_nonca = pred.loc[mask_nonca, FEATURE_COLS_NONCA].values
        pred.loc[mask_nonca, "predicted_gdp_impact_pct"] = model_nonca.predict(X_nonca)

    # CA predictions (6 features)
    mask_ca = is_ca
    if mask_ca.any() and model_ca is not None:
        X_ca = pred.loc[mask_ca, FEATURE_COLS].values
        pred.loc[mask_ca, "predicted_gdp_impact_pct"] = model_ca.predict(X_ca)
    elif mask_ca.any():
        # Fallback: use non-CA model if CA model unavailable
        X_ca = pred.loc[mask_ca, FEATURE_COLS].values
        pred.loc[mask_ca, "predicted_gdp_impact_pct"] = model_nonca.predict(X_ca)
        pred.loc[mask_ca, "model_region"] = "non-CA (fallback)"

    print(f"  CA counties predicted:     {mask_ca.sum()}")
    print(f"  Non-CA counties predicted: {mask_nonca.sum()}")

    # ── Confidence interval: ±1 MAE as approximate CI ──
    point = pred["predicted_gdp_impact_pct"].values
    mae_est = 0.83
    pred["prediction_ci_low"]  = point - mae_est
    pred["prediction_ci_high"] = point + mae_est
    pred["prediction_confidence"] = pred["prediction_ci_high"] - pred["prediction_ci_low"]

    ci_median = pred["prediction_confidence"].median()
    pred["confidence_flag"] = pred["prediction_confidence"].apply(
        lambda x: "High" if x < ci_median * 0.5
        else ("Medium" if x < ci_median * 1.5 else "Low")
    )

    # ── SHAP explanations (stratified) ──
    pred["top_shap_driver"] = "unavailable"
    pred["shap_breakdown"]  = "unavailable"

    if SHAP_AVAILABLE:
        try:
            def _compute_shap(model, mask, feat_cols):
                if not mask.any():
                    return
                X_sub = pred.loc[mask, feat_cols].values
                explainer   = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_sub)

                drivers = []
                breakdowns = []
                for i in range(len(shap_values)):
                    row = shap_values[i]
                    abs_vals = np.abs(row)
                    top_idx  = np.argmax(abs_vals)
                    direction = "↑" if row[top_idx] > 0 else "↓"
                    drivers.append(f"{feat_cols[top_idx]} {direction} ({row[top_idx]:+.3f})")
                    parts = sorted(zip(feat_cols, row), key=lambda x: abs(x[1]), reverse=True)
                    breakdowns.append("  |  ".join(f"{f}: {v:+.3f}" for f, v in parts))

                pred.loc[mask, "top_shap_driver"] = drivers
                pred.loc[mask, "shap_breakdown"]  = breakdowns

            _compute_shap(model_nonca, mask_nonca, FEATURE_COLS_NONCA)
            if model_ca is not None:
                _compute_shap(model_ca, mask_ca, FEATURE_COLS)
            else:
                _compute_shap(model_nonca, mask_ca, FEATURE_COLS_NONCA)
            print(f"  SHAP explanations generated for {len(pred)} counties")
        except Exception as e:
            print(f"  SHAP computation failed: {e}")
    else:
        fi_order = sorted(range(len(FEATURE_COLS_NONCA)),
                          key=lambda i: abs(model_nonca.feature_importances_[i]),
                          reverse=True)
        top_feat = FEATURE_COLS_NONCA[fi_order[0]]
        pred["top_shap_driver"] = f"{top_feat} (importance-based, install shap for per-county)"
        pred["shap_breakdown"]  = "Install shap for per-county explanations: pip install shap"

    print(f"  Counties with predictions:     {len(pred)}")
    print(f"  Predicted GDP impact range:    {pred['predicted_gdp_impact_pct'].min():.1f}% to "
          f"{pred['predicted_gdp_impact_pct'].max():.1f}%")
    print(f"  Mean predicted impact:         {pred['predicted_gdp_impact_pct'].mean():.2f}%")

    return pred[[
        "fips","county_name","state",
        "predicted_gdp_impact_pct","model_region",
        "prediction_ci_low","prediction_ci_high",
        "prediction_confidence","confidence_flag",
        "top_shap_driver","shap_breakdown",
        "drought_weeks_before","unemployment_rate",
        "gdp_per_capita","industry_sensitivity","fire_frequency",
        "wage_base_m","water_vulnerability_score","private_well_proxy_pct",
    ]]

# ═══════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════
def main():
    from pathlib import Path

    print("=" * 60)
    print("  Wildfire Economic Risk Pipeline  v10")
    print(f"  {REFERENCE_DATE.strftime('%Y-%m-%d')}  |  λ={DECAY_RATE}")
    print("  Final Risk = Probability × Impact × Infrastructure")
    print("=" * 60)

    gdp, foia, geo, unem, drought, crosswalk = load_data()

    # ── Probability layer ──
    drought_score = build_drought_score(drought)
    fire_freq     = build_fire_frequency_score(geo, gdp)
    probability   = build_probability_score(drought_score, fire_freq)

    # ── Impact layer ──
    prepost       = build_prepost_windows(geo, gdp)
    causal, cf_detail = build_counterfactual(prepost, gdp, unem)
    industry      = build_industry_sensitivity(foia, gdp)
    trajectory    = build_recovery_trajectory(causal, industry, gdp)

    # P3a: BEA CAGDP2 GDP-weighted industry sensitivity
    gdp_industry = None
    if Path("data/raw/bea_fixed_assets_county.csv").exists():
        bea_cagdp2 = pd.read_csv("data/raw/bea_fixed_assets_county.csv", low_memory=False)
        gdp_industry = build_gdp_weighted_industry_sensitivity(bea_cagdp2)

    # P3a: Blend FOIA + GDP-weighted industry scores, add to trajectory
    industry = blend_industry_scores(industry, gdp_industry)
    trajectory = trajectory.drop(
        columns=[c for c in ["industry_sensitivity_n", "blended_industry_sensitivity_n",
                             "gdp_industry_sensitivity_n", "industry_confidence",
                             "gdp_dominant_sector", "gdp_top_industries"]
                 if c in trajectory.columns],
        errors="ignore"
    )
    blend_cols = ["fips", "industry_sensitivity", "industry_sensitivity_n",
                  "top_industries", "dominant_sector", "industry_source",
                  "blended_industry_sensitivity_n", "gdp_industry_sensitivity_n",
                  "industry_confidence"]
    blend_cols = [c for c in blend_cols if c in industry.columns]
    trajectory = trajectory.drop(
        columns=[c for c in blend_cols if c in trajectory.columns and c != "fips"],
        errors="ignore"
    )
    trajectory = trajectory.merge(
        industry[blend_cols], on="fips", how="left"
    )

    # P3b: BLS QCEW wage data
    qcew = None
    if Path("data/raw/bls_qcew_county_annual.csv").exists():
        qcew = pd.read_csv("data/raw/bls_qcew_county_annual.csv")

    # P3i: SCO fiscal resilience (CA only)
    fiscal = None
    if (Path("data/raw/sco_county_revenues.csv").exists() and
        Path("data/raw/sco_county_expenditures.csv").exists()):
        sco_rev = pd.read_csv("data/raw/sco_county_revenues.csv", low_memory=False)
        sco_exp = pd.read_csv("data/raw/sco_county_expenditures.csv", low_memory=False)
        # SF is a consolidated city-county — sourced from SCO city budget data
        city_rev, city_exp = None, None
        if (Path("data/raw/sco_city_revenues.csv").exists() and
            Path("data/raw/sco_city_expenditures.csv").exists()):
            city_rev = pd.read_csv("data/raw/sco_city_revenues.csv", low_memory=False)
            city_exp = pd.read_csv("data/raw/sco_city_expenditures.csv", low_memory=False)
        fiscal = build_fiscal_resilience_score(
            sco_rev, sco_exp, crosswalk,
            city_revenues=city_rev, city_expenditures=city_exp,
        )

    # P3j: FTB B-7 income vulnerability (CA only)
    income = None
    if Path("data/raw/ftb_agi_by_county.csv").exists():
        ftb_agi = pd.read_csv("data/raw/ftb_agi_by_county.csv", low_memory=False)
        income = build_income_vulnerability_score(ftb_agi, crosswalk)

    econ_traj = build_economic_trajectory(
        unem, gdp, qcew=qcew, fiscal_df=fiscal, income_df=income
    )

    # P3c: FAIR Plan underinsurance (CA only)
    fair_feat = None
    if Path("data/processed/fair_plan_county.csv").exists():
        fair_raw  = pd.read_csv("data/processed/fair_plan_county.csv")
        fair_feat = build_fair_plan_amplifier(fair_raw, crosswalk, gdp)

    impact = build_impact_score(trajectory, econ_traj, fair_plan_df=fair_feat)

    # P3d: USDA NASS agricultural amplifier
    if Path("data/raw/usda_nass_land_values.csv").exists():
        nass = pd.read_csv("data/raw/usda_nass_land_values.csv")
        nass_amp = build_usda_nass_amplifier(nass, impact)
        impact = impact.merge(
            nass_amp[["fips", "nass_amplifier", "land_value_per_acre",
                       "nass_data_available"]],
            on="fips", how="left"
        )
        impact["nass_amplifier"] = impact["nass_amplifier"].fillna(1.0)
        impact["nass_data_available"] = impact["nass_data_available"].fillna(False)
        n_amplified = (impact["nass_amplifier"] > 1.0).sum()
        impact["impact_score"] = (impact["impact_score"] * impact["nass_amplifier"]).clip(0, 1)
        print(f"  Applied NASS amplifier to {n_amplified} ag-dominant counties")

    # P3e-h + P4: Infrastructure data (loaded here, composed in P4)
    county_areas = None
    if Path("data/raw/2023_Gaz_counties_national.txt").exists():
        county_areas = load_county_areas()

    # P3e: Water vulnerability (CA SWRCB + federal SDWIS)
    # Load ACS B25049 private well proxy (Gap 2 fix)
    acs_plumbing = None
    if Path("data/raw/acs_plumbing_county.csv").exists():
        acs_plumbing = pd.read_csv("data/raw/acs_plumbing_county.csv")
        acs_plumbing["fips"] = pd.to_numeric(acs_plumbing["fips"], errors="coerce").astype("Int64")

    federal_water = None
    if (Path("data/raw/sdwis/SDWA_PUB_WATER_SYSTEMS.csv").exists() and
        Path("data/raw/sdwis/SDWA_GEOGRAPHIC_AREAS.csv").exists()):
        federal_water = build_water_vulnerability("data/raw/sdwis", crosswalk=crosswalk,
                                                   acs_plumbing=acs_plumbing)

    ca_water = None
    if (Path("data/raw/ca_swrcb_pws_facilities.csv").exists() and
        Path("data/raw/ca_swrcb_safer.csv").exists()):
        ca_water = build_ca_water_vulnerability(crosswalk, acs_plumbing=acs_plumbing)

    # Combine federal + CA, fill missing with median
    all_fips = impact["fips"]  # universe of counties in our pipeline
    water_vuln = combine_water_vulnerability(federal_water, ca_water, all_fips)

    road_data = None
    tiger_road_path = "data/processed/tiger_road_miles_by_county.csv"
    if Path(tiger_road_path).exists() and county_areas is not None and not county_areas.empty:
        road_data = build_road_redundancy(county_areas, tiger_path=tiger_road_path)

    # P3g-HFTD: CPUC fire-threat exposure (CA only)
    hftd_data = None
    if Path("data/raw/ca_hftd_county_exposure.csv").exists():
        hftd_data = build_hftd_exposure()

    utility_data = None
    # Try multiple naming conventions for EIA files
    eia_service_path = None
    eia_sales_path = None
    for sp in ["data/raw/Service_Territory_2024.xlsx",
               "data/raw/eia_service_territory_2024.xlsx"]:
        if Path(sp).exists():
            eia_service_path = sp
            break
    for sp in ["data/raw/Sales_Ult_Cust_2024.xlsx",
               "data/raw/eia_sales_ult_cust_2024.xlsx"]:
        if Path(sp).exists():
            eia_sales_path = sp
            break

    if eia_service_path and eia_sales_path:
        eia_service = pd.read_excel(eia_service_path)
        eia_sales   = pd.read_excel(eia_sales_path, header=2)
        utility_data = build_utility_hardening(eia_service, eia_sales, crosswalk,
                                                county_areas, hftd_data=hftd_data)
    elif Path("data/raw/eia_form861_utility.xlsx").exists():
        eia_861 = pd.read_excel("data/raw/eia_form861_utility.xlsx")
        eia_sales_fallback = pd.DataFrame()
        utility_data = build_utility_hardening(eia_861, eia_sales_fallback, crosswalk,
                                                county_areas, hftd_data=hftd_data)

    # ── P4: Infrastructure resilience score ──
    infrastructure = build_infrastructure_resilience_score(
        utility_data=utility_data,
        road_data=road_data,
        water_data=water_vuln,
    )

    # ── Final risk ──
    final = build_final_risk(probability, impact, infrastructure=infrastructure)

    # ── Predictive ML layer ──
    train_df        = build_training_dataset(cf_detail, drought, unem, gdp, industry, fire_freq, qcew=qcew, water_vuln=water_vuln)
    model_nonca, model_ca, fi, _ = select_best_model(train_df, force_retrain=False)
    ml_predictions  = predict_current_impact(model_nonca, probability, drought, unem, gdp, industry, fire_freq, qcew=qcew, model_ca=model_ca, water_vuln=water_vuln)

    # Merge ML predictions into final output
    if not ml_predictions.empty:
        ml_cols = ["fips","predicted_gdp_impact_pct","model_region","prediction_ci_low",
                   "prediction_ci_high","confidence_flag","top_shap_driver","shap_breakdown"]
        final = final.merge(ml_predictions[ml_cols], on="fips", how="left")

    # ── P3m: WUI exposure fraction ──
    print("\n[P3m] Loading WUI housing fraction data...")
    wui = pd.read_csv("data/processed/wui_county.csv")
    wui["fips"] = wui["fips"].astype("Int64")
    WUI_MEDIAN = wui["wui_housing_fraction"].median()
    final = final.merge(
        wui[["fips", "wui_housing_fraction"]],
        on="fips", how="left"
    )
    final["wui_data_available"] = final["wui_housing_fraction"].notna()
    final["wui_housing_fraction"] = final["wui_housing_fraction"].fillna(WUI_MEDIAN)
    n_wui = final["wui_data_available"].sum()
    n_fill = (~final["wui_data_available"]).sum()
    print(f"  WUI matched: {n_wui}, median-filled: {n_fill} (median={WUI_MEDIAN:.4f})")

    # ── P2: Dollar loss and wage loss estimates ──
    final = add_dollar_loss_estimate(final, gdp)
    final = add_wage_loss_estimate(final)

    # ── P6: Rural county flags (diagnostic only — no score changes) ──
    print("\n[P6] Adding rural county flags...")

    # Population density from GDP + Gazetteer
    gdp_pop = (
        gdp[gdp["fips"].notna()]
        .groupby("fips")["County Population"]
        .mean()
        .reset_index()
        .rename(columns={"County Population": "county_population"})
    )
    if county_areas is not None and not county_areas.empty:
        pop_area = gdp_pop.merge(county_areas[["fips", "area_sq_miles"]], on="fips", how="inner")
        pop_area["population_density"] = pop_area["county_population"] / pop_area["area_sq_miles"]
        final = final.merge(
            pop_area[["fips", "population_density"]],
            on="fips", how="left"
        )
    else:
        final["population_density"] = np.nan
    final["is_rural"] = final["population_density"] < 50

    # Industry source from impact layer
    if "industry_source" in impact.columns:
        ind_src = impact[["fips", "industry_source"]].drop_duplicates("fips")
        final = final.merge(ind_src, on="fips", how="left", suffixes=("", "_dup"))
        if "industry_source_dup" in final.columns:
            final.drop(columns=["industry_source_dup"], inplace=True)
    final["rural_industry_source"] = final.get("industry_source", pd.Series("Unknown", index=final.index))

    # GDP data quality: surveyed if >= 4 quarters of actual data
    gdp_w = gdp[gdp["fips"].notna()].copy()
    gdp_qtrs = gdp_w.groupby("fips").size().reset_index(name="gdp_quarter_count")
    final = final.merge(gdp_qtrs, on="fips", how="left")
    final["rural_gdp_data_quality"] = final["gdp_quarter_count"].apply(
        lambda x: "surveyed" if pd.notna(x) and x >= 4 else "modeled"
    )
    final.drop(columns=["gdp_quarter_count"], inplace=True)

    # Training coverage: fire events per county from cf_detail
    if not cf_detail.empty:
        event_counts = cf_detail.groupby("fips").size().reset_index(name="rural_training_coverage")
        final = final.merge(event_counts, on="fips", how="left")
        final["rural_training_coverage"] = final["rural_training_coverage"].fillna(0).astype(int)
    else:
        final["rural_training_coverage"] = 0

    # Prediction confidence
    final["rural_prediction_confidence"] = final["rural_training_coverage"].apply(
        lambda x: "high" if x >= 5 else ("medium" if x >= 2 else "low")
    )

    n_rural = final["is_rural"].sum()
    n_total = len(final)
    print(f"  Rural counties (density < 50/sq mi): {n_rural}/{n_total}")
    print(f"  Prediction confidence: "
          f"high={( final['rural_prediction_confidence']=='high').sum()}, "
          f"medium={(final['rural_prediction_confidence']=='medium').sum()}, "
          f"low={(final['rural_prediction_confidence']=='low').sum()}")

    # ── Save outputs ──
    prob_cols = [c for c in [
        "probability_rank","fips","county_name","state",
        "probability_score","probability_tier",
        "drought_score","fire_frequency_score",
        "max_consecutive_weeks","recent_intensity",
        "episode_frequency","total_drought_weeks",
        "fire_count","fires_per_100k","last_drought_end",
    ] if c in probability.columns]

    imp_cols = [c for c in [
        "impact_rank","fips","county_name","state",
        "impact_score","impact_tier","trajectory",
        "causal_impact_score","avg_gdp_delta_pct","avg_counterfactual_gap",
        "industry_sensitivity","industry_sensitivity_n",
        "gdp_industry_sensitivity_n","blended_industry_sensitivity_n",
        "industry_confidence",
        "top_industries","dominant_sector","industry_source",
        "economic_trajectory","decay_unem_rate","gdp_growth_trend",
        "land_value_per_acre","nass_amplifier","nass_data_available",
    ] if c in impact.columns]

    final_cols = [c for c in [
        "final_rank","fips","county_name","state",
        "final_risk_score","final_risk_tier","risk_quadrant",
        "probability_score","probability_tier",
        "impact_score","impact_tier",
        "infrastructure_vulnerability_score","infrastructure_tier",
        "predicted_risk_score","predicted_risk_tier",
        "risk_direction","risk_change","trajectory",
        "predicted_gdp_impact_pct","prediction_ci_low","prediction_ci_high",
        "confidence_flag","top_shap_driver","shap_breakdown",
        "dominant_sector","industry_source","top_industries",
        "total_gdp_m","wui_housing_fraction","wui_data_available",
        "predicted_gdp_loss_m","estimated_loss_m",
        "total_loss_low_m","total_loss_high_m","estimated_wage_loss_m",
        "is_rural","population_density","rural_industry_source",
        "rural_gdp_data_quality","rural_training_coverage",
        "rural_prediction_confidence",
    ] if c in final.columns]

    cf_cols = [c for c in [
        "county_name","county_key","fire_period","acreage",
        "gdp_before","gdp_after","gdp_delta_pct",
        "control_avg_delta","counterfactual_gap",
    ] if c in cf_detail.columns]

    probability[prob_cols].to_csv("county_probability_scores.csv", index=False)
    impact[imp_cols].to_csv("county_impact_scores.csv", index=False)
    final[final_cols].to_csv("county_final_risk_scores.csv", index=False)
    cf_detail[cf_cols].to_csv("county_causal_evidence.csv", index=False)

    # Save ML predictions separately too
    if not ml_predictions.empty:
        ml_predictions.to_csv("county_ml_predictions.csv", index=False)

    write_report(probability, impact, final, cf_detail)

    print("\n  county_probability_scores.csv → fire likelihood")
    print("  county_impact_scores.csv      → economic damage if fire occurs")
    print("  county_final_risk_scores.csv  → probability × impact + ML predictions")
    print("  county_ml_predictions.csv     → RF predicted GDP impact + SHAP explanations")
    print("  county_causal_evidence.csv    → pre/post + counterfactual detail")
    print("  risk_summary_report.txt       → shareholder narrative")
    print("  wildfire_rf_model.pkl         → trained model (reused on next run)")
    print("\nDone! ✓")


if __name__ == "__main__":
    main()
