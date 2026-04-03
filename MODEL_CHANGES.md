# Model Update Specification

This document tells Claude Code exactly what to change in
`Wildfire_Risk_Analysis_1.py`, in what order, and why.
Read alongside CLAUDE.md before touching any code.

---

## Ground Rules

1. Never change the core formula without explicit instruction:
   `final_risk_score = normalize(probability_score × impact_score)`

2. Never change these config values without explicit instruction:
   - `DECAY_RATE = 0.3`
   - `PRE_POST_QTR = 4`
   - `N_CONTROLS = 5`
   - `MIN_ACREAGE = 100`
   - Tier cut points: `[0, 0.25, 0.50, 0.75, 1.0]`

3. After every change, confirm output county counts match baseline.
   Run the pipeline before and after. If counts drop, a join broke — fix before proceeding.

4. All new functions follow the existing pattern:
   - Accept DataFrames as input, return DataFrames
   - Print progress with `[TAG]` prefix matching existing style
   - Normalize sub-scores to [0,1] using existing `normalize()` helper
   - Use existing `invert()` helper for inverse relationships
   - Join on FIPS (`Int64`) not county name strings
   - Fill missing values with median (conservative, not zero)

5. New data files go in `data/raw/`. Processed versions go in `data/processed/`.
   Never modify raw files.

6. The model measures economic output disruption, NOT residential property loss.
   Never use Zillow, CoreLogic, or residential home price data.

---

## What Does NOT Change

These functions and values are explicitly frozen:

- `normalize()` and `invert()` helper functions
- `decay_weight()` function and `DECAY_RATE = 0.3`
- `build_drought_score()` — all weights and sub-indicator logic
- `build_fire_frequency_score()` — acreage weighting, population normalization
- `build_probability_score()` — `drought × 0.60 + fire_frequency × 0.40`
  _(these weights will be replaced by trained weights in P5, but not before)_
  **DESIGN NOTE — RETROSPECTIVE INDEX, NOT FORECAST:**
  probability_score is a recency-weighted fire frequency index. The geo_events
  dataset covers 2021-08-11 to 2025-09-05, so top-scoring counties have already
  burned in this window. This is correct for a risk index but is NOT an
  independent ignition forecast. True forward-looking probability (fuel load,
  WUI, fire weather index) is in scope for P8 (sfa_analysis.py).
  The ML layer (XGBoost, CV R²=0.516) IS genuinely predictive — it predicts
  GDP impact if a fire occurs, trained on pre-fire conditions.
- `build_prepost_windows()` — `PRE_POST_QTR=4`, `MIN_ACREAGE=100`
- `build_counterfactual()` — `N_CONTROLS=5`, cosine similarity on 3 features
- `build_recovery_trajectory()` — trajectory classification logic and thresholds
  are frozen. The industry merge IS being updated in P1 to include
  `dominant_sector` and `industry_source` — this is a bug fix, not a logic change.
- The 4-tier classification: Low/Moderate/High/Critical at 0.25 intervals
- The quadrant logic — UPDATED from median splits to 75th percentile thresholds.
  Median splits produce exactly 219 counties per quadrant (not meaningful).
  Fix applied in build_final_risk():
    prob_high_threshold   = df["probability_score"].quantile(0.75)
    impact_high_threshold = df["impact_score"].quantile(0.75)
  Result: 63 High/High, 156 High/Low, 156 Low/High, 501 Low/Low (876 total)
  P75 thresholds: probability=0.4292, impact=0.5105

---

## P1 — County Crosswalk

### Problem Being Fixed
GDP and several downstream functions join on `county.lower().strip()` string
matching. This silently drops counties where names differ across datasets
(e.g. "St. Louis" vs "Saint Louis", "De Kalb" vs "DeKalb"). Every new dataset
we add inherits this fragility unless fixed first.

### Data Needed
```
URL:  https://www2.census.gov/geo/docs/reference/codes2020/national_county2020.txt
Save: data/raw/national_county2020.txt
```

### New Function (add near top, after existing helpers)
```python
def load_county_crosswalk(
    path: str = "data/raw/national_county2020.txt"
) -> pd.DataFrame:
    """
    Loads Census ANSI county code list as authoritative FIPS lookup.

    Provides canonical mapping:
      fips (5-digit Int64) → county_name_canonical, state_abbr, state_name

    Use this to standardize all county name joins to FIPS joins.
    Any dataset with only county name gets joined here first,
    then joins everything else via FIPS.

    county_key_short strips common suffixes (" County", " Parish",
    " Borough") for fuzzy matching of datasets that omit them.
    """
    df = pd.read_csv(
        path, sep="|", dtype=str,
        names=["state_abbr", "state_fips", "county_fips",
               "county_name_canonical", "class_code"]
    )
    df["fips"] = (df["state_fips"] + df["county_fips"]).astype("Int64")
    df["county_key"] = df["county_name_canonical"].str.lower().str.strip()
    df["county_key_short"] = (
        df["county_key"]
        .str.replace(
            r"\s+(county|parish|borough|census area|municipality)$",
            "", regex=True
        )
    )
    return df[[
        "fips", "state_fips", "county_fips",
        "county_name_canonical", "county_key",
        "county_key_short", "state_abbr"
    ]]
```

### Functions to Update
- `load_data()`: add crosswalk load, return as 6th element
- `build_fire_frequency_score()`: join population via FIPS not county_key
- `build_counterfactual()`: join GDP features via FIPS not county_key
- `build_recovery_trajectory()`: fix industry merge AND join GDP data via FIPS not county_key (see below)
- `build_economic_trajectory()`: join GDP trends via FIPS not county_key
- `predict_current_impact()`: join GDP per capita via FIPS not county_key

### Additional Fix — dominant_sector Data Flow Bug

**Problem:** `dominant_sector` and `industry_source` are computed by
`build_industry_sensitivity()` but silently dropped in `build_recovery_trajectory()`
because the merge only selects four columns:

```python
# Current — drops dominant_sector and industry_source
full = causal_w.merge(
    industry[["fips","industry_sensitivity","industry_sensitivity_n","top_industries"]],
    on="fips", how="outer"
)
```

As a result, `dominant_sector` never reaches `build_impact_score()`, never
appears in any output CSV, and cannot be used for sector-specific logic
(such as the USDA NASS agricultural amplifier in P3d).

`dominant_sector` is a critical field — it's the primary signal for determining
which sector-specific amplifiers apply to a county (USDA NASS for agriculture,
FAIR Plan underinsurance for residential-heavy economies, etc.).

**Fix — one line change in `build_recovery_trajectory()`:**

```python
# Fixed — preserves dominant_sector and industry_source through the pipeline
full = causal_w.merge(
    industry[[
        "fips",
        "industry_sensitivity",
        "industry_sensitivity_n",
        "top_industries",
        "dominant_sector",     # added — needed for sector-specific amplifiers
        "industry_source",     # added — needed for output transparency
    ]],
    on="fips", how="outer"
)
```

**Downstream effects of this fix:**
- `dominant_sector` now flows into `build_impact_score()` output
- `dominant_sector` now appears in `county_impact_scores.csv` and
  `county_final_risk_scores.csv` as populated columns (not silently absent)
- P3d USDA NASS amplifier can now condition on
  `dominant_sector == "Agriculture & Forestry"` as intended
- County narratives (P7) can reference the dominant sector accurately

**Also add to `build_impact_score()` output columns** — confirm these are
included in the return DataFrame and in `imp_cols` in `main()`:
```python
"dominant_sector",
"industry_source",
```

### Validation
After change: output county counts should be equal to or greater than baseline.
If counts increase, the fix is working (previously dropped counties now join).
Additionally verify: `dominant_sector` column appears and is populated in
`county_impact_scores.csv` — if it shows all NaN the fix did not take.

---

## P2 — Dollar Loss and Wage Loss Estimates

### Problem Being Fixed
All current outputs are percentages and normalized 0–1 scores. No dollar figures
exist anywhere in the pipeline. Insurance underwriters and county officials need
dollar amounts to act, not abstract scores.

### Methodology
UCLA Anderson (2025) finding: total capital loss = insured loss × 2.2 to 2.8.
- 2.2 = California wildfire historical baseline (Camp/Woolsey/Tubbs/Thomas)
- 2.8 = Hurricane Katrina infrastructure ratio (higher infra damage)
BEA national labor share of income = 0.53 (proportion of GDP that is wages).

### New Functions (add after build_final_risk())

```python
def add_dollar_loss_estimate(
    final_df: pd.DataFrame,
    gdp: pd.DataFrame
) -> pd.DataFrame:
    """
    Converts predicted GDP impact % into a dollar loss range.

    UCLA Anderson multipliers (2025):
      2.2 = California wildfire historical baseline
      2.8 = Hurricane Katrina infrastructure ratio (upper bound)

    New columns:
      total_gdp_m:           county total GDP in $M
      predicted_gdp_loss_m:  predicted GDP loss in $M
      total_loss_low_m:      lower bound total economic loss (× 2.2)
      total_loss_high_m:     upper bound total economic loss (× 2.8)

    Only applied where predicted_gdp_impact_pct is available.
    Counties without ML predictions receive NaN.
    """
    UCLA_LOW  = 2.2
    UCLA_HIGH = 2.8

    gdp_totals = (
        gdp[gdp["Year"] == gdp["Year"].max()]
        .groupby("County")
        .agg(total_gdp_m=("Est. County GDP (Current $M)", "sum"))
        .reset_index()
    )
    gdp_totals["county_key"] = gdp_totals["County"].str.lower().str.strip()

    df = final_df.copy()
    if "county_key" not in df.columns:
        df["county_key"] = df["county_name"].str.lower().str.strip()

    df = df.merge(gdp_totals[["county_key", "total_gdp_m"]],
                  on="county_key", how="left")

    if "predicted_gdp_impact_pct" in df.columns:
        df["predicted_gdp_loss_m"] = (
            df["total_gdp_m"] *
            df["predicted_gdp_impact_pct"].abs() / 100
        )
        df["total_loss_low_m"]  = df["predicted_gdp_loss_m"] * UCLA_LOW
        df["total_loss_high_m"] = df["predicted_gdp_loss_m"] * UCLA_HIGH
    else:
        df["predicted_gdp_loss_m"] = np.nan
        df["total_loss_low_m"]     = np.nan
        df["total_loss_high_m"]    = np.nan

    return df


def add_wage_loss_estimate(final_df: pd.DataFrame) -> pd.DataFrame:
    """
    Estimates wage loss from predicted GDP loss.

    Method: wage_loss = GDP_loss × labor_share_of_income
    BEA national average labor share = 0.53

    This is the number municipal governments need for tax revenue
    forecasting: lost wages = lost income tax = lost county revenue.

    New column:
      estimated_wage_loss_m: estimated wage loss in $M
    """
    LABOR_SHARE = 0.53

    df = final_df.copy()
    if "predicted_gdp_loss_m" in df.columns:
        df["estimated_wage_loss_m"] = (
            df["predicted_gdp_loss_m"] * LABOR_SHARE
        )
    else:
        df["estimated_wage_loss_m"] = np.nan
    return df
```

### Changes to main()
```python
# After build_final_risk():
final = add_dollar_loss_estimate(final, gdp)
final = add_wage_loss_estimate(final)
```

### Changes to Output Column Lists
Add to `final_cols`:
```python
"total_gdp_m",
"predicted_gdp_loss_m",
"total_loss_low_m",
"total_loss_high_m",
"estimated_wage_loss_m",
```

### Changes to generate_county_narrative()
If `total_loss_low_m` is available, add line:
`"Estimated total economic loss: ${low:.0f}M – ${high:.0f}M (UCLA Anderson methodology)"`

### Validation
Spot-check: LA County `predicted_gdp_impact_pct × LA GDP` should produce
a loss in the billions — consistent with UCLA's $4.6B finding for 2025.

---

## P2.5 — Training Label Fix (counterfactual_gap)

### Problem Being Fixed
`LABEL_COL = "gdp_delta_pct"` trains the ML model on raw GDP change, which is
contaminated by macroeconomic trends (COVID recovery, national growth). Most
counties saw GDP growth from 2019–2023 regardless of fires. The fire signal
is buried under macro noise, causing the model to predict GDP *growth* after
fires — the opposite of what fire damage looks like.

### Fix
```python
# Old:
LABEL_COL = "gdp_delta_pct"   # raw GDP change — contaminated by macro trends

# New:
LABEL_COL = "counterfactual_gap"   # fire-attributable GDP change only
```

### Rationale
`counterfactual_gap` is already computed in `build_counterfactual()`. It
subtracts the GDP trajectory of N=5 matched unburned counties, removing
the macro trend and isolating fire-attributable damage. This is the correct
training label — model selection and weight training cannot compensate for
a wrong label.

### Expected Effects
- Cross-val R² should improve (signal-to-noise ratio increases)
- LA County prediction should flip from positive to negative
- SHAP feature drivers should shift toward fire-relevant features
  (drought weeks, industry sensitivity) away from macro proxies (GDP per capita)

### Validation
After change: retrain with `force_retrain=True`. Compare cross-val R² before
and after. Confirm LA County predicted impact is negative (GDP decline).

---

## P3 — New Data Sources

### 3a. BEA CAGDP2 — GDP-Weighted Industry Sensitivity
```
NOTE: BEA does not publish private fixed assets at county level.
P3a was revised to use BEA CAGDP2 regional GDP by industry instead —
this gives GDP-weighted industry earnings mix per county, which is
used to blend with FOIA industry sensitivity for a more robust score.

URL:  https://apps.bea.gov/api/data
      Table: CAGDP2 (GDP by County and Metropolitan Area)
      GeoFips: COUNTY, Year: most recent available
Save: fetched via BEA API (no flat file download)
Join: FIPS (GeoFIPS field)
```

`build_bea_industry_sensitivity()` derives GDP-weighted sensitivity by
mapping BEA industry line codes to NAICS sensitivity values, then
blending 50/50 with FOIA loan-weighted sensitivity:
  - Where both FOIA and BEA available: 50/50 blend
  - Where only FOIA: use FOIA only
  - Where only BEA: use BEA only
  - Confidence flag: "high" if sources agree (|diff| < 0.10), else "low"

Output: fips, gdp_weighted_sensitivity, industry_sensitivity_blended,
        blend_confidence (high/low), industry_source

### 3b. BLS QCEW — Wage Base
```
URL:  https://www.bls.gov/cew/downloadable-data.htm
      → County High-Level → most recent annual file
Save: data/raw/bls_qcew_county_annual.csv
Cols: area_fips, own_code, industry_code, year,
      annual_avg_emplvl, total_annual_wages, annual_avg_estabs
Filter: industry_code = "10" (all industries), own_code = "0"
Join: area_fips = county FIPS
```

Update `build_economic_trajectory()` to accept `qcew=None` parameter.
Add wage sub-indicators and rebalance weights:
```python
# Updated formula (was: unem×0.45 + unem_trend×0.25 + gdp_trend×0.30)
result["economic_trajectory"] = (
    result["unem_n"]        * 0.35 +   # was 0.45
    result["unem_trend_n"]  * 0.20 +   # was 0.25
    result["gdp_trend_n"]   * 0.25 +   # was 0.30
    result["wage_base_n"]   * 0.10 +   # new
    result["wage_trend_n"]  * 0.10     # new (inverted — declining = vulnerable)
)
```

### 3c. CA FAIR Plan
```
URLs (ZIP-level PDFs, aggregated to county):
  PIF: https://www.cfpnet.com/wp-content/uploads/2025/11/CFP-5-yr-PIF-County-FY25-All-251114.pdf
  TIV: https://www.cfpnet.com/wp-content/uploads/2025/11/CFP-5-yr-TIV-County-FY25-All-251114.pdf
  Residential PIF by ZIP: https://www.cfpnet.com/wp-content/uploads/2026/03/Policies-by-category-DWE-as-of-251231-DL-260319v001-002.pdf
  Commercial PIF by ZIP: (separate commercial PDF at cfpnet.com)
  Residential TIV by ZIP: (separate residential TIV PDF at cfpnet.com)
  Commercial TIV by ZIP: (separate commercial TIV PDF at cfpnet.com)
Save: data/processed/fair_plan_pif_by_county.csv   (58 CA counties, FY2025)
      data/processed/fair_plan_tiv_by_county.csv   (58 CA counties, FY2025)
      data/processed/fair_plan_county.csv           (pipeline-ready, merged)
Cols: county, pif_residential, pif_commercial, total_pif,
      exposure_residential_m, exposure_commercial_m, total_tiv_m
Join: county name → crosswalk[state_abbr == "CA"] → FIPS
NOTE: California ONLY. Non-CA counties get fair_plan_available=False,
      fair_plan_amplifier=1.0

Coverage gap uses actual per-policy exposure vs legal caps:
  RESIDENTIAL_CAP_M = 3.0   ($3M — CA law, do not train)
  COMMERCIAL_CAP_M  = 20.0  ($20M — CA law, do not train)
  avg_residential_exposure = exposure_residential_m / pif_residential
  avg_commercial_exposure  = exposure_commercial_m  / pif_commercial
  residential_gap = (avg_residential_exposure - RESIDENTIAL_CAP_M).clip(0)
  commercial_gap  = (avg_commercial_exposure  - COMMERCIAL_CAP_M).clip(0)
  coverage_gap = residential_gap × res_share + commercial_gap × com_share

NOTE from diagnostics: avg residential exposure ~$0.935M (below $3M cap).
Coverage gap is zero for all 58 counties — amplifier differentiates purely
by fair_plan_share (private insurer market withdrawal signal).
```

```python
def build_fair_plan_amplifier(
    fair: pd.DataFrame,
    crosswalk: pd.DataFrame,
    gdp: pd.DataFrame,
    max_amplifier: float = 1.20    # default — trained in P5 Part E
) -> pd.DataFrame:
    """
    FAIR Plan underinsurance amplifier (California only).

    High FAIR Plan concentration = private insurers have exited =
    post-fire recovery will be systematically worse because residents
    and businesses are underinsured.

    FACTUAL CONSTANTS (do not train — set by California law):
      RESIDENTIAL_CAP_M = 3.0    ($3M FAIR Plan residential policy cap)
      COMMERCIAL_CAP_M  = 20.0   ($20M FAIR Plan commercial policy cap)

    MODEL PARAMETER (trained in P5 Part E):
      max_amplifier: maximum impact amplification from FAIR Plan concentration
      Default: 1.20 (+20%). Trained via optimization against GDP outcomes.

    The fair_plan_share vs coverage_gap weights (0.60, 0.40) within
    the raw FAIR Plan score are also trained in P5 Part E.

    fair_plan_amplifier range: 1.0 (no effect) to max_amplifier
    """
    RESIDENTIAL_CAP_M = 3.0    # legal constant — do not train
    COMMERCIAL_CAP_M  = 20.0   # legal constant — do not train
    # max_amplifier comes from parameter (trained value or default 1.20)
    # ... (full implementation)
```

Update `build_impact_score()` signature to accept `fair_plan_df=None`.

### 3d. USDA NASS — Farmland Values
```
URL:  https://quickstats.nass.usda.gov/
      Sector: Economics
      Group: Farms & Land & Assets
      Commodity: AG LAND
      Category: ASSET VALUE
      Data Item: AG LAND, INCL BUILDINGS - ASSET VALUE, MEASURED IN $ / ACRE
      Domain: TOTAL
      Geographic Level: County
      Year: 2022 (Census of Agriculture — county level only available every 5 years)
      Period Type: POINT IN TIME
Save: data/raw/usda_nass_land_values.csv
Actual columns in file: Program, Year, Period, Geo Level, State, State ANSI,
      County, County ANSI, Commodity, Data Item, Domain, Value, CV (%)
Join: State ANSI (zero-pad to 2) + County ANSI (zero-pad to 3) = FIPS
Use:  Ag damage amplifier for counties where dominant_sector =
      "Agriculture & Forestry" (NAICS 11, sensitivity 0.95)

Suppressed values: "(D)" or "(Z)" in Value column — coerce to NaN,
fill with state median.

NOTE: Value column contains commas — strip before numeric conversion.
NOTE: Row count ~39,111 total; filter to Data Item = $/acre + Domain = TOTAL
      gives ~3,074 unique county FIPS.

Amplifier formula:
  magnitude = _load_weights("NASS_AMPLIFIER", {"magnitude": 0.20})["magnitude"]
  land_value_n = normalize(land_value_per_acre) across ag-dominant counties only
  nass_amplifier = 1.0 + (land_value_n * magnitude)
  Applied ONLY where dominant_sector == "Agriculture & Forestry"
  All other counties: nass_amplifier = 1.0 (no effect)

Applied in build_impact_score() after score computation:
  impact_score = normalize((impact_score * nass_amplifier).clip(0, 1))

Output columns added: nass_land_value_per_acre, nass_amplifier, nass_data_available
```

### 3e. EPA SDWIS/ECHO + CA State Small WQ Risk
```
Three water data sources combined for full coverage:
  1. CA SWRCB SAFER       → CA public systems (already in P4)
  2. CA State Small WQ Risk → CA domestic wells + state small systems (Gap 2)
  3. EPA SDWIS/ECHO        → non-CA public systems

EPA SDWIS:
URL:  https://echo.epa.gov/tools/data-downloads/sdwa-download-summary
Save: data/raw/sdwis/SDWA_PUB_WATER_SYSTEMS.csv
      data/raw/sdwis/SDWA_VIOLATIONS.csv
Cols: PWSID, COUNTIES_SERVED (FIPS), POPULATION_SERVED_COUNT,
      PRIMARY_SOURCE_CODE, PWS_ACTIVITY_CODE, violation records
Filter: PWS_ACTIVITY_CODE == "A" (active), exclude CA (state == "CA")
Join: COUNTIES_SERVED = county FIPS

CA State Small WQ Risk (Gap 2 fix):
URL:  https://gis.data.ca.gov/datasets/waterboards::state-small-wq-risk/about
Save: data/raw/ca_state_small_wq_risk.csv
Cols: COUNTY, POPULATION, FINAL_SAFER_STATUS, WATER_QUALITY_SCORE
Covers: domestic wells + state small systems (1–15 households)
Merged into CA water vulnerability as population-weighted component.

Coverage achieved: 3,280 counties (CA SWRCB: 58, Federal SDWIS: 3,167,
Estimated/median fill: 55). Std=0.160, range=0.914 on 876-county set.

ACS B25049 Private Well Proxy (Gap 2 fix — national):
URL:  https://api.census.gov/data/2023/acs/acs5?get=NAME,B25049_001E,B25049_002E,B25049_003E,B25049_004E,B25049_005E,B25049_006E,B25049_007E&for=county:*
Save: data/raw/acs_plumbing_county.csv
Cols: NAME, B25049_001E (total), B25049_004E (owner lacking plumbing),
      B25049_007E (renter lacking plumbing), state, county (FIPS parts)
Join: state + county (zero-padded) = FIPS
Use:  private_well_proxy_pct = (B25049_004E + B25049_007E) / B25049_001E
      Added as sub-indicator in water_vulnerability_score with weight 0.15
```

### 3f. Census TIGER/Line Roads (replaces FHWA HM-71)
```
NOTE: FHWA Highway Statistics tables are state-level only — no county
breakdowns exist in Highway Statistics. County-level road data comes
from Census TIGER/Line roads instead.

URL:  https://www2.census.gov/geo/tiger/TIGER2024/ROADS/
      One ZIP per county: tl_2024_SSCCC_roads.zip
      (SS = 2-digit state FIPS, CCC = 3-digit county FIPS)
Save: data/raw/tiger_roads/  (download all ZIPs, extract shapefiles)
Cols: LINEARID, FULLNAME, MTFCC (functional class code), geometry
Join: county FIPS embedded in filename — no spatial join needed

MTFCC → functional class mapping:
  S1100 = Primary road (Interstate equivalent)
  S1200 = Secondary road (Principal/Minor Arterial equivalent)
  S1400 = Local neighborhood road
  S1500 = Vehicular trail
  S1630 = Ramp
  S1640 = Service drive
  (use S1100, S1200, S1400 only — discard others)

Assumed lane counts (same as FHWA estimation methodology):
  S1100 → 4 lanes (divided highway)
  S1200 → 2 lanes
  S1400 → 2 lanes

road_miles_by_class = sum(segment_length_miles × assumed_lanes)
                      grouped by county FIPS and MTFCC
```


### 3f Pre-processing Step
TIGER road shapefiles must be pre-processed into a summary CSV before
the main pipeline runs. Add a standalone script `scripts/process_tiger_roads.py`:

```python
import geopandas as gpd
import pandas as pd
from pathlib import Path

TIGER_DIR = Path("data/raw/tiger_roads")
MTFCC_LANES = {"S1100": 4, "S1200": 2, "S1400": 2}

records = []
for fp in TIGER_DIR.glob("tl_2024_*_roads.shp"):
    # Extract county FIPS from filename: tl_2024_06037_roads.shp → 6037
    fips = int(fp.stem.split("_")[2])
    gdf  = gpd.read_file(fp, columns=["MTFCC","geometry"])
    gdf  = gdf[gdf["MTFCC"].isin(MTFCC_LANES)]
    gdf["length_miles"] = gdf.geometry.length / 1609.34  # meters → miles
    gdf["lane_miles"]   = gdf["length_miles"] * gdf["MTFCC"].map(MTFCC_LANES)
    for mtfcc, grp in gdf.groupby("MTFCC"):
        records.append({
            "fips":        fips,
            "mtfcc":       mtfcc,
            "lane_miles":  grp["lane_miles"].sum(),
        })

df = pd.DataFrame(records)
df.to_csv("data/processed/tiger_road_miles_by_county.csv", index=False)
print(f"Done: {df['fips'].nunique()} counties")
```

Run once before the pipeline. Output: `data/processed/tiger_road_miles_by_county.csv`
Cols: fips, mtfcc (S1100/S1200/S1400), lane_miles

Note: Requires geopandas. Install: `pip install geopandas`.
Download all county TIGER ZIPs for the 50 states + DC. Fastest approach:
use the bulk state-level roads file instead of individual county files:
`https://www2.census.gov/geo/tiger/TIGER2024/ROADS/` — filter by state FIPS.

### 3g. EIA Form 861
```
URL:  https://www.eia.gov/electricity/data/eia861/zip/f8612024.zip
Save: data/raw/eia_form861_service_territory.xlsx  ← utility→county mapping
      data/raw/eia_form861_sales_ult_cust.xlsx      ← customer counts
NOTE: Do NOT use Utility_Data_2024.xlsx — it has no county or customer data.

Service Territory file: utility-county pairs (one row per utility-county)
  Key cols: Utility Number, State, County (name string)
Sales file: customer counts per utility
  Key cols: Utility Number, Total Customers

Build utility→county crosswalk from Service Territory.
Join customer counts on Utility Number.
Distribute customers evenly across counties served by each utility.
Aggregate to county: sum distributed customers per FIPS.

Coverage: 3,074 counties matched (97.8%). CA counties get HFTD+density
blend (hftd_exposure=0.60, customer_density=0.40). Non-CA uses
density-only proxy (inverted — rural = more overhead lines = higher vuln).
Join: county name + state → crosswalk → FIPS
```

### 3h. Census Gazetteer (county areas)
```
URL:  https://www.census.gov/geographies/reference-files/time-series/geo/gazetteer-files.html
      → Counties → 2023_Gaz_counties_national.txt
Save: data/raw/2023_Gaz_counties_national.txt
Cols: USPS (state abbr), GEOID (5-digit FIPS), ALAND (land area sq meters)
Use:  county area in sq miles for road redundancy ratio computation
      area_sq_miles = ALAND / 2,590,000
```

### 3i. SCO County Revenues + Expenditures (Fiscal Resilience)
```
URL:  https://bythenumbers.sco.ca.gov/Raw-Data/
      → County Budgets → Revenues (all years)
      → County Budgets → Expenditures (all years)
      → City Budgets → Revenues (for San Francisco only — FIPS 6075)
      → City Budgets → Expenditures (for San Francisco only)
Save: data/raw/sco_county_revenues.csv
      data/raw/sco_county_expenditures.csv
      data/raw/sco_city_revenues.csv      ← SF only
      data/raw/sco_city_expenditures.csv  ← SF only
Coverage: 58/58 CA counties (SF sourced from city dataset FY2019-2021,
          all other counties use FY2022-2024 3-year average)
Join: county name → crosswalk[state_abbr == "CA"] → FIPS
NOTE: Non-CA counties get fiscal_resilience_available = False,
      fiscal_resilience_score = national median fill
```

New function: build_fiscal_resilience_score(revenues, expenditures,
              crosswalk, city_revenues=None, city_expenditures=None)

Three sub-indicators:
  property_tax_dependence = property_tax_revenue / total_revenues (weight 0.45)
  fiscal_surplus_ratio    = (revenues - expenditures) / revenues  (weight 0.35) INVERTED
  debt_service_burden     = debt_service / total_expenditures      (weight 0.20)

Intergovernmental dependence penalty (confirmed needed from diagnostics):
  intergovernmental_share = intergovernmental_revenue / total_revenues
  ig_penalty_multiplier   = 1.0 + (intergovernmental_share - 0.50).clip(lower=0) * 1.5
  property_tax_dependence_adj = (property_tax_dependence * ig_penalty_multiplier).clip(upper=1.0)

  This corrects the formula for transfer-dependent counties (Del Norte,
  Imperial, Modoc, Butte) where low property_tax_dependence was being
  misread as resilience. ig_penalty_scale=1.5 is in optimized_weights.json
  for P5 tuning.

Output columns:
  fips, fiscal_resilience_score, fiscal_resilience_available,
  property_tax_dependence, fiscal_surplus_ratio, debt_service_burden,
  intergovernmental_share, ig_penalty_multiplier, fiscal_data_note
  (fiscal_data_note = "high_ig_dependence" where intergov_share > 0.50)

Add to optimized_weights.json:
  "FISCAL_RESILIENCE_WEIGHTS": {
      "property_tax_dependence": 0.45,
      "fiscal_surplus_ratio":    0.35,
      "debt_service_burden":     0.20,
      "ig_penalty_scale":        1.5,
      "ig_penalty_threshold":    0.50,
      "trained": false,
      "reason": "pending_P5",
      "scope": "CA_only"
  },
  "ECONOMIC_TRAJECTORY_WEIGHTS_CA": {
      "unem_n":                  0.30,
      "unem_trend_n":            0.15,
      "gdp_trend_n":             0.20,
      "wage_base_n":             0.10,
      "wage_trend_n":            0.10,
      "fiscal_resilience_score": 0.15,
      "trained": false,
      "reason": "pending_P5",
      "scope": "CA_only"
  }

Integration: update build_economic_trajectory() to accept fiscal_df=None.
When fiscal data available (CA), use ECONOMIC_TRAJECTORY_WEIGHTS_CA.
Non-CA counties use existing national formula unchanged.

Validation spot-checks (confirmed from diagnostics):
  Butte:       0.334 (was 0.236 before penalty — correct direction)
  Del Norte:   0.208 (transfer-dependent, positive surplus, defensible)
  Modoc:       0.297 (was 0.178 before penalty)
  Santa Clara: 0.436 (unchanged — IG share 0.298, no penalty applied)
  SF:          0.475 (self-funded, IG share 0.135, high PT dependence)
  Top 3 vulnerable: Sierra, Mono, San Luis Obispo
  Bottom 3 resilient: Plumas, Del Norte, Imperial

---

### 3j. FTB B-7 AGI by County (Income Vulnerability)
```
URL:  https://data.ca.gov/dataset/personal-income-tax-statistics-by-county
      (CA Franchise Tax Board B-7 report)
Save: data/raw/ftb_agi_by_county.csv
Coverage: 58 CA counties, tax year 2022
Cols: county, agi_bracket, num_returns, total_agi
AGI brackets: $0–$10K, $10K–$25K, $25K–$50K, $50K–$100K,
              $100K–$200K, $200K–$500K, $500K–$1M, $1M+
Join: county name → crosswalk[state_abbr == "CA"] → FIPS
NOTE: Non-CA counties get income_vulnerability_available = False,
      income_vulnerability_score = national median fill
```

New function: build_income_vulnerability_score(ftb_agi, crosswalk)

Three sub-indicators:
  low_income_share     = returns with AGI < $25K / total returns     (weight 0.50)
  income_concentration = AGI from brackets >$500K / total county AGI  (weight 0.30)
  agi_per_return       = total_agi / num_returns — INVERTED            (weight 0.20)

  income_vulnerability_score = (
      normalize(low_income_share)            * 0.50 +
      normalize(income_concentration)        * 0.30 +
      invert(normalize(agi_per_return))      * 0.20
  )

NOTE on income_concentration direction: high concentration = high vulnerability.
Rationale: when high earners relocate post-fire (common in high-risk areas),
county loses disproportionate tax base and consumer demand simultaneously.
Sign assumption validated in P5 weight training — ElasticNet will push this
weight toward zero if the direction is empirically wrong.

Output columns:
  fips, income_vulnerability_score, income_vulnerability_available,
  low_income_share, income_concentration, agi_per_return

Add to optimized_weights.json:
  "INCOME_VULNERABILITY_WEIGHTS": {
      "low_income_share":     0.50,
      "income_concentration": 0.30,
      "agi_per_return":       0.20,
      "trained": false,
      "reason": "pending_P5",
      "scope": "CA_only"
  },
  "ECONOMIC_TRAJECTORY_WEIGHTS_CA_FULL": {
      "unem_n":                      0.25,
      "unem_trend_n":                0.10,
      "gdp_trend_n":                 0.15,
      "wage_base_n":                 0.10,
      "wage_trend_n":                0.10,
      "fiscal_resilience_score":     0.15,
      "income_vulnerability_score":  0.15,
      "trained": false,
      "reason": "pending_P5",
      "scope": "CA_only — requires both SCO and FTB data"
  }

Integration: update build_economic_trajectory() to accept income_df=None.
Three-tier formula logic:
  Both fiscal + income available (CA full): ECONOMIC_TRAJECTORY_WEIGHTS_CA_FULL
  Fiscal only (CA partial):                 ECONOMIC_TRAJECTORY_WEIGHTS_CA
  Neither (non-CA):                         existing national formula

Expected validation results:
  Top 3 most vulnerable: agricultural/rural counties (Imperial, Tulare, Fresno)
  Bottom 3 least vulnerable: wealthy Bay Area (San Mateo, Santa Clara, Marin)
  Correlation with fiscal_resilience_score: r > 0.30 expected
  (poor counties tend to score high on both)


---


### 3k. SCO TOT Revenue — Tourism Concentration Amplifier (CA only)
```
Source: Already downloaded — data/raw/sco_county_revenues.csv
        (same file used for fiscal_resilience_score in P3i)

TOT = Transient Occupancy Tax — hotel/lodging tax collected by county.
High TOT share signals tourism-dependent economy: Bryce Canyon gateway
towns, Lake Tahoe, Yosemite corridor counties. These counties have
disproportionate wildfire impact because fire closures destroy an entire
visitor season regardless of physical damage.

Computation:
  tot_revenue = revenue rows where category == "Transient Occupancy Tax"
                (or equivalent in SCO file — confirm column name)
  total_revenue = total general fund revenues
  tot_share = tot_revenue / total_revenue

  tot_amplifier = 1.0 + (normalize(tot_share) * TOT_AMPLIFIER_MAGNITUDE)
  TOT_AMPLIFIER_MAGNITUDE = 0.20 (untrained default, add to optimized_weights.json)

Apply in build_impact_score() after nass_amplifier:
  impact_score = normalize((impact_score * tot_amplifier).clip(0, 1))
  (CA only — non-CA counties get tot_amplifier = 1.0)

Output columns: tot_share, tot_amplifier, tot_data_available

Add to optimized_weights.json:
  "TOT_AMPLIFIER": {
      "magnitude": 0.20,
      "trained": false,
      "reason": "pending_P5_rerun",
      "scope": "CA_only"
  }

Why this matters for government users:
  TOT is a primary revenue source for many CA counties. A wildfire during
  peak season destroys TOT revenue for the fiscal year, directly threatening
  county operating budgets. This is the mechanism by which high-probability
  fires create fiscal crises, not just economic damage.
```

### 3l. FEMA National Structure Inventory — Housing at Risk (National)
```
URL:  https://www.fema.gov/flood-maps/products-tools/national-structure-inventory
      Bulk download by state: https://nsi.sec.usace.army.mil/downloads/
      API: https://nsi.sec.usace.army.mil/nsiapi/structures?fips={FIPS}&fmt=fc

Data: Address-level structure counts, occupancy type, replacement values
      ~120M structures nationwide, updated annually by FEMA/USACE
Save: data/raw/fema_nsi/  (download by state, aggregate to county)
Cols: fd_id, x (lon), y (lat), occtype (RES1/RES2/COM/IND etc),
      val_struct (structure replacement value $), val_cont (contents value $),
      pop2amu65 (daytime population), county (FIPS)
Join: county FIPS field → direct join

Aggregation to county level:
  total_structures     = count of all structures
  residential_units    = count where occtype starts with "RES"
  residential_value_m  = sum(val_struct) for RES structures / 1e6
  total_value_m        = sum(val_struct + val_cont) / 1e6
  wui_structures       = structures in WUI zone (join with USFS WUI layer)

Output columns added to county_impact_scores.csv:
  total_structures, residential_units, residential_value_m,
  total_replacement_value_m, nsi_data_available

Integration into dollar loss estimate (add_dollar_loss_estimate()):
  total_dollar_loss_m = gdp_loss_m + wage_loss_m + residential_value_m * burn_probability
  burn_probability = fire_frequency_score × 0.15  (conservative: not all structures burn)

Why this matters for government use:
  - FEMA HMGP applications require documented structures at risk
  - Property tax base at risk = residential_value_m × local_tax_rate
  - Justifies pre-disaster mitigation investment with specific dollar figures
  - FEMA NSI is FEMA's own data — using it in HMGP applications is authoritative

Priority: Implement before P7 narrative engine so narratives can cite
structure counts and replacement values.
```


### Dollar Loss Exposure Fraction — WUI-Based (P3m)
```
PROBLEM: predicted_gdp_loss_m = county_gdp × predicted_impact_pct
This scales linearly with county size. LA County at $140B predicted
loss is not credible — the 2025 Palisades+Eaton fires caused ~$250B
total economic damage but business disruption within the fire perimeter
was estimated at $4.6–8.9B. A single county fire event should not
produce $140B loss estimates.

CORRECT APPROACH: Apply WUI exposure fraction before multiplying by GDP.
Only the fraction of county economic activity in WUI-exposed zones
is at risk from a typical wildfire.

  fire_exposure_fraction = wui_housing_fraction  (from USFS WUI 2020 data)
  predicted_gdp_loss_m   = county_gdp_m
                           × abs(predicted_gdp_impact_pct / 100)
                           × fire_exposure_fraction

DATA SOURCE:
  USFS Wildland-Urban Interface 2020 Dataset (Silvis Lab / Radeloff et al.)
  URL:  https://www.fs.usda.gov/rds/archive/catalog/RDS-2015-0012-4
  Save: data/raw/wui/RDS-2015-0012-4.gdb
  Format: geodatabase, census block level, ~500MB
  Requires: geopandas for spatial aggregation to county

  wui_housing_fraction = (housing_units_in_WUI) / (total_county_housing_units)
  WUI classes to include: intermix + interface (both types)
  Year: 2020 (most recent)

  Output: data/processed/wui_county.csv
  Cols: fips, wui_housing_fraction, wui_intermix_fraction,
        wui_interface_fraction, wui_data_available

Expected result after fix:
  LA County: wui_housing_fraction ≈ 0.15–0.25 (hills/canyons only)
  Rural fire counties: wui_housing_fraction ≈ 0.60–0.90 (most land is WUI)
  LA predicted_gdp_loss_m: ~$140B × 0.20 ≈ $28B (plausible for major urban fire)

This is P3m — implement after P3k (TOT) and P3l (FEMA NSI).
Priority: required before P7 narrative uses dollar figures.

Add to optimized_weights.json:
  "WUI_EXPOSURE": {
      "source": "USFS RDS-2015-0012-4",
      "year": 2020,
      "classes_used": ["intermix", "interface"],
      "trained": false,
      "note": "wui_housing_fraction used directly as exposure multiplier"
  }
```

## P4 — Infrastructure Resilience Score

### What
New function `build_infrastructure_resilience_score()` combining three
infrastructure vulnerability components. Runs after `build_impact_score()`,
modifies final risk computation.

### Why
Infrastructure damage has a different economic signature than property damage.
It doesn't just destroy value — it blocks recovery. The 2025 LA fires: business
disruption projected to cost $4.6–8.9B in lost output from 2025–2029 — that
multi-year tail is driven by infrastructure recovery timelines, not the fire
itself. No competitor model prices this. This separates "how bad is the damage"
from "how long will recovery take."

### Formula
```
infrastructure_vulnerability_score =
    utility_hardening_index    × w_util    (trained, default: 0.40)
  + road_redundancy_ratio      × w_road    (trained, default: 0.35)
  + water_vulnerability_score  × w_water   (trained, default: 0.25)

infrastructure_amplifier = 1.0 + (vulnerability_score × α)
  α = trained amplifier magnitude (default: 0.35)
  Range: 1.0 (resilient) to 1 + α_max (critical infrastructure failure)

NOTE: All four values (w_util, w_road, w_water, α) are trained via
constrained optimization in P5 Part D. Hardcoded defaults are used
only on the first pipeline run before training data is available.
See P5 Part D for full training methodology and two-pass call order.

impact_score_adjusted = impact_score × infrastructure_amplifier
final_risk_score = normalize(probability_score × impact_score_adjusted)
```

### Component 1: Utility Hardening Index (default weight: 0.40, trained in P5)
Source: EIA Form 861 + CPUC HFTD tier map (CA only)

Proxy when overhead line miles not directly available:
- High customer density → urban area → more hardening investment → lower vulnerability
- Rural counties with low customer density = more overhead lines = higher vulnerability
- Inverted customer density score = utility hardening vulnerability

CPUC HFTD tier for CA counties:
```
URL:  https://www.cpuc.ca.gov/industries-and-topics/wildfires
      → High Fire Threat District Map → Shapefile download
Tiers: 1 = acceptable, 2 = elevated, 3 = extreme
```
`hftd_exposure = weighted % of county in Tier 2/3`

### Component 2: Road Redundancy Ratio (default weight: 0.35, trained in P5)
Source: Census TIGER/Line Roads 2024

```python
# FC_WEIGHTS encode relative criticality of each road class.
# Default values below are intuition-based — trained in P5 Part E.
# Ordering constraint enforced during training: w[0] >= w[1] >= w[2] >= w[3] >= w[4]
# (interstates must remain more critical than local roads after training)
# MTFCC codes from Census TIGER roads
FC_WEIGHTS = {
    "S1100_lane_miles": 1.0,   # Primary road (Interstate) — trained in P5 Part E
    "S1200_lane_miles": 0.5,   # Secondary road (Arterial) — trained in P5 Part E
    "S1400_lane_miles": 0.05,  # Local road               — trained in P5 Part E
}
# Ordering constraint: S1100 >= S1200 >= S1400 enforced during training
road_density = weighted_lane_miles / county_area_sq_miles
road_redundancy_ratio = invert(normalize(road_density))
# Low density = low redundancy = high vulnerability
```

Single corridor flag: `arterial_concentration > 0.60 AND road_density < 25th percentile`

### Component 3: Water Vulnerability Score (default weight: 0.25, trained in P5)
Source: EPA SDWIS/ECHO

Three sub-indicators:
- `violation_rate` (weight 0.50): violations per 1000 people served, recency-weighted
  (last 5 years × 2.0, older × 1.0). Chronic violators fail catastrophically post-fire.
  Camp Fire precedent: benzene contamination cost $250M and took years to remediate.
- `system_fragmentation` (weight 0.25): water systems per 100k population.
  Fragmented = harder to repair/reroute when multiple systems damaged simultaneously.
- `surface_water_dependency` (weight 0.25): % population served by surface water.
  Surface water systems are MORE vulnerable to wildfire contamination (benzene, ash,
  sediment) than groundwater. Camp Fire was surface water. PRIMARY_SOURCE_CODE
  in ['SW', 'SWP', 'GUP'] = surface water.

### Changes to build_final_risk()
Update function signature and core computation:
```python
def build_final_risk(
    probability: pd.DataFrame,
    impact: pd.DataFrame,
    infrastructure: pd.DataFrame = None   # new optional parameter
) -> pd.DataFrame:
    # ...
    # Merge infrastructure
    if infrastructure is not None:
        df = df.merge(
            infrastructure[[
                "fips", "infrastructure_amplifier",
                "infrastructure_vulnerability_score",
                "infrastructure_resilience_score",
                "infrastructure_tier"
            ]],
            on="fips", how="left"
        )
        df["infrastructure_amplifier"] = df["infrastructure_amplifier"].fillna(1.0)
        df["impact_score_adjusted"] = (
            df["impact_score"] * df["infrastructure_amplifier"]
        ).clip(0, 1)
    else:
        df["impact_score_adjusted"] = df["impact_score"]

    # Updated core formula
    df["final_risk_score"] = normalize(
        df["probability_score"] * df["impact_score_adjusted"]
    )
```

### Changes to main()
```python
# Load infrastructure data
# TIGER roads: pre-process into county × MTFCC summary CSV before main()
    # See build_tiger_road_summary() in P3f
    tiger_roads  = pd.read_csv("data/processed/tiger_road_miles_by_county.csv")
sdwis        = pd.read_csv("data/raw/sdwis_violations.csv")
eia_861      = pd.read_excel("data/raw/eia_form861_utility.xlsx")
gaz          = pd.read_csv("data/raw/2023_Gaz_counties_national.txt",
                            sep="\t", dtype={"GEOID": str})
gaz["fips"]          = gaz["GEOID"].astype("Int64")
gaz["area_sq_miles"] = gaz["ALAND"] / 2_590_000
county_areas         = gaz[["fips", "area_sq_miles"]]

infrastructure = build_infrastructure_resilience_score(
    fhwa, sdwis, eia_861, crosswalk, county_areas
)
final = build_final_risk(probability, impact, infrastructure)
```

---

## P5 — Model Selection + Weight Training

### Part A: Model Selection

**Problem**: Random Forest was chosen as a reasonable first pass. XGBoost is
strictly better for tabular data — same SHAP support, better regularization,
better performance on small training sets. But we should select empirically.

**Candidate Models**:
```python
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import BayesianRidge, ElasticNet
import xgboost as xgb

CANDIDATE_MODELS = {
    "random_forest": RandomForestRegressor(
        n_estimators=300, max_depth=6,
        min_samples_leaf=3, random_state=42, n_jobs=-1
    ),
    "xgboost": xgb.XGBRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0, random_state=42, n_jobs=-1
    ),
    "gradient_boosting": GradientBoostingRegressor(
        n_estimators=200, max_depth=3,
        learning_rate=0.05, subsample=0.8, random_state=42
    ),
    "bayesian_ridge": BayesianRidge(max_iter=300),
    "elastic_net": ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=1000),
}
```

**New Function**:
```python
def select_best_model(
    X: np.ndarray,
    y: np.ndarray,
    cv_folds: int = 5
) -> tuple:
    """
    Runs cross-validation across all candidate models.
    Returns (best_model, best_name, results_df).

    Selection criteria: highest mean cross-val R².
    Tie-break: lowest MAE.
    Saves comparison results to outputs/model_comparison_results.csv.
    """
    results = []
    for name, model in CANDIDATE_MODELS.items():
        cv = min(cv_folds, len(X) // 5)
        r2  = cross_val_score(model, X, y, cv=cv, scoring="r2")
        mae = cross_val_score(model, X, y, cv=cv,
                              scoring="neg_mean_absolute_error")
        results.append({
            "model":      name,
            "cv_r2_mean": r2.mean(),
            "cv_r2_std":  r2.std(),
            "cv_mae_mean": -mae.mean(),
        })
        print(f"  {name:<22} R²={r2.mean():.3f}±{r2.std():.3f}  "
              f"MAE={-mae.mean():.2f}%")

    results_df = pd.DataFrame(results).sort_values(
        "cv_r2_mean", ascending=False
    )
    best_name  = results_df.iloc[0]["model"]
    best_model = CANDIDATE_MODELS[best_name]
    best_model.fit(X, y)

    results_df.to_csv("outputs/model_comparison_results.csv", index=False)
    print(f"\n  Best: {best_name} (R²={results_df.iloc[0]['cv_r2_mean']:.3f})")
    return best_model, best_name, results_df
```

Replace `train_random_forest()` call in `main()`:
```python
# Old: rf_model, fi, _ = train_random_forest(train_df, force_retrain=False)
# New:
X = train_df[FEATURE_COLS].values
y = train_df[LABEL_COL].values
best_model, best_model_name, model_comparison = select_best_model(X, y)
joblib.dump(best_model, "outputs/best_model.pkl")
```

### Part B: Trained Pillar Weights

**Problem**: Current hardcoded weights (drought 0.60, fire_freq 0.40, etc.)
cannot be defended to a methodological reviewer or sophisticated investor.
"We guessed" is not an acceptable answer. Trained weights can be defended:
"The data assigned X% to drought because it has the highest correlation with
post-fire GDP decline across N historical fire events in our training set."

**Approach**: Constrained optimization using `scipy.optimize.minimize` with
SLSQP method. Minimizes negative correlation between weighted sub-scores and
observed GDP outcomes. Constraints: weights ≥ 0, sum = 1. Bounds: each weight
between 5% and 80% (prevents any single indicator from dominating).

```python
from scipy.optimize import minimize

def train_pillar_weights(
    causal_df: pd.DataFrame,
    drought_sub_scores: pd.DataFrame,
    fire_freq_scores: pd.DataFrame,
    industry_scores: pd.DataFrame,
    econ_traj_scores: pd.DataFrame
) -> dict:
    """
    Learns optimal pillar weights via constrained optimization.

    Uses counterfactual GDP evidence as ground truth.
    Replaces hardcoded DROUGHT_WEIGHTS, PROB_WEIGHTS, IMPACT_WEIGHTS.

    Returns dict of optimized weights for each group.
    Saves to outputs/optimized_weights.json.

    Optimization objective: maximize |correlation(weighted_score, gdp_delta)|
    Constraint: all weights ≥ 0, each group sums to 1.0
    Bounds: each weight between 0.05 and 0.80
    """
    print("\n[Weight Training] Optimizing pillar weights...")

    df = causal_df.dropna(subset=["avg_gdp_delta_pct"]).copy()
    # ... join all sub-scores
    if len(df) < 20:
        print(f"  WARNING: Only {len(df)} rows. Using defaults.")
        return None

    y = df["avg_gdp_delta_pct"].values
    bounds = [(0.05, 0.80)]
    eq_constraint = {"type": "eq", "fun": lambda w: np.abs(w).sum() - 1}

    def neg_corr(w, *features):
        w = np.abs(w) / np.abs(w).sum()
        score = sum(f * wi for f, wi in zip(features, w))
        return -abs(np.corrcoef(score, y)[0, 1])

    # Optimize drought sub-weights
    drought_result = minimize(
        neg_corr,
        x0=[0.40, 0.35, 0.25],
        args=(
            df["max_consecutive_n"],
            df["recent_intensity_n"],
            df["frequency_n"]
        ),
        method="SLSQP",
        constraints=eq_constraint,
        bounds=bounds * 3
    )
    w_drought = np.abs(drought_result.x) / np.abs(drought_result.x).sum()

    # Optimize probability weights
    drought_score = (
        df["max_consecutive_n"]  * w_drought[0] +
        df["recent_intensity_n"] * w_drought[1] +
        df["frequency_n"]        * w_drought[2]
    )
    prob_result = minimize(
        neg_corr,
        x0=[0.60, 0.40],
        args=(drought_score, df["fire_frequency_score"]),
        method="SLSQP",
        constraints=eq_constraint,
        bounds=bounds * 2
    )
    w_prob = np.abs(prob_result.x) / np.abs(prob_result.x).sum()

    # Optimize impact weights
    impact_result = minimize(
        neg_corr,
        x0=[0.40, 0.25, 0.35],
        args=(
            df["causal_impact_score"],
            df["industry_sensitivity_n"],
            df["economic_trajectory"]
        ),
        method="SLSQP",
        constraints=eq_constraint,
        bounds=bounds * 3
    )
    w_impact = np.abs(impact_result.x) / np.abs(impact_result.x).sum()

    optimized = {
        "DROUGHT_WEIGHTS": {
            "max_consecutive_n":  round(float(w_drought[0]), 4),
            "recent_intensity_n": round(float(w_drought[1]), 4),
            "frequency_n":        round(float(w_drought[2]), 4),
        },
        "PROB_WEIGHTS": {
            "drought":        round(float(w_prob[0]), 4),
            "fire_frequency": round(float(w_prob[1]), 4),
        },
        "IMPACT_WEIGHTS": {
            "causal_impact":        round(float(w_impact[0]), 4),
            "industry_sensitivity": round(float(w_impact[1]), 4),
            "economic_trajectory":  round(float(w_impact[2]), 4),
        },
        "training_rows": len(df),
        "training_date": datetime.today().strftime("%Y-%m-%d")
    }

    import json
    with open("outputs/optimized_weights.json", "w") as f:
        json.dump(optimized, f, indent=2)

    print("\n  Optimized vs default weights:")
    print(f"  DROUGHT:  max={w_drought[0]:.3f}(was 0.40)  "
          f"recent={w_drought[1]:.3f}(was 0.35)  "
          f"freq={w_drought[2]:.3f}(was 0.25)")
    print(f"  PROB:     drought={w_prob[0]:.3f}(was 0.60)  "
          f"fire_freq={w_prob[1]:.3f}(was 0.40)")
    print(f"  IMPACT:   causal={w_impact[0]:.3f}(was 0.40)  "
          f"industry={w_impact[1]:.3f}(was 0.25)  "
          f"econ={w_impact[2]:.3f}(was 0.35)")

    return optimized
```

### Part C: Stacked Ensemble (final architecture)

After weight training, the final model architecture uses a stacked approach:
- Layer 1: existing sub-score pipeline (drought, fire freq, causal impact, etc.)
- Layer 2: ElasticNetCV learns optimal weights to combine sub-scores

```python
from sklearn.linear_model import ElasticNetCV

def build_stacked_scorer(
    sub_scores: pd.DataFrame,
    labels: pd.Series
) -> object:
    """
    Layer 2 of stacked ensemble. ElasticNet learns optimal combination
    of sub-scores to predict GDP outcomes.

    Why ElasticNet:
    - L1 zeros out sub-scores that don't predict outcomes (auto feature selection)
    - L2 prevents any single sub-score from dominating spuriously
    - CV auto-selects regularization strength
    - Coefficients ARE the learned weights — fully interpretable and citable

    The coefficients replace hardcoded weights AND provide model explainability:
    "The model assigned X% weight to drought because drought weeks has the
    highest correlation with post-fire GDP decline in our training set."
    """
    feature_cols = [
        "drought_score", "fire_frequency_score",
        "causal_impact_score", "industry_sensitivity_n",
        "economic_trajectory"
    ]
    X = sub_scores[feature_cols].fillna(
        sub_scores[feature_cols].median()
    ).values
    y = labels.values

    model = ElasticNetCV(
        l1_ratio=[0.1, 0.5, 0.7, 0.9, 1.0],
        alphas=np.logspace(-4, 0, 50),
        cv=min(5, len(X) // 5),
        max_iter=2000,
        random_state=42
    )
    model.fit(X, y)

    print("\n  Learned sub-score weights (ElasticNet coefficients):")
    for feat, coef in zip(feature_cols, model.coef_):
        direction = "+" if coef > 0 else "-"
        bar = "█" * int(abs(coef) * 20)
        print(f"    {feat:<30} {direction}{abs(coef):.4f} {bar}")

    return model
```

### Part D: Infrastructure Weight Training

The same argument that applies to pillar weights applies here. Saying "utility
hardening matters 40% and roads matter 35%" is just as arbitrary as saying
"drought matters 60%." You'd face the same question from a reviewer: *why those
numbers?*

**The key difference from pillar weight training:** Infrastructure vulnerability
is primarily a *recovery duration* signal, not a *damage severity* signal. A
county with fragile infrastructure doesn't necessarily have a worse GDP delta in
year 1 — it has a worse trajectory in years 2–5. So you cannot train against
`gdp_delta_pct` (immediate damage). You must train against recovery duration,
which your `build_recovery_trajectory()` already classifies.

**Two-pass approach required:** Infrastructure scores must be computed with
defaults first, then recovery trajectory labels become available, then weights
are trained, then infrastructure scores are recomputed with trained weights.

```python
from scipy.optimize import minimize, minimize_scalar

def train_infrastructure_weights(
    infra_sub_scores: pd.DataFrame,
    recovery_labels: pd.DataFrame
) -> dict:
    """
    Trains infrastructure component weights via constrained optimization.

    Target: recovery_score (0=fast, 1=chronic) derived from trajectory
    classification on counties with historical fire events.

    Logic: infrastructure vulnerability should predict recovery duration.
    A county with one road corridor and an aging water system takes longer
    to restart — regardless of how bad the fire itself was.

    Optimization: maximize |correlation(weighted_infra_score, recovery_score)|
    Constraints:  weights >= 0, sum = 1.0
    Bounds:       each weight between 0.10 and 0.70
                  (prevents any single component dominating)

    Falls back to defaults if < 15 training rows. Documents reason.
    """
    print("\n[Infra Weight Training] Optimizing infrastructure weights...")

    RECOVERY_SCORE = {
        "Resilient":      0.0,
        "Slow Recovery":  0.5,
        "Chronic Impact": 1.0,
        # "At Risk" excluded — no observed fire outcome to train against
    }

    df = infra_sub_scores.merge(
        recovery_labels[["fips", "trajectory"]],
        on="fips", how="inner"
    )
    df["recovery_score"] = df["trajectory"].map(RECOVERY_SCORE)
    df = df.dropna(subset=[
        "recovery_score",
        "utility_hardening_index",
        "road_redundancy_ratio",
        "water_vulnerability_score"
    ])

    if len(df) < 15:
        print(f"  WARNING: Only {len(df)} rows with recovery labels.")
        print("  Insufficient data — using defaults (0.40, 0.35, 0.25).")
        print("  To fix: lower MIN_ACREAGE or expand geo_events date range.")
        return {
            "utility_hardening":   0.40,
            "road_redundancy":     0.35,
            "water_vulnerability": 0.25,
            "training_rows":       len(df),
            "trained":             False,
            "reason":              "insufficient_data"
        }

    y = df["recovery_score"].values

    def neg_abs_corr(w):
        w_norm = np.abs(w) / np.abs(w).sum()
        score = (
            df["utility_hardening_index"]  * w_norm[0] +
            df["road_redundancy_ratio"]    * w_norm[1] +
            df["water_vulnerability_score"] * w_norm[2]
        ).values
        if np.std(score) == 0:
            return 0.0
        return -abs(np.corrcoef(score, y)[0, 1])

    result = minimize(
        neg_abs_corr,
        x0=[0.40, 0.35, 0.25],
        method="SLSQP",
        constraints={"type": "eq", "fun": lambda w: np.abs(w).sum() - 1},
        bounds=[(0.10, 0.70)] * 3
    )

    w = np.abs(result.x) / np.abs(result.x).sum()
    final_corr = -result.fun

    print(f"  Training rows:          {len(df)}")
    print(f"  Achieved correlation:   {final_corr:.3f}")
    print(f"  utility_hardening:      {w[0]:.3f}  (default: 0.40)")
    print(f"  road_redundancy:        {w[1]:.3f}  (default: 0.35)")
    print(f"  water_vulnerability:    {w[2]:.3f}  (default: 0.25)")

    if final_corr < 0.15:
        print("  WARNING: Low correlation — trained weights have low confidence.")
        print("  Weights saved but flagged. Expand data before relying on them.")

    return {
        "utility_hardening":   round(float(w[0]), 4),
        "road_redundancy":     round(float(w[1]), 4),
        "water_vulnerability": round(float(w[2]), 4),
        "training_rows":       len(df),
        "correlation":         round(float(final_corr), 4),
        "trained":             True
    }


def train_amplifier_magnitude(
    final_df: pd.DataFrame,
    causal_df: pd.DataFrame
) -> dict:
    """
    Trains the infrastructure amplifier magnitude constant.

    Current hardcoded formula:
      infrastructure_amplifier = 1.0 + (vulnerability_score × 0.35)

    The 0.35 constant is arbitrary — it says infrastructure can amplify
    impact by at most 35%. This function finds the value α that maximizes
    correlation between amplified impact score and observed GDP outcomes.

    Constraint: α between 0.10 and 0.60
    (infrastructure cannot amplify impact by less than 10% or more than 60%)

    Returns dict with trained magnitude and metadata.
    """
    print("\n[Infra Amplifier Training] Optimizing amplifier magnitude...")

    df = final_df.merge(
        causal_df[["fips", "avg_gdp_delta_pct"]],
        on="fips", how="inner"
    ).dropna(subset=[
        "avg_gdp_delta_pct",
        "impact_score",
        "infrastructure_vulnerability_score"
    ])

    if len(df) < 15:
        print(f"  WARNING: Only {len(df)} rows. Using default magnitude: 0.35")
        return {
            "magnitude": 0.35,
            "training_rows": len(df),
            "trained": False,
            "reason": "insufficient_data"
        }

    y = df["avg_gdp_delta_pct"].values

    def neg_corr_alpha(alpha):
        adjusted = (
            df["impact_score"] *
            (1.0 + df["infrastructure_vulnerability_score"] * alpha)
        ).clip(0, 1)
        # Higher adjusted impact should correlate with more negative GDP delta
        corr = np.corrcoef(adjusted, -y)[0, 1]
        return -corr

    result = minimize_scalar(
        neg_corr_alpha,
        bounds=(0.10, 0.60),
        method="bounded"
    )

    trained_alpha = round(float(result.x), 3)
    achieved_corr = round(float(-result.fun), 3)

    print(f"  Training rows:          {len(df)}")
    print(f"  Trained magnitude:      {trained_alpha:.3f}  (default: 0.350)")
    print(f"  Achieved correlation:   {achieved_corr:.3f}")

    return {
        "magnitude":      trained_alpha,
        "training_rows":  len(df),
        "correlation":    achieved_corr,
        "trained":        True
    }
```

**Loading trained weights in build_infrastructure_resilience_score():**

Replace hardcoded weight constants with a loader that reads from
`outputs/optimized_weights.json` if available, falling back to defaults:

```python
def _load_infra_weights() -> tuple:
    """
    Returns (w_utility, w_road, w_water, amplifier_magnitude).
    Loads from optimized_weights.json if trained weights exist,
    otherwise returns hardcoded defaults.
    """
    import json
    from pathlib import Path

    path = Path("outputs/optimized_weights.json")
    if path.exists():
        with open(path) as f:
            saved = json.load(f)
        iw  = saved.get("INFRASTRUCTURE_WEIGHTS", {})
        amp = saved.get("INFRASTRUCTURE_AMPLIFIER", {})
        if iw.get("trained") and amp.get("trained"):
            print(f"  [Infra] Using trained weights "
                  f"(n={iw.get('training_rows')}, "
                  f"r={iw.get('correlation', '?')})")
            return (
                iw["utility_hardening"],
                iw["road_redundancy"],
                iw["water_vulnerability"],
                amp["magnitude"]
            )
        elif iw.get("trained") and not amp.get("trained"):
            print("  [Infra] Using trained component weights, "
                  "default amplifier magnitude (0.35)")
            return (
                iw["utility_hardening"],
                iw["road_redundancy"],
                iw["water_vulnerability"],
                0.35
            )

    print("  [Infra] No trained weights found — using defaults")
    return (0.40, 0.35, 0.25, 0.35)
```

**Updated optimized_weights.json schema** (full file after P5 complete):

```json
{
  "DROUGHT_WEIGHTS": {
    "max_consecutive_n":  0.4120,
    "recent_intensity_n": 0.3380,
    "frequency_n":        0.2500,
    "training_rows": 47,
    "trained": true
  },
  "PROB_WEIGHTS": {
    "drought":        0.6200,
    "fire_frequency": 0.3800,
    "training_rows": 47,
    "trained": true
  },
  "IMPACT_WEIGHTS": {
    "causal_impact":        0.4350,
    "industry_sensitivity": 0.2280,
    "economic_trajectory":  0.3370,
    "training_rows": 47,
    "trained": true
  },
  "INFRASTRUCTURE_WEIGHTS": {
    "utility_hardening":   0.3800,
    "road_redundancy":     0.4100,
    "water_vulnerability": 0.2100,
    "training_rows": 31,
    "correlation": 0.38,
    "trained": true
  },
  "INFRASTRUCTURE_AMPLIFIER": {
    "magnitude": 0.280,
    "training_rows": 31,
    "correlation": 0.41,
    "trained": true
  },
  "training_date": "2026-04-02",
  "pipeline_version": "v8"
}
```

**The honest caveat — document this clearly:**

Infrastructure weight training requires counties that have BOTH historical fire
events AND infrastructure sub-scores (SDWIS + FHWA + EIA all populated). This
intersection may be small. If `trained: false` appears in the JSON, it means
the data wasn't sufficient and defaults were used. This is documented honestly
and is more credible than silently using trained weights from 8 data points.

When presenting to investors or reviewers:
- If `trained: true`: "Infrastructure weights were optimized against recovery
  trajectory outcomes across N counties with historical fire events."
- If `trained: false`: "Infrastructure weights use evidence-based defaults
  pending sufficient historical data. The training threshold is 15 counties
  with both fire history and complete infrastructure profiles."

### Two-Pass Call Order in main()

Infrastructure weight training requires two passes through the pipeline:

```python
# ── PASS 1: compute with defaults ──
infrastructure = build_infrastructure_resilience_score(
    fhwa, sdwis, eia_861, crosswalk, county_areas
    # uses defaults (0.40, 0.35, 0.25) on first run
)
final_pass1 = build_final_risk(probability, impact, infrastructure)

# ── Train infrastructure weights (requires trajectory labels from impact) ──
infra_weights  = train_infrastructure_weights(infrastructure, trajectory)
amplifier_meta = train_amplifier_magnitude(final_pass1, cf_detail)

# ── Train FAIR Plan and road FC weights (Part E) ──
fair_weights   = train_fair_plan_weights(fair_feat, cf_detail)
fc_weights     = train_fc_weights(infrastructure, trajectory)

# ── Save all trained weights to single JSON ──
import json
all_weights = {
    **optimized_weights,   # pillar weights from Part B
    "INFRASTRUCTURE_WEIGHTS":   infra_weights,
    "INFRASTRUCTURE_AMPLIFIER": amplifier_meta,
    "FAIR_PLAN_WEIGHTS":        fair_weights,
    "FC_WEIGHTS":               fc_weights,
    "training_date": datetime.today().strftime("%Y-%m-%d"),
    "pipeline_version": "v7"
}
with open("outputs/optimized_weights.json", "w") as f:
    json.dump(all_weights, f, indent=2)

# ── PASS 2: recompute with all trained weights ──
fair_feat      = build_fair_plan_amplifier(
    fair_raw, crosswalk, gdp,
    max_amplifier=fair_weights.get("max_amplifier", 1.20)
)
infrastructure = build_infrastructure_resilience_score(
    fhwa, sdwis, eia_861, crosswalk, county_areas
    # now reads infra + FC trained weights from optimized_weights.json
)
impact  = build_impact_score(trajectory, econ_traj, fair_plan_df=fair_feat)
final   = build_final_risk(probability, impact, infrastructure)
```

### Part E: FAIR Plan and Road FC Weight Training

Two additional sets of model parameters identified as hardcoded guesses
that require the same treatment as pillar and infrastructure weights.

**What is NOT trained (factual legal constants):**
- `RESIDENTIAL_CAP_M = 3.0` — California FAIR Plan residential policy cap ($3M)
- `COMMERCIAL_CAP_M = 20.0` — California FAIR Plan commercial policy cap ($20M)
These are set by California law, not by the model. Never train or change them.

**What IS trained:**

#### FAIR Plan MAX_AMPLIFIER and internal score weights

`MAX_AMPLIFIER = 1.20` says FAIR Plan concentration can amplify impact by
at most 20%. The internal score weights (fair_plan_share vs coverage_gap:
currently 0.60 / 0.40) are also arbitrary.

Training target: `avg_gdp_delta_pct` from counterfactual evidence — higher
FAIR Plan concentration in California counties should predict more negative
GDP outcomes post-fire (worse recovery due to underinsurance).

```python
def train_fair_plan_weights(
    fair_features: pd.DataFrame,
    causal_df: pd.DataFrame
) -> dict:
    """
    Trains two sets of FAIR Plan model parameters:

    1. Internal score weights: fair_plan_share vs coverage_gap
       (currently hardcoded as 0.60 / 0.40)
       These determine which FAIR Plan signal matters more —
       raw policy count share, or the dollar coverage gap.

    2. MAX_AMPLIFIER: maximum impact amplification (currently 1.20)
       Trained via minimize_scalar to find the ceiling that best
       predicts actual GDP deterioration in CA counties with high
       FAIR Plan concentration.

    Training target: avg_gdp_delta_pct (negative = worse outcome)
    Scope: California counties only (FAIR Plan is CA-only)

    Falls back to defaults if < 10 CA training rows available.
    """
    print("\n[FAIR Plan Weight Training] Optimizing FAIR Plan parameters...")

    # Join FAIR Plan features with observed GDP outcomes (CA only)
    df = fair_features.merge(
        causal_df[["fips", "avg_gdp_delta_pct"]],
        on="fips", how="inner"
    ).dropna(subset=[
        "avg_gdp_delta_pct",
        "fair_plan_share",
        "coverage_gap_score"
    ])

    if len(df) < 10:
        print(f"  WARNING: Only {len(df)} CA training rows.")
        print("  Using defaults: share_weight=0.60, gap_weight=0.40, max_amp=1.20")
        return {
            "share_weight":   0.60,
            "gap_weight":     0.40,
            "max_amplifier":  1.20,
            "training_rows":  len(df),
            "trained":        False,
            "reason":         "insufficient_data"
        }

    y = df["avg_gdp_delta_pct"].values  # more negative = worse outcome

    # ── Train internal score weights (share vs coverage_gap) ──
    def neg_corr_internal(w):
        w = np.abs(w) / np.abs(w).sum()
        score = (
            df["fair_plan_share"]    * w[0] +
            df["coverage_gap_score"] * w[1]
        ).values
        # Higher fair plan score should correlate with more negative GDP delta
        return -abs(np.corrcoef(score, -y)[0, 1])

    internal_result = minimize(
        neg_corr_internal,
        x0=[0.60, 0.40],
        method="SLSQP",
        constraints={"type": "eq", "fun": lambda w: np.abs(w).sum() - 1},
        bounds=[(0.20, 0.80)] * 2   # neither signal below 20%
    )
    w_internal = np.abs(internal_result.x) / np.abs(internal_result.x).sum()

    # ── Train MAX_AMPLIFIER ──
    def neg_corr_amplifier(max_amp):
        # Compute fair_plan_raw with trained internal weights
        raw_score = (
            df["fair_plan_share"]    * w_internal[0] +
            df["coverage_gap_score"] * w_internal[1]
        )
        # Amplifier at each county
        amplifier = 1.0 + (raw_score * (max_amp - 1.0))
        # This amplified score should predict GDP decline
        return -abs(np.corrcoef(amplifier, -y)[0, 1])

    amp_result = minimize_scalar(
        neg_corr_amplifier,
        bounds=(1.05, 1.50),   # amplifier between +5% and +50%
        method="bounded"
    )
    trained_max = round(float(amp_result.x), 3)
    final_corr  = round(float(-amp_result.fun), 3)

    print(f"  Training rows:       {len(df)}")
    print(f"  share_weight:        {w_internal[0]:.3f}  (default: 0.60)")
    print(f"  gap_weight:          {w_internal[1]:.3f}  (default: 0.40)")
    print(f"  max_amplifier:       {trained_max:.3f}  (default: 1.20)")
    print(f"  Achieved correlation:{final_corr:.3f}")

    return {
        "share_weight":   round(float(w_internal[0]), 4),
        "gap_weight":     round(float(w_internal[1]), 4),
        "max_amplifier":  trained_max,
        "training_rows":  len(df),
        "correlation":    final_corr,
        "trained":        True
    }
```

#### Road Functional Class Weights (FC_WEIGHTS)

The FC_WEIGHTS values (1.0, 0.7, 0.4, 0.2, 0.05) say an interstate is
20x more critical than a local road. The ordering is intuitive but the
specific magnitudes are arbitrary.

Training target: recovery trajectory — counties where higher-class road
removal causes more disconnection should have longer recovery times.

Key constraint: **ordering must be preserved after training.** Interstates
must remain more critical than arterials, which must remain more critical
than locals. This is enforced via ordering constraints in the optimizer,
not left to the data alone — the ordering is structural reality, only
the magnitudes are learned.

```python
def train_fc_weights(
    road_sub_scores: pd.DataFrame,
    recovery_labels: pd.DataFrame
) -> dict:
    """
    Trains functional class weights for road redundancy calculation.

    FC_WEIGHTS currently: {interstate: 1.0, principal: 0.7,
                           minor: 0.4, collector: 0.2, local: 0.05}

    These encode relative criticality — how much economic disruption
    results if a road of that class is cut. The ordering (higher class
    = more critical) is structural and preserved as a hard constraint.
    Only the magnitudes are learned.

    Training target: recovery_score from trajectory classification.
    Counties where road network is more fragile (fewer high-class roads
    relative to area) should have longer recovery trajectories.

    Ordering constraint: w[0] >= w[1] >= w[2] >= w[3] >= w[4]
    Interstate must always be >= principal arterial, etc.
    Interstates normalized to 1.0 after training (relative weights).

    Falls back to defaults if < 15 training rows.
    """
    print("\n[FC Weight Training] Optimizing road functional class weights...")

    RECOVERY_SCORE = {
        "Resilient":      0.0,
        "Slow Recovery":  0.5,
        "Chronic Impact": 1.0,
    }

    FC_COLS = [
        "interstate_lane_miles",
        "principal_arterial_miles",
        "minor_arterial_miles",
        "collector_miles",
        "local_miles"
    ]
    DEFAULTS = [1.0, 0.7, 0.4, 0.2, 0.05]

    df = road_sub_scores.merge(
        recovery_labels[["fips", "trajectory"]],
        on="fips", how="inner"
    )
    df["recovery_score"] = df["trajectory"].map(RECOVERY_SCORE)
    df = df.dropna(subset=["recovery_score"] + FC_COLS)

    if len(df) < 15:
        print(f"  WARNING: Only {len(df)} rows. Using defaults.")
        return {
            "interstate":         1.00,
            "principal_arterial": 0.70,
            "minor_arterial":     0.40,
            "collector":          0.20,
            "local":              0.05,
            "training_rows":      len(df),
            "trained":            False,
            "reason":             "insufficient_data"
        }

    y = df["recovery_score"].values

    def neg_corr_fc(w):
        # Normalize so interstate = 1.0 (relative weights)
        w_norm = np.abs(w) / np.abs(w[0])
        weighted = sum(
            df[col] * w_norm[i]
            for i, col in enumerate(FC_COLS)
        )
        road_density = weighted / df["area_sq_miles"].replace(0, np.nan)
        # Low density = high vulnerability = should predict high recovery_score
        vuln = 1 - (road_density - road_density.min()) / (
            road_density.max() - road_density.min() + 1e-9
        )
        if np.std(vuln) == 0:
            return 0.0
        return -abs(np.corrcoef(vuln.fillna(0).values, y)[0, 1])

    # Ordering constraints: w[i] >= w[i+1] for all i
    ordering_constraints = [
        {"type": "ineq", "fun": lambda w, i=i: w[i] - w[i+1]}
        for i in range(4)
    ]

    result = minimize(
        neg_corr_fc,
        x0=DEFAULTS,
        method="SLSQP",
        constraints=ordering_constraints,
        bounds=[(0.001, 5.0)] * 5   # all positive, interstate can go up to 5×
    )

    # Normalize so interstate = 1.0
    w_raw  = np.abs(result.x)
    w_norm = w_raw / w_raw[0]
    final_corr = round(float(-result.fun), 3)

    labels = ["interstate", "principal_arterial",
              "minor_arterial", "collector", "local"]
    defaults = [1.00, 0.70, 0.40, 0.20, 0.05]

    print(f"  Training rows:  {len(df)}")
    print(f"  Correlation:    {final_corr:.3f}")
    for lbl, val, dflt in zip(labels, w_norm, defaults):
        print(f"  {lbl:<22} {val:.3f}  (default: {dflt:.2f})")

    return {
        "interstate":         round(float(w_norm[0]), 4),
        "principal_arterial": round(float(w_norm[1]), 4),
        "minor_arterial":     round(float(w_norm[2]), 4),
        "collector":          round(float(w_norm[3]), 4),
        "local":              round(float(w_norm[4]), 4),
        "training_rows":      len(df),
        "correlation":        final_corr,
        "trained":            True
    }
```

**Loading trained FC weights in build_infrastructure_resilience_score():**

```python
def _load_fc_weights() -> dict:
    """
    Returns FC_WEIGHTS dict. Loads trained values if available,
    otherwise returns hardcoded defaults.
    """
    import json
    from pathlib import Path

    path = Path("outputs/optimized_weights.json")
    if path.exists():
        with open(path) as f:
            saved = json.load(f)
        fc = saved.get("FC_WEIGHTS", {})
        if fc.get("trained"):
            print(f"  [FC] Using trained weights "
                  f"(n={fc.get('training_rows')}, "
                  f"r={fc.get('correlation', '?')})")
            return {
                "interstate_lane_miles":    fc["interstate"],
                "principal_arterial_miles": fc["principal_arterial"],
                "minor_arterial_miles":     fc["minor_arterial"],
                "collector_miles":          fc["collector"],
                "local_miles":              fc["local"],
            }
    print("  [FC] No trained FC weights — using defaults")
    return {
        "interstate_lane_miles":    1.00,
        "principal_arterial_miles": 0.70,
        "minor_arterial_miles":     0.40,
        "collector_miles":          0.20,
        "local_miles":              0.05,
    }
```

**Updated complete optimized_weights.json schema** (all trained parameters):

```json
{
  "DROUGHT_WEIGHTS": {
    "max_consecutive_n":  0.4120,
    "recent_intensity_n": 0.3380,
    "frequency_n":        0.2500,
    "training_rows": 47,
    "trained": true
  },
  "PROB_WEIGHTS": {
    "drought":        0.6200,
    "fire_frequency": 0.3800,
    "training_rows": 47,
    "trained": true
  },
  "IMPACT_WEIGHTS": {
    "causal_impact":        0.4350,
    "industry_sensitivity": 0.2280,
    "economic_trajectory":  0.3370,
    "training_rows": 47,
    "trained": true
  },
  "INFRASTRUCTURE_WEIGHTS": {
    "utility_hardening":   0.3800,
    "road_redundancy":     0.4100,
    "water_vulnerability": 0.2100,
    "training_rows": 31,
    "correlation": 0.38,
    "trained": true
  },
  "INFRASTRUCTURE_AMPLIFIER": {
    "magnitude": 0.280,
    "training_rows": 31,
    "correlation": 0.41,
    "trained": true
  },
  "FAIR_PLAN_WEIGHTS": {
    "share_weight":   0.6500,
    "gap_weight":     0.3500,
    "max_amplifier":  1.15,
    "training_rows": 18,
    "correlation": 0.29,
    "trained": true
  },
  "NASS_AMPLIFIER": {
    "magnitude": 0.20,
    "trained": false,
    "reason": "added_post_P5 — retrain in P5 rerun once all amplifiers stable",
    "constraint": "0.05 to 0.40",
    "scope": "ag_dominant_counties_only"
  },
  "FC_WEIGHTS": {
    "S1100_lane_miles": 1.0000,
    "S1200_lane_miles": 0.5000,
    "S1400_lane_miles": 0.1000,
    "trained": false,
    "reason": "TIGER MTFCC codes — S1100=Primary, S1200=Secondary, S1400=Local",
    "constraint": "S1100 >= S1200 >= S1400 always enforced",
    "note": "P5 FC training used old FHWA keys — keys updated to TIGER after P3f"
  },
  "training_date": "2026-04-02",
  "pipeline_version": "v8"
}
```

**The honest caveat — same principle as infrastructure weights:**

If any weight group shows `trained: false`, document it. The training
threshold for FAIR Plan weights is 10 CA counties with fire history.
The threshold for FC weights is 15 counties with both road data and
trajectory labels. Below those thresholds, defaults are used and
flagged transparently.

One additional note on FC weight training: the ordering constraint
(interstates ≥ arterials ≥ collectors ≥ locals) is always enforced
even if the unconstrained optimization would produce a different result.
This is correct — the constraint encodes structural reality that the
data alone cannot invalidate with small sample sizes.

### Changes to main()
```python
# P5 additions (add after build_training_dataset()):
optimized_weights = train_pillar_weights(
    cf_detail, drought_sub_scores, fire_freq,
    industry, econ_traj
)
if optimized_weights:
    # Apply trained weights to scoring functions
    # Pass as parameters rather than modifying globals
    # Rebuild scores with trained weights
    pass

X = train_df[FEATURE_COLS].values
y = train_df[LABEL_COL].values
best_model, best_model_name, _ = select_best_model(X, y)
```

---

## P6 — ML Feature Expansion

### Updated FEATURE_COLS
Add one at a time, validate R² after each addition:
```python
# FINAL FEATURE_COLS after P6 completion (6 features kept):
FEATURE_COLS = [
    "drought_weeks_before",   # kept — strong signal (importance 17.6%)
    "unemployment_rate",      # kept — strong signal (importance 19.7%)
    "gdp_per_capita",         # kept — strong signal (importance 18.2%)
    "industry_sensitivity",   # kept — strong signal (importance 13.5%)
    "fire_frequency",         # kept — strong signal (importance 16.3%)
    "wage_base_m",            # ADDED — +0.0148 R² improvement (importance 14.7%)
]

# Features tested and DROPPED (below +0.01 R² threshold):
#   road_redundancy_ratio     +0.0007 — DROP
#   water_vulnerability_score -0.0010 — DROP
#   fiscal_resilience_score   -0.0214 — DROP (CA-only, median fill hurts nationally)
#   income_vulnerability_score -0.0084 — DROP (CA-only, same issue)
#   fair_plan_share           — SKIP (data/processed/fair_plan_county.csv
#                               was missing at time of P6; retest after P3c complete)
```

Baseline CV R²: 0.5045 (XGBoost, fresh run)
After wage_base_m: 0.5193
Final retrain (2026-04-02, P3-complete data): 0.516 ± 0.058
Training rows: 1,526 (CA: 518/34%, non-CA: 1,008/66%)
CA counties: 57/58 | non-CA counties: 260
Training R²: 0.906
Model: XGBoost saved to outputs/best_model.pkl

fair_plan_share tested post-P3c (data/processed/fair_plan_county.csv now exists):
  Delta CV R²: +0.002 → DROPPED (below 0.01 threshold)
  Reason: CA-only feature, 66% of training rows get identical median fill
  → no signal for non-CA fire events. Would need separate CA model to test properly.

Stratified CA/non-CA model — ADOPTED (2026-04-02)

Model selection on CA dataset (518 rows, 5-fold CV):
  XGBoost (6 features):          0.554 ± 0.106  ← WINNER
  Gradient Boosting (9 features): 0.518 ± 0.126
  XGBoost (9 features):           0.511 ± 0.210
  Random Forest (9 features):     0.282 ± 0.247
  Ridge (9 features):            -0.108 ± 0.153
  ElasticNet (9 features):       -0.032 ± 0.148
  Non-CA XGBoost (6 features):    0.528 ± 0.121  (baseline, before feature expansion)

Decision: CA model CV R² = 0.554 > threshold 0.526 → ADOPTED

Individual feature tests on stratified models:
  CA model (518 rows) — all 6 candidates DROP:
    fair_plan_share:            0.5501 → 0.4879  (-0.062) DROP
    fiscal_resilience_score:    0.5501 → 0.5218  (-0.028) DROP
    income_vulnerability_score: 0.5501 → 0.5107  (-0.039) DROP
    road_redundancy_ratio:      0.5501 → 0.5384  (-0.012) DROP
    water_vulnerability_score:  0.5501 → 0.5164  (-0.034) DROP
    private_well_proxy_pct:     0.5501 → 0.5034  (-0.047) DROP
  CA infrastructure features add noise — CA has homogeneous statewide coverage.

  Non-CA model — combination tests (best wins):
    6 + water + well (8 features):       0.5626 ± 0.126  (+0.035) KEEP ← BEST
    6 + water + well + road (9 features): 0.5591 ± 0.148  (+0.032)
    6 + water + road (8 features):        0.5530 ± 0.124  (+0.025)
  road_redundancy_ratio dropped — redundant once water + well included.
  Rural non-CA counties vary enormously in water/well infrastructure.

Final feature sets:
  FEATURE_COLS_CA    (6 features): drought_weeks_before, unemployment_rate,
    gdp_per_capita, industry_sensitivity, fire_frequency, wage_base_m
  FEATURE_COLS_NONCA (8 features): above 6 + water_vulnerability_score,
    private_well_proxy_pct

Saved:
  outputs/best_model_ca.pkl  ← CA XGBoost, 518 rows, 6 features,  CV R²=0.554
  outputs/best_model.pkl     ← non-CA XGBoost, 1,008 rows, 8 features, CV R²=0.563

Routing in predict_current_impact():
  if fips in CA_FIPS (6001–6115): use best_model_ca.pkl + FEATURE_COLS_CA
  else: use best_model.pkl + FEATURE_COLS_NONCA

Validation against actual historical outcomes (277 counties):
  National: Pearson r=0.626, MAE=1.68%, bias=-0.36%
  CA:       Pearson r=0.754, MAE=1.87%, bias=-1.69%
  Range compression: model predicts -12.3% to -9.7% vs actual -14.1% to -7.4%

optimized_weights.json:
  ca_model_cv_r2: 0.554, ca_model_cv_std: 0.106, ca_model_adopted: true
  nonca_model_cv_r2: 0.563, nonca_model_cv_std: 0.126
  nonca_model_features: FEATURE_COLS_NONCA (8 features)
  nonca_features_added: [water_vulnerability_score, private_well_proxy_pct]
  nonca_features_dropped: [road_redundancy_ratio] — redundant with water+well

### Validation Protocol (run after adding each feature)
1. Set `force_retrain=True` in model training call
2. Compare cross-val R² before and after — drop if R² drops > 0.02
3. Check SHAP importance — new feature should appear in top 5 to justify keeping
4. Confirm county rank ordering does not change dramatically for top/bottom 10

### Pattern for build_training_dataset() additions
Each new feature must be measured BEFORE the fire date (no data leakage).
Fall back to current value if historical value unavailable.
Fill remaining NaN with median.

```python
# Pattern for each new feature:
feature_lookup = new_dataset.set_index("fips")["feature_col"].to_dict()
df["feature_name"] = df["fips"].map(feature_lookup).fillna(
    new_dataset["feature_col"].median()
)
```

### Pattern for predict_current_impact() additions
```python
# Pattern for each new feature in current-conditions assembly:
pred = pred.merge(
    new_dataset[["fips", "feature_col"]],
    on="fips", how="left"
)
pred["feature_col"] = pd.to_numeric(pred["feature_col"], errors="coerce")
pred["feature_col"] = pred["feature_col"].fillna(pred["feature_col"].median())
```

---

## P7 — Claude API Narrative Engine

### Dependencies
```bash
pip install anthropic
export ANTHROPIC_API_KEY="your_key_here"
```

### Government Use Case Framework
The narrative engine serves two distinct audiences with different needs:

**Investors/Insurers** — need relative ranking, probability × impact scores,
portfolio exposure. Current template serves this well.

**County Governments/Emergency Managers** — need answers to three operational
questions:
  1. Should we act right now?
     → Current conditions alert: drought threshold status, probability tier,
       comparison to historical ignition conditions
  2. If a fire happens, what specifically breaks?
     → Infrastructure vulnerabilities in operational language (not scores):
       evacuation routes, water system coverage gaps, utility hardening status
     → Structure count and replacement value at risk (FEMA NSI)
  3. What does recovery cost and how long does it take?
     → Dollar loss breakdown: GDP impact + wage loss + structure replacement
     → TOT revenue loss estimate (CA counties) = fiscal year budget impact
     → Recovery trajectory classification with timeline estimate
     → Whether county fiscal reserves can absorb the shock (fiscal_resilience_score)

**FEMA HMGP grant support** — generate_fema_grant_section() produces a
Benefit-Cost Analysis justification section using:
  - probability_score as annualized risk
  - total_dollar_loss_m as avoided damages
  - FEMA NSI structure counts as at-risk assets
  - infrastructure vulnerability scores as pre-mitigation baseline
  This output can be used directly in HMGP grant applications.

### What Changes
Replace `generate_county_narrative()` with Claude API call.
Add `generate_fema_grant_section()` for HMGP BCA generation.
Both fall back to existing template behavior if API unavailable.

```python
import anthropic
import os

def generate_county_narrative(
    row: pd.Series,
    use_ai: bool = True
) -> str:
    """
    Generates county risk narrative.

    use_ai=True + ANTHROPIC_API_KEY set: calls Claude API (claude-sonnet-4-6)
    for contextual 250-300 word narrative.

    Fallback: existing template-based string building (renamed to
    _generate_narrative_template, preserves current behavior exactly).

    New fields used when available:
    - total_loss_low_m, total_loss_high_m (dollar figures from P2)
    - estimated_wage_loss_m
    - infrastructure_tier, road_redundancy_ratio (from P4)
    - water_vulnerability_score
    - fair_plan_share (from P3c)
    """
    if not use_ai or not os.getenv("ANTHROPIC_API_KEY"):
        return _generate_narrative_template(row)

    try:
        client = anthropic.Anthropic()
        context_parts = [
            f"County: {row.get('county_name')}, {row.get('state')}",
            f"Final Risk: {row.get('final_risk_tier')} "
            f"(Score: {row.get('final_risk_score', 0):.3f})",
            f"Probability: {row.get('probability_score', 0):.3f}",
            f"Impact: {row.get('impact_score', 0):.3f}",
            f"Risk quadrant: {row.get('risk_quadrant', '')}",
            f"Risk direction: {row.get('risk_direction', 'Stable')}",
        ]
        # Add dollar figures if available (from P2)
        if pd.notna(row.get("total_loss_low_m")):
            context_parts.append(
                f"Estimated total economic loss: "
                f"${row.get('total_loss_low_m', 0):.0f}M – "
                f"${row.get('total_loss_high_m', 0):.0f}M"
            )
        if pd.notna(row.get("estimated_wage_loss_m")):
            context_parts.append(
                f"Estimated wage loss: "
                f"${row.get('estimated_wage_loss_m', 0):.0f}M"
            )
        # Add infrastructure (from P4)
        if row.get("infrastructure_tier"):
            context_parts.append(
                f"Infrastructure tier: {row.get('infrastructure_tier')}"
            )
        if pd.notna(row.get("fair_plan_share")):
            context_parts.append(
                f"FAIR Plan share: {row.get('fair_plan_share', 0):.1%}"
            )
        if pd.notna(row.get("max_consecutive_weeks")):
            context_parts.append(
                f"Max consecutive drought weeks: "
                f"{int(row.get('max_consecutive_weeks', 0))}"
            )
        if pd.notna(row.get("avg_counterfactual_gap")):
            context_parts.append(
                f"Counterfactual GDP gap vs similar counties: "
                f"{row.get('avg_counterfactual_gap', 0):.1f}pp"
            )
        if row.get("top_industries"):
            context_parts.append(f"Top industries: {row.get('top_industries')}")
        if row.get("trajectory"):
            context_parts.append(
                f"Recovery trajectory: {row.get('trajectory')}"
            )

        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=600,
            messages=[{
                "role": "user",
                "content": (
                    "You are generating a wildfire economic risk narrative for a "
                    "dashboard used by insurance underwriters, municipal officials, "
                    "and ESG investors.\n\n"
                    f"COUNTY DATA:\n{chr(10).join(context_parts)}\n\n"
                    "Write a 250-300 word professional narrative covering:\n"
                    "1. Fire probability drivers (drought, historical frequency)\n"
                    "2. Economic impact if fire occurs (GDP loss, wage loss, "
                    "dollars where available)\n"
                    "3. Infrastructure and recovery risk (if data available)\n"
                    "4. Recovery outlook (trajectory classification)\n\n"
                    "End with one action-oriented sentence for insurers or "
                    "county planners. Plain professional language. No bullets. "
                    "Do not repeat raw numbers verbatim — interpret them."
                )
            }]
        )
        return message.content[0].text

    except Exception as e:
        print(f"  [AI narrative] Failed for {row.get('county_name')}: {e}")
        return _generate_narrative_template(row)


def generate_fema_grant_section(row: pd.Series) -> str:
    """
    Generates FEMA HMGP Benefit-Cost Analysis justification section.

    HMGP = Hazard Mitigation Grant Program. Counties use this to apply
    for federal wildfire mitigation funding (75% federal / 25% local match).

    FEMA's BCA tool requires exactly what our model outputs:
    - Annualized probability of significant wildfire
    - Projected losses avoided in dollar terms
    - Affected economic sectors
    - Critical infrastructure at risk

    Returns structured grant narrative with 4 required sections.
    """
    if not os.getenv("ANTHROPIC_API_KEY"):
        return "ANTHROPIC_API_KEY not set — cannot generate FEMA grant section."

    client = anthropic.Anthropic()
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1000,
        messages=[{
            "role": "user",
            "content": (
                "Generate a FEMA HMGP Benefit-Cost Analysis justification "
                "section for a wildfire hazard mitigation grant application. "
                "Use formal government grant language. Prose paragraphs only.\n\n"
                f"COUNTY: {row.get('county_name')}, {row.get('state')}\n"
                f"ANNUALIZED FIRE PROBABILITY: "
                f"{row.get('probability_score', 0):.2%}\n"
                f"PREDICTED GDP IMPACT: "
                f"{row.get('predicted_gdp_impact_pct', 0):.1f}%\n"
                f"PROJECTED LOSSES AVOIDED (annual):\n"
                f"  Low estimate:  ${row.get('total_loss_low_m', 0):.0f}M\n"
                f"  High estimate: ${row.get('total_loss_high_m', 0):.0f}M\n"
                f"WAGE LOSS AT RISK: "
                f"${row.get('estimated_wage_loss_m', 0):.0f}M\n"
                f"DOMINANT SECTORS: {row.get('top_industries', '')}\n"
                f"INFRASTRUCTURE TIER: {row.get('infrastructure_tier', 'Unknown')}\n"
                f"FAIR PLAN SHARE: {row.get('fair_plan_share', 0):.1%}\n"
                f"RECOVERY TRAJECTORY: {row.get('trajectory', 'Unknown')}\n\n"
                "Structure as:\n"
                "Section 1: Project Need and Threat Description (2 paragraphs)\n"
                "Section 2: Affected Assets and Population at Risk (1 paragraph)\n"
                "Section 3: Projected Losses Without Mitigation (1 paragraph)\n"
                "Section 4: Benefit Summary (1 paragraph)"
            )
        }]
    )
    return message.content[0].text
```

### Changes to main()
```python
# After write_report():
print("\n[FEMA] Generating grant sections for top 10 counties...")
top10 = final.nlargest(10, "final_risk_score")
fema_records = []
for _, row in top10.iterrows():
    fema_records.append({
        "county_name": row["county_name"],
        "state": row["state"],
        "final_risk_score": row["final_risk_score"],
        "fema_grant_section": generate_fema_grant_section(row)
    })
pd.DataFrame(fema_records).to_csv(
    "outputs/county_fema_grant_sections.csv", index=False
)
print("  county_fema_grant_sections.csv → FEMA grant narratives")
```

---


### Gap 4 — Tourism-Dependent County Revenue Concentration (CA)
TOT amplifier planned in P3k using SCO revenue data already downloaded.
Addresses fire closures destroying visitor seasons independent of physical damage.
Priority: implement before P7 narrative engine.

### Gap 5 — Housing Stock at Risk (National)
FEMA NSI planned in P3l.
Critical for HMGP grant applications and complete dollar loss estimates.
Addresses property tax base at risk — direct fiscal impact on county budgets.
Priority: implement before P7 narrative engine.

## P8 — Secondary Fire Area Module

### File
Create `sfa_analysis.py` as a separate module. The main pipeline runs
without it. The dashboard calls it on demand for specific fire events.

### Dependencies
```bash
# PostGIS
createdb wildfireshield
psql -d wildfireshield -c "CREATE EXTENSION postgis; CREATE EXTENSION pgrouting;"

# TIGER roads (per state — replace XX with state FIPS)
shp2pgsql -I -s 4326 tl_2023_06_roads.shp public.roads | psql -d wildfireshield

# DuckDB for Overture Maps
pip install duckdb
```

### Three SFA Components

**Component 1 — Evacuation Zone**
Buffer from fire perimeter using county OES standard tier distances:
- 1.6km (1 mile) = mandatory evacuation
- 3.2km (2 miles) = warning zone
For predictive use: California OES community wildfire protection plan tier
boundaries (public GIS data) define evacuation zones by risk tier.

**Component 2 — Smoke Plume Zone**
Directional ellipse based on wind data. NOT a circular buffer.
- Source: NOAA HRRR-Smoke model (free, 3km grid, hourly PM2.5 forecasts)
- EPA AirNow API for historical validation
- Threshold: PM2.5 > 35.5 μg/m³ (EPA "Unhealthy" threshold)
- Scale: plume extends 15× further downwind than upwind

**Component 3 — Road Access Loss Zone**
Network connectivity analysis — not a buffer.
Algorithm:
1. Load road network into pgRouting graph
2. Mark road segments intersecting fire/evacuation zone as impassable
3. For each business cluster, test if route exists to nearest commercial center
4. Disconnected nodes = access loss zone
Key metric: betweenness centrality identifies critical corridors
(road segments whose removal disconnects the most economic activity)

**SFA Formula**
```python
sfa_combined = union(evacuation_zone, smoke_zone, access_loss_zone)
sfa_final    = sfa_combined.difference(fire_perimeter)
```

### Key Finding to Communicate
LA 2025: businesses in SFA generated $6.8B/yr annual revenue vs $1.4B in
direct burn area. The indirect economic damage from road closures and access
loss was 5× the direct burn damage. No current competitor model prices this.

### Data Sources
```
NIFC WFIGS perimeters:
  https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/
  WFIGS_Interagency_Perimeters/FeatureServer

Census TIGER roads:
  https://census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html
  → All Roads → per state shapefile

Overture Maps POIs (via DuckDB):
  INSTALL httpfs; LOAD httpfs;
  SELECT id, names.primary, categories.primary, addresses, geometry
  FROM read_parquet('s3://overturemaps-us-west-2/release/2024-07-22/theme=places/type=place/*')
  WHERE bbox.xmin BETWEEN -124.5 AND -114.0  -- California
  AND bbox.ymin BETWEEN 32.5 AND 42.0

Census LODES WAC (jobs by block):
  https://lehd.ces.census.gov/data/lodes/LODES8/{state}/wac/
  {state}_wac_S000_JT00_{year}.csv.gz
```

---

## Updated main() — Complete Target Flow

```python
def main():
    print("=" * 60)
    print("  Wildfire Economic Risk Pipeline  v8")
    print(f"  {REFERENCE_DATE.strftime('%Y-%m-%d')}  |  λ={DECAY_RATE}")
    print("  Final Risk = Probability × Impact × Infrastructure")
    print("=" * 60)

    # Load base data
    gdp, foia, geo, unem, drought = load_data()
    crosswalk = load_county_crosswalk()                           # P1

    # Probability layer (unchanged)
    drought_score = build_drought_score(drought)
    fire_freq     = build_fire_frequency_score(geo, gdp)
    probability   = build_probability_score(drought_score, fire_freq)

    # Impact layer
    prepost           = build_prepost_windows(geo, gdp)
    causal, cf_detail = build_counterfactual(prepost, gdp, unem)
    industry          = build_industry_sensitivity(foia, gdp)

    # P3b — BLS QCEW wage data
    qcew         = pd.read_csv("data/raw/bls_qcew_county_annual.csv")

    # P3i — SCO fiscal resilience (CA only)
    sco_rev      = pd.read_csv("data/raw/sco_county_revenues.csv")
    sco_exp      = pd.read_csv("data/raw/sco_county_expenditures.csv")
    sco_city_rev = pd.read_csv("data/raw/sco_city_revenues.csv")   # SF only
    sco_city_exp = pd.read_csv("data/raw/sco_city_expenditures.csv")
    fiscal       = build_fiscal_resilience_score(
        sco_rev, sco_exp, crosswalk,
        city_revenues=sco_city_rev, city_expenditures=sco_city_exp
    )

    # P3j — FTB AGI income vulnerability (CA only)
    ftb_agi      = pd.read_csv("data/raw/ftb_agi_by_county.csv")
    income       = build_income_vulnerability_score(ftb_agi, crosswalk)

    # build_economic_trajectory uses all three CA sources where available
    econ_traj    = build_economic_trajectory(
        unem, gdp, qcew=qcew, fiscal_df=fiscal, income_df=income
    )

    # P3c — CA FAIR Plan underinsurance amplifier
    fair_raw     = pd.read_csv("data/processed/fair_plan_county.csv")
    fair_feat    = build_fair_plan_amplifier(fair_raw, crosswalk, gdp)

    # P3d — USDA NASS agricultural land value amplifier
    nass_raw     = pd.read_csv("data/raw/usda_nass_land_values.csv")
    nass_feat    = build_usda_nass_amplifier(nass_raw, crosswalk)

    trajectory   = build_recovery_trajectory(causal, industry, gdp)

    impact       = build_impact_score(
        trajectory, econ_traj,
        fair_plan_df=fair_feat,   # P3c
        nass_df=nass_feat         # P3d
    )

    # Infrastructure layer
    # TIGER roads: pre-process into county × MTFCC summary CSV before main()
    # See build_tiger_road_summary() in P3f
    tiger_roads  = pd.read_csv("data/processed/tiger_road_miles_by_county.csv")  # P4
    sdwis        = pd.read_csv("data/raw/sdwis_violations.csv")
    eia_861      = pd.read_excel("data/raw/eia_form861_utility.xlsx")
    gaz          = pd.read_csv("data/raw/2023_Gaz_counties_national.txt",
                               sep="\t", dtype={"GEOID": str})
    gaz["fips"]          = gaz["GEOID"].astype("Int64")
    gaz["area_sq_miles"] = gaz["ALAND"] / 2_590_000
    county_areas         = gaz[["fips", "area_sq_miles"]]

    infrastructure = build_infrastructure_resilience_score(               # P4
        tiger_roads, sdwis, eia_861, crosswalk, county_areas
    )

    # Final risk (updated to accept infrastructure)
    final = build_final_risk(probability, impact, infrastructure)         # P4

    # Dollar estimates
    final = add_dollar_loss_estimate(final, gdp)                          # P2
    final = add_wage_loss_estimate(final)                                 # P2

    # Weight training
    optimized_weights = train_pillar_weights(                             # P5
        cf_detail, drought_score, fire_freq, industry, econ_traj
    )

    # ML layer — model selection
    train_df       = build_training_dataset(
        cf_detail, drought, unem, gdp, industry, fire_freq,
        asset_features=asset_feat, wage_features=econ_traj,
        fair_features=fair_feat, infra_features=infrastructure
    )
    X = train_df[FEATURE_COLS].values
    y = train_df[LABEL_COL].values
    best_model, best_name, _ = select_best_model(X, y)                   # P5
    ml_predictions = predict_current_impact(
        best_model, probability, drought, unem, gdp, industry, fire_freq,
        asset_features=asset_feat, wage_features=econ_traj,
        fair_features=fair_feat, infra_features=infrastructure
    )

    if not ml_predictions.empty:
        final = final.merge(
            ml_predictions[[
                "fips", "predicted_gdp_impact_pct",
                "prediction_ci_low", "prediction_ci_high",
                "confidence_flag", "top_shap_driver", "shap_breakdown"
            ]],
            on="fips", how="left"
        )

    # Outputs
    probability.to_csv("outputs/county_probability_scores.csv", index=False)
    impact.to_csv("outputs/county_impact_scores.csv", index=False)
    final.to_csv("outputs/county_final_risk_scores.csv", index=False)
    cf_detail.to_csv("outputs/county_causal_evidence.csv", index=False)
    if not ml_predictions.empty:
        ml_predictions.to_csv("outputs/county_ml_predictions.csv", index=False)

    # Rural county flags (added post-P6)
    # is_rural, population_density, rural_industry_source,
    # rural_gdp_data_quality, rural_training_coverage,
    # rural_prediction_confidence, confidence_note
    # prediction_ci_low_adj, prediction_ci_high_adj
    # (all computed in build_final_risk() from county_areas + cf_detail)

    write_report(probability, impact, final, cf_detail)                   # P7

    # FEMA grant sections
    top10 = final.nlargest(10, "final_risk_score")                        # P7
    pd.DataFrame([{
        "county_name": r["county_name"], "state": r["state"],
        "fema_grant_section": generate_fema_grant_section(r)
    } for _, r in top10.iterrows()]).to_csv(
        "outputs/county_fema_grant_sections.csv", index=False
    )

    print("\nDone! ✓")
```
