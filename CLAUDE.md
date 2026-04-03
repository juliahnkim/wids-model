# Wildfire Shield — Project Brief for Claude Code

## What This Is

Wildfire Shield is a predictive economic risk platform built by DataHuskies
(Northeastern University, Khoury College of Computer Sciences). It identifies
which U.S. counties are most economically vulnerable to wildfire — before the
fire starts.

The platform produces county-level risk scores for governments, insurers, and
ESG investors. The core output is not "where will fires burn" but "what happens
to the economy when they do."

**This model measures economic output disruption, NOT residential property loss.**
Do not use Zillow or residential home price data anywhere in this pipeline.

---

## Current Codebase

### Entry Point
`Wildfire_Risk_Analysis_1.py` — Python pipeline, currently v10

### Current Data Files (do not modify paths without instruction)
- `County Quarterly GDP Estimate.xlsx` — quarterly county GDP + population
- `FOIA_2019_PRESENT_with_borrcounty.xlsx` — SBA loan records with NAICS codes
- `geo_events_geoevent_with_county.csv` — wildfire events with acreage + FIPS
- `Unemployment Data 2025.csv` — county unemployment rates by year (long format)
- `Drought_Monitor_Data_20192023_1.csv` — drought episodes with consecutive weeks

### Pipeline Architecture (current)

```
load_data() → 19 datasets
    │
    ├── PROBABILITY LAYER
    │   build_drought_score()               # drought × 0.60
    │   build_fire_frequency_score()        # fire frequency × 0.40
    │   build_probability_score()           # inner join on FIPS
    │
    ├── IMPACT LAYER
    │   build_prepost_windows()             # 4-quarter pre/post GDP windows
    │   build_counterfactual()              # cosine similarity, N=5 controls
    │   build_industry_sensitivity()        # FOIA → CBP → BEA fallback chain
    │   build_recovery_trajectory()         # Resilient/Slow/Chronic/At Risk
    │   build_economic_trajectory()         # unem trend + GDP trend + wages
    │   build_impact_score()               # causal×0.40 + industry×0.25 + econ×0.35
    │
    ├── INFRASTRUCTURE LAYER (P4)
    │   build_ca_water_vulnerability()      # CA SWRCB SAFER + facilities (CA only)
    │   build_infrastructure_resilience_score()  # utility + road + water
    │
    ├── FINAL RISK
    │   build_final_risk()                  # normalize(probability × impact_adjusted)
    │                                       # impact_adjusted = impact × infra_amplifier
    │
    └── ML LAYER
        build_training_dataset()            # historical fire events, pre-fire features
        select_best_model()                 # RF vs XGBoost vs GBM vs Ridge vs ElasticNet
        predict_current_impact()            # current conditions → GDP impact %
```

### Key Config Values (frozen — all confirmed by sensitivity analysis)
- `DECAY_RATE = 0.3` — confirmed, no change (tested 0.1–0.5)
- `PRE_POST_QTR = 4` — confirmed, no change (tested 2–8)
- `N_CONTROLS = 5` — confirmed, no change (tested 3–10)
- `MIN_ACREAGE = 50` — changed from 100 (tested 50–1000, +39% training rows)
- Final score formula: `probability × impact` (multiplication, not addition — intentional)
- Tier cut points: `[0, 0.25, 0.50, 0.75, 1.0]` — confirmed, no change (tested 4 schemes)

### Current ML Features
```python
# CA model (6 features)
FEATURE_COLS = [
    "drought_weeks_before",    # drought weeks in 52 weeks before fire
    "unemployment_rate",       # unemployment rate at time of fire
    "gdp_per_capita",          # county GDP per capita at time of fire
    "industry_sensitivity",    # NAICS fragility score (0–1)
    "fire_frequency",          # historical fires per 100k population
    "wage_base_m",             # BLS QCEW total annual wages ($M) — added P6
]
# Non-CA model (8 features) — adds water infrastructure features
FEATURE_COLS_NONCA = FEATURE_COLS + [
    "water_vulnerability_score",  # combined federal/CA water vulnerability
    "private_well_proxy_pct",     # ACS B25049 private well proxy
]
LABEL_COL = "counterfactual_gap"  # GDP delta vs matched controls post-fire
                                   # (was gdp_delta_pct — fixed in P2.5)
```

### Known Issues (resolved)
1. ~~County name string joins~~ — Fixed in P1 (FIPS crosswalk)
2. ~~dominant_sector silently dropped~~ — Fixed in P1
3. ~~No dollar figures~~ — Fixed in P2 (UCLA Anderson multiplier)
4. ~~Hardcoded weights~~ — Fixed in P5 (optimized_weights.json with SLSQP training)
5. **CBP API call in `_naics_from_cbp()`** is live HTTP — can fail silently.

### Current Outputs (7 files)
```
county_probability_scores.csv    — fire likelihood per county
county_impact_scores.csv         — economic damage severity if fire occurs
county_final_risk_scores.csv     — probability × impact + ML predictions
county_ml_predictions.csv        — predicted GDP impact + SHAP
county_causal_evidence.csv       — pre/post + counterfactual fire event detail
best_model.pkl / best_model_ca.pkl — stratified XGBoost models (non-CA / CA)
risk_summary_report.txt          — shareholder narrative (template-based, pre-AI)
```

---

## Planned Enhancements

See `MODEL_CHANGES.md` for full implementation specification including exact
function signatures, formulas, and validation steps.

### Priority Order

**P1 — County Crosswalk (do first, before any new data)**
Fix the string join problem. Download `national_county2020.txt` from Census ANSI.
Build `county_crosswalk.csv`. Replace all county name string joins with FIPS joins.
Also fix: add `dominant_sector` and `industry_source` to merge in
`build_recovery_trajectory()` — one-line bug fix, blocks P3d.

**P2 — Dollar Loss Estimates (no new data needed)**
Add `add_dollar_loss_estimate()` and `add_wage_loss_estimate()` after
`build_final_risk()`. Uses UCLA Anderson methodology (2.2–2.8x multiplier).

**P2.5 — Training Label Fix**
Switch `LABEL_COL` from `gdp_delta_pct` to `counterfactual_gap`.
Raw GDP delta includes macro noise. Counterfactual gap isolates fire-attributable
impact. Training set grew from 1 row to 1,526 rows after FIPS crosswalk fix.

**P3 — New Data Sources**
| ID | Dataset | Source | Feeds | Scope |
|---|---|---|---|---|
| 3a | BEA CAINC5N | bea.gov → Regional → CAINC5N | Industry earnings mix | National |
| 3b | BLS QCEW | bls.gov/cew → County High-Level | Wage base + trend | National |
| 3c | CA FAIR Plan | cfpnet.com → Key Statistics (PDF) | Underinsurance amplifier | CA only |
| 3d | USDA NASS | nass.usda.gov → Quick Stats → Land Values | Ag county damage | National |
| 3e | EPA SDWIS/ECHO | echo.epa.gov → SDWA Download | Water vuln (non-CA) | Non-CA |
| 3f | FHWA HM-71 | fhwa.dot.gov → Highway Statistics | Road redundancy | National |
| 3g | EIA Form 861 | eia.gov/electricity/data/eia861 | Utility hardening | National |
| 3h | Census Gazetteer | census.gov → Gazetteer Files | County area (sq miles) | National |
| 3i | SCO Revenues + Expenditures | bythenumbers.sco.ca.gov | Fiscal resilience | CA only |
| 3j | FTB B-7 AGI | data.ca.gov → FTB B-7 report | Income vulnerability | CA only |
| — | CA SWRCB SAFER | data.ca.gov → SAFER dashboard | Water vuln (CA) | CA only |
| — | CA SWRCB Facilities | data.ca.gov → DWW facilities | Surface water dep (CA) | CA only |
| — | CPUC HFTD | ArcGIS REST API | Utility hardening (CA) | CA only |
| — | FEMA NRI | hazards.fema.gov → NRI | Validation only | National |

**P4 — Infrastructure Resilience Score (new function)**
`build_infrastructure_resilience_score()` combining:
- Utility hardening index × 0.40
  - CA: CPUC HFTD tier exposure (0.60) + EIA customer density (0.40)
  - Non-CA: EIA customer density only
- Road redundancy ratio × 0.35 (FHWA HM-71)
- Water vulnerability score × 0.25
  - CA: build_ca_water_vulnerability() using SAFER + facilities files
  - Non-CA: federal SDWIS where available, median fill elsewhere

All component weights load from `optimized_weights.json`. See weight registry
in MODEL_CHANGES.md. Infrastructure plugs into final risk as amplifier:
`impact_score_adjusted = impact_score × (1.0 + vulnerability × α)`
where α = INFRASTRUCTURE_AMPLIFIER (default 0.35, trained in P5).

**P5 — Model Selection + Weight Training**
Replace Random Forest with model comparison (RF vs XGBoost vs Gradient Boosting
vs Bayesian Ridge vs ElasticNet). Select best by cross-val R².
Train ALL weights via constrained SLSQP optimization. See complete weight
registry in MODEL_CHANGES.md — every weight loads from optimized_weights.json.

**P6 — ML Feature Expansion**
After P3–P5 complete, add to FEATURE_COLS one at a time, validate R² after each:
- `gdp_weighted_sensitivity` (BEA CAINC5N)
- `wage_base_m` (BLS QCEW)
- `fair_plan_share` (CA only, median fill non-CA)
- `road_redundancy_ratio` (FHWA)
- `water_vulnerability_score` (SDWIS / CA SWRCB)
- `fiscal_resilience_score` (SCO — CA only)
- `income_vulnerability_score` (FTB — CA only)

**P7 — Claude API Narrative Engine**
Replace template-based `generate_county_narrative()` with Anthropic API call.
Add `generate_fema_grant_section()` for HMGP Benefit-Cost Analysis generation.
Model: `claude-sonnet-4-6`. Requires `ANTHROPIC_API_KEY` env var.

**P8 — Secondary Fire Area Module (separate file: `sfa_analysis.py`)**
Requires PostGIS, pgRouting, DuckDB. Optional — main pipeline runs without it.
Rural county flag: counties below 50 persons/sq mile where fire exceeds 10%
of land area get a rural_coverage_ratio multiplier on SFA impact.
Key finding: LA 2025 — SFA businesses generated $6.8B/yr vs $1.4B in burn area.

---

## Complete Weight Registry

Every weight in the model loads from `optimized_weights.json` with hardcoded
defaults as fallback. No weight is a hardcoded constant in any function body.

| Key | Components | Constraint | Scope |
|---|---|---|---|
| DROUGHT_WEIGHTS | max_consecutive_n, recent_intensity_n, frequency_n | sum=1 | National |
| PROB_WEIGHTS | drought, fire_frequency | sum=1 | National |
| IMPACT_WEIGHTS | causal_impact, industry_sensitivity, econ_trajectory | sum=1 | National |
| INFRASTRUCTURE_WEIGHTS | utility_hardening, road_redundancy, water_vulnerability | sum=1 | National |
| INFRASTRUCTURE_AMPLIFIER | magnitude α | 0.10–0.60 | National |
| FC_WEIGHTS | interstate, principal_arterial, minor_arterial, collector, local | ordering enforced | National |
| FAIR_PLAN_WEIGHTS | share_weight, gap_weight, max_amplifier | — | CA only |
| CA_WATER_WEIGHTS | system_risk_score, surface_water_dependency, system_fragmentation | sum=1, each 0.10–0.70 | CA only |
| CA_UTILITY_HARDENING_WEIGHTS | hftd_exposure, customer_density | hftd >= customer_density | CA only |
| HFTD_TIER_WEIGHTS | tier2, tier3 | tier3 >= tier2 | CA only |
| FISCAL_RESILIENCE_WEIGHTS | property_tax_dependence, fiscal_surplus_ratio, debt_service_burden | sum=1 | CA only |
| INCOME_VULNERABILITY_WEIGHTS | low_income_share, income_concentration, agi_per_return | sum=1 | CA only |
| ECONOMIC_TRAJECTORY_WEIGHTS_CA | unem_n, unem_trend_n, gdp_trend_n, wage_base_n, wage_trend_n, fiscal_resilience_score | sum=1 | CA only |
| ECONOMIC_TRAJECTORY_WEIGHTS_CA_FULL | all above + income_vulnerability_score | sum=1 | CA only, requires SCO+FTB |

---

## Business Context

### Target Customers
1. **Insurance carriers and reinsurers** — pay $50K+ for catastrophe modeling.
   Need economic damage by industry sector, not just property loss.
2. **Municipal governments** — applying for FEMA HMGP grants need quantified
   risk exposure. Our outputs directly populate the required BCA sections.
3. **ESG investors and lenders** — need industry sensitivity layer for geographic
   credit exposure to fire-prone regions.

### FEMA HMGP Context
HMGP = Hazard Mitigation Grant Program. Federal grants (75% federal / 25% local
match) for wildfire mitigation. Requires Benefit-Cost Analysis with:
- Annualized probability of significant wildfire
- Projected losses avoided in dollar terms
- Affected economic sectors and assets
- Critical infrastructure at risk
Our model outputs exactly these numbers. The FEMA grant report generator (P7)
auto-populates required BCA sections.

### Key Differentiators vs Competitors
- Only platform separating probability from impact and multiplying them
- Counterfactual matching removes macroeconomic noise from GDP impact measurement
- Industry sensitivity via NAICS — ag county vs finance county get different scores
- Infrastructure resilience score separates damage severity from recovery duration
- CA-specific data (SWRCB, CPUC HFTD, FAIR Plan, SCO, FTB) for primary market
- Secondary Fire Area quantifies indirect disruption beyond burn perimeter
- FEMA grant report generator — direct revenue from municipal segment

### Validation Anchor (use for every demo)
LA County 2025: UCLA Anderson Forecast independently measured -0.48% GDP / $4.6B
post-fire. Model predicts -12.01% for significant hypothetical wildfire scenario
(correct direction; magnitude reflects conditional expected damage).

---

## Methodology Notes

### Why Multiplication Not Addition
`final_risk = probability × impact` (not summed). A county certain to burn but
economically resilient does NOT score Critical. A fragile economy with zero fire
history does NOT score Critical. Only HIGH on BOTH axes reaches Critical tier.

### UCLA Anderson Loss Multiplier
Total economic loss = insured loss × 2.2 to 2.8.
- 2.2 = California wildfire historical baseline (Camp/Woolsey/Tubbs/Thomas fires)
- 2.8 = Hurricane Katrina infrastructure ratio (used when infra damage is severe)

### CA-Specific Data Sources
Several components use California-specific data because CA is the primary market
and federal sources have poor CA coverage (federal SDWIS misses 47/58 CA counties).
All CA-specific features are flagged in output columns with `_available` boolean.
Non-CA counties get median fill with `source = "estimated"` flags.
- Water vulnerability: CA SWRCB SAFER + facilities (58/58 CA counties)
- Utility hardening: CPUC HFTD tier map (58/58 CA counties)
- Underinsurance: CA FAIR Plan county exposure (56/58 CA counties)
- Fiscal resilience: SCO county revenues/expenditures (58/58 CA counties)
- Income vulnerability: FTB B-7 AGI by county (58/58 CA counties)

### FAIR Plan Is California-Only
FAIR Plan amplifier applies only to California counties. Non-CA counties get
`fair_plan_available = False` and `fair_plan_amplifier = 1.0`.
Residential cap: $3M. Commercial cap: $20M (legal constants, never trained).

### No Residential Data
Do NOT use: Zillow, CoreLogic, residential property value.
Use instead: BEA CAINC5N (industry earnings), BLS QCEW (wage base),
USDA NASS (farmland value), SCO revenues (fiscal capacity).

### Rural County Handling
Rural counties (< 50 persons/sq mile) are handled with more assumptions and
less empirical grounding. Key gaps and fixes documented in MODEL_CHANGES.md:
- Gap 1: Industry sensitivity — ag/timber counties may fall through to median fill
- Gap 2: Private well exposure — SDWIS/SAFER miss domestic well users
- Gap 3: SFA logic — single fire can encompass entire rural county economy
- Gap 4: Tourism-dependent counties — TOT revenue share as sectoral amplifier
All rural counties get `is_rural` flag and `rural_prediction_confidence` in output.

### Weight Training Rationale
All weights are trained via scipy SLSQP constrained optimization using
counterfactual GDP evidence as ground truth. Weights are saved to
`optimized_weights.json` with `trained: true/false` and `training_rows` metadata.
Default values are used until training runs in P5.

---

## File Structure

### Target
```
/project
  CLAUDE.md                                ← this file (read every session)
  MODEL_CHANGES.md                         ← implementation spec (read before coding)
  QUERIES.md                               ← PostGIS + DuckDB reference queries
  Wildfire_Risk_Analysis_1.py              ← main pipeline (modify in place)
  sfa_analysis.py                          ← new: Secondary Fire Area module (P8)
  county_crosswalk.csv                     ← new: FIPS → canonical county name map
  data/
    raw/
      national_county2020.txt              ← Census ANSI county codes
      bls_qcew_county_annual.csv           ← BLS QCEW wage data
      usda_nass_land_values.csv            ← farmland values
      fhwa_hm71_lane_miles.xlsx            ← road lane miles by county
      eia_form861_service_territory.xlsx   ← utility → county mapping
      eia_form861_sales_ult_cust.xlsx      ← utility customer counts
      2024_Gaz_counties_national.txt       ← county areas (sq miles)
      ca_swrcb_safer.csv                   ← CA water system risk (SAFER)
      ca_swrcb_pws_facilities.csv          ← CA water system facilities
      sco_county_revenues.csv              ← CA county fiscal revenues
      sco_county_expenditures.csv          ← CA county fiscal expenditures
      ftb_agi_by_county.csv                ← CA income distribution by bracket
      sdwis/                               ← EPA federal SDWIS (non-CA)
        SDWA_PUB_WATER_SYSTEMS.csv
        SDWA_VIOLATIONS.csv
    processed/
      fair_plan_county.csv                 ← extracted from FAIR Plan PDFs
  outputs/
    county_probability_scores.csv
    county_impact_scores.csv
    county_final_risk_scores.csv
    county_ml_predictions.csv
    county_causal_evidence.csv
    county_fema_grant_sections.csv         ← new (P7)
    best_model.pkl                         ← replaces wildfire_rf_model.pkl (P5)
    model_comparison_results.csv           ← new (P5)
    optimized_weights.json                 ← all trained weights (P5)
    risk_summary_report.txt
```

---

## Session Start Instructions

Every Claude Code session should begin with:
```
Read CLAUDE.md and MODEL_CHANGES.md completely before doing anything.
State which priority (P1–P8) we are implementing today.
Confirm what you will change, in which function, and what validation you will run.
Do not write any code until I confirm your plan.
```

Work one priority at a time. Confirm output county counts match baseline
(879 counties) before moving to next priority. If counts drop after a change,
a join broke — fix before proceeding.

---

## Status Tracking

### Completed
- P1: County crosswalk (FIPS joins, dominant_sector fix, training set 1→1,526 rows)
- P2: Dollar loss estimates (UCLA Anderson multiplier, wage loss)
- P2.5: Training label fix (counterfactual_gap replaces gdp_delta_pct)
- P3a: BEA CAGDP2 GDP-weighted industry sensitivity
- P3b: BLS QCEW wage base + wage growth trend
- P3c: CA FAIR Plan underinsurance amplifier
- P3d: USDA NASS agricultural land value amplifier
       (54 ag-dominant counties in 879 set, amplifier range 1.012–1.155,
        Glenn CA highest at $10,262/acre, magnitude from optimized_weights.json,
        879 baseline held)
- P3e: EPA SDWIS non-CA water vulnerability + CA SWRCB SAFER
       + ACS B25049 private well proxy (Gap 2 fix, weight=0.15,
        113 counties with note, rural mean 1.66x urban)
- P3f: Census TIGER/Line Roads 2024 road redundancy
- P3g: EIA Form 861 utility hardening (HFTD blend for 58 CA counties)
- P3h: Census Gazetteer county areas
- P3i: SCO county revenues + expenditures fiscal resilience
       (58/58 CA counties, ig_penalty implemented, SF from city dataset)
- P3j: FTB B-7 AGI income vulnerability
       (58/58 CA counties, r=-0.458 with fiscal — complementary not redundant)
- P3k: SCO TOT tourism amplifier (CA only)
       (57/58 CA counties, tot_share 0.000–0.098, amplifier 1.000–1.200,
        Mariposa highest at 0.098 (Yosemite gateway), clip-only pattern,
        tier impact: +1 High, -1 Moderate, 879 baseline held)
- P3: ALL DATA SOURCES COMPLETE (P3a–P3k + FAIR Plan res/com split + private well proxy)
- P4: Infrastructure resilience score — all 3 pillars complete
       (std=0.157, range=0.885, 879 baseline held)

- P5: Model selection + weight training
       (XGBoost selected, DROUGHT+PROB weights trained,
        IMPACT weights via QCEW employment delta alt label)

- P6: ML feature expansion + stratified model deployment
       (wage_base_m kept, fair_plan_share tested +0.002 → dropped;
        IMPACT_WEIGHTS trained: causal=0.06, industry=0.45, econ=0.49;
        stratified CA/non-CA models adopted;
        CA-specific features all tested and dropped — hurt even within CA;
        predict_current_impact() routes by FIPS 6001–6115;
        non-CA expanded to 8 features: +water_vulnerability_score, +private_well_proxy_pct;
        MIN_ACREAGE lowered 100→50, +39% training rows, both models improved;
        final: CA R²=0.629 (792 rows, 6 feat), non-CA R²=0.589 (1324 rows, 8 feat))

### Pipeline Status
- Version: v10
- Counties: 879
- MIN_ACREAGE: 50
- Model: Stratified XGBoost — CA R²=0.629 (792 rows, 6 feat), non-CA R²=0.589 (1324 rows, 8 feat)
- Ready for: P7 (narrative engine), P8 (SFA module), dashboard integration

### In Progress

### Not Started
- P7: Claude API narrative engine
- P8: Secondary Fire Area module
