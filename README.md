# NYC Night Signals Atlas (2021-2023)

A data-driven analysis of nighttime disturbance patterns across New York City, using 311 noise complaint data to identify behavioral profiles, temporal stability, and concentrated hotspots.

**[View the Project Website](https://yoelplutchok.github.io/Sleep_ESI_NYC/)**

---

## What This Project Is

This project maps **nighttime 311 noise complaints** (10pm-7am) across NYC's 59 Community Districts and 197 residential Neighborhood Tabulation Areas, producing:

- **Behavioral profiles** ("night types") based on complaint composition, timing, and intensity
- **Stability overlays** showing which patterns persist vs. change year-to-year
- **Hotspot layers** with artifact flagging to distinguish genuine clusters from data anomalies
- **Heat sensitivity analysis** linking nighttime temperature to complaint rates
- **Environmental Sleep Index** combining objective exposure data (noise, light, heat, air quality)

All outputs are reproducible, versioned, and designed for exploratory visualization.

---

## What This Project Is NOT

- **Not a public health assessment.** Profiles describe complaint patterns, not noise exposure or sleep outcomes.
- **Not a land-use classification.** A "Nightlife & Entertainment" profile reflects complaint behavior, not zoning.
- **Not a ranking of "bad" neighborhoods.** High complaint rates may reflect reporting culture, not lived experience.
- **Not causal.** Correlations with demographics or outcomes are descriptive only.

---

## Key Statistics

| Metric | Value |
|--------|-------|
| Total 311 Complaints | 2,194,187 |
| Nighttime Complaints (22:00-07:00) | 405,226 |
| Community Districts | 59 |
| Behavioral Clusters | 9 |
| Complaint Rate Disparity | 27x (highest vs lowest CD) |
| Persistent Hotspots (all 3 years) | 19 cells |

---

## Core Analytical Components

### 1. Night Typology (Profiles)

Clusters of areas with similar complaint patterns (K-Means, K=9):

| Level | File | Rows |
|-------|------|------|
| CD | `data/processed/reports/cd_types_with_stability.geojson` | 59 |
| NTA | `data/processed/reports/nta_types_with_stability_residential.geojson` | 197 |

**Key columns:**
- `cluster_label_short` - Human-readable profile name
- `stability_class2` - "Strict Structural", "Mostly Structural", or "Episodic"
- `is_outlier_cluster` - True for singleton clusters (interpret with caution)

### 2. Hotspot Layers

Grid-based concentration of complaints (250m cells):

| File | Description |
|------|-------------|
| `data/processed/hotspots/hotspot_cells_ge50.geojson` | >=50 complaints (map-grade) |
| `data/processed/hotspots/hotspot_cells_ge10.geojson` | >=10 complaints (analysis-grade) |
| `data/processed/reports/hotspot_persistent_all3.geojson` | Cells in top 100 all 3 years (19 cells) |

**Key columns:**
- `complaint_count` - Total complaints 2021-2023
- `is_suspected_artifact` - True if >=95% at one coordinate OR >=200 in one day
- `is_repeat_location_dominant` - True if >=90% at one coordinate

### 3. Stability Analysis

How consistent are profiles across 2021, 2022, 2023?

| File | Description |
|------|-------------|
| `data/processed/reports/cd_stability.geojson` | CD-level stability metrics |
| `data/processed/reports/nta_stability_residential.geojson` | NTA-level stability metrics |

**Stability classes:**
- **Strict Structural** (entropy = 0): Same cluster all 3 years
- **Mostly Structural** (entropy <= 0.6, stable 2 of 3): Reliable pattern
- **Episodic** (otherwise): Volatile, interpret cautiously

### 4. Heat Sensitivity

Temperature-complaint relationship modeled via Poisson GLM:

| File | Description |
|------|-------------|
| `data/processed/weather/cd_heat_sensitivity.parquet` | Per-CD slopes and standard errors |
| `data/processed/weather/citywide_temp_curve.csv` | Citywide temperature-response curve |

**Finding:** Each 1C increase in nighttime minimum temperature is associated with a ~2.3% increase in complaints citywide.

---

## How to Explore the Results

### Option A: Kepler.gl (Recommended)

1. Go to [kepler.gl](https://kepler.gl/demo)
2. Drag and drop any `.geojson` file from `data/processed/reports/`
3. Color by `cluster_label_short` or `stability_class2`
4. Filter by `is_mostly_structural = true` to see reliable patterns

### Option B: Mapshaper

1. Go to [mapshaper.org](https://mapshaper.org)
2. Import `.geojson` file
3. Click features to inspect properties
4. Export to other formats if needed

### Recommended Starting Layers

| Purpose | File | Suggested Filter |
|---------|------|------------------|
| CD profiles | `cd_types_with_stability.geojson` | Color by `cluster_label_short` |
| NTA profiles | `nta_types_with_stability_residential.geojson` | Filter `is_mostly_structural = true` |
| Reliable hotspots | `hotspot_persistent_all3.geojson` | None (already filtered) |
| Map-grade hotspots | `hotspot_cells_ge50.geojson` | Filter `is_suspected_artifact = false` |

---

## Key Findings

1. **Most areas are episodic, not structural.** Only 3% of CDs and NTAs maintain the same cluster across all 3 years. Pooled profiles are more reliable than single-year.

2. **Hotspot persistence is low.** Of the top 100 hotspot cells per year, only 19 persist across all 3 years. Most hotspots are transient.

3. **Artifact flagging changes interpretation.** The #1 hotspot cell (31,207 complaints) is 91% concentrated at a single address - likely a data artifact, not a crisis zone.

4. **Weekend and late-night patterns vary by profile.** Some areas show strong weekend uplift; others peak in late-night hours. These behavioral signatures are consistent but don't explain causation.

5. **Heat increases complaints.** Higher nighttime temperatures are associated with more noise complaints, with sensitivity varying across districts.

---

## Data Sources

| Dataset | Source | Coverage |
|---------|--------|----------|
| 311 Noise Complaints | NYC Open Data API | 2021-2023 |
| Transportation Noise | BTS NTNM 2020 | Road + Rail + Aviation |
| Light at Night | NASA VIIRS VNP46A4 | 2021-2023 annual |
| Nighttime Temperature | Oregon State PRISM | Daily Tmin 2021-2023 |
| Air Quality | NYC DOHMH NYCCAS | PM2.5, NO2 2021-2023 |
| Demographics | US Census ACS 5-Year | 2022 estimates |

---

## Reproducibility

### Pipeline Architecture

- **14 numbered scripts** in `scripts/` executed in sequence
- **Centralized configuration** in `configs/params.yml`
- **430+ automated tests** in `tests/`
- **Metadata sidecars** for all outputs with provenance hashes

### To Regenerate Outputs

```bash
# Create environment
conda env create -f environment.yml
conda activate sleep-esi

# Run pipeline (requires raw data)
python scripts/00_build_geographies.py
python scripts/01_build_crosswalks.py
python scripts/02_fetch_311_noise.py
python scripts/03_build_311_night_features.py
# ... continue with remaining scripts
```

### Raw Data

Raw data files are not included in this repository due to size (7.3GB). To reproduce:

1. Run `scripts/02_fetch_311_noise.py` to fetch 311 data from NYC Open Data API
2. Download PRISM, VIIRS, and NYCCAS rasters manually (see data source links)
3. Place files in `data/raw/` following the structure in the scripts

---

## Project Structure

```
Sleep_ESI_NYC/
├── configs/           # Configuration files (params.yml, weights.yml)
├── data/
│   ├── processed/     # Analysis outputs (included)
│   └── raw/           # Raw data (not included, 7.3GB)
├── docs/              # GitHub Pages website
├── scripts/           # Pipeline scripts (00-14)
├── src/sleep_esi/     # Utility modules
└── tests/             # Automated tests (430+)
```

---

## License

Code: MIT License | Data outputs: CC BY 4.0

---

## Citation

```
Plutchok, Y.Y. (2025). NYC Night Signals Atlas: Nighttime Noise Disturbance
Patterns Across New York City Community Districts, 2021-2023.
https://github.com/yoelplutchok/Sleep_ESI_NYC
```

---

**Author:** Yoel Y. Plutchok
**Data Period:** January 2021 - December 2023
**Last Updated:** January 2025
