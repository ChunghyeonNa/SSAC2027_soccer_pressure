# Exploring Factors Influencing Human Classification of Pressure in Soccer Matches

This repository contains the code and materials supporting our abstract submitted to the MIT Sloan Sports Analytics Conference (SSAC) 2027.

The study examines which factors explain human-collected pressure labels on passes in the 2022 FIFA World Cup event data, compares them with published pressure metrics, and tests whether information from reception to pass (the on-ball phase) adds to the moment of the pass.

## Data

The analysis uses the publicly released event and tracking data for all 64 matches of the 2022 FIFA World Cup (Gradient Sports, USA; formerly PFF FC). The data are free but are not redistributed here. Request access through the form on the provider's release page:

https://www.gradientsports.com/blog/pff-fc-release-2022-world-cup-data

After downloading, place the data as follows:

```
data/World_cup/
├── Event Data/       {game_id}_event.json
├── Tracking Data/    {game_id}.jsonl_tracking.bz2
├── Metadata/         {game_id}_meta.json
└── Rosters/          {game_id}_roster.json
```

## Setup

Python 3.12 and [uv](https://docs.astral.sh/uv/) are used.

```bash
uv sync
mkdir -p outputs/figs
```

Run the notebooks in `notebooks/` in numerical order with the project environment (for example, `uv run jupyter lab`). Intermediate files and figures are written to `outputs/`. Tracking extraction in notebooks 05 and 11 takes several minutes and is cached after the first run.

## Notebooks

| Notebook | Content |
|---|---|
| 00_helper_functions | Walk-through of the helper functions in `src/football` |
| 01_build_events | Event table with passes, pressure labels, and player coordinates |
| 05_pressure_benchmark | Published pressure metrics (Andrienko et al., 2017; Bekkers, 2025; Narizuka et al., 2026) compared with the labels |
| 06_pressure_rule_vs_judgement | Written collection rule implemented on tracking data; label rates by distance, movement, and direction |
| 07_abstract_table_figure | Benchmark summary table (input to notebook 12) |
| 08_pressure_determinants | Hierarchical logistic regression of moment-of-pass variables; 50% labeling distance; SHAP |
| 09_possession_window | Nearest-defender distance before the pass (input to notebook 10) |
| 10_sequence_context | Labels recorded at reception and on the previous pass |
| 11_possession_exposure | Exposure from reception to pass; pass failure models |
| 12_abstract_v2_table_figure | Table 1 and Figure 1 of the abstract |
| 13_supervised_pressure_classifier | LightGBM classifier without pressure labels as inputs |
| 14_abstract_statistics | Group comparisons and distance-adjusted odds ratio reported in the abstract |

## Numbers in the abstract

| Result | Notebook |
|---|---|
| AUC 0.886 for distance; +0.007 for other moment-of-pass variables; 50% labeling probability at 2.8 m | 08 |
| Published metrics, AUC 0.795–0.874 | 05, 07 |
| Written rule, weighted κ 0.43 vs. 0.59 | 06 |
| 34.2% vs. 9.1%; distance-adjusted odds ratio 4.42 | 14 |
| Exposure, AUC 0.863 to 0.878 | 11 |
| 91.2% vs. 12.7% by the label recorded at reception | 14 |
| Classifier accuracy 84.9% (majority baseline 57.7%); AUC 0.913 to 0.922 | 13, 14 |
| Pass failure, baseline AUC 0.842; +0.012 for exposure vs. +0.003 for distance | 11 |

## Code

`src/football/` contains the shared functions: coordinate handling and match-grouped cross-validation with cluster bootstrap (`common.py`), pressure metrics and the per-defender table (`pressure.py`), and tracking extraction including exposure from reception to pass (`tracking.py`).
