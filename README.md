# DEPOXY Project

**Multimodal physiological signal analysis during incremental exercise : muscle oxygenation, force output, metabolic demand, and cardiac response**

| | |
|---|---|
| **Student** | AUGROS Baptiste |
| **Course** | Python – R – Git |
| **Institution** | Master STAPS IEAP - 2SIA Track |
| **Year** | 2025 – 2026 |
| **Raw data** | [Google Drive](https://drive.google.com/file/d/1jqL7KQI9zYLx_VKepTDGp9bGdf7xI97Y/view?usp=sharing) |

---

## Scientific Question

Does the NIRS-derived deoxygenation breakpoint (Sd) correspond temporally and functionally to the ventilatory threshold (VT1) detected from breath-by-breath gas exchange? Are these two physiological markers synchronised, or do they reflect distinct mechanisms of oxygen supply-demand regulation?

---

## Project Structure

```
Augros.Baptiste/
│
├── data/
│   │   ├── Données FP Allan.csv (see Google Drive link providden before - too heavy)
│   │   ├── Données NIRS Allan.csv
│   │   ├── Données K5 Allan.csv
│   │   └── Données FC Allan.csv
│
├── sources/                  # Scripts will be useful for my futur analysis        │                              pipeline and generalization to all my partcipants.
│
├── results/                  # Empty folder which will be created and filled once   │                                both notebooks will be completly run
│
│   ├── load_data.py          # Data ingestion and schema normalisation
│   ├── synchronization.py    # Multi-device time alignment (trigger-based)
│   ├── filtering.py          # Butterworth low-pass + rolling-median filters
│   ├── segmentation.py       # Stage detection and segmentation
│   ├── features.py           # Feature extraction per stage per modality
│   ├── statistics.py         # Python-side correlations and exports
│   ├── visualization.py      # Matplotlib/Seaborn publication figures
│
├── Summary file DEPOXY Project # Summarizes all informations about the participants │                                 (not used here but will be used for               │                                  reproducibility in my futur pipeline)
│
├── LICENCE
│
├── main.ipynb                # Python main notebook
│
├── main.Rmd                  # R main notebook
│
├── main.Rproj                # R main project
│
├── environment.yaml          # Pipeline configuration (paths, filter params)
├── requirements.txt          # Python dependencies
├── augros.baptiste.html      # Final HTML technical report (grading copy)
└── README.md                 # This file
```

---

## Data Sources

Because raw physiological files exceed GitHub's file size limits, all raw data are stored externally.

**Google Drive (raw dataset):** https://drive.google.com/file/d/1jqL7KQI9zYLx_VKepTDGp9bGdf7xI97Y/view?usp=sharing

Place downloaded files into `data/raw/` before running the pipeline.

The four modalities collected are listed below.

| Device | Signal | Sampling rate | File |
|---|---|---|---|
| Force plate (Kistler / AMTI) | Fz, Fx, Fy (N) | 1000 Hz | `Données FP Allan.csv` |
| NIRS (Moxy / Artinis) | SmO2 (%), [O2Hb], [HHb] (µM) | 10–50 Hz | `Données IRS Allan.csv` |
| COSMED K5 | VO2, VCO2, VE, RER (breath-by-breath) | ~0.2–0.4 Hz | `Données K5 Allan.csv` |
| Polar H10 / chest strap | HR (bpm), R-R interval (ms) | 1 Hz / ~250 Hz | `Données FC Allan.csv` |

---

## Pipeline Overview

The processing pipeline is fully scripted in Python 3.10 and follows a strict step-by-step architecture. Each script is a self-contained module and can be run independently for debugging.

**Step 1 — Loading (`load_data.py`):** ingests all raw files, normalises column names, and converts timestamps to seconds relative to a common origin.

**Step 2 — Synchronisation (`synchronization.py`):** detects the hardware trigger pulse embedded in the force-plate file, computes per-device time offsets using cross-correlation, and resamples all streams to 100 Hz using `scipy.signal.resample_poly()`.

**Step 3 — Filtering (`filtering.py`):** applies signal-appropriate zero-phase Butterworth low-pass filters via `scipy.signal.sosfiltfilt()`. Cut-off frequencies are: force = 50 Hz, NIRS = 0.1 Hz, HR = 0.05 Hz. VO2 uses a 5-point rolling median followed by a 0.03 Hz Butterworth.

**Step 4 — Segmentation (`segmentation.py`):** splits the continuous recording into exercise stages using protocol timestamp metadata. Stage onset and offset are detected automatically.

**Step 5 — Feature extraction (`features.py`):** computes mean, SD, slope, and peak values per stage for each modality. The NIRS deoxygenation breakpoint (Sd) is detected via double-linear segmented regression using `scipy.optimize.curve_fit()`.

**Step 6 — Statistics (`statistics.py`):** runs Pearson and Spearman correlations (via `pingouin`) between threshold estimates.

**Step 7 — Visualisation (`visualization.py`):** generates all figures and csv (time series, Bland-Altman, correlation scatter plots, boxplots...) saved in results folder as CSV, PNG...

**Step 8 — R analysis (`main.Rmd`):** fits linear mixed-effects models with `lme4`/`lmerTest`, runs repeated-measures ANOVA with `afex`, and applies Bonferroni-corrected post-hoc tests.

---

## How to Reproduce

### Python pipeline

```bash
# 1. Clone the repository
git clone https://github.com/Baptoudu36/Augros.Baptiste.git
cd Augros.Baptiste

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Download raw data from Google Drive and place in data/raw/

# 4. Run the full pipeline
python scripts/run_pipeline.py --config config.yaml
```

### R analysis

Open `main.Rmd` in RStudio and click **Knit**, or run from the terminal:

```r
rmarkdown::render("R/analysis.Rmd")
```

Required R packages: `lme4`, `lmerTest`, `afex`, `ggplot2`, `tidyverse`, `rmarkdown`.

---

## Key Results (summary)

The NIRS-derived deoxygenation breakpoint (Sd) and the ventilatory threshold (VT1) show a strong positive correlation (Pearson r = .84, p < .01) with clinically acceptable Bland-Altman limits of agreement (±0.6 exercise stages). A linear mixed-effects model confirms that SmO2 decreases significantly with both VO2 intensity and stage number (both p < .001), with substantial inter-individual variability in baseline oxygenation (random intercept SD = 4.2%).

---

## Dependencies

### Python (requirements.txt)

```
# ============================================================================
# DEPOXY Project — Python requirements (pip-installable subset)
# ============================================================================
# For a full reproducible environment use environment.yml with conda:
#
#   conda env create -f environment.yml
#   conda activate depoxy
#
# If you prefer pip-only installation:
#
#   pip install -r requirements.txt
#
# Note: conda is strongly preferred because it resolves binary dependencies
# (numpy, scipy, matplotlib) more reliably than pip on all platforms.
# ============================================================================

# ── Data manipulation ────────────────────────────────────────────────────────
numpy==1.26.4
pandas==2.2.1

# ── Scientific computing ──────────────────────────────────────────────────────
scipy==1.13.0
scikit-learn==1.4.2

# ── Visualisation ─────────────────────────────────────────────────────────────
matplotlib==3.8.4
seaborn==0.13.2

# ── Statistical modelling ─────────────────────────────────────────────────────
statsmodels==0.14.2
pingouin==0.5.4

# ── Jupyter ───────────────────────────────────────────────────────────────────
jupyter==1.0.0
notebook==7.1.3
ipykernel==6.29.4
```

### R

```r
install.packages(c("lme4", "lmerTest", "afex", "ggplot2",
                   "tidyverse", "rmarkdown", "knitr"))
```

---

## Report

The final grading HTML report is: **`augros.baptiste.html`**

It is self-contained, requires no internet access (other than Google Fonts) and can be opened directly in any browser.

---

## License

This project is submitted for academic evaluation (Master STAPS IEAP, 2025–2026). Reuse for academic or educational purposes is permitted with attribution.    
    ├── synchronization.py
    ├── filtering.py
    ├── segmentation.py
    ├── features.py
    ├── statistics.py
    └── visualization.py
```

## What Each File/Folder Does

- LICENCE: project license file.
- README.md: project documentation and usage guide.
- environment.yml: Conda environment definition (recommended setup).
- requirements.txt: pip package list for Python dependencies.
- main.ipynb: main Python notebook for preprocessing, metrics, plots, and CSV exports.
- main.Rmd: R Markdown report using exported data for additional statistics and reporting.
- main.Rproj: RStudio project file for easier R workflow.
- Summary file DEPOXY Project.csv: participant/session summary metadata used in analysis.

- data/: raw input files.
- data/Données FC Allan.csv: heart rate time series.
- data/Données FP Allan.csv: force plate signals.
- data/Données K5 Allan.csv: COSMED K5 metabolic data.
- data/Données NIRS Allan.csv: NIRS oxygenation signals.

- sources/: modular Python pipeline code.
- sources/load_data.py: load and standardize raw datasets.
- sources/synchronization.py: align signals on a common timeline.
- sources/filtering.py: apply signal filtering/cleaning.
- sources/segmentation.py: split data into protocol phases/blocks.
- sources/features.py: compute derived physiological metrics.
- sources/statistics.py: run statistical analyses and table outputs.
- sources/visualization.py: generate plots and figure exports.

- results/: output folder kept empty in versioned project state.
- results/ (runtime): populated only after running main.ipynb and main.Rmd end-to-end.

## Quick Start

### 1) Python environment

Conda (recommended):

```bash
conda env create -f environment.yml
conda activate depoxy
```

pip alternative:

```bash
pip install -r requirements.txt
```

### 2) Run Python analysis

```bash
jupyter notebook main.ipynb
```

Then run all notebook cells in order.

### 3) Run R report

From RStudio (open main.Rproj) or from terminal:

```bash
Rscript -e "rmarkdown::render('main.Rmd')"
```

## Important Notes

- The results folder is intentionally empty before execution.
- Output files (CSVs and figures) are generated only after complete runs of both notebooks/workflows.
- Local technical files such as .Rproj.user, .Rhistory, .RData, and .DS_Store are not part of the scientific pipeline.
