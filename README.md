# DEPOXY Project

Physiological data analysis project (Python + R) for the Master STAPS IEAP program.

## Project Info

- Author: Baptiste AUGROS
- Period: March 2026
- Subject: Allan LEPLAT
- Goal: compare physiological responses across running shoe conditions

## Repository Structure (exact project layout)

```text
.
├── LICENCE
├── README.md
├── environment.yml
├── requirements.txt
├── main.ipynb
├── main.Rmd
├── main.Rproj
├── Summary file DEPOXY Project.csv
├── data/
│   ├── Données FC Allan.csv
│   ├── Données FP Allan.csv
│   ├── Données K5 Allan.csv
│   └── Données NIRS Allan.csv
├── results/
│   └── (empty by default; generated after full Python + R files fully runned)
└── sources/
    ├── load_data.py
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