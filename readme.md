# Machine Learning Framework – Dataset Analysis & Automated Random Forest Experiments

A framework that **analyzes a dataset** (profiling, visualization, schema extraction) and **automatically trains a Random Forest model** with multiple trials and experiments.  
Designed to be modular so that dataset analysis, preprocessing, and model training can be run separately or together.

---

## 📦 Project Structure

- `dataset_analysis.py` – Toolkit: profiling, charts, YAML schema, `profile_dataset()`
- `main.py` – CLI entrypoint (run with `--analyze` to profile the dataset)
- `datasets/` – Default location for input datasets (ignored by git)
  - `.gitkeep`
- `dataset_analisys/` – Default output folder for analysis results (ignored by git)
  - `.gitkeep`
- `environment.yml` – Conda environment definition
- `.github/workflows/ci.yml` – GitHub Actions CI workflow
- `LICENSE` – MIT License
- `README.md`



---

## ⚙️ Requirements

- [Conda](https://docs.conda.io/) (Anaconda or Miniconda)
- Python 3.10+

---

## 📥 Installation (Conda)

```bash
# Create the environment
conda env create -f environment.yml

# Activate it
conda activate mlframework

# (Optional) verify versions
python -V
python -c "import pandas, numpy, matplotlib, yaml; print('OK')"
```

If you prefer manual setup:

```bash

conda create -n mlframework python=3.10 -y
conda activate mlframework
pip install pandas numpy matplotlib pyyaml
```
▶️ Usage
Place your CSV dataset inside the datasets/ folder, for example:
datasets/my_dataset.csv

1) Run without analysis (default: skipped)
```bash

python main.py --dataset datasets/my_dataset.csv
```
2) Run with analysis
```bash

python main.py --dataset datasets/my_dataset.csv --analyze
```
3) Specify the label column (example: labels)
```bash

python main.py --dataset datasets/my_dataset.csv --analyze --label-column labels
```
4) Useful options
```bash

# Top-k categories for frequency analysis
python main.py --dataset datasets/my_dataset.csv --analyze --freq-top-k 30

# Number of bins for numeric histograms
python main.py --dataset datasets/my_dataset.csv --analyze --numeric-bins 40

# Limit number of generated plots
python main.py --dataset datasets/my_dataset.csv --analyze --max-num-plots 10 --max-cat-plots 8

# Do not save HTML overview
python main.py --dataset datasets/my_dataset.csv --analyze --no-html

```

## 📊 Output Structure
When analysis is enabled, results are saved in:

- `dataset_analisys/`
  - `my_dataset/`
    - **tables/**
      - `column_types.csv`
      - `numeric_stats.csv`
      - `categorical_<col>.csv`
      - `label_distribution.csv` – only if `--label-column` is valid
    - **figures/**
      - `label_balance.png` – only if `--label-column` is valid
      - `hist_<numcol>.png`
      - `freq_<catcol>.png`
    - `schema.yaml`
    - `overview.html` – only if not disabled

## 🔬 Next Steps
This framework will be extended to:

- Automatically train Random Forest models on the dataset

- Perform multiple trials with:

    - Different tree counts

    - Different subsets of features

    - Various preprocessing strategies

    - Compare results automatically

## 🧪 GitHub Actions CI
A CI workflow runs on every push and pull request:

Lints the code with ruff

Creates a sample dataset

Runs a “dry-run” analysis in headless mode (no HTML overview)

## 📜 License
MIT License – you are free to use, modify, and distribute this code for any purpose.