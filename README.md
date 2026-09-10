# AI Programming Foundations Project — Wine Quality Data Workflow

**Author:** Cameron Doelling

A complete, reproducible data workflow built in a single Jupyter notebook. The notebook ingests
the UCI Wine Quality dataset, cleans it with documented and reusable functions, explores it with
summary statistics and correlation checks, and communicates the findings through four labelled
visualizations and a written interpretation. No models are trained — this is the data foundation
that later machine learning work builds on.

## What is in this repository

| File | Purpose |
| --- | --- |
| `data_workflow.ipynb` | The full workflow: load, clean, explore, visualize, summarize |
| `winequality.csv` | The raw dataset, committed so the notebook runs offline |
| `winequality_clean.csv` | The cleaned table written out by the notebook |
| `figures/` | The four figures saved by the notebook |
| `module_summary.pdf` | Written report with in-text citations and references |
| `requirements.txt` | Pinned dependencies |
| `build_notebook.py` | Regenerates `data_workflow.ipynb` from source cells |

## Dataset

**UCI Wine Quality** — 6,497 Portuguese *Vinho Verde* wines (1,599 red, 4,898 white) with eleven
physicochemical measurements and a 0–10 sensory quality score.
Source: https://archive.ics.uci.edu/dataset/186/wine+quality

## How to run

Requires Python 3.9 or newer.

```bash
git clone https://github.com/doellingcameron92/ai-programming-foundations-project.git
cd ai-programming-foundations-project

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
jupyter lab data_workflow.ipynb
```

Then run all cells from top to bottom (`Run > Run All Cells`). The notebook reads
`winequality.csv` from this folder, writes `winequality_clean.csv`, and saves the figures into
`figures/`.

To regenerate `requirements.txt` after adding a dependency:

```bash
pip freeze > requirements.txt
```

## Branches

- `main` — reviewed, working state of the project
- `development` — branch used to build out the cleaning, EDA and visualization work
