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

## Reflections

### Where poor data cleaning could introduce bias

The riskiest step in this project was the outlier decision. Sixty-two wines carry a measurement
more than five standard deviations from its column mean. Deleting them would have been the tidier
choice, and it would also have systematically removed the very sweet and the heavily sulphured
wines — real wines whose chemistry is unusual but valid. The cleaned dataset would then have
described a narrower population than the one it claims to represent, and every summary statistic
would have been quietly biased toward the typical wine. I flagged those rows in an `is_extreme`
column instead of dropping them, so the decision stays visible and reversible.

Deduplication carries the opposite risk. Removing the 1,177 exact duplicates was necessary — they
would otherwise have pulled every mean and correlation toward whichever wine profiles happen to
repeat — but it is an irreversible judgement that identical rows are repeated records rather than
distinct bottles. Because that is a judgement, the function prints exactly how many rows it removed
rather than doing it silently.

Two biases are inherent in the data and no amount of cleaning removes them. The `quality` score is
the median of at least three expert tasters' opinions, so it measures perceived quality within one
panel and one wine culture, not an objective standard. And white wines outnumber red three to one,
so any statistic pooled across both colours is effectively a statistic about white wine — which is
why the analysis reports results by colour wherever the two differ. Careless imputation would add a
third bias: filling gaps with a column mean would shrink variance and pull skewed chemical
measurements toward a centre that no real wine occupies, which is why the missing-value function
uses the median and reports every value it fills.

### How this workflow changes when machine learning is added

The cleaning and EDA stages stay, but three things change. First, the data has to be split before
anything is fitted, and every statistic learned from the data — imputation medians, scaling
parameters, category encodings — must be computed on the training split alone and applied to the
test split, or information leaks across the boundary and the reported accuracy becomes optimistic.
Second, deduplication stops being a convenience and becomes a correctness requirement: with 18% of
rows duplicated, a random split would place copies of the same wine in both training and test data,
and the model would be scored partly on rows it had memorised. Third, the class imbalance visible
in Figure 1 forces the evaluation to change — a model that only ever predicts 5 or 6 would already
look accurate, so the split needs to be stratified by quality and the results reported per class
rather than as a single accuracy number. The cleaning functions themselves would move out of the
notebook into a module so that the same transformations run identically at training time and at
prediction time.

### What would need to change to prepare this data for a neural network

Neural networks need numeric input on a comparable scale, so the `color` column would be encoded
(one-hot, or a single binary indicator since it has two levels) and every measurement standardised
to zero mean and unit variance — necessary here because the raw features span wildly different
ranges, from chlorides near 0.05 to total sulphur dioxide near 140, and unscaled inputs of that
kind make gradient descent converge slowly and unevenly. The extreme values I chose to flag rather
than delete matter more for a network than for the correlation analysis, since a single value five
standard deviations out produces a large gradient; the `is_extreme` column makes it easy to clip or
exclude them for a training run and compare. The dataset also needs a three-way split into
training, validation and test rather than two, because a network's early stopping and
hyperparameter choices are themselves fitted to the validation data. Finally, 5,320 rows is a small
dataset by neural network standards, which argues for a small network with regularisation over a
deep one, and makes the redundancy Figure 2 exposed — density against alcohol and sugar, free
against total sulphur dioxide — worth resolving before adding model capacity.

### Where agentic automation could take over parts of this workflow

The mechanical parts of this workflow are the natural candidates. An agent could run the data audit
on every new data drop, compare the schema, dtypes, missing-value rates and duplicate counts
against the last accepted version, and open an issue when something moves — the kind of check that
is easy to write once and easy to forget to run. It could regenerate the notebook, the figures and
the PDF report on every commit, so the artefacts never drift from the code that produced them, and
it could keep `requirements.txt` current and flag when a dependency upgrade changes a reported
number.

What should not be automated is the judgement. Whether identical rows are duplicate records or
distinct samples, whether an extreme value is a measurement error or a genuinely unusual wine, and
whether a correlation is worth reporting are all decisions that depend on knowing what the data
represents. An agent that deleted outliers automatically would silently produce exactly the bias
described above. The useful division is that the agent detects, reports and reruns, while a person
decides — and the fact that every cleaning step here is a small named function with a docstring is
what would make that division workable, since an agent can call the same functions a human reviews.
