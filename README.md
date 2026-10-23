# Wine Quality Lab

A reproducible baseline for classifying UCI Wine Quality ratings from physicochemical measurements. The task is treated as ordinal-label classification; the model reports class-aware metrics because quality scores are imbalanced.

## Dataset

The script downloads UCI dataset 186 with the ucimlrepo package. See the [UCI Wine Quality dataset card](https://archive.ics.uci.edu/dataset/186/wine). No dataset copy is committed here.

## Run

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

The first run needs network access to fetch the dataset. Use `python main.py --output-dir results` to choose another output directory.

## Method and outputs

- Fixed random 75/25 holdout (seed 42), stratified by quality label.
- Median imputation, a 400-tree random forest classifier, and a majority-class dummy baseline.
- Accuracy, balanced accuracy, macro/weighted F1, and per-class precision/recall/F1.
- Writes `artifacts/metrics.json`, `artifacts/holdout_predictions.csv`, and `artifacts/model.joblib`.

This is a public-dataset benchmark, not a production wine-quality system. A single random split does not establish performance on new producers, vintages, or regions. No performance result is claimed until the script has been run and independently checked.
