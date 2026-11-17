# Wine Quality Lab

Predict ordered wine-quality labels and inspect class-wise errors; rare quality classes have very little holdout support.

## Dataset

[UCI Wine Quality dataset](https://archive.ics.uci.edu/dataset/186/wine%2Bquality) (dataset ID 186), 6,497 records. The source is public and is fetched by the runner; the raw dataset is not committed.

## Run

```bash
python -m pip install -r requirements.txt
python main.py --output-dir artifacts
```

The runner saves metrics, holdout predictions, and a reusable model in the ignored output directory. The public [`reports/benchmark.json`](reports/benchmark.json) contains aggregate results only; no row-level records are published.

## Holdout results

A stratified random 75/25 split (seed 42) is used. Numeric inputs are median-imputed; the random-forest classifier uses 400 trees and balanced subsampling.

| Accuracy | Balanced accuracy | Macro F1 | Weighted F1 | Majority baseline accuracy |
|---:|---:|---:|---:|---:|
| 0.679 | 0.374 | 0.405 | 0.671 | 0.436 |

Full aggregate metrics and the exact split seed are in [`reports/benchmark.json`](reports/benchmark.json).

## Limits

Quality is ordinal and highly imbalanced. The low macro F1 and poor recall on rare labels are visible limitations; aggregate accuracy should not be read as reliable performance across all quality classes. These are one-time benchmark results, not service-level, production, or causal claims.

## Reproducibility

The model and split use seed 42. Re-running downloads source data from UCI and regenerates local artifacts. Model binaries, raw inputs, and row-level predictions are excluded from this public repository.
