"""UCI dataset benchmark runner. Produces holdout metrics, predictions and a reusable model."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, f1_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from ucimlrepo import fetch_ucirepo

DATASET_ID = 186
PROJECT_TITLE = "Wine Quality Lab"
CLASSIFICATION_IDS = {602, 228, 159, 357, 186}

def load_data():
    dataset = fetch_ucirepo(id=DATASET_ID)
    X, y = dataset.data.features.copy(), dataset.data.targets.copy()
    split_mode = "random"
    if DATASET_ID == 357:
        dates = pd.to_datetime(X["date"], errors="coerce")
        order = dates.sort_values(kind="stable").index
        X = X.loc[order].drop(columns=[c for c in ["id", "date"] if c in X.columns])
        y, split_mode = y.loc[order], "chronological"
    if DATASET_ID == 477:
        X = X.drop(columns=[c for c in X.columns if c.lower() in {"no", "id"}], errors="ignore")
    keep = y.notna().all(axis=1)
    return X.loc[keep].reset_index(drop=True), y.loc[keep].reset_index(drop=True), split_mode

def main():
    parser = argparse.ArgumentParser(description=PROJECT_TITLE)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    X, y, split_mode = load_data()
    classification = DATASET_ID in CLASSIFICATION_IDS
    if DATASET_ID == 357:
        cut = int(len(X) * 0.7)
        X_train, X_test, y_train, y_test = X.iloc[:cut], X.iloc[cut:], y.iloc[:cut], y.iloc[cut:]
    else:
        stratify = y.iloc[:, 0] if classification and y.shape[1] == 1 else None
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=stratify)
    y_train_fit = y_train.iloc[:, 0] if classification or y_train.shape[1] == 1 else y_train
    y_test_eval = y_test.iloc[:, 0] if classification or y_test.shape[1] == 1 else y_test
    text_cols = [c for c in X.columns if X[c].dtype == "object" and X[c].astype(str).str.len().mean() > 25]
    numeric = [c for c in X.select_dtypes(include=np.number).columns if c not in text_cols]
    categorical = [c for c in X.columns if c not in numeric and c not in text_cols]
    transformers = []
    if numeric:
        transformers.append(("numeric", SimpleImputer(strategy="median"), numeric))
    if categorical:
        cat_pipe = Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))])
        transformers.append(("categorical", cat_pipe, categorical))
    for i, col in enumerate(text_cols):
        text_pipe = Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="")), ("flatten", FunctionTransformer(lambda a: a.ravel(), validate=False)), ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000, sublinear_tf=True))])
        transformers.append((f"text_{i}", text_pipe, [col]))
    preprocess = ColumnTransformer(transformers, remainder="drop")
    if classification:
        estimator = RandomForestClassifier(n_estimators=400, min_samples_leaf=2, class_weight="balanced_subsample", random_state=42, n_jobs=-1)
        baseline = DummyClassifier(strategy="most_frequent")
    else:
        estimator = RandomForestRegressor(n_estimators=400, min_samples_leaf=2, random_state=42, n_jobs=-1)
        baseline = DummyRegressor(strategy="median")
    model = Pipeline([("preprocess", preprocess), ("model", estimator)])
    model.fit(X_train, y_train_fit)
    pred = model.predict(X_test)
    baseline.fit(X_train, y_train_fit)
    base_pred = baseline.predict(X_test)
    metrics = {"project": PROJECT_TITLE, "dataset_id": DATASET_ID, "rows": len(X), "features": list(X.columns), "targets": list(y.columns), "split": "chronological 70/30 forward holdout" if split_mode == "chronological" else "random 75/25 holdout; seed=42", "task": "classification" if classification else "regression"}
    if classification:
        metrics["holdout"] = {"accuracy": float(accuracy_score(y_test_eval, pred)), "balanced_accuracy": float(balanced_accuracy_score(y_test_eval, pred)), "macro_f1": float(f1_score(y_test_eval, pred, average="macro")), "weighted_f1": float(f1_score(y_test_eval, pred, average="weighted")), "majority_baseline_accuracy": float(accuracy_score(y_test_eval, base_pred)), "per_class": classification_report(y_test_eval, pred, output_dict=True, zero_division=0)}
        if len(np.unique(y_test_eval)) == 2:
            metrics["holdout"]["roc_auc"] = float(roc_auc_score(y_test_eval, model.predict_proba(X_test)[:, 1]))
    else:
        actual = np.asarray(y_test_eval)
        prediction = np.asarray(pred)
        base_prediction = np.asarray(base_pred)
        if actual.ndim == 1: actual = actual.reshape(-1, 1)
        if prediction.ndim == 1: prediction = prediction.reshape(-1, 1)
        if base_prediction.ndim == 1: base_prediction = base_prediction.reshape(-1, 1)
        metrics["holdout"] = {str(y.columns[i]): {"mae": float(mean_absolute_error(actual[:, i], prediction[:, i])), "rmse": float(np.sqrt(mean_squared_error(actual[:, i], prediction[:, i]))), "r2": float(r2_score(actual[:, i], prediction[:, i])), "median_baseline_mae": float(mean_absolute_error(actual[:, i], base_prediction[:, i]))} for i in range(actual.shape[1])}
    metrics["limitations"] = "Public benchmark; holdout results do not establish production performance." + (" Telescope events are Monte Carlo simulations." if DATASET_ID == 159 else "")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    actual = np.asarray(y_test_eval)
    prediction = np.asarray(pred)
    if actual.ndim == 1: actual = actual.reshape(-1, 1)
    if prediction.ndim == 1: prediction = prediction.reshape(-1, 1)
    predictions = pd.DataFrame({f"actual_{y.columns[i]}": actual[:, i] for i in range(actual.shape[1])})
    for i in range(prediction.shape[1]): predictions[f"predicted_{y.columns[i]}"] = prediction[:, i]
    predictions.to_csv(args.output_dir / "holdout_predictions.csv", index=False)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    joblib.dump(model, args.output_dir / "model.joblib")
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
