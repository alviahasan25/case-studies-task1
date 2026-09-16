"""
COSC2816 Individual Task 2, Part 2 - evaluation diagnostics.

Re-runs the Task 1 models under repeated stratified cross-validation so that the
deliberation can be written against measured numbers instead of assertions.

Produces four things:
  1. The original single 75/25 split result (reproduces Task 1, Table 1).
  2. Repeated stratified k-fold: mean, std and 95% range for each metric.
  3. Where the seed-42 single-split result sits in the CV distribution
     (i.e. how lucky or unlucky that one split was).
  4. Subgroup metrics, to test whether performance is even across student groups.

Usage
-----
    pip install pandas scikit-learn numpy
    python evaluation_diagnostics.py --dropout data.csv --performance student-por.csv

Both files are the raw semicolon-delimited CSVs from UCI:
  dropout     -> https://archive.ics.uci.edu/dataset/697  (4424 rows)
  performance -> student-por.csv from
                 https://archive.ics.uci.edu/dataset/320  (649 rows)
"""

import argparse
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import (RepeatedStratifiedKFold, cross_validate,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

SEED = 42
N_SPLITS = 5
N_REPEATS = 10          # 50 fits per model - enough for a stable std
SCORING = ["accuracy", "precision", "recall", "f1", "roc_auc"]


# --------------------------------------------------------------------------- #
# Pipeline construction (identical preprocessing to Task 1)
# --------------------------------------------------------------------------- #

def build_pipeline(X, model):
    """Impute -> scale numerics / one-hot categoricals -> classifier."""
    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), num_cols),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
    ])
    return Pipeline([("pre", pre), ("clf", model)])


def models():
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=SEED),
        "SVM (RBF)": SVC(
            kernel="rbf", class_weight="balanced", random_state=SEED),
    }


# --------------------------------------------------------------------------- #
# 1. Single split (reproduces Task 1)
# --------------------------------------------------------------------------- #

def single_split(X, y, model, seed=SEED):
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=seed)
    pipe = build_pipeline(X, model).fit(Xtr, ytr)
    pred = pipe.predict(Xte)
    score = (pipe.decision_function(Xte) if hasattr(pipe, "decision_function")
             else pipe.predict_proba(Xte)[:, 1])
    return {
        "accuracy":  accuracy_score(yte, pred),
        "precision": precision_score(yte, pred, zero_division=0),
        "recall":    recall_score(yte, pred, zero_division=0),
        "f1":        f1_score(yte, pred, zero_division=0),
        "roc_auc":   roc_auc_score(yte, score),
    }


# --------------------------------------------------------------------------- #
# 2 & 3. Repeated stratified CV + where the single split sits in it
# --------------------------------------------------------------------------- #

def repeated_cv(X, y, model):
    cv = RepeatedStratifiedKFold(
        n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    res = cross_validate(build_pipeline(X, model), X, y,
                         cv=cv, scoring=SCORING, n_jobs=-1)
    return {m: res[f"test_{m}"] for m in SCORING}


def report(name, dataset, X, y):
    print(f"\n{'=' * 74}\n{dataset}  |  {name}\n{'=' * 74}")
    mdl = models()[name]

    single = single_split(X, y, mdl)
    dist = repeated_cv(X, y, mdl)

    print(f"{'metric':<11}{'Task 1':>9}{'CV mean':>10}{'CV std':>9}"
          f"{'95% range':>18}{'pctile':>9}")
    print("-" * 74)
    for m in SCORING:
        d = dist[m]
        lo, hi = np.percentile(d, [2.5, 97.5])
        pct = (d < single[m]).mean() * 100      # where seed 42 sits
        print(f"{m:<11}{single[m]:>9.3f}{d.mean():>10.3f}{d.std():>9.3f}"
              f"{f'[{lo:.3f}, {hi:.3f}]':>18}{pct:>8.0f}%")

    spread = {m: np.percentile(dist[m], 97.5) - np.percentile(dist[m], 2.5)
              for m in SCORING}
    print(f"\nWidest 95% interval: {max(spread, key=spread.get)} "
          f"(+/-{max(spread.values()) / 2:.3f})")
    print("A single split reports one draw from these distributions.")


# --------------------------------------------------------------------------- #
# 4. Subgroup metrics - the fairness check Task 1 never ran
# --------------------------------------------------------------------------- #

def subgroups(X, y, model, groups: dict, dataset: str):
    """groups: {'label': pandas Series aligned to X} -> per-level metrics."""
    print(f"\n{'=' * 74}\n{dataset}  |  subgroup performance "
          f"(single split, seed {SEED})\n{'=' * 74}")
    idx = np.arange(len(X))
    tr, te = train_test_split(idx, test_size=0.25, stratify=y, random_state=SEED)
    pipe = build_pipeline(X, model).fit(X.iloc[tr], y.iloc[tr])
    pred = pipe.predict(X.iloc[te])
    yte = y.iloc[te].to_numpy()

    for label, series in groups.items():
        print(f"\n-- by {label} --")
        print(f"{'level':<22}{'n':>6}{'base rate':>11}{'recall':>9}"
              f"{'precision':>11}{'flag rate':>11}")
        g = series.iloc[te].to_numpy()
        for level in pd.unique(g):
            m = g == level
            if m.sum() < 20:
                continue                      # too few to read anything into
            print(f"{str(level):<22}{m.sum():>6}{yte[m].mean():>11.3f}"
                  f"{recall_score(yte[m], pred[m], zero_division=0):>9.3f}"
                  f"{precision_score(yte[m], pred[m], zero_division=0):>11.3f}"
                  f"{pred[m].mean():>11.3f}")
    print("\nUnequal recall across levels = the model misses some groups more "
          "than others,\neven when overall accuracy looks acceptable.")


# --------------------------------------------------------------------------- #
# Dataset loaders
# --------------------------------------------------------------------------- #

def load_dropout(path):
    df = pd.read_csv(path, sep=";")
    df.columns = [c.strip() for c in df.columns]
    target = "Target" if "Target" in df.columns else df.columns[-1]
    y = (df[target].astype(str).str.strip() == "Dropout").astype(int)  # minority = 1
    X = df.drop(columns=[target])

    grp = {}
    for col, name in [("Gender", "gender"), ("International", "international"),
                      ("Scholarship holder", "scholarship holder"),
                      ("Debtor", "debtor")]:
        if col in X.columns:
            grp[name] = X[col]
    if "Age at enrollment" in X.columns:
        grp["age band"] = pd.cut(X["Age at enrollment"],
                                 [0, 20, 23, 30, 200],
                                 labels=["<=20", "21-23", "24-30", "30+"])
    return X, y, grp


def load_performance(path, positive="fail"):
    df = pd.read_csv(path, sep=";")
    df.columns = [c.strip().strip('"') for c in df.columns]
    passed = (df["G3"] >= 10).astype(int)
    # Task 1 used positive = pass (the 84.6% majority). 'fail' flips it so the
    # positive class is the minority "needs attention" group the rationale describes.
    y = passed if positive == "pass" else 1 - passed
    X = df.drop(columns=[c for c in ["G1", "G2", "G3"] if c in df.columns])

    grp = {}
    if "sex" in X.columns:
        grp["sex"] = X["sex"]
    if "address" in X.columns:
        grp["address (urban/rural)"] = X["address"]
    if "Medu" in X.columns:
        grp["mother's education"] = X["Medu"]
    if "age" in X.columns:
        grp["age band"] = pd.cut(X["age"], [0, 16, 17, 18, 100],
                                 labels=["<=16", "17", "18", "19+"])
    return X, y, grp


# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dropout", help="UCI 697 data.csv")
    ap.add_argument("--performance", help="UCI 320 student-por.csv")
    a = ap.parse_args()

    if a.dropout:
        X, y, grp = load_dropout(a.dropout)
        print(f"\nDropout: {len(X)} rows, positive (Dropout) rate {y.mean():.3f}")
        for name in models():
            report(name, "Student dropout", X, y)
        subgroups(X, y, models()["Logistic Regression"], grp, "Student dropout")

    if a.performance:
        for pos in ["pass", "fail"]:
            X, y, grp = load_performance(a.performance, positive=pos)
            print(f"\nPerformance: {len(X)} rows, positive ('{pos}') "
                  f"rate {y.mean():.3f}")
            for name in models():
                report(name, f"Student performance [positive = {pos}]", X, y)
        X, y, grp = load_performance(a.performance, positive="fail")
        subgroups(X, y, models()["SVM (RBF)"], grp, "Student performance")

    if not (a.dropout or a.performance):
        ap.error("give --dropout and/or --performance")


if __name__ == "__main__":
    main()
