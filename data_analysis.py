"""
Individual Task 1: Part 1.3 -- Data Analysis
Case Studies in Data Science, RMIT University
Mohammad Alvi Ahasan (s4198032)

Trains Logistic Regression and an RBF-SVM on two education datasets and reports
Accuracy, Precision, Recall, F1 and ROC-AUC, plus the top logistic coefficients.

Datasets (download and place in ./data/):
  Dropout:     https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success
               (semicolon-separated CSV with a 'Target' column) -> data/dropout.csv
  Performance: https://archive.ics.uci.edu/dataset/320/student+performance
               (student-por.csv, semicolon-separated)          -> data/performance.csv

Run:  python data_analysis.py
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)

RANDOM_STATE = 42


def run(tag, X, y, num_cols, cat_cols):
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y)
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), num_cols),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
    ])
    print(f"\n===== {tag}  (n={len(X)}, positive rate = {y.mean():.1%}) =====")
    print(f"{'Model':22}{'Acc':>7}{'Prec':>7}{'Rec':>7}{'F1':>7}{'AUC':>7}")
    lr = None
    for name, clf in {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "SVM (RBF)": SVC(kernel="rbf", class_weight="balanced", probability=True),
    }.items():
        pipe = Pipeline([("pre", pre), ("clf", clf)]).fit(Xtr, ytr)
        pred = pipe.predict(Xte)
        proba = pipe.predict_proba(Xte)[:, 1]
        print(f"{name:22}"
              f"{accuracy_score(yte, pred):7.3f}"
              f"{precision_score(yte, pred):7.3f}"
              f"{recall_score(yte, pred):7.3f}"
              f"{f1_score(yte, pred):7.3f}"
              f"{roc_auc_score(yte, proba):7.3f}")
        if name.startswith("Logistic"):
            lr = pipe
    names = lr.named_steps["pre"].get_feature_names_out()
    coef = lr.named_steps["clf"].coef_[0]
    order = np.argsort(coef)
    print("  Top + drivers:", ", ".join(names[i] for i in order[::-1][:5]))
    print("  Top - drivers:", ", ".join(names[i] for i in order[:5]))


def main():
    # ---------- Student Dropout (at-risk detection) ----------
    d = pd.read_csv("data/dropout.csv", sep=";")
    d.columns = [c.strip() for c in d.columns]
    y1 = (d["Target"].str.strip() == "Dropout").astype(int)
    X1 = d.drop(columns=["Target"])
    num1 = X1.select_dtypes(include=[np.number]).columns.tolist()
    cat1 = [c for c in X1.columns if c not in num1]
    run("STUDENT DROPOUT (at-risk detection)", X1, y1, num1, cat1)

    # ---------- Student Performance (predict pass, no prior grades) ----------
    p = pd.read_csv("data/performance.csv", sep=";")
    y2 = (p["G3"] >= 10).astype(int)          # pass on the 0-20 scale
    X2 = p.drop(columns=["G1", "G2", "G3"])   # drop grade leakage -> early-warning
    num2 = X2.select_dtypes(include=[np.number]).columns.tolist()
    cat2 = [c for c in X2.columns if c not in num2]
    run("STUDENT PERFORMANCE (predict pass)", X2, y2, num2, cat2)


if __name__ == "__main__":
    main()
