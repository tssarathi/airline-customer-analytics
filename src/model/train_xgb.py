import argparse
import json
import os

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import roc_auc_score, average_precision_score

try:
    from xgboost import XGBClassifier
except ImportError:
    import subprocess
    import sys

    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "xgboost", "-q"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    from xgboost import XGBClassifier

import joblib


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--recency_threshold", type=int, default=3)
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.join(args.out, "artifacts"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "predictions"), exist_ok=True)

    df = pd.read_parquet(args.input)
    df["churn_label"] = (
        df["is_cancelled"] | (df["recency"] >= args.recency_threshold)
    ).astype(int)

    num = ["frequency", "monetary", "tenure_months", "clv"]
    cat = ["province", "city", "gender", "education", "loyalty_card"]

    X = df[num + cat].copy()
    y = df["churn_label"].astype(int)

    X[num] = X[num].replace([np.inf, -np.inf], np.nan).fillna(0)
    for c in cat:
        X[c] = X[c].astype("string").fillna("Unknown")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    preprocess = ColumnTransformer(
        [
            ("num", StandardScaler(with_mean=False), num),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
        ]
    )

    model = Pipeline(
        [
            ("preprocess", preprocess),
            (
                "clf",
                XGBClassifier(
                    n_estimators=400,
                    learning_rate=0.05,
                    max_depth=4,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_lambda=1.0,
                    objective="binary:logistic",
                    eval_metric="aucpr",
                    tree_method="hist",
                    random_state=42,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)

    proba_test = model.predict_proba(X_test)[:, 1]
    metrics = {
        "roc_auc": float(roc_auc_score(y_test, proba_test)),
        "pr_auc": float(average_precision_score(y_test, proba_test)),
        "churn_rate": float(df["churn_label"].mean()),
        "recency_threshold_months": int(args.recency_threshold),
        "n_customers": int(len(df)),
    }

    churn_prob = model.predict_proba(X)[:, 1]
    preds = pd.DataFrame(
        {
            "loyalty_number": df["loyalty_number"],
            "rfm_segment": df["rfm_segment"],
            "churn_label": df["churn_label"],
            "churn_prob": churn_prob,
            "clv": df["clv"],
        }
    )
    preds["priority_score"] = preds["churn_prob"] * preds["clv"]
    preds = preds.sort_values("priority_score", ascending=False).reset_index(drop=True)

    joblib.dump(model, os.path.join(args.out, "artifacts", "model.pkl"))
    with open(os.path.join(args.out, "artifacts", "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    preds.to_parquet(
        os.path.join(args.out, "predictions", "predictions.parquet"), index=False
    )

    print(metrics)


if __name__ == "__main__":
    main()
