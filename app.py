#!/usr/bin/env python3
import argparse, json
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

def load_stock_csv(path):
    df = pd.read_csv(path, parse_dates=["Date"])
    if "Close" not in df.columns:
        raise ValueError("CSV must have 'Close' column.")
    df = df.sort_values("Date").reset_index(drop=True)
    df["day_index"] = np.arange(1, len(df)+1)
    return df

def train_models(df):
    X, y = df[["day_index"]].values, df["Close"].values
    models = {
        "rbf": make_pipeline(StandardScaler(), SVR(kernel="rbf", C=100, epsilon=0.1, gamma="scale")),
        "linear": make_pipeline(StandardScaler(), SVR(kernel="linear", C=100, epsilon=0.1)),
        "poly": make_pipeline(StandardScaler(), SVR(kernel="poly", C=100, epsilon=0.1, degree=2)),
    }
    for m in models.values():
        m.fit(X, y)
    return models

def plot_models(df, models, out="svr_models.png"):
    X, y = df[["day_index"]].values, df["Close"].values
    plt.figure(figsize=(10,6))
    plt.scatter(df["day_index"], y, color="black", label="Data")
    for name, model in models.items():
        plt.plot(df["day_index"], model.predict(X), label=f"{name.upper()} model")
    plt.xlabel("Day Index")
    plt.ylabel("Close Price")
    plt.title("Stock Price Prediction with SVR")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()

def predict(models, day_index):
    X_pred = np.array([[day_index]])
    return {name: float(m.predict(X_pred)[0]) for name,m in models.items()}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stock_csv", required=True)
    p.add_argument("--predict_day", type=int, default=None)
    args = p.parse_args()

    df = load_stock_csv(args.stock_csv)
    models = train_models(df)
    plot_models(df, models)
    next_day = args.predict_day or (df["day_index"].max() + 1)
    preds = predict(models, next_day)
    with open("predictions.json","w") as f:
        json.dump({"predict_day": int(next_day), **preds}, f, indent=2)
    print("Predictions saved to predictions.json")
    print(preds)

if __name__ == "__main__":
    main()
