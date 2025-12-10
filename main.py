# loan_pipeline_balanced.py
# Balanced loan pipeline (relaxed overrides + blended probability + softer thresholds)
# Requirements:
# pip install pandas scikit-learn requests joblib

import requests
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
import joblib
import json

# ------------- CONFIG -------------
SHEET_API_URL = "https://api.sheetbest.com/sheets/62c95aec-f89d-4c81-8194-ef274a80c0b9"
RANDOM_STATE = 42
MODEL_OUT = "loan_default_lr_balanced.joblib"
JSON_OUT = "sheet_with_predictions_balanced.json"

# Interest-rate params (tweakable)
BASE_RATE = 0.08           # 8% base annual rate
MAX_RISK_PREMIUM = 0.20    # up to +20% premium for high risk (so max rate = 28%)

# ------------- HELPERS -------------
def fetch_data(url):
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    return r.json()

def safe_numeric(series, dtype=float):
    return pd.to_numeric(series, errors="coerce").fillna(0).astype(dtype)

def suggested_annual_rate(risk_score):
    rate = BASE_RATE + float(risk_score) * MAX_RISK_PREMIUM
    return min(rate, BASE_RATE + MAX_RISK_PREMIUM)

def monthly_emi(principal, annual_rate, months):
    principal = float(principal)
    months = int(max(1, months))
    if principal <= 0:
        return 0.0
    r = float(annual_rate) / 12.0
    if r == 0:
        return principal / months
    emi = principal * r * (1 + r) ** months / ((1 + r) ** months - 1)
    return float(emi)

def annualized_roi(total_interest, principal, months):
    principal = float(principal)
    total_interest = float(total_interest)
    months = int(max(1, months))
    if principal <= 0 or total_interest <= 0:
        return 0.0
    years = months / 12.0
    return float((total_interest / principal) / years)

def compute_priority_score(row):
    risk_component = 1 - float(row["risk_score"])
    credit_component = float(row["credit_score"]) / 900.0
    asset_component = float(row["asset_coverage"])
    asset_component = min(asset_component, 2.0) / 2.0
    score = 0.60 * risk_component + 0.30 * credit_component + 0.10 * asset_component
    return float(max(0.0, min(1.0, score)))

def priority_level_from_score(s):
    if s >= 0.75:
        return "High"
    if s >= 0.50:
        return "Medium"
    if s >= 0.25:
        return "Low"
    return "Very Low"

# NEW: relaxed override logic
def is_grey_by_override(row):
    # Immediate Grey only for strong signals
    if str(row.get("legal_issues", "")).lower() == "yes":
        return True
    if float(row.get("credit_score", 999)) < 520:
        return True
    if float(row.get("past_defaults", 0)) >= 3:
        return True
    return False

# NEW: softer thresholds
def category_from_prob(p, row):
    # p is combined_prob in [0..1]
    if is_grey_by_override(row):
        return "Grey"
    if p < 0.40:
        return "Green"
    if p < 0.60:
        return "Yellow"
    if p < 0.80:
        return "Red"
    return "Grey"

# ------------- MAIN -------------
def main():
    print("Fetching sheet data...")
    raw = fetch_data(SHEET_API_URL)
    df = pd.DataFrame(raw)

    # normalize columns
    df.columns = [c.strip() for c in df.columns]

    # numeric conversions
    num_cols = [
        "age", "years_employed", "monthly_income", "total_assets",
        "total_debt", "existing_emi", "loan_amount_requested",
        "loan_term_months", "credit_score", "past_defaults"
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = safe_numeric(df[c])

    # normalize text columns
    for c in ["pan_verified", "legal_issues", "employment_type", "name", "id"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    # Derived metrics (same as before)
    df["DTI"] = df["total_debt"] / df["monthly_income"].replace({0: 1})
    df["EMI_burden"] = df["existing_emi"] / df["monthly_income"].replace({0: 1})
    df["asset_coverage"] = df["total_assets"] / df["loan_amount_requested"].replace({0: 1})
    df["credit_factor"] = 1 - (df["credit_score"] / 900.0)

    # normalize factors to avoid extreme domination
    DTI_max = df["DTI"].max() if df["DTI"].max() > 0 else 1.0
    EMI_max = df["EMI_burden"].max() if df["EMI_burden"].max() > 0 else 1.0
    past_defaults_max = df["past_defaults"].replace({0:1}).max()

    df["risk_score"] = (
        0.35 * (df["DTI"] / DTI_max) +
        0.25 * (df["EMI_burden"] / EMI_max) +
        0.25 * df["credit_factor"] +
        0.15 * (df["past_defaults"] / (past_defaults_max if past_defaults_max > 0 else 1))
    )
    df["risk_score"] = df["risk_score"].clip(0, 1)

    # create labels (less aggressive than before)
    df["default_label"] = 0
    # label defaults only for stronger signals
    df.loc[(df["credit_score"] < 500) | (df["past_defaults"] >= 3), "default_label"] = 1
    df.loc[(df["DTI"] > 2.0) & (df["EMI_burden"] > 1.0), "default_label"] = 1

    # Features
    feature_cols = ["DTI", "EMI_burden", "asset_coverage", "credit_score", "past_defaults", "monthly_income", "loan_amount_requested"]
    X = df[feature_cols].fillna(0)
    y = df["default_label"].astype(int)

    # Train model if possible
    clf = None
    if y.nunique() > 1 and len(df) >= 10:
        print("Training LogisticRegression model (balanced)...")
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y)
            clf = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
            clf.fit(X_train, y_train)
            # safer cross-val
            try:
                scores = cross_val_score(clf, X, y, cv=min(5, max(2, len(X)//10)), scoring="roc_auc")
                print(f"CV AUC scores: {scores.round(3)}  mean={scores.mean():.3f}")
            except Exception:
                pass
            # Evaluate
            y_pred_proba = clf.predict_proba(X_test)[:, 1]
            y_pred = (y_pred_proba >= 0.5).astype(int)
            print("Test ROC AUC:", roc_auc_score(y_test, y_pred_proba))
            print("Confusion matrix (test):")
            print(confusion_matrix(y_test, y_pred))
            print("Classification report (test):")
            print(classification_report(y_test, y_pred, zero_division=0))
            joblib.dump({"model": clf, "features": feature_cols}, MODEL_OUT)
            print(f"Model saved to {MODEL_OUT}")
        except Exception as e:
            print("Model training failed (fallback to risk_score). Error:", e)
            clf = None
    else:
        print("Not enough label variety or data to train model; using risk_score as proxy for probability.")

    # model probabilities or fallback
    if clf is not None:
        model_prob = clf.predict_proba(X)[:, 1]
    else:
        model_prob = df["risk_score"].values

    # BLEND model_prob and risk_score to avoid extremes
    alpha = 0.5  # weight for model_prob; (1-alpha) for rule risk_score
    df["model_prob"] = model_prob
    df["combined_prob"] = alpha * df["model_prob"] + (1 - alpha) * df["risk_score"]
    df["combined_prob"] = df["combined_prob"].clip(0, 1)

    # Map to categories with relaxed overrides + softer thresholds
    df["category"] = df.apply(lambda r: category_from_prob(float(r["combined_prob"]), r), axis=1)

    # Additional finance fields (EMI, ROI, rate)
    df["suggested_annual_rate"] = df["risk_score"].apply(suggested_annual_rate)
    df["suggested_monthly_emi"] = df.apply(lambda r: monthly_emi(
        principal=float(r["loan_amount_requested"]),
        annual_rate=float(r["suggested_annual_rate"]),
        months=int(max(1, r["loan_term_months"]))
    ), axis=1)
    df["total_repayment"] = df["suggested_monthly_emi"] * df["loan_term_months"]
    df["total_interest"] = df["total_repayment"] - df["loan_amount_requested"]
    df["annualized_roi"] = df.apply(lambda r: annualized_roi(
        total_interest=float(max(0, r["total_interest"])),
        principal=float(max(1, r["loan_amount_requested"])),
        months=int(max(1, r["loan_term_months"]))
    ), axis=1)

    # Priority
    df["priority_score"] = df.apply(compute_priority_score, axis=1)
    df["priority_level"] = df["priority_score"].apply(priority_level_from_score)
    df.loc[df["category"] == "Grey", ["priority_level", "priority_score"]] = ["Reject", 0.0]

    # human-readable
    df["suggested_annual_rate_pct"] = df["suggested_annual_rate"].apply(lambda r: f"{r*100:.2f}%")
    df["annualized_roi_pct"] = df["annualized_roi"].apply(lambda r: f"{r*100:.2f}%")

    # Output columns
    out_cols = [
        "id", "name", "age", "employment_type", "monthly_income", "total_assets", "total_debt",
        "loan_amount_requested", "loan_term_months", "credit_score", "past_defaults",
        "DTI", "EMI_burden", "asset_coverage", "risk_score", "model_prob", "combined_prob", "category",
        "suggested_annual_rate", "suggested_annual_rate_pct", "suggested_monthly_emi",
        "total_repayment", "total_interest", "annualized_roi", "annualized_roi_pct",
        "priority_score", "priority_level"
    ]
    out_cols = [c for c in out_cols if c in df.columns]
    df_out = df[out_cols].copy()

    # Save JSON
    df_out.to_json(JSON_OUT, orient="records", indent=4)
    print(f"Balanced JSON written to {JSON_OUT}")

    # preview
    preview = df_out.head(20).to_dict(orient="records")
    print("\n--- Preview (first 20 rows) ---")
    print(json.dumps(preview, indent=2))

if __name__ == "__main__":
    main()
