# loan_pipeline_updated.py
# Robust loan pipeline with improved DTI/EMI handling and normalization
# Requirements: pip install pandas scikit-learn requests joblib

import requests
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
import joblib
import json

# ------------- CONFIG -------------
# SHEET_API_URL = "https://api.sheetbest.com/sheets/62c95aec-f89d-4c81-8194-ef274a80c0b9"
from dotenv import load_dotenv
import os

load_dotenv()

SHEET_API_URL = os.getenv("SHEET_API_URL")

RANDOM_STATE = 42
MODEL_OUT = "loan_default_lr_updated.joblib"
JSON_OUT = "sheet_with_predictions_updated.json"

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
    asset_component = float(row["asset_coverage"]) if row["asset_coverage"] is not None else 0.0
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

def is_grey_by_override(row):
    if str(row.get("legal_issues", "")).lower() == "yes":
        return True
    if float(row.get("credit_score", 999)) < 520:
        return True
    if float(row.get("past_defaults", 0)) >= 3:
        return True
    return False

def category_from_prob(p, row):
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
    if df.empty:
        print("No data loaded. Exiting.")
        return

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

    # ---------------- Improved DTI & EMI handling ----------------
    # Annual income (monthly * 12)
    df["annual_income"] = df["monthly_income"] * 12.0

    # DTI_raw = total_debt / annual_income (NaN if income==0)
    df["DTI_raw"] = df["total_debt"] / df["annual_income"].replace({0: np.nan})

    # Fallback for unemployed or zero income: use debt / max(total_assets, loan_amount_requested)
    # This gives a relative sense rather than exploding to huge numbers
    fallback_denom = df[["total_assets", "loan_amount_requested"]].max(axis=1).replace({0: np.nan})
    df["DTI_raw"] = df["DTI_raw"].fillna(df["total_debt"] / fallback_denom)

    # If still NaN (both assets and loan_amount zero), set to a small value
    df["DTI_raw"] = df["DTI_raw"].fillna(0.0)

    # Clip extreme values: DTI can't be infinite; cap to a sensible maximum (e.g., 10)
    df["DTI_raw"] = df["DTI_raw"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["DTI"] = df["DTI_raw"].clip(upper=10.0)

    # EMI burden robust:
    # Try existing_emi / monthly_income; fallback to existing_emi / max(1, loan_amount_requested/term)
    df["EMI_burden_raw"] = df["existing_emi"] / df["monthly_income"].replace({0: np.nan})
    fallback_monthly_payment = (df["loan_amount_requested"] / df["loan_term_months"].replace({0:1})).replace({0: np.nan})
    df["EMI_burden_raw"] = df["EMI_burden_raw"].fillna(df["existing_emi"] / fallback_monthly_payment)
    df["EMI_burden_raw"] = df["EMI_burden_raw"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    # cap EMI burden to 1 (100% of income) so it doesn't explode
    df["EMI_burden"] = df["EMI_burden_raw"].clip(upper=1.0)

    # asset_coverage (safe)
    df["asset_coverage"] = df["total_assets"] / df["loan_amount_requested"].replace({0: np.nan})
    df["asset_coverage"] = df["asset_coverage"].fillna(0.0)

    # ---------------- Winsorize & normalize DTI and EMI to 0..1 ----------------
    # Winsorize DTI at 99th percentile to remove extreme outliers influence
    p99 = df["DTI"].quantile(0.99)
    if np.isnan(p99) or p99 <= 0:
        p99 = df["DTI"].max() if df["DTI"].max() > 0 else 1.0
    df["DTI_w"] = df["DTI"].clip(upper=p99)

    # Normalize by 5th-95th percentile to get robust 0..1
    q05, q95 = df["DTI_w"].quantile(0.05), df["DTI_w"].quantile(0.95)
    if q95 - q05 <= 0:
        # fallback normalization
        df["DTI_norm"] = (df["DTI_w"] / (df["DTI_w"].max() if df["DTI_w"].max() > 0 else 1.0)).clip(0,1)
    else:
        df["DTI_norm"] = ((df["DTI_w"] - q05) / (q95 - q05)).clip(0,1)

    # Normalize EMI burden similarly (cap then scale)
    p99_emi = df["EMI_burden"].quantile(0.99)
    if np.isnan(p99_emi) or p99_emi <= 0:
        p99_emi = df["EMI_burden"].max() if df["EMI_burden"].max() > 0 else 1.0
    df["EMI_w"] = df["EMI_burden"].clip(upper=p99_emi)
    q05e, q95e = df["EMI_w"].quantile(0.05), df["EMI_w"].quantile(0.95)
    if q95e - q05e <= 0:
        df["EMI_norm"] = (df["EMI_w"] / (df["EMI_w"].max() if df["EMI_w"].max() > 0 else 1.0)).clip(0,1)
    else:
        df["EMI_norm"] = ((df["EMI_w"] - q05e) / (q95e - q05e)).clip(0,1)

    # credit_factor: 1 - score/900
    df["credit_factor"] = 1 - (df["credit_score"] / 900.0)
    df["credit_factor"] = df["credit_factor"].clip(0,1)

    # past_defaults normalized (cap at 5)
    df["past_defaults_capped"] = df["past_defaults"].clip(upper=5)
    df["past_defaults_norm"] = df["past_defaults_capped"] / 5.0

    # ---------------- risk_score using normalized features ----------------
    # Weights: DTI 35%, EMI 25%, credit 25%, past_defaults 15%
    df["risk_score"] = (
        0.35 * df["DTI_norm"] +
        0.25 * df["EMI_norm"] +
        0.25 * df["credit_factor"] +
        0.15 * df["past_defaults_norm"]
    ).clip(0,1)

    # ---------------- create training label (less aggressive) ----------------
    df["default_label"] = 0
    df.loc[(df["credit_score"] < 500) | (df["past_defaults"] >= 3), "default_label"] = 1
    df.loc[(df["DTI"] > 5.0) & (df["EMI_burden"] > 0.8), "default_label"] = 1

    # ---------------- Features for ML ----------------
    feature_cols = ["DTI_norm", "EMI_norm", "asset_coverage", "credit_score", "past_defaults"]
    X = df[feature_cols].fillna(0)
    y = df["default_label"].astype(int)

    # ---------------- Train model if possible ----------------
    clf = None
    if y.nunique() > 1 and len(df) >= 10:
        print("Training LogisticRegression model (robust)...")
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y)
            clf = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
            clf.fit(X_train, y_train)
            # safe cross-val
            try:
                scores = cross_val_score(clf, X, y, cv=min(5, max(2, len(X)//10)), scoring="roc_auc")
                print(f"CV AUC scores: {scores.round(3)}  mean={scores.mean():.3f}")
            except Exception:
                pass
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
            print("Model training failed; falling back to risk_score. Error:", e)
            clf = None
    else:
        print("Not enough label variety or data to train model; using risk_score as proxy for probability.")

    # ---------------- Probabilities & blending ----------------
    if clf is not None:
        model_prob = clf.predict_proba(X)[:, 1]
    else:
        model_prob = df["risk_score"].values

    alpha = 0.5  # blending weight
    df["model_prob"] = model_prob
    df["combined_prob"] = (alpha * df["model_prob"] + (1 - alpha) * df["risk_score"]).clip(0,1)

    # ---------------- Map to category ----------------
    df["category"] = df.apply(lambda r: category_from_prob(float(r["combined_prob"]), r), axis=1)

    # ---------------- Financial outputs (EMI, ROI, rate) ----------------
    df["suggested_annual_rate"] = df["risk_score"].apply(suggested_annual_rate)
    df["suggested_monthly_emi"] = df.apply(lambda r: monthly_emi(
        principal=float(r["loan_amount_requested"]),
        annual_rate=float(r["suggested_annual_rate"]),
        months=int(max(1, r.get("loan_term_months", 1)))
    ), axis=1)
    df["total_repayment"] = df["suggested_monthly_emi"] * df["loan_term_months"]
    df["total_interest"] = df["total_repayment"] - df["loan_amount_requested"]
    df["annualized_roi"] = df.apply(lambda r: annualized_roi(
        total_interest=float(max(0, r["total_interest"])),
        principal=float(max(1, r["loan_amount_requested"])),
        months=int(max(1, r["loan_term_months"]))
    ), axis=1)

    # ---------------- Priority ----------------
    df["priority_score"] = df.apply(compute_priority_score, axis=1)
    df["priority_level"] = df["priority_score"].apply(priority_level_from_score)
    df.loc[df["category"] == "Grey", ["priority_level", "priority_score"]] = ["Reject", 0.0]

    # human-readable percent strings
    df["suggested_annual_rate_pct"] = df["suggested_annual_rate"].apply(lambda r: f"{r*100:.2f}%")
    df["annualized_roi_pct"] = df["annualized_roi"].apply(lambda r: f"{r*100:.2f}%")

    # Select output columns
    out_cols = [
        "id", "name", "age", "employment_type", "monthly_income", "total_assets", "total_debt",
        "loan_amount_requested", "loan_term_months", "credit_score", "past_defaults",
        "DTI", "DTI_raw", "DTI_norm", "EMI_burden", "EMI_norm", "asset_coverage",
        "risk_score", "model_prob", "combined_prob", "category",
        "suggested_annual_rate", "suggested_annual_rate_pct", "suggested_monthly_emi",
        "total_repayment", "total_interest", "annualized_roi", "annualized_roi_pct",
        "priority_score", "priority_level"
    ]
    out_cols = [c for c in out_cols if c in df.columns]
    df_out = df[out_cols].copy()

    # Save JSON
    df_out.to_json(JSON_OUT, orient="records", indent=4)
    print(f"Updated JSON written to {JSON_OUT}")

    # Print preview
    preview = df_out.head(20).to_dict(orient="records")
    print("\n--- Preview (first 20 rows) ---")
    print(json.dumps(preview, indent=2))

if __name__ == "__main__":
    main()
