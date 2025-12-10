# streamlit_app.py
# Requirements: pip install streamlit pandas requests
import streamlit as st
import pandas as pd
import requests
import json
from io import StringIO

st.set_page_config(page_title="Loan Approval Dashboard", layout="wide", initial_sidebar_state="expanded")

# ---------- Helpers ----------
def load_json(source: str):
    """Load JSON from a local path or HTTP URL."""
    try:
        if source.startswith("http://") or source.startswith("https://"):
            r = requests.get(source, timeout=10)
            r.raise_for_status()
            return pd.DataFrame(r.json())
        else:
            with open(source, "r", encoding="utf-8") as f:
                return pd.DataFrame(json.load(f))
    except Exception as e:
        st.error(f"Could not load data: {e}")
        return pd.DataFrame()

def to_numeric_df(df):
    # Attempt to convert common numeric columns
    num_cols = [
        "age", "years_employed", "monthly_income", "total_assets",
        "total_debt", "existing_emi", "loan_amount_requested",
        "loan_term_months", "credit_score", "past_defaults",
        "DTI", "EMI_burden", "asset_coverage", "risk_score",
        "prob_default", "suggested_annual_rate", "suggested_monthly_emi",
        "total_repayment", "total_interest", "annualized_roi", "priority_score"
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # Keep categories clean
    if "category" in df.columns:
        df["category"] = df["category"].astype(str)
    return df

def color_box_html(color_hex, title, big_text, sub_text):
    # Return a small HTML card to show as a colored box
    html = f"""
    <div style="
        background: linear-gradient(180deg, {color_hex}22, {color_hex}11);
        border-radius: 14px;
        padding: 14px;
        min-height: 110px;
        display:flex;
        flex-direction:column;
        justify-content:space-between;
        box-shadow: 0 6px 18px rgba(0,0,0,0.06);
        border: 1px solid {color_hex}40;
    ">
      <div style="font-size:13px; color:#333; opacity:0.9">{title}</div>
      <div style="font-size:22px; font-weight:700; color:{color_hex}; margin-top:6px">{big_text}</div>
      <div style="font-size:12px; color:#666; margin-top:6px">{sub_text}</div>
    </div>
    """
    return html

# ---------- Sidebar ----------
st.sidebar.header("Data source / Settings")
default_local = "sheet_with_predictions_updated.json"
data_source = st.sidebar.text_input("JSON file path or URL", value=default_local)
refresh = st.sidebar.button("Load / Refresh data")

st.sidebar.markdown("---")
st.sidebar.markdown("**Filters**")
min_income = st.sidebar.number_input("Min monthly income", value=0, step=1000)
max_income = st.sidebar.number_input("Max monthly income (0 = no limit)", value=0, step=1000)
show_only_priority = st.sidebar.selectbox("Show priority level", options=["All","High","Medium","Low","Very Low","Reject"])

# ---------- Load Data ----------
if "df" not in st.session_state or refresh:
    df = load_json(data_source)
    df = to_numeric_df(df)
    st.session_state.df = df
else:
    df = st.session_state.df

if df.empty:
    st.stop()

# Apply filters
if max_income and max_income > 0:
    df = df[(df["monthly_income"] >= min_income) & (df["monthly_income"] <= max_income)]
else:
    df = df[df["monthly_income"] >= min_income]

if "priority_level" in df.columns and show_only_priority != "All":
    df = df[df["priority_level"] == show_only_priority]

# ---------- Top summary cards ----------
st.markdown("<h2 style='margin-bottom:6px'>Loan Approval — AI Lab Dashboard</h2>", unsafe_allow_html=True)
st.markdown("<div style='color:#555;margin-top:-10px;margin-bottom:18px'>Visual summary of applicants by category. Click a card to expand details below.</div>", unsafe_allow_html=True)

# counts and avg risk per category
categories = ["Green","Yellow","Red","Grey"]
colors = {"Green":"#16a34a","Yellow":"#f59e0b","Red":"#ef4444","Grey":"#6b7280"}
cards = []
for c in categories:
    sub = df[df["category"]==c]
    count = len(sub)
    avg_risk = sub["risk_score"].mean() if count>0 else 0.0
    avg_rate = (sub["suggested_annual_rate"].mean()*100) if "suggested_annual_rate" in sub.columns and count>0 else 0.0
    cards.append((c, count, avg_risk, avg_rate))

col1, col2, col3, col4 = st.columns(4)
cols = [col1, col2, col3, col4]
for i, (cname, cnt, avgrisk, avgrate) in enumerate(cards):
    with cols[i]:
        html = color_box_html(
            colors[cname],
            f"{cname}",
            f"{cnt} applicants",
            f"Avg risk: {avgrisk:.2f} — Avg rate: {avgrate:.2f}%"
        )
        st.markdown(html, unsafe_allow_html=True)

st.markdown("---")

# ---------- Main content layout ----------
left, right = st.columns((2,3))

with left:
    st.subheader("Category breakdown")
    # bar chart counts
    counts = df["category"].value_counts().reindex(categories).fillna(0)
    st.bar_chart(counts)

    st.subheader("Risk distribution")
    # show histogram of risk_score
    if "risk_score" in df.columns:
        st.bar_chart(df["risk_score"].fillna(0), height=200)

    st.subheader("Quick filters")
    sel_cat = st.multiselect("Select categories to show", options=categories, default=categories)
    min_score = st.slider("Min risk score", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
    max_score = st.slider("Max risk score", min_value=0.0, max_value=1.0, value=1.0, step=0.01)

    # Apply these filters live
    df_filtered = df[df["category"].isin(sel_cat)]
    df_filtered = df_filtered[(df_filtered["risk_score"] >= min_score) & (df_filtered["risk_score"] <= max_score)]

    st.markdown("### Legend")
    st.write("- **Green**: low risk — prefer (lower interest).")
    st.write("- **Yellow**: acceptable — higher interest.")
    st.write("- **Red**: high risk — collateral or higher scrutiny.")
    st.write("- **Grey**: blacklist / reject.")

with right:
    st.subheader("Applicants (filtered)")
    st.dataframe(df_filtered.sort_values(["priority_score"], ascending=False).reset_index(drop=True), use_container_width=True)

    st.markdown("### Download")
    # prepare download JSON or CSV
    json_str = df_filtered.to_json(orient="records", indent=2)
    st.download_button("Download filtered JSON", data=json_str, file_name="filtered_applicants.json", mime="application/json")

st.markdown("---")

# ---------- Expandable category lists ----------
st.subheader("Applicants by category")
cats_exp = st.columns(4)
for i, cat in enumerate(categories):
    with cats_exp[i]:
        st.markdown(f"### {cat}")
        subset = df[df["category"]==cat].sort_values("priority_score", ascending=False)
        if subset.empty:
            st.write("No applicants")
            continue
        # show top 6
        for _, r in subset.head(6).iterrows():
            st.markdown(
                f"<div style='padding:8px;border-radius:8px;border:1px solid #eee;margin-bottom:8px'>"
                f"<b>{r.get('name','-')} (id: {r.get('id','-')})</b> — <span style='color:{colors[cat]};font-weight:700'>{cat}</span><br>"
                f"Income: ₹{int(r.get('monthly_income',0)):,} • Loan: ₹{int(r.get('loan_amount_requested',0)):,} • Risk: {r.get('risk_score',0):.2f} • Priority: {r.get('priority_level','-')}"
                f"</div>", unsafe_allow_html=True
            )
        if len(subset) > 6:
            if st.button(f"Show all {cat}", key=f"showall_{cat}"):
                st.dataframe(subset.reset_index(drop=True), use_container_width=True)

st.markdown("---")

# ---------- Single applicant inspector ----------
st.subheader("Inspect single applicant")
app_id = st.text_input("Enter applicant id or name (partial match)")
if app_id:
    # try numeric id first
    sub = df[df["id"].astype(str).str.lower() == app_id.strip().lower()]
    if sub.empty:
        sub = df[df["name"].str.lower().str.contains(app_id.strip().lower(), na=False)]
    if sub.empty:
        st.info("No matching applicant found.")
    else:
        r = sub.iloc[0]
        cols1, cols2 = st.columns(2)
        with cols1:
            st.markdown(f"**{r.get('name','-')}** (id: {r.get('id','-')})")
            st.write(f"Category: **{r.get('category','-')}**")
            st.write(f"Priority: **{r.get('priority_level','-')}** (score: {r.get('priority_score',0):.2f})")
            st.write(f"Risk score: {r.get('risk_score',0):.3f}")
            st.write(f"Prob default: {r.get('prob_default',0):.3f}")
        with cols2:
            st.write(f"Monthly income: ₹{int(r.get('monthly_income',0)):,}")
            st.write(f"Loan requested: ₹{int(r.get('loan_amount_requested',0)):,} for {int(r.get('loan_term_months',0))} months")
            st.write(f"Suggested rate: {r.get('suggested_annual_rate_pct','-')}  • EMI: ₹{float(r.get('suggested_monthly_emi',0)):.2f}")
            st.write(f"Total interest: ₹{float(r.get('total_interest',0)):.2f}  • ROI: {r.get('annualized_roi_pct','-')}")

st.markdown("<div style='color:#777;font-size:12px'>Built with ❤️ for quick demos. Use the JSON source box in the sidebar to point to your online file (sheet_with_predictions.json or a URL).</div>", unsafe_allow_html=True)
