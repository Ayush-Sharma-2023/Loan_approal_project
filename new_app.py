# streamlit_app_aesthetic.py
# Requirements: pip install streamlit pandas requests plotly
import streamlit as st
import pandas as pd
import requests
import json
import plotly.express as px
import plotly
from math import ceil
from html import escape

st.set_page_config(page_title="Loan Approval — Aesthetic", layout="wide", initial_sidebar_state="expanded")

# ---------- CSS / THEME ----------
# ---------- DARK THEME CSS ----------
st.markdown(
    """
    <style>
    /* Dark background */
    .stApp {
        background: #0d1117;
        color: #e5e7eb;
    }

    /* Hide hamburger + footer */
    #MainMenu, footer, header {visibility: hidden;}

    /* Card styling */
    .card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 12px;
        padding: 14px;
        transition: 0.15s ease-in-out;
        box-shadow: 0 4px 12px rgba(0,0,0,0.4);
    }
    .card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 24px rgba(0,0,0,0.6);
        background: rgba(255,255,255,0.05);
    }

    .muted {
        color: #94a3b8 !important;
        font-size: 12px;
    }

    .small {
        font-size: 13px;
        color: #e5e7eb;
    }

    /* Avatars */
    .avatar {
        width: 44px; height: 44px;
        border-radius: 50%;
        font-weight: 700;
        display:flex;
        align-items:center;
        justify-content:center;
        color: white;
        box-shadow: 0 0 10px rgba(255,255,255,0.15);
    }

    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 6px;
    }
    ::-webkit-scrollbar-thumb {
        background: #475569;
        border-radius: 10px;
    }

    /* Inputs / widgets */
    .stTextInput>div>div>input,
    .stNumberInput input {
        background: rgba(255,255,255,0.07) !important;
        color: white !important;
        border-radius: 6px;
    }

    /* Dropdowns */
    .stSelectbox div[data-baseweb="select"] {
        background: rgba(255,255,255,0.07) !important;
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)
# ---------------- Helpers ----------------
def load_json(source: str):
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
    num_cols = [
        "age","years_employed","monthly_income","total_assets","total_debt","existing_emi",
        "loan_amount_requested","loan_term_months","credit_score","past_defaults",
        "DTI","EMI_burden","asset_coverage","risk_score","model_prob","combined_prob",
        "suggested_annual_rate","suggested_monthly_emi","total_repayment","total_interest",
        "annualized_roi","priority_score"
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "category" in df.columns:
        df["category"] = df["category"].astype(str)
    if "priority_level" in df.columns:
        df["priority_level"] = df["priority_level"].astype(str)
    return df

def initials(name):
    if not name: return "?"
    parts = str(name).split()
    if len(parts) == 1: return parts[0][:2].upper()
    return (parts[0][0] + parts[-1][0]).upper()

def color_for(cat):
    return {
        "Green": "#22c55e",   # neon green
        "Yellow": "#facc15",  # bright yellow
        "Red": "#ff5252",     # neon red
        "Grey": "#9ca3af"     # soft grey
    }.get(cat, "#94a3b8")


def render_small_card(title, value, subtitle=""):
    return f"""
    <div class="card" style="background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
                              padding:12px;border-radius:12px;border:1px solid rgba(255,255,255,0.04);">
      <div style="font-size:12px;color:#94a3b8">{title}</div>
      <div style="font-size:18px;font-weight:700;margin-top:6px;color:#e6eef3">{value}</div>
      <div style="font-size:12px;color:#8b95a3;margin-top:6px">{subtitle}</div>
    </div>
    """


def render_applicant_card(r):
    name = escape(str(r.get("name","-")))
    aid = escape(str(r.get("id","-")))
    cat = r.get("category","-")
    pri = r.get("priority_level","-")
    risk = r.get("risk_score",0.0)
    inc = int(r.get("monthly_income",0))
    loan = int(r.get("loan_amount_requested",0))
    rate = r.get("suggested_annual_rate_pct","-")
    avatar_bg = color_for(cat)
    html = f"""
    <div class="card" style="background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)); padding:14px; border-radius:12px; border:1px solid rgba(255,255,255,0.04); margin-bottom:10px">

      <div style="display:flex;align-items:center;justify-content:space-between">
        <div style="display:flex;align-items:center;gap:12px">
          <div class="avatar" style="background:{avatar_bg}">{initials(name)}</div>
          <div>
            <div style="font-weight:700">{name} <span style="color:#6b7280;font-size:12px">id:{aid}</span></div>
            <div class="muted">{r.get('employment_type','')} • {r.get('age','-')} yrs</div>
          </div>
        </div>
        <div style="text-align:right">
          <div style="font-weight:800;color:{avatar_bg}">{cat}</div>
          <div class="muted" style="margin-top:6px">Priority: <b>{pri}</b></div>
        </div>
      </div>
      <div style="display:flex;justify-content:space-between;margin-top:12px">
        <div class="small">Income<br><b>₹{inc:,}</b></div>
        <div class="small">Loan<br><b>₹{loan:,}</b></div>
        <div class="small">Rate<br><b>{rate}</b></div>
        <div class="small">Risk<br><b>{risk:.2f}</b></div>
      </div>
    </div>
    """
    return html

# -------------- Sidebar controls --------------
st.sidebar.header("Source & Controls")
default_local = "sheet_with_predictions_updated.json"
data_source = st.sidebar.text_input("JSON file path or URL", value=default_local)
if st.sidebar.button("Load / Refresh"):
    st.session_state.pop("df", None)

st.sidebar.markdown("---")
st.sidebar.markdown("Filters")
min_income = st.sidebar.number_input("Min monthly income", value=0, step=1000)
max_income = st.sidebar.number_input("Max monthly income (0 = no limit)", value=0, step=1000)
show_priority = st.sidebar.selectbox("Priority", options=["All","High","Medium","Low","Very Low","Reject"])
show_table_toggle = st.sidebar.checkbox("Show compact table", value=False)
per_page = st.sidebar.number_input("Cards per page", value=8, min_value=4, max_value=24, step=4)

# -------------- Load data --------------
if "df" not in st.session_state:
    df = load_json(data_source)
    df = to_numeric_df(df)
    st.session_state.df = df
else:
    df = st.session_state.df

if df.empty:
    st.stop()

# filters
if max_income and max_income > 0:
    df = df[(df["monthly_income"] >= min_income) & (df["monthly_income"] <= max_income)]
else:
    df = df[df["monthly_income"] >= min_income]

if "priority_level" in df.columns and show_priority != "All":
    df = df[df["priority_level"] == show_priority]

# -------------- Header & KPIs --------------
st.markdown("<h1 style='margin-bottom:4px'>💳 Loan Approval — Demo</h1>", unsafe_allow_html=True)
st.markdown("<div class='muted'>A clean, aesthetic demo for your AI lab project — interactive and explorable.</div>", unsafe_allow_html=True)
st.markdown("<br>")

# KPIs row
total = len(df)
avg_risk = df["risk_score"].mean() if total>0 else 0.0
avg_rate = (df["suggested_annual_rate"].mean()*100) if "suggested_annual_rate" in df.columns and total>0 else 0.0
col1, col2, col3, col4 = st.columns([1.5,1,1,1])
col1.markdown(render_small_card("Total applicants", f"{total}", "Current dataset"), unsafe_allow_html=True)
col2.markdown(render_small_card("Avg risk", f"{avg_risk:.2f}", "Lower is better"), unsafe_allow_html=True)
col3.markdown(render_small_card("Avg suggested rate", f"{avg_rate:.2f}%", "Estimated"), unsafe_allow_html=True)
col4.markdown(render_small_card("Green ratio", f"{(df['category']=='Green').mean()*100:.0f}%", "Healthy demo"), unsafe_allow_html=True)
st.markdown("---")

# -------------- Category strip with interactive selection --------------
categories = ["Green","Yellow","Red","Grey"]
cards = []
for c in categories:
    sub = df[df["category"]==c]
    cnt = len(sub)
    avg_risk = sub["risk_score"].mean() if cnt>0 else 0.0
    cards.append((c, cnt, avg_risk))

cols = st.columns(4)
for i, (name, cnt, ar) in enumerate(cards):
    with cols[i]:
        if st.button(f"{name} ({cnt})", key=f"cat_{name}"):
            st.session_state.selected_category = name
        st.markdown(
            f"""<div class="card" style="padding:12px;border-radius:12px;border:1px solid rgba(15,23,42,0.04);">
                 <div style="font-size:13px;color:#374151">{name}</div>
                 <div style="font-size:18px;font-weight:700;color:{color_for(name)};margin-top:6px">{cnt}</div>
                 <div class="muted" style="margin-top:6px">Avg risk {ar:.2f}</div>
               </div>""",
            unsafe_allow_html=True)

if "selected_category" in st.session_state:
    st.markdown(f"**Filtered by:** {st.session_state.selected_category} — [clear]")
    if st.button("Clear filter"):
        st.session_state.pop("selected_category", None)

st.markdown("---")

st.subheader("Inspect single applicant")
app_id = st.text_input("Enter applicant id or name (partial match)")
if app_id:
    sub = df[df["id"].astype(str).str.lower() == app_id.strip().lower()]
    if sub.empty:
        sub = df[df["name"].str.lower().str.contains(app_id.strip().lower(), na=False)]
    if sub.empty:
        st.info("No matching applicant found.")
    else:
        r = sub.iloc[0]
        c1, c2 = st.columns([2,1])
        with c1:
            st.markdown(f"## {r.get('name','-')}  <span style='color:{color_for(r.get('category'))};font-weight:800'>{r.get('category')}</span>", unsafe_allow_html=True)
            st.markdown(f"<div class='muted'>Priority: <b>{r.get('priority_level','-')}</b> — Risk {r.get('risk_score',0):.3f}</div>", unsafe_allow_html=True)
            st.markdown("### Financials")
            st.write("Monthly income", f"₹{int(r.get('monthly_income',0)):,}")
            st.write("Loan requested", f"₹{int(r.get('loan_amount_requested',0)):,}")
            st.write("Tenure", f"{int(r.get('loan_term_months',0))} months")
            st.write("Suggested rate", r.get('suggested_annual_rate_pct','-'))
        with c2:
            st.markdown("### Lender view")
            st.write("EMI", f"₹{float(r.get('suggested_monthly_emi',0)):.2f}")
            st.write("Total interest", f"₹{float(r.get('total_interest',0)):.2f}")
            st.write("Annualized ROI", r.get('annualized_roi_pct','-'))
        st.markdown("#### Full data")
        st.json(r.to_dict())



# -------------- Layout: charts left, cards right --------------
left, right = st.columns((2,3))

with left:
    st.subheader("Quick filters")
    sel_cat = st.multiselect("Select categories", options=categories, default=categories if "selected_category" not in st.session_state else [st.session_state.selected_category])
    min_score, max_score = st.slider("Risk score range", 0.0, 1.0, (0.0, 1.0), 0.01)
    df_filtered = df[df["category"].isin(sel_cat)]
    df_filtered = df_filtered[(df_filtered["risk_score"] >= min_score) & (df_filtered["risk_score"] <= max_score)]

    st.subheader("Charts & Distribution")
    # Category donut
    fig = px.pie(df, names="category", title="Category share", color="category",
                 color_discrete_map={"Green":"#16a34a","Yellow":"#f59e0b","Red":"#ef4444","Grey":"#6b7280"})
    fig.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig, use_container_width=True)

    # Risk histogram
    # Risk histogram
    if "risk_score" in df.columns:
        fig2 = px.histogram(
            df, 
            x="risk_score", 
            nbins=15,
            title="Risk score distribution", 
            marginal="box"
        )

        fig2.update_layout(bargap=0.15)  
        fig2.update_traces(
            marker_line_width=1,
            marker_line_color="white"
        )

        st.plotly_chart(fig2, use_container_width=True)



with right:
    st.subheader("Applicants")
    # apply session selected category
    if "selected_category" in st.session_state:
        df_filtered = df_filtered[df_filtered["category"] == st.session_state.selected_category]

    df_filtered = df_filtered.sort_values(["priority_score","risk_score"], ascending=[False, True]).reset_index(drop=True)

    # pagination
    page = st.session_state.get("page", 1)
    per_page = int(per_page)
    total_pages = max(1, ceil(len(df_filtered)/per_page))
    start = (page-1)*per_page
    end = start + per_page
    page_slice = df_filtered.iloc[start:end]

    # grid of cards (2 columns)
    grid_cols = st.columns(2)
    i = 0
    for _, r in page_slice.iterrows():
        with grid_cols[i % 2]:
            st.markdown(render_applicant_card(r), unsafe_allow_html=True)
            with st.expander("Details & actions"):
                # pretty details (selective)
                left_d, right_d = st.columns(2)
                with left_d:
                    st.write("Income", f"₹{int(r.get('monthly_income',0)):,}")
                    st.write("Loan", f"₹{int(r.get('loan_amount_requested',0)):,}")
                    st.write("Tenure", f"{int(r.get('loan_term_months',0))} months")
                    st.write("Rate", r.get("suggested_annual_rate_pct","-"))
                with right_d:
                    st.write("Risk score", f"{r.get('risk_score',0):.3f}")
                    st.write("Combined prob", f"{r.get('combined_prob',0):.3f}")
                    st.write("Priority", r.get("priority_level","-"))
                st.markdown("---")
                st.write("Full record")
                st.json(r.to_dict())
                # action buttons
                action = st.radio("Action", options=["No action","Approve","Review","Reject"], index=0, key=f"act_{r.get('id')}")
        i += 1

    # pagination controls
    p1, p2, p3 = st.columns([1,2,1])
    with p1:
        if st.button("◀ Prev") and page > 1:
            st.session_state.page = page - 1
    with p2:
        st.write(f"Page {page} of {total_pages}")
    with p3:
        if st.button("Next ▶") and page < total_pages:
            st.session_state.page = page + 1

    # compact table toggle
    if show_table_toggle:
        st.markdown("### Compact table")
        show_cols = ["id","name","category","risk_score","priority_level","monthly_income","loan_amount_requested","suggested_annual_rate_pct"]
        show_cols = [c for c in show_cols if c in df_filtered.columns]
        st.table(df_filtered[show_cols].head(80).reset_index(drop=True))

    # download
    json_str = df_filtered.to_json(orient="records", indent=2)
    st.download_button("Download filtered JSON", data=json_str, file_name="filtered_applicants.json", mime="application/json")

st.markdown("---")



st.markdown("<div style='color:#777;font-size:12px'>Built with ❤️ — polished UI for demos. Want theme tweaks (dark mode / colors / compactness)? Tell me what vibe.</div>", unsafe_allow_html=True)
