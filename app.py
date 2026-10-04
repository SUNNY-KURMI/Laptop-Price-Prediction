import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st

BASE = Path(__file__).parent
FEATURES = ["Company", "TypeName", "CpuBrand", "GpuBrand", "OpSys",
            "Inches", "Ram", "Weight", "Touchscreen", "IPS", "SSD", "HDD"]

st.set_page_config(
    page_title="Laptop Price Tracker",
    page_icon="\U0001F4BB",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ---------------- Secrets (works locally and on Streamlit Cloud) ----------------
def get_secret(key: str, default=None):
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.getenv(key, default)


SERPAPI_KEY = get_secret("SERPAPI_KEY")


# ---------------- ML model (loaded once, in-process) ----------------
@st.cache_resource(show_spinner=False)
def load_model():
    path = BASE / "model" / "laptop_model.pkl"
    if not path.exists():
        return None
    return joblib.load(path)


@st.cache_data(show_spinner=False)
def load_metrics():
    path = BASE / "model" / "metrics.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


model = load_model()
metrics = load_metrics()


def predict_price(specs: dict, live_price: float | None = None) -> dict:
    X = pd.DataFrame([specs])[FEATURES]
    predicted = float(np.exp(model.predict(X)[0]))
    result = {"predicted_price": round(predicted)}
    if live_price:
        ratio = live_price / predicted
        if ratio < 0.9:
            verdict = "Good deal"
        elif ratio > 1.1:
            verdict = "Overpriced"
        else:
            verdict = "Fair price"
        result.update({
            "live_price": live_price,
            "difference_percent": round((ratio - 1) * 100, 1),
            "verdict": verdict,
        })
    return result


# ---------------- Firestore (optional; app works without it) ----------------
@st.cache_resource(show_spinner=False)
def get_db():
    try:
        import firebase_admin
        from firebase_admin import credentials, firestore
    except ImportError:
        return None

    try:
        if "firebase" in st.secrets:
            cred_dict = dict(st.secrets["firebase"])
            cred = credentials.Certificate(cred_dict)
        else:
            key_path = BASE / "serviceAccountKey.json"
            if not key_path.exists():
                return None
            cred = credentials.Certificate(str(key_path))
        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred)
        return firestore.client()
    except Exception as e:
        print("Firestore not configured:", e)
        return None


db = get_db()
_price_cache = {}  # {query: (timestamp, data)}


def fetch_prices(q: str) -> list:
    key = q.lower().strip()
    if key in _price_cache and time.time() - _price_cache[key][0] < 1800:
        return _price_cache[key][1]
    if not SERPAPI_KEY:
        return []

    r = requests.get(
        "https://serpapi.com/search.json",
        params={"engine": "google_shopping", "q": q, "gl": "in", "hl": "en", "api_key": SERPAPI_KEY},
        timeout=20,
    )
    r.raise_for_status()
    items = []
    for it in r.json().get("shopping_results", [])[:10]:
        if it.get("extracted_price"):
            items.append({
                "title": it.get("title"),
                "price": it["extracted_price"],
                "source": it.get("source"),
                "link": it.get("product_link") or it.get("link"),
            })

    if items and db is not None:
        try:
            ts = datetime.now(timezone.utc)
            batch = db.batch()
            for it in items:
                doc = {"query": key, "title": it["title"], "price": it["price"],
                       "source": it["source"], "timestamp": ts}
                batch.set(db.collection("price_history").document(), doc)
            batch.commit()
        except Exception as e:
            print("Firestore write failed:", e)

    _price_cache[key] = (time.time(), items)
    return items


def fetch_history(q: str) -> list:
    if db is None:
        return []
    from google.cloud.firestore_v1.base_query import FieldFilter
    key = q.lower().strip()
    docs = db.collection("price_history").where(filter=FieldFilter("query", "==", key)).limit(500).stream()
    rows = []
    for d in docs:
        row = d.to_dict()
        row["timestamp"] = row["timestamp"].isoformat()
        rows.append(row)
    rows.sort(key=lambda r: r["timestamp"])
    return rows


# ---------------- Design tokens ----------------
INK = "#14211E"
MUTED = "#5B6B67"
LINE = "#D9E0DD"
MIST = "#F3F5F4"
PINE = "#0F5C4D"
TAG = "#F5B301"
GOOD = "#15803D"
WARN = "#B42318"

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700;12..96,800&family=Figtree:wght@400;500;600;700&display=swap');

    :root {{
        --ink: {INK}; --muted: {MUTED}; --line: {LINE}; --mist: {MIST};
        --pine: {PINE}; --tag: {TAG}; --good: {GOOD}; --warn: {WARN};
    }}

    html, body, [class*="css"], .stApp {{
        font-family: 'Figtree', -apple-system, BlinkMacSystemFont, sans-serif;
        color: var(--ink);
    }}
    .stApp {{ background: var(--mist); }}
    header[data-testid="stHeader"] {{ background: transparent; }}
    .block-container {{ max-width: 1180px; padding-top: 2rem; padding-bottom: 4rem; }}

    h1, h2, h3, h4, .brand-name, .kpi-value, .tag-price {{
        font-family: 'Bricolage Grotesque', 'Figtree', sans-serif;
        letter-spacing: -0.02em;
    }}

    .masthead {{
        display: flex; align-items: center; justify-content: space-between;
        gap: 16px; flex-wrap: wrap;
        padding-bottom: 20px; margin-bottom: 8px;
        border-bottom: 2px solid var(--ink);
    }}
    .brand {{ display: flex; align-items: center; gap: 14px; }}
    .brand-mark {{
        width: 44px; height: 44px; border-radius: 10px 10px 10px 2px;
        background: var(--pine); color: #fff; font-size: 1.4rem;
        display: grid; place-items: center;
    }}
    .brand-name {{ font-size: 1.75rem; font-weight: 800; line-height: 1.1; }}
    .brand-sub {{ color: var(--muted); font-size: 0.95rem; margin-top: 2px; }}
    .status {{
        display: inline-flex; align-items: center; gap: 8px;
        border: 1px solid var(--line); background: #fff; border-radius: 999px;
        padding: 6px 14px; font-size: 0.85rem; font-weight: 600;
    }}
    .dot {{ width: 9px; height: 9px; border-radius: 50%; background: var(--good); }}
    .dot.off {{ background: var(--warn); }}

    .stTabs [data-baseweb="tab-list"] {{ gap: 4px; border-bottom: 1px solid var(--line); }}
    .stTabs [data-baseweb="tab"] {{
        height: 48px; padding: 0 18px; font-weight: 600; font-size: 0.95rem;
        color: var(--muted); border-radius: 8px 8px 0 0;
    }}
    .stTabs [aria-selected="true"] {{ color: var(--pine) !important; }}
    .stTabs [data-baseweb="tab-highlight"] {{ background-color: var(--pine); height: 3px; }}
    .stTabs [data-baseweb="tab-panel"] {{ padding-top: 1.4rem; }}

    div[data-testid="stVerticalBlockBorderWrapper"] {{ border-radius: 12px; }}
    div[data-testid="stVerticalBlockBorderWrapper"]:has(> div > div[data-testid="stVerticalBlock"] .panel-title) {{
        background: #fff; border-color: var(--line);
    }}
    .panel-title {{ font-weight: 700; font-size: 1rem; margin-bottom: 2px; }}
    .panel-hint {{ color: var(--muted); font-size: 0.88rem; margin-bottom: 10px; }}

    .stButton > button, .stFormSubmitButton > button {{
        border-radius: 10px; font-weight: 600; min-height: 44px;
        border: 1px solid var(--line);
    }}
    .stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"] {{
        background: var(--pine); border-color: var(--pine); color: #fff;
    }}
    .stButton > button[kind="primary"]:hover, .stFormSubmitButton > button[kind="primary"]:hover {{
        background: #0B4A3E; border-color: #0B4A3E;
    }}
    .stButton > button:focus-visible, .stFormSubmitButton > button:focus-visible {{
        outline: 3px solid var(--tag); outline-offset: 2px;
    }}

    .kpis {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 12px; margin: 6px 0 18px; }}
    .kpi {{ background: #fff; border: 1px solid var(--line); border-radius: 12px; padding: 14px 18px; }}
    .kpi.lead {{ border-left: 5px solid var(--tag); }}
    .kpi-label {{ font-size: 0.85rem; color: var(--muted); font-weight: 600; }}
    .kpi-value {{ font-size: 1.75rem; font-weight: 800; margin-top: 2px; }}
    .kpi-note {{ font-size: 0.82rem; color: var(--muted); margin-top: 2px; }}
    .kpi-note.down {{ color: var(--good); font-weight: 600; }}
    .kpi-note.up {{ color: var(--warn); font-weight: 600; }}

    .listing {{
        display: grid; grid-template-columns: 1fr auto; gap: 6px 20px; align-items: center;
        background: #fff; border: 1px solid var(--line); border-radius: 12px;
        padding: 14px 18px; margin-bottom: 10px;
    }}
    .listing.best {{ border: 2px solid var(--pine); }}
    .listing-title {{ font-weight: 600; line-height: 1.35; }}
    .listing-meta {{ color: var(--muted); font-size: 0.85rem; }}
    .listing-price {{ font-family: 'Bricolage Grotesque', sans-serif; font-weight: 800; font-size: 1.4rem; text-align: right; }}
    .listing a {{ color: var(--pine); font-weight: 600; font-size: 0.88rem; text-decoration: none; }}
    .listing a:hover {{ text-decoration: underline; }}
    .flag {{
        display: inline-block; background: var(--tag); color: var(--ink);
        font-size: 0.75rem; font-weight: 700; padding: 2px 8px; border-radius: 4px; margin-right: 8px;
    }}

    .tag-wrap {{ display: flex; gap: 24px; align-items: stretch; flex-wrap: wrap; margin-top: 8px; }}
    .price-tag {{
        position: relative; background: var(--tag); color: var(--ink);
        padding: 22px 34px 22px 46px; min-width: 280px;
        clip-path: polygon(24px 0, 100% 0, 100% 100%, 24px 100%, 0 50%);
    }}
    .price-tag::before {{
        content: ""; position: absolute; left: 24px; top: 50%; width: 12px; height: 12px;
        margin-top: -6px; border-radius: 50%; background: var(--mist);
    }}
    .tag-label {{ font-size: 0.85rem; font-weight: 700; }}
    .tag-price {{ font-size: 2.6rem; font-weight: 800; line-height: 1.1; }}
    .verdict {{
        flex: 1; min-width: 280px; background: #fff; border: 1px solid var(--line);
        border-radius: 12px; padding: 18px 22px;
    }}
    .verdict-head {{ font-family: 'Bricolage Grotesque', sans-serif; font-size: 1.4rem; font-weight: 800; }}
    .verdict.good .verdict-head {{ color: var(--good); }}
    .verdict.over .verdict-head {{ color: var(--warn); }}
    .verdict.fair .verdict-head {{ color: var(--pine); }}
    .verdict-copy {{ color: var(--muted); margin: 4px 0 16px; }}
    .scale {{ position: relative; height: 10px; border-radius: 6px;
        background: linear-gradient(90deg, var(--good) 0 33.3%, #9FB8B1 33.3% 66.6%, var(--warn) 66.6% 100%); }}
    .pin {{ position: absolute; top: -6px; width: 4px; height: 22px; background: var(--ink); border-radius: 2px; }}
    .scale-labels {{ display: flex; justify-content: space-between; font-size: 0.78rem; color: var(--muted); margin-top: 8px; }}

    .chips {{ display: flex; flex-wrap: wrap; gap: 8px; margin: 14px 0 4px; }}
    .chip {{ background: #fff; border: 1px solid var(--line); border-radius: 999px; padding: 4px 12px; font-size: 0.85rem; font-weight: 500; }}

    .empty {{
        border: 1.5px dashed var(--line); border-radius: 12px; padding: 36px 20px;
        text-align: center; color: var(--muted); background: rgba(255,255,255,0.5);
    }}
    .empty strong {{ display: block; color: var(--ink); font-size: 1.05rem; margin-bottom: 4px; }}

    .metric-card {{ background: #fff; border: 1px solid var(--line); border-radius: 12px; padding: 16px 18px; }}
    .metric-label {{ font-size: 0.82rem; color: var(--muted); font-weight: 600; }}
    .metric-value {{ font-size: 1.5rem; font-weight: 800; margin-top: 2px; }}

    @media (max-width: 640px) {{
        .brand-name {{ font-size: 1.4rem; }}
        .listing {{ grid-template-columns: 1fr; }}
        .listing-price {{ text-align: left; }}
        .price-tag {{ min-width: 100%; }}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------- Helpers ----------------
def inr(value) -> str:
    return f"₹{int(round(value)):,}"


def kpi(label, value, note="", cls="", lead=False) -> str:
    note_html = f'<div class="kpi-note {cls}">{note}</div>' if note else ""
    return (
        f'<div class="kpi{" lead" if lead else ""}"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>{note_html}</div>'
    )


def set_query(key: str, value: str):
    st.session_state[key] = value


def panel_open(title: str, hint: str = ""):
    st.markdown(f'<div class="panel-title">{title}</div>', unsafe_allow_html=True)
    if hint:
        st.markdown(f'<div class="panel-hint">{hint}</div>', unsafe_allow_html=True)


SUGGESTIONS = ["HP Pavilion 15", "Dell XPS 13", "Asus ROG Zephyrus", "MacBook Air M2", "Lenovo IdeaPad Slim 3"]

# ---------------- Masthead ----------------
model_ready = model is not None
st.markdown(
    f"""
    <div class="masthead">
        <div class="brand">
            <div class="brand-mark">\U0001F4BB</div>
            <div>
                <div class="brand-name">Laptop Price Tracker</div>
                <div class="brand-sub">Compare live prices, check what a laptop should cost, and watch prices over time.</div>
            </div>
        </div>
        <div class="status"><span class="dot{'' if model_ready else ' off'}"></span>{'Model loaded' if model_ready else 'Model missing'}</div>
    </div>
    """,
    unsafe_allow_html=True,
)
if not model_ready:
    st.error("model/laptop_model.pkl was not found. Run `python train.py` inside the `ml` folder first.")
if not SERPAPI_KEY:
    st.info("No SERPAPI_KEY configured, so 'Compare prices' will return no live listings. Add one in `.streamlit/secrets.toml` to enable it.")
if db is None:
    st.caption("Price history isn't configured (no Firebase credentials found) — predictions and searches still work, but won't be saved across sessions.")

tab1, tab2, tab3, tab4 = st.tabs(["Compare prices", "Fair price check", "Price history", "Model info"])

# ---------------- Tab 1: Live prices ----------------
with tab1:
    with st.container(border=True):
        panel_open("Search for a laptop", "Enter a model name to see what sellers are charging right now.")
        c_search, c_btn = st.columns([5, 1], vertical_alignment="bottom")
        with c_search:
            q = st.text_input("Laptop model", "HP Pavilion 15", key="live_q", label_visibility="collapsed",
                              placeholder="e.g. Asus ROG Zephyrus, Dell XPS 13")
        with c_btn:
            fetch_btn = st.button("Find prices", width="stretch", type="primary", key="fetch")
        chip_cols = st.columns(len(SUGGESTIONS))
        for col, s in zip(chip_cols, SUGGESTIONS):
            col.button(s, key=f"sug_{s}", width="stretch", on_click=set_query, args=("live_q", s))

    if fetch_btn:
        with st.spinner("Checking seller prices..."):
            try:
                rows = fetch_prices(q)
                st.session_state["live_result"] = {"q": q, "rows": rows, "error": None}
            except Exception as err:
                st.session_state["live_result"] = {"q": q, "rows": [], "error": str(err)}

    result = st.session_state.get("live_result")
    if not result:
        st.markdown(
            '<div class="empty"><strong>No search yet</strong>Pick a suggestion or type a model, then choose Find prices.</div>',
            unsafe_allow_html=True,
        )
    elif result["error"]:
        st.error(f"Couldn't load prices for “{result['q']}”. {result['error']}")
    elif not result["rows"]:
        st.warning(f"No listings found for “{result['q']}”. Try a shorter model name, or check that SERPAPI_KEY is set.")
    else:
        dfp = pd.DataFrame(result["rows"])
        low, avg = dfp["price"].min(), dfp["price"].mean()
        best = dfp.loc[dfp["price"].idxmin()]
        save = avg - low
        st.markdown(
            '<div class="kpis">'
            + kpi("Lowest price", inr(low), f"at {best['source']}", lead=True)
            + kpi("Average price", inr(avg), f"{inr(save)} above the lowest", "")
            + kpi("Listings", f"{len(dfp)}", f"from {dfp['source'].nunique()} sellers")
            + "</div>",
            unsafe_allow_html=True,
        )

        list_col, chart_col = st.columns([3, 2], gap="large")
        with list_col:
            top, sort_col = st.columns([2, 1], vertical_alignment="center")
            top.markdown(f"#### Results for “{result['q']}”")
            order = sort_col.selectbox("Sort by", ["Lowest price", "Highest price", "Seller"], label_visibility="collapsed")
            if order == "Lowest price":
                view = dfp.sort_values("price")
            elif order == "Highest price":
                view = dfp.sort_values("price", ascending=False)
            else:
                view = dfp.sort_values(["source", "price"])

            html = ""
            for _, r in view.iterrows():
                is_best = r["price"] == low
                flag = '<span class="flag">Lowest</span>' if is_best else ""
                link = f'<a href="{r["link"]}" target="_blank" rel="noopener">View at {r["source"]} ↗</a>' if r.get("link") else ""
                title = str(r["title"]).replace("<", "&lt;")
                html += (
                    f'<div class="listing{" best" if is_best else ""}">'
                    f'<div><div class="listing-title">{flag}{title}</div>'
                    f'<div class="listing-meta">{r["source"]}</div></div>'
                    f'<div><div class="listing-price">{inr(r["price"])}</div>'
                    f'<div style="text-align:right">{link}</div></div></div>'
                )
            st.markdown(html, unsafe_allow_html=True)

        with chart_col:
            st.markdown("#### Price by seller")
            by_seller = dfp.groupby("source", as_index=False)["price"].min().sort_values("price")
            bars = (
                alt.Chart(by_seller)
                .mark_bar(cornerRadiusEnd=4, color=PINE)
                .encode(
                    y=alt.Y("source:N", sort="x", title=None),
                    x=alt.X("price:Q", title="Lowest listing (₹)", axis=alt.Axis(format="~s")),
                    tooltip=[alt.Tooltip("source:N", title="Seller"), alt.Tooltip("price:Q", title="Price (₹)", format=",.0f")],
                )
                .properties(height=max(160, 34 * len(by_seller)))
            )
            avg_rule = alt.Chart(pd.DataFrame({"p": [avg]})).mark_rule(color=TAG, strokeWidth=3).encode(x="p:Q")
            st.altair_chart(bars + avg_rule, width="stretch")
            st.caption("The yellow line marks the average price across all listings.")

# ---------------- Tab 2: Prediction ----------------
with tab2:
    with st.form("predict_form", border=False):
        with st.container(border=True):
            panel_open("Brand and performance", "The basics that drive most of a laptop's price.")
            c1, c2, c3 = st.columns(3)
            with c1:
                company = st.selectbox("Brand", ["HP", "Dell", "Lenovo", "Asus", "Acer", "Apple", "MSI", "Toshiba", "Samsung"])
                type_name = st.selectbox("Category", ["Notebook", "Ultrabook", "Gaming", "2 in 1 Convertible", "Workstation", "Netbook"])
            with c2:
                cpu = st.selectbox("Processor", ["Intel Core i3", "Intel Core i5", "Intel Core i7", "Other Intel", "AMD"], index=1)
                gpu = st.selectbox("Graphics", ["Intel", "Nvidia", "AMD"])
            with c3:
                os_name = st.selectbox("Operating system", ["Windows 10", "macOS", "Linux", "Chrome OS", "No OS"])
                ram = st.selectbox("Memory (GB RAM)", [4, 8, 16, 32, 64], index=1)

        with st.container(border=True):
            panel_open("Storage and display", "Storage sizes are in GB; 1024 GB is 1 TB.")
            c4, c5, c6 = st.columns(3)
            with c4:
                ssd = st.selectbox("SSD storage (GB)", [0, 128, 256, 512, 1024], index=2)
                hdd = st.selectbox("HDD storage (GB)", [0, 500, 1024, 2048])
            with c5:
                inches = st.slider("Screen size (inches)", 11.0, 18.0, 15.6, 0.1)
                weight = st.number_input("Weight (kg)", 0.8, 4.5, 1.9, 0.1)
            with c6:
                st.markdown("**Screen features**")
                touch = st.checkbox("Touchscreen")
                ips = st.checkbox("IPS panel")

        with st.container(border=True):
            panel_open("Have a price in mind? (optional)", "Enter the asking price and we'll tell you if it's a good deal.")
            live_price = st.number_input("Asking price (₹)", min_value=0, value=0, step=1000)

        predict_btn = st.form_submit_button("Check fair price", type="primary", width="stretch", disabled=not model_ready)

    if predict_btn:
        specs = {
            "Company": company, "TypeName": type_name, "Inches": inches, "Ram": ram,
            "Weight": weight, "Touchscreen": int(touch), "IPS": int(ips),
            "SSD": ssd, "HDD": hdd, "CpuBrand": cpu, "GpuBrand": gpu, "OpSys": os_name,
        }
        spec_chips = [company, type_name, cpu, f"{gpu} graphics", f"{ram} GB RAM",
                      f"{ssd} GB SSD" if ssd else None, f"{hdd} GB HDD" if hdd else None,
                      f'{inches}"', f"{weight} kg", "Touchscreen" if touch else None, "IPS" if ips else None, os_name]
        with st.spinner("Estimating fair price..."):
            try:
                out = predict_price(specs, live_price or None)
                st.session_state["pred_result"] = {"ok": True, "out": out, "specs": [s for s in spec_chips if s]}
            except Exception as err:
                st.session_state["pred_result"] = {"ok": False, "out": str(err), "specs": []}

    pr = st.session_state.get("pred_result")
    if pr:
        if not pr["ok"]:
            st.error(f"Couldn't get an estimate. {pr['out']}")
        else:
            out = pr["out"]
            pred_price = out["predicted_price"]
            chips = "".join(f'<span class="chip">{s}</span>' for s in pr["specs"])

            verdict_html = ""
            if "verdict" in out:
                v, diff = out["verdict"], out.get("difference_percent", 0)
                cls = {"Good deal": "good", "Overpriced": "over"}.get(v, "fair")
                if diff < 0:
                    copy = f"The asking price is {abs(diff)}% below the estimate."
                elif diff > 0:
                    copy = f"The asking price is {diff}% above the estimate."
                else:
                    copy = "The asking price matches the estimate."
                pos = min(max((diff + 30) / 60 * 100, 2), 98)
                verdict_html = (
                    f'<div class="verdict {cls}"><div class="verdict-head">{v}</div>'
                    f'<div class="verdict-copy">{copy} Asking: {inr(out.get("live_price", 0))}</div>'
                    f'<div class="scale"><div class="pin" style="left:{pos}%"></div></div>'
                    f'<div class="scale-labels"><span>30% below</span><span>Fair range</span><span>30% above</span></div></div>'
                )
            else:
                verdict_html = (
                    '<div class="verdict fair"><div class="verdict-head">Estimate only</div>'
                    '<div class="verdict-copy">Add an asking price in the form above to see how it compares.</div></div>'
                )

            st.markdown("#### Your estimate")
            st.markdown(
                f'<div class="tag-wrap"><div class="price-tag"><div class="tag-label">Estimated fair price</div>'
                f'<div class="tag-price">{inr(pred_price)}</div></div>{verdict_html}</div>'
                f'<div class="chips">{chips}</div>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            '<div class="empty"><strong>No estimate yet</strong>Fill in the specs above and choose Check fair price.</div>',
            unsafe_allow_html=True,
        )

# ---------------- Tab 3: History ----------------
with tab3:
    with st.container(border=True):
        panel_open("Look up a price history", "Prices are saved each time you compare a model, so history builds up over time.")
        h_search, h_btn = st.columns([5, 1], vertical_alignment="bottom")
        with h_search:
            hq = st.text_input("Laptop model", "HP Pavilion 15", key="hist_q", label_visibility="collapsed")
        with h_btn:
            show_hist_btn = st.button("Show history", width="stretch", type="primary", key="hist", disabled=db is None)

    if db is None:
        st.markdown(
            '<div class="empty"><strong>History isn\'t configured</strong>Add your Firebase credentials to .streamlit/secrets.toml to enable this tab.</div>',
            unsafe_allow_html=True,
        )
    else:
        if show_hist_btn:
            with st.spinner("Loading saved prices..."):
                try:
                    rows = fetch_history(hq)
                    st.session_state["hist_result"] = {"q": hq, "rows": rows, "error": None}
                except Exception as err:
                    st.session_state["hist_result"] = {"q": hq, "rows": [], "error": str(err)}

        hr = st.session_state.get("hist_result")
        if not hr:
            st.markdown(
                '<div class="empty"><strong>No model selected</strong>Enter a model name and choose Show history.</div>',
                unsafe_allow_html=True,
            )
        elif hr["error"]:
            st.error(f"Couldn't load history. {hr['error']}")
        elif not hr["rows"]:
            st.info(f"No saved prices for “{hr['q']}” yet. Run a search in Compare prices on a few different days to start a history.")
        else:
            dfh = pd.DataFrame(hr["rows"])
            dfh["timestamp"] = pd.to_datetime(dfh["timestamp"])

            sources = sorted(dfh["source"].dropna().unique())
            picked = st.multiselect("Sellers", sources, default=sources, placeholder="Choose sellers")
            dfh = dfh[dfh["source"].isin(picked)] if picked else dfh

            daily = dfh.groupby("timestamp")["price"].agg(Lowest="min", Average="mean").reset_index()
            first_low, last_low = daily["Lowest"].iloc[0], daily["Lowest"].iloc[-1]
            change = (last_low - first_low) / first_low * 100 if first_low else 0
            cls = "down" if change < 0 else ("up" if change > 0 else "")
            note = f"{'▼' if change < 0 else '▲' if change > 0 else ''} {abs(change):.1f}% since first scan" if len(daily) > 1 else "Only one scan so far"

            st.markdown(
                '<div class="kpis">'
                + kpi("Lowest ever", inr(dfh["price"].min()), lead=True)
                + kpi("Latest lowest", inr(last_low), note, cls)
                + kpi("Scans", f"{daily.shape[0]}", f"{len(dfh)} listings saved")
                + "</div>",
                unsafe_allow_html=True,
            )

            st.markdown(f"#### Price trend for “{hr['q']}”")
            long = daily.melt("timestamp", var_name="Series", value_name="Price")
            base = alt.Chart(long).encode(
                x=alt.X("timestamp:T", title=None),
                y=alt.Y("Price:Q", title="Price (₹)", scale=alt.Scale(zero=False), axis=alt.Axis(format="~s")),
                color=alt.Color("Series:N", scale=alt.Scale(domain=["Lowest", "Average"], range=[PINE, TAG]),
                                legend=alt.Legend(orient="top", title=None)),
                tooltip=[alt.Tooltip("timestamp:T", title="Scanned", format="%d %b %Y, %H:%M"),
                         "Series:N", alt.Tooltip("Price:Q", format=",.0f", title="Price (₹)")],
            )
            st.altair_chart((base.mark_line(strokeWidth=3) + base.mark_point(size=60, filled=True)).properties(height=320),
                            width="stretch")

            with st.expander("All saved listings"):
                st.dataframe(
                    dfh[["timestamp", "title", "price", "source"]].sort_values("timestamp", ascending=False),
                    hide_index=True,
                    width="stretch",
                    column_config={
                        "timestamp": st.column_config.DatetimeColumn("Saved on", format="D MMM YYYY, h:mm a"),
                        "title": st.column_config.TextColumn("Laptop", width="large"),
                        "price": st.column_config.NumberColumn("Price (₹)", format="₹%d"),
                        "source": st.column_config.TextColumn("Seller"),
                    },
                )

# ---------------- Tab 4: Model info ----------------
with tab4:
    with st.container(border=True):
        panel_open("Model evaluation", "Metrics computed on a held-out 20% test split, in Indian Rupees.")
        if metrics is None:
            st.warning("metrics.json not found. Run `python train.py` inside the `ml` folder to generate it.")
        else:
            m = metrics["metrics"]
            st.markdown(
                '<div class="kpis">'
                + kpi("Algorithm", metrics["best_model"], lead=True)
                + kpi("R²", f"{m['R2']:.3f}")
                + kpi("MAPE", f"{m['MAPE']:.1f}%")
                + kpi("MAE", inr(m["MAE"]))
                + kpi("RMSE", inr(m["RMSE"]))
                + "</div>",
                unsafe_allow_html=True,
            )
            with st.expander("Compare all trained models"):
                st.dataframe(pd.DataFrame(metrics["all_models"]), hide_index=True, width="stretch")
            st.caption(f"Trained on {metrics['rows_trained_on']} laptops.")

    with st.container(border=True):
        panel_open("About this project")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Objective**")
            st.caption("Predict a laptop's fair price from its specifications and compare it against live market prices.")
            st.markdown("**Dataset**")
            st.caption("1,303 real laptop listings (brand, type, CPU, GPU, RAM, storage, screen, weight, OS) with price as the target.")
        with c2:
            st.markdown("**Algorithm**")
            st.caption("Linear Regression, Random Forest and Gradient Boosting compared; the best by R² is used. One-hot encoding for categorical features, trained on log(price).")
            st.markdown("**Technologies**")
            st.caption("Streamlit, scikit-learn, Pandas, NumPy, Joblib, Firebase Firestore (optional), SerpAPI / Google Shopping (optional).")
