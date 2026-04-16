import os

import requests
import streamlit as st

# ── Config ───────────────────────────────────────────────────────────────────
API_URL = os.environ.get("API_URL", "http://localhost:8000")

TIER_LABELS = {
    "budget":       "Budget (below $129k)",
    "mid_market":   "Mid Market ($129k – $210k)",
    "above_market": "Above Market ($210k – $300k)",
    "premium":      "Premium (above $300k)",
}

FEATURE_DISPLAY = {
    "overall_qual":  "Overall Quality (1–10)",
    "gr_liv_area":   "Living Area (sqft)",
    "garage_area":   "Garage Area (sqft)",
    "total_bsmt_sf": "Basement Area (sqft)",
    "year_built":    "Year Built",
    "full_bath":     "Full Bathrooms",
    "mas_vnr_area":  "Masonry Veneer (sqft)",
    "bsmt_qual":     "Basement Quality",
    "exter_qual":    "Exterior Quality",
    "kitchen_qual":  "Kitchen Quality",
    "fireplace_qu":  "Fireplace Quality",
    "neighborhood":  "Neighborhood",
}


def call_predict(query: str) -> dict:
    response = requests.post(
        f"{API_URL}/predict",
        json={"query": query},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


# ── Page ─────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Real Estate Agent",
    page_icon="🏠",
    layout="centered",
)

st.title("AI Real Estate Agent")
st.caption(
    "Describe a property in plain English and get an instant price estimate "
    "powered by a Random Forest model and Gemini."
)

# ── Session state ─────────────────────────────────────────────────────────────
for key, default in [
    ("accumulated_query", ""),
    ("missing_details",   []),
    ("result",            None),
    ("turn",              0),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ── Input area ────────────────────────────────────────────────────────────────
if st.session_state.turn == 0:
    user_input = st.text_area(
        "Describe the property",
        placeholder=(
            "e.g. A 2005-built home in StoneBr, 2400 sqft living area, "
            "excellent exterior and kitchen, good fireplace, 2 full baths, "
            "600 sqft garage, overall quality 8."
        ),
        height=130,
    )
    submit_label = "Get Estimate"

else:
    missing = st.session_state.missing_details
    st.warning(
        f"I need {len(missing)} more detail(s) to complete the estimate. "
        "Please provide the information below."
    )

    for detail in missing:
        label = detail.get("display_name", detail["field"])
        desc  = detail.get("description", "")
        with st.expander(f"{label} — {desc}"):
            if "valid_range" in detail:
                st.write(f"**Valid range:** `{detail['valid_range']}`")
            if "valid_options" in detail:
                options = ", ".join(f"`{o}`" for o in detail["valid_options"])
                st.write(f"**Options:** {options}")

    user_input = st.text_area(
        "Provide the missing details",
        placeholder="e.g. good kitchen quality, 600 sqft garage, typical basement...",
        height=100,
    )
    submit_label = "Continue"


# ── Buttons ───────────────────────────────────────────────────────────────────
col_reset, col_submit = st.columns([1, 4])

with col_reset:
    if st.button("Reset", use_container_width=True):
        for key in ("accumulated_query", "missing_details", "result", "turn"):
            st.session_state[key] = {"accumulated_query": "", "missing_details": [],
                                     "result": None, "turn": 0}[key]
        st.rerun()

with col_submit:
    submitted = st.button(submit_label, type="primary", use_container_width=True)


# ── On submit ─────────────────────────────────────────────────────────────────
if submitted:
    if not user_input.strip():
        st.warning("Please enter a description before submitting.")
        st.stop()

    # Accumulate turns
    if st.session_state.turn == 0:
        st.session_state.accumulated_query = user_input.strip()
    else:
        st.session_state.accumulated_query += f"\n{user_input.strip()}"

    with st.spinner("Analysing..."):
        try:
            data = call_predict(st.session_state.accumulated_query)
        except requests.exceptions.ConnectionError:
            st.error(
                f"Cannot reach the API at `{API_URL}`. "
                "Make sure the server is running."
            )
            st.stop()
        except requests.exceptions.HTTPError as exc:
            st.error(f"API error {exc.response.status_code}: {exc.response.text}")
            st.stop()
        except Exception as exc:
            st.error(f"Unexpected error: {exc}")
            st.stop()

    if data["is_complete"]:
        st.session_state.result = data["response"]
        st.session_state.missing_details = []
        st.rerun()
    else:
        st.session_state.missing_details = data["missing_details"]
        st.session_state.turn += 1
        st.rerun()


# ── Result display ────────────────────────────────────────────────────────────
if st.session_state.result:
    r = st.session_state.result
    st.divider()

    # ── Price headline ────────────────────────────────────────────────────────
    st.subheader("Estimated Price")
    st.metric(
        label="Estimated Sale Price",
        value=f"${r['predicted_price']:,.0f}",
        label_visibility="collapsed",
    )
    tier_label = TIER_LABELS.get(r["price_tier"], r["price_tier"])
    st.caption(f"Price tier: **{tier_label}**")

    # ── Market context ────────────────────────────────────────────────────────
    st.subheader("Market Context")
    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            label=f"Neighbourhood Median ({r['extracted_features'].get('neighborhood', '')})",
            value=f"${r['neighborhood_median']:,.0f}",
            delta=f"{r['pct_vs_neighborhood']:+.1f}%",
        )
    with col2:
        st.metric(
            label="Ames Dataset Median",
            value=f"${r['dataset_median']:,.0f}",
            delta=f"{r['pct_vs_dataset']:+.1f}%",
        )

    # ── Interpretation ────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Analysis")
    st.write(r["interpretation"])

    # ── Feature breakdown ─────────────────────────────────────────────────────
    with st.expander("Extracted Features"):
        rows = {
            FEATURE_DISPLAY.get(k, k): str(v)
            for k, v in r["extracted_features"].items()
        }
        st.table(rows)
