"""Sydney Housing Price Decision Support System — Streamlit prototype (SIG720 Task 8D).

Run with:
    streamlit run app/app.py

The app imports the same ``hpp_features`` module used to train the model and loads the
pickled scikit-learn ``Pipeline``, so a user's input is transformed exactly as the
training data was. Sharing the module is what prevents training/serving skew.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import joblib  # noqa: E402
import hpp_features as hf  # noqa: E402

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "price_model.joblib")
META_PATH = os.path.join(PROJECT_ROOT, "models", "model_meta.json")

#: Fallback band if the metadata file is unavailable.
DEFAULT_ERROR_PCT = 12.0

st.set_page_config(page_title="Sydney Housing Price Estimator", page_icon="🏠", layout="wide")


# --- Loading ---------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading the trained model…")
def load_artifacts() -> tuple[object, dict]:
    """Load the pickled pipeline and its metadata, failing loudly if either is missing."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. Run all cells of Task_8D.ipynb first — "
            "Part 6 writes the model artefact."
        )
    model = joblib.load(MODEL_PATH)
    meta: dict = {}
    if os.path.exists(META_PATH):
        with open(META_PATH, encoding="utf-8") as fh:
            meta = json.load(fh)
    return model, meta


try:
    MODEL, META = load_artifacts()
except Exception as exc:  # surfaced to the user rather than a stack trace in the console
    st.error(f"**The application could not start.**\n\n{exc}")
    st.stop()

SUBURBS: list[str] = META.get("suburbs", ["Bondi", "Chatswood", "Liverpool"])
PROPERTY_TYPES: list[str] = META.get(
    "property_types", ["Apartment", "House", "Other", "Studio", "Townhouse", "Unit"]
)
SUBURB_DEFAULTS: dict = META.get("suburb_defaults", {})
ERROR_PCT: float = float(META.get("holdout_median_ape", DEFAULT_ERROR_PCT))
STRATA_TYPES = {"Apartment", "Unit", "Studio"}


# --- Prediction ------------------------------------------------------------------


def predict_prices(rows: pd.DataFrame) -> np.ndarray:
    """Return predicted prices in AUD for a frame of raw (un-engineered) listings."""
    missing = [c for c in hf.RAW_INPUT_COLUMNS if c not in rows.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(missing)}")
    features = hf.engineer_features(rows)[hf.all_model_columns()]
    return np.exp(MODEL.predict(features))


def money(value: float) -> str:
    """Format as AUD. The dollar sign is escaped because Streamlit renders a pair of
    unescaped `$` in Markdown as LaTeX, which mangles any range like "$A - $B"."""
    return f"\\${value:,.0f}"


# --- Sidebar: single-property form -----------------------------------------------

st.sidebar.header("Property details")

suburb = st.sidebar.selectbox("Suburb", SUBURBS, index=0)
property_type = st.sidebar.selectbox("Property type", PROPERTY_TYPES, index=0)

col_a, col_b = st.sidebar.columns(2)
bedrooms = col_a.number_input("Bedrooms", min_value=0.0, max_value=12.0, value=2.0, step=1.0)
bathrooms = col_b.number_input("Bathrooms", min_value=0.0, max_value=8.0, value=1.0, step=1.0)
parking = col_a.number_input("Car spaces", min_value=0.0, max_value=8.0, value=1.0, step=1.0)
sale_year = col_b.number_input("Year of sale", min_value=2024, max_value=2030, value=2026, step=1)

is_strata = property_type in STRATA_TYPES
st.sidebar.caption(
    f"`{property_type}` is treated as **{'strata' if is_strata else 'freestanding'}** tenure."
)

land_size = st.sidebar.number_input(
    "Land size (m²) — 0 if not applicable", min_value=0.0, max_value=5000.0,
    value=0.0 if is_strata else 550.0, step=10.0,
)
building_size = st.sidebar.number_input(
    "Internal size (m²) — 0 if unknown", min_value=0.0, max_value=1000.0, value=0.0, step=5.0
)

auction_choice = st.sidebar.radio("Method of sale", ["Unknown", "Auction", "Private treaty"],
                                  horizontal=True)
auction_value = {"Unknown": np.nan, "Auction": 1.0, "Private treaty": 0.0}[auction_choice]

with st.sidebar.expander("Location (pre-filled from suburb)"):
    defaults = SUBURB_DEFAULTS.get(suburb, {})
    latitude = st.number_input("Latitude", value=float(defaults.get("latitude", -33.87)), format="%.5f")
    longitude = st.number_input("Longitude", value=float(defaults.get("longitude", 151.21)), format="%.5f")
    dist_cbd = st.number_input("Distance to CBD (km)", min_value=0.0,
                               value=float(defaults.get("dist_cbd_km", 10.0)), step=0.1)
    dist_transport = st.number_input("Distance to transport hub (km)", min_value=0.0,
                                     value=float(defaults.get("dist_transport_km", 1.0)), step=0.1)

description = st.sidebar.text_area(
    "Agent description",
    value="Renovated apartment with an open-plan living area, stone kitchen, built-in wardrobes "
          "and a sunny balcony, moments from shops and transport.",
    height=140,
    help="The model reads this text. Condition, aspect and outlook are only visible here.",
)


def build_single_row() -> pd.DataFrame:
    return pd.DataFrame([{
        "suburb": suburb,
        "property_type": property_type,
        "bedrooms": float(bedrooms),
        "bathrooms": float(bathrooms),
        "parking": float(parking),
        "land_size_m2": float(land_size) if land_size > 0 else np.nan,
        "building_size_m2": float(building_size) if building_size > 0 else np.nan,
        "is_strata": int(is_strata),
        "latitude": float(latitude),
        "longitude": float(longitude),
        "dist_cbd_km": float(dist_cbd),
        "dist_transport_km": float(dist_transport),
        "sold_by_auction": auction_value,
        "sale_year": int(sale_year),
        "description": description,
    }])


# --- Main panel ------------------------------------------------------------------

st.title("🏠 Sydney Housing Price Estimator")
st.caption(
    f"{META.get('model', 'Gradient Boosting')} trained on {META.get('trained_on_rows', 927):,} "
    f"Bondi, Chatswood and Liverpool sales from {META.get('sale_year_min', 2024)} onwards."
)

single_tab, batch_tab, about_tab = st.tabs(["Single property", "Batch upload", "About this model"])

with single_tab:
    st.subheader("Estimate a single property")
    st.write("Enter the details in the sidebar, then press **Estimate price**.")

    if st.button("Estimate price", type="primary"):
        try:
            estimate = float(predict_prices(build_single_row())[0])
        except Exception as exc:
            st.error(f"Could not produce an estimate: {exc}")
        else:
            low, high = estimate * (1 - ERROR_PCT / 100), estimate * (1 + ERROR_PCT / 100)
            left, right = st.columns([1, 1])
            left.metric("Estimated sale price", money(estimate))
            right.metric(f"Likely range (±{ERROR_PCT:.0f}%)", f"{money(low)} – {money(high)}")

            st.info(
                f"**{suburb} {property_type.lower()}**, {bedrooms:.0f} bed / {bathrooms:.0f} bath / "
                f"{parking:.0f} car. The range reflects the model's median absolute error of "
                f"{ERROR_PCT:.0f}% on listings it had never seen."
            )
            st.warning(
                "**This is an indicative estimate, not a formal valuation.** It is least reliable for "
                "properties above $5M, for unusual dwelling types, and for properties in exceptionally "
                "good or poor condition — the dataset has no field recording condition."
            )

with batch_tab:
    st.subheader("Score a CSV of properties")
    st.write(
        "Upload a CSV with the same columns as the training file. "
        f"Required: `{'`, `'.join(hf.RAW_INPUT_COLUMNS)}`."
    )
    uploaded = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded is not None:
        try:
            batch = pd.read_csv(uploaded)
        except Exception as exc:
            st.error(f"Could not read that file: {exc}")
        else:
            st.write(f"Loaded **{len(batch):,}** rows.")
            try:
                batch_out = batch.copy()
                batch_out["predicted_price"] = predict_prices(batch)
            except Exception as exc:
                st.error(f"Could not score the file: {exc}")
            else:
                preview_cols = [c for c in ["suburb", "property_type", "bedrooms", "bathrooms",
                                            "predicted_price"] if c in batch_out.columns]
                st.dataframe(batch_out[preview_cols].head(25), width="stretch")
                if "price_sold" in batch_out.columns:
                    ape = ((batch_out.predicted_price - batch_out.price_sold).abs()
                           / batch_out.price_sold * 100)
                    c1, c2 = st.columns(2)
                    c1.metric("Mean absolute % error", f"{ape.mean():.1f}%")
                    c2.metric("Median absolute % error", f"{ape.median():.1f}%")
                st.download_button(
                    "Download predictions as CSV",
                    batch_out.to_csv(index=False).encode("utf-8"),
                    file_name="predicted_prices.csv",
                    mime="text/csv",
                )

with about_tab:
    st.subheader("How this model works and where it fails")
    st.markdown(
        f"""
**Model.** {META.get('model', 'Histogram Gradient Boosting')} regression on `log(price_sold)`, selected by
5-fold cross-validation against a regularised linear model and a random forest.

**Accuracy on unseen listings.** Mean absolute error {META.get('holdout_mape', '—')}% ·
median absolute error {META.get('holdout_median_ape', '—')}%.

**What it reads.** Suburb, dwelling type, room counts, land and internal area, coordinates and distances,
method of sale, year of sale, and the agent's description. Currency figures in the description are masked
before use so the text cannot leak a price.

**Known limitations.**

- It under-predicts expensive properties and over-predicts cheap ones — an unavoidable consequence of
  fitting a squared-error model when the features cannot fully explain the extremes.
- It has **no field for condition, aspect, outlook or floor level**, which are among the largest real drivers
  of price. These are only partly recoverable from marketing copy.
- It covers **three suburbs only**. Predictions for anywhere else are meaningless.
- It was trained on {META.get('trained_on_rows', 927):,} sales — small enough that accuracy is still limited
  by sample size rather than by the algorithm.

**Use it for** triage, sanity-checking a vendor's expectations and prioritising which properties deserve a
full appraisal. **Do not use it as** a valuation, or as the sole basis for a lending or purchase decision.
"""
    )
