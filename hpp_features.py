"""Feature engineering for the Sydney housing price model (SIG720 Task 8D).

This module is imported by both ``Task_8D.ipynb`` and ``app/app.py``. Keeping the
transformations here (rather than as lambdas inside the notebook) means the fitted
scikit-learn pipeline can be pickled with ``joblib`` and re-loaded by the web app.

Every transformation is strictly row-wise: no statistic is computed across rows, so
applying this before a train/test split cannot leak information about the target.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

# --- Constants -------------------------------------------------------------------

#: Reference year for the linear time trend, so the coefficient is interpretable.
BASE_YEAR = 2024

#: Currency amounts in agent copy are strata levies, council rates and weekly rents.
#: They are masked before vectorising so no numeric string can act as a price proxy.
CURRENCY_RE = re.compile(r"\$\s?[\d,]+(?:\.\d+)?\s?(?:m|million|k|pw|pa)?", re.IGNORECASE)
NUMBER_RE = re.compile(r"\b\d[\d,\.]*\b")
WHITESPACE_RE = re.compile(r"\s+")

#: Marketing phrases grouped into themes that plausibly move price in opposite
#: directions. Chosen from reading a sample of descriptions, not from the target.
KEYWORD_THEMES: dict[str, tuple[str, ...]] = {
    "kw_prestige": ("luxur", "prestige", "designer", "architect", "bespoke", "premium", "exclusive"),
    "kw_renovated": ("renovated", "refurbished", "immaculate", "brand new", "as new", "recently updated"),
    "kw_potential": (
        "potential",
        "renovator",
        "opportunity to",
        "scope to",
        "original condition",
        "deceased estate",
        "handyman",
        "needs work",
    ),
    "kw_view": ("ocean view", "harbour view", "district view", "water view", "sea view", "panoramic"),
    "kw_outdoor": ("pool", "garden", "courtyard", "terrace", "balcony", "north facing", "north-facing"),
    "kw_devsite": ("da approved", "development", "torrens", "duplex", "subdivi", "r3 zoning", "block of"),
    "kw_investor": ("tenanted", "rental return", "yield", "investor", "investment"),
}

#: Numeric columns fed to the models (after engineering).
NUMERIC_FEATURES: list[str] = [
    "bedrooms",
    "bathrooms",
    "parking",
    "log_land_size",
    "log_building_size",
    "dist_cbd_km",
    "dist_transport_km",
    "latitude",
    "longitude",
    "years_since_base",
    "total_rooms",
    "bath_per_bed",
    "park_per_bed",
    "has_land_size",
    "has_building_size",
    "desc_word_count",
    "desc_bullet_count",
    *KEYWORD_THEMES,
]

#: Low-cardinality categoricals, one-hot encoded.
CATEGORICAL_FEATURES: list[str] = ["suburb", "property_type"]

#: Binary flags. ``sold_by_auction`` is ~19% missing, so it is imputed with an
#: explicit "unknown" level rather than a guess.
BINARY_FEATURES: list[str] = ["is_strata", "sold_by_auction"]

#: Free-text column consumed by the TF-IDF -> SVD branch.
TEXT_FEATURE: str = "description_clean"

#: Raw columns a user must supply to the deployed app.
RAW_INPUT_COLUMNS: list[str] = [
    "suburb",
    "property_type",
    "bedrooms",
    "bathrooms",
    "parking",
    "land_size_m2",
    "building_size_m2",
    "is_strata",
    "latitude",
    "longitude",
    "dist_cbd_km",
    "dist_transport_km",
    "sold_by_auction",
    "sale_year",
    "description",
]


# --- Text cleaning ---------------------------------------------------------------


def clean_description(text: object) -> str:
    """Lower-case agent copy with all currency amounts and bare numbers masked.

    Masking is a deliberate leakage control: 292 of the 927 recent listings quote a
    dollar figure, and although inspection shows these are levies and rents rather
    than sale prices, removing them entirely makes the text branch defensible.
    """
    if not isinstance(text, str):
        return ""
    masked = CURRENCY_RE.sub(" moneytoken ", text)
    masked = NUMBER_RE.sub(" numtoken ", masked)
    return WHITESPACE_RE.sub(" ", masked.lower()).strip()


def _count_bullets(text: object) -> int:
    if not isinstance(text, str):
        return 0
    return text.count("*") + text.count("-") + text.count("•")


# --- Feature engineering ---------------------------------------------------------


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a new frame with engineered columns added. The input is not mutated."""
    out = df.copy()

    description = out["description"] if "description" in out else pd.Series("", index=out.index)
    description = description.fillna("")

    # Size: heavily right-skewed and missing-by-design for strata, so log the value
    # and keep an explicit presence flag rather than pretending a median is the truth.
    out["log_land_size"] = np.log1p(out["land_size_m2"])
    out["log_building_size"] = np.log1p(out["building_size_m2"])
    out["has_land_size"] = out["land_size_m2"].notna().astype(int)
    out["has_building_size"] = out["building_size_m2"].notna().astype(int)

    # Layout: totals and ratios capture "how the rooms are arranged", which raw
    # counts alone do not (a 3-bed/2-bath is not the same product as a 3-bed/1-bath).
    beds = out["bedrooms"]
    beds_safe = beds.where(beds.notna() & (beds > 0), 1.0)
    out["total_rooms"] = beds.fillna(0) + out["bathrooms"].fillna(0)
    out["bath_per_bed"] = out["bathrooms"] / beds_safe
    out["park_per_bed"] = out["parking"] / beds_safe

    # Time: a single linear term. Three years of data cannot support a richer trend.
    out["years_since_base"] = out["sale_year"].astype(float) - BASE_YEAR

    # Text-derived numerics.
    out["description_clean"] = description.map(clean_description)
    out["desc_word_count"] = out["description_clean"].str.split().str.len().fillna(0)
    out["desc_bullet_count"] = description.map(_count_bullets)

    lowered = description.str.lower()
    for theme, phrases in KEYWORD_THEMES.items():
        pattern = "|".join(re.escape(p) for p in phrases)
        out[theme] = lowered.str.contains(pattern, regex=True, na=False).astype(int)

    return out


def all_model_columns() -> list[str]:
    return [*NUMERIC_FEATURES, *CATEGORICAL_FEATURES, *BINARY_FEATURES, TEXT_FEATURE]
