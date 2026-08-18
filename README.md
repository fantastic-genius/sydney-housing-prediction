# Sydney Housing Price Prediction and Decision Support System

A housing price prediction system for **Bondi**, **Chatswood** and **Liverpool**, covering the full ML
lifecycle: data understanding, feature engineering, model selection, failure analysis, a three-way
comparison against an LLM and human judgement, and a deployed Streamlit application.

**Scope:** only sales with `sale_year >= 2024` are used, giving **927 listings**.

## Results at a glance

| | Held-out test set (140 unseen listings) |
|---|---:|
| Model | Histogram Gradient Boosting on `log(price_sold)` |
| R² (log) | 0.962 |
| Mean absolute error | \$187,593 |
| Mean absolute % error | 11.2% |
| Median absolute % error | 9.0% |
| Within 20% of actual | 84.3% |

## Files

| Path | Contents |
|---|---|
| `Task_8D.ipynb` | Full analysis, Parts 1–6, executed with all outputs visible |
| `hpp_features.py` | Feature engineering used by the app |
| `app/app.py` | Streamlit decision support application |
| `app/requirements.txt` | Dependencies |
| `models/price_model.joblib` | Fitted pipeline (preprocessing + estimator) |
| `models/model_meta.json` | Model metadata and per-suburb location defaults |
| `sydney_housing_data.csv` | Dataset |

## Reproducing the results

```bash
cd "sydney-housing-prediction"

# create virtual environment
python -m venv .venv

# activate virtual environment
source .venv/bin/activate

# install packages
pip install -r app/requirements.txt

# Re-run the whole analysis; writes figures/, results.json and models/
jupyter nbconvert --to notebook --execute --inplace Task_8D.ipynb
```

## Running the application

```bash
streamlit run app/app.py     # opens http://localhost:8501
```

The app requires `models/price_model.joblib`, which the notebook writes. If it is missing, the app reports that clearly rather than crashing.

**Single property** — fill the sidebar form and press *Estimate price*. Land and internal size accept `0` for
"not applicable / unknown"; latitude, longitude and distances are pre-filled from the chosen suburb.

**Batch upload** — upload a CSV with the same columns as `sydney_housing_data.csv`. The app appends a
`predicted_price` column, reports accuracy if `price_sold` is present, and offers the result as a download.

**About this model** — accuracy figures and documented failure modes.

## Design note

The app imports an `hpp_features` module used for preprocessing and loads the pickled `Pipeline`, so user input is transformed identically to the training data. This is deliberate: it makes training/serving skew structurally impossible rather than merely unlikely.

## Known limitations

- Covers **three suburbs only**; predictions elsewhere are meaningless.
- Under-predicts expensive property and over-predicts cheap property (median bias runs from +2.5% below
  \$600k to −14.7% above \$4M) — the tail of the market is thin and tree ensembles cannot extrapolate.
- No structured field for condition, aspect, outlook or floor level, which are among the largest real price drivers. Only partly recoverable from agent copy.
- Development sites, boarding houses and blocks of units are priced as if they were single dwellings, and account for three of the five largest errors.
- Trained on 927 sales — the learning curve is still falling, so accuracy is limited by sample size rather than by the algorithm.

