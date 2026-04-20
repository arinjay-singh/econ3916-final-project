# Spotify Hit Prediction

Predicting whether a song will be a "hit" (Spotify popularity score > 70) based on its audio characteristics and genre. Final project for ECON 3916 (Statistics & Machine Learning for Economics).

## Problem

Can we forecast whether a track crosses a popularity threshold from its audio features alone? This is a **prediction** problem, not a causal one. We're asking whether audio signatures correlate with hit-likely tracks, not whether changing any specific feature would cause a song to succeed. Popularity is driven by many unobserved factors (marketing budget, artist fame, playlist placement, release timing) that aren't in the feature set.

**Stakeholder:** A&R representatives at record labels deciding which tracks to prioritize for marketing spend, playlist pitching, and promotional investment. A reliable hit-prediction model lets them screen tracks early and allocate limited budgets toward songs with audio profiles most associated with commercial success.

## Dataset

- **Source:** [Spotify Songs dataset](https://github.com/rfordatascience/tidytuesday/blob/master/data/2020/2020-01-21/readme.md) via TidyTuesday, originally from Spotify's Web API via the `spotifyr` R package
- **N:** 32,833 tracks (32,828 after dropping 5 rows with missing metadata)
- **Features:** 12 audio features (danceability, energy, key, loudness, mode, speechiness, acousticness, instrumentalness, liveness, valence, tempo, duration_ms) + genre metadata
- **Target:** Binary `is_hit`, engineered from `track_popularity > 70`. Class balance: ~12% hits, ~88% non-hits (imbalanced)
- **Access date:** April 17, 2026

The data loads directly from the TidyTuesday GitHub mirror in the notebook, so no separate download step is needed.

## Approach

**EDA:**
- Missing data assessed as MCAR (0.015% of rows, confined to text metadata columns). Dropped the 5 affected rows.
- Outliers flagged by Tukey fences but kept with justification (skew reflects real music categories like instrumental tracks and spoken-word, not data errors).
- No single audio feature correlates strongly with popularity (max |r| = 0.15), motivating a model that can capture feature interactions.
- Multicollinearity between `energy` and `loudness` (r = 0.68) noted for logistic regression coefficient interpretation.

**Modeling:**
- **Baseline:** Logistic regression with standardized features and `class_weight='balanced'`
- **Comparison:** Random forest with 200 trees, `min_samples_leaf=5`, `class_weight='balanced'`
- Train/test split: 80/20, stratified, `random_state=42`
- Primary metric: ROC-AUC (threshold-independent, appropriate for imbalanced classification)
- Validation: Stratified 5-fold cross-validation

## Files

- `3916_final_project_starter.ipynb` — main notebook (EDA, modeling, cross-validation)
- `app.py` — Streamlit dashboard *(coming)*
- `report.pdf` — 5-page SCR-structured writeup *(coming)*

## Running the Notebook

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
jupyter notebook 3916_final_project_starter.ipynb
```

Or open directly in Google Colab — the data loads from a public URL, no setup needed.

## Key Caveats

- The model predicts popularity patterns at the time of the snapshot (Jan 2020). Popularity is dynamic and this dataset represents a single point in time.
- Audio features don't capture marketing, artist reputation, or playlist placement, which drive much of real-world popularity. Model performance has a ceiling set by these unobserved confounders.
- "Hit" is defined by a popularity threshold of 70, which is an analytical choice rather than an industry-standard cutoff.

## Author

Arinjay — Northeastern University, CS + Economics (May 2026)
