import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# Page config
# ============================================================
st.set_page_config(
    page_title="Spotify Hit Predictor",
    page_icon="🎵",
    layout="wide"
)

# ============================================================
# Load model artifacts
# ============================================================
@st.cache_resource
def load_artifacts():
    model = joblib.load('model.pkl')
    feature_columns = joblib.load('feature_columns.pkl')
    genres = joblib.load('genres.pkl')
    metrics = joblib.load('metrics.pkl')
    return model, feature_columns, genres, metrics

model, feature_columns, genres, metrics = load_artifacts()

# ============================================================
# Title and description
# ============================================================
st.title("🎵 Spotify Hit Predictor")
st.markdown("""
This tool predicts whether a track is likely to be a **hit** on Spotify
(popularity score > 70) based on its audio characteristics and genre.

**For A&R representatives:** use this as a screening tool to rank tracks
in your release slate by hit-likelihood, not as a hard accept/reject filter.

> ⚠️ **Predictive, not causal.** The model identifies audio profiles correlated
> with hit-likely tracks. It does not mean changing a feature will *cause* a song
> to become a hit. Real-world popularity also depends on marketing, artist fame,
> and playlist placement, which this model cannot see.
""")

st.divider()

# ============================================================
# Sidebar: input controls
# ============================================================
st.sidebar.header("🎚️ Track Audio Features")
st.sidebar.markdown("Adjust the sliders to describe the track:")

danceability = st.sidebar.slider("Danceability", 0.0, 1.0, 0.65, 0.01,
    help="How suitable a track is for dancing (0 = least, 1 = most)")
energy = st.sidebar.slider("Energy", 0.0, 1.0, 0.70, 0.01,
    help="Perceptual measure of intensity (0 = calm, 1 = intense)")
loudness = st.sidebar.slider("Loudness (dB)", -30.0, 5.0, -7.0, 0.1,
    help="Overall loudness in decibels")
speechiness = st.sidebar.slider("Speechiness", 0.0, 1.0, 0.07, 0.01,
    help="Presence of spoken words (high = rap/podcasts, low = music)")
acousticness = st.sidebar.slider("Acousticness", 0.0, 1.0, 0.20, 0.01,
    help="Confidence the track is acoustic")
instrumentalness = st.sidebar.slider("Instrumentalness", 0.0, 1.0, 0.05, 0.01,
    help="Likelihood the track contains no vocals")
liveness = st.sidebar.slider("Liveness", 0.0, 1.0, 0.18, 0.01,
    help="Presence of a live audience")
valence = st.sidebar.slider("Valence (positivity)", 0.0, 1.0, 0.55, 0.01,
    help="Musical positivity (0 = sad/angry, 1 = happy/cheerful)")
tempo = st.sidebar.slider("Tempo (BPM)", 50.0, 220.0, 120.0, 1.0,
    help="Estimated tempo in beats per minute")
duration_min = st.sidebar.slider("Duration (minutes)", 0.5, 8.0, 3.5, 0.1,
    help="Track length in minutes")

st.sidebar.divider()
st.sidebar.subheader("🎼 Musical Properties")
key = st.sidebar.selectbox("Key", list(range(12)), index=5,
    format_func=lambda x: ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'][x])
mode = st.sidebar.radio("Mode", [1, 0],
    format_func=lambda x: 'Major' if x == 1 else 'Minor', horizontal=True)

st.sidebar.divider()
genre = st.sidebar.selectbox("Genre", genres, index=genres.index('pop') if 'pop' in genres else 0)

# ============================================================
# Build input row matching the model's expected feature columns
# ============================================================
input_dict = {
    'danceability': danceability,
    'energy': energy,
    'key': key,
    'loudness': loudness,
    'mode': mode,
    'speechiness': speechiness,
    'acousticness': acousticness,
    'instrumentalness': instrumentalness,
    'liveness': liveness,
    'valence': valence,
    'tempo': tempo,
    'duration_ms': duration_min * 60000,
}

# add genre dummies (drop_first=True was used, so the first genre alphabetically is the reference)
reference_genre = genres[0]
for g in genres:
    if g != reference_genre:
        input_dict[f'genre_{g}'] = 1 if genre == g else 0

# build dataframe in the exact order the model expects
input_df = pd.DataFrame([input_dict])[feature_columns]

# ============================================================
# Predict
# ============================================================
hit_prob = model.predict_proba(input_df)[0, 1]
prediction = "HIT" if hit_prob > 0.5 else "NOT A HIT"

# ============================================================
# Main panel: prediction display
# ============================================================
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="Prediction",
        value=prediction,
        delta=f"{hit_prob*100:.1f}% hit probability"
    )

with col2:
    # confidence band based on cross-validation std
    auc_mean = metrics['rf_auc_mean']
    auc_std = metrics['rf_auc_std']
    st.metric(
        label="Model Confidence",
        value=f"AUC {auc_mean:.3f}",
        delta=f"± {auc_std:.3f} (CV)"
    )

with col3:
    base_rate = metrics['class_balance']
    lift = hit_prob / base_rate if base_rate > 0 else 0
    st.metric(
        label="Lift over Base Rate",
        value=f"{lift:.2f}×",
        delta=f"Base rate: {base_rate*100:.1f}%"
    )

st.divider()

# ============================================================
# Visualization 1: Hit probability gauge / bar
# ============================================================
st.subheader("📊 Hit Probability")

fig1, ax1 = plt.subplots(figsize=(10, 1.5))
ax1.barh([''], [hit_prob], color='#1DB954', height=0.5)
ax1.barh([''], [1 - hit_prob], left=[hit_prob], color='#E0E0E0', height=0.5)
ax1.axvline(x=0.5, color='red', linestyle='--', linewidth=1.5, label='Decision threshold')
ax1.axvline(x=base_rate, color='gray', linestyle=':', linewidth=1.5, label=f'Base rate ({base_rate*100:.0f}%)')
ax1.set_xlim(0, 1)
ax1.set_xlabel('Predicted Hit Probability')
ax1.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
ax1.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
ax1.legend(loc='lower right', fontsize=8)
ax1.set_yticks([])
for spine in ['top', 'right', 'left']:
    ax1.spines[spine].set_visible(False)
st.pyplot(fig1)

# ============================================================
# Visualization 2: Feature importance
# ============================================================
st.subheader("🔍 What Drives the Prediction?")
st.markdown(
    "These are the features the Random Forest relies on most when ranking tracks. "
    "**Predictive importance only — does not imply causation.**"
)

importances = pd.Series(model.feature_importances_, index=feature_columns).sort_values(ascending=True).tail(10)

fig2, ax2 = plt.subplots(figsize=(8, 5))
ax2.barh(importances.index, importances.values, color='#1DB954')
ax2.set_xlabel('Feature Importance (Gini)')
ax2.set_title('Top 10 Most Important Features')
plt.tight_layout()
st.pyplot(fig2)

# ============================================================
# Uncertainty + caveats
# ============================================================
st.divider()
st.subheader("⚠️ How to Interpret This")

st.markdown(f"""
**Cross-validation performance:** ROC-AUC = {auc_mean:.3f} ± {auc_std:.3f} (5-fold stratified CV).
This means the model ranks hits above non-hits meaningfully better than chance,
but it is **not** a perfect classifier. Use predictions as one input among many.

**Limitations:**
- The model only sees audio features and genre. It cannot account for marketing
  spend, artist reputation, label support, playlist placement, or release timing —
  factors that drive much of real-world popularity.
- "Hit" is defined as Spotify popularity > 70 in this dataset (a snapshot from
  January 2020). Popularity is dynamic and the threshold is an analytical choice.
- Class imbalance (~12% hits) means the model has higher recall on negatives
  than positives. Treat high-probability predictions with more weight than
  low-probability ones.

**Recommended use:** Rank a release slate by hit probability, concentrate
promotion budget on the top quartile, and keep human A&R judgment in the loop
for final allocation decisions.
""")

st.caption("Built for ECON 3916 Final Project • Data: Spotify Web API via TidyTuesday (Jan 2020 snapshot)")