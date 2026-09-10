"""
streamlit_app.py
-----------------
Interactive dashboard for the EA Player Review & Sentiment Analytics
project. Lets a reviewer explore the same findings as the notebook
(analysis/ea_player_sentiment_analysis.ipynb) without running any code.

Run locally:   streamlit run app/streamlit_app.py
Deploy:        push this repo to GitHub, then deploy free on
                Streamlit Community Cloud (share.streamlit.io) pointing
                at app/streamlit_app.py -- see README.md for the exact steps.
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# --- data loading -----------------------------------------------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(APP_DIR, "..", "data", "player_reviews.csv")

st.set_page_config(page_title="EA Player Sentiment Analytics", layout="wide")

try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon")


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["review_date"])
    sia = SentimentIntensityAnalyzer()
    df["vader_compound"] = df["review_text"].apply(lambda t: sia.polarity_scores(t)["compound"])
    return df


df = load_data()

# --- sidebar filters ----------------------------------------------------
st.sidebar.title("EA Player Sentiment Analytics")
st.sidebar.caption(
    "Simulated EA Product Analyst project — player review analytics "
    "across four EA titles. Built for the EA Hyderabad (Slingshot "
    "Studios) Product Analyst Intern application."
)

games = st.sidebar.multiselect("Game(s)", sorted(df["game"].unique()), default=sorted(df["game"].unique()))
date_min, date_max = df["review_date"].min().date(), df["review_date"].max().date()
date_range = st.sidebar.date_input("Date range", (date_min, date_max), min_value=date_min, max_value=date_max)

filtered = df[df["game"].isin(games)]
if isinstance(date_range, tuple) and len(date_range) == 2:
    start, end = date_range
    filtered = filtered[(filtered["review_date"].dt.date >= start) & (filtered["review_date"].dt.date <= end)]

st.sidebar.metric("Reviews in view", f"{len(filtered):,}")
st.sidebar.metric("Avg rating", f"{filtered['rating'].mean():.2f}" if len(filtered) else "-")
st.sidebar.metric("Avg sentiment (VADER)", f"{filtered['vader_compound'].mean():.2f}" if len(filtered) else "-")

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Full analysis:** [`ea_player_sentiment_analysis.ipynb`](.) covers "
    "hypothesis testing, a TF-IDF + Logistic Regression classifier, "
    "player segmentation, anomaly detection, and a 14-day forecast."
)

# --- header ---------------------------------------------------------
st.title("🎮 EA Player Review & Sentiment Analytics")
st.caption(
    "Data: ~8,000 synthetic player reviews (Jan–Jun 2024) across "
    "EA SPORTS FC 25, Apex Legends, The Sims 4, and Battlefield 2042. "
    "Generated to simulate real player feedback patterns, since live "
    "EA/Steam review data wasn't reachable in this build environment "
    "— see `data/generate_data.py` for exactly how and why."
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total reviews", f"{len(filtered):,}")
col2.metric("Games in view", len(games))
col3.metric("Avg rating", f"{filtered['rating'].mean():.2f} / 5" if len(filtered) else "-")
col4.metric("Negative review rate", f"{(filtered['rating'] <= 2).mean():.1%}" if len(filtered) else "-")

st.markdown("---")

# --- tabs -------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Overview", "🗣️ Sentiment", "🚨 Anomaly Detection", "🧩 Player Segments"]
)

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Review volume by game")
        fig, ax = plt.subplots(figsize=(6, 4))
        filtered["game"].value_counts().plot(kind="bar", ax=ax, color=sns.color_palette("Set2"))
        ax.set_ylabel("Reviews")
        plt.xticks(rotation=20)
        st.pyplot(fig)
    with c2:
        st.subheader("Average rating by game")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(data=filtered, x="game", y="rating", ax=ax, errorbar=("ci", 95))
        plt.xticks(rotation=20)
        st.pyplot(fig)

    st.subheader("Daily review volume")
    fig, ax = plt.subplots(figsize=(11, 3.5))
    daily = filtered.groupby(filtered["review_date"].dt.date).size()
    daily.plot(ax=ax)
    ax.set_ylabel("Reviews / day")
    st.pyplot(fig)

    st.subheader("Platform & region mix")
    c3, c4 = st.columns(2)
    with c3:
        fig, ax = plt.subplots(figsize=(6, 4))
        filtered["platform"].value_counts().plot(kind="bar", ax=ax, color=sns.color_palette("Set3"))
        plt.xticks(rotation=20)
        st.pyplot(fig)
    with c4:
        fig, ax = plt.subplots(figsize=(6, 4))
        filtered["region"].value_counts().plot(kind="bar", ax=ax, color=sns.color_palette("Set1"))
        plt.xticks(rotation=20)
        st.pyplot(fig)

with tab2:
    st.subheader("VADER sentiment vs star rating")
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=filtered, x="rating", y="vader_compound", ax=ax)
    ax.set_xlabel("Player rating (1-5)")
    ax.set_ylabel("VADER compound score")
    st.pyplot(fig)

    st.subheader("Try the sentiment scorer")
    sample_text = st.text_area(
        "Paste a review-style sentence to score it live:",
        "Matchmaking has been broken since the last patch, really frustrating.",
    )
    if sample_text:
        sia = SentimentIntensityAnalyzer()
        scores = sia.polarity_scores(sample_text)
        st.json(scores)
        verdict = (
            "🟢 Positive" if scores["compound"] > 0.05
            else "🔴 Negative" if scores["compound"] < -0.05
            else "🟡 Neutral"
        )
        st.markdown(f"**Verdict:** {verdict}")

with tab3:
    st.subheader("Daily sentiment with anomaly flags")
    st.caption(
        "Rolling 14-day z-score on mean daily sentiment per game. Any day "
        "more than 2.5 standard deviations below trend is flagged — this "
        "is how a live 'sentiment health' alert would catch a bad patch "
        "automatically instead of waiting on a support-ticket backlog."
    )
    pick_game = st.selectbox("Game", sorted(df["game"].unique()), index=list(sorted(df["game"].unique())).index("Battlefield 2042") if "Battlefield 2042" in df["game"].unique() else 0)

    game_df = df[df["game"] == pick_game].copy()
    daily_sent = (
        game_df.groupby(game_df["review_date"].dt.date)["vader_compound"]
        .mean()
        .reset_index()
        .rename(columns={"review_date": "date"})
        .sort_values("date")
        .reset_index(drop=True)
    )
    roll_mean = daily_sent["vader_compound"].rolling(14, min_periods=5).mean()
    roll_std = daily_sent["vader_compound"].rolling(14, min_periods=5).std()
    daily_sent["z_score"] = (daily_sent["vader_compound"] - roll_mean) / roll_std
    daily_sent["is_anomaly"] = daily_sent["z_score"] < -2.5

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(daily_sent["date"], daily_sent["vader_compound"], label="Daily avg sentiment")
    flagged = daily_sent[daily_sent["is_anomaly"]]
    ax.scatter(flagged["date"], flagged["vader_compound"], color="red", zorder=5, label="Flagged anomaly")
    ax.set_title(f"{pick_game} — Daily Sentiment")
    ax.legend()
    plt.xticks(rotation=30)
    st.pyplot(fig)

    if len(flagged):
        st.warning(f"{len(flagged)} anomalous low-sentiment day(s) detected for {pick_game}.")
        st.dataframe(flagged[["date", "vader_compound", "z_score"]].round(3))
    else:
        st.success(f"No sentiment anomalies detected for {pick_game} in this window.")

with tab4:
    st.subheader("Player segmentation (KMeans)")
    st.caption("Segmenting players by playtime, rating, and review length into casual / regular / power-user cohorts.")

    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans

    seg_df = filtered.copy()
    if len(seg_df) < 10:
        st.info("Not enough reviews in the current filter to segment. Widen the filters in the sidebar.")
    else:
        seg_df["review_length"] = seg_df["review_text"].str.len()
        feats = seg_df[["playtime_hours", "rating", "review_length"]]
        scaled = StandardScaler().fit_transform(feats)
        km = KMeans(n_clusters=3, random_state=42, n_init=10)
        seg_df["cluster"] = km.fit_predict(scaled)

        summary = seg_df.groupby("cluster")[["playtime_hours", "rating", "review_length"]].mean().round(2)
        order = summary["playtime_hours"].sort_values().index
        name_map = {order[0]: "Casual", order[1]: "Regular", order[2]: "Power User"}
        summary.index = [name_map[i] for i in summary.index]
        summary["n_players"] = seg_df["cluster"].map(name_map).value_counts()
        st.dataframe(summary)

        seg_df["segment"] = seg_df["cluster"].map(name_map)
        fig, ax = plt.subplots(figsize=(8, 5))
        sample = seg_df.sample(min(1500, len(seg_df)), random_state=1)
        sns.scatterplot(data=sample, x="playtime_hours", y="rating", hue="segment", alpha=0.5, ax=ax)
        ax.set_title("Player Segments: Playtime vs Rating")
        st.pyplot(fig)

st.markdown("---")
st.caption(
    "Built by [your name] as a Product Analyst portfolio project. "
    "Source & full write-up: see the project README. Data is synthetic; "
    "methodology (SQL, VADER + ML classification, hypothesis testing, "
    "segmentation, anomaly detection, forecasting) mirrors real "
    "player-analytics workflows."
)
