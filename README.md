# 🎮 EA Player Review & Sentiment Analytics

A player-analytics project built specifically for **EA Hyderabad's Product
Analyst Intern (Slingshot Studios)** role — it mirrors the actual workflow
in the job description: pull player feedback data with SQL, analyze it
with Python and statistics, apply basic data science (classification,
clustering, anomaly detection, forecasting), and communicate findings a
producer could act on.

**[Notebook with full analysis →](analysis/ea_player_sentiment_analysis.ipynb)**
**[Interactive dashboard →](app/streamlit_app.py)** (run locally or deploy in ~2 minutes — see below)

---

## Why this dataset is synthetic (read this first)

EA's internal review pipeline and the Steam API weren't reachable from the
environment this was built in. Rather than fake a "real" data source,
`data/generate_data.py` **generates a structurally realistic dataset**:
~7,900 player reviews across four EA titles (EA SPORTS FC 25, Apex
Legends, The Sims 4, Battlefield 2042), Jan–Jun 2024, with:

- rating ↔ sentiment correlation baked in (not random)
- weekday/weekend seasonality in review volume
- platform and region distributions
- **one deliberate "bad patch" event** injected into Battlefield 2042's
  data — a real dip a hypothesis test and an anomaly detector both need
  to independently rediscover

This means every result below is a genuine output of the analysis code
running against this data, not a cherry-picked example — you can
regenerate the dataset with a different random seed and rerun everything.

---

## Key findings

| Question | Method | Result |
|---|---|---|
| Did the patch actually hurt Battlefield 2042 sentiment? | Welch's t-test | Rating dropped from 3.09 → 1.75 post-patch, **p = 3.4e-86** — not noise |
| Does playtime predict satisfaction? | Pearson correlation + OLS | r ≈ 0.02 — negligible in practice, despite being technically "significant" at n≈7,900 (a real trap worth flagging, not just p-hacking around) |
| Can we auto-detect a bad patch from sentiment alone? | Rolling 14-day z-score anomaly detection | Flags the patch window automatically, no manual date lookup needed |
| Can we classify review sentiment from text? | TF-IDF + Logistic Regression | 99.3% test accuracy, 93% recall on the minority negative class — checked against a 5-fold CV gap to rule out overfitting |
| Are casual and power users affected differently by the patch? | KMeans segmentation (3 clusters) | Both segments dropped similarly — argues for a universal hotfix, not a targeted one |
| What's next week's review volume? | Holt-Winters exponential smoothing | 14-day forecast with weekly seasonality, for community-management staffing |

See the notebook's final "Business Takeaways" section for the full
write-up in plain language.

---

## How this maps to the role

| Job description ask | Where it's covered |
|---|---|
| SQL: joins, aggregations, CTEs, window functions | `sql/analysis_queries.sql` — 6 queries against a real SQLite DB, including `RANK()`, `PERCENT_RANK()`, rolling averages, and share-of-total windows |
| Python for analysis, hypothesis testing, insights | `analysis/ea_player_sentiment_analysis.ipynb` |
| Statistics: hypothesis testing, correlation, regression | Welch's t-test, Pearson correlation, OLS regression (Steps 4–5) |
| Data science: classification, clustering, model evaluation, overfitting | TF-IDF + Logistic Regression classifier with CV and confusion matrix (Step 3); KMeans segmentation (Step 6) |
| Experimentation, forecasting, anomaly detection, segmentation | All four are explicit sections (Steps 4, 6, 7, 8) |
| Visualization & communication | Every step ends in a chart and a plain-language takeaway; the dashboard makes it explorable |
| Documenting methodology, assumptions, limitations | See "Why this dataset is synthetic" above, plus the caveats called out inline (e.g. the classifier's accuracy note, the correlation effect-size note) |

---

## Project structure

```
ea-player-sentiment/
├── data/
│   ├── generate_data.py          # synthetic dataset generator (documented, seeded)
│   └── ea_player_reviews.csv      # generated output (~7,900 reviews)
├── sql/
│   ├── build_database.py          # loads CSV into SQLite
│   ├── analysis_queries.sql       # joins, CTEs, window functions
│   └── ea_analytics.db            # generated SQLite DB
├── analysis/
│   ├── ea_player_sentiment_analysis.py   # source (jupytext percent format)
│   └── ea_player_sentiment_analysis.ipynb # executed notebook with outputs
├── app/
│   └── streamlit_app.py           # interactive dashboard
├── requirements.txt
└── README.md
```

---

## Running it locally

```bash
git clone https://github.com/divya5-11-04/Sentiment_Analysis.git
cd Sentiment_Analysis
pip install -r requirements.txt
python -c "import nltk; nltk.download('vader_lexicon')"

# regenerate data + database (optional — already included)
python data/generate_data.py
python sql/build_database.py

# open the notebook
jupyter notebook analysis/ea_player_sentiment_analysis.ipynb

# or run the dashboard
streamlit run app/streamlit_app.py
```

---

## Deploying the dashboard (free, ~2 minutes)

This app has no secrets or paid dependencies, so it deploys cleanly on
**Streamlit Community Cloud**:

1. Push this folder to a public GitHub repo.
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with
   GitHub, click **"New app"**.
3. Pick your repo, set the main file path to `app/streamlit_app.py`,
   and deploy.
4. Streamlit Cloud installs `requirements.txt` automatically. You'll get
   a public URL like `https://<your-app-name>.streamlit.app`.

Once deployed, add the live Streamlit link at the top of this README and in your
application/resume.

---

## Limitations & honest caveats

- **Data is synthetic.** It's built to have realistic structure (see
  above), but it is not real EA telemetry. Every finding in this project
  is a demonstration of method, not a claim about real EA games.
- **The classifier's 99.3% accuracy is a ceiling, not a floor.** Real
  review text is messier (sarcasm, mixed languages, spam) — production
  accuracy would be lower, which is exactly why the notebook checks the
  train/test gap rather than trusting the headline number.
- **The anomaly detector and forecast use simple, explainable methods**
  (rolling z-score; Holt-Winters) deliberately, since a baseline you can
  explain to a producer in one sentence is more useful on day one than a
  more complex model you can't.
