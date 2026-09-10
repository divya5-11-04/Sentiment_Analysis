# Player Sentiment Analysis

An end-to-end NLP and analytics project that analyzes player reviews using sentiment analysis, SQL, and an interactive Streamlit dashboard.

The project takes unstructured player reviews, processes the text, extracts sentiment signals, stores the results in SQLite, and presents insights through an interactive dashboard.

**Live Dashboard:** [Add Streamlit link here]
**GitHub Repository:** https://github.com/divya5-11-04/Sentiment_Analysis

---

## Project Overview

Online player reviews contain useful signals about how users experience a game. However, manually reading thousands of reviews makes it difficult to identify common complaints, positive feedback, and changes in sentiment over time.

This project builds a complete pipeline to:

* Clean and preprocess review text
* Perform sentiment analysis
* Classify reviews into sentiment categories
* Analyze sentiment by game and time period
* Store processed data in SQLite
* Use SQL queries to answer analytical questions
* Identify unusual changes in sentiment
* Present the results through an interactive Streamlit dashboard

---

## Key Questions

The analysis focuses on questions such as:

* What proportion of reviews are positive, neutral, and negative?
* Which games receive the most positive or negative sentiment?
* How does sentiment change over time?
* Which games show unusually negative sentiment?
* What topics or words are associated with different sentiment categories?
* Are there differences between model-based sentiment scores and classified sentiment?
* Which games may require further investigation based on sentiment trends?

---

## Dataset

The project uses **synthetic player-review data**.

The dataset was generated because direct access to a reliable real-world player-review API was not available for this project. The generator was designed to create realistic review structures, including:

* Review text
* Game title
* Review date
* Rating
* Player/reviewer information
* Sentiment-related language patterns

The dataset is intended for demonstrating the analytical pipeline and should not be interpreted as real player feedback.

---

## Project Pipeline

```text
Synthetic Review Data
        ↓
Data Cleaning
        ↓
Text Preprocessing
        ↓
Sentiment Analysis
        ↓
Feature Extraction
        ↓
SQLite Database
        ↓
SQL Analysis
        ↓
Streamlit Dashboard
```

---

## Sentiment Analysis

The project uses NLP techniques to analyze the sentiment expressed in player reviews.

The sentiment pipeline includes:

1. Text normalization
2. Tokenization and preprocessing
3. Sentiment scoring
4. Sentiment classification
5. Aggregation for analytical reporting

The analysis uses sentiment scores to identify positive, neutral, and negative reviews.

### Model Evaluation

The classifier achieved approximately **99.3% accuracy** on the evaluation data.

However, accuracy alone does not tell the complete story.

The minority class achieved approximately **93% recall**, meaning some examples from the less-represented sentiment class were still missed.

The dataset is also synthetic, so this performance should not be interpreted as expected performance on real-world player reviews. Real reviews contain more varied language, sarcasm, spelling errors, slang, mixed sentiment, and context that can make classification harder.

For that reason, the project reports the evaluation result together with its limitations rather than presenting 99.3% accuracy as a production-level result.

---

## SQL Analytics

Processed sentiment data is stored in a SQLite database.

SQL queries are used to perform analysis such as:

* Sentiment distribution
* Game-level sentiment comparison
* Average ratings
* Review volume
* Sentiment trends
* Negative-review analysis
* Time-based aggregation

Example analytical workflow:

```sql
SELECT
    game,
    sentiment,
    COUNT(*) AS review_count
FROM reviews
GROUP BY game, sentiment
ORDER BY review_count DESC;
```

This allows the NLP output to be combined with structured analytical techniques.

---

## Streamlit Dashboard

The project includes an interactive Streamlit dashboard for exploring the results.

The dashboard provides views for:

### Overview

High-level metrics including:

* Total reviews
* Average rating
* Average sentiment score
* Positive/negative review proportions

### Game Analysis

Compare sentiment across different games and identify games with stronger positive or negative feedback.

### Sentiment Trends

Analyze how average sentiment changes over time.

### Review Exploration

Filter and inspect individual reviews and their associated sentiment information.

### Anomaly Analysis

Identify unusual changes in sentiment that may require further investigation.

---

## Repository Structure

```text
Sentiment_Analysis/
│
├── app/
│   └── streamlit_app.py
│
├── analysis/
│   ├── player_sentiment_analysis.py
│   └── player_sentiment_analysis.ipynb
│
├── data/
│   ├── generate_data.py
│   └── player_reviews.csv
│
├── sql/
│   ├── build_database.py
│   ├── analysis_queries.sql
│   └── player_reviews.db
│
├── requirements.txt
├── README.md
└── .gitignore
```

> File names above should match the names currently present in the repository.

---

## Technologies Used

**Programming**

* Python

**Data Analysis**

* pandas
* NumPy
* Matplotlib
* Seaborn

**Machine Learning / NLP**

* scikit-learn
* NLTK
* SciPy
* statsmodels

**Database**

* SQLite
* SQL

**Dashboard**

* Streamlit

**Development**

* Jupyter Notebook
* Git
* GitHub

---

## Running the Project Locally

Clone the repository:

```bash
git clone https://github.com/divya5-11-04/Sentiment_Analysis.git
cd Sentiment_Analysis
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit dashboard:

```bash
streamlit run app/streamlit_app.py
```

The dashboard should open automatically in your browser.

---

## Reproducing the Dataset

The synthetic dataset can be regenerated using the data-generation script:

```bash
python data/generate_data.py
```

This creates the review data used by the analysis pipeline.

---

## Reproducing the Database

To build the SQLite database:

```bash
python sql/build_database.py
```

The resulting database can then be queried using the SQL scripts in the `sql/` directory.

---

## Limitations

This project has several important limitations:

### Synthetic Data

The reviews are generated rather than collected from real players. Therefore, the dataset does not represent actual player opinions.

### Model Performance

The reported 99.3% accuracy is measured on the project evaluation data. It should not be treated as evidence that the same performance would be achieved on real-world reviews.

### Language Complexity

Real reviews may contain:

* Sarcasm
* Slang
* Typos
* Mixed positive and negative opinions
* Very short reviews
* Context-dependent language

These factors can reduce sentiment-classification performance.

### Generalization

Further validation using a real, independently collected dataset would be required before using the system in a production environment.

---

## What I Learned

This project helped me work through an end-to-end analytics workflow rather than treating sentiment classification as an isolated machine-learning problem.

Key areas covered:

* Preparing unstructured text for analysis
* Building an NLP classification pipeline
* Evaluating models beyond overall accuracy
* Working with imbalanced classes
* Designing analytical SQL queries
* Connecting Python analysis with a database
* Building an interactive analytics dashboard
* Thinking about model limitations and generalization
* Deploying a Python application using Streamlit

---

## Future Improvements

Potential improvements include:

* Evaluate the model on a real-world review dataset
* Experiment with transformer-based sentiment models
* Add topic modeling to identify common complaint categories
* Add review-level search and filtering
* Add automated monitoring for significant sentiment changes
* Improve anomaly detection with statistical thresholds
* Add model explainability for individual predictions
* Connect the dashboard to a continuously updated review source

---

## Project Links

**GitHub:**
https://github.com/divya5-11-04/Sentiment_Analysis

**Live Dashboard:**
Add your deployed Streamlit URL here.

---

## Disclaimer

This project is an educational and portfolio project. The player-review dataset is synthetic and was created for demonstrating the data, NLP, SQL, and dashboarding workflow. The reported model metrics should therefore be interpreted within the context of this dataset and its limitations.
