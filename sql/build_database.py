"""
build_database.py
Loads the generated CSV into a SQLite database (ea_analytics.db) with a
proper schema, so the project has a real SQL layer instead of just a
flat file -- matching the "extract and transform data with SQL" part
of the role.
"""
import sqlite3
import pandas as pd

df = pd.read_csv("data/ea_player_reviews.csv")

conn = sqlite3.connect("sql/ea_analytics.db")
df.to_sql("reviews", conn, if_exists="replace", index=False)

conn.execute("CREATE INDEX IF NOT EXISTS idx_game ON reviews(game)")
conn.execute("CREATE INDEX IF NOT EXISTS idx_date ON reviews(review_date)")
conn.commit()
conn.close()
print(f"Loaded {len(df):,} rows into sql/ea_analytics.db (table: reviews)")
