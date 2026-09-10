"""
generate_data.py
-----------------
Generates a synthetic-but-realistic dataset of EA player reviews across
four EA titles, simulating what a Product Analyst at EA Hyderabad
(Slingshot Studios) might pull from an internal review/feedback pipeline.

Why synthetic data: EA's internal review data and the Steam API aren't
reachable from this environment. The generator below encodes realistic
structure on purpose -- rating <-> sentiment correlation, a real patch
date that causes a negative sentiment spike (so the anomaly detector has
something real to find), weekday/weekend seasonality in review volume,
and per-game/platform/region distributions -- so every downstream
analysis (SQL, stats, ML, anomaly detection, forecasting) is working on
signal, not noise.

Run: python generate_data.py
Output: data/ea_player_reviews.csv
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

rng = np.random.default_rng(42)

GAMES = [
    {"name": "EA SPORTS FC 25", "genre": "Sports", "base_rating": 3.6},
    {"name": "Apex Legends", "genre": "Battle Royale", "base_rating": 3.9},
    {"name": "The Sims 4", "genre": "Simulation", "base_rating": 4.1},
    {"name": "Battlefield 2042", "genre": "Shooter", "base_rating": 3.1},
]

PLATFORMS = ["PC", "PS5", "Xbox Series X", "PS4", "Xbox One"]
PLATFORM_WEIGHTS = [0.35, 0.28, 0.17, 0.12, 0.08]

REGIONS = ["North America", "EU", "APAC", "LATAM", "MEA"]
REGION_WEIGHTS = [0.32, 0.30, 0.20, 0.12, 0.06]

START_DATE = datetime(2024, 1, 1)
END_DATE = datetime(2024, 6, 30)
N_DAYS = (END_DATE - START_DATE).days

# A patch that ships mid-window and tanks sentiment for ~10 days
# (this is the anomaly the anomaly-detection notebook section will surface)
PATCH_DATE = START_DATE + timedelta(days=95)   # ~ Apr 5, 2024
PATCH_GAME = "Battlefield 2042"
PATCH_WINDOW = 10

# Building blocks for a combinatorial text generator, rather than a small
# fixed snippet pool -- a handful of static templates repeated thousands of
# times would make the review text trivially separable by any bag-of-words
# model (train/test accuracy ~1.00, which looks fake to anyone reviewing the
# notebook). Composing clauses from separate pools, sharing some vocabulary
# across positive/negative/neutral, and adding light noise gets closer to
# how real review text behaves -- a classifier that has to work for its
# accuracy number.

ASPECTS = ["matchmaking", "servers", "the new patch", "the progression system",
           "balance", "graphics", "the battle pass", "hit registration",
           "squad play", "the store prices", "load times", "the UI"]

POS_OPENERS = [
    "Really enjoying {aspect} right now.",
    "{aspect} feels great this week.",
    "Honestly {aspect} is in a good spot.",
    "Been having a lot of fun, {aspect} is solid.",
    "Big fan of how {aspect} turned out.",
]
POS_CLOSERS = [
    "Keeps me coming back every day.",
    "Would recommend picking this up.",
    "Devs are clearly listening to feedback.",
    "Best it's felt in a while.",
    "Genuinely having a blast with friends.",
    "Worth the price of entry.",
]
NEG_OPENERS = [
    "{aspect} is a mess right now.",
    "Really frustrated with {aspect} lately.",
    "{aspect} has been broken since the last update.",
    "Not happy with {aspect} at all.",
    "{aspect} ruined an otherwise good session.",
]
NEG_CLOSERS = [
    "Please fix this soon.",
    "Losing patience with this game.",
    "Considering a refund at this point.",
    "Unplayable some nights.",
    "Lost progress because of it, not okay.",
    "Feels like it's getting worse, not better.",
]
NEU_OPENERS = [
    "{aspect} is fine, nothing special.",
    "{aspect} is okay I guess.",
    "Mixed feelings about {aspect}.",
    "{aspect} works as expected.",
]
NEU_CLOSERS = [
    "Some good moments, some frustrating ones.",
    "Wouldn't say it's great, wouldn't say it's bad.",
    "Gets a bit repetitive after a while.",
    "Does the job for a casual session.",
    "Not sure if I'd recommend it at full price.",
]

TYPO_SWAPS = [("realy", "really"), ("sinse", "since"), ("definately", "definitely"),
              ("progresion", "progression"), ("frustraded", "frustrated")]


ALL_OPENERS = {"pos": POS_OPENERS, "neu": NEU_OPENERS, "neg": NEG_OPENERS}
ALL_CLOSERS = {"pos": POS_CLOSERS, "neu": NEU_CLOSERS, "neg": NEG_CLOSERS}


def compose_review(rating, rng, genre, game):
    aspect = rng.choice(ASPECTS)
    bucket = "pos" if rating >= 4 else ("neu" if rating == 3 else "neg")

    # ~14% of reviews mix in an opener/closer from a neighboring sentiment
    # bucket (e.g. a 2-star review that still has one grudging positive
    # line) -- this is what real ambivalent reviews look like, and it's
    # what keeps a text classifier from hitting an unrealistic 100%.
    neighbor_map = {"pos": ["neu"], "neu": ["pos", "neg"], "neg": ["neu"]}
    opener_bucket = rng.choice(neighbor_map[bucket]) if rng.random() < 0.14 else bucket
    closer_bucket = rng.choice(neighbor_map[bucket]) if rng.random() < 0.14 else bucket

    opener = rng.choice(ALL_OPENERS[opener_bucket]).format(aspect=aspect)
    closer = rng.choice(ALL_CLOSERS[closer_bucket])

    text = opener + " " + closer

    # ~30% chance of tacking on an unrelated second aspect, sometimes
    # a mildly conflicting one -- real reviews are rarely one-note, and
    # this is what keeps the classes from being perfectly separable.
    if rng.random() < 0.30:
        mix_rating = rating if rng.random() < 0.7 else rng.integers(1, 6)
        aspect2 = rng.choice(ASPECTS)
        if mix_rating >= 4:
            extra = rng.choice(POS_OPENERS).format(aspect=aspect2)
        elif mix_rating == 3:
            extra = rng.choice(NEU_OPENERS).format(aspect=aspect2)
        else:
            extra = rng.choice(NEG_OPENERS).format(aspect=aspect2)
        text = text + " " + extra

    # light noise: an occasional typo, matching how real reviews are typed
    if rng.random() < 0.15:
        for wrong, right in TYPO_SWAPS:
            if right in text:
                text = text.replace(right, wrong, 1)
                break

    return text


records = []
review_id = 100000

for day_offset in range(N_DAYS):
    date = START_DATE + timedelta(days=day_offset)
    is_weekend = date.weekday() >= 5
    base_volume = 55 if is_weekend else 38  # weekend seasonality

    for game in GAMES:
        game_name = game["name"]
        genre = game["genre"]
        base_rating = game["base_rating"]

        vol_noise = rng.poisson(base_volume / len(GAMES))
        n_reviews_today = max(0, vol_noise)

        # inject the patch-driven anomaly for one game
        sentiment_shock = 0.0
        volume_shock = 1.0
        if game_name == PATCH_GAME and 0 <= (date - PATCH_DATE).days < PATCH_WINDOW:
            sentiment_shock = -1.3
            volume_shock = 1.8  # angry players post more
            n_reviews_today = int(n_reviews_today * volume_shock)

        for _ in range(n_reviews_today):
            review_id += 1
            platform = rng.choice(PLATFORMS, p=PLATFORM_WEIGHTS)
            region = rng.choice(REGIONS, p=REGION_WEIGHTS)
            playtime_hours = round(float(rng.gamma(shape=2.2, scale=25)), 1)

            noise = rng.normal(0, 0.55)
            raw_score = base_rating + noise + sentiment_shock
            rating = int(np.clip(round(raw_score), 1, 5))

            text = compose_review(rating, rng, genre, game_name)

            records.append({
                "review_id": review_id,
                "game": game_name,
                "genre": genre,
                "platform": platform,
                "region": region,
                "rating": rating,
                "review_text": text,
                "playtime_hours": playtime_hours,
                "review_date": date.strftime("%Y-%m-%d"),
                "is_post_patch_window": bool(
                    game_name == PATCH_GAME and 0 <= (date - PATCH_DATE).days < PATCH_WINDOW
                ),
            })

df = pd.DataFrame(records)
df["user_id"] = ["u_" + str(rng.integers(10_000, 99_999)) for _ in range(len(df))]
df = df.sort_values(["review_date", "game"]).reset_index(drop=True)

out_path = "data/ea_player_reviews.csv"
df.to_csv(out_path, index=False)
print(f"Generated {len(df):,} reviews -> {out_path}")
print(df["game"].value_counts())
print("\nPatch anomaly window:", PATCH_DATE.date(), "to", (PATCH_DATE + timedelta(days=PATCH_WINDOW)).date(), "for", PATCH_GAME)
