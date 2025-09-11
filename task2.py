# ===============================
# AMAZON REVIEWS - FULL ANALYSIS
# ===============================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from collections import Counter
import re

# ===============================
# Load Dataset
# ===============================
file_path = "data/amazon_reviews.csv"
df = pd.read_csv(file_path, parse_dates=["review_date"])

# ===============================
# Data Cleaning
# ===============================
df = df.drop_duplicates()
df["month"] = df["review_date"].dt.to_period("M")
df["review_length"] = df["review_text"].apply(len)

# ===============================
# Sentiment Analysis
# ===============================
def get_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"

df["sentiment"] = df["review_text"].apply(get_sentiment)

# ===============================
# GRID 1 - Basic Distributions
# ===============================
fig, axes = plt.subplots(2, 2, figsize=(9, 6))
fig.suptitle("Amazon Reviews - Grid 1", fontsize=14, x=0.5, ha="center")

sns.countplot(x="rating", data=df, palette="viridis", ax=axes[0,0])
axes[0,0].set_title("Distribution of Ratings")

sns.histplot(df["helpful_votes"], bins=10, kde=True, ax=axes[0,1])
axes[0,1].set_title("Helpful Votes Distribution")

sns.barplot(x="product_category", y="rating", data=df, ci=None, palette="Set2", ax=axes[1,0])
axes[1,0].set_title("Average Rating by Category")
axes[1,0].tick_params(axis="x", rotation=45)

sns.boxplot(x="rating", y="helpful_votes", data=df, palette="coolwarm", ax=axes[1,1])
axes[1,1].set_title("Helpful Votes by Rating")

plt.tight_layout()
plt.savefig('outputs/grid1_basic_distributions.png')
plt.show()

# ===============================
# GRID 2 - Trends & Sentiment
# ===============================
fig, axes = plt.subplots(2, 2, figsize=(9, 6))
fig.suptitle("Amazon Reviews - Grid 2", fontsize=14, x=0.5, ha="center")

monthly_avg_rating = df.groupby("month")["rating"].mean()
monthly_avg_rating.plot(marker="o", ax=axes[0,0])
axes[0,0].set_title("Monthly Avg Rating Trend")
axes[0,0].set_ylabel("Average Rating")

sns.scatterplot(x="review_length", y="helpful_votes", hue="rating", data=df, palette="plasma", ax=axes[0,1])
axes[0,1].set_title("Review Length vs Helpful Votes")

sns.countplot(x="sentiment", data=df, palette="Set1", ax=axes[1,0])
axes[1,0].set_title("Sentiment Distribution")

sns.boxplot(x="sentiment", y="rating", data=df, palette="coolwarm", ax=axes[1,1])
axes[1,1].set_title("Ratings by Sentiment")

plt.tight_layout()
plt.savefig('outputs/grid2_trends_sentiment.png')
plt.show()

# ===============================
# GRID 3 - Deep Dive & Text
# ===============================
fig, axes = plt.subplots(2, 2, figsize=(9, 6))
fig.suptitle("Amazon Reviews - Grid 3", fontsize=14, x=0.5, ha="center")

sns.countplot(x="product_category", hue="sentiment", data=df, palette="viridis", ax=axes[0,0])
axes[0,0].set_title("Sentiment Across Categories")
axes[0,0].tick_params(axis="x", rotation=45)

monthly_sentiment = df.groupby(["month", "sentiment"]).size().unstack().fillna(0)
monthly_sentiment.plot(kind="line", marker="o", ax=axes[0,1])
axes[0,1].set_title("Monthly Sentiment Trend")
axes[0,1].set_ylabel("Review Count")

corr = df[["rating", "helpful_votes", "review_length"]].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=axes[1,0])
axes[1,0].set_title("Correlation Heatmap")

# Top 10 Most Common Words
def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text.split()

all_words = []
for review in df["review_text"]:
    all_words.extend(tokenize(review))

word_counts = Counter(all_words).most_common(10)
words, counts = zip(*word_counts)

sns.barplot(x=list(counts), y=list(words), palette="magma", ax=axes[1,1])
axes[1,1].set_title("Top 10 Most Common Words in Reviews")

plt.tight_layout()
plt.savefig('outputs/grid3_deep_dive.png')
plt.show()
