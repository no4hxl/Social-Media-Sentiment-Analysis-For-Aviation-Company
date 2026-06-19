# Airline Sentiment Analysis for Brand Reputation: Comprehensive Project Documentation

## 📌 Project Overview
In the modern aviation industry, social media has evolved into the primary arena for real-time customer feedback. For an airline like Virgin America, public perception directly impacts customer retention, repeat bookings, and long-term brand equity. Traditional feedback loops (such as post-flight email surveys) operate with severe limitations—they are slow, suffer from low response rates, and cannot scale efficiently. 

This project establishes an end-to-end Big Data Analytics pipeline to systematically process, clean, score, and analyze **14,485 tweets**. By pairing algorithmic natural language processing (VADER) with enterprise business intelligence tools (Power BI), we transform high-velocity social media noise into a structured, evidence-based roadmap for executive decision-making.

---

## 🛑 Problem Statement
Customer feedback on modern social media platforms arrives at a scale, velocity, and density that completely breaks manual monitoring workflows. For Virgin America, failing to track this conversational stream creates two critical business vulnerabilities:

1. **Reputational Blind Spots:** Without an automated classification mechanism, the organization cannot track macro-level sentiment shifts or evaluate how operational adjustments influence public perception in real time.
2. **Escalation Risks:** Unaddressed negative customer experiences can achieve virality rapidly. Lacking an algorithmic alerting threshold means acute service failure crises go completely undetected until long after structural brand damage has solidified.

This project directly answers the operational question: **How can Big Data Analytics be deployed to build a resilient pipeline that processes, quantifies, and mitigates public customer friction at scale?**

---

## 🛠️ Data Engineering & Analytics Pipeline

The architecture is built as a modular, 4-phase data analytics pipeline:

```text
[Local SQLite Storage] ➔ [Pandas Subsetting] ➔ [Regex Cleaning Engine] ➔ [VADER Lexicon Scoring] ➔ [Scikit-Learn Evaluation] ➔ [Power BI Output]

```

### Phase 1: Data Sourcing and Structural Reduction

The dataset was sourced manually from **Kaggle** (originally tracking Twitter sentiment for major US airlines). The acquired unstructured text data was structured and stored locally inside a relational SQLite database file (`database.sqlite`).

Using `sqlite3` and `pandas`, the data is loaded directly from local database storage into the Python development environment. To optimize processing efficiency and eliminate computational overhead, immediate schema pruning is performed to drop unnecessary columns, keeping only the 7 core features required for execution:

```python
import sqlite3
import pandas as pd

# Establish connection to local relational storage layer
conn = sqlite3.connect('database.sqlite')
df = pd.read_sql_query("SELECT * FROM Tweets", conn)
conn.close()

# Schema optimization: isolate features relevant to sentiment modeling
columns_to_keep = [
    'text', 'airline_sentiment', 'airline_sentiment_confidence',
    'negativereason', 'airline', 'retweet_count', 'tweet_created'
]
df = df[columns_to_keep].copy()

```

### Phase 2: Algorithmic Text Preprocessing

Raw social media text contains high-entropy elements (noise) that distort lexical sentiment engines. We deploy a vectorized regular expression engine via Python’s `re` module to purge hyperlinks, mentions, hashtags, and numbers, before formatting everything to uniform lowercase:

```python
import re

def clean_tweet(text):
    text = re.sub(r'http\S+', '', text)        # Strip hyperlinks
    text = re.sub(r'@\w+', '', text)           # Strip account mentions (@handle)
    text = re.sub(r'#\w+', '', text)           # Strip hashtags
    text = re.sub(r'[^A-Za-z\s]', '', text)    # Strip numericals and special punctuation
    return text.lower().strip()                # Standardize casing and white spacing

df['clean_text'] = df['text'].apply(clean_tweet)

```

### Phase 3: Lexicon-Based Sentiment Modeling

We pass the cleaned strings into **VADER (Valence Aware Dictionary and sEntiment Reasoner)**. VADER maps individual tokens to semantic intensity values and calculates an aggregated `compound` score bounded between $-1.0$ (highly negative) and $+1.0$ (highly positive). We apply standardized mathematical thresholds to bin these scores into discrete business targets:

```python
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

def classify_sentiment(text):
    score = analyzer.polarity_scores(text)['compound']
    if score >= 0.05:
        return 'positive'
    elif score <= -0.05:
        return 'negative'
    else:
        return 'neutral'

df['vader_sentiment'] = df['clean_text'].apply(classify_sentiment)

```

### Phase 4: Statistical Validation

To establish data integrity for leadership review, we evaluate the rule-based VADER model against the human-annotated ground-truth labels manually sourced from Kaggle using statistical metrics via `scikit-learn`.

---

## 📊 Core Empirical Findings

### 1. The Macro Sentiment Landscape

Our analysis reveals a severe, systemic brand health crisis for Virgin America. Unhappy passengers dominate the platform, with negative interactions thoroughly outnumbering positive expressions.

| Metric Class | Raw Volume | Total Percentage |
| --- | --- | --- |
| **Negative Sentiment** | 9,082 tweets | 62.70% |
| **Neutral Sentiment** | 3,069 tweets | 21.19% |
| **Positive Sentiment** | 2,334 tweets | 16.11% |

> ⚠️ **Data Insight:** A negative footprint exceeding **62%** indicates that customer friction is structural, consistent, and pervasive across the operational ecosystem, rather than driven by isolated outliers. Conversely, satisfied passengers exhibit low organic tendencies to vocalize their experiences on Twitter.

### 2. Algorithmic Pipeline Performance & Limitations

The evaluation phase shows an overall accuracy of **54.15%**. While modest, a deep-dive look at the classification matrix reveals major operational utility along with distinct lexical limitations:

```text
              precision    recall  f1-score   support

    negative       0.90      0.50      0.64      9082
     neutral       0.38      0.43      0.40      3069
    positive       0.34      0.87      0.48      2334

```

* **The Negative Precision Edge (90%):** VADER achieved a precision score of **0.90** on the negative class. This means when the pipeline flags a tweet as a complaint, it is correct 9 out of 10 times. For brand management teams seeking to prioritize crisis mitigation, this makes the pipeline highly reliable for filtering incoming problem streams.
* **The Sarcasm Bottleneck:** The system heavily over-counts positive tweets (assigning 6,015 positive classifications vs. 2,334 actual ground-truth positives). This occurs because VADER evaluates words individually. It struggles to identify context, irony, or sarcasm—meaning a passive-aggressive customer complaint like *"Great job losing my luggage again"* gets incorrectly binned as positive due to the high positive weight of the word *"Great"*.

### 3. Root-Cause Defect Taxonomy

By dissecting the `negativereason` feature, we can categorize and rank the concrete operational failures that damage Virgin America's market perception:

```text
Rank  Operational Failure Category    Tweet Count   Proportional Contribution
-----------------------------------------------------------------------------
1     Customer Service Issue          2,885         ███████████████ 32%
2     Late Flight                     1,650         █████████ 18%
3     Can't Tell                      1,176         ██████ 13%
4     Cancelled Flight                  829         ████ 9%
5     Lost Luggage                      719         ████ 8%

```

* **The Human Interface Failure:** **Customer Service Issues** represent the largest, most acute defect category with 2,885 recorded instances (~32% of total complaints). Frontline human interaction is currently the primary point of failure for passengers.
* **Operational Disruption:** Combined **Late Flights and Cancelled Flights** account for 2,479 tweets (over 27% of negative volume), establishing core operational flight logistics as the second largest structural driver of public brand degradation.
* **High-Emotion Volatility:** While luggage issues (**Lost and Damaged Luggage**) total 792 instances, semantic text processing reveals these updates are highly descriptive and carry severe emotional distress, which drives high retweet amplification among secondary networks.

### 4. High-Velocity Temporal Crises

Plotting the temporal footprint across the 7-day window shows a steady baseline of 750–838 daily complaints, followed by an aggressive, sharp escalation on **February 22, peaking at 2,261 negative tweets**. Because positive and neutral distributions remained completely flat, this massive volume spike points to a major systemic service failure or an unmitigated viral event. The delay in bringing this volume back down confirms that the organization lacked a real-time listening framework to trigger immediate damage-control protocols.

---

## 📈 Strategic Business Recommendations

Based on these findings, Virgin America should transition from passive tracking to proactive data-driven mitigation:

1. **Deploy an Automated Streaming Alert Framework:** Future iterations of this pipeline should engineer automated infrastructure on top of real-time streaming data interfaces. Setting an automated threshold alert (e.g., standard deviation over baseline volume) ensures that anomalous surges like the February 22 spike immediately page on-call communications managers to intervene within hours, preventing localized flight delays from ballooning into national PR crises.
2. **Restructure Customer Service KPI Architecture:** Address the 2,885 customer service complaints by redesigning frontline support training protocols. Move past simple resolution speed metrics and formally incorporate localized social media sentiment scores and qualitative feedback indices directly into employee performance evaluations and compensation dashboards.
3. **Stand Up a Dedicated, Empowered Social Response Unit:** Move social media management out of marketing and establish an agile, dedicated customer support triage cell on Twitter. Providing immediate public updates and resolution options demonstrates brand transparency, actively reducing customer churn and converting public complaints into highly visible examples of stellar service recovery.
4. **Initiate Data-Driven Root-Cause Inquiries on Flight Logistical Operations:** Given that delayed and canceled flights generate over 27% of overall negative sentiment, the operations team must carry out detailed audits of routine technical delays and invest heavily in proactive messaging workflows. Pushing real-time status updates, alternative bookings, and digital meal/hotel vouchers directly to passengers' mobile apps *before* they arrive at an airport gate significantly defuses public frustration and prevents complaints from hitting social media channels.
5. **Overhaul Luggage Management with Real-Time Customer Tracking:** To mitigate the high emotional damage and reach of baggage complaints, integrate an end-to-end RFID baggage tracking interface directly into the airline's passenger app. Giving travelers transparent visibility over their belongings completely eliminates blind anxiety, while an automated, stress-free compensation tracking module for the remaining delayed bags protects customer relationships during operational failures.

---

## ⚠️ Architectural Limitations & Safeguards

To maintain analytical transparency, note the following constraints within the current setup:

* **Temporal Horizon:** The underlying dataset represents a highly concentrated 7-day snapshot from February 2015. It lacks the capacity to capture broader seasonality, major holiday operational bottlenecks, or multi-year structural improvements. Additionally, because Virgin America was subsequently acquired by Alaska Airlines, these metrics represent a historical baseline of brand reputation at that specific historical point.
* **Sarcasm and Context Gaps:** VADER’s dictionary-reliant, rule-based approach can introduce misclassifications when processing complex expressions like irony or regional slang. Moving forward, the text analytics engine should be upgraded to use a supervised transformer architecture (e.g., fine-tuning a `RoBERTa` or `BERT` model trained natively on large-scale aviation datasets) to better capture contextual syntax.
* **Platform Ingestion Gaps:** This analysis exclusively processes text fields manually sourced from Twitter. For a comprehensive, modern assessment of brand health, the analytical pipeline should expand to include multi-channel connections to Instagram, Facebook, Google Reviews, and aviation review platforms like Skytrax.
