import feedparser
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download VADER lexicon (only needed the first time)
nltk.download('vader_lexicon')


def fetch_crypto_news(keyword):
    rss_feeds = [
        "https://cointelegraph.com/rss",
        "https://www.coindesk.com/feed"
    ]

    articles = []
    # Parse each RSS feed
    for feed_url in rss_feeds:
        feed = feedparser.parse(feed_url)
        for entry in feed.entries:
            # Check keyword presence in title or summary (case-insensitive)
            if keyword.lower() in entry.title.lower() or keyword.lower() in entry.get("summary", "").lower():
                articles.append({
                    "title": entry.title,
                    "link": entry.link,
                    "published": entry.get("published", "No Date"),
                    "summary": entry.get("summary", "")
                })

    # Initialize sentiment analyzer
    analyzer = SentimentIntensityAnalyzer()

    # Analyze sentiment for each article
    for article in articles:
        # Combine title and summary for sentiment evaluation
        text = article["title"] + " " + article["summary"]
        sentiment_score = analyzer.polarity_scores(text)
        compound = sentiment_score["compound"]
        # Define thresholds for sentiment categorization
        if compound >= 0.05:
            sentiment = "Bullish"
        elif compound <= -0.05:
            sentiment = "Bearish"
        else:
            sentiment = "Neutral"
        article["sentiment"] = sentiment

    # Output the top 10 articles
    if articles:
        print(f"\nLatest news for '{keyword}':\n" + "-" * 50)
        for article in articles[:10]:
            print(f"Title: {article['title']}")
            print(f"Published: {article['published']}")
            print(f"URL: {article['link']}")
            print(f"Sentiment: {article['sentiment']}\n")
    else:
        print(f"No news found for '{keyword}'. Try a different keyword.")


if __name__ == "__main__":
    crypto_keyword = input("Enter a crypto symbol or keyword (e.g., BTC, Ethereum): ")
    fetch_crypto_news(crypto_keyword)
