from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Create the analyzer
analyzer = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    """Return sentiment label for given text"""
    score = analyzer.polarity_scores(text)['compound']
    if score >= 0.05:
        return "positive"
    elif score <= -0.05:
        return "negative"
    else:
        return "neutral"

# Example usage
if __name__ == "__main__":
    sample = "Stock markets are doing great today!"
    print(analyze_sentiment(sample))