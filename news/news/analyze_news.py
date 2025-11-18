import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# -----------------------------
# Download required NLTK data
# -----------------------------
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# -----------------------------
# Text Preprocessing
# -----------------------------
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www.\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 1]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)

# -----------------------------
# Sentiment Analysis
# -----------------------------
analyzer = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    score = analyzer.polarity_scores(text)['compound']
    if score >= 0.05:
        return "positive"
    elif score <= -0.05:
        return "negative"
    else:
        return "neutral"

# -----------------------------
# Main Program
# -----------------------------
if __name__ == "__main__":
    # Load your news CSV
    file_path = "news.csv"   # make sure this file exists
    df = pd.read_csv(file_path)

    # Adjust column name if needed
    if 'headline' in df.columns:
        df['text'] = df['headline']
    elif 'title' in df.columns:
        df['text'] = df['title']
    elif 'summary' in df.columns:
        df['text'] = df['summary']
    else:
        raise ValueError("CSV must have a column named 'headline', 'title', or 'summary'")

    # Clean text and analyze sentiment
    print("Cleaning and analyzing news... please wait ⏳")
    df['clean_text'] = df['text'].apply(clean_text)
    df['sentiment'] = df['clean_text'].apply(analyze_sentiment)

    # Save results
    output_file = "news_with_sentiment.csv"
    df.to_csv(output_file, index=False)
    print(f"\n✅ Done! Results saved to '{output_file}'")
    print(df[['text', 'sentiment']].head())
