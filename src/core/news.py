
import requests
from textblob import TextBlob
from typing import List, Dict, Any
import time

class NewsAnalyzer:
    """
    Analyzes crypto news for sentiment and importance.
    Uses CoinGecko API (public) for news and TextBlob for sentiment analysis.
    """
    
    BASE_URL = "https://min-api.cryptocompare.com/data/v2/news/?lang=EN"
    
    def __init__(self):
        self.session = requests.Session()
        
    def get_latest_news(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Fetch latest crypto news from CryptoCompare.
        """
        try:
            response = self.session.get(self.BASE_URL, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            articles = data.get("Data", [])
            news_items = []
            
            # Slice to limit
            for item in articles[:limit]:
                title = item.get("title", "")
                body = item.get("body", "")
                source = item.get("source_info", {}).get("name", "CryptoCompare")
                published_on = item.get("published_on", 0)
                
                # Convert timestamp to string
                published_at = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(published_on))
                
                # Perform sentiment analysis on Title + Body
                full_text = f"{title}. {body}"
                sentiment = self._analyze_sentiment(full_text)
                
                news_items.append({
                    "title": title,
                    "description": body,
                    "url": item.get("url", ""),
                    "author": source,
                    "published_at": published_at,
                    "sentiment": sentiment
                })
                
            return news_items
            
        except Exception as e:
            print(f"Error fetching news: {e}")
            return []

    def _analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment of text using TextBlob.
        Returns polarity, subjectivity, and a label.
        """
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        if polarity > 0.1:
            label = "POSITIVE"
            color = "#00ff88" # Green
        elif polarity < -0.1:
            label = "NEGATIVE"
            color = "#ff4444" # Red
        else:
            label = "NEUTRAL"
            color = "#8892b0" # Grey
            
        return {
            "polarity": polarity,
            "subjectivity": subjectivity,
            "label": label,
            "color": color
        }

if __name__ == "__main__":
    # Test
    analyzer = NewsAnalyzer()
    news = analyzer.get_latest_news()
    for n in news:
        print(f"[{n['sentiment']['label']}] {n['title']}: {n['description'][:50]}...")
