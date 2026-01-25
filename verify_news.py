
import sys
import os
sys.path.append(os.getcwd())

from src.core.news import NewsAnalyzer

def verify_news():
    print("Initializing NewsAnalyzer...")
    analyzer = NewsAnalyzer()
    
    print("Fetching latest news...")
    news = analyzer.get_latest_news(limit=5)
    
    print(f"Found {len(news)} items.")
    for i, item in enumerate(news):
        print(f"\nItem {i+1}:")
        print(f"Title: {item['title']}")
        print(f"Sentiment: {item['sentiment']['label']} (Polarity: {item['sentiment']['polarity']:.2f})")
        print(f"Description: {item['description'][:100]}...")

if __name__ == "__main__":
    verify_news()
