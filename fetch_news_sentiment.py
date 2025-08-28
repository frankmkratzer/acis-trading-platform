#!/usr/bin/env python3
"""
Fetch News Sentiment from Alpha Vantage
Includes sentiment analysis, relevance scoring, and topic classification
"""

import os
import sys
import time
import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from logging_config import setup_logger, log_script_start, log_script_end

# Configuration
load_dotenv()
API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
POSTGRES_URL = os.getenv("POSTGRES_URL")

if not API_KEY or not POSTGRES_URL:
    print("[ERROR] Required environment variables not set")
    sys.exit(1)

engine = create_engine(POSTGRES_URL)
logger = setup_logger("fetch_news_sentiment")

# Constants
AV_URL = "https://www.alphavantage.co/query"
MAX_RETRIES = 3
RATE_LIMIT_DELAY = 12  # 5 calls per minute for news API
MAX_ARTICLES_PER_SYMBOL = 1000  # Alpha Vantage limit
SENTIMENT_LOOKBACK_DAYS = 30  # How far back to fetch news

class SentimentLabel(Enum):
    """Alpha Vantage sentiment labels"""
    BEARISH = "Bearish"
    SOMEWHAT_BEARISH = "Somewhat-Bearish"
    NEUTRAL = "Neutral"
    SOMEWHAT_BULLISH = "Somewhat-Bullish"
    BULLISH = "Bullish"

@dataclass
class NewsArticle:
    """Structure for news article with sentiment"""
    article_id: str
    symbol: str
    title: str
    summary: str
    source: str
    category: str
    published_at: datetime
    url: str
    relevance_score: float
    sentiment_score: float
    sentiment_label: str
    ticker_sentiment_score: float
    ticker_sentiment_label: str
    ticker_relevance_score: float
    topics: List[Dict]
    
def verify_news_sentiment_tables():
    """Verify news sentiment tables exist"""
    with engine.connect() as conn:
        # Check if tables exist
        result = conn.execute(text("""
            SELECT COUNT(*) FROM information_schema.tables 
            WHERE table_name IN ('news_articles', 'news_sentiment_by_symbol', 
                                 'news_topics', 'daily_sentiment_summary')
        """)).scalar()
        
        if result != 4:
            logger.error("News sentiment tables not found. Please run setup_schema.py first.")
            return False
        
        logger.info("News sentiment tables verified")
        return True

def hash_article(title: str, published_at: str) -> str:
    """Generate unique ID for article"""
    content = f"{title}_{published_at}"
    return hashlib.md5(content.encode()).hexdigest()

def parse_sentiment_score(score_str: str) -> float:
    """Convert Alpha Vantage sentiment score to numeric"""
    try:
        return float(score_str) if score_str else 0.0
    except:
        return 0.0

def fetch_news_sentiment(symbols: List[str] = None, topics: List[str] = None) -> List[NewsArticle]:
    """Fetch news sentiment from Alpha Vantage"""
    articles = []
    
    params = {
        "function": "NEWS_SENTIMENT",
        "apikey": API_KEY,
        "limit": 1000  # Max allowed
    }
    
    # Add optional parameters
    if symbols:
        params["tickers"] = ",".join(symbols[:50])  # Max 50 symbols per request
    
    if topics:
        params["topics"] = ",".join(topics)
    
    # Add time range
    time_from = (datetime.now() - timedelta(days=SENTIMENT_LOOKBACK_DAYS)).strftime("%Y%m%dT%H%M")
    params["time_from"] = time_from
    
    try:
        logger.info(f"Fetching news sentiment for {len(symbols) if symbols else 'market'}")
        
        response = requests.get(AV_URL, params=params, timeout=30)
        if response.status_code != 200:
            logger.error(f"API request failed with status {response.status_code}")
            return articles
        
        data = response.json()
        
        # Check for API errors
        if "Error Message" in data:
            logger.error(f"API error: {data['Error Message']}")
            return articles
        
        if "Note" in data:
            logger.warning(f"API note: {data['Note']}")
            time.sleep(60)  # Rate limit hit
            return articles
        
        # Parse feed items
        feed = data.get("feed", [])
        logger.info(f"Retrieved {len(feed)} news articles")
        
        for item in feed:
            # Parse article metadata
            article_id = hash_article(
                item.get("title", ""),
                item.get("time_published", "")
            )
            
            # Parse overall sentiment
            overall_sentiment = item.get("overall_sentiment_score", 0)
            overall_label = item.get("overall_sentiment_label", "Neutral")
            
            # Parse ticker-specific sentiment
            ticker_sentiments = item.get("ticker_sentiment", [])
            
            # Create base article
            base_article = {
                "article_id": article_id,
                "title": item.get("title", ""),
                "summary": item.get("summary", ""),
                "source": item.get("source", ""),
                "category": item.get("category_within_source", ""),
                "published_at": datetime.strptime(
                    item.get("time_published", ""),
                    "%Y%m%dT%H%M%S"
                ).replace(tzinfo=timezone.utc),
                "url": item.get("url", ""),
                "overall_sentiment_score": parse_sentiment_score(overall_sentiment),
                "overall_sentiment_label": overall_label,
                "topics": item.get("topics", [])
            }
            
            # Create article entry for each mentioned ticker
            for ticker_data in ticker_sentiments:
                symbol = ticker_data.get("ticker", "")
                if not symbol:
                    continue
                
                article = NewsArticle(
                    article_id=article_id,
                    symbol=symbol,
                    title=base_article["title"],
                    summary=base_article["summary"],
                    source=base_article["source"],
                    category=base_article["category"],
                    published_at=base_article["published_at"],
                    url=base_article["url"],
                    relevance_score=parse_sentiment_score(
                        ticker_data.get("relevance_score", 0)
                    ),
                    sentiment_score=base_article["overall_sentiment_score"],
                    sentiment_label=base_article["overall_sentiment_label"],
                    ticker_sentiment_score=parse_sentiment_score(
                        ticker_data.get("ticker_sentiment_score", 0)
                    ),
                    ticker_sentiment_label=ticker_data.get(
                        "ticker_sentiment_label", "Neutral"
                    ),
                    ticker_relevance_score=parse_sentiment_score(
                        ticker_data.get("ticker_relevance_score", 0)
                    ),
                    topics=base_article["topics"]
                )
                articles.append(article)
        
        return articles
        
    except Exception as e:
        logger.error(f"Error fetching news sentiment: {e}")
        return articles

def store_news_articles(articles: List[NewsArticle]):
    """Store news articles in database"""
    if not articles:
        return
    
    # Prepare dataframes
    articles_df = pd.DataFrame([{
        "article_id": a.article_id,
        "title": a.title[:500],  # Truncate long titles
        "summary": a.summary[:2000] if a.summary else None,
        "source": a.source,
        "category": a.category,
        "published_at": a.published_at,
        "url": a.url,
        "overall_sentiment_score": a.sentiment_score,
        "overall_sentiment_label": a.sentiment_label,
        "fetched_at": datetime.now(timezone.utc)
    } for a in articles])
    
    # Remove duplicates
    articles_df = articles_df.drop_duplicates(subset=["article_id"])
    
    # Symbol sentiment
    sentiment_df = pd.DataFrame([{
        "article_id": a.article_id,
        "symbol": a.symbol,
        "relevance_score": a.ticker_relevance_score,
        "sentiment_score": a.ticker_sentiment_score,
        "sentiment_label": a.ticker_sentiment_label
    } for a in articles])
    
    # Topics
    topics_data = []
    for article in articles:
        for topic in article.topics:
            topics_data.append({
                "article_id": article.article_id,
                "topic": topic.get("topic", ""),
                "relevance_score": parse_sentiment_score(
                    topic.get("relevance_score", 0)
                )
            })
    topics_df = pd.DataFrame(topics_data) if topics_data else pd.DataFrame()
    
    # Store in database
    with engine.begin() as conn:
        # Upsert articles
        if not articles_df.empty:
            temp_table = f"temp_articles_{int(time.time())}"
            articles_df.to_sql(temp_table, conn, if_exists='replace', index=False)
            
            conn.execute(text(f"""
                INSERT INTO news_articles 
                SELECT * FROM {temp_table}
                ON CONFLICT (article_id) DO NOTHING
            """))
            
            conn.execute(text(f"DROP TABLE {temp_table}"))
        
        # Upsert symbol sentiment
        if not sentiment_df.empty:
            temp_table = f"temp_sentiment_{int(time.time())}"
            sentiment_df.to_sql(temp_table, conn, if_exists='replace', index=False)
            
            conn.execute(text(f"""
                INSERT INTO news_sentiment_by_symbol 
                (article_id, symbol, relevance_score, sentiment_score, sentiment_label)
                SELECT article_id, symbol, relevance_score, sentiment_score, sentiment_label
                FROM {temp_table}
                ON CONFLICT (article_id, symbol) DO UPDATE SET
                    relevance_score = EXCLUDED.relevance_score,
                    sentiment_score = EXCLUDED.sentiment_score,
                    sentiment_label = EXCLUDED.sentiment_label
            """))
            
            conn.execute(text(f"DROP TABLE {temp_table}"))
        
        # Insert topics
        if not topics_df.empty:
            temp_table = f"temp_topics_{int(time.time())}"
            topics_df.to_sql(temp_table, conn, if_exists='replace', index=False)
            
            conn.execute(text(f"""
                INSERT INTO news_topics (article_id, topic, relevance_score)
                SELECT article_id, topic, relevance_score
                FROM {temp_table}
                WHERE topic != ''
                ON CONFLICT DO NOTHING
            """))
            
            conn.execute(text(f"DROP TABLE {temp_table}"))
    
    logger.info(f"Stored {len(articles_df)} unique articles with sentiment")

def calculate_daily_sentiment_summary():
    """Calculate aggregated daily sentiment metrics"""
    logger.info("Calculating daily sentiment summaries")
    
    with engine.begin() as conn:
        # Get date range to process
        result = conn.execute(text("""
            SELECT 
                MIN(DATE(na.published_at)) as min_date,
                MAX(DATE(na.published_at)) as max_date
            FROM news_articles na
            WHERE na.published_at > CURRENT_DATE - INTERVAL '30 days'
        """)).fetchone()
        
        if not result or not result[0]:
            logger.warning("No news data to summarize")
            return
        
        # Calculate daily summaries
        conn.execute(text("""
            INSERT INTO daily_sentiment_summary (
                symbol, date, avg_sentiment_score, weighted_sentiment_score,
                bullish_count, bearish_count, neutral_count, total_articles,
                avg_relevance_score, sentiment_momentum, sentiment_volatility
            )
            WITH daily_sentiment AS (
                SELECT 
                    ns.symbol,
                    DATE(na.published_at) as date,
                    ns.sentiment_score,
                    ns.sentiment_label,
                    ns.relevance_score
                FROM news_sentiment_by_symbol ns
                JOIN news_articles na ON ns.article_id = na.article_id
                WHERE na.published_at > CURRENT_DATE - INTERVAL '30 days'
            ),
            daily_stats AS (
                SELECT 
                    symbol,
                    date,
                    AVG(sentiment_score) as avg_sentiment,
                    SUM(sentiment_score * relevance_score) / NULLIF(SUM(relevance_score), 0) as weighted_sentiment,
                    SUM(CASE WHEN sentiment_score > 0.15 THEN 1 ELSE 0 END) as bullish_count,
                    SUM(CASE WHEN sentiment_score < -0.15 THEN 1 ELSE 0 END) as bearish_count,
                    SUM(CASE WHEN sentiment_score BETWEEN -0.15 AND 0.15 THEN 1 ELSE 0 END) as neutral_count,
                    COUNT(*) as total_articles,
                    AVG(relevance_score) as avg_relevance,
                    STDDEV(sentiment_score) as sentiment_vol
                FROM daily_sentiment
                GROUP BY symbol, date
            ),
            momentum AS (
                SELECT 
                    symbol,
                    date,
                    avg_sentiment,
                    weighted_sentiment,
                    bullish_count,
                    bearish_count,
                    neutral_count,
                    total_articles,
                    avg_relevance,
                    sentiment_vol,
                    avg_sentiment - LAG(avg_sentiment, 1) OVER (
                        PARTITION BY symbol ORDER BY date
                    ) as sentiment_momentum
                FROM daily_stats
            )
            SELECT * FROM momentum
            ON CONFLICT (symbol, date) DO UPDATE SET
                avg_sentiment_score = EXCLUDED.avg_sentiment_score,
                weighted_sentiment_score = EXCLUDED.weighted_sentiment_score,
                bullish_count = EXCLUDED.bullish_count,
                bearish_count = EXCLUDED.bearish_count,
                neutral_count = EXCLUDED.neutral_count,
                total_articles = EXCLUDED.total_articles,
                avg_relevance_score = EXCLUDED.avg_relevance_score,
                sentiment_momentum = EXCLUDED.sentiment_momentum,
                sentiment_volatility = EXCLUDED.sentiment_volatility,
                updated_at = CURRENT_TIMESTAMP
        """))
        
        # Get summary stats
        result = conn.execute(text("""
            SELECT 
                COUNT(DISTINCT symbol) as symbols,
                COUNT(DISTINCT date) as days,
                SUM(total_articles) as articles
            FROM daily_sentiment_summary
            WHERE date > CURRENT_DATE - INTERVAL '7 days'
        """)).fetchone()
        
        if result:
            logger.info(f"Summarized {result[2]} articles for {result[0]} symbols over {result[1]} days")

def fetch_news_for_universe(batch_size: int = 50):
    """Fetch news for entire symbol universe"""
    # Get active symbols
    with engine.connect() as conn:
        symbols = pd.read_sql(text("""
            SELECT symbol 
            FROM symbol_universe 
            WHERE is_active = true 
                AND is_etf = false
                AND market_cap_group IN ('large_cap', 'mid_cap')
            ORDER BY market_cap DESC
        """), conn)["symbol"].tolist()
    
    if not symbols:
        logger.warning("No symbols found in universe")
        return
    
    logger.info(f"Fetching news for {len(symbols)} symbols")
    
    # Process in batches (API accepts max 50 symbols)
    all_articles = []
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i+batch_size]
        logger.info(f"Processing batch {i//batch_size + 1}/{(len(symbols)-1)//batch_size + 1}")
        
        # Fetch news for batch
        articles = fetch_news_sentiment(symbols=batch)
        all_articles.extend(articles)
        
        # Store articles
        if articles:
            store_news_articles(articles)
        
        # Rate limiting
        if i + batch_size < len(symbols):
            time.sleep(RATE_LIMIT_DELAY)
    
    logger.info(f"Fetched {len(all_articles)} total articles")
    
    # Calculate summaries
    calculate_daily_sentiment_summary()

def fetch_topic_news():
    """Fetch news for specific market topics"""
    topics = [
        "earnings",
        "ipo",
        "mergers_and_acquisitions", 
        "financial_markets",
        "economy_fiscal",
        "economy_monetary",
        "economy_macro",
        "technology",
        "finance",
        "retail_wholesale"
    ]
    
    for topic in topics:
        logger.info(f"Fetching news for topic: {topic}")
        articles = fetch_news_sentiment(topics=[topic])
        
        if articles:
            store_news_articles(articles)
        
        time.sleep(RATE_LIMIT_DELAY)
    
    # Calculate summaries
    calculate_daily_sentiment_summary()

def main():
    """Main execution"""
    start_time = time.time()
    log_script_start(logger, "fetch_news_sentiment", 
                    "Fetch market news sentiment from Alpha Vantage")
    
    try:
        # Verify tables exist
        if not verify_news_sentiment_tables():
            logger.error("Cannot proceed without database tables")
            sys.exit(1)
        
        # Fetch news for top symbols
        fetch_news_for_universe(batch_size=50)
        
        # Also fetch topic-based news
        fetch_topic_news()
        
        # Get final statistics
        with engine.connect() as conn:
            stats = conn.execute(text("""
                SELECT 
                    COUNT(DISTINCT article_id) as articles,
                    COUNT(DISTINCT symbol) as symbols,
                    MIN(published_at) as oldest,
                    MAX(published_at) as newest
                FROM news_articles na
                JOIN news_sentiment_by_symbol ns ON na.article_id = ns.article_id
                WHERE na.published_at > CURRENT_DATE - INTERVAL '30 days'
            """)).fetchone()
            
            if stats:
                logger.info(f"Database now contains {stats[0]} articles for {stats[1]} symbols")
                logger.info(f"Date range: {stats[2]} to {stats[3]}")
        
        duration = time.time() - start_time
        log_script_end(logger, "fetch_news_sentiment", True, duration, {
            "Status": "Success",
            "Articles fetched": stats[0] if stats else 0
        })
        
    except Exception as e:
        logger.error(f"Script failed: {e}")
        log_script_end(logger, "fetch_news_sentiment", False, 
                      time.time() - start_time)
        sys.exit(1)

if __name__ == "__main__":
    main()