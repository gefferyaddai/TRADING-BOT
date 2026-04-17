"""
sentiment.py — News fetching and Claude-powered sentiment scoring.

Public API:
    get_sentiment_scores(symbols)  → Dict[str, float]
        Returns a score per symbol in range [-1.0, +1.0]
        0.0 returned for any symbol with insufficient news data.

How it works:
    1. Fetch last SENTIMENT_LOOKBACK_HOURS of headlines via Finnhub
    2. For each symbol, send headlines + summaries to Claude for scoring
    3. Apply recency weighting (newer articles weighted more)
    4. Require SENTIMENT_CONSENSUS_MIN headlines to agree before trusting score
    5. Return composite score per symbol

Score interpretation:
    < -0.3  → blocked from entry (SENTIMENT_GATE_THRESHOLD)
    -0.3 to 0.3  → neutral, normal size
     0.3 to 0.6  → mildly positive, normal size
     0.6 to 0.8  → strong positive, 1.5× size
     0.8 to 1.0  → very strong, 2× size
"""

import json
import logging
import re
import time
from datetime import datetime, timedelta
from typing import Dict, List

import requests

from config import CFG

log = logging.getLogger(__name__)

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
FINNHUB_NEWS_URL  = "https://finnhub.io/api/v1/company-news"


# ══════════════════════════════════════════════════════════════════════════════
# Finnhub news fetching
# ══════════════════════════════════════════════════════════════════════════════

def _fetch_headlines(symbol: str) -> List[dict]:
    """
    Fetch recent news articles for a symbol from Finnhub.
    Returns list of dicts with: headline, summary, datetime (unix timestamp).
    """
    end   = datetime.utcnow()
    start = end - timedelta(hours=CFG.SENTIMENT_LOOKBACK_HOURS)

    params = {
        "symbol": symbol,
        "from":   start.strftime("%Y-%m-%d"),
        "to":     end.strftime("%Y-%m-%d"),
        "token":  CFG.FINNHUB_API_KEY,
    }

    try:
        resp = requests.get(FINNHUB_NEWS_URL, params=params, timeout=10)
        resp.raise_for_status()
        articles = resp.json()

        # Filter to only articles within the lookback window
        cutoff = start.timestamp()
        recent = [a for a in articles if a.get("datetime", 0) >= cutoff]

        log.debug(f"  {symbol}: {len(recent)} articles in last {CFG.SENTIMENT_LOOKBACK_HOURS}h")
        return recent

    except Exception as e:
        log.warning(f"Finnhub fetch failed for {symbol}: {e}")
        return []


# ══════════════════════════════════════════════════════════════════════════════
# Claude sentiment scoring
# ══════════════════════════════════════════════════════════════════════════════

def _score_articles_with_claude(symbol: str, articles: List[dict]) -> List[dict]:
    """
    Send article headlines and summaries to Claude for sentiment scoring.

    Returns a list of scored articles:
        [{"headline": ..., "score": float, "confidence": float, "age_hours": float}]

    Each score is in [-1.0, +1.0]:
        -1.0 = very negative (e.g. fraud, bankruptcy, major miss)
         0.0 = neutral
        +1.0 = very positive (e.g. blowout earnings, major contract win)
    """
    if not articles:
        return []

    # Build the prompt — send up to 10 most recent articles
    recent = sorted(articles, key=lambda x: x.get("datetime", 0), reverse=True)[:10]

    articles_text = ""
    for i, a in enumerate(recent, 1):
        headline = a.get("headline", "").strip()
        summary  = a.get("summary",  "").strip()[:300]   # cap summary length
        articles_text += f"\n{i}. HEADLINE: {headline}\n   SUMMARY: {summary}\n"

    prompt = f"""You are a financial news sentiment analyst. Score each news article about {symbol} stock.

For each article, provide:
- score: a float from -1.0 (very negative) to +1.0 (very positive) for the stock
- confidence: a float from 0.0 to 1.0 indicating how clearly the article affects the stock

Consider:
- Earnings beats/misses, revenue growth, guidance changes
- Product launches, partnerships, regulatory approvals
- Executive changes, legal issues, fraud allegations
- Macro headwinds/tailwinds specific to this company
- Analyst upgrades/downgrades

Be conservative — only give extreme scores for clear-cut news.

Articles:
{articles_text}

Respond ONLY with a JSON array, no other text. Example format:
[{{"index": 1, "score": 0.7, "confidence": 0.9}}, {{"index": 2, "score": -0.3, "confidence": 0.6}}]"""

    try:
        resp = requests.post(
            ANTHROPIC_API_URL,
            headers={
                "Content-Type": "application/json",
                "x-api-key": CFG.ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
            },
            json={
                "model": "claude-sonnet-4-6",
                "max_tokens": 1000,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=30,
        )
        resp.raise_for_status()
        raw_text = resp.json()["content"][0]["text"].strip()
        raw_text = re.sub(r"```[\w]*\n?", "", raw_text).strip()
        scores = json.loads(raw_text)

        # Attach scores back to articles
        scored = []
        now_ts = datetime.utcnow().timestamp()
        for s in scores:
            idx = s.get("index", 1) - 1
            if 0 <= idx < len(recent):
                article   = recent[idx]
                age_hours = (now_ts - article.get("datetime", now_ts)) / 3600
                scored.append({
                    "headline":   article.get("headline", ""),
                    "score":      float(s.get("score", 0.0)),
                    "confidence": float(s.get("confidence", 0.5)),
                    "age_hours":  age_hours,
                })

        return scored

    except Exception as e:
        log.warning(f"Claude scoring failed for {symbol}: {e}")
        return []


# ══════════════════════════════════════════════════════════════════════════════
# Composite score calculation
# ══════════════════════════════════════════════════════════════════════════════

def _composite_score(scored_articles: List[dict]) -> float:
    """
    Combine individual article scores into a single composite score.

    Two combat measures applied here:
    1. Recency weighting — articles from the last 6 hours get 2× weight
    2. Consensus requirement — if fewer than SENTIMENT_CONSENSUS_MIN articles
       agree on direction (positive or negative), return 0.0 (neutral)
       to avoid acting on a single outlier headline.
    """
    if not scored_articles:
        return 0.0

    # Recency weighting
    weighted_scores = []
    weights         = []

    for a in scored_articles:
        age_hours   = a["age_hours"]
        recency_wt  = 2.0 if age_hours <= 6 else 1.0
        conf_wt     = a["confidence"]
        total_wt    = recency_wt * conf_wt

        weighted_scores.append(a["score"] * total_wt)
        weights.append(total_wt)

    if sum(weights) == 0:
        return 0.0

    raw_composite = sum(weighted_scores) / sum(weights)

    # Consensus check — count how many articles agree with the composite direction.
    # A perfectly neutral composite (0.0) has no direction, so skip the check.
    if raw_composite == 0.0:
        return 0.0
    direction = 1 if raw_composite > 0 else -1
    agreeing  = sum(1 for a in scored_articles if (a["score"] * direction) > 0.1)

    if agreeing < CFG.SENTIMENT_CONSENSUS_MIN:
        log.debug(
            f"Consensus not met ({agreeing} < {CFG.SENTIMENT_CONSENSUS_MIN}) "
            f"— returning neutral score"
        )
        return 0.0

    # Clamp to [-1, 1]
    return max(-1.0, min(1.0, raw_composite))


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════

def get_sentiment_scores(symbols: List[str]) -> Dict[str, float]:
    """
    Fetch and score news sentiment for a list of symbols.

    Returns dict of {symbol: composite_score} for all symbols.
    Symbols with no news or failed fetches default to 0.0 (neutral).
    """
    scores = {}
    log.info(f"Fetching sentiment for {len(symbols)} symbols ...")

    for symbol in symbols:
        try:
            articles = _fetch_headlines(symbol)

            if not articles:
                log.info(f"  {symbol}: no recent news — neutral (0.0)")
                scores[symbol] = 0.0
                time.sleep(0.3)
                continue

            scored   = _score_articles_with_claude(symbol, articles)
            score    = _composite_score(scored)
            scores[symbol] = score

            # Log the result with context
            direction = "POSITIVE" if score > 0.1 else ("NEGATIVE" if score < -0.1 else "NEUTRAL")
            log.info(
                f"  {symbol:<6}  sentiment={score:+.2f}  "
                f"({direction})  articles={len(articles)}"
            )

            # Log individual headlines at debug level
            for a in scored:
                log.debug(
                    f"    [{a['score']:+.2f} conf={a['confidence']:.1f}] "
                    f"{a['headline'][:80]}"
                )

            # Rate limit — Finnhub free tier is 60 calls/min
            time.sleep(0.5)

        except Exception as e:
            log.warning(f"Sentiment failed for {symbol}: {e}")
            scores[symbol] = 0.0

    log.info("Sentiment scoring complete.")
    return scores