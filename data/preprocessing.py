# data/preprocessing.py

import pandas as pd
import numpy as np
import streamlit as st
from typing import Optional

from utils.helpers import (
    compute_weighted_daily_sentiment,
    normalize_sentiment_columns,
    safe_utc_localize,
)


def process_sentiment_data(
    articles_df: pd.DataFrame,
    decay: float = 0.95,
    min_days: int = 15,
    debug: bool = True,
) -> pd.DataFrame:
    """Convert article-level sentiment into daily weighted sentiment."""
    if articles_df is None or articles_df.empty:
        st.warning("No sentiment data available for processing.")
        return pd.DataFrame(columns=["DateOnly", "sentiment_score", "Date"])

    df = articles_df.copy()
    if debug:
        print(f"🔍 process_sentiment_data: starting with {len(df)} articles")
        print("🔍 sample article columns:", df.columns.tolist())

    # --- Ensure numeric sentiment_score ---
    if "sentiment_score" not in df.columns:
        if "sentiment" in df.columns:
            st.warning("⚠️ 'sentiment_score' missing — coercing from 'sentiment'")
            df["sentiment_score"] = pd.to_numeric(df["sentiment"], errors="coerce")
        else:
            st.error("❌ Missing 'sentiment_score' and no fallback column.")
            return pd.DataFrame(columns=["DateOnly", "sentiment_score", "Date"])

    df["sentiment_score"] = (
        pd.to_numeric(df["sentiment_score"], errors="coerce")
        .fillna(0.0)
        .astype(float)
    )

    # --- Date handling (support Date, DateOnly, fallback to published) ---
    if "Date" in df.columns:
        df["Date"] = safe_utc_localize(df["Date"])
    elif "DateOnly" in df.columns:
        df["DateOnly"] = pd.to_datetime(df["DateOnly"], errors="coerce").dt.normalize()
        df["Date"] = safe_utc_localize(df["DateOnly"])
    elif "published" in df.columns:
        st.warning("⚠️ Using 'published' as date source")
        df["Date"] = safe_utc_localize(df["published"])
    else:
        st.error("❌ No date column found in sentiment data")
        return pd.DataFrame(columns=["DateOnly", "sentiment_score", "Date"])

    # Drop missing
    before = len(df)
    df = df.dropna(subset=["Date"]).reset_index(drop=True)
    after = len(df)
    if debug:
        print(f"🔍 DEBUG: dropped {before - after} rows with invalid dates")

    # --- Daily weighted sentiment ---
    daily = compute_weighted_daily_sentiment(
        df, decay_val=decay, min_days=min_days, debug=debug
    )

    return daily


def combine_data(
    sentiment_df: Optional[pd.DataFrame],
    stock_df: Optional[pd.DataFrame],
    lag_days: int = 1,
    join_type: str = "left",
) -> pd.DataFrame:
    """Safely merge stock and sentiment data, ensuring consistency and clear deduplication.
    Includes detailed debug logs to diagnose 'computed_score' issues.
    """
    print("DEBUG combine_data: start")
    if stock_df is None or stock_df.empty:
        print("❌ combine_data: stock_df missing or empty")
        st.error("❌ Missing stock data.")
        return pd.DataFrame()

    stock = stock_df.copy()
    sent = pd.DataFrame() if sentiment_df is None else sentiment_df.copy()

    print(f"DEBUG combine_data: stock shape={stock.shape}, sent shape={sent.shape}")

    # --- Date normalization ---
    try:
        stock["Date"] = safe_utc_localize(stock.get("Date", pd.NaT))
    except Exception as e:
        print("ERROR combine_data: safe_utc_localize(stock.Date) failed:", e)
        stock["Date"] = pd.to_datetime(stock.get("Date", pd.NaT), errors="coerce")

    stock["DateOnly"] = stock["Date"].dt.tz_localize(None).dt.normalize()
    print("DEBUG combine_data: stock DateOnly sample:", stock["DateOnly"].head().tolist())

    # --- Prepare sentiment ---
    if sent.empty:
        print("DEBUG combine_data: sentiment_df empty -> generating neutral placeholder")
        sent = pd.DataFrame({"DateOnly": stock["DateOnly"].unique()})
        sent["sentiment_score"] = 0.0
        sent["computed_score"] = 0.0
    else:
        if "DateOnly" not in sent.columns:
            if "Date" in sent.columns:
                sent["Date"] = safe_utc_localize(sent["Date"])
                sent["DateOnly"] = sent["Date"].dt.tz_convert(None).dt.normalize()
            else:
                sent["DateOnly"] = stock["DateOnly"].unique()

    # --- Column alignment ---
    print("DEBUG combine_data: before column fix, sent.columns =", list(sent.columns))

    for col in ["sentiment_score", "computed_score"]:
        if col not in sent.columns:
            sent[col] = 0.0

    sent["sentiment_score"] = pd.to_numeric(sent["sentiment_score"], errors="coerce").fillna(0.0)
    sent["computed_score"] = pd.to_numeric(sent["computed_score"], errors="coerce").fillna(sent["sentiment_score"])
    sent = sent.drop_duplicates(subset=["DateOnly"])

    print("DEBUG combine_data: after column fix, sent dtypes:")
    print(sent.dtypes)
    print("DEBUG combine_data: sent head:\n", sent.head())

    # --- Close & Volume fallback ---
    close_col = next((c for c in ["Close", "Adj Close", "close", "adjclose"] if c in stock.columns), None)
    if close_col is not None:
        stock["Close"] = pd.to_numeric(stock[close_col], errors="coerce")
    else:
        print("⚠️ combine_data: No close column found — filling NaN")
        stock["Close"] = np.nan

    if "Volume" not in stock.columns:
        stock["Volume"] = 0.0

    if "Pct_Change" not in stock.columns:
        stock["Pct_Change"] = stock["Close"].pct_change() * 100.0

    # --- Merge with clear suffix control ---
    merged = pd.merge(
        stock,
        sent[["DateOnly", "computed_score", "sentiment_score"]],
        on="DateOnly",
        how=join_type,
        validate="m:1",
        suffixes=("", "_sent")
    )

    print(f"DEBUG combine_data: merged shape={merged.shape}")
    print(f"DEBUG combine_data: merged columns={list(merged.columns)}")

    # --- Deduplicate computed/sentiment columns ---
    for base in ["computed_score", "sentiment_score"]:
        if f"{base}_sent" in merged.columns:
            if base not in merged.columns:
                merged[base] = merged[f"{base}_sent"]
            else:
                merged[base] = merged[base].combine_first(merged[f"{base}_sent"])
            merged.drop(columns=[f"{base}_sent"], inplace=True, errors="ignore")

    # --- Add lagged sentiment & rolling positivity ---
    merged["Lagged_Sentiment"] = merged["computed_score"].shift(lag_days)
    merged["is_positive"] = (merged["computed_score"] > 0).astype(int)
    merged["7day_pct_positive"] = (
        merged["is_positive"].rolling(7, min_periods=1).mean() * 100
    )

    merged = merged.loc[:, ~merged.columns.duplicated()]
    merged = merged.replace([np.inf, -np.inf], np.nan).fillna(0)

    print(f"DEBUG combine_data: final columns={list(merged.columns)}")
    print(f"DEBUG combine_data: end, returning merged with shape {merged.shape}")
    return merged



def prepare_sentiment_for_forecast(daily_sentiment: pd.DataFrame) -> pd.DataFrame:
    """Prepare daily sentiment for forecasting (no stock merge)."""
    if daily_sentiment is None or daily_sentiment.empty:
        return pd.DataFrame(columns=["DateOnly", "computed_score"])

    df = daily_sentiment.copy()
    if "DateOnly" not in df.columns and "Date" in df.columns:
        df["DateOnly"] = safe_utc_localize(df["Date"]).dt.normalize()

    df["DateOnly"] = safe_utc_localize(df["DateOnly"]).dt.normalize()
    df = df.dropna(subset=["DateOnly"]).sort_values("DateOnly").reset_index(drop=True)

    return df[["DateOnly", "computed_score"]].set_index("DateOnly")
