import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from data.preprocessing import combine_data
from data.api import get_news_data
import pandas_datareader.data as web
from typing import Optional

@st.cache_data(ttl=3600)
def get_stock_data(
    ticker: str,
    start_date: str,
    end_date: str,
    include_sentiment: bool = True,
    daily_sentiment: Optional[pd.DataFrame] = None,
):
    """
    Fetch OHLCV stock data and merge with benchmark, macro indicators, and optional sentiment.
    Prevents duplicate sentiment fetching and guarantees computed_score integrity.
    """
    st.info(f"📈 Fetching stock data for {ticker}...")

    try:
        # --- Base stock data ---
        # If the selected date window is too short (<30 days), automatically extend it to 1 year for modeling stability.
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        if (end_dt - start_dt).days < 30:
            extended_start = (end_dt - pd.Timedelta(days=180)).strftime("%Y-%m-%d")
            st.info(f"📅 Extending data window to 1 year back: {extended_start} → {end_date}")
        else:
            extended_start = start_date

        df = yf.download(ticker, start=extended_start, end=end_date, progress=False, auto_adjust=False)

        if df is None or df.empty:
            st.error(f"❌ No stock data found for {ticker}.")
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ['_'.join(col).strip() for col in df.columns.values]
        df = df.reset_index()
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.tz_localize(
            "UTC", nonexistent="shift_forward", ambiguous="NaT"
        )
        df.dropna(subset=["Date"], inplace=True)
        df["DateOnly"] = df["Date"].dt.normalize()

        # --- Close / Volume / Features ---
        close_col = next((c for c in df.columns if "Close" in c), None)
        if close_col is None:
            st.error("❌ No Close column found.")
            return None

        df["Close"] = pd.to_numeric(df[close_col], errors="coerce")
        df["Pct_Change"] = df["Close"].pct_change() * 100

        df["Volume"] = (
            pd.to_numeric(df.get("Volume", pd.Series(0.0, index=df.index)), errors="coerce")
            .fillna(method="ffill")
            .fillna(0.0)
        )

        df["MA20"] = df["Close"].rolling(20, min_periods=1).mean()
        df["MA50"] = df["Close"].rolling(50, min_periods=1).mean()
        df["Volatility"] = df["Pct_Change"].rolling(7, min_periods=1).std()
        df["RSI"] = compute_rsi(df["Close"])
        df["LogVolume"] = np.log1p(df["Volume"])

                # --- Extra technical indicators for model learning ---
        # Momentum indicators
        df["Momentum_5"] = df["Close"].pct_change(5) * 100
        df["Momentum_14"] = df["Close"].pct_change(14) * 100

        # Bollinger Bands (20-day)
        rolling_mean = df["Close"].rolling(20)
        df["Bollinger_Mid"] = rolling_mean.mean()
        df["Bollinger_Upper"] = df["Bollinger_Mid"] + 2 * rolling_mean.std()
        df["Bollinger_Lower"] = df["Bollinger_Mid"] - 2 * rolling_mean.std()

        # MACD (12/26 EMA)
        ema_fast = df["Close"].ewm(span=12, adjust=False).mean()
        ema_slow = df["Close"].ewm(span=26, adjust=False).mean()
        df["MACD"] = ema_fast - ema_slow
        df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

        # Price rate of change
        df["ROC_10"] = df["Close"].pct_change(10) * 100

        # Lagged volatility measures
        df["Volatility_14"] = df["Pct_Change"].rolling(14, min_periods=1).std()


        # --- Benchmark (S&P500) ---
        try:
            benchmark = yf.download("^GSPC", start=start_date, end=end_date, progress=False)
            if isinstance(benchmark.columns, pd.MultiIndex):
                benchmark.columns = ['_'.join(col).strip() for col in benchmark.columns.values]
            bench_close = next((c for c in benchmark.columns if "Close" in c), None)
            if bench_close:
                benchmark = benchmark.reset_index()
                benchmark["Date"] = pd.to_datetime(benchmark["Date"], errors="coerce").dt.tz_localize("UTC")
                df = pd.merge(df, benchmark[["Date", bench_close]], on="Date", how="left", suffixes=("", "_SPX"))
                df["RelStrength"] = df["Close"] / df[bench_close]
            else:
                df["RelStrength"] = 1.0
        except Exception:
            df["RelStrength"] = 1.0

        # --- Macro Data ---
        macro = fetch_macro_data(start_date, end_date)
        df = pd.merge(df, macro, on="DateOnly", how="left")

        # --- Sentiment Integration ---
        if include_sentiment and daily_sentiment is not None:
            daily_sent = _prepare_sentiment_data(ticker, daily_sentiment)
            combined = combine_data(daily_sent, df)
        else:
            st.info("💤 Skipping sentiment merge (not provided).")
            combined = df.copy()

                # --- Sentiment lag features (3-day & 7-day) ---
        if "computed_score" in combined.columns:
            combined = combined.sort_values("DateOnly")
            combined["Lagged_Sentiment_3d"] = combined["computed_score"].shift(3)
            combined["Lagged_Sentiment_7d"] = combined["computed_score"].shift(7)
            combined["Sentiment_MA7"] = combined["computed_score"].rolling(7, min_periods=1).mean()
            combined["Sentiment_STD7"] = combined["computed_score"].rolling(7, min_periods=1).std()
        else:
            combined["Lagged_Sentiment_3d"] = 0.0
            combined["Lagged_Sentiment_7d"] = 0.0
            combined["Sentiment_MA7"] = 0.0
            combined["Sentiment_STD7"] = 0.0


        # --- Correlation filter ---
        if "sentiment_score" in combined.columns:
            corr = combined["sentiment_score"].corr(combined["Pct_Change"])
            if pd.notna(corr) and abs(corr) < 0.2:
                st.info(f"🪶 Ignoring weak sentiment correlation ({corr:.3f}).")
                combined = combined.drop(columns=[c for c in ["sentiment_score", "computed_score"] if c in combined.columns])

        # --- Scaling ---
        numeric_cols = combined.select_dtypes(include=[np.number]).columns
        combined[numeric_cols] = (
            combined[numeric_cols]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(method="ffill")
            .fillna(method="bfill")
        )
        scaler = StandardScaler()
        scaled = pd.DataFrame(scaler.fit_transform(combined[numeric_cols]), columns=numeric_cols, index=combined.index)
        for col in numeric_cols:
            combined[f"{col}_scaled"] = scaled[col]

        st.success(f"✅ Stock data ready: {len(combined)} rows, {len(combined.columns)} columns")
        return combined

    except Exception as e:
        st.error(f"❌ Failed to fetch stock data: {e}")
        return None


# === Sentiment Handling Helper ===
from utils.helpers import safe_utc_localize, ensure_daily_sentiment

from utils.helpers import safe_utc_localize, ensure_daily_sentiment

def _prepare_sentiment_data(ticker: str, daily_sentiment: Optional[pd.DataFrame]) -> pd.DataFrame:
    """
    Prepare sentiment safely. Never fetch from API here.
    Added diagnostic prints so you can see what the function receives and returns.
    """
    try:
        print(f"DEBUG _prepare_sentiment_data: ticker={ticker}, incoming daily_sentiment type={type(daily_sentiment)}")
        if daily_sentiment is None or (isinstance(daily_sentiment, pd.DataFrame) and daily_sentiment.empty):
            print("DEBUG _prepare_sentiment_data: no daily_sentiment provided or empty - returning empty placeholder")
            st.warning("⚠️ No sentiment data provided — defaulting to neutral.")
            return pd.DataFrame({
                "DateOnly": pd.to_datetime([], utc=True),
                "sentiment_score": [],
                "computed_score": []
            })

        print("DEBUG _prepare_sentiment_data: received daily_sentiment.shape =", daily_sentiment.shape)
        print("DEBUG _prepare_sentiment_data: received columns =", daily_sentiment.columns.tolist())
        print("DEBUG _prepare_sentiment_data: head =")
        print(daily_sentiment.head(5))

        daily_sent = ensure_daily_sentiment(daily_sentiment)
        print("DEBUG _prepare_sentiment_data: after ensure_daily_sentiment shape =", None if daily_sent is None else daily_sent.shape)

        if daily_sent is None or daily_sent.empty:
            print("DEBUG _prepare_sentiment_data: ensure_daily_sentiment returned empty -> neutral placeholder")
            return pd.DataFrame({
                "DateOnly": pd.to_datetime([], utc=True),
                "sentiment_score": [],
                "computed_score": []
            })

        # --- Normalize datetime safely ---
        if "Date" in daily_sent.columns:
            print("DEBUG _prepare_sentiment_data: converting Date -> DateOnly")
            daily_sent["Date"] = safe_utc_localize(daily_sent["Date"])
            # convert to tz-naive DateOnly
            daily_sent["DateOnly"] = daily_sent["Date"].dt.tz_convert(None).dt.normalize()
        elif "DateOnly" in daily_sent.columns:
            daily_sent["DateOnly"] = pd.to_datetime(daily_sent["DateOnly"], errors="coerce").dt.tz_localize(None)

        # --- Ensure sentiment columns ---
        if "computed_score" not in daily_sent.columns and "sentiment_score" in daily_sent.columns:
            print("DEBUG _prepare_sentiment_data: creating computed_score from sentiment_score")
            daily_sent["computed_score"] = pd.to_numeric(daily_sent["sentiment_score"], errors="coerce").fillna(0.0).astype(float)
        if "sentiment_score" not in daily_sent.columns and "computed_score" in daily_sent.columns:
            daily_sent["sentiment_score"] = pd.to_numeric(daily_sent["computed_score"], errors="coerce").fillna(0.0).astype(float)

        # final debug before return
        daily_sent = daily_sent.drop_duplicates(subset=["DateOnly"])
        print("DEBUG _prepare_sentiment_data: final daily_sent columns:", daily_sent.columns.tolist())
        print("DEBUG _prepare_sentiment_data: final head:")
        print(daily_sent.head(10))

        # Keep only needed columns
        return daily_sent[["DateOnly", "sentiment_score", "computed_score"]]

    except Exception as e:
        print("ERROR _prepare_sentiment_data:", e)
        st.error(f"❌ Failed preparing sentiment data: {e}")
        return pd.DataFrame({
            "DateOnly": pd.to_datetime([], utc=True),
            "sentiment_score": [],
            "computed_score": []
        })



# === RSI & Macro ===
def compute_rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(com=(period - 1), adjust=False).mean()
    ma_down = down.ewm(com=(period - 1), adjust=False).mean()
    rs = ma_up / (ma_down + 1e-9)
    return 100 - (100 / (1 + rs)).fillna(50)


def fetch_macro_data(start_date, end_date):
    macros = {}
    macro_tickers = {
        "VIX": "^VIX",
        "US10Y": "^TNX",
        "DXY": "DX-Y.NYB",
        "GOLD": "GC=F",
        "OIL": "CL=F"
    }

    for label, sym in macro_tickers.items():
        try:
            data = yf.download(sym, start=start_date, end=end_date, progress=False)
            if data is None or data.empty:
                continue
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = ['_'.join(col).strip() for col in data.columns.values]

            close_col = next((c for c in data.columns if "Close" in c), None)
            if not close_col:
                continue

            temp = data.reset_index()[["Date", close_col]].rename(columns={close_col: label})
            temp["Date"] = pd.to_datetime(temp["Date"], errors="coerce").dt.tz_localize("UTC", nonexistent="shift_forward", ambiguous="NaT")
            temp["DateOnly"] = temp["Date"].dt.normalize()
            macros[label] = temp[["DateOnly", label]]

        except Exception:
            continue

    if not macros:
        return pd.DataFrame({"DateOnly": pd.date_range(start=start_date, end=end_date, freq="D", tz="UTC")})

    df_macro = None
    for label, data in macros.items():
        df_macro = data if df_macro is None else pd.merge(df_macro, data, on="DateOnly", how="outer")

    return df_macro.sort_values("DateOnly").fillna(method="ffill").fillna(method="bfill")
