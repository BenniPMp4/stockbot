"""
fundamentals.py
---------------
Fetches company fundamentals, peer comparables, valuation ratios,
and performs a simple DCF intrinsic value estimate for integration
with app.py or the main Streamlit UI.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, Any


@st.cache_data(ttl=86400)
def get_fundamentals(ticker: str) -> pd.DataFrame:
    """Fetch key financial and valuation ratios for a ticker."""
    st.info(f"🏦 Fetching fundamental data for {ticker}...")

    try:
        t = yf.Ticker(ticker)
        info = t.info or {}

        df = pd.DataFrame([{
            "Ticker": ticker,
            "MarketCap": info.get("marketCap"),
            "EnterpriseValue": info.get("enterpriseValue"),
            "Revenue": info.get("totalRevenue"),
            "EBITDA": info.get("ebitda"),
            "FreeCashFlow": info.get("freeCashflow"),
            "OperatingCashFlow": info.get("operatingCashflow"),
            "DebtToEquity": info.get("debtToEquity"),
            "ProfitMargin": info.get("profitMargins"),
            "EarningsGrowth": info.get("earningsGrowth"),
            "RevenueGrowth": info.get("revenueGrowth"),
            "TrailingPE": info.get("trailingPE"),
            "ForwardPE": info.get("forwardPE"),
            "PEG": info.get("pegRatio"),
            "PriceToBook": info.get("priceToBook"),
            "DividendYield": info.get("dividendYield"),
            "Beta": info.get("beta"),
        }])

        # Derived metrics
        df["P/S"] = df["MarketCap"] / df["Revenue"]
        df["EV/EBITDA"] = df["EnterpriseValue"] / df["EBITDA"]

        st.success(f"✅ Fundamentals fetched for {ticker}")
        return df

    except Exception as e:
        st.error(f"❌ Failed to fetch fundamentals for {ticker}: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=86400)
def get_comparables(ticker: str, peer_count: int = 5) -> pd.DataFrame:
    """Fetch fundamental data for comparable companies within the same sector."""
    st.info(f"🤝 Building comparable-company set for {ticker}...")

    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
        sector = info.get("sector")
        industry = info.get("industry")

        if not sector:
            st.warning("⚠️ Sector info not available — skipping comparables.")
            return pd.DataFrame()

        sector_peers = {
            "Technology": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "ORCL", "NVDA"],
            "Financial Services": ["JPM", "BAC", "GS", "MS", "WFC"],
            "Healthcare": ["JNJ", "PFE", "ABBV", "MRK", "UNH"],
            "Energy": ["XOM", "CVX", "COP", "SLB"],
            "Consumer Cyclical": ["TSLA", "NKE", "HD", "MCD", "SBUX"],
        }

        peers = sector_peers.get(sector, [])
        if ticker in peers:
            peers.remove(ticker)
        peers = peers[:peer_count]

        peer_dfs = [get_fundamentals(p) for p in peers if p]
        peer_dfs = [df for df in peer_dfs if not df.empty]
        if not peer_dfs:
            return pd.DataFrame()

        df_peers = pd.concat(peer_dfs, ignore_index=True)
        return df_peers

    except Exception as e:
        st.error(f"❌ Failed to fetch comparables: {e}")
        return pd.DataFrame()


def compute_valuation_gap(ticker: str, peers_df: pd.DataFrame) -> pd.DataFrame:
    """Compute valuation gaps (% over/under) vs peer medians."""
    try:
        if peers_df.empty:
            return pd.DataFrame()

        peer_medians = peers_df.median(numeric_only=True)
        target = get_fundamentals(ticker).iloc[0]
        metrics = ["ForwardPE", "EV/EBITDA", "P/S", "PriceToBook"]

        rows = []
        for m in metrics:
            peer_val = peer_medians.get(m)
            tgt_val = target.get(m)
            if pd.notna(peer_val) and pd.notna(tgt_val) and peer_val != 0:
                diff_pct = (tgt_val - peer_val) / peer_val * 100
                rows.append({"Metric": m, "Peer Median": peer_val, "Target": tgt_val, "V-Gap (%)": diff_pct})

        return pd.DataFrame(rows)

    except Exception as e:
        st.error(f"❌ V-gap computation failed: {e}")
        return pd.DataFrame()


# === DCF Estimation ===
def run_dcf_valuation(ticker: str, discount_rate: float = 0.09, terminal_growth: float = 0.025, years: int = 5):
    """
    Perform a basic Discounted Cash Flow valuation.
    Uses trailing free cash flow and growth rates from Yahoo Finance.
    """
    try:
        st.info(f"💰 Running DCF model for {ticker}...")

        t = yf.Ticker(ticker)
        info = t.info or {}

        fcf = info.get("freeCashflow") or info.get("operatingCashflow")
        shares_outstanding = info.get("sharesOutstanding")
        growth = info.get("earningsGrowth", 0.08)  # fallback to 8%
        market_price = info.get("currentPrice", np.nan)

        if not fcf or not shares_outstanding:
            st.warning("⚠️ Insufficient FCF or share data for DCF.")
            return pd.DataFrame()

        fcf_list = []
        pv_list = []

        for i in range(1, years + 1):
            fcf_next = fcf * ((1 + growth) ** i)
            pv = fcf_next / ((1 + discount_rate) ** i)
            fcf_list.append(fcf_next)
            pv_list.append(pv)

        terminal_value = fcf_list[-1] * (1 + terminal_growth) / (discount_rate - terminal_growth)
        pv_terminal = terminal_value / ((1 + discount_rate) ** years)

        intrinsic_value = (sum(pv_list) + pv_terminal) / shares_outstanding
        upside = (intrinsic_value - market_price) / market_price * 100 if market_price else np.nan

        df_dcf = pd.DataFrame({
            "Year": [f"Year {i}" for i in range(1, years + 1)] + ["Terminal Value"],
            "FCF (Projected)": fcf_list + [terminal_value],
            "PV of FCF": pv_list + [pv_terminal],
        })

        summary = pd.DataFrame([{
            "Ticker": ticker,
            "Intrinsic Value": intrinsic_value,
            "Market Price": market_price,
            "Upside (%)": upside,
        }])

        st.success("✅ DCF valuation complete.")
        return df_dcf, summary

    except Exception as e:
        st.error(f"❌ DCF model failed: {e}")
        return pd.DataFrame(), pd.DataFrame()
