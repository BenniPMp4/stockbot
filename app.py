import streamlit as st
import pandas as pd
from data.api import get_news_data
from data.stockdata import get_stock_data
from data.preprocessing import process_sentiment_data, combine_data
from models.forecasting import fit_and_forecast
from visualisation.plots import create_plot, plot_price_with_forecast
from utils.helpers import calc_correlation, safe_utc_localize

# -------------------------------
# Streamlit App Setup
# -------------------------------
st.sidebar.title("📊 Predicting Stock Prices by News Sentiment")
ticker = st.sidebar.text_input("Enter stock ticker (e.g., MSFT):", value="MSFT")
run_button = st.sidebar.button("Run Analysis")

def ensure_daily_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure computed_score and sentiment_score exist and are numeric."""
    if df is None:
        return pd.DataFrame(columns=["DateOnly", "sentiment_score", "computed_score"])
    df = df.copy()
    # Normalize column names
    for col in ["sentiment_score", "computed_score"]:
        if col not in df.columns:
            df[col] = 0.0
    df["sentiment_score"] = pd.to_numeric(df["sentiment_score"], errors="coerce").fillna(0.0)
    df["computed_score"] = pd.to_numeric(df["computed_score"], errors="coerce").fillna(df["sentiment_score"])
    return df

if run_button:
    try:
        print("✅ Script started")

        # -------------------------------
        # Fetch & Cache Sentiment Data (once)
        # -------------------------------
        if "sentiment_cache" not in st.session_state or st.session_state["sentiment_cache"] is None or st.session_state["sentiment_cache"].empty:
            print("🚀 Fetching fresh news data...")
            news_data = get_news_data(ticker)
            # Defensive: validate structure
            if not news_data or "daily_sentiment" not in news_data:
                st.error("❌ get_news_data returned no daily_sentiment.")
                st.stop()
            st.session_state["sentiment_cache"] = news_data.get("daily_sentiment", pd.DataFrame())
            # keep articles if you want (not displayed by default)
            st.session_state["articles_cache"] = news_data.get("articles", pd.DataFrame())
        else:
            print("🧠 Using cached sentiment data...")
            news_data = {
                "daily_sentiment": st.session_state.get("sentiment_cache", pd.DataFrame()),
                "articles": st.session_state.get("articles_cache", pd.DataFrame())
            }

        articles_df = news_data.get("articles", pd.DataFrame())
        daily_sentiment = news_data.get("daily_sentiment", pd.DataFrame())

        # -------------------------------
        # Process Sentiment (if needed)
        # -------------------------------
        # If the API gave raw articles but not daily sentiment, process them
        if (daily_sentiment is None or daily_sentiment.empty) and (not articles_df.empty):
            print("🔄 Processing articles -> daily sentiment")
            daily_sentiment = process_sentiment_data(articles_df)

        # Ensure required columns and datetimes are normalized
        if daily_sentiment is None:
            daily_sentiment = pd.DataFrame(columns=["DateOnly", "sentiment_score", "computed_score"])

        # Normalize Date / DateOnly to naive-pandas dates (floor to day)
        for col in ["Date", "DateOnly"]:
            if col in daily_sentiment.columns:
                daily_sentiment[col] = pd.to_datetime(daily_sentiment[col], errors="coerce").dt.floor("D")

        daily_sentiment = ensure_cols = ensure_daily_columns(daily_sentiment)

        # (UI) skip showing article tables unless user explicitly wants them
        # st.subheader("📰 Recent Articles & Sentiment")
        # if not articles_df.empty:
        #     show_cols = [c for c in ["Date", "Title", "sentiment", "sentiment_score"] if c in articles_df.columns]
        #     st.dataframe(articles_df[show_cols].head(10))

        # -------------------------------
        # Date Window from Sentiment (drives stock fetch)
        # -------------------------------
        if "DateOnly" not in daily_sentiment.columns or daily_sentiment["DateOnly"].dropna().empty:
            st.error("❌ No valid DateOnly values in daily sentiment to determine date range.")
            st.stop()

        start_date = daily_sentiment["DateOnly"].min()
        end_date = daily_sentiment["DateOnly"].max()

        if pd.isna(start_date) or pd.isna(end_date):
            st.error("❌ No valid sentiment date range.")
            st.stop()

        # Convert to UTC-aware timestamps for yfinance
        start_date_utc = safe_utc_localize(pd.Series([start_date]))[0]
        end_date_utc = safe_utc_localize(pd.Series([end_date]))[0]

        # -------------------------------
        # Fetch Stock Data (Single Call) — pass daily_sentiment to avoid re-fetch
        # -------------------------------
        print(f"📈 Fetching stock data for {ticker} from {start_date_utc} to {end_date_utc}")
        stock_data = get_stock_data(
            ticker,
            start_date=start_date_utc,
            end_date=end_date_utc,
            daily_sentiment=daily_sentiment
        )

        if stock_data is None or stock_data.empty:
            st.error("❌ No stock data returned.")
            st.stop()

        st.subheader("📈 Stock Data")
        st.dataframe(stock_data.head(10))

        # -------------------------------
        # Merge Stock + Sentiment (single safe merge)
        # -------------------------------
        # combine_data is defensive; supply daily_sentiment and stock_data
        combined_df = combine_data(daily_sentiment, stock_data)
        if combined_df is None or combined_df.empty:
            st.error("❌ Combined DataFrame is empty — check overlap.")
            st.stop()

        # Normalize datetimes for display/indexing
        for col in ["Date", "DateOnly"]:
            if col in combined_df.columns:
                combined_df[col] = pd.to_datetime(combined_df[col], errors="coerce").dt.floor("D")

        st.subheader("🔍 Combined Data Sample")
        st.dataframe(combined_df.head(15))

        # -------------------------------
        # Sanity & Ranges
        # -------------------------------
        if "Pct_Change" not in combined_df.columns and "Close" in combined_df.columns:
            combined_df["Pct_Change"] = combined_df["Close"].pct_change() * 100.0

        # Avoid KeyError
        if "sentiment_score" in combined_df.columns:
            st.write(f"sentiment_score range: {combined_df['sentiment_score'].min():.3f} → {combined_df['sentiment_score'].max():.3f}")
        else:
            st.write("sentiment_score: not present in combined_df")

        if "Pct_Change" in combined_df.columns:
            st.write(f"Pct_Change range: {combined_df['Pct_Change'].min():.3f} → {combined_df['Pct_Change'].max():.3f}")

        # -------------------------------
        # Correlation Check (safe)
        # -------------------------------
        correlation = calc_correlation(stock_data, daily_sentiment)
        if correlation is None:
            st.warning("⚠️ Correlation could not be calculated.")
            st.write("📈 Correlation (lagged sentiment vs. % change): n/a")
        else:
            st.write(f"📈 Correlation (lagged sentiment vs. % change): {correlation:.4f}")

        # -------------------------------
        # Forecasting with Sentiment Integration
        # -------------------------------
        st.subheader("🔮 Forecasting Results")
        results = fit_and_forecast(combined_df, steps=7)

        if results:
            # show recommendation & certainty
            st.write(f"**Recommendation:** {results.get('recommendation','HOLD')} (certainty {results.get('certainty', 0.0):.2f})")

            # project 1-day and 7-day if produced by forecasting; otherwise derive from daily_forecast if available
            proj_1d = results.get("forecast_1d")
            proj_7d = results.get("forecast_7d")
            if proj_1d is None and "daily_forecast" in results and not results["daily_forecast"].empty:
                proj_1d = float(results["daily_forecast"]["Forecast"].iloc[0]) if "Forecast" in results["daily_forecast"].columns else "N/A"
            if proj_7d is None and "weekly_forecast" in results and not results["weekly_forecast"].empty:
                # weekly forecast is aggregated; take sum of first week or first row
                try:
                    proj_7d = float(results["weekly_forecast"]["Forecast"].iloc[0]) if "Forecast" in results["weekly_forecast"].columns else "N/A"
                except Exception:
                    proj_7d = "N/A"

            st.write(f"**Projected 1-day change:** {proj_1d}")
            st.write(f"**Projected 7-day change:** {proj_7d}")

            st.write("### Daily Forecast with Confidence Intervals")
            st.dataframe(results["daily_forecast"])

            st.write("### Weekly Forecast Summary")
            st.dataframe(results["weekly_forecast"])

            # Visualizations
            create_plot(combined_df, forecast_df=results["daily_forecast"])
            plot_price_with_forecast(ticker, hist_days=30, forecast_df=results["daily_forecast"])
        else:
            st.warning("⚠️ Forecasting failed or returned no result.")

        # -------------------------------
        # Fundamental & Valuation Analysis
        # -------------------------------
        from data.fundmentals import get_fundamentals, get_comparables, compute_valuation_gap, run_dcf_valuation

        st.header("🏦 Fundamental & Valuation Analysis")

        # Fetch core fundamentals
        fundamentals_df = get_fundamentals(ticker)
        if not fundamentals_df.empty:
            st.subheader("📊 Key Financial Metrics")
            st.dataframe(
                fundamentals_df[
                    [
                        "MarketCap", "Revenue", "EBITDA", "FreeCashFlow",
                        "ForwardPE", "EV/EBITDA", "P/S", "PriceToBook", "DividendYield"
                    ]
                ].T.rename(columns={0: ticker})
            )

        # Fetch comparable companies and valuation gaps
        comparables_df = get_comparables(ticker)
        if not comparables_df.empty:
            st.subheader("🤝 Comparable Company Analysis")
            st.dataframe(
                comparables_df[
                    ["Ticker", "MarketCap", "ForwardPE", "EV/EBITDA", "P/S", "PriceToBook"]
                ]
            )

            vgap_df = compute_valuation_gap(ticker, comparables_df)
            if not vgap_df.empty:
                st.subheader("📉 Valuation Gap vs Peers")
                st.dataframe(vgap_df.style.background_gradient(cmap="RdYlGn", subset=["V-Gap (%)"]))

        # Run DCF intrinsic value estimator
        st.subheader("💰 DCF Intrinsic Valuation")
        dcf_detail, dcf_summary = run_dcf_valuation(ticker)

        if not dcf_summary.empty:
            st.write("### DCF Summary")
            st.table(dcf_summary.style.format({"Intrinsic Value": "{:,.2f}", "Market Price": "{:,.2f}", "Upside (%)": "{:+.2f}%"}))

        if not dcf_detail.empty:
            st.write("### Discounted Cash Flow Projection")
            st.dataframe(dcf_detail.style.format({"FCF (Projected)": "{:,.0f}", "PV of FCF": "{:,.0f}"}))

        # -------------------------------
        # Integrated Hedge Fund Analyst Summary
        # -------------------------------
        st.header("🧾 Integrated Analyst Summary")

        try:
            # Extract key metrics from each module
            forecast_reco = results.get("recommendation", "HOLD")
            forecast_certainty = results.get("certainty", 0.0)
            dcf_val = dcf_summary.get("Intrinsic Value", [None])[0] if not dcf_summary.empty else None
            market_price = dcf_summary.get("Market Price", [None])[0] if not dcf_summary.empty else None
            upside = dcf_summary.get("Upside (%)", [None])[0] if not dcf_summary.empty else None
            corr_val = correlation if correlation is not None else 0.0

            # Construct qualitative view
            sentiment_bias = "Bullish" if forecast_reco == "BUY" else "Bearish" if forecast_reco == "SELL" else "Neutral"
            value_bias = "Undervalued" if upside and upside > 10 else "Overvalued" if upside and upside < -10 else "Fair Value"

            # Composite analyst score (simple weighted mix)
            # Example weights: 0.4 forecasting certainty + 0.4 DCF upside + 0.2 sentiment correlation
            comp_score = (0.4 * forecast_certainty*100) + (0.4 * (upside or 0)) + (0.2 * corr_val*100)

            if comp_score > 20:
                overall = "Strong Buy"
            elif comp_score > 5:
                overall = "Buy"
            elif comp_score > -5:
                overall = "Hold"
            elif comp_score > -20:
                overall = "Sell"
            else:
                overall = "Strong Sell"

            st.subheader("🧮 Quant + Value Alignment")
            summary_df = pd.DataFrame([{
                "Sentiment Bias": sentiment_bias,
                "Forecast Certainty": f"{forecast_certainty:.2f}",
                "DCF Value Bias": value_bias,
                "DCF Upside (%)": f"{upside:+.2f}%" if upside is not None else "n/a",
                "Price Sentiment Corr.": f"{corr_val:.2f}",
                "Composite Analyst Score": f"{comp_score:.2f}",
                "Final Recommendation": overall,
            }])

            st.table(summary_df)

            # Optional radar or bar chart
            st.bar_chart(pd.DataFrame({
                "Metric": ["Forecast Certainty", "DCF Upside", "Sentiment Correlation"],
                "Score": [forecast_certainty*100, upside or 0, corr_val*100]
            }).set_index("Metric"))

            st.success(f"📊 Integrated Hedge Fund View: **{overall.upper()}**")

        except Exception as e:
            st.error(f"⚠️ Could not generate combined summary: {e}")


    except Exception as e:
        # Print a trace to the terminal to help you debug where computed_score might be missing
        print("❌ Error during analysis:", e)
        st.error(f"❌ Error during analysis: {e}")
