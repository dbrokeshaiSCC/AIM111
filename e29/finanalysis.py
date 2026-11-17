"""AIM111 Streamlit Financial Analysis Lab.

Run: streamlit run finanalysis.py
Based on the instructor's financial analysis lab PowerPoint (slides 5-16).
"""

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_data(tickers, start_date, end_date):
    """Fetch daily close prices for tickers from Yahoo Finance."""
    data_frames = []
    available_tickers = []
    failed_tickers = []

    for ticker in tickers:
        try:
            stock_data = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                auto_adjust=True,
                progress=False,
                threads=False,
            )
            if stock_data.empty:
                failed_tickers.append(ticker)
                continue

            # yfinance may return MultiIndex columns even for a single ticker.
            close = stock_data["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            close = pd.to_numeric(close, errors="coerce").rename(ticker)
            if close.dropna().empty:
                failed_tickers.append(ticker)
                continue

            data_frames.append(close)
            available_tickers.append(ticker)
        except Exception:
            failed_tickers.append(ticker)

    if not data_frames:
        return pd.DataFrame(), failed_tickers

    data = pd.concat(data_frames, axis=1)
    data.columns = available_tickers
    data = data.sort_index().dropna(how="all")
    return data, failed_tickers


def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    """Annualized equal-weighted portfolio Sharpe ratio (252 trading days)."""
    portfolio_daily_returns = returns.mean(axis=1)
    volatility = portfolio_daily_returns.std() * np.sqrt(252)
    if pd.isna(volatility) or volatility <= 0:
        return np.nan
    annualized_return = portfolio_daily_returns.mean() * 252
    return (annualized_return - risk_free_rate) / volatility


def calculate_sortino_ratio(returns, risk_free_rate=0.02):
    """Annualized Sortino ratio against a 2% annual target return."""
    portfolio_daily_returns = returns.mean(axis=1)
    daily_target = risk_free_rate / 252
    downside = np.minimum(portfolio_daily_returns - daily_target, 0)
    downside_deviation = np.sqrt(np.mean(np.square(downside))) * np.sqrt(252)
    if pd.isna(downside_deviation) or downside_deviation <= 0:
        return np.nan
    annualized_return = portfolio_daily_returns.mean() * 252
    return (annualized_return - risk_free_rate) / downside_deviation


def portfolio_optimization(returns):
    """Lab's simple equal-weight allocation (not numerical optimization)."""
    return np.ones(len(returns.columns)) / len(returns.columns)


def main():
    st.title("Financial Analysis Web App")
    st.write("Explore historical stock prices, returns, and portfolio performance.")

    # User inputs from the PowerPoint lab.
    ticker_input = st.text_input(
        "Enter stock tickers (comma separated)", "AAPL, MSFT, GOOG"
    )
    tickers = list(dict.fromkeys(
        ticker.strip().upper() for ticker in ticker_input.split(",")
        if ticker.strip()
    ))
    start_date = st.date_input("Start Date", pd.to_datetime("2020-01-01"))
    end_date = st.date_input("End Date", pd.to_datetime("2023-01-01"))

    if not tickers:
        st.warning("Enter at least one stock ticker.")
        st.stop()
    if start_date >= end_date:
        st.error("End Date must be after Start Date.")
        st.stop()

    # Fetch stock data.
    with st.spinner("Downloading historical stock data..."):
        stock_data, failed_tickers = fetch_data(tickers, start_date, end_date)

    if failed_tickers:
        st.warning("No price data for: " + ", ".join(failed_tickers))
    if stock_data.empty:
        st.error("No stock data was returned. Check tickers, dates, and internet access.")
        st.stop()

    # Calculate returns only for dates with prices for all selected stocks.
    aligned_prices = stock_data.dropna(how="any")
    returns_data = aligned_prices.pct_change(fill_method=None).dropna()

    # Visualizations.
    st.subheader("Stock Prices")
    st.line_chart(stock_data)
    st.subheader("Stock Returns")
    if returns_data.empty:
        st.warning("Insufficient overlapping daily stock prices to calculate returns.")
    else:
        st.line_chart(returns_data)

    # Performance metrics.
    st.subheader("Performance Metrics")
    if len(returns_data) < 2:
        st.info("At least two daily return observations are needed for risk metrics.")
    else:
        sharpe_ratio = calculate_sharpe_ratio(returns_data)
        sortino_ratio = calculate_sortino_ratio(returns_data)
        st.write(
            f"Sharpe Ratio: {sharpe_ratio:.2f}"
            if np.isfinite(sharpe_ratio) else "Sharpe Ratio: N/A"
        )
        st.write(
            f"Sortino Ratio: {sortino_ratio:.2f}"
            if np.isfinite(sortino_ratio) else "Sortino Ratio: N/A"
        )
        st.caption(
            "Ratios are annualized using 252 trading days and a 2% annual "
            "risk-free/target rate. Portfolio returns are equally weighted."
        )

    # Portfolio allocation.
    st.subheader("Portfolio Optimization")
    st.write("Equal-Weighted Portfolio Weights:")
    optimal_weights = portfolio_optimization(stock_data)
    for ticker, weight in zip(stock_data.columns, optimal_weights):
        st.write(f"{ticker}: {weight:.2%}")
    st.caption("The lab uses equal weights; it does not solve an optimization problem.")

    # Download option.
    csv_data = stock_data.to_csv()
    st.download_button(
        "Download Stock Data CSV",
        data=csv_data,
        file_name="stock_data.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
