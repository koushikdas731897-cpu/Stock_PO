import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from scipy.optimize import minimize
from datetime import date, timedelta

# ------------------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------------------
st.set_page_config(
    page_title="NIFTY 50 | MVP vs HRP Portfolio Optimisation",
    page_icon="📊",
    layout="wide"
)

DEFAULT_STOCKS = {
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "Reliance Industries": "RELIANCE.NS",
    "Bharti Airtel": "BHARTIARTL.NS",
    "Larsen & Toubro": "LT.NS",
    "State Bank of India": "SBIN.NS",
    "Axis Bank": "AXISBANK.NS",
    "Infosys": "INFY.NS",
    "Kotak Mahindra Bank": "KOTAKBANK.NS",
    "Mahindra & Mahindra": "M&M.NS",
}
TRADING_DAYS = 252

# ------------------------------------------------------------
# DATA DOWNLOAD: RAW CLOSE PRICES ONLY
# ------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def download_close_prices(tickers, start_date, end_date):
    data = yf.download(
        tickers=list(tickers),
        start=start_date,
        end=end_date,
        interval="1d",
        auto_adjust=False,
        actions=False,
        group_by="column",
        progress=False,
        threads=True
    )
    if data is None or data.empty:
        raise ValueError("Yahoo Finance returned no data for the selected dates. Check the dates or try again later.")

    if isinstance(data.columns, pd.MultiIndex):
        # yfinance commonly returns (Price, Ticker). Handle either level order.
        if "Close" in data.columns.get_level_values(0):
            close = data.xs("Close", axis=1, level=0).copy()
        elif "Close" in data.columns.get_level_values(1):
            close = data.xs("Close", axis=1, level=1).copy()
        else:
            raise ValueError("The downloaded data does not contain a Close column.")
    else:
        if "Close" not in data.columns:
            raise ValueError("The downloaded data does not contain a Close column.")
        close = data[["Close"]].copy()
        if len(tickers) == 1:
            close.columns = list(tickers)

    if isinstance(close, pd.Series):
        close = close.to_frame(name=list(tickers)[0])

    available = [ticker for ticker in tickers if ticker in close.columns]
    close = close.loc[:, available].copy()
    close = close.apply(pd.to_numeric, errors="coerce")
    close = close[~close.index.duplicated(keep="last")].sort_index()
    close.index = pd.to_datetime(close.index).tz_localize(None)
    return close.dropna(how="all")

# ------------------------------------------------------------
# PORTFOLIO CONSTRUCTION
# ------------------------------------------------------------
def calculate_mvp(train_returns):
    cov = train_returns.cov() * TRADING_DAYS
    n_assets = len(train_returns.columns)
    initial = np.repeat(1.0 / n_assets, n_assets)

    def objective(weights):
        return float(weights.T @ cov.to_numpy(copy=True) @ weights)

    constraints = [{"type": "eq", "fun": lambda weights: np.sum(weights) - 1.0}]
    bounds = tuple((0.0, 1.0) for _ in range(n_assets))
    result = minimize(
        objective, initial, method="SLSQP", bounds=bounds,
        constraints=constraints, options={"maxiter": 2000, "ftol": 1e-12}
    )
    if not result.success:
        raise RuntimeError(f"MVP optimisation failed: {result.message}")
    weights = pd.Series(result.x, index=train_returns.columns, name="MVP")
    weights[weights.abs() < 1e-8] = 0.0
    return weights / weights.sum()


def calculate_hrp(train_returns):
    covariance = train_returns.cov()
    correlation = train_returns.corr().clip(-1, 1)
    distance = np.sqrt(((1.0 - correlation) / 2.0).clip(lower=0.0))
    distance_values = distance.to_numpy(copy=True)
    np.fill_diagonal(distance_values, 0.0)
    condensed = squareform(distance_values, checks=False)
    tree = linkage(condensed, method="single", optimal_ordering=True)
    order = leaves_list(tree).astype(int).tolist()
    ordered_assets = train_returns.columns[order].tolist()

    weights = pd.Series(1.0, index=ordered_assets)

    def cluster_variance(items):
        block = covariance.loc[items, items].to_numpy(copy=True)
        diagonal = np.diag(block)
        diagonal = np.where(diagonal <= 0, 1e-12, diagonal)
        inv_var = 1.0 / diagonal
        cluster_weights = inv_var / inv_var.sum()
        return float(cluster_weights @ block @ cluster_weights)

    clusters = [ordered_assets]
    while clusters:
        next_clusters = []
        for cluster in clusters:
            if len(cluster) <= 1:
                continue
            split = len(cluster) // 2
            left, right = cluster[:split], cluster[split:]
            left_var, right_var = cluster_variance(left), cluster_variance(right)
            total_var = left_var + right_var
            left_weight = 0.5 if total_var <= 0 else 1.0 - left_var / total_var
            weights[left] *= left_weight
            weights[right] *= (1.0 - left_weight)
            next_clusters.extend([left, right])
        clusters = next_clusters

    weights = weights.reindex(train_returns.columns).fillna(0.0)
    weights[weights < 1e-10] = 0.0
    if not np.isfinite(weights.sum()) or weights.sum() <= 0:
        raise ValueError("HRP could not produce valid portfolio weights.")
    return weights / weights.sum(), tree, correlation

# ------------------------------------------------------------
# BACKTEST METRICS
# ------------------------------------------------------------
def evaluate_strategy(test_returns, weights, risk_free_rate):
    daily = (test_returns @ weights).dropna()
    if len(daily) < 2:
        raise ValueError("Not enough test-period observations to calculate performance.")
    wealth = (1.0 + daily).cumprod()
    total_return = float(wealth.iloc[-1] - 1.0)
    annual_return = float(wealth.iloc[-1] ** (TRADING_DAYS / len(daily)) - 1.0)
    annual_risk = float(daily.std(ddof=1) * np.sqrt(TRADING_DAYS))
    annual_mean = float(daily.mean() * TRADING_DAYS)
    sharpe = (annual_mean - risk_free_rate) / annual_risk if annual_risk > 0 else np.nan
    drawdown = wealth / wealth.cummax() - 1.0
    metrics = {
        "Total Return": total_return,
        "Annualised Return": annual_return,
        "Annualised Risk": annual_risk,
        "Sharpe Ratio": float(sharpe) if np.isfinite(sharpe) else np.nan,
        "Maximum Drawdown": float(drawdown.min()),
    }
    return metrics, daily, wealth, drawdown


def as_percent(value):
    return f"{value:.2%}" if pd.notna(value) else "N/A"

# ------------------------------------------------------------
# SIDEBAR: EDITABLE DATE RANGES
# ------------------------------------------------------------
st.sidebar.title("Model settings")
st.sidebar.subheader("Training period")
training_start = st.sidebar.date_input("Training start", value=date(2025, 1, 1), key="training_start")
training_end = st.sidebar.date_input("Training end", value=date(2026, 4, 30), key="training_end")
st.sidebar.markdown("---")
st.sidebar.subheader("Testing period")
testing_start = st.sidebar.date_input("Testing start", value=date(2026, 5, 1), key="testing_start")
testing_end = st.sidebar.date_input("Testing end", value=date(2026, 10, 31), key="testing_end")
st.sidebar.markdown("---")
risk_free_pct = st.sidebar.number_input(
    "Annual risk-free rate (%)", min_value=0.0, max_value=20.0,
    value=6.0, step=0.25,
    help="Assumption used to calculate the annualised Sharpe ratio."
)
risk_free_rate = risk_free_pct / 100.0
st.sidebar.caption("Prices are downloaded from Yahoo Finance. The analysis uses only the daily Close field.")
st.sidebar.caption("Default universe: 10 large NIFTY 50 constituents. Tickers can be changed in DEFAULT_STOCKS in the source code.")

# ------------------------------------------------------------
# MAIN PAGE
# ------------------------------------------------------------
st.title("NIFTY 50 Portfolio Optimisation")
st.subheader("Minimum Variance Portfolio (MVP) vs Hierarchical Risk Parity (HRP)")
st.caption("Daily Close-price analysis | Build portfolios on the training period | Evaluate on the separate testing period")

st.markdown("### Portfolio optimisation and backtesting")
run_clicked = st.button("Run Portfolio Optimization", type="primary", use_container_width=False)

if run_clicked:
    if training_start >= training_end:
        st.error("Training start date must be earlier than training end date.")
        st.stop()
    if testing_start >= testing_end:
        st.error("Testing start date must be earlier than testing end date.")
        st.stop()
    if testing_start <= training_end:
        st.error("The testing period must start after the training period ends. Keep the periods separate to avoid data leakage.")
        st.stop()

    tickers = list(DEFAULT_STOCKS.values())
    download_start = training_start.isoformat()
    # yfinance end is exclusive, so add one day to include the selected test-end date.
    download_end = (testing_end + timedelta(days=1)).isoformat()

    try:
        with st.spinner("Downloading daily Close prices and building the portfolios..."):
            close_prices = download_close_prices(tickers, download_start, download_end)

        missing = [ticker for ticker in tickers if ticker not in close_prices.columns]
        if missing:
            st.warning("Not returned by Yahoo Finance and excluded: " + ", ".join(missing))

        # Keep a common complete price history so both methods use the same assets.
        usable = close_prices.dropna(axis=1, how="any").copy()
        excluded = sorted(set(close_prices.columns) - set(usable.columns))
        if excluded:
            st.warning("Excluded because of missing prices in the selected date range: " + ", ".join(excluded))
        if usable.shape[1] < 2:
            raise ValueError("Fewer than two complete stock price series are available for the selected dates.")

        daily_returns = usable.pct_change(fill_method=None).dropna(how="all")
        train_returns = daily_returns.loc[training_start:training_end].dropna()
        test_returns = daily_returns.loc[testing_start:testing_end].dropna()
        common = [col for col in train_returns.columns if col in test_returns.columns]
        train_returns = train_returns[common].dropna()
        test_returns = test_returns[common].dropna()
        if len(common) < 2:
            raise ValueError("Fewer than two stocks have valid data in both periods.")
        if train_returns.empty or test_returns.empty:
            raise ValueError("Training or testing returns are empty. Check the selected dates and data availability.")

        mvp_weights = calculate_mvp(train_returns)
        hrp_weights, linkage_matrix, correlation = calculate_hrp(train_returns)
        mvp_metrics, mvp_daily, mvp_wealth, mvp_drawdown = evaluate_strategy(test_returns, mvp_weights, risk_free_rate)
        hrp_metrics, hrp_daily, hrp_wealth, hrp_drawdown = evaluate_strategy(test_returns, hrp_weights, risk_free_rate)

        company_by_ticker = {ticker: company for company, ticker in DEFAULT_STOCKS.items()}
        allocation_df = pd.DataFrame({
            "Company": [company_by_ticker[t] for t in common],
            "Ticker": common,
            "MVP Weight": [float(mvp_weights[t]) for t in common],
            "HRP Weight": [float(hrp_weights[t]) for t in common],
        })
        allocation_df["MVP Allocation (%)"] = allocation_df["MVP Weight"] * 100
        allocation_df["HRP Allocation (%)"] = allocation_df["HRP Weight"] * 100

        metrics_df = pd.DataFrame({"MVP": mvp_metrics, "HRP": hrp_metrics}).T
        backtest_df = metrics_df.copy()
        backtest_df.insert(0, "Strategy", backtest_df.index)
        backtest_df.index = range(len(backtest_df))
        daily_df = pd.DataFrame({"MVP": mvp_daily, "HRP": hrp_daily})
        wealth_df = pd.DataFrame({"MVP": mvp_wealth, "HRP": hrp_wealth})
        drawdown_df = pd.DataFrame({"MVP": mvp_drawdown, "HRP": hrp_drawdown})

        st.session_state["portfolio_results"] = {
            "close_prices": usable,
            "allocation_df": allocation_df,
            "metrics_df": metrics_df,
            "backtest_df": backtest_df,
            "daily_df": daily_df,
            "wealth_df": wealth_df,
            "drawdown_df": drawdown_df,
            "correlation": correlation,
            "stocks": len(common),
            "train_obs": len(train_returns),
            "test_obs": len(test_returns),
            "training_start": training_start,
            "training_end": training_end,
            "testing_start": testing_start,
            "testing_end": testing_end,
        }
        st.success("Optimisation and out-of-sample backtest completed successfully.")
    except Exception as exc:
        st.error(f"Unable to complete the analysis: {exc}")
        st.exception(exc)

# ------------------------------------------------------------
# RESULTS: DISPLAY FROM SESSION STATE (PERSISTS ON DOWNLOAD)
# ------------------------------------------------------------
results = st.session_state.get("portfolio_results")
if results:
    st.markdown("## Results summary")
    a, b, c, d = st.columns(4)
    a.metric("Stocks analysed", results["stocks"])
    b.metric("Training observations", results["train_obs"])
    c.metric("Testing observations", results["test_obs"])
    d.metric("Testing period", f"{results['testing_start']:%d %b %Y} – {results['testing_end']:%d %b %Y}")

    st.markdown("### Portfolio performance comparison")
    metric_view = results["metrics_df"].copy()
    for col in metric_view.columns:
        if col == "Sharpe Ratio":
            metric_view[col] = metric_view[col].map(lambda x: f"{x:.3f}" if pd.notna(x) else "N/A")
        else:
            metric_view[col] = metric_view[col].map(as_percent)
    st.dataframe(metric_view, use_container_width=True)

    sharpe_values = results["metrics_df"]["Sharpe Ratio"]
    if sharpe_values.notna().any():
        best = sharpe_values.idxmax()
        st.info(f"Higher test-period Sharpe ratio: **{best}** ({sharpe_values[best]:.3f}). A higher value indicates more historical excess return per unit of realised risk during this test period.")

    st.markdown("### Portfolio allocation")
    left, right = st.columns(2)
    with left:
        fig_mvp = px.pie(results["allocation_df"], names="Company", values="MVP Weight", hole=0.30, title="MVP portfolio weights")
        fig_mvp.update_traces(textposition="inside", textinfo="label+percent")
        st.plotly_chart(fig_mvp, use_container_width=True)
    with right:
        fig_hrp = px.pie(results["allocation_df"], names="Company", values="HRP Weight", hole=0.30, title="HRP portfolio weights")
        fig_hrp.update_traces(textposition="inside", textinfo="label+percent")
        st.plotly_chart(fig_hrp, use_container_width=True)

    allocation_view = results["allocation_df"][["Company", "Ticker", "MVP Allocation (%)", "HRP Allocation (%)"]].copy()
    allocation_view["MVP Allocation (%)"] = allocation_view["MVP Allocation (%)"].map(lambda x: f"{x:.2f}%")
    allocation_view["HRP Allocation (%)"] = allocation_view["HRP Allocation (%)"].map(lambda x: f"{x:.2f}%")
    st.dataframe(allocation_view, use_container_width=True)

    st.markdown("### Portfolio growth during the testing period")
    growth_fig = go.Figure()
    growth_fig.add_trace(go.Scatter(x=results["wealth_df"].index, y=results["wealth_df"]["MVP"], mode="lines", name="MVP"))
    growth_fig.add_trace(go.Scatter(x=results["wealth_df"].index, y=results["wealth_df"]["HRP"], mode="lines", name="HRP"))
    growth_fig.update_layout(xaxis_title="Date", yaxis_title="Growth of ₹1", hovermode="x unified")
    st.plotly_chart(growth_fig, use_container_width=True)

    st.markdown("### Portfolio drawdown")
    dd_fig = go.Figure()
    dd_fig.add_trace(go.Scatter(x=results["drawdown_df"].index, y=results["drawdown_df"]["MVP"], mode="lines", name="MVP", fill="tozeroy"))
    dd_fig.add_trace(go.Scatter(x=results["drawdown_df"].index, y=results["drawdown_df"]["HRP"], mode="lines", name="HRP", fill="tozeroy"))
    dd_fig.update_layout(xaxis_title="Date", yaxis_title="Drawdown", yaxis_tickformat=".1%", hovermode="x unified")
    st.plotly_chart(dd_fig, use_container_width=True)

    st.markdown("### Training-period correlation matrix")
    corr_fig = px.imshow(results["correlation"], text_auto=".2f", aspect="auto", color_continuous_scale="RdBu", zmin=-1, zmax=1, title="Correlation of daily Close-to-Close returns")
    st.plotly_chart(corr_fig, use_container_width=True)

    st.markdown("### Download Results")
    weights_csv = results["allocation_df"].to_csv(index=False).encode("utf-8")
    backtest_csv = results["backtest_df"].to_csv(index=False).encode("utf-8")
    daily_csv = results["daily_df"].to_csv(index=True).encode("utf-8")
    prices_csv = results["close_prices"].to_csv(index=True).encode("utf-8")
    d1, d2 = st.columns(2)
    with d1:
        st.download_button("Download Portfolio Weights", data=weights_csv, file_name="nifty50_mvp_hrp_portfolio_weights.csv", mime="text/csv", use_container_width=True)
    with d2:
        st.download_button("Download Backtesting Results", data=backtest_csv, file_name="nifty50_mvp_hrp_backtesting_results.csv", mime="text/csv", use_container_width=True)
    d3, d4 = st.columns(2)
    with d3:
        st.download_button("Download Daily Portfolio Returns", data=daily_csv, file_name="nifty50_mvp_hrp_daily_returns.csv", mime="text/csv", use_container_width=True)
    with d4:
        st.download_button("Download Close Prices", data=prices_csv, file_name="nifty50_top10_close_prices.csv", mime="text/csv", use_container_width=True)

    with st.expander("Methodology and assumptions"):
        st.markdown("""
        **Minimum Variance Portfolio (MVP):** Long-only optimisation that minimises training-period portfolio variance. Weights are constrained between 0% and 100% and sum to 100%.

        **Hierarchical Risk Parity (HRP):** Converts the training-period correlation matrix into a distance matrix, applies hierarchical clustering and recursively allocates weights based on cluster variance.

        **Backtesting:** Portfolio weights are estimated using training-period returns and held fixed during the separate testing period. Daily returns are calculated from raw Close prices only. Dividends are excluded because Adjusted Close is not used.

        **Risk and return:** Annualised volatility uses daily standard deviation × √252. Sharpe ratio uses the annual risk-free-rate assumption entered in the sidebar. No transaction costs, taxes or slippage are included. Historical performance does not guarantee future results.
        """)
else:
    st.markdown("### Download Results")
    st.caption("Run the portfolio optimisation to generate the downloadable portfolio weights and backtesting results.")
    st.markdown("The dashboard will show the performance comparison, portfolio allocations, charts and downloadable CSV files after the analysis completes.")
