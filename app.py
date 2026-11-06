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

# ------------------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------------------
st.set_page_config(
    page_title="NIFTY 50 | MVP vs HRP Optimisation",
    page_icon="📊",
    layout="wide"
)

st.title("NIFTY 50 Portfolio Optimisation")
st.subheader("Minimum Variance Portfolio (MVP) vs Hierarchical Risk Parity (HRP)")
st.caption(
    "Price-only analysis using daily Close prices | "
    "Training: Jan 2025–Apr 2026 | Out-of-sample test: May–Oct 2026"
)

# Top 10 constituents by NIFTY 50 index weight, reference date: 30-Sep-2026.
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

START_DATE = "2025-01-01"
END_DATE = "2026-11-01"  # yfinance end date is exclusive
TRAIN_END = "2026-04-30"
TEST_START = "2026-05-01"
TEST_END = "2026-10-31"
TRADING_DAYS = 252


# ------------------------------------------------------------
# DATA DOWNLOAD: USE ONLY THE RAW "CLOSE" FIELD
# ------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def download_close_prices(tickers, start_date, end_date):
    data = yf.download(
        tickers=list(tickers),
        start=start_date,
        end=end_date,
        interval="1d",
        auto_adjust=False,   # Keep the Close field separate from Adj Close
        actions=False,
        group_by="column",
        progress=False,
        threads=True
    )

    if data is None or data.empty:
        raise ValueError("Yahoo Finance returned no data. Try again later.")

    # yfinance may return MultiIndex columns, including for a single ticker.
    if isinstance(data.columns, pd.MultiIndex):
        if "Close" not in data.columns.get_level_values(0):
            raise ValueError("The downloaded data does not contain a Close column.")
        close = data["Close"].copy()
    else:
        if "Close" not in data.columns:
            raise ValueError("The downloaded data does not contain a Close column.")
        close = data[["Close"]].copy()
        if len(tickers) == 1:
            close.columns = list(tickers)

    if isinstance(close, pd.Series):
        close = close.to_frame(name=list(tickers)[0])

    # Match the requested ticker order and retain only available symbols.
    available = [ticker for ticker in tickers if ticker in close.columns]
    close = close[available].copy()
    close = close.apply(pd.to_numeric, errors="coerce")
    close = close[~close.index.duplicated(keep="last")].sort_index()
    close.index = pd.to_datetime(close.index).tz_localize(None)

    # Do not forward-fill prices: that would manufacture zero-return days.
    close = close.dropna(how="all")
    return close


# ------------------------------------------------------------
# PORTFOLIO OPTIMISATION
# ------------------------------------------------------------
def portfolio_stats(weights, mean_returns, cov_matrix, risk_free_rate):
    annual_return = float(np.dot(weights, mean_returns) * TRADING_DAYS)
    annual_volatility = float(
        np.sqrt(weights.T @ cov_matrix.values @ weights) * np.sqrt(TRADING_DAYS)
    )
    sharpe = (
        (annual_return - risk_free_rate) / annual_volatility
        if annual_volatility > 0 else np.nan
    )
    return annual_return, annual_volatility, sharpe


def calculate_mvp(train_returns):
    """Long-only global minimum-variance portfolio; weights sum to one."""
    cov_matrix = train_returns.cov() * TRADING_DAYS
    n_assets = len(train_returns.columns)
    initial_weights = np.repeat(1 / n_assets, n_assets)

    def objective(weights):
        return float(weights.T @ cov_matrix.values @ weights)

    constraints = [{"type": "eq", "fun": lambda weights: np.sum(weights) - 1}]
    bounds = tuple((0.0, 1.0) for _ in range(n_assets))

    result = minimize(
        objective,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 2000, "ftol": 1e-12}
    )

    if not result.success:
        raise RuntimeError(f"MVP optimisation failed: {result.message}")

    weights = pd.Series(result.x, index=train_returns.columns, name="MVP")
    weights[weights.abs() < 1e-8] = 0
    return weights / weights.sum()


def get_quasi_diag(link):
    """Return the hierarchical clustering leaf order."""
    return leaves_list(link).astype(int).tolist()


def get_cluster_variance(covariance, cluster_items):
    """Inverse-variance portfolio variance for a cluster."""
    cov_slice = covariance.loc[cluster_items, cluster_items]
    diagonal = np.diag(cov_slice.values)
    diagonal = np.where(diagonal <= 0, 1e-12, diagonal)
    inv_variance = 1 / diagonal
    weights = inv_variance / inv_variance.sum()
    return float(weights @ cov_slice.values @ weights)


def calculate_hrp(train_returns):
    """
    Hierarchical Risk Parity:
    1. Correlation-based distance matrix
    2. Hierarchical clustering
    3. Quasi-diagonal ordering
    4. Recursive bisection using cluster variance
    """
    covariance = train_returns.cov()
    correlation = train_returns.corr().clip(-1, 1)

    distance = np.sqrt((1 - correlation) / 2)
    np.fill_diagonal(distance.values, 0)

    condensed_distance = squareform(distance.values, checks=False)
    linkage_matrix = linkage(
        condensed_distance,
        method="single",
        optimal_ordering=True
    )

    ordered_indices = get_quasi_diag(linkage_matrix)
    ordered_assets = train_returns.columns[ordered_indices].tolist()

    weights = pd.Series(1.0, index=ordered_assets)

    clusters = [ordered_assets]
    while clusters:
        next_clusters = []
        for cluster in clusters:
            if len(cluster) <= 1:
                continue

            split = len(cluster) // 2
            left = cluster[:split]
            right = cluster[split:]

            left_variance = get_cluster_variance(covariance, left)
            right_variance = get_cluster_variance(covariance, right)

            # Allocate more weight to the lower-variance cluster.
            left_weight = 1 - left_variance / (left_variance + right_variance)

            weights[left] *= left_weight
            weights[right] *= (1 - left_weight)

            next_clusters.extend([left, right])

        clusters = next_clusters

    weights = weights.reindex(train_returns.columns).fillna(0)
    weights[weights < 1e-10] = 0
    return weights / weights.sum(), linkage_matrix, correlation


# ------------------------------------------------------------
# OUT-OF-SAMPLE PERFORMANCE
# ------------------------------------------------------------
def evaluate_strategy(test_returns, weights, risk_free_rate):
    daily_portfolio_returns = test_returns @ weights
    daily_portfolio_returns = daily_portfolio_returns.dropna()

    if len(daily_portfolio_returns) < 2:
        raise ValueError("Not enough test-period returns to calculate performance.")

    wealth = (1 + daily_portfolio_returns).cumprod()
    total_return = float(wealth.iloc[-1] - 1)
    annualised_return = float(
        wealth.iloc[-1] ** (TRADING_DAYS / len(daily_portfolio_returns)) - 1
    )
    annualised_volatility = float(
        daily_portfolio_returns.std(ddof=1) * np.sqrt(TRADING_DAYS)
    )
    annualised_mean_return = float(daily_portfolio_returns.mean() * TRADING_DAYS)
    sharpe = (
        (annualised_mean_return - risk_free_rate) / annualised_volatility
        if annualised_volatility > 0 else np.nan
    )
    drawdown = wealth / wealth.cummax() - 1
    max_drawdown = float(drawdown.min())

    metrics = {
        "Total Return": total_return,
        "Annualised Return": annualised_return,
        "Annualised Volatility (Risk)": annualised_volatility,
        "Sharpe Ratio": sharpe,
        "Maximum Drawdown": max_drawdown,
    }

    return metrics, daily_portfolio_returns, wealth, drawdown


def format_percent_columns(frame):
    formatted = frame.copy()
    for col in formatted.columns:
        if col != "Sharpe Ratio":
            formatted[col] = formatted[col].map(
                lambda x: f"{x:.2%}" if pd.notna(x) else "N/A"
            )
        else:
            formatted[col] = formatted[col].map(
                lambda x: f"{x:.3f}" if pd.notna(x) else "N/A"
            )
    return formatted


# ------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------
st.sidebar.header("Model settings")
risk_free_pct = st.sidebar.number_input(
    "Annual risk-free rate (%)",
    min_value=0.0,
    max_value=20.0,
    value=6.0,
    step=0.25,
    help="Used for the out-of-sample Sharpe ratio. This is an assumption, not a downloaded market rate."
)
risk_free_rate = risk_free_pct / 100

st.sidebar.markdown("---")
st.sidebar.write("**Analysis dates**")
st.sidebar.write("Data: 01-Jan-2025 to 31-Oct-2026")
st.sidebar.write("Training: 01-Jan-2025 to 30-Apr-2026")
st.sidebar.write("Testing: 01-May-2026 to 31-Oct-2026")

st.sidebar.markdown("---")
st.sidebar.caption(
    "The default stock universe is the 10 largest NIFTY 50 constituents "
    "by index weight on 30-Sep-2026. You can edit the tickers in the source code."
)


# ------------------------------------------------------------
# MAIN APP
# ------------------------------------------------------------
if st.button("Run portfolio optimisation and backtest", type="primary"):
    tickers = list(DEFAULT_STOCKS.values())

    try:
        with st.spinner("Downloading daily Close prices from Yahoo Finance..."):
            close_prices = download_close_prices(tickers, START_DATE, END_DATE)

        missing = [ticker for ticker in tickers if ticker not in close_prices.columns]
        if missing:
            st.warning(
                "These tickers were not returned by Yahoo Finance and will be excluded: "
                + ", ".join(missing)
            )

        # Require a complete common history for a fair pairwise comparison.
        usable = close_prices.dropna(axis=1, how="any").copy()
        excluded = sorted(set(close_prices.columns) - set(usable.columns))
        if excluded:
            st.warning(
                "Excluded tickers with missing observations in the requested period: "
                + ", ".join(excluded)
            )

        if usable.shape[1] < 2:
            st.error("Fewer than two complete price series are available. Cannot optimise.")
            st.stop()

        # Use only Close prices. No Adj Close, Open, High, Low or Volume.
        daily_returns = usable.pct_change(fill_method=None).dropna(how="all")

        train_returns = daily_returns.loc[:TRAIN_END].dropna()
        test_returns = daily_returns.loc[TEST_START:TEST_END].dropna()

        if train_returns.empty or test_returns.empty:
            st.error(
                "Training or test data is empty. Check data availability and the date settings."
            )
            st.stop()

        # Ensure identical assets in train and test.
        common_assets = [
            asset for asset in train_returns.columns
            if asset in test_returns.columns
        ]
        train_returns = train_returns[common_assets].dropna()
        test_returns = test_returns[common_assets].dropna()

        if len(common_assets) < 2:
            st.error("Not enough common stocks with valid training and test data.")
            st.stop()

        with st.spinner("Building MVP and HRP portfolios..."):
            mvp_weights = calculate_mvp(train_returns)
            hrp_weights, linkage_matrix, correlation = calculate_hrp(train_returns)

        mvp_metrics, mvp_daily, mvp_wealth, mvp_drawdown = evaluate_strategy(
            test_returns, mvp_weights, risk_free_rate
        )
        hrp_metrics, hrp_daily, hrp_wealth, hrp_drawdown = evaluate_strategy(
            test_returns, hrp_weights, risk_free_rate
        )

        # ----------------------------------------------------
        # RESULTS: DATA SUMMARY
        # ----------------------------------------------------
        st.success("Optimisation and out-of-sample backtest completed.")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Stocks analysed", len(common_assets))
        c2.metric("Training observations", len(train_returns))
        c3.metric("Test observations", len(test_returns))
        c4.metric("Test period", "May–Oct 2026")

        with st.expander("View downloaded Close-price data"):
            st.dataframe(usable, use_container_width=True)
            st.download_button(
                "Download Close prices (CSV)",
                data=usable.to_csv().encode("utf-8"),
                file_name="nifty50_top10_close_prices.csv",
                mime="text/csv"
            )

        # ----------------------------------------------------
        # PERFORMANCE TABLE
        # ----------------------------------------------------
        st.header("1. Out-of-sample performance comparison")

        metrics_df = pd.DataFrame(
            {"MVP": mvp_metrics, "HRP": hrp_metrics}
        ).T
        st.dataframe(
            format_percent_columns(metrics_df),
            use_container_width=True
        )

        best_sharpe = max(
            [("MVP", mvp_metrics["Sharpe Ratio"]),
             ("HRP", hrp_metrics["Sharpe Ratio"])],
            key=lambda x: x[1] if pd.notna(x[1]) else -np.inf
        )
        st.info(
            f"**Higher out-of-sample Sharpe ratio:** {best_sharpe[0]} "
            f"({best_sharpe[1]:.3f}). A higher Sharpe ratio indicates better "
            "historical excess return per unit of realised volatility over this test window."
        )

        # ----------------------------------------------------
        # ALLOCATION TABLE + PIE CHARTS
        # ----------------------------------------------------
        st.header("2. Portfolio allocation")

        allocation_df = pd.DataFrame({
            "Company": [name for name, ticker in DEFAULT_STOCKS.items()
                        if ticker in common_assets],
            "Ticker": common_assets,
            "MVP Allocation": [mvp_weights[t] for t in common_assets],
            "HRP Allocation": [hrp_weights[t] for t in common_assets],
        })
        allocation_df["MVP Allocation (%)"] = allocation_df["MVP Allocation"] * 100
        allocation_df["HRP Allocation (%)"] = allocation_df["HRP Allocation"] * 100

        display_alloc = allocation_df[
            ["Company", "Ticker", "MVP Allocation (%)", "HRP Allocation (%)"]
        ].copy()
        display_alloc["MVP Allocation (%)"] = display_alloc["MVP Allocation (%)"].map(
            lambda x: f"{x:.2f}%"
        )
        display_alloc["HRP Allocation (%)"] = display_alloc["HRP Allocation (%)"].map(
            lambda x: f"{x:.2f}%"
        )
        st.dataframe(display_alloc, use_container_width=True)

        left, right = st.columns(2)
        with left:
            fig_mvp = px.pie(
                allocation_df,
                names="Company",
                values="MVP Allocation",
                hole=0.35,
                title="MVP – Asset Allocation"
            )
            fig_mvp.update_traces(textposition="inside", textinfo="label+percent")
            st.plotly_chart(fig_mvp, use_container_width=True)

        with right:
            fig_hrp = px.pie(
                allocation_df,
                names="Company",
                values="HRP Allocation",
                hole=0.35,
                title="HRP – Asset Allocation"
            )
            fig_hrp.update_traces(textposition="inside", textinfo="label+percent")
            st.plotly_chart(fig_hrp, use_container_width=True)

        st.download_button(
            "Download allocation table (CSV)",
            data=allocation_df.to_csv(index=False).encode("utf-8"),
            file_name="mvp_hrp_allocations.csv",
            mime="text/csv"
        )

        # ----------------------------------------------------
        # CUMULATIVE PERFORMANCE
        # ----------------------------------------------------
        st.header("3. Cumulative portfolio growth – test period")

        wealth_df = pd.DataFrame({
            "MVP": mvp_wealth,
            "HRP": hrp_wealth
        })
        wealth_fig = go.Figure()
        wealth_fig.add_trace(go.Scatter(
            x=wealth_df.index, y=wealth_df["MVP"],
            mode="lines", name="MVP"
        ))
        wealth_fig.add_trace(go.Scatter(
            x=wealth_df.index, y=wealth_df["HRP"],
            mode="lines", name="HRP"
        ))
        wealth_fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Growth of ₹1 (before costs)",
            hovermode="x unified",
            legend_title="Strategy"
        )
        st.plotly_chart(wealth_fig, use_container_width=True)

        # ----------------------------------------------------
        # DRAWDOWN
        # ----------------------------------------------------
        st.header("4. Portfolio drawdown")

        drawdown_df = pd.DataFrame({
            "MVP": mvp_drawdown,
            "HRP": hrp_drawdown
        })
        dd_fig = go.Figure()
        dd_fig.add_trace(go.Scatter(
            x=drawdown_df.index, y=drawdown_df["MVP"],
            mode="lines", name="MVP", fill="tozeroy"
        ))
        dd_fig.add_trace(go.Scatter(
            x=drawdown_df.index, y=drawdown_df["HRP"],
            mode="lines", name="HRP", fill="tozeroy"
        ))
        dd_fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Drawdown",
            yaxis_tickformat=".1%",
            hovermode="x unified"
        )
        st.plotly_chart(dd_fig, use_container_width=True)

        # ----------------------------------------------------
        # TRAINING CORRELATION HEATMAP
        # ----------------------------------------------------
        st.header("5. Training-period correlation matrix")
        corr_fig = px.imshow(
            correlation,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="RdBu",
            zmin=-1,
            zmax=1,
            title="Correlation of daily Close-to-Close returns (training period)"
        )
        st.plotly_chart(corr_fig, use_container_width=True)

        # ----------------------------------------------------
        # DAILY TEST RETURNS
        # ----------------------------------------------------
        st.header("6. Daily portfolio returns – test period")
        daily_df = pd.DataFrame({
            "MVP": mvp_daily,
            "HRP": hrp_daily
        })
        daily_fig = px.line(
            daily_df,
            x=daily_df.index,
            y=["MVP", "HRP"],
            labels={"value": "Daily return", "index": "Date", "variable": "Strategy"},
            title="Daily portfolio returns"
        )
        daily_fig.update_yaxes(tickformat=".1%")
        st.plotly_chart(daily_fig, use_container_width=True)

        st.download_button(
            "Download daily test-period portfolio returns (CSV)",
            data=daily_df.to_csv().encode("utf-8"),
            file_name="mvp_hrp_test_daily_returns.csv",
            mime="text/csv"
        )

        # ----------------------------------------------------
        # METHOD NOTES
        # ----------------------------------------------------
        with st.expander("Methodology, assumptions and limitations"):
            st.markdown("""
            **MVP**
            - Uses the training-period sample covariance matrix.
            - Minimises portfolio variance subject to long-only weights (0–100%) and weights summing to 100%.
            - This is a global minimum-variance portfolio; it does not maximise expected return.

            **HRP**
            - Converts the training-period correlation matrix into a distance matrix.
            - Applies hierarchical clustering, quasi-diagonal ordering and recursive bisection.
            - Allocates more weight to the lower-variance side of each cluster split.

            **Performance calculations**
            - Daily simple returns: `Close[t] / Close[t-1] - 1`.
            - Annualised volatility: daily standard deviation × √252.
            - Annualised Sharpe: (annualised arithmetic mean return − assumed annual risk-free rate) / annualised volatility.
            - Annualised return: geometric annualisation of the realised test-period growth.
            - Maximum drawdown: largest peak-to-trough decline in the test-period wealth index.

            **Important limitations**
            - Uses raw Yahoo Finance `Close`, not `Adj Close`; dividends are excluded.
            - Results are price-return results, not total-return results.
            - No brokerage, taxes, slippage, liquidity constraints or rebalancing costs are included.
            - Weights are estimated once and held fixed through the test period.
            - The stock universe is selected using a September 2026 constituent snapshot, creating potential survivorship/look-ahead bias.
            - Historical results do not guarantee future performance.
            """)

    except Exception as error:
        st.error(f"Unable to complete the analysis: {error}")
        st.exception(error)

else:
    st.markdown("""
    ### What this application produces
    - MVP and HRP portfolio weights using the training period.
    - Out-of-sample total return, annualised return, risk, Sharpe ratio and maximum drawdown.
    - Side-by-side allocation pie charts and an allocation table.
    - Cumulative portfolio growth and drawdown charts.
    - Training-period correlation heatmap.
    - Downloadable CSV files for prices, allocations and daily portfolio returns.

    Click **Run portfolio optimisation and backtest** in the main panel to begin.
    """)
    st.warning(
        "The data window ends on 31 October 2026. Yahoo Finance must have the "
        "required historical observations available for the application to run."
    )
