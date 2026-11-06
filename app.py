import warnings
warnings.filterwarnings("ignore")

from datetime import date
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from scipy.optimize import minimize

# ------------------------------------------------------------
# App configuration
# ------------------------------------------------------------
st.set_page_config(
    page_title="Nifty 50: MVP vs HRP",
    page_icon="📊",
    layout="wide"
)

START_DATE = "2025-01-01"
END_DATE_EXCLUSIVE = "2026-11-01"  # Includes data through 30-Oct-2026
TEST_START = "2026-05-01"

# Top 10 Nifty 50 constituents by index weight, using 30-Sep-2026
# published constituent weights as the reference selection date.
DEFAULT_TICKERS = {
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
# Market data: ONLY the unadjusted "Close" field is selected
# ------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def download_close_prices(tickers, start_date, end_date):
    raw = yf.download(
        tickers=list(tickers),
        start=start_date,
        end=end_date,
        auto_adjust=False,
        actions=False,
        progress=False,
        group_by="column",
        threads=True
    )

    if raw.empty:
        raise ValueError("Yahoo Finance returned no price data.")

    # With multiple tickers, yfinance normally returns MultiIndex columns.
    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" not in raw.columns.get_level_values(0):
            raise ValueError('The downloaded data does not contain a "Close" field.')
        close = raw["Close"].copy()
    else:
        # Defensive handling if the provider returns a single-level frame.
        if "Close" not in raw.columns:
            raise ValueError('The downloaded data does not contain a "Close" field.')
        close = raw[["Close"]].copy()
        if len(tickers) == 1:
            close.columns = [tickers[0]]

    if isinstance(close, pd.Series):
        close = close.to_frame(name=list(tickers)[0])

    # Retain requested ticker order and remove unusable columns/rows.
    available = [ticker for ticker in tickers if ticker in close.columns]
    close = close[available].apply(pd.to_numeric, errors="coerce")
    close = close.dropna(axis=1, how="all").dropna(how="all")
    return close


# ------------------------------------------------------------
# Portfolio optimisation methods
# ------------------------------------------------------------
def minimum_variance_weights(returns):
    """Long-only global minimum-variance portfolio, estimated in-sample."""
    cov = returns.cov().values
    n = cov.shape[0]

    def portfolio_variance(weights):
        return float(weights @ cov @ weights)

    result = minimize(
        portfolio_variance,
        x0=np.repeat(1 / n, n),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n,
        constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}],
        options={"maxiter": 2000, "ftol": 1e-12}
    )

    if not result.success:
        raise RuntimeError(f"MVP optimisation failed: {result.message}")

    weights = np.clip(result.x, 0, 1)
    return pd.Series(weights / weights.sum(), index=returns.columns, name="MVP")


def get_cluster_variance(cluster_items, cov):
    """Compute the variance of an inverse-variance-weighted cluster."""
    sub_cov = cov.loc[cluster_items, cluster_items]
    diagonal = np.diag(sub_cov.values)
    inv_variance = 1.0 / np.maximum(diagonal, 1e-12)
    cluster_weights = inv_variance / inv_variance.sum()
    return float(cluster_weights @ sub_cov.values @ cluster_weights)


def quasi_diagonal(linkage_matrix):
    """Return asset indices ordered by hierarchical clustering."""
    link = linkage_matrix.astype(int)
    sort_ix = pd.Series([link[-1, 0], link[-1, 1]])

    num_items = link[-1, 3]
    while sort_ix.max() >= num_items:
        sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
        clusters = sort_ix[sort_ix >= num_items]
        if clusters.empty:
            break

        cluster_indices = clusters.index
        cluster_values = clusters.values - num_items
        sort_ix[cluster_indices] = link[cluster_values, 0]
        new_values = pd.Series(
            link[cluster_values, 1],
            index=cluster_indices + 1
        )
        sort_ix = pd.concat([sort_ix, new_values]).sort_index()
        sort_ix.index = range(sort_ix.shape[0])

    return sort_ix.astype(int).tolist()


def hrp_weights(returns):
    """
    Hierarchical Risk Parity:
    correlation distance -> hierarchical clustering ->
    quasi-diagonal ordering -> recursive bisection.
    """
    corr = returns.corr().clip(-1, 1)

    # Use a writable NumPy array: DataFrame.values can be read-only.
    distance = np.sqrt(np.maximum((1 - corr.to_numpy(dtype=float)) / 2, 0))
    np.fill_diagonal(distance, 0.0)

    condensed = squareform(distance, checks=False)
    link = linkage(condensed, method="single")
    ordered_indices = quasi_diagonal(link)
    ordered_assets = returns.columns[ordered_indices].tolist()

    cov = returns.cov()
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

            left_var = get_cluster_variance(left, cov)
            right_var = get_cluster_variance(right, cov)

            variance_sum = left_var + right_var
            if not np.isfinite(variance_sum) or variance_sum <= 0:
                alpha = 0.5
            else:
                alpha = 1.0 - left_var / variance_sum

            alpha = float(np.clip(alpha, 0.0, 1.0))
            weights[left] *= alpha
            weights[right] *= (1.0 - alpha)

            next_clusters.extend([left, right])
        clusters = next_clusters

    weights = weights.reindex(returns.columns).fillna(0)
    weights = weights / weights.sum()
    weights.name = "HRP"
    return weights


# ------------------------------------------------------------
# Performance calculations
# ------------------------------------------------------------
def portfolio_daily_returns(asset_returns, weights):
    return asset_returns @ weights.reindex(asset_returns.columns)


def performance_metrics(daily_returns, annual_risk_free_rate):
    r = daily_returns.dropna()
    if len(r) < 2:
        return {
            "Total Return": np.nan,
            "Annualised Return (CAGR)": np.nan,
            "Annualised Volatility": np.nan,
            "Sharpe Ratio": np.nan,
        }

    wealth = (1 + r).cumprod()
    total_return = wealth.iloc[-1] - 1
    cagr = wealth.iloc[-1] ** (TRADING_DAYS / len(r)) - 1
    volatility = r.std(ddof=1) * np.sqrt(TRADING_DAYS)
    daily_rf = annual_risk_free_rate / TRADING_DAYS
    sharpe = (
        (r.mean() - daily_rf) * TRADING_DAYS ** 0.5 / r.std(ddof=1)
        if r.std(ddof=1) > 0 else np.nan
    )
    return {
        "Total Return": total_return,
        "Annualised Return (CAGR)": cagr,
        "Annualised Volatility": volatility,
        "Sharpe Ratio": sharpe,
    }


def drawdown_series(daily_returns):
    wealth = (1 + daily_returns).cumprod()
    return wealth / wealth.cummax() - 1


def format_metrics_table(metrics):
    frame = pd.DataFrame(metrics).T
    for col in ["Total Return", "Annualised Return (CAGR)", "Annualised Volatility"]:
        frame[col] = frame[col].map(lambda x: f"{x:.2%}" if pd.notna(x) else "N/A")
    frame["Sharpe Ratio"] = frame["Sharpe Ratio"].map(
        lambda x: f"{x:.3f}" if pd.notna(x) else "N/A"
    )
    return frame


# ------------------------------------------------------------
# Streamlit interface
# ------------------------------------------------------------
st.title("📊 Nifty 50 Portfolio Optimisation")
st.subheader("Minimum Variance Portfolio (MVP) vs Hierarchical Risk Parity (HRP)")
st.caption(
    "Uses daily NSE-listed stock Close prices from Yahoo Finance. "
    "This is an educational historical backtest, not investment advice."
)

with st.sidebar:
    st.header("Settings")
    st.write("**Historical period:** 01-Jan-2025 to 30-Oct-2026")
    st.write("**Training period:** 01-Jan-2025 to 30-Apr-2026")
    st.write("**Out-of-sample test:** 01-May-2026 to 30-Oct-2026")

    annual_rf = st.number_input(
        "Annual risk-free rate (%)",
        min_value=0.0,
        max_value=20.0,
        value=6.0,
        step=0.25,
        help="Used only for Sharpe ratio calculation. Change this to your chosen Indian risk-free proxy."
    ) / 100

    st.markdown("---")
    st.markdown("**Stock universe**")
    st.caption(
        "Default list: 10 highest-weight Nifty 50 constituents as of "
        "30-Sep-2026. You can edit the Yahoo Finance symbols below."
    )
    ticker_text = st.text_area(
        "Yahoo Finance tickers (one per line)",
        value="\n".join(DEFAULT_TICKERS.values()),
        height=220
    )
    run_button = st.button("Run portfolio analysis", type="primary", use_container_width=True)

tickers = [x.strip().upper() for x in ticker_text.splitlines() if x.strip()]
tickers = list(dict.fromkeys(tickers))

if not tickers:
    st.error("Enter at least two valid Yahoo Finance tickers.")
    st.stop()

if len(tickers) < 2:
    st.error("At least two tickers are required for portfolio optimisation.")
    st.stop()

if run_button or "analysis_has_run" not in st.session_state:
    st.session_state.analysis_has_run = True

if st.session_state.analysis_has_run:
    try:
        with st.spinner("Downloading Close prices and calculating portfolio weights..."):
            close = download_close_prices(
                tickers,
                START_DATE,
                END_DATE_EXCLUSIVE
            )

            missing = sorted(set(tickers) - set(close.columns))
            if missing:
                st.warning(
                    "No usable Close-price series was returned for: "
                    + ", ".join(missing)
                )

            # Require complete price history for a fair, common comparison.
            close = close.dropna(axis=1, how="any")
            if close.shape[1] < 2:
                raise ValueError(
                    "Fewer than two tickers have complete data for the full period. "
                    "Check ticker symbols or use stocks with complete histories."
                )

            close = close.loc[
                (close.index >= pd.Timestamp(START_DATE)) &
                (close.index < pd.Timestamp(END_DATE_EXCLUSIVE))
            ].sort_index()

            asset_returns = close.pct_change().dropna(how="any")
            train_returns = asset_returns.loc[asset_returns.index < pd.Timestamp(TEST_START)]
            test_returns = asset_returns.loc[asset_returns.index >= pd.Timestamp(TEST_START)]

            if train_returns.empty or test_returns.empty:
                raise ValueError(
                    "Training or testing data is empty. Check the available date range."
                )

            if len(train_returns) < 60 or len(test_returns) < 20:
                raise ValueError(
                    "Insufficient observations. Need at least 60 training and "
                    "20 testing daily return observations."
                )

            # Estimate weights using training data only.
            mvp_w = minimum_variance_weights(train_returns)
            hrp_w = hrp_weights(train_returns)

            weights = pd.DataFrame({
                "MVP": mvp_w,
                "HRP": hrp_w
            }).fillna(0)

            # Apply fixed, training-estimated weights to the untouched test set.
            test_portfolios = pd.DataFrame({
                "MVP": portfolio_daily_returns(test_returns, mvp_w),
                "HRP": portfolio_daily_returns(test_returns, hrp_w)
            })

            # Training performance is included for context, not as test results.
            train_portfolios = pd.DataFrame({
                "MVP": portfolio_daily_returns(train_returns, mvp_w),
                "HRP": portfolio_daily_returns(train_returns, hrp_w)
            })

        st.success(
            f"Analysis complete. {len(close.columns)} stocks included; "
            f"{len(train_returns)} training days and {len(test_returns)} testing days."
        )

        st.info(
            "Methodology: portfolio weights are estimated only from the training period "
            "(Jan-2025 to Apr-2026), then held fixed during the test period "
            "(May-2026 to Oct-2026). No test-period returns are used to optimise weights."
        )

        # -------------------- Test metrics --------------------
        st.header("1. Out-of-sample performance — test period")
        test_metrics = {
            "MVP": performance_metrics(test_portfolios["MVP"], annual_rf),
            "HRP": performance_metrics(test_portfolios["HRP"], annual_rf),
        }
        metric_table = format_metrics_table(test_metrics)
        st.dataframe(metric_table, use_container_width=True)

        c1, c2, c3, c4 = st.columns(4)
        mvp_m = test_metrics["MVP"]
        hrp_m = test_metrics["HRP"]
        c1.metric("MVP test return", f"{mvp_m['Total Return']:.2%}")
        c2.metric("HRP test return", f"{hrp_m['Total Return']:.2%}")
        c3.metric("MVP Sharpe", f"{mvp_m['Sharpe Ratio']:.3f}")
        c4.metric("HRP Sharpe", f"{hrp_m['Sharpe Ratio']:.3f}")

        # -------------------- Cumulative performance --------------------
        st.header("2. Growth of ₹1 during the test period")
        growth = (1 + test_portfolios).cumprod()
        growth.index.name = "Date"
        growth_long = growth.reset_index().melt(
            id_vars="Date", var_name="Strategy", value_name="Portfolio value"
        )
        fig_growth = px.line(
            growth_long,
            x="Date",
            y="Portfolio value",
            color="Strategy",
            title="Out-of-sample cumulative portfolio value"
        )
        fig_growth.update_layout(legend_title_text="", hovermode="x unified")
        st.plotly_chart(fig_growth, use_container_width=True)

        # -------------------- Risk / drawdown --------------------
        st.header("3. Risk and drawdown")
        risk_col1, risk_col2 = st.columns(2)

        with risk_col1:
            rolling_vol = test_portfolios.rolling(21).std() * np.sqrt(TRADING_DAYS)
            vol_long = rolling_vol.reset_index().melt(
                id_vars="Date", var_name="Strategy", value_name="Annualised volatility"
            )
            fig_vol = px.line(
                vol_long,
                x="Date",
                y="Annualised volatility",
                color="Strategy",
                title="21-trading-day rolling annualised volatility"
            )
            st.plotly_chart(fig_vol, use_container_width=True)

        with risk_col2:
            dd = pd.DataFrame({
                "MVP": drawdown_series(test_portfolios["MVP"]),
                "HRP": drawdown_series(test_portfolios["HRP"])
            })
            dd_long = dd.reset_index().melt(
                id_vars="Date", var_name="Strategy", value_name="Drawdown"
            )
            fig_dd = px.area(
                dd_long,
                x="Date",
                y="Drawdown",
                color="Strategy",
                title="Test-period drawdown"
            )
            st.plotly_chart(fig_dd, use_container_width=True)

        # -------------------- Allocation pies and table --------------------
        st.header("4. Portfolio allocations")
        alloc_col1, alloc_col2 = st.columns(2)

        with alloc_col1:
            mvp_alloc = weights["MVP"].sort_values(ascending=False).reset_index()
            mvp_alloc.columns = ["Stock", "Weight"]
            fig_mvp = px.pie(
                mvp_alloc,
                names="Stock",
                values="Weight",
                title="MVP allocation",
                hole=0.25
            )
            fig_mvp.update_traces(textposition="inside", textinfo="label+percent")
            st.plotly_chart(fig_mvp, use_container_width=True)

        with alloc_col2:
            hrp_alloc = weights["HRP"].sort_values(ascending=False).reset_index()
            hrp_alloc.columns = ["Stock", "Weight"]
            fig_hrp = px.pie(
                hrp_alloc,
                names="Stock",
                values="Weight",
                title="HRP allocation",
                hole=0.25
            )
            fig_hrp.update_traces(textposition="inside", textinfo="label+percent")
            st.plotly_chart(fig_hrp, use_container_width=True)

        allocation_table = weights.copy()
        allocation_table["MVP"] = allocation_table["MVP"].map(lambda x: f"{x:.2%}")
        allocation_table["HRP"] = allocation_table["HRP"].map(lambda x: f"{x:.2%}")
        allocation_table.index.name = "Stock"
        st.subheader("Allocation comparison table")
        st.dataframe(allocation_table, use_container_width=True)

        # -------------------- Training / test comparison --------------------
        st.header("5. Training vs testing performance")
        train_metrics = {
            "MVP": performance_metrics(train_portfolios["MVP"], annual_rf),
            "HRP": performance_metrics(train_portfolios["HRP"], annual_rf),
        }
        compare = {}
        for strategy in ["MVP", "HRP"]:
            compare[f"{strategy} — Training"] = train_metrics[strategy]
            compare[f"{strategy} — Testing"] = test_metrics[strategy]
        st.dataframe(format_metrics_table(compare), use_container_width=True)

        # -------------------- Correlation matrix --------------------
        st.header("6. Training-period correlation matrix")
        corr = train_returns.corr()
        fig_corr = px.imshow(
            corr,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            title="Asset return correlations (training data only)"
        )
        st.plotly_chart(fig_corr, use_container_width=True)

        # -------------------- Downloadable outputs --------------------
        st.header("7. Download results")
        weights_csv = weights.to_csv().encode("utf-8")
        test_csv = test_portfolios.to_csv().encode("utf-8")
        close_csv = close.to_csv().encode("utf-8")
        d1, d2, d3 = st.columns(3)
        d1.download_button(
            "Download allocation table (CSV)",
            data=weights_csv,
            file_name="mvp_hrp_allocations.csv",
            mime="text/csv"
        )
        d2.download_button(
            "Download test returns (CSV)",
            data=test_csv,
            file_name="mvp_hrp_test_returns.csv",
            mime="text/csv"
        )
        d3.download_button(
            "Download Close prices (CSV)",
            data=close_csv,
            file_name="nifty50_top10_close_prices.csv",
            mime="text/csv"
        )

        with st.expander("View downloaded Close-price data"):
            st.dataframe(close.tail(10), use_container_width=True)

        st.caption(
            "Data note: Yahoo Finance may revise or omit historical observations. "
            "The app uses the unadjusted Close field only, not Adj Close, dividends, "
            "or corporate-action-adjusted total returns. Results therefore represent "
            "price-only returns and may differ from total-return portfolio performance."
        )
        st.caption(
            "Selection-bias note: the default universe is based on Nifty 50 index weights "
            "as of Sep-2026 and is applied retrospectively to Jan-2025. This introduces "
            "look-ahead/survivorship bias. For a fully point-in-time study, use the "
            "constituents that were known at the start of the training period."
        )

    except Exception as exc:
        st.error(f"Unable to complete the analysis: {exc}")
        st.info(
            "Check your internet connection and Yahoo Finance ticker symbols. "
            "If a ticker has a recent listing or missing prices, replace it with "
            "another stock having a complete history for the selected period."
        )
