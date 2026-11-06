# NIFTY 50 MVP vs HRP Streamlit App

## Files
- `app.py`: Streamlit portfolio optimisation and backtesting application.
- `requirements.txt`: Python dependencies.

## Run locally
1. Install Python 3.10 or newer.
2. Open a terminal in this folder.
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Start Streamlit:

   ```bash
   streamlit run app.py
   ```

## Model design
- Data: Yahoo Finance daily `Close` prices only (`auto_adjust=False`).
- Window: 1 Jan 2025 to 31 Oct 2026.
- Training: 1 Jan 2025 to 30 Apr 2026.
- Test: 1 May 2026 to 31 Oct 2026.
- MVP: long-only global minimum variance optimisation.
- HRP: correlation-distance hierarchical clustering and recursive bisection.
- Performance: total return, annualised return, annualised volatility, Sharpe ratio and maximum drawdown.

## Important caveats
The default universe is based on the top ten NIFTY 50 index-weight constituents as of 30 Sep 2026. Applying this later snapshot to the earlier period introduces survivorship/look-ahead bias. The app uses raw `Close`, not `Adj Close`, so dividends are excluded. It also excludes transaction costs, taxes and slippage. This is a historical educational backtest, not investment advice.
