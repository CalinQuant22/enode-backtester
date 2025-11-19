Interactive Dashboard
====================

The Enode Backtester includes a professional interactive dashboard built with Dash and Plotly for comprehensive visual analysis of your backtest results.

Launching the Dashboard
-----------------------

Method 1: Direct from Results (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Run backtest and launch dashboard immediately
    results = run_backtest(data_dict, MyStrategy, 100_000.0)
    results.dashboard()  # Opens at http://localhost:8050

Method 2: From Saved Results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Save results first
    results.save("my_backtest.json")

    # Launch later from CLI
    # uv run python -m backtester.cli dashboard my_backtest.json

Method 3: Custom Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    results.dashboard(port=8051)  # Custom port
    results.dashboard(save=True)  # Save results file automatically

Dashboard Overview
------------------

The dashboard provides four main analysis tabs:

1. **Performance Overview** - Equity curves and key metrics
2. **Risk Analysis** - Returns distribution and drawdown analysis  
3. **Trade Analysis** - Individual trade statistics and patterns
4. **Monte Carlo** - Scenario analysis and risk assessment

Performance Overview Tab
------------------------

Equity Curve Chart
~~~~~~~~~~~~~~~~~~~

The main equity curve shows your portfolio value over time with several key features:

- **Portfolio Equity**: Your strategy's performance
- **Buy & Hold Benchmark**: Equal-weight passive strategy for comparison
- **Interactive Zoom**: Click and drag to zoom into specific periods
- **Hover Details**: Exact values and dates on mouseover

Key Metrics Cards
~~~~~~~~~~~~~~~~~

Essential performance indicators displayed prominently:

.. list-table::
   :header-rows: 1

   * - Metric
     - Description
     - Good Values
   * - Total Return
     - Overall portfolio return
     - > 10% annually
   * - Sharpe Ratio
     - Risk-adjusted return
     - > 1.0 good, > 2.0 excellent
   * - Max Drawdown
     - Largest decline from peak
     - < 10% excellent, < 20% acceptable
   * - Calmar Ratio
     - Return per unit drawdown
     - > 1.0 good, > 2.0 excellent

Portfolio Composition
~~~~~~~~~~~~~~~~~~~~~

Shows which assets were traded and their relative activity:

- **Symbols Traded**: List of all assets in your strategy
- **Trade Counts**: Number of transactions per symbol
- **Allocation Insights**: Understanding of strategy diversification

Risk Analysis Tab
-----------------

Returns Distribution
~~~~~~~~~~~~~~~~~~~~

Histogram showing the distribution of daily returns with statistical overlays:

- **Normal Distribution Overlay**: Compare to theoretical normal distribution
- **Statistical Measures**: Mean, standard deviation, skewness, kurtosis
- **Tail Analysis**: Identify fat tails and extreme events

Key insights from returns distribution:

- **Symmetry**: Skewed distributions indicate directional bias
- **Fat Tails**: Higher kurtosis suggests more extreme events than normal
- **Outliers**: Identify days with exceptional performance

Drawdown Analysis
~~~~~~~~~~~~~~~~~

Detailed view of portfolio declines from peaks:

- **Drawdown Curve**: Shows decline from running maximum
- **Recovery Periods**: Time to recover from drawdowns
- **Underwater Curve**: Continuous view of drawdown periods

Risk Metrics Table
~~~~~~~~~~~~~~~~~~

Comprehensive risk statistics with explanations:

**Value at Risk (VaR)**
  Expected loss in worst 5% of cases. A 95% VaR of 3% means: "95% of the time, daily loss won't exceed 3%"

**Conditional VaR (CVaR)**
  Average loss when VaR threshold is exceeded. Also called "Expected Shortfall"

**Volatility**
  Annualized standard deviation of returns. Measures price fluctuation intensity

Trade Analysis Tab
------------------

Trade Statistics Summary
~~~~~~~~~~~~~~~~~~~~~~~~

Key trading performance metrics:

**Win Rate**
  Percentage of profitable trades. Formula: Winning Trades / Total Trades

**Profit Factor**
  Total gains divided by total losses. Must be > 1.0 for profitability

**Average Win vs Average Loss**
  Compare typical gain to typical loss. Good strategies often have Average Win > Average Loss

Recent Trades Table
~~~~~~~~~~~~~~~~~~~

Detailed view of individual transactions:

.. list-table::
   :header-rows: 1

   * - Column
     - Description
   * - Timestamp
     - When the trade occurred
   * - Symbol
     - Asset traded
   * - Signal
     - Buy or sell action
   * - Quantity
     - Number of shares
   * - Fill Price
     - Execution price
   * - Commission
     - Transaction cost

Trade Pattern Analysis
~~~~~~~~~~~~~~~~~~~~~~

Visual analysis of trading patterns:

- **Trade Frequency**: How often trades occur
- **Position Holding Periods**: Time between buy and sell
- **Seasonal Patterns**: Monthly or weekly trading tendencies

Monte Carlo Tab
---------------

Scenario Analysis
~~~~~~~~~~~~~~~~~

Range of possible future outcomes based on historical patterns:

**Scenario Percentiles**
  - 95th percentile: Only 5% of simulations performed better
  - 50th percentile: Median outcome (half above, half below)  
  - 5th percentile: Only 5% of simulations performed worse

**Confidence Intervals**
  Statistical ranges for key metrics like returns and Sharpe ratio

Risk Assessment
~~~~~~~~~~~~~~~

**Probability of Loss**
  Likelihood of losing money based on historical patterns:
  
  - < 20%: Low risk strategy
  - 20-40%: Moderate risk  
  - > 40%: High risk strategy

**Stress Testing**
  How the strategy performs under adverse conditions

Understanding Dashboard Metrics
-------------------------------

Performance Metrics Explained
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Total Return**
  Overall portfolio return from start to finish
  
  Formula: :math:`\frac{\text{Final Value} - \text{Initial Value}}{\text{Initial Value}}`

**Sharpe Ratio**
  Risk-adjusted return measure
  
  Formula: :math:`\frac{\text{Mean Excess Return}}{\text{Standard Deviation of Excess Returns}}`
  
  Interpretation: > 1.0 is good, > 2.0 is excellent, > 3.0 is exceptional

**Sortino Ratio**
  Like Sharpe but only penalizes downside volatility
  
  Formula: :math:`\frac{\text{Mean Excess Return}}{\text{Downside Deviation}}`
  
  Generally higher than Sharpe; > 1.5 is good

**Maximum Drawdown**
  Largest peak-to-trough decline
  
  Formula: :math:`\max_t \left( \frac{\text{Peak}_t - \text{Trough}_t}{\text{Peak}_t} \right)`
  
  Lower is better: < 10% is excellent, < 20% is acceptable

**Calmar Ratio**
  Annual return divided by maximum drawdown
  
  Formula: :math:`\frac{\text{Annualized Return}}{|\text{Max Drawdown}|}`
  
  Higher is better: > 1.0 is good, > 2.0 is excellent

Risk Metrics Explained
~~~~~~~~~~~~~~~~~~~~~~

**Value at Risk (VaR)**
  Expected loss in worst 5% of cases. Lower absolute values are better.

**Conditional VaR (CVaR)**
  Average loss when VaR threshold is exceeded. Tells you how bad losses can be in the tail.

**Volatility**
  Annualized standard deviation of returns. Lower is generally better for risk-averse investors.

Trade Metrics Explained
~~~~~~~~~~~~~~~~~~~~~~~

**Win Rate**
  Percentage of trades that were profitable. 50%+ is generally good, but depends on profit factor.

**Profit Factor**
  Total gains divided by total losses. Must be > 1.0 for profitability; > 1.5 is good.

**Average Win vs Average Loss**
  Compare typical gain to typical loss. Can compensate for lower win rate.

Dashboard Customization
-----------------------

Theming
~~~~~~~

The dashboard uses a professional dark theme optimized for financial data visualization:

- **Dark Background**: Reduces eye strain during long analysis sessions
- **High Contrast**: Ensures readability of all text and charts
- **Color Coding**: Consistent use of green/red for gains/losses

Interactive Features
~~~~~~~~~~~~~~~~~~~~

**Chart Interactions**
  - Zoom: Click and drag to zoom into specific time periods
  - Pan: Hold shift and drag to pan across time
  - Hover: Detailed information on mouseover
  - Reset: Double-click to reset zoom

**Data Export**
  - Download charts as PNG images
  - Export underlying data as CSV
  - Save dashboard state for later review

Technical Implementation
------------------------

Dashboard Architecture
~~~~~~~~~~~~~~~~~~~~~~

The dashboard is built using:

- **Dash**: Web application framework
- **Plotly**: Interactive charting library  
- **Bootstrap**: Responsive CSS framework
- **Pandas**: Data manipulation and analysis

Key Components:

.. code-block:: python

    # Dashboard structure
    backtester/dashboard/
    ├── app.py          # Main Dash application
    ├── layout.py       # UI layout and components
    ├── plots.py        # Chart generation functions
    ├── components.py   # Reusable UI components
    ├── callbacks.py    # Interactive behavior
    └── loaders.py      # Data loading utilities

Performance Considerations
~~~~~~~~~~~~~~~~~~~~~~~~~~

The dashboard is optimized for performance:

- **Lazy Loading**: Charts generated only when tabs are accessed
- **Data Caching**: Computed metrics cached to avoid recalculation
- **Efficient Rendering**: Plotly's WebGL rendering for large datasets

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**Dashboard Won't Start**
  - Check that required dependencies are installed: ``pip install dash plotly dash-bootstrap-components``
  - Verify port 8050 is not in use by another application
  - Try a different port: ``results.dashboard(port=8051)``

**Charts Not Displaying**
  - Ensure your browser supports modern JavaScript
  - Check browser console for error messages
  - Try refreshing the page

**Performance Issues**
  - Large datasets (>10,000 data points) may load slowly
  - Consider filtering data to shorter time periods
  - Close other browser tabs to free memory

**Data Not Loading**
  - Verify your backtest results contain valid data
  - Check that equity curve and trade log are populated
  - Ensure timestamps are properly formatted

Browser Compatibility
~~~~~~~~~~~~~~~~~~~~~

The dashboard works best with modern browsers:

- **Chrome/Chromium**: Recommended for best performance
- **Firefox**: Full compatibility
- **Safari**: Full compatibility  
- **Edge**: Full compatibility

Mobile Support
~~~~~~~~~~~~~~

The dashboard is responsive and works on mobile devices, though desktop is recommended for detailed analysis.

Next Steps
----------

- Explore the :doc:`examples` for dashboard usage patterns
- Learn about :doc:`risk_management` to improve your metrics
- Check the API reference for programmatic dashboard access