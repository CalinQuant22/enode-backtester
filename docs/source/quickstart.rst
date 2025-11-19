Quick Start Guide
=================

Installation
------------

Using uv (Recommended)
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    git clone <repository-url>
    cd enode-backtester
    uv sync

Using pip
~~~~~~~~~~

.. code-block:: bash

    git clone <repository-url>
    cd enode-backtester
    pip install -r requirements.txt

Optional Dependencies
~~~~~~~~~~~~~~~~~~~~~

For the interactive dashboard:

.. code-block:: bash

    pip install dash plotly dash-bootstrap-components

Your First Backtest
-------------------

Here's a minimal example to get you started:

.. code-block:: python

    import pandas as pd
    import numpy as np
    from backtester import run_backtest, FixedSizeSizer
    from backtester.strategy import BaseStrategy
    from backtester.event import StockEvent

    # 1. Create a simple strategy
    class BuyAndHoldStrategy(BaseStrategy):
        def __init__(self, event_queue, data_handler):
            super().__init__(event_queue, data_handler)
            self.bought = False
        
        def on_stock_event(self, event: StockEvent) -> None:
            if not self.bought:
                self.signal(event.payload.symbol, event.payload.timestamp, 'buy')
                self.bought = True

    # 2. Prepare sample data
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    np.random.seed(42)
    prices = 100 * (1 + np.random.normal(0.001, 0.02, len(dates))).cumprod()

    data = pd.DataFrame({
        'timestamp': dates,
        'symbol': 'AAPL',
        'open': prices,
        'high': prices * 1.02,
        'low': prices * 0.98,
        'close': prices,
        'volume': 1000000
    })

    # 3. Run backtest
    results = run_backtest(
        data_dict={'AAPL': data},
        strategy_class=BuyAndHoldStrategy,
        initial_cash=100_000.0,
        sizer=FixedSizeSizer(100)
    )

    # 4. Analyze results
    print(f"Total Return: {results.metrics.total_return:.2%}")
    print(f"Sharpe Ratio: {results.metrics.sharpe_ratio:.2f}")

    # 5. Launch dashboard
    results.dashboard()

Key Concepts
------------

Event-Driven Architecture
~~~~~~~~~~~~~~~~~~~~~~~~~

The framework follows an event-driven architecture where components communicate through events:

.. code-block:: text

    Market Data → Strategy → Signals → Portfolio → Orders → Execution → Fills → Portfolio

Core Components
~~~~~~~~~~~~~~~

- **Strategy**: Your trading logic that generates buy/sell signals
- **Portfolio**: Manages positions, cash, and converts signals to orders  
- **DataHandler**: Streams historical data chronologically
- **ExecutionHandler**: Simulates realistic order execution
- **RiskManager**: Validates orders against risk rules
- **Engine**: Coordinates the event loop

Data Format Requirements
------------------------

Your data must be pandas DataFrames with these columns:

.. list-table::
   :header-rows: 1

   * - Column
     - Type
     - Description
   * - timestamp
     - datetime
     - Price timestamp (timezone-aware preferred)
   * - symbol
     - string
     - Stock symbol (e.g., 'AAPL', 'GOOGL')
   * - open
     - float
     - Opening price
   * - high
     - float
     - Highest price
   * - low
     - float
     - Lowest price
   * - close
     - float
     - Closing price
   * - volume
     - int
     - Trading volume (optional)

Next Steps
----------

- Read the :doc:`strategies` guide to learn strategy development
- Explore the :doc:`dashboard` for visual analysis
- Check out :doc:`examples` for more complex strategies
- Review :doc:`risk_management` for capital protection