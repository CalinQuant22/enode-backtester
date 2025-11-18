## enode-backtester

Lightweight event-driven backtesting framework that wires together data handlers, strategies, sizing logic, a portfolio, execution simulation, and analysis helpers. The library is designed as a foundation you can extend with richer metrics, visualisations, and strategy research.

---

### Repository Layout

- `backtester/__init__.py` – Public entry point exporting `run_backtest`, utility helpers, and default sizing/execution models.
- `backtester/data.py` – Data-handling abstractions. `DataFrameDataHandler` streams pandas data frames into the event queue as `StockEvent`s.
- `backtester/event.py` – Event hierarchy that flows through the engine (market data, signals, orders, fills).
- `backtester/strategy.py` – Base strategy contract; provides the `signal` helper for emitting `SignalEvent`s.
- `backtester/sizer.py` – Base sizing abstraction plus a simple `FixedSizeSizer`.
- `backtester/portfolio.py` – Tracks cash, positions, trade log, equity curve, and reacts to incoming events.
- `backtester/execution.py` – Simulated execution handler with configurable slippage and commission models.
- `backtester/engine.py` – Core event loop that coordinates the handlers and processes the event queue.
- `backtester/analysis.py` – Post-run utilities, including an optional PyFolio tear sheet and trade log export.
- `backtester/models.py` – Pydantic models shared across modules (e.g., `StockCandle`).

---

### Installation

```bash
# using uv (recommended)
uv sync

# or using pip / virtualenv
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`pyfolio` is optional; install it (`pip install pyfolio`) if you need tear-sheet generation.

---

### Creating a Strategy

1. Subclass `backtester.strategy.BaseStrategy`.
2. Implement `on_stock_event(self, event)` to react to incoming candles (`event.payload` is a `StockCandle`).
3. Use `self.signal(symbol, timestamp, action)` to enqueue `SignalEvent`s (`action` is typically `"buy"` or `"sell"`).

```python
from collections import deque

from backtester.strategy import BaseStrategy
from backtester.event import StockEvent


class MySMAStrategy(BaseStrategy):
    def __init__(self, event_queue, data_handler, window: int = 20):
        super().__init__(event_queue, data_handler)
        self.window = window
        self.prices: dict[str, deque[float]] = {}

    def on_stock_event(self, event: StockEvent) -> None:
        symbol = event.payload.symbol
        price = event.payload.close
        history = self.prices.setdefault(symbol, deque(maxlen=self.window))
        history.append(price)

        if len(history) < history.maxlen:
            return

        avg_price = sum(history) / len(history)
        if price > avg_price:
            self.signal(symbol, event.payload.timestamp, "buy")
        elif price < avg_price:
            self.signal(symbol, event.payload.timestamp, "sell")
```

---

### Running a Backtest

```python
import pandas as pd

from backtester import FixedSizeSizer, run_backtest, get_trade_log_df
from backtester.data import DataFrameDataHandler
from my_strategies import MySMAStrategy

# 1. Prepare your market data as pandas DataFrames keyed by symbol
data_dict: dict[str, pd.DataFrame] = {
    "AAPL": aapl_df,
    "GOOGL": googl_df,
}

# 2. Choose a sizing model (optional)
sizer = FixedSizeSizer(default_size=10)

# 3. Execute the backtest
portfolio = run_backtest(
    data_dict=data_dict,
    strategy_class=MySMAStrategy,
    initial_cash=100_000.0,
    sizer=sizer,
)

# 4. Inspect results
trade_log = get_trade_log_df(portfolio)
print(trade_log.tail())
print(f"Final equity: {portfolio.equity_curve[-1][1]:,.2f}")
```

`run_backtest` instantiates all core components for you:
- `DataFrameDataHandler` feeds candles into the event queue.
- Your strategy consumes `StockEvent`s and emits `SignalEvent`s.
- `Portfolio` generates `OrderEvent`s using the provided `Sizer`.
- `SimulatedExecutionHandler` fills orders with configurable commission and slippage.

After the engine finishes processing all data, you receive the final `Portfolio` object for downstream analysis.

---

### Extending the Framework

- **Metrics & Visuals:** Add new functions under `backtester.analysis` or integrate notebooks that consume the trade log and equity curve.
- **Strategies:** Drop additional strategy classes alongside your own module; they only need to inherit from `BaseStrategy`.
- **Execution/Sizing:** Create custom classes that follow the abstract base class signatures in `execution.py` and `sizer.py`.
- **Data:** Implement new handlers by subclassing `BaseDataHandler` if you need live feeds, streaming data, or alternative storage.

---

### Contributing

Pull requests and issues welcome—especially around additional strategy templates, validation rules, and analytical tooling.
