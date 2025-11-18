
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class StockCandle(BaseModel):
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    symbol: str
    stock_id: Optional[int] = None
    timestamp: datetime
    resolution: Optional[str] = None
    open: float
    high: float
    low: float
    close: float
    volume: Optional[float] = None