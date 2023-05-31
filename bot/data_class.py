from pydantic import BaseModel
from typing import Literal, Any


class BBParams(BaseModel):
    bb_period: int
    bb_dev: float
    risk: float


class Strategy(BaseModel):
    name: str
    market: Literal[
        'spot',
        'spot-cross',
        'spot-isolated',
        'um-futures-cross',
        'um-futures-isolated',
        'cm-futures-cross',
        'cm-futures-isolated',
    ]
    symbol: str
    tf: str
    window: int  # length of history window
    params: BBParams

    def __init__(self, **data: Any) -> None:
        if isinstance(data.get("params"), dict):
            data["params"] = BBParams(**data["params"])
        super().__init__(**data)
        if self.window <= self.params.bb_period:
            raise ValueError('window must be greater than bb_period in the strategy parameters.')