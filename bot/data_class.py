from pydantic import BaseModel
from typing import Literal


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
    params: dict
