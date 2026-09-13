from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class Config:
    start: str = "2018-01-01"
    end: str = "2025-12-31"
    train_end: str = "2021-12-31"
    validation_end: str = "2022-12-31"
    test_start: str = "2023-01-01"
    horizon: int = 5
    sequence_length: int = 20
    rebalance_days: int = 21
    transaction_cost: float = 0.001
    max_weight: float = 0.20
    top_k: int = 8
    risk_lambda: dict = field(default_factory=lambda: {
        "conservative": 12.0, "moderate": 5.0, "aggressive": 1.5
    })
    data_dir: Path = Path("data")
    results_dir: Path = Path("results")
