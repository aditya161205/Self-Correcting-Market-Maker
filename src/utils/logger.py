from dataclasses import dataclass, field
import pandas as pd


@dataclass
class SimulationLogger:
    rows: list[dict] = field(default_factory=list)

    def log(self, row: dict) -> None:
        self.rows.append(row)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)